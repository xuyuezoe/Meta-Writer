"""
矩阵驱动器：batch_driver

功能：
    把"任务 × 方法 × 模型 × 重复"的实验矩阵展开为运行单元序列，依次调用对应运行器，
    并提供断点续跑、token 预算监控、逐单元异常隔离与进度清单落盘。

设计动机（第一性原理）：
    顶会实验是数百个运行单元的笛卡尔积（如 40 任务 × 6 方法 × 3 次 = 720）。
    驱动器必须满足：
        1. 幂等可续跑：已完成单元跳过，崩溃后可从断点继续（依据 summary.json 是否存在）。
        2. 预算可控：累计 token 超过预算时立即停止，防止超支。
        3. 故障隔离：单个单元失败不应中断整批（除非 fail_fast），并被如实记录。
        4. 进度可观测：每步把进度清单写盘，便于外部监控与事后审计。
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from experiments.config.backbone import ResolvedBackbone, get_backbone

from ._common import RunnerError
from .run_baseline import run_baseline_task
from .run_metawriter import run_metawriter_task


@dataclass(frozen=True)
class RunSpec:
    """
    单个运行单元规格

    参数：
        kind:    "metawriter" 或 "baseline"
        task_id: 任务 ID
        method:  metawriter 时为消融预设名（如 "full"/"a1_no_dsl"）；
                 baseline 时为系统名（如 "direct-llm"/"autosurvey"）
        model:   骨干模型别名（如 "minimax"/"gpt-4o"）
        run_id:  重复运行编号（如 "r1"）
    """

    kind: str
    task_id: str
    method: str
    model: str
    run_id: str


def build_ablation_matrix(
    task_ids: Sequence[str],
    preset_names: Sequence[str],
    model: str,
    runs: int,
) -> List[RunSpec]:
    """
    构建消融/主实验矩阵（EXP-I / EXP-IV）

    参数：
        task_ids:     任务 ID 序列
        preset_names: 消融预设名序列（含 "full"）
        model:        骨干模型别名
        runs:         每个组合的重复次数

    返回值：
        List[RunSpec]：metawriter 运行单元列表
    """
    specs: List[RunSpec] = []
    for task_id in task_ids:
        for preset in preset_names:
            for r in range(1, runs + 1):
                specs.append(
                    RunSpec(kind="metawriter", task_id=task_id, method=preset, model=model, run_id=f"r{r}")
                )
    return specs


def build_comparison_matrix(
    task_ids: Sequence[str],
    systems: Sequence[str],
    model: str,
    runs: int,
) -> List[RunSpec]:
    """
    构建系统对比矩阵（EXP-II）

    参数：
        task_ids: 任务 ID 序列
        systems:  对比系统名序列（如 direct-llm / autosurvey / ...）
        model:    骨干模型别名
        runs:     每个组合的重复次数

    返回值：
        List[RunSpec]：baseline 运行单元列表
    """
    specs: List[RunSpec] = []
    for task_id in task_ids:
        for system in systems:
            for r in range(1, runs + 1):
                specs.append(
                    RunSpec(kind="baseline", task_id=task_id, method=system, model=model, run_id=f"r{r}")
                )
    return specs


def build_backbone_matrix(
    task_ids: Sequence[str],
    models: Sequence[str],
    runs: int,
    *,
    include_direct_llm: bool = True,
) -> List[RunSpec]:
    """
    构建骨干模型泛化矩阵（EXP-III）：每个模型上跑 Direct-LLM 与 MetaWriter Full

    参数：
        task_ids:           任务 ID 序列
        models:             骨干模型别名序列
        runs:               重复次数
        include_direct_llm: 是否纳入 Direct-LLM 对照

    返回值：
        List[RunSpec]：混合 metawriter(full) 与 baseline(direct-llm) 的运行单元
    """
    specs: List[RunSpec] = []
    for task_id in task_ids:
        for model in models:
            for r in range(1, runs + 1):
                specs.append(
                    RunSpec(kind="metawriter", task_id=task_id, method="full", model=model, run_id=f"r{r}")
                )
                if include_direct_llm:
                    specs.append(
                        RunSpec(kind="baseline", task_id=task_id, method="direct-llm", model=model, run_id=f"r{r}")
                    )
    return specs


def run_matrix(
    specs: Sequence[RunSpec],
    *,
    root_dir: str = "./experiments_out/runs",
    manifest_dir: str = "./experiments_out/manifests",
    manifest_name: str = "matrix_progress.json",
    token_budget: Optional[int] = None,
    overwrite: bool = False,
    fail_fast: bool = False,
    envs_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    驱动整个实验矩阵

    参数：
        specs:         运行单元序列
        root_dir:      运行产物根目录
        manifest_dir:  进度清单目录
        manifest_name: 进度清单文件名
        token_budget:  累计 token 上限；超过则停止（None 表示不限）
        overwrite:     是否强制重跑已完成单元
        fail_fast:     单元失败时是否立即中断整批
        envs_dir:      外部系统 env 配置目录

    返回值：
        Dict：批次报告（完成/失败/跳过计数、累计 token、各单元结果摘要）

    异常：
        仅在 fail_fast=True 且某单元失败时向上抛出；否则失败被记录并继续。
    """
    manifest_path = Path(manifest_dir)
    manifest_path.mkdir(parents=True, exist_ok=True)
    progress_file = manifest_path / manifest_name

    backbone_cache: Dict[str, ResolvedBackbone] = {}
    records: List[Dict[str, Any]] = []
    total_tokens = 0
    counts = {"completed": 0, "skipped": 0, "failed": 0, "stopped_budget": 0}

    for index, spec in enumerate(specs):
        record: Dict[str, Any] = {
            "index": index,
            "spec": asdict(spec),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        # 预算闸门：在执行前检查，超预算则停止
        if token_budget is not None and total_tokens >= token_budget:
            record["status"] = "stopped_budget"
            counts["stopped_budget"] += 1
            records.append(record)
            break

        try:
            summary = _dispatch(spec, root_dir=root_dir, overwrite=overwrite, envs_dir=envs_dir, backbone_cache=backbone_cache)
            spent = _extract_tokens(summary)
            total_tokens += spent
            record["status"] = "completed"
            record["run_status"] = summary.get("status")
            record["tokens"] = spent
            record["meta_bench_scores"] = summary.get("meta_bench_scores", {})
            counts["completed"] += 1
        except Exception as exc:  # 故障隔离：记录并继续（除非 fail_fast）
            record["status"] = "failed"
            record["error"] = str(exc)
            record["traceback"] = traceback.format_exc()
            counts["failed"] += 1
            records.append(record)
            _write_progress(progress_file, specs, records, counts, total_tokens)
            if fail_fast:
                raise
            continue

        records.append(record)
        _write_progress(progress_file, specs, records, counts, total_tokens)

    report = {
        "total_specs": len(specs),
        "counts": counts,
        "total_tokens": total_tokens,
        "token_budget": token_budget,
        "records": records,
        "progress_file": str(progress_file),
    }
    _write_progress(progress_file, specs, records, counts, total_tokens, final_report=report)
    return report


# ----------------------------------------------------------------------
# 内部辅助
# ----------------------------------------------------------------------

def _dispatch(
    spec: RunSpec,
    *,
    root_dir: str,
    overwrite: bool,
    envs_dir: Optional[Path],
    backbone_cache: Dict[str, ResolvedBackbone],
) -> Dict[str, Any]:
    """把单个运行单元分派到对应运行器"""
    if spec.kind == "metawriter":
        from experiments.config.ablation import from_preset

        ablation = from_preset(spec.method)
        resolved = _resolve_backbone(spec.model, backbone_cache)
        return run_metawriter_task(
            task_id=spec.task_id,
            ablation=ablation,
            resolved_backbone=resolved,
            run_id=spec.run_id,
            root_dir=root_dir,
            overwrite=overwrite,
        )

    if spec.kind == "baseline":
        resolved: Optional[ResolvedBackbone] = None
        if spec.method.strip().lower() == "direct-llm":
            resolved = _resolve_backbone(spec.model, backbone_cache)
        return run_baseline_task(
            task_id=spec.task_id,
            system_name=spec.method,
            model_label=spec.model,
            resolved_backbone=resolved,
            run_id=spec.run_id,
            root_dir=root_dir,
            overwrite=overwrite,
            envs_dir=envs_dir,
        )

    raise RunnerError(f"[矩阵驱动失败] 未知运行类型 kind='{spec.kind}'")


def _resolve_backbone(alias: str, cache: Dict[str, ResolvedBackbone]) -> ResolvedBackbone:
    """解析骨干模型（带缓存，避免重复读环境变量）"""
    if alias not in cache:
        cache[alias] = get_backbone(alias).resolve()
    return cache[alias]


def _extract_tokens(summary: Dict[str, Any]) -> int:
    """从摘要中提取本次 token 消耗（无统计时计 0）"""
    llm_stats = summary.get("llm_stats")
    if isinstance(llm_stats, dict):
        value = llm_stats.get("total_tokens")
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def _write_progress(
    progress_file: Path,
    specs: Sequence[RunSpec],
    records: List[Dict[str, Any]],
    counts: Dict[str, int],
    total_tokens: int,
    *,
    final_report: Optional[Dict[str, Any]] = None,
) -> None:
    """把进度清单写盘（每个单元后调用，保证崩溃后仍有最新进度）"""
    payload = {
        "total_specs": len(specs),
        "counts": counts,
        "total_tokens": total_tokens,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "records": records,
    }
    if final_report is not None:
        payload["final"] = True
    progress_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
