"""
对比系统运行器：run_baseline

功能：
    运行对比系统（Direct-LLM 或外部已发表系统）单任务，用与 MetaWriter 完全相同的
    MetaBench 评估口径打分，产出标准化运行摘要。覆盖 EXP-II（系统横向对比）。

设计动机：
    对比公平的关键是"同输入、同评估"。本运行器复用 _common 中与 run_metawriter
    一致的评估与落盘逻辑，保证对比系统与 MetaWriter 的摘要结构、评分方式完全一致。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from experiments.baselines import BaselineTask, get_baseline_adapter
from experiments.config.backbone import ResolvedBackbone
from experiments.config.run_context import RunContext

from ._common import (
    RunnerError,
    build_llm_client,
    evaluate_if_possible,
    extract_ordered_metric_scores,
    resolve_task_config,
    summary_exists,
    write_run_bundle,
)


def run_baseline_task(
    *,
    task_id: str,
    system_name: str,
    model_label: str,
    resolved_backbone: Optional[ResolvedBackbone] = None,
    run_id: str = "r1",
    root_dir: str = "./experiments_out/runs",
    overwrite: bool = False,
    envs_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    运行对比系统单任务

    参数：
        task_id:           任务 ID
        system_name:       对比系统名（direct-llm / autosurvey / lira / surveyforge / paperorchestra）
        model_label:       骨干模型别名（用于命名与分组）。Direct-LLM 即其真实骨干；
                           外部系统为该系统所用模型的标识（其真实模型由 baseline_env 决定）。
        resolved_backbone: Direct-LLM 必需（构造 LLMClient）；外部系统可为 None。
        run_id:            重复运行编号
        root_dir:          产物根目录
        overwrite:         为 False 时若 summary 已存在则复用（断点续跑）
        envs_dir:          外部系统 env 配置目录（默认 baselines/envs）

    返回值：
        Dict：运行摘要（结构与 run_metawriter 对齐，含 meta_bench_scores 与 llm_stats）

    异常：
        RunnerError：任务未注册，或 direct-llm 缺少 resolved_backbone 时抛出。
    """
    run_context = RunContext(
        task_id=task_id,
        method=system_name.strip().lower(),
        model=model_label,
        run_id=run_id,
        root_dir=root_dir,
    )

    if not overwrite and summary_exists(run_context):
        return json.loads(
            run_context.artifact_path("summary.json").read_text(encoding="utf-8")
        )

    _task_name, config = resolve_task_config(task_id)
    task = BaselineTask.from_meta_bench_config(task_id, config)

    bundle = run_context.bundle_dir
    bundle.mkdir(parents=True, exist_ok=True)

    # 构造适配器：Direct-LLM 需进程内 LLM 客户端；外部系统经子进程桥接
    llm_client = None
    if system_name.strip().lower() == "direct-llm":
        if resolved_backbone is None:
            raise RunnerError("[对比运行失败] direct-llm 需要 resolved_backbone 以构造 LLM 客户端")
        llm_client = build_llm_client(resolved_backbone)

    adapter = get_baseline_adapter(system_name, llm_client=llm_client, envs_dir=envs_dir)

    start = time.time()
    result = adapter.run(task, work_dir=str(bundle))
    wall_time = time.time() - start

    run_status = result.status if result.status else "completed"
    evaluation = evaluate_if_possible(result.final_text, task.reference, run_status=run_status)
    meta_bench_scores = extract_ordered_metric_scores(evaluation)

    backbone_provenance = (
        resolved_backbone.as_provenance() if resolved_backbone is not None else {"alias": model_label}
    )

    summary: Dict[str, Any] = {
        "provenance": run_context.provenance(
            extra={
                "kind": "baseline",
                "system": system_name,
                "backbone": backbone_provenance,
                "system_extra": result.extra,
            }
        ),
        "task_id": task_id,
        "method": run_context.method,
        "model": model_label,
        "run_id": run_id,
        "status": run_status,
        "word_count": len(result.final_text.split()),
        "wall_time_seconds": round(wall_time, 3),
        "meta_bench_scores": meta_bench_scores,
        "meta_bench_evaluation": evaluation,
        "llm_stats": result.llm_stats(),
    }

    write_run_bundle(run_context, final_text=result.final_text, summary=summary)
    return summary
