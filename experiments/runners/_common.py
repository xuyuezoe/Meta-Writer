"""
执行层共享工具：_common

功能：
    汇集 run_metawriter 与 run_baseline 共用的逻辑：任务配置解析、LLM 客户端构造、
    MetaBench 评估封装、运行产物落盘。

设计动机：
    避免两个 runner 重复实现"按 task_id 取配置""把 summary 写进 bundle"等样板，
    并把这些跨 runner 的契约集中在一处，保证产物结构一致（利于分析层聚合）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from experiments.config.backbone import ResolvedBackbone
from experiments.config.run_context import RunContext


class RunnerError(RuntimeError):
    """执行层异常（任务未注册、配置非法等）"""


def resolve_task_config(task_id: str) -> Tuple[str, Dict[str, Any]]:
    """
    按 task_id 解析 MetaBench 任务配置

    参数：
        task_id: 任务 ID（如 "med_s001"）

    返回值：
        Tuple[str, Dict]：(task_name, 任务配置)

    异常：
        RunnerError：当 task_id 未注册时抛出。
    """
    # 延迟导入，避免在不需要任务注册表的场景强依赖
    from examples.tasks import TASK_ID_REGISTRY, TASK_REGISTRY

    task_name = TASK_ID_REGISTRY.get(task_id)
    if task_name is None:
        known = ", ".join(sorted(TASK_ID_REGISTRY.keys())) or "（空）"
        raise RunnerError(
            f"[任务未注册] 未知 task_id: '{task_id}'。已注册任务: {known}"
        )
    config = TASK_REGISTRY[task_name]()
    return task_name, config


def build_llm_client(resolved: ResolvedBackbone) -> Any:
    """
    从解析后的骨干配置构造 LLMClient

    参数：
        resolved: 已解析的骨干模型运行时配置

    返回值：
        LLMClient 实例

    关键实现细节：
        延迟导入 LLMClient，避免在仅做配置/分析时触发 openai SDK 初始化。
    """
    from src.utils.llm_client import LLMClient

    return LLMClient(
        api_key=resolved.api_key,
        model=resolved.model,
        base_url=resolved.base_url,
    )


def evaluate_if_possible(
    final_text: str,
    reference: Optional[Mapping[str, Any]],
    *,
    run_status: str,
) -> Optional[Dict[str, Any]]:
    """
    在条件满足时执行 MetaBench 评估

    参数：
        final_text: 生成的最终文本
        reference:  MetaBench reference（缺失则不评估）
        run_status: 运行状态（"completed" / "degraded"）。不再用于阻断评分——
                    降级运行同样评分，仅通过 generation_degraded 标记，
                    使消融实验在含降级章节时仍能产出连续指标供统计检验。

    返回值：
        Optional[Dict]：评估结果（始终含真实 metric_scores）；无 reference 时返回 None。
            附加正交字段：generation_degraded（被评估文本是否含降级章节）、
            run_status（原始运行状态），与评估自身 status（"completed"）相互独立。

    设计动机（第一性原理）：
        "评估是否完成"与"被评估文本是否降级"是两个正交事实。旧实现把后者
        当作前者的闸门——只要有 1 节降级就丢弃全部指标，使消融实验无法量化
        "关闭某机制后差了多少"。此处解耦二者：始终评分 + 显式标记降级，
        下游可凭 generation_degraded / status 字段按需筛选。
    """
    if not isinstance(reference, Mapping):
        return None

    from meta_bench import evaluate_meta_bench

    evaluation = evaluate_meta_bench(final_text, reference)
    # 正交标记：评估自身 status 恒为 "completed"；generation_degraded 单独
    # 表达"被评估文本含彻底失败/降级章节"，二者独立，互不覆盖。
    evaluation["generation_degraded"] = run_status != "completed"
    evaluation["run_status"] = run_status
    return evaluation


def extract_ordered_metric_scores(
    evaluation: Optional[Mapping[str, Any]],
) -> Dict[str, float]:
    """
    从 MetaBench 评估结果提取有序指标分数

    参数：
        evaluation: evaluate_meta_bench 的返回值（可能为 None 或 skipped）

    返回值：
        Dict[str, float]：按 META_BENCH_METRIC_ORDER 排序的指标分数；
            无有效评分时返回空字典。
    """
    if not isinstance(evaluation, Mapping):
        return {}
    if str(evaluation.get("status")) == "skipped":
        return {}

    from meta_bench import META_BENCH_METRIC_ORDER

    raw_scores = evaluation.get("metric_scores")
    if not isinstance(raw_scores, Mapping):
        return {}

    ordered: Dict[str, float] = {}
    for metric_name in META_BENCH_METRIC_ORDER:
        value = raw_scores.get(metric_name)
        if isinstance(value, (int, float)):
            ordered[metric_name] = float(value)
    # 追加 META_BENCH_METRIC_ORDER 之外的额外指标，保证不丢信息
    for metric_name, value in raw_scores.items():
        if metric_name not in ordered and isinstance(value, (int, float)):
            ordered[str(metric_name)] = float(value)
    return ordered


def write_run_bundle(
    run_context: RunContext,
    *,
    final_text: str,
    summary: Dict[str, Any],
) -> Path:
    """
    将运行产物写入该次运行的独立 bundle 目录

    参数：
        run_context: 运行上下文（决定 bundle 目录与命名）
        final_text:  最终文本
        summary:     运行摘要（含 provenance、指标、统计）

    返回值：
        Path：bundle 目录路径

    产物：
        <bundle>/text.txt       最终文本
        <bundle>/summary.json   运行摘要（分析层聚合的唯一数据源）
    """
    bundle = run_context.bundle_dir
    bundle.mkdir(parents=True, exist_ok=True)

    (bundle / "text.txt").write_text(final_text, encoding="utf-8")
    (bundle / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return bundle


def summary_exists(run_context: RunContext) -> bool:
    """
    判断该次运行的 summary 是否已存在（断点续跑依据）

    参数：
        run_context: 运行上下文

    返回值：
        bool：summary.json 已存在则 True
    """
    return run_context.artifact_path("summary.json").exists()
