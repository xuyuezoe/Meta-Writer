"""
MetaWriter 运行器：run_metawriter

功能：
    在指定骨干模型上，以指定消融配置运行 MetaWriter 单任务，产出标准化运行摘要。
    覆盖 EXP-I（主实验，ablation=full）与 EXP-IV（消融，ablation=各变体）。

设计动机：
    把"配置（消融 + 骨干）→ Orchestrator 执行 → 标准化摘要落盘"封装为一个纯函数，
    使矩阵驱动（batch_driver）只需以不同参数反复调用它。
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict

from experiments.config.backbone import ResolvedBackbone
from experiments.config.run_context import RunContext
from src.core.ablation import AblationConfig

from ._common import (
    RunnerError,
    build_llm_client,
    evaluate_if_possible,
    extract_ordered_metric_scores,
    resolve_task_config,
    summary_exists,
    write_run_bundle,
)


def run_metawriter_task(
    *,
    task_id: str,
    ablation: AblationConfig,
    resolved_backbone: ResolvedBackbone,
    run_id: str = "r1",
    root_dir: str = "./experiments_out/runs",
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    运行 MetaWriter 单任务

    参数：
        task_id:           任务 ID（如 "med_s001"）
        ablation:          消融配置（method 标签由其 method_label 决定）
        resolved_backbone: 已解析的骨干模型运行时配置
        run_id:            重复运行编号（如 "r1"）
        root_dir:          产物根目录
        overwrite:         为 False 时若 summary 已存在则直接复用（断点续跑）

    返回值：
        Dict：运行摘要（含 provenance、meta_bench_scores、各类统计）

    异常：
        RunnerError：任务未注册或配置非法时抛出。
    """
    run_context = RunContext(
        task_id=task_id,
        method=ablation.method_label,
        model=resolved_backbone.alias,
        run_id=run_id,
        root_dir=root_dir,
    )

    # 断点续跑：已完成则复用既有摘要，避免重复消耗 token
    if not overwrite and summary_exists(run_context):
        return json.loads(
            run_context.artifact_path("summary.json").read_text(encoding="utf-8")
        )

    _task_name, config = resolve_task_config(task_id)
    constraints, outline, task_description, reference, corpus_dir = _unpack_config(config)

    bundle = run_context.bundle_dir
    bundle.mkdir(parents=True, exist_ok=True)

    client = build_llm_client(resolved_backbone)

    # 延迟导入 Orchestrator，避免分析/配置场景触发重型模型加载
    from src.orchestrator_v2 import SelfCorrectingOrchestrator

    orchestrator = SelfCorrectingOrchestrator(
        client,
        memory_path=str(bundle / "sessions"),
        session_name=run_context.run_prefix,
        output_dir=str(bundle),
        corpus_dir=corpus_dir,
        ablation=ablation,
    )

    start = time.time()
    final_text, _decisions, correction_log = orchestrator.generate_with_self_correction(
        task=task_description,
        constraints=constraints,
        outline=outline,
        reference=reference,
    )
    wall_time = time.time() - start

    correction_stats = correction_log.get_statistics()
    dtg_stats = orchestrator.dtg.get_statistics()
    metric_summary = orchestrator.metric_collector.summary()
    g3 = metric_summary.get("g3_memory_effectiveness", {})
    if isinstance(g3, dict):
        g3["memory_trust_level"] = round(orchestrator.meta_state.memory_trust_level, 3)
    llm_stats = client.get_statistics()

    run_status = "completed" if correction_stats.get("total_failures", 0) == 0 else "degraded"
    # 注入运行时引用证据（chunk_map / citation_manifest），使引用维度指标
    # （source_fidelity / citation_count / source_balance）可被正确评分，
    # 与默认模式（main.py）的评估口径一致，保证跨模式对比公平。
    runtime_reference = reference
    if isinstance(reference, dict):
        runtime_reference = dict(reference)
        runtime_reference["chunk_map"] = list(orchestrator.last_chunk_map)
        runtime_reference["citation_manifest"] = list(orchestrator.last_citation_manifest)
    evaluation = evaluate_if_possible(final_text, runtime_reference, run_status=run_status)
    meta_bench_scores = extract_ordered_metric_scores(evaluation)

    summary: Dict[str, Any] = {
        "provenance": run_context.provenance(
            extra={
                "kind": "metawriter",
                "ablation": ablation.as_dict(),
                "backbone": resolved_backbone.as_provenance(),
            }
        ),
        "task_id": task_id,
        "method": run_context.method,
        "model": run_context.model,
        "run_id": run_id,
        "status": run_status,
        "word_count": len(final_text.split()),
        "wall_time_seconds": round(wall_time, 3),
        "meta_bench_scores": meta_bench_scores,
        "meta_bench_evaluation": evaluation,
        "correction_stats": correction_stats,
        "dtg_stats": dtg_stats,
        "metric_summary": metric_summary,
        "llm_stats": llm_stats,
    }

    write_run_bundle(run_context, final_text=final_text, summary=summary)
    return summary


def _unpack_config(config: Dict[str, Any]):
    """
    从任务配置解包 Orchestrator 所需字段

    返回值：
        (constraints, outline, task_description, reference, corpus_dir)

    异常：
        RunnerError：缺少必要字段或类型非法时抛出。
    """
    for field_name in ("task", "constraints", "outline", "session_name"):
        if field_name not in config:
            raise RunnerError(f"[任务配置非法] 缺少字段: {field_name}")

    constraints_value = config["constraints"]
    if not isinstance(constraints_value, list):
        raise RunnerError("[任务配置非法] constraints 必须为列表")
    constraints = [str(c) for c in constraints_value]

    outline_value = config["outline"]
    if not isinstance(outline_value, dict):
        raise RunnerError("[任务配置非法] outline 必须为字典")
    outline = {str(k): str(v) for k, v in outline_value.items()}

    task_description = str(config["task"])

    reference = config.get("reference")
    reference_dict = dict(reference) if isinstance(reference, dict) else None

    corpus_dir_value = config.get("corpus_dir")
    corpus_dir = (
        str(corpus_dir_value)
        if isinstance(corpus_dir_value, str) and corpus_dir_value.strip()
        else "./data_sample/med_papers_review_augmented_strict"
    )

    return constraints, outline, task_description, reference_dict, corpus_dir
