"""MetaWriter entry point."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from examples.tasks import (
    META_BENCH_ALL_TASK_NAMES,
    META_BENCH_TASK_NAMES,
    META_BENCH_TASK_TIERS,
    TASK_ID_REGISTRY,
    TASK_REGISTRY,
)
from meta_bench import META_BENCH_METRIC_ORDER, evaluate_meta_bench


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a MetaWriter task or MetaBench task")
    parser.add_argument(
        "-t",
        "--task",
        "--task-name",
        dest="task_name",
        type=str,
        help="Run a registered task such as scifi_story or metabench_sample.",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        help="Run a registered MetaBench task by task_id, for example med_s001.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run the default MetaBench benchmark tier in batch mode.",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List available tasks and exit.",
    )
    parser.add_argument(
        "--print-response",
        action="store_true",
        help="Print the full generated text in the terminal.",
    )
    # ── 实验模式参数（委托 experiments 执行层）─────────────────────────────
    # 触发条件：提供 --ablation / --model / --baseline 任一即进入实验模式，
    # 产物写入 experiments_out/，与默认 outputs/ 隔离。
    parser.add_argument(
        "--ablation",
        type=str,
        default=None,
        help="Experiment mode: run MetaWriter with an ablation preset (e.g. full, a1_no_dsl). Requires --task-id.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Experiment mode: backbone model alias (minimax/gpt-4o/claude-sonnet/deepseek-v3/qwen2.5-72b).",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Experiment mode: run a comparison system (direct-llm/autosurvey/lira/surveyforge/paperorchestra). Requires --task-id.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="r1",
        help="Experiment mode: repeat-run identifier (e.g. r1/r2/r3).",
    )
    parser.add_argument(
        "--exp-root",
        type=str,
        default="./experiments_out/runs",
        help="Experiment mode: root directory for run artifacts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Experiment mode: re-run even if a summary already exists (disable resume).",
    )
    args = parser.parse_args()

    if args.all and (args.task_name or args.task_id):
        parser.error("--all cannot be combined with --task/--task-name or --task-id")
    if args.task_name and args.task_id:
        parser.error("--task/--task-name cannot be combined with --task-id")

    if (args.ablation or args.baseline) and not args.task_id:
        parser.error("--ablation/--baseline require --task-id")
    if args.ablation and args.baseline:
        parser.error("--ablation and --baseline cannot be combined (choose MetaWriter or a baseline)")

    return args


def _task_id_for_name(task_name: str) -> str | None:
    task_factory = TASK_REGISTRY.get(task_name)
    if task_factory is None:
        return None

    config = task_factory()
    reference = config.get("reference")
    if not isinstance(reference, dict):
        return None

    raw_task_id = reference.get("task_id")
    if not isinstance(raw_task_id, str):
        return None

    task_id = raw_task_id.strip()
    return task_id or None


def _resolve_requested_task_names(args: argparse.Namespace) -> list[str]:
    if args.all:
        return list(META_BENCH_TASK_NAMES)

    if args.task_id:
        task_name = TASK_ID_REGISTRY.get(args.task_id)
        if task_name is None:
            raise ValueError(f"Unknown task_id: {args.task_id}")
        return [task_name]

    if args.task_name:
        return [args.task_name]

    if META_BENCH_TASK_NAMES:
        return [META_BENCH_TASK_NAMES[0]]

    raise ValueError("No MetaBench tasks are registered.")


def _print_available_tasks() -> None:
    regular_tasks = [
        task_name
        for task_name in sorted(TASK_REGISTRY.keys())
        if task_name not in META_BENCH_ALL_TASK_NAMES
    ]

    print("Available regular tasks:")
    for task_name in regular_tasks:
        print(f"  - {task_name}")

    print("\nAvailable MetaBench benchmark tasks:")
    for task_name in META_BENCH_TASK_TIERS["benchmark"]:
        task_id = _task_id_for_name(task_name)
        if task_id is None:
            print(f"  - {task_name}")
        else:
            print(f"  - {task_name} (task_id: {task_id})")

    print("\nAvailable MetaBench task tiers:")
    for tier_name, task_names in META_BENCH_TASK_TIERS.items():
        print(f"  - {tier_name}: {len(task_names)} task(s)")


def _load_runtime_settings() -> dict[str, str | None]:
    load_dotenv(override=True)
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise EnvironmentError("Please set API_KEY in the .env file")

    return {
        "api_key": api_key,
        "model": os.getenv("MODEL", "MiniMax-M2.5"),
        "base_url": os.getenv("BASE_URL"),
    }


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _clean_output_files(output_dir: Path, session_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in output_dir.glob(f"{session_name}_*"):
        if old_file.is_file():
            old_file.unlink()
    bundle_dir = output_dir / session_name
    if bundle_dir.exists() and bundle_dir.is_dir():
        shutil.rmtree(bundle_dir)


def _load_task_config(task_name: str) -> dict[str, object]:
    if task_name not in TASK_REGISTRY:
        available_tasks = sorted(TASK_REGISTRY.keys())
        raise KeyError(f"Unknown task '{task_name}'. Available tasks: {available_tasks}")

    config = TASK_REGISTRY[task_name]()
    required_fields = ("task", "constraints", "outline", "session_name")
    for field_name in required_fields:
        if field_name not in config:
            raise KeyError(f"Task configuration is missing required field: {field_name}")
    return config


def _build_task_input_snapshot(
    *,
    task_name: str,
    config: dict[str, object],
    session_name: str,
) -> dict[str, object]:
    snapshot: dict[str, object] = {
        "framework": "meta_bench",
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "task_name": task_name,
        "task_id": _task_id_for_name(task_name),
        "session_name": session_name,
        "task": str(config["task"]),
        "constraints": list(config["constraints"]) if isinstance(config["constraints"], list) else [],
        "outline": dict(config["outline"]) if isinstance(config["outline"], dict) else {},
    }

    reference = config.get("reference")
    if isinstance(reference, dict):
        snapshot["reference"] = reference
    corpus_dir = config.get("corpus_dir")
    if isinstance(corpus_dir, str) and corpus_dir.strip():
        snapshot["corpus_dir"] = corpus_dir

    return snapshot


def _write_run_bundle(
    *,
    output_dir: Path,
    session_name: str,
    task_input_file: Path,
    text_file: Path,
    correction_log_file: Path,
    dtg_file: Path,
    summary_file: Path,
    session_file: Path,
    run_log_file: Path | None,
    meta_bench_eval_file: Path | None,
    citation_manifest_file: Path | None = None,
    chunk_map_file: Path | None = None,
) -> Path:
    bundle_dir = output_dir / session_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    bundle_artifacts: dict[str, str] = {}
    copy_plan = {
        "task_input": (task_input_file, "task_input.json"),
        "text": (text_file, "text.txt"),
        "correction_log": (correction_log_file, "correction_log.json"),
        "dtg": (dtg_file, "dtg.json"),
        "summary": (summary_file, "summary.json"),
        "session": (session_file, "session.json"),
    }
    if run_log_file is not None:
        copy_plan["run_log"] = (run_log_file, "run.log")
    if meta_bench_eval_file is not None:
        copy_plan["meta_bench_eval"] = (meta_bench_eval_file, "meta_bench_eval.json")
    if citation_manifest_file is not None:
        copy_plan["citation_manifest"] = (citation_manifest_file, "citation_manifest.json")
    if chunk_map_file is not None:
        copy_plan["chunk_map"] = (chunk_map_file, "chunk_map.json")

    for artifact_name, (source_path, target_name) in copy_plan.items():
        if not source_path.exists():
            continue
        target_path = bundle_dir / target_name
        shutil.copy2(source_path, target_path)
        bundle_artifacts[artifact_name] = str(target_path)

    manifest = {
        "framework": "meta_bench",
        "session_name": session_name,
        "bundle_dir": str(bundle_dir),
        "artifacts": bundle_artifacts,
    }
    _write_json(bundle_dir / "artifact_manifest.json", manifest)
    return bundle_dir


def _build_run_result(
    *,
    task_name: str,
    config: dict[str, object],
    session_name: str,
    final_text: str,
    run_status: str,
    correction_stats: dict[str, object],
    dtg_stats: dict[str, object],
    metric_summary: dict[str, object],
    llm_stats: dict[str, object],
    artifacts: dict[str, str],
    meta_bench_evaluation: dict[str, object] | None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "status": run_status,
        "task_name": task_name,
        "session_name": session_name,
        "task": str(config["task"]),
        "constraints": list(config["constraints"]) if isinstance(config["constraints"], list) else [],
        "outline": dict(config["outline"]) if isinstance(config["outline"], dict) else {},
        "text_word_count": len(final_text.split()),
        "text_length_chars": len(final_text),
        "text_length": len(final_text),
        "correction_stats": correction_stats,
        "dtg_stats": dtg_stats,
        "metric_summary": metric_summary,
        "llm_stats": llm_stats,
        "artifacts": artifacts,
    }

    task_id = _task_id_for_name(task_name)
    if task_id is not None:
        result["task_id"] = task_id

    if meta_bench_evaluation is not None:
        metric_scores = meta_bench_evaluation.get("metric_scores", {})
        if isinstance(metric_scores, dict):
            ordered_metric_scores = {
                metric_name: float(metric_scores[metric_name])
                for metric_name in META_BENCH_METRIC_ORDER
                if isinstance(metric_scores.get(metric_name), (int, float))
            }
            extra_metric_scores = {
                str(metric_name): float(metric_value)
                for metric_name, metric_value in metric_scores.items()
                if metric_name not in ordered_metric_scores
                and isinstance(metric_value, (int, float))
            }
            ordered_metric_scores.update(extra_metric_scores)
            result["meta_bench_scores"] = ordered_metric_scores

        result["meta_bench_evaluation"] = meta_bench_evaluation

    return result


def _print_task_header(
    task_name: str,
    task_description: str,
    constraints: Sequence[str],
    outline: dict[str, object],
) -> None:
    print("=" * 60)
    print(f"MetaWriter | Task: {task_name}")
    task_id = _task_id_for_name(task_name)
    if task_id is not None:
        print(f"MetaBench task_id: {task_id}")
    print("=" * 60)
    print(f"\nTask description: {task_description}")
    print(f"Constraint count: {len(constraints)}")
    for constraint in constraints:
        print(f"  - {constraint}")
    print(f"\nOutline ({len(outline)} sections):")
    for section_id, title in outline.items():
        print(f"  [{section_id}] {title}")
    print()


def _print_run_summary(
    *,
    task_name: str,
    final_text: str,
    correction_stats: dict[str, object],
    dtg_stats: dict[str, object],
    metric_summary: dict[str, object],
    llm_stats: dict[str, object],
    meta_bench_evaluation: dict[str, object] | None,
    print_response: bool,
    show_preview: bool,
    summary_file: Path,
) -> None:
    print("\n" + "=" * 60)
    print("Correction Statistics")
    print("=" * 60)
    print(f"  Total sections:    {correction_stats['total_sections']}")
    print(
        f"  First-pass wins:   {correction_stats['success_first_try']} "
        f"({correction_stats['success_rate_first_try']:.1%})"
    )
    print(f"  Total retries:     {correction_stats['total_retries']}")
    print(f"  Total rollbacks:   {correction_stats['total_rollbacks']}")
    print(f"  Failed sections:   {correction_stats['total_failures']}")
    print(f"  Avg attempts:      {correction_stats['avg_attempts']:.2f}")

    if correction_stats["total_rollbacks"] > 0:
        print(f"  Avg rollback span: {correction_stats['avg_rollback_distance']:.1f} sections")

    retry_by_action = correction_stats.get("retry_by_action", {})
    if isinstance(retry_by_action, dict) and retry_by_action:
        print("\nRepair Strategy Distribution:")
        for action, count in sorted(retry_by_action.items(), key=lambda item: -item[1]):
            print(f"  {action}: {count} times")

    print("\nDTG Statistics:")
    print(f"  Total decisions:   {dtg_stats['total_decisions']}")
    print(f"  Intent nodes:      {dtg_stats['total_intent_nodes']}")
    print(f"  Avg confidence:    {dtg_stats['avg_confidence']:.3f}")
    print(f"  Avg refs/decision: {dtg_stats['avg_references_per_decision']:.2f}")
    print(f"  Rollback count:    {dtg_stats['rollback_count']}")

    g2 = metric_summary.get("g2_repair_efficiency", {})
    g3 = metric_summary.get("g3_memory_effectiveness", {})
    print("\nMetrics:")
    print(f"  First-pass rate:   {g2.get('first_pass_rate', 'N/A')}")
    print(f"  Active DSL items:  {g3.get('final_active_entries', 'N/A')}")
    if "memory_trust_level" in g3:
        print(f"  DSL trust level:   {g3['memory_trust_level']}")

    print("\nLLM Statistics:")
    print(f"  Total tokens:      {llm_stats['total_tokens']:,}")
    print(f"  Request count:     {llm_stats['request_count']}")

    if meta_bench_evaluation is not None:
        print("\nMetaBench Evaluation:")
        print(json.dumps(meta_bench_evaluation, ensure_ascii=False, indent=2))

    if print_response:
        print("\n" + "=" * 60)
        print(f"Full Generated Text | {task_name}")
        print("=" * 60)
        print(final_text)
    elif show_preview:
        print("\n" + "=" * 60)
        print("Generated Text Preview (first 500 characters)")
        print("=" * 60)
        print(final_text[:500] + ("..." if len(final_text) > 500 else ""))

    print(f"\nRun summary: {summary_file}")


def _run_single_task(
    task_name: str,
    runtime_settings: dict[str, str | None],
    *,
    output_dir: Path,
    memory_dir: Path,
    print_response: bool,
    show_preview: bool,
) -> dict[str, object]:
    from src.orchestrator_v2 import SelfCorrectingOrchestrator
    from src.utils.llm_client import LLMClient

    config = _load_task_config(task_name)

    constraints_object = config["constraints"]
    if not isinstance(constraints_object, list):
        raise TypeError("task_config.constraints must be a list")
    constraints = [str(item) for item in constraints_object]

    outline_object = config["outline"]
    if not isinstance(outline_object, dict):
        raise TypeError("task_config.outline must be a dictionary")
    outline = {str(section_id): str(title) for section_id, title in outline_object.items()}

    task_description = str(config["task"])
    session_name = str(config["session_name"])

    _clean_output_files(output_dir, session_name)
    _print_task_header(task_name, task_description, constraints, outline)

    client = LLMClient(
        api_key=str(runtime_settings["api_key"]),
        model=str(runtime_settings["model"]),
        base_url=runtime_settings["base_url"],
    )
    corpus_dir_value = config.get("corpus_dir")
    corpus_dir = str(corpus_dir_value) if isinstance(corpus_dir_value, str) else "./data_sample/med_papers_review_augmented_strict"
    orchestrator = SelfCorrectingOrchestrator(
        client,
        memory_path=str(memory_dir),
        session_name=session_name,
        output_dir=str(output_dir),
        corpus_dir=corpus_dir,
    )

    final_text, _decisions, correction_log = orchestrator.generate_with_self_correction(
        task=task_description,
        constraints=constraints,
        outline=outline,
        reference=config.get("reference") if isinstance(config.get("reference"), dict) else None,
    )

    correction_stats = correction_log.get_statistics()
    dtg_stats = orchestrator.dtg.get_statistics()
    metric_summary = orchestrator.metric_collector.summary()
    g3 = metric_summary.get("g3_memory_effectiveness", {})
    if isinstance(g3, dict):
        g3["memory_trust_level"] = round(orchestrator.meta_state.memory_trust_level, 3)
    llm_stats = client.get_statistics()

    text_file = output_dir / f"{session_name}_text.txt"
    text_file.write_text(final_text, encoding="utf-8")

    task_input_snapshot = _build_task_input_snapshot(
        task_name=task_name,
        config=config,
        session_name=session_name,
    )
    task_input_file = output_dir / f"{session_name}_task_input.json"
    _write_json(task_input_file, task_input_snapshot)

    correction_log_file = output_dir / f"{session_name}_correction_log.json"
    correction_log.save(str(correction_log_file))

    dtg_file = output_dir / f"{session_name}_dtg.json"
    dtg_file.write_text(
        json.dumps(orchestrator.dtg.export_dtg(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    orchestrator.dtg.save_to_disk(session_name)
    session_file = memory_dir / f"{session_name}.json"
    run_log_file = output_dir / f"{session_name}_run.log"
    citation_manifest_file: Path | None = None
    chunk_map_file: Path | None = None

    run_status = "completed" if correction_stats["total_failures"] == 0 else "degraded"

    meta_bench_evaluation: dict[str, object] | None = None
    meta_bench_eval_file: Path | None = None
    reference = config.get("reference")
    if isinstance(reference, dict):
        runtime_reference = dict(reference)
        chunk_map = list(orchestrator.last_chunk_map)
        citation_manifest = list(orchestrator.last_citation_manifest)
        runtime_reference["chunk_map"] = chunk_map
        runtime_reference["citation_manifest"] = citation_manifest

        chunk_map_file = output_dir / f"{session_name}_chunk_map.json"
        _write_json(chunk_map_file, chunk_map)

        citation_manifest_file = output_dir / f"{session_name}_citation_manifest.json"
        _write_json(citation_manifest_file, citation_manifest)

        # 方案一：降级运行同样评分，仅显式标记，不再跳过正式评分，
        # 与实验执行层 evaluate_if_possible 保持一致，避免含降级节即丢失全部指标。
        meta_bench_evaluation = evaluate_meta_bench(final_text, runtime_reference)
        meta_bench_evaluation["generation_degraded"] = run_status != "completed"
        meta_bench_evaluation["run_status"] = run_status
        if run_status != "completed":
            meta_bench_evaluation["degraded_details"] = {
                "total_failures": correction_stats["total_failures"],
                "message": (
                    "One or more sections were accepted in a degraded state; "
                    "scores include degraded content and must be read together "
                    "with the generation_degraded flag."
                ),
            }
        meta_bench_eval_file = output_dir / f"{session_name}_meta_bench_eval.json"
        _write_json(meta_bench_eval_file, meta_bench_evaluation)

    artifacts = {
        "task_input_file": str(task_input_file),
        "text_file": str(text_file),
        "correction_log_file": str(correction_log_file),
        "dtg_file": str(dtg_file),
        "session_file": str(session_file),
    }
    if run_log_file.exists():
        artifacts["run_log_file"] = str(run_log_file)
    if meta_bench_eval_file is not None:
        artifacts["meta_bench_eval_file"] = str(meta_bench_eval_file)
    if citation_manifest_file is not None:
        artifacts["citation_manifest_file"] = str(citation_manifest_file)
    if chunk_map_file is not None:
        artifacts["chunk_map_file"] = str(chunk_map_file)

    summary_file = output_dir / f"{session_name}_summary.json"
    artifacts["summary_file"] = str(summary_file)

    result = _build_run_result(
        task_name=task_name,
        config=config,
        session_name=session_name,
        final_text=final_text,
        run_status=run_status,
        correction_stats=correction_stats,
        dtg_stats=dtg_stats,
        metric_summary=metric_summary,
        llm_stats=llm_stats,
        artifacts=artifacts,
        meta_bench_evaluation=meta_bench_evaluation,
    )
    _write_json(summary_file, result)

    bundle_dir = _write_run_bundle(
        output_dir=output_dir,
        session_name=session_name,
        task_input_file=task_input_file,
        text_file=text_file,
        correction_log_file=correction_log_file,
        dtg_file=dtg_file,
        summary_file=summary_file,
        session_file=session_file,
        run_log_file=run_log_file if run_log_file.exists() else None,
        meta_bench_eval_file=meta_bench_eval_file,
        citation_manifest_file=citation_manifest_file,
        chunk_map_file=chunk_map_file,
    )
    result["artifacts"]["bundle_dir"] = str(bundle_dir)
    _write_json(summary_file, result)
    _write_json(bundle_dir / "summary.json", result)

    _print_run_summary(
        task_name=task_name,
        final_text=final_text,
        correction_stats=correction_stats,
        dtg_stats=dtg_stats,
        metric_summary=metric_summary,
        llm_stats=llm_stats,
        meta_bench_evaluation=meta_bench_evaluation,
        print_response=print_response,
        show_preview=show_preview,
        summary_file=summary_file,
    )

    return result


def _build_batch_summary(results: list[dict[str, object]]) -> dict[str, object]:
    success_results = [item for item in results if str(item.get("status")) == "completed"]
    degraded_results = [item for item in results if str(item.get("status")) == "degraded"]
    failed_results = [item for item in results if str(item.get("status")) == "failed"]

    evaluations: list[dict[str, object]] = []
    for item in success_results:
        evaluation = item.get("meta_bench_evaluation")
        if isinstance(evaluation, dict) and str(evaluation.get("status")) != "skipped":
            evaluations.append(evaluation)

    average_metric_scores: dict[str, float] = {}

    if evaluations:
        for metric_name in META_BENCH_METRIC_ORDER:
            metric_values: list[float] = []
            for evaluation in evaluations:
                metric_scores = evaluation.get("metric_scores", {})
                if not isinstance(metric_scores, dict):
                    continue
                raw_value = metric_scores.get(metric_name)
                if isinstance(raw_value, (int, float)):
                    metric_values.append(float(raw_value))
            if metric_values:
                average_metric_scores[metric_name] = round(
                    sum(metric_values) / len(metric_values),
                    4,
                )

    return {
        "task_count": len(results),
        "success_count": len(success_results),
        "degraded_count": len(degraded_results),
        "failure_count": len(failed_results),
        "average_metric_scores": average_metric_scores,
        "results": results,
    }


def _is_experiment_mode(args: argparse.Namespace) -> bool:
    """是否进入实验模式（提供消融/模型/对比系统任一即触发）"""
    return bool(args.ablation or args.baseline or args.model)


def _run_experiment_mode(args: argparse.Namespace) -> None:
    """
    实验模式分派：委托 experiments 执行层运行单次 MetaWriter/对比系统任务

    功能：
        - --baseline 给定：运行对比系统（Direct-LLM 需骨干模型解析）。
        - 否则：运行 MetaWriter（消融预设默认 full）。
        骨干模型别名默认 "minimax"，从 baseline_env / 环境变量解析真实模型与密钥。

    参数：
        args: 已解析的命令行参数（保证含 task_id）

    关键实现细节：
        延迟导入 experiments，避免默认路径加载实验依赖。
    """
    from experiments.config.ablation import from_preset
    from experiments.config.backbone import get_backbone
    from experiments.runners import run_baseline_task, run_metawriter_task

    # 载入配置来源（不覆盖已存在的环境变量，命令行环境优先）：
    #   1. 项目根 .env（legacy 三元组 API_KEY/MODEL/BASE_URL，供 "default" 骨干别名）
    #   2. baseline_env.env（五骨干模型的 *_MODEL/*_BASE_URL/*_API_KEY）
    load_dotenv(override=False)
    baseline_env_file = Path("experiments/baselines/envs/baseline_env.env")
    if baseline_env_file.exists():
        load_dotenv(dotenv_path=str(baseline_env_file), override=False)

    model_alias = args.model or "minimax"

    if args.baseline:
        system_name = args.baseline.strip().lower()
        resolved = None
        if system_name == "direct-llm":
            resolved = get_backbone(model_alias).resolve()
        summary = run_baseline_task(
            task_id=args.task_id,
            system_name=system_name,
            model_label=model_alias,
            resolved_backbone=resolved,
            run_id=args.run_id,
            root_dir=args.exp_root,
            overwrite=bool(args.overwrite),
        )
    else:
        ablation = from_preset(args.ablation) if args.ablation else from_preset("full")
        resolved = get_backbone(model_alias).resolve()
        summary = run_metawriter_task(
            task_id=args.task_id,
            ablation=ablation,
            resolved_backbone=resolved,
            run_id=args.run_id,
            root_dir=args.exp_root,
            overwrite=bool(args.overwrite),
        )

    print("=" * 60)
    print("Experiment run complete")
    print("=" * 60)
    print(f"  task_id:   {summary.get('task_id')}")
    print(f"  method:    {summary.get('method')}")
    print(f"  model:     {summary.get('model')}")
    print(f"  run_id:    {summary.get('run_id')}")
    print(f"  status:    {summary.get('status')}")
    print(f"  words:     {summary.get('word_count')}")
    scores = summary.get("meta_bench_scores", {})
    if isinstance(scores, dict) and scores:
        print("  MetaBench scores:")
        for metric_name, value in scores.items():
            print(f"    {metric_name}: {value}")
    prefix = summary.get("provenance", {}).get("run_prefix", "")
    print(f"\n  bundle: {args.exp_root}/{prefix}/")


def main() -> None:
    args = _parse_args()

    if args.list_tasks:
        _print_available_tasks()
        return

    if _is_experiment_mode(args):
        try:
            _run_experiment_mode(args)
        except Exception as exc:
            print(f"Experiment run failed: {exc}")
            print(traceback.format_exc())
        return

    try:
        task_names = _resolve_requested_task_names(args)
    except Exception as exc:
        print(f"Argument error: {exc}")
        return

    try:
        runtime_settings = _load_runtime_settings()
    except Exception as exc:
        print(f"Environment configuration error: {exc}")
        return

    output_dir = Path("./outputs")
    memory_dir = Path("./sessions")
    is_batch_mode = len(task_names) > 1

    batch_results: list[dict[str, object]] = []
    batch_failed = False

    for task_name in task_names:
        try:
            run_result = _run_single_task(
                task_name,
                runtime_settings,
                output_dir=output_dir,
                memory_dir=memory_dir,
                print_response=bool(args.print_response),
                show_preview=not is_batch_mode,
            )
            batch_results.append(run_result)
            if is_batch_mode:
                task_label = run_result.get("task_id", task_name)
                summary_file = run_result["artifacts"]["summary_file"]
                print(f"\n[OK] {task_label} -> {summary_file}")
        except Exception:
            batch_failed = True
            error_result = {
                "status": "failed",
                "task_name": task_name,
                "task_id": _task_id_for_name(task_name),
                "error": traceback.format_exc().splitlines()[-1] if traceback.format_exc() else "unknown error",
                "traceback": traceback.format_exc(),
            }
            batch_results.append(error_result)
            print(f"\nRun failed: {task_name}")
            print(error_result["traceback"])
            if not is_batch_mode:
                return

    if is_batch_mode:
        batch_summary = _build_batch_summary(batch_results)
        batch_summary_file = output_dir / "meta_bench_batch_summary.json"
        _write_json(batch_summary_file, batch_summary)

        print("\n" + "=" * 60)
        print("MetaBench Batch Run Complete")
        print("=" * 60)
        print(f"Total tasks: {batch_summary['task_count']}")
        print(f"Completed:   {batch_summary['success_count']}")
        print(f"Degraded:    {batch_summary['degraded_count']}")
        print(f"Failed:      {batch_summary['failure_count']}")
        if batch_summary["average_metric_scores"]:
            print("Average metric scores:")
            for metric_name, score in batch_summary["average_metric_scores"].items():
                print(f"  {metric_name}: {score}")
        print(f"\nBatch summary: {batch_summary_file}")

        if batch_failed:
            print("\nSome tasks failed during the batch run. Check the tracebacks and summaries above.")


if __name__ == "__main__":
    main()
