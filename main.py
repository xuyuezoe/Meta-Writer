"""
MetaWriter entry point.

Usage:
    python main.py
    python main.py --task survey_paper
    python main.py --task-id med_s010
    python main.py --all

    python -m main
    python -m main --task-name argumentative_essay
    python -m main --task-id med_s010
    python -m main --all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Sequence

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from examples.benchmark_template import evaluate_output, list_benchmark_task_ids
from examples.tasks import TASK_REGISTRY


# 目的：
#   保持 `python main.py` 无参数时直接跑完整 benchmark 主链路，
#   让默认入口与正式批量评测行为保持一致。
BENCHMARK_TASK_PREFIX = "metabench_"


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Run a MetaWriter task or benchmark sample")
    parser.add_argument(
        "-t",
        "--task",
        "--task-name",
        dest="task_name",
        type=str,
        help="Run a registered task such as survey_paper, argumentative_essay, or metabench_med_s010",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        help="Run a benchmark sample by ID, for example med_s010",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmark samples in batch mode and score each generated result",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List available tasks and exit",
    )
    parser.add_argument(
        "--print-response",
        action="store_true",
        help="Print the full generated text in the terminal; otherwise only a preview is shown",
    )
    args = parser.parse_args()

    if args.all and (args.task_name or args.task_id):
        parser.error("--all cannot be combined with --task/--task-name or --task-id")
    if args.task_name and args.task_id:
        parser.error("--task/--task-name cannot be combined with --task-id")

    return args


def _benchmark_task_name(task_id: str) -> str:
    """将 benchmark 样本 ID 转换为任务注册表中的动态任务名。"""
    return f"{BENCHMARK_TASK_PREFIX}{task_id}"


def _extract_benchmark_task_id(task_name: str) -> str | None:
    """如果任务名对应动态 benchmark 样本，则返回样本 ID。"""
    if not task_name.startswith(BENCHMARK_TASK_PREFIX):
        return None

    task_id = task_name[len(BENCHMARK_TASK_PREFIX) :]
    if task_id in set(list_benchmark_task_ids()):
        return task_id
    return None


def _resolve_requested_task_names(args: argparse.Namespace) -> List[str]:
    """根据 CLI 参数解析实际要运行的任务名列表。"""
    benchmark_task_ids = list_benchmark_task_ids()
    benchmark_task_names = [
        _benchmark_task_name(task_id) for task_id in benchmark_task_ids
    ]

    if args.all:
        return benchmark_task_names

    if args.task_id:
        if args.task_id not in benchmark_task_ids:
            raise ValueError(f"Unknown benchmark task_id: {args.task_id}")
        return [_benchmark_task_name(args.task_id)]

    if args.task_name:
        return [args.task_name]

    return benchmark_task_names


def _print_available_tasks() -> None:
    """列出当前可用普通任务与 benchmark 样本。"""
    regular_tasks = [
        task_name
        for task_name in sorted(TASK_REGISTRY.keys())
        if _extract_benchmark_task_id(task_name) is None
    ]
    benchmark_task_ids = list_benchmark_task_ids()

    print("Available regular tasks:")
    for task_name in regular_tasks:
        print(f"  - {task_name}")

    print("\nAvailable benchmark samples:")
    print(f"  {len(benchmark_task_ids)} samples available. You can run one directly with --task-id, for example:")
    print("  python main.py --task-id med_s010")
    print("  python -m main --task-id med_s010")
    print("  python main.py --all")


def _load_runtime_settings() -> Dict[str, str | None]:
    """加载运行所需环境变量。"""
    load_dotenv(override=True)
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise EnvironmentError("Please set API_KEY in the .env file")

    return {
        "api_key": api_key,
        "model": os.getenv("MODEL", "MiniMax-M2.5"),
        "base_url": os.getenv("BASE_URL"),
    }


def _write_json(path: Path, payload: Dict[str, object]) -> Path:
    """写入 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _clean_output_files(output_dir: Path, session_name: str) -> None:
    """清理当前任务已有输出，避免旧结果混入新实验。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in output_dir.glob(f"{session_name}_*"):
        if old_file.is_file():
            old_file.unlink()


def _load_task_config(task_name: str) -> Dict[str, object]:
    """从任务注册表加载任务配置。"""
    if task_name not in TASK_REGISTRY:
        available_tasks = sorted(TASK_REGISTRY.keys())
        raise KeyError(f"Unknown task '{task_name}'. Available tasks: {available_tasks}")

    config = TASK_REGISTRY[task_name]()
    required_fields = ("task", "constraints", "outline", "session_name")
    for field_name in required_fields:
        if field_name not in config:
            raise KeyError(f"Task configuration is missing required field: {field_name}")
    return config


def _build_run_result(
    *,
    task_name: str,
    config: Dict[str, object],
    session_name: str,
    final_text: str,
    run_status: str,
    correction_stats: Dict[str, object],
    dtg_stats: Dict[str, object],
    metric_summary: Dict[str, object],
    llm_stats: Dict[str, object],
    artifacts: Dict[str, str],
    benchmark_evaluation: Dict[str, object] | None,
) -> Dict[str, object]:
    """构造单次运行的结构化结果。"""
    result: Dict[str, object] = {
        "status": run_status,
        "task_name": task_name,
        "session_name": session_name,
        "task": str(config["task"]),
        "constraints": list(config["constraints"]) if isinstance(config["constraints"], list) else [],
        "outline": dict(config["outline"]) if isinstance(config["outline"], dict) else {},
        "text_word_count": len(final_text.split()),
        "text_length_chars": len(final_text),
        "correction_stats": correction_stats,
        "dtg_stats": dtg_stats,
        "metric_summary": metric_summary,
        "llm_stats": llm_stats,
        "artifacts": artifacts,
    }

    benchmark_task_id = _extract_benchmark_task_id(task_name)
    if benchmark_task_id is not None:
        result["benchmark_task_id"] = benchmark_task_id

    if benchmark_evaluation is not None:
        result["benchmark_evaluation"] = benchmark_evaluation

    return result


def _print_task_header(
    task_name: str,
    task_description: str,
    constraints: Sequence[str],
    outline: Dict[str, object],
) -> None:
    """打印任务基本信息。"""
    print("=" * 60)
    print(f"MetaWriter | Task: {task_name}")
    benchmark_task_id = _extract_benchmark_task_id(task_name)
    if benchmark_task_id is not None:
        print(f"Benchmark sample: {benchmark_task_id}")
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
    correction_stats: Dict[str, object],
    dtg_stats: Dict[str, object],
    metric_summary: Dict[str, object],
    llm_stats: Dict[str, object],
    benchmark_evaluation: Dict[str, object] | None,
    print_response: bool,
    show_preview: bool,
    summary_file: Path,
) -> None:
    """打印单次运行摘要。"""
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

    if benchmark_evaluation is not None:
        print("\nBenchmark Evaluation:")
        print(json.dumps(benchmark_evaluation, ensure_ascii=False, indent=2))

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
    runtime_settings: Dict[str, str | None],
    *,
    output_dir: Path,
    memory_dir: Path,
    print_response: bool,
    show_preview: bool,
) -> Dict[str, object]:
    """运行单个任务。"""
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
    orchestrator = SelfCorrectingOrchestrator(
        client,
        memory_path=str(memory_dir),
        session_name=session_name,
        output_dir=str(output_dir),
    )

    final_text, _decisions, correction_log = orchestrator.generate_with_self_correction(
        task=task_description,
        constraints=constraints,
        outline=outline,
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

    correction_log_file = output_dir / f"{session_name}_correction_log.json"
    correction_log.save(str(correction_log_file))

    dtg_file = output_dir / f"{session_name}_dtg.json"
    dtg_file.write_text(
        json.dumps(orchestrator.dtg.export_dtg(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    orchestrator.dtg.save_to_disk(session_name)
    session_file = memory_dir / f"{session_name}.json"

    run_status = "completed" if correction_stats["total_failures"] == 0 else "degraded"

    benchmark_evaluation: Dict[str, object] | None = None
    benchmark_eval_file: Path | None = None
    reference = config.get("reference")
    if isinstance(reference, (dict, str)):
        if run_status == "completed":
            benchmark_evaluation = evaluate_output(final_text, reference)
        else:
            benchmark_evaluation = {
                "status": "skipped",
                "reason": "generation_degraded",
                "details": {
                    "total_failures": correction_stats["total_failures"],
                    "message": "One or more sections failed generation, so degraded sections are excluded from formal benchmark scoring.",
                },
            }
        benchmark_eval_file = output_dir / f"{session_name}_benchmark_eval.json"
        _write_json(benchmark_eval_file, benchmark_evaluation)

    artifacts = {
        "text_file": str(text_file),
        "correction_log_file": str(correction_log_file),
        "dtg_file": str(dtg_file),
        "session_file": str(session_file),
    }
    if benchmark_eval_file is not None:
        artifacts["benchmark_eval_file"] = str(benchmark_eval_file)

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
        benchmark_evaluation=benchmark_evaluation,
    )
    _write_json(summary_file, result)

    _print_run_summary(
        task_name=task_name,
        final_text=final_text,
        correction_stats=correction_stats,
        dtg_stats=dtg_stats,
        metric_summary=metric_summary,
        llm_stats=llm_stats,
        benchmark_evaluation=benchmark_evaluation,
        print_response=print_response,
        show_preview=show_preview,
        summary_file=summary_file,
    )

    return result


def _build_batch_summary(results: List[Dict[str, object]]) -> Dict[str, object]:
    """构造批量 benchmark 运行汇总。"""
    success_results = [item for item in results if str(item.get("status")) == "completed"]
    degraded_results = [item for item in results if str(item.get("status")) == "degraded"]
    failed_results = [item for item in results if str(item.get("status")) == "failed"]

    benchmark_eval_results = []
    for item in success_results:
        evaluation = item.get("benchmark_evaluation")
        if isinstance(evaluation, dict):
            if str(evaluation.get("status")) == "skipped":
                continue
            benchmark_eval_results.append(evaluation)

    average_scores: Dict[str, float] = {}
    if benchmark_eval_results:
        average_scores = {
            "constraint_violation_rate": round(
                sum(float(item["constraint_violation_rate"]) for item in benchmark_eval_results)
                / len(benchmark_eval_results),
                4,
            ),
            "entity_consistency_score": round(
                sum(float(item["entity_consistency_score"]) for item in benchmark_eval_results)
                / len(benchmark_eval_results),
                4,
            ),
            "logical_coherence": round(
                sum(float(item["logical_coherence"]) for item in benchmark_eval_results)
                / len(benchmark_eval_results),
                4,
            ),
        }

    return {
        "task_count": len(results),
        "success_count": len(success_results),
        "degraded_count": len(degraded_results),
        "failure_count": len(failed_results),
        "average_benchmark_scores": average_scores,
        "results": results,
    }


def main() -> None:
    """程序主入口。"""
    args = _parse_args()

    if args.list_tasks:
        _print_available_tasks()
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

    batch_results: List[Dict[str, object]] = []
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
                result_task_id = run_result.get("benchmark_task_id", task_name)
                summary_file = run_result["artifacts"]["summary_file"]
                print(f"\n[OK] {result_task_id} -> {summary_file}")
        except Exception:
            batch_failed = True
            error_result = {
                "status": "failed",
                "task_name": task_name,
                "benchmark_task_id": _extract_benchmark_task_id(task_name),
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
        batch_summary_file = output_dir / "metabench_batch_summary.json"
        _write_json(batch_summary_file, batch_summary)

        print("\n" + "=" * 60)
        print("Benchmark Batch Run Complete")
        print("=" * 60)
        print(f"Total tasks: {batch_summary['task_count']}")
        print(f"Completed:   {batch_summary['success_count']}")
        print(f"Degraded:    {batch_summary['degraded_count']}")
        print(f"Failed:      {batch_summary['failure_count']}")
        average_scores = batch_summary["average_benchmark_scores"]
        if average_scores:
            print(f"Average ECS: {average_scores['entity_consistency_score']}")
            print(f"Average LC:  {average_scores['logical_coherence']}")
            print(f"Average CVR: {average_scores['constraint_violation_rate']}")
        print(f"\nBatch summary: {batch_summary_file}")

        if batch_failed:
            print("\nSome tasks failed during the batch run. Check the tracebacks and summaries above.")


if __name__ == "__main__":
    main()
