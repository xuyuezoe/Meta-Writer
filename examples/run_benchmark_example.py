"""本地 benchmark example 运行脚本。

功能：
    只使用 bench 与 examples 接口层完成可复现的加载、取样、评估闭环，
    并支持单样本、批量样本、结果落盘、精简汇总与正文打印。

用法：
    python -m examples.run_benchmark_example
    python -m examples.run_benchmark_example --task-id s2
    python -m examples.run_benchmark_example --all
    python -m examples.run_benchmark_example --task-id s1 --save-text --print-response
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from .benchmark_template import (
    evaluate_output,
    list_benchmark_task_ids,
    load_benchmark_task,
)


OUTPUTS_PATH: Path = (
    Path(__file__).resolve().parent.parent / "metabench" / "examples" / "outputs.jsonl"
)
RESULTS_DIR: Path = Path(__file__).resolve().parent.parent / "outputs"


def _read_jsonl_rows(file_path: Path) -> List[Dict[str, object]]:
    """读取 JSONL 文件。

    参数：
        file_path: JSONL 文件路径。

    返回值：
        List[Dict[str, object]]：逐行解析后的数据。

    关键实现细节：
        仅依赖 examples 层本地文件，确保 bench 演示链路不绑定主程序运行状态。
    """
    if not file_path.exists():
        raise FileNotFoundError(f"未找到示例输出文件：{file_path}")

    rows: List[Dict[str, object]] = []
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        stripped_line = raw_line.strip()
        if stripped_line == "":
            continue
        parsed_row = json.loads(stripped_line)
        if not isinstance(parsed_row, dict):
            raise ValueError("JSONL 行必须解析为字典")
        rows.append(parsed_row)
    return rows


def _load_demo_output(task_id: str) -> Dict[str, object]:
    """加载指定 benchmark 样本对应的示例输出。

    参数：
        task_id: benchmark 样本 ID。

    返回值：
        Dict[str, object]：输出样本字典。

    关键实现细节：
        按 `sample_id` 精确匹配，避免不同样本之间误用演示输出。
    """
    for output_row in _read_jsonl_rows(OUTPUTS_PATH):
        if str(output_row.get("sample_id")) == task_id:
            return output_row
    raise ValueError(f"未找到 task_id={task_id} 的示例输出")


def _write_json_result(file_name: str, payload: Dict[str, object]) -> Path:
    """写入 JSON 结果文件。

    参数：
        file_name: 输出文件名。
        payload: 待写入数据。

    返回值：
        Path：落盘路径。

    关键实现细节：
        所有 example 产物统一写入 `outputs/`，保持与仓库现有输出目录一致。
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / file_name
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_path


def _write_text_result(task_id: str, response_text: str) -> Path:
    """写入示例输出文本。

    参数：
        task_id: benchmark 样本 ID。
        response_text: 输出文本。

    返回值：
        Path：文本文件路径。

    关键实现细节：
        单独保留文本文件，方便快速查看示例输出内容而无需打开 JSON。
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"benchmark_example_{task_id}.txt"
    output_path.write_text(response_text, encoding="utf-8")
    return output_path


def _run_single(task_id: str) -> Dict[str, object]:
    """运行单个样本的 benchmark example。

    参数：
        task_id: benchmark 样本 ID。

    返回值：
        Dict[str, object]：包含任务信息、输出和评估结果的完整字典。

    关键实现细节：
        该函数既服务命令行打印，也服务批量模式汇总，避免重复实现评估逻辑。
    """
    benchmark_task = load_benchmark_task(task_id)
    output_row = _load_demo_output(task_id)
    response_text = str(output_row["response"])
    evaluation_result = evaluate_output(response_text, benchmark_task["reference"])
    return {
        "task_id": task_id,
        "task": benchmark_task["task"],
        "constraints": benchmark_task["constraints"],
        "outline": benchmark_task["outline"],
        "response": response_text,
        "evaluation": evaluation_result,
    }


def _build_summary(results: List[Dict[str, object]]) -> Dict[str, object]:
    """构造精简汇总结果。

    参数：
        results: 批量运行得到的完整结果列表。

    返回值：
        Dict[str, object]：更适合榜单查看的汇总字典。

    关键实现细节：
        summary 只保留 task_id 与三项核心分数，并补一个简单综合分用于快速排序。
    """
    compact_results: List[Dict[str, object]] = []
    for item in results:
        evaluation = item["evaluation"]
        composite_score = (
            evaluation["entity_consistency_score"]
            + evaluation["logical_coherence"]
            + (1.0 - evaluation["constraint_violation_rate"])
        ) / 3.0
        compact_results.append(
            {
                "task_id": item["task_id"],
                "cvr": evaluation["constraint_violation_rate"],
                "ecs": evaluation["entity_consistency_score"],
                "lc": evaluation["logical_coherence"],
                "score": round(composite_score, 4),
            }
        )

    compact_results.sort(key=lambda item: (-item["score"], item["task_id"]))
    return {
        "task_count": len(compact_results),
        "ranking": compact_results,
    }


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。

    参数：
        无。

    返回值：
        argparse.Namespace：命令行参数对象。

    关键实现细节：
        `--all` 与 `--task-id` 二选一语义由调用约定保证，默认单样本运行 `s1`。
    """
    parser = argparse.ArgumentParser(description="运行本地 benchmark example")
    parser.add_argument(
        "--task-id", type=str, default="s1", help="指定 benchmark 样本 ID"
    )
    parser.add_argument(
        "--all", action="store_true", help="批量运行全部 benchmark 样本"
    )
    parser.add_argument("--save-text", action="store_true", help="额外保存示例输出文本")
    parser.add_argument(
        "--print-response", action="store_true", help="在终端打印示例输出正文"
    )
    return parser.parse_args()


def main() -> None:
    """执行 benchmark example 演示。

    参数：
        无。

    返回值：
        无。

    关键实现细节：
        默认运行 `s1`，支持指定样本 ID，也支持 `--all` 批量验证全部样本。
    """
    args = _parse_args()

    if args.all:
        all_results: List[Dict[str, object]] = []
        for task_id in list_benchmark_task_ids():
            single_result = _run_single(task_id)
            all_results.append(single_result)
            result_path = _write_json_result(
                f"benchmark_example_{task_id}.json", single_result
            )
            if args.save_text:
                _write_text_result(task_id, single_result["response"])
            print(f"[OK] {task_id} -> {result_path}")

        full_summary = {
            "task_count": len(all_results),
            "task_ids": [item["task_id"] for item in all_results],
            "results": all_results,
        }
        summary_payload = _build_summary(all_results)
        full_summary_path = _write_json_result(
            "benchmark_example_all.json", full_summary
        )
        compact_summary_path = _write_json_result(
            "benchmark_example_summary.json", summary_payload
        )
        print(f"\n批量结果：{full_summary_path}")
        print(f"精简汇总：{compact_summary_path}")
        return

    single_result = _run_single(args.task_id)
    result_path = _write_json_result(
        f"benchmark_example_{args.task_id}.json", single_result
    )
    text_path: Path | None = None
    if args.save_text:
        text_path = _write_text_result(args.task_id, single_result["response"])

    print("=" * 60)
    print(f"Benchmark Example | task_id={args.task_id}")
    print("=" * 60)
    print(f"任务：{single_result['task']}")
    print(f"约束：{single_result['constraints']}")
    print(f"大纲：{single_result['outline']}")
    if args.print_response:
        print("\n示例输出：")
        print(single_result["response"])
    print("\n评估结果：")
    print(json.dumps(single_result["evaluation"], ensure_ascii=False, indent=2))
    print(f"\n结果文件：{result_path}")
    if text_path is not None:
        print(f"文本文件：{text_path}")


if __name__ == "__main__":
    main()
