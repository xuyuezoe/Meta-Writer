"""MetaBench 主流水线。

该模块负责串联读取、校验、计数、打分与结果导出。
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Dict, List

from metabench.io_utils import (
    build_eval_metrics,
    build_input_sample,
    build_model_output,
    read_json,
    read_jsonl,
    sample_result_to_dict,
    write_json,
    write_jsonl,
    write_radar_csv,
)
from metabench.scoring import score_one_sample
from metabench.schemas import SampleResult
from metabench.tokenization import TokenizerConfig, count_tokens, resolve_usage_total_tokens
from metabench.validators import (
    validate_baseline_map,
    validate_eval_metrics,
    validate_input_sample,
    validate_model_output,
)


def _build_index(rows: List[Dict[str, object]], key_field: str) -> Dict[str, Dict[str, object]]:
    """按主键构建索引。"""

    indexed: Dict[str, Dict[str, object]] = {}
    for row in rows:
        row_key = str(row[key_field])
        if row_key in indexed:
            raise ValueError(f"检测到重复主键 {key_field}={row_key}")
        indexed[row_key] = row
    return indexed


def _compute_summary(sample_results: List[SampleResult]) -> Dict[str, object]:
    """计算模型级汇总结果。"""

    valid_results = [result for result in sample_results if result.overall_score is not None]
    if len(valid_results) == 0:
        raise ValueError("没有可用于汇总的样本，所有样本 overall_score 均为 None")

    overall_scores = [float(result.overall_score) for result in valid_results]
    summary: Dict[str, object] = {
        "sample_count": len(sample_results),
        "valid_sample_count": len(valid_results),
        "overall_mean": statistics.fmean(overall_scores),
        "overall_stdev": statistics.pstdev(overall_scores),
        "overall_p50": statistics.median(overall_scores),
    }
    return summary


def run_pipeline(
    samples_path: Path,
    outputs_path: Path,
    metrics_path: Path,
    baseline_path: Path,
    run_output_dir: Path,
    run_id: str,
    model_name: str,
    judge_model: str,
    data_version: str,
    tokenizer_config: TokenizerConfig,
    power_p: float,
) -> Dict[str, object]:
    """执行 MetaBench 评测流水线。

    参数:
        samples_path: 输入样本 JSONL 路径。
        outputs_path: 模型输出 JSONL 路径。
        metrics_path: 中间指标 JSONL 路径。
        baseline_path: 成本基线 JSON 路径。
        run_output_dir: 运行产物目录。
        run_id: 运行标识。
        model_name: 被评测模型名。
        judge_model: 裁判模型名。
        data_version: 数据版本。
        tokenizer_config: tokenizer 配置。
        power_p: 幂平均参数。

    返回:
        汇总结果字典。
    """

    # 第一阶段：读取与索引
    sample_rows = read_jsonl(samples_path)
    output_rows = read_jsonl(outputs_path)
    metric_rows = read_jsonl(metrics_path)
    baseline_map_raw = read_json(baseline_path)

    sample_index = _build_index(sample_rows, "sample_id")
    output_index = _build_index(output_rows, "sample_id")
    metric_index = _build_index(metric_rows, "sample_id")
    baseline_map: Dict[str, float] = {str(k): float(v) for k, v in baseline_map_raw.items()}
    validate_baseline_map(baseline_map)

    # 第二阶段：样本计算
    sample_results: List[SampleResult] = []
    for sample_id, sample_row in sample_index.items():
        if sample_id not in output_index:
            raise ValueError(f"sample_id={sample_id} 缺少输出数据")
        if sample_id not in metric_index:
            raise ValueError(f"sample_id={sample_id} 缺少中间指标数据")

        input_sample = build_input_sample(sample_row)
        raw_output_row = output_index[sample_id]
        raw_metric_row = metric_index[sample_id]

        prompt_tokens = count_tokens(tokenizer_config, input_sample.prompt)
        completion_tokens = count_tokens(tokenizer_config, str(raw_output_row["response"]))
        total_tokens = prompt_tokens + completion_tokens
        usage_total_tokens = resolve_usage_total_tokens(
            None if raw_output_row.get("usage_total_tokens") is None else int(raw_output_row["usage_total_tokens"])
        )

        normalized_output_row = {
            "sample_id": raw_output_row["sample_id"],
            "model_name": raw_output_row["model_name"],
            "response": raw_output_row["response"],
            "latency_seconds": raw_output_row["latency_seconds"],
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "usage_total_tokens": usage_total_tokens,
            },
        }

        model_output = build_model_output(normalized_output_row)
        eval_metrics = build_eval_metrics(raw_metric_row)

        validate_input_sample(input_sample)
        validate_model_output(model_output)
        validate_eval_metrics(eval_metrics)

        if input_sample.task_type not in baseline_map:
            raise ValueError(f"任务类型 {input_sample.task_type} 未配置 baseline raw cost")

        score_breakdown, overall_score, diagnostics = score_one_sample(
            input_sample=input_sample,
            model_output=model_output,
            eval_metrics=eval_metrics,
            baseline_raw_cost=baseline_map[input_sample.task_type],
            power_p=power_p,
        )

        sample_results.append(
            SampleResult(
                sample_id=input_sample.sample_id,
                task_type=input_sample.task_type,
                model_name=model_output.model_name,
                scores=score_breakdown,
                overall_score=overall_score,
                diagnostics=diagnostics,
            )
        )

    # 第三阶段：汇总导出
    summary = _compute_summary(sample_results)
    summary["run_id"] = run_id
    summary["model_name"] = model_name
    summary["judge_model"] = judge_model
    summary["data_version"] = data_version

    samples_output_path = run_output_dir / run_id / "samples.jsonl"
    summary_output_path = run_output_dir / run_id / "summary.json"
    radar_output_path = run_output_dir / run_id / "radar.csv"

    write_jsonl(samples_output_path, [sample_result_to_dict(result) for result in sample_results])
    write_json(summary_output_path, summary)
    write_radar_csv(radar_output_path, sample_results)

    return summary
