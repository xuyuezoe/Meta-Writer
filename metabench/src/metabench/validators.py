"""MetaBench 数据校验模块。

该模块提供运行时校验，确保关键字段范围合法，
避免默认值掩盖错误并保证后续打分可解释。
"""

from __future__ import annotations

from typing import Dict

from metabench.schemas import EvalMetrics, InputSample, ModelOutput


def _assert_ratio(name: str, value: float) -> None:
    """校验比例值在 [0, 1]。

    参数:
        name: 字段名。
        value: 待校验值。
    """

    if not (0.0 <= value <= 1.0):
        raise ValueError(f"字段 {name} 必须在 [0,1]，当前值为 {value}")


def validate_input_sample(input_sample: InputSample) -> None:
    """校验输入样本。

    参数:
        input_sample: 输入样本对象。
    """

    if input_sample.sample_id == "":
        raise ValueError("sample_id 不能为空")
    if input_sample.prompt == "":
        raise ValueError("prompt 不能为空")
    if input_sample.constraints.required_length_words <= 0:
        raise ValueError("required_length_words 必须为正整数")


def validate_model_output(model_output: ModelOutput) -> None:
    """校验模型输出。

    参数:
        model_output: 模型输出对象。
    """

    if model_output.sample_id == "":
        raise ValueError("model_output.sample_id 不能为空")
    if model_output.model_name == "":
        raise ValueError("model_output.model_name 不能为空")
    if model_output.latency_seconds <= 0:
        raise ValueError("latency_seconds 必须大于 0")
    if model_output.token_usage.prompt_tokens < 0:
        raise ValueError("prompt_tokens 不能小于 0")
    if model_output.token_usage.completion_tokens < 0:
        raise ValueError("completion_tokens 不能小于 0")
    if model_output.token_usage.total_tokens != (
        model_output.token_usage.prompt_tokens + model_output.token_usage.completion_tokens
    ):
        raise ValueError("total_tokens 必须等于 prompt_tokens + completion_tokens")


def validate_eval_metrics(eval_metrics: EvalMetrics) -> None:
    """校验评测中间指标。

    参数:
        eval_metrics: 中间指标对象。
    """

    _assert_ratio("completion_rate", eval_metrics.completion_rate)
    _assert_ratio("acc_once", eval_metrics.acc_once)
    _assert_ratio("acc_range", eval_metrics.acc_range)
    _assert_ratio("acc_periodic", eval_metrics.acc_periodic)
    _assert_ratio("syntax_pass_rate", eval_metrics.syntax_pass_rate)
    _assert_ratio("schema_pass_rate", eval_metrics.schema_pass_rate)

    if eval_metrics.instruction_total <= 0:
        raise ValueError("instruction_total 必须大于 0")
    if not (0 <= eval_metrics.instruction_hits <= eval_metrics.instruction_total):
        raise ValueError("instruction_hits 必须在 [0, instruction_total]")

    if eval_metrics.instruction_hard_total < 0:
        raise ValueError("instruction_hard_total 不能小于 0")
    if not (0 <= eval_metrics.instruction_hard_hits <= eval_metrics.instruction_hard_total):
        raise ValueError("instruction_hard_hits 必须在 [0, instruction_hard_total]")

    if eval_metrics.instruction_soft_total < 0:
        raise ValueError("instruction_soft_total 不能小于 0")
    if not (0 <= eval_metrics.instruction_soft_hits <= eval_metrics.instruction_soft_total):
        raise ValueError("instruction_soft_hits 必须在 [0, instruction_soft_total]")

    if eval_metrics.proxy_qa_total <= 0:
        raise ValueError("proxy_qa_total 必须大于 0")
    if not (0 <= eval_metrics.proxy_qa_correct <= eval_metrics.proxy_qa_total):
        raise ValueError("proxy_qa_correct 必须在 [0, proxy_qa_total]")

    for dim_name, dim_score in eval_metrics.quality_scores.items():
        if not (1.0 <= dim_score <= 5.0):
            raise ValueError(f"质量维度 {dim_name} 分数必须在 [1,5]，当前值为 {dim_score}")


def validate_baseline_map(baseline_map: Dict[str, float]) -> None:
    """校验成本基线映射。

    参数:
        baseline_map: 任务类型到 raw_cost 基线的映射。
    """

    if len(baseline_map) == 0:
        raise ValueError("baseline_map 不能为空")
    for task_type, baseline_value in baseline_map.items():
        if task_type == "":
            raise ValueError("baseline_map 中的 task_type 不能为空")
        if baseline_value <= 0:
            raise ValueError(f"任务 {task_type} 的 baseline 必须大于 0")
