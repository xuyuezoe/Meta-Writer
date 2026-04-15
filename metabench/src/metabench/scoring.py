"""MetaBench 七维打分模块。

该模块负责将中间指标映射为七维主分，并执行幂平均聚合。
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

from metabench.schemas import EvalMetrics, InputSample, ModelOutput, ScoreBreakdown


def _clamp_zero_one(value: float) -> float:
    """将数值限制在 [0, 1]。"""

    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _word_count(text: str) -> int:
    """统计中文长文本的长度单元。

    参数:
        text: 待统计文本。

    返回:
        int：长度单元数量。

    关键实现细节:
        对中文连续文本按 CJK 字符计数，对英文/数字片段按单词计数，
        使 benchmark 中“目标字数”与实际评分口径更一致。
    """

    stripped_text = text.strip()
    if stripped_text == "":
        return 0

    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", stripped_text))
    latin_token_count = len(re.findall(r"[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)*", stripped_text))
    return cjk_count + latin_token_count


def compute_s_stability(eval_metrics: EvalMetrics) -> float:
    """计算稳定性分数。"""

    score_value = (
        0.4 * eval_metrics.completion_rate
        + 0.2 * eval_metrics.acc_once
        + 0.2 * eval_metrics.acc_range
        + 0.2 * eval_metrics.acc_periodic
    )
    return _clamp_zero_one(score_value)


def compute_s_quality(eval_metrics: EvalMetrics) -> float:
    """计算质量分数。"""

    normalized_scores: List[float] = []
    for dim_score in eval_metrics.quality_scores.values():
        normalized_scores.append((dim_score - 1.0) / 4.0)
    if len(normalized_scores) == 0:
        raise ValueError("quality_scores 为空，无法计算 S_quality")
    return _clamp_zero_one(sum(normalized_scores) / len(normalized_scores))


def compute_s_length(required_length_words: int, response_text: str) -> float:
    """计算长度控制分数。"""

    output_word_count = _word_count(response_text)
    if output_word_count <= 0:
        raise ValueError("response 文本词数为 0，无法计算 S_length")

    if required_length_words <= 0:
        raise ValueError("required_length_words 必须大于 0")

    length_ratio = output_word_count / required_length_words

    # 分段惩罚：<50%、50-80%、80-120%、>120%
    # 使用不同斜率，避免“轻微偏短”和“严重过短”被同样惩罚。
    if length_ratio < 0.5:
        # 严重过短：惩罚最重，分数上限控制在 0.4
        length_score = 0.8 * length_ratio
    elif length_ratio < 0.8:
        # 中度偏短：线性恢复到 0.8
        length_score = 0.4 + (length_ratio - 0.5) * (4.0 / 3.0)
    elif length_ratio <= 1.2:
        # 合理区间：轻惩罚，中心点(100%)最高分
        length_score = 1.0 - abs(length_ratio - 1.0) * 0.5
    else:
        # 明显过长：按独立斜率下降，避免与偏短区间混淆
        length_score = 0.9 - (length_ratio - 1.2) * 0.5
    return _clamp_zero_one(length_score)


def compute_s_info(eval_metrics: EvalMetrics, response_text: str) -> float:
    """计算信息密度与有效性分数。"""

    output_word_count = _word_count(response_text)
    if output_word_count <= 0:
        raise ValueError("response 文本词数为 0，无法计算 S_info")

    proxy_qa_acc = eval_metrics.proxy_qa_correct / eval_metrics.proxy_qa_total
    info_density = eval_metrics.proxy_qa_correct / output_word_count
    info_score = min(1.0, 0.8 * proxy_qa_acc + 0.2 * min(1.0, 300.0 * info_density))
    return _clamp_zero_one(info_score)


def compute_s_instruction(eval_metrics: EvalMetrics) -> float:
    """计算指令遵循分数（硬约束/软约束）。

    规则:
    - 硬约束优先，软约束提供细粒度区分。
    - 若硬约束未全满足，施加明显惩罚，避免“全有或全无”之外仍缺少惩戒强度。
    """

    if eval_metrics.instruction_hard_total > 0:
        hard_score = eval_metrics.instruction_hard_hits / eval_metrics.instruction_hard_total
    else:
        hard_score = 1.0

    if eval_metrics.instruction_soft_total > 0:
        soft_score = eval_metrics.instruction_soft_hits / eval_metrics.instruction_soft_total
    else:
        soft_score = 1.0

    if hard_score < 1.0:
        # 硬约束未满足时，先按硬约束主导聚合，再施加额外惩罚
        base_score = 0.8 * hard_score + 0.2 * soft_score
        instruction_score = base_score * 0.6
    else:
        # 硬约束全满足后，软约束负责细粒度拉开差距
        instruction_score = 0.7 * hard_score + 0.3 * soft_score

    return _clamp_zero_one(instruction_score)


def compute_s_structure(eval_metrics: EvalMetrics) -> float:
    """计算结构完整性分数。"""

    structure_score = 0.5 * eval_metrics.syntax_pass_rate + 0.5 * eval_metrics.schema_pass_rate
    return _clamp_zero_one(structure_score)


def compute_raw_cost(latency_seconds: float, total_tokens: int) -> float:
    """计算原始成本分。"""

    if latency_seconds <= 0:
        raise ValueError("latency_seconds 必须大于 0")
    if total_tokens <= 0:
        raise ValueError("total_tokens 必须大于 0")

    return 1.0 / (1.0 + latency_seconds / 3600.0 + total_tokens / 1000000.0)


def compute_s_cost(raw_cost: float, baseline_raw_cost: float) -> float:
    """计算成本效率分。"""

    if baseline_raw_cost <= 0:
        raise ValueError("baseline_raw_cost 必须大于 0")
    return _clamp_zero_one(raw_cost / baseline_raw_cost)


def power_mean(values: List[float], power_p: float) -> float:
    """计算幂平均总分。"""

    if len(values) == 0:
        raise ValueError("values 不能为空")
    safe_values = [max(value, 1e-12) for value in values]
    if power_p == 0.0:
        return math.exp(sum(math.log(value) for value in safe_values) / len(safe_values))
    powered_average = sum(value ** power_p for value in safe_values) / len(safe_values)
    return powered_average ** (1.0 / power_p)


def score_one_sample(
    input_sample: InputSample,
    model_output: ModelOutput,
    eval_metrics: EvalMetrics,
    baseline_raw_cost: float,
    power_p: float,
) -> Tuple[ScoreBreakdown, Optional[float], Dict[str, object]]:
    """计算单样本七维分与总分。

    参数:
        input_sample: 输入样本。
        model_output: 模型输出。
        eval_metrics: 中间评测指标。
        baseline_raw_cost: 任务成本基线。
        power_p: 幂平均参数。

    返回:
        七维分数、总分、诊断信息。
    """

    # 第一阶段：七维主分计算
    s_stability = compute_s_stability(eval_metrics)
    s_quality = compute_s_quality(eval_metrics)
    s_length = compute_s_length(input_sample.constraints.required_length_words, model_output.response)
    s_info = compute_s_info(eval_metrics, model_output.response)
    s_instruction = compute_s_instruction(eval_metrics)
    s_structure = compute_s_structure(eval_metrics)

    raw_cost_value = compute_raw_cost(model_output.latency_seconds, model_output.token_usage.total_tokens)
    s_cost = compute_s_cost(raw_cost_value, baseline_raw_cost)

    score_breakdown = ScoreBreakdown(
        s_stability=s_stability,
        s_quality=s_quality,
        s_length=s_length,
        s_info=s_info,
        s_instruction=s_instruction,
        s_structure=s_structure,
        s_cost=s_cost,
    )

    # 第二阶段：总分聚合
    score_values = [
        s_stability,
        s_quality,
        s_length,
        s_info,
        s_instruction,
        s_structure,
        s_cost,
    ]
    if any(score is None for score in score_values):
        overall_score = None
    else:
        typed_values: List[float] = [float(score) for score in score_values]
        overall_score = power_mean(typed_values, power_p)

    # 第三阶段：诊断信息写入
    diagnostics: Dict[str, object] = {
        "required_length_words": input_sample.constraints.required_length_words,
        "response_word_count": _word_count(model_output.response),
        "length_ratio": _word_count(model_output.response) / input_sample.constraints.required_length_words,
        "instruction_hard_hits": eval_metrics.instruction_hard_hits,
        "instruction_hard_total": eval_metrics.instruction_hard_total,
        "instruction_soft_hits": eval_metrics.instruction_soft_hits,
        "instruction_soft_total": eval_metrics.instruction_soft_total,
        "proxy_qa_correct": eval_metrics.proxy_qa_correct,
        "proxy_qa_total": eval_metrics.proxy_qa_total,
        "usage_total_tokens": model_output.token_usage.usage_total_tokens,
        "tokenizer_total_tokens": model_output.token_usage.total_tokens,
        "raw_cost": raw_cost_value,
        "baseline_raw_cost": baseline_raw_cost,
        "score_breakdown": asdict(score_breakdown),
    }
    return score_breakdown, overall_score, diagnostics