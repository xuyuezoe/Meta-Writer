"""MetaBench 数据结构定义。

该模块负责定义评测输入、输出与中间指标的数据结构，
用于保证流水线各阶段字段一致性与可追溯性。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ConstraintSpec:
    """约束配置。

    参数:
        required_length_words: 目标输出词数。
        must_include: 必须包含元素列表。
        periodic_requirements: 周期性要求列表。
    """

    required_length_words: int
    must_include: List[str]
    periodic_requirements: List[str]


@dataclass
class ProxyQuestion:
    """代理问题定义。

    参数:
        qid: 问题唯一标识。
        question: 问题文本。
        answer: 标准答案。
    """

    qid: str
    question: str
    answer: str


@dataclass
class InputSample:
    """评测输入样本。

    参数:
        sample_id: 样本唯一标识。
        task_type: 任务类型。
        prompt: 用户输入。
        constraints: 约束定义。
        proxy_questions: 代理问题列表。
        checklist: 检查项列表。
    """

    sample_id: str
    task_type: str
    prompt: str
    constraints: ConstraintSpec
    proxy_questions: List[ProxyQuestion]
    checklist: List[str]


@dataclass
class TokenUsage:
    """Token 统计结果。

    参数:
        prompt_tokens: 输入 token 数。
        completion_tokens: 输出 token 数。
        total_tokens: 总 token 数。
        usage_total_tokens: 接口返回 token 数，仅用于对账。
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    usage_total_tokens: Optional[int]


@dataclass
class ModelOutput:
    """模型输出样本。

    参数:
        sample_id: 样本唯一标识。
        model_name: 模型名称。
        response: 模型输出文本。
        latency_seconds: 推理耗时。
        token_usage: token 统计信息。
    """

    sample_id: str
    model_name: str
    response: str
    latency_seconds: float
    token_usage: TokenUsage


@dataclass
class EvalMetrics:
    """样本级中间评测指标。

    参数:
        completion_rate: 结构完成率。
        acc_once: 单次约束命中率。
        acc_range: 区间约束命中率。
        acc_periodic: 周期约束命中率。
        quality_scores: LLM 裁判分数字典，分值范围为 1-5。
        instruction_hits: 指令命中数量（硬约束+软约束汇总）。
        instruction_total: 指令总数（硬约束+软约束汇总）。
        instruction_hard_hits: 硬约束命中数量。
        instruction_hard_total: 硬约束总数。
        instruction_soft_hits: 软约束命中数量。
        instruction_soft_total: 软约束总数。
        syntax_pass_rate: 语法通过率。
        schema_pass_rate: 结构校验通过率。
        proxy_qa_correct: 代理问答答对数量。
        proxy_qa_total: 代理问答总题数。
    """

    completion_rate: float
    acc_once: float
    acc_range: float
    acc_periodic: float
    quality_scores: Dict[str, float]
    instruction_hits: int
    instruction_total: int
    instruction_hard_hits: int
    instruction_hard_total: int
    instruction_soft_hits: int
    instruction_soft_total: int
    syntax_pass_rate: float
    schema_pass_rate: float
    proxy_qa_correct: int
    proxy_qa_total: int


@dataclass
class ScoreBreakdown:
    """七维打分结果。

    参数:
        s_stability: 稳定性分数。
        s_quality: 质量分数。
        s_length: 长度控制分数。
        s_info: 信息密度与有效性分数。
        s_instruction: 指令遵循分数。
        s_structure: 结构完整性分数。
        s_cost: 成本效率分数。
    """

    s_stability: Optional[float]
    s_quality: Optional[float]
    s_length: Optional[float]
    s_info: Optional[float]
    s_instruction: Optional[float]
    s_structure: Optional[float]
    s_cost: Optional[float]


@dataclass
class SampleResult:
    """样本级最终结果。

    参数:
        sample_id: 样本唯一标识。
        task_type: 任务类型。
        model_name: 模型名称。
        scores: 七维分数明细。
        overall_score: 总分。
        diagnostics: 诊断信息。
    """

    sample_id: str
    task_type: str
    model_name: str
    scores: ScoreBreakdown
    overall_score: Optional[float]
    diagnostics: Dict[str, object]
