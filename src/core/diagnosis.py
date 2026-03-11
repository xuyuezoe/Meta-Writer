"""
诊断结果数据类：ErrorTier、ErrorSource、DecodingConfig、DiagnosisResult

功能：
    定义 MRSD 算法的输出格式，包含两轴错误分类、修复作用域、
    解码配置和诊断置信度。

依赖：无
被依赖：MRSD（生成）、OnlineValidator（读取策略）、Orchestrator（执行修复）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class ErrorTier(Enum):
    """错误层级（两轴分类的维度一）"""
    PLAN        = "plan"
    DECISION    = "decision"
    REALIZATION = "realization"
    STATE       = "state"


class ErrorSource(Enum):
    """错误来源（两轴分类的维度二）"""
    INTRINSIC  = "intrinsic"   # 当前节自身产生
    PROPAGATED = "propagated"  # 上游污染传播
    AMBIGUOUS  = "ambiguous"   # 预算内无法区分


@dataclass
class DecodingConfig:
    """
    由错误类型决定的生成配置

    功能：
        MRSD 输出的一部分，决定下一次生成调用的参数。
        这不是效率优化，而是修复机制正确性的一部分。

    参数：
        temperature: 生成温度
        use_two_step_generation: 是否先生成大纲再展开（约束密集节）
        strengthen_dsl_injection: 是否强化 DSL 条目注入
        trigger_section_intent_revision: 是否先重规划 Section Intent

    关键实现细节：
        realization → 高温（0.8+），鼓励多样性
        decision    → 低温（0.3），强约束保守重写
        state       → 低温，强化 DSL 注入，purge 后生成
        plan        → 先重规划 Section Intent，再低温生成
    """
    temperature: float
    use_two_step_generation: bool
    strengthen_dsl_injection: bool
    trigger_section_intent_revision: bool

    @classmethod
    def for_tier(cls, tier: ErrorTier) -> "DecodingConfig":
        """
        根据错误层级构造对应的解码配置

        返回：对应错误类型的 DecodingConfig 实例
        """
        configs: Dict[ErrorTier, "DecodingConfig"] = {
            ErrorTier.REALIZATION: cls(
                temperature=0.85,
                use_two_step_generation=False,
                strengthen_dsl_injection=False,
                trigger_section_intent_revision=False,
            ),
            ErrorTier.DECISION: cls(
                temperature=0.3,
                use_two_step_generation=False,
                strengthen_dsl_injection=True,
                trigger_section_intent_revision=False,
            ),
            ErrorTier.STATE: cls(
                temperature=0.3,
                use_two_step_generation=False,
                strengthen_dsl_injection=True,
                trigger_section_intent_revision=False,
            ),
            ErrorTier.PLAN: cls(
                temperature=0.3,
                use_two_step_generation=True,
                strengthen_dsl_injection=True,
                trigger_section_intent_revision=True,
            ),
        }
        return configs[tier]


@dataclass
class DiagnosisResult:
    """
    MRSD 诊断输出

    功能：
        包含两轴错误分类、最小责任子图、修复作用域和解码配置。
        通过 should_rollback() 实现置信度门控，避免不确定的诊断
        触发确定的破坏性操作。

    参数：
        error_tier: 错误层级（ErrorTier 枚举）
        error_source: 错误来源（ErrorSource 枚举）
        repair_scope: 修复范围（local_rewrite / partial_rollback / memory_purge）
        target_section: 回退目标节 ID（仅 partial_rollback 时使用）
        causal_subgraph: 责任节点集合（decision_id 列表）
        confidence: 整体诊断置信度 [0, 1]
        decoding_config: 下一次生成的配置参数
        evidence: 支持诊断的证据描述列表
        alternative_repairs: 备选修复方案及其概率 [(scope, prob), ...]

    返回值：
        should_rollback(): bool，是否执行回退
    """
    error_tier: ErrorTier
    error_source: ErrorSource
    repair_scope: str
    target_section: Optional[str]
    causal_subgraph: List[str]
    confidence: float
    decoding_config: DecodingConfig
    evidence: List[str]
    alternative_repairs: List[Tuple[str, float]] = field(default_factory=list)

    def should_rollback(self) -> bool:
        """
        是否执行回退的门控逻辑

        关键实现细节：
            置信度 < 0.6 时不回退（不确定的诊断不应触发破坏性操作）
            error_source 为 ambiguous 时不回退（来源不明时保守处理，防止过多的llm资源消耗）
        """
        return (
            self.repair_scope == "partial_rollback"
            and self.confidence >= 0.6
            and self.error_source != ErrorSource.AMBIGUOUS
        )

    def to_dict(self) -> Dict:
        """序列化为字典（用于日志记录）"""
        return {
            "error_tier":       self.error_tier.value,
            "error_source":     self.error_source.value,
            "repair_scope":     self.repair_scope,
            "target_section":   self.target_section,
            "causal_subgraph":  self.causal_subgraph,
            "confidence":       round(self.confidence, 3),
            "evidence":         self.evidence,
        }
