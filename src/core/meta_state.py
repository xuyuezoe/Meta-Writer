"""
元认知状态：MetaState

功能：
    Orchestrator 对"当前自身状态"的显式建模。
    通过 gate_action() 接口对三类关键动作进行门控，
    使 MetaState 成为实质性控制变量而非观察面板。

依赖：无
被依赖：Orchestrator（在每次生成前更新，在策略选择时读取）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MetaState:
    """
    元认知调度器 - 为减少系统与模型的幻觉，构建元认知对系统自身不可靠性来源的显式建模，让系统显式的了解自己的能力边界

    功能：
        1. 建模系统各组件的当前可信度（不可靠性来源）
        2. 通过 gate_action() 明确控制三类关键动作
        3. 计算 EIV（预期干预价值），判断修复是否值得执行

    参数：
        memory_trust_level: DSL 整体可信度（所有活跃条目 trust_level 的均值）
        validator_stability_estimate: 验证器评分稳定性（方差的倒数）
        diagnosis_uncertainty_profile: 按 error_tier 分类的历史诊断准确率
        section_complexity_estimate: 当前节约束密度（约束数 / 节目标词数估计）
        expected_intervention_value: EIV = P(成功)×ΔQ - C(代价)
        high_risk_constraint_ids: 最可能在后续节被违反的约束 ID 列表
        contamination_risk_score: 当前节的上游污染风险 [0, 1]（用于 BCP 软短路）
        frequent_failure_tiers: 按 error_tier 统计的历史失败次数
        frequent_failure_sources: 按 error_source 统计的历史失败次数
        remaining_rollback_budget: 剩余可用回退次数
        remaining_retry_budget: 当前节剩余可用重试次数
        validator_score_history: 验证器最近 5 次 DCAS 评分（用于计算稳定性）

    关键实现细节：
        validator_stability_estimate 由最近 5 次 DCAS 评分方差的倒数计算
        EIV = P(修复成功) × ΔQ(质量提升) - C(额外 LLM 开销)
        EIV < 0 时修复在期望意义上不划算，gate_action("allow_rollback") 返回 False
    """
    memory_trust_level: float = 1.0
    validator_stability_estimate: float = 1.0
    diagnosis_uncertainty_profile: Dict[str, float] = field(default_factory=dict)
    section_complexity_estimate: float = 0.5
    expected_intervention_value: float = 1.0

    high_risk_constraint_ids: List[str] = field(default_factory=list)
    contamination_risk_score: float = 0.0

    frequent_failure_tiers: Dict[str, int] = field(default_factory=dict)
    frequent_failure_sources: Dict[str, int] = field(default_factory=dict)

    remaining_rollback_budget: int = 5
    remaining_retry_budget: int = 3

    validator_score_history: List[float] = field(default_factory=list)

    def gate_action(self, action: str) -> bool:
        """
        元认知行为门控接口

        功能：
            明确控制三类关键动作，使 MetaState 从观察面板变为实质性控制变量。

        参数：
            action: 动作名称，三种合法值：
                "trust_validator_major" — 是否信任验证器触发 MAJOR 级别修复
                "allow_rollback"        — 是否允许执行回退操作
                "strengthen_dsl_injection" — 是否强化 DSL 注入

        返回值：
            True 表示允许执行该动作，False 表示拒绝/降级

        关键实现细节：
            "trust_validator_major"：
                validator_stability_estimate < 0.5 时返回 False
                （验证器本身不稳定时，MAJOR 可能是误报）

            "allow_rollback"：
                remaining_rollback_budget <= 0 时返回 False
                或 当前 error_tier 的诊断准确率 < 0.4 时返回 False
                或 expected_intervention_value < 0 时返回 False
                （修复期望收益为负时，不执行破坏性操作）

            "strengthen_dsl_injection"：
                memory_trust_level < 0.5 时返回 True
                或 contamination_risk_score > 0.6 时返回 True
        """
        if action == "trust_validator_major":
            return self.validator_stability_estimate >= 0.5

        elif action == "allow_rollback":
            if self.remaining_rollback_budget <= 0:
                return False
            if self.expected_intervention_value < 0:
                return False
            current_diag_accuracy = min(
                self.diagnosis_uncertainty_profile.values(),
                default=1.0,
            )
            return current_diag_accuracy >= 0.4

        elif action == "strengthen_dsl_injection":
            return (
                self.memory_trust_level < 0.5
                or self.contamination_risk_score > 0.6
            )

        return True

    def update_validator_stability(self, new_dcas_score: float) -> None:
        """
        更新验证器稳定性估计

        参数：
            new_dcas_score: 最新一次 DCAS 总分

        关键实现细节：
            保留最近 5 次评分，计算方差，稳定性 = 1 / (1 + variance)
        """
        self.validator_score_history.append(new_dcas_score)
        if len(self.validator_score_history) > 5:
            self.validator_score_history.pop(0)

        if len(self.validator_score_history) >= 2:
            mean = sum(self.validator_score_history) / len(self.validator_score_history)
            variance = sum(
                (s - mean) ** 2 for s in self.validator_score_history
            ) / len(self.validator_score_history)
            self.validator_stability_estimate = 1.0 / (1.0 + variance)

    def update_eiv(
        self,
        current_tier: str,
        current_dcas: float,
        dcas_threshold: float,
        estimated_extra_calls: int,
        avg_tokens_per_call: int,
        token_budget: int,
    ) -> None:
        """
        更新预期干预价值（EIV）

        参数：
            current_tier: 当前错误层级字符串
            current_dcas: 当前 DCAS 总分
            dcas_threshold: DCAS 通过阈值
            estimated_extra_calls: 预计需要的额外 LLM 调用次数
            avg_tokens_per_call: 每次调用平均 token 数
            token_budget: 当前剩余 token 预算

        关键实现细节：
            P(成功) ≈ 当前错误层级的历史诊断准确率
            ΔQ ≈ 当前 DCAS 到阈值的差距
            C ≈ 额外调用 token 数 / 总 token 预算
        """
        p_success = self.diagnosis_uncertainty_profile.get(current_tier, 0.5)
        delta_q = max(0.0, dcas_threshold - current_dcas)
        c_cost = (estimated_extra_calls * avg_tokens_per_call) / max(token_budget, 1)
        self.expected_intervention_value = p_success * delta_q - c_cost

    def record_failure(self, error_tier: str, error_source: str) -> None:
        """记录失败事件，更新历史失败模式"""
        self.frequent_failure_tiers[error_tier] = (
            self.frequent_failure_tiers.get(error_tier, 0) + 1
        )
        self.frequent_failure_sources[error_source] = (
            self.frequent_failure_sources.get(error_source, 0) + 1
        )

    def record_diagnosis_outcome(self, error_tier: str, was_correct: bool) -> None:
        """
        记录诊断结果，更新诊断准确率

        参数：
            error_tier: 错误层级字符串
            was_correct: 该诊断事后是否正确（修复是否成功）
        """
        current = self.diagnosis_uncertainty_profile.get(error_tier, 0.5)
        alpha = 0.3
        self.diagnosis_uncertainty_profile[error_tier] = (
            current * (1 - alpha) + (1.0 if was_correct else 0.0) * alpha
        )

    def update_contamination_risk(
        self,
        low_trust_ref_ratio: float,
        recent_same_tier_failure_rate: float,
    ) -> None:
        """
        更新上游污染风险分数

        参数：
            low_trust_ref_ratio: 当前节引用的低 trust DSL 条目比例
            recent_same_tier_failure_rate: 最近 2 节中相同层级失败频率

        关键实现细节：
            CRS = 0.4 × low_trust_ref_ratio
                + 0.3 × recent_same_tier_failure_rate
                + 0.3 × (1 - memory_trust_level)
            CRS < 0.3 时 BCP 可以短路；CRS >= 0.3 时进入轻量溯源

        三个风险因子的设计逻辑：

        1.low_trust_ref_ratio (40% 权重)：

            表示当前节生成时引用的 DSL 条目中，低信任度条目的比例
            核心指标：直接反映当前生成依赖的约束质量
            如果大量使用不可信约束，生成内容很可能受污染
        2.recent_same_tier_failure_rate (30% 权重)：

        表示最近几节中相同错误层级的失败频率
        模式识别：如果近期频繁出现同类型错误，可能存在系统性污染
        避免将偶然失败误判为污染
        3.(1.0 - memory_trust_level) (30% 权重)：

        DSL 整体信任度的倒数
        全局污染指示：如果整个 DSL 都不太可信，那么局部污染风险更高

        通过综合这三个因子，MetaState 能够动态评估当前节的上游污染风险，从而指导是否需要启用更严格的修复策略（如强化 DSL 注入）或直接拒绝修复操作（如拒绝回退）。
        """
        self.contamination_risk_score = (
            0.4 * low_trust_ref_ratio
            + 0.3 * recent_same_tier_failure_rate
            + 0.3 * (1.0 - self.memory_trust_level)
        )

    def to_dict(self) -> Dict:
        """序列化为字典（用于日志记录）"""
        return {
            "memory_trust_level":          round(self.memory_trust_level, 3),
            "validator_stability":         round(self.validator_stability_estimate, 3),
            "section_complexity":          round(self.section_complexity_estimate, 3),
            "expected_intervention_value": round(self.expected_intervention_value, 3),
            "contamination_risk_score":    round(self.contamination_risk_score, 3),
            "remaining_rollback_budget":   self.remaining_rollback_budget,
            "remaining_retry_budget":      self.remaining_retry_budget,
        }
