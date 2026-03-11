"""
评估指标收集器：MetricCollector

功能：
    在系统运行过程中收集四组内部机制指标（Internal Metrics），
    支持诊断可靠性、修复效率、记忆有效性和干预遗憾的事后分析。

依赖：无外部依赖（纯数据收集）
被依赖：Orchestrator（在关键事件发生时调用 record_*() 方法）
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class DiagnosisEvent:
    """
    单次 MRSD 诊断事件记录

    参数：
        event_id: 事件唯一标识（时间戳+节ID）
        section_id: 被诊断的节 ID
        predicted_tier: MRSD 预测的错误层级
        predicted_source: MRSD 预测的错误来源
        confidence: 诊断置信度
        repair_scope: 修复范围
        repair_succeeded: 修复是否成功（后验写入）
        ground_truth_tier: 人工标注根因层级（可选，用于 H3 验证）
        causal_subgraph_size: 责任子图节点数
        llm_calls_used: 本次诊断消耗的 LLM 调用次数
        timestamp: 事件时间戳
    """
    event_id: str
    section_id: str
    predicted_tier: str
    predicted_source: str
    confidence: float
    repair_scope: str
    causal_subgraph_size: int
    llm_calls_used: int
    timestamp: int

    repair_succeeded: Optional[bool] = None
    ground_truth_tier: Optional[str] = None


@dataclass
class RepairEvent:
    """
    单次修复操作事件记录

    参数：
        event_id: 事件唯一标识
        section_id: 被修复的节 ID
        repair_scope: 修复类型（local_rewrite / partial_rollback / memory_purge）
        triggered_by_tier: 触发修复的错误层级
        triggered_by_confidence: 触发诊断的置信度
        succeeded: 修复是否成功
        extra_llm_calls: 本次修复额外消耗的 LLM 调用次数
        rollback_distance: 回退距离（partial_rollback 时的节数，其他为 0）
        was_false_rollback: 是否为误触发回退（后验判断）
        timestamp: 事件时间戳
    """
    event_id: str
    section_id: str
    repair_scope: str
    triggered_by_tier: str
    triggered_by_confidence: float
    succeeded: bool
    extra_llm_calls: int
    rollback_distance: int
    timestamp: int

    was_false_rollback: bool = False


@dataclass
class DslMemorySnapshot:
    """
    DSL 记忆状态快照（每节生成后记录）

    参数：
        section_id: 快照对应的节 ID
        total_entries: 当前活跃条目总数
        high_stability_count: stability_score > 0.7 的条目数
        revoked_count: 本次生成周期内被 revoke 的条目数
        open_loop_count: 未闭合 OPEN_LOOP 数量
        closed_loop_count: 本节闭合的 OPEN_LOOP 数量
        memory_trust_level: 当前 DSL 整体可信度
        timestamp: 快照时间戳
    """
    section_id: str
    total_entries: int
    high_stability_count: int
    revoked_count: int
    open_loop_count: int
    closed_loop_count: int
    memory_trust_level: float
    timestamp: int


class MetricCollector:
    """
    内部机制指标收集器

    功能：
        在 Orchestrator 的关键执行点收集事件数据，
        事后通过 compute_*() 方法计算统计指标。

    指标组（对应 plan.md 5.1 节）：
        G1 诊断可靠性：MRSD 责任节点与人工标注一致率；置信度-成功率相关性
        G2 修复效率：每错误额外 LLM 调用次数；False Rollback Rate；Over-Correction Rate
        G3 记忆有效性：高稳定性条目比例；污染率；OPEN_LOOP 闭合率
        G4 干预遗憾：EIV 与实际修复结果相关性

    参数：
        无

    关键实现细节：
        所有 record_*() 方法在 Orchestrator 关键事件处调用，不修改系统行为。
        compute_*() 方法在运行结束后调用，进行事后统计。
    """

    def __init__(self) -> None:
        self._diagnosis_events: List[DiagnosisEvent] = []
        self._repair_events: List[RepairEvent] = []
        self._dsl_snapshots: List[DslMemorySnapshot] = []
        self._eiv_records: List[Tuple[float, bool]] = []  # (eiv, repair_succeeded)
        self._section_first_pass: Dict[str, bool] = {}   # section_id → 首次通过

        # 总 LLM 调用计数（由 Orchestrator 更新）
        self.total_llm_calls: int = 0
        self.total_sections_generated: int = 0

    # ------------------------------------------------------------------
    # 第一组：诊断事件记录
    # ------------------------------------------------------------------

    def record_diagnosis(
        self,
        section_id: str,
        predicted_tier: str,
        predicted_source: str,
        confidence: float,
        repair_scope: str,
        causal_subgraph_size: int,
        llm_calls_used: int,
    ) -> str:
        """
        记录一次 MRSD 诊断事件

        参数：
            section_id: 被诊断的节 ID
            predicted_tier: 预测错误层级（plan/decision/realization/state）
            predicted_source: 预测来源（intrinsic/propagated/ambiguous）
            confidence: 诊断置信度
            repair_scope: 修复范围
            causal_subgraph_size: 责任子图节点数
            llm_calls_used: 本次诊断消耗的 LLM 调用次数

        返回值：
            str：事件 ID（用于后续写入 repair_succeeded 和 ground_truth_tier）
        """
        event_id = f"{int(time.time())}_{section_id}_{len(self._diagnosis_events)}"
        event = DiagnosisEvent(
            event_id=event_id,
            section_id=section_id,
            predicted_tier=predicted_tier,
            predicted_source=predicted_source,
            confidence=confidence,
            repair_scope=repair_scope,
            causal_subgraph_size=causal_subgraph_size,
            llm_calls_used=llm_calls_used,
            timestamp=int(time.time()),
        )
        self._diagnosis_events.append(event)
        return event_id

    def record_diagnosis_outcome(self, event_id: str, succeeded: bool) -> None:
        """
        后验写入诊断结果（修复是否成功）

        参数：
            event_id: 诊断事件 ID（由 record_diagnosis 返回）
            succeeded: 修复是否成功（下一轮验证通过即为 True）
        """
        for event in self._diagnosis_events:
            if event.event_id == event_id:
                event.repair_succeeded = succeeded
                return

    def record_ground_truth(self, event_id: str, ground_truth_tier: str) -> None:
        """
        写入人工标注根因层级（H3 假设验证用）

        参数：
            event_id: 诊断事件 ID
            ground_truth_tier: 人工标注的真实错误层级
        """
        for event in self._diagnosis_events:
            if event.event_id == event_id:
                event.ground_truth_tier = ground_truth_tier
                return

    # ------------------------------------------------------------------
    # 第二组：修复事件记录
    # ------------------------------------------------------------------

    def record_repair(
        self,
        section_id: str,
        repair_scope: str,
        triggered_by_tier: str,
        triggered_by_confidence: float,
        succeeded: bool,
        extra_llm_calls: int,
        rollback_distance: int = 0,
    ) -> str:
        """
        记录一次修复操作事件

        参数：
            section_id: 被修复的节 ID
            repair_scope: 修复类型
            triggered_by_tier: 触发此修复的错误层级
            triggered_by_confidence: 触发诊断的置信度
            succeeded: 修复是否成功
            extra_llm_calls: 本次修复额外的 LLM 调用次数
            rollback_distance: 回退距离（节数，非回退时为 0）

        返回值：
            str：修复事件 ID
        """
        event_id = f"repair_{int(time.time())}_{len(self._repair_events)}"
        event = RepairEvent(
            event_id=event_id,
            section_id=section_id,
            repair_scope=repair_scope,
            triggered_by_tier=triggered_by_tier,
            triggered_by_confidence=triggered_by_confidence,
            succeeded=succeeded,
            extra_llm_calls=extra_llm_calls,
            rollback_distance=rollback_distance,
            timestamp=int(time.time()),
        )
        self._repair_events.append(event)
        return event_id

    def mark_false_rollback(self, event_id: str) -> None:
        """
        标记某次回退为误触发（False Rollback）

        参数：
            event_id: 修复事件 ID
        """
        for event in self._repair_events:
            if event.event_id == event_id:
                event.was_false_rollback = True
                return

    def record_section_first_pass(self, section_id: str, passed_on_first_try: bool) -> None:
        """
        记录节是否在第一次尝试就通过验证

        参数：
            section_id: 节 ID
            passed_on_first_try: 是否首次通过
        """
        self._section_first_pass[section_id] = passed_on_first_try
        self.total_sections_generated += 1

    # ------------------------------------------------------------------
    # 第三组：DSL 记忆快照
    # ------------------------------------------------------------------

    def record_dsl_snapshot(
        self,
        section_id: str,
        total_entries: int,
        high_stability_count: int,
        revoked_count: int,
        open_loop_count: int,
        closed_loop_count: int,
        memory_trust_level: float,
    ) -> None:
        """
        记录当前 DSL 记忆状态快照

        参数：
            section_id: 快照对应的节 ID
            total_entries: 当前活跃条目数
            high_stability_count: stability_score > 0.7 的条目数
            revoked_count: 本节周期内被 revoke 的条目数
            open_loop_count: 未闭合 OPEN_LOOP 数量
            closed_loop_count: 本节闭合的 OPEN_LOOP 数量
            memory_trust_level: DSL 整体可信度
        """
        self._dsl_snapshots.append(DslMemorySnapshot(
            section_id=section_id,
            total_entries=total_entries,
            high_stability_count=high_stability_count,
            revoked_count=revoked_count,
            open_loop_count=open_loop_count,
            closed_loop_count=closed_loop_count,
            memory_trust_level=memory_trust_level,
            timestamp=int(time.time()),
        ))

    # ------------------------------------------------------------------
    # 第四组：EIV 干预遗憾记录
    # ------------------------------------------------------------------

    def record_eiv_outcome(self, eiv: float, repair_succeeded: bool) -> None:
        """
        记录 EIV 与实际修复结果的对应关系（H4/干预遗憾分析用）

        参数：
            eiv: 本次修复前的 EIV 值
            repair_succeeded: 实际修复是否成功
        """
        self._eiv_records.append((eiv, repair_succeeded))

    # ------------------------------------------------------------------
    # 指标计算：G1 诊断可靠性
    # ------------------------------------------------------------------

    def compute_diagnosis_reliability(self) -> Dict:
        """
        计算 G1 诊断可靠性指标

        功能：
            - MRSD 责任节点与人工标注根因一致率（需 ground_truth_tier 数据）
            - 诊断置信度与修复成功率的 Pearson 相关系数

        返回值：
            Dict：包含 tier_accuracy、confidence_success_correlation 等指标
        """
        annotated = [
            e for e in self._diagnosis_events
            if e.ground_truth_tier is not None
        ]
        tier_accuracy: Optional[float] = None
        if annotated:
            correct = sum(
                1 for e in annotated
                if e.predicted_tier == e.ground_truth_tier
            )
            tier_accuracy = correct / len(annotated)

        # 置信度-成功率 Pearson 相关
        completed = [
            e for e in self._diagnosis_events
            if e.repair_succeeded is not None
        ]
        correlation: Optional[float] = None
        if len(completed) >= 3:
            confidences = [e.confidence for e in completed]
            successes = [1.0 if e.repair_succeeded else 0.0 for e in completed]
            correlation = self._pearson(confidences, successes)

        return {
            "annotated_events": len(annotated),
            "tier_accuracy": tier_accuracy,
            "confidence_success_correlation": correlation,
            "avg_diagnosis_confidence": (
                sum(e.confidence for e in self._diagnosis_events) / len(self._diagnosis_events)
                if self._diagnosis_events else None
            ),
            "avg_causal_subgraph_size": (
                sum(e.causal_subgraph_size for e in self._diagnosis_events) / len(self._diagnosis_events)
                if self._diagnosis_events else None
            ),
        }

    # ------------------------------------------------------------------
    # 指标计算：G2 修复效率
    # ------------------------------------------------------------------

    def compute_repair_efficiency(self) -> Dict:
        """
        计算 G2 修复效率指标

        功能：
            - 首次成功率（First-Pass Rate）
            - 每修复一个真实错误的额外 LLM 调用次数
            - False Rollback Rate
            - 平均回退距离

        返回值：
            Dict：修复效率统计字典
        """
        first_pass_rate: Optional[float] = None
        if self._section_first_pass:
            passed = sum(1 for v in self._section_first_pass.values() if v)
            first_pass_rate = passed / len(self._section_first_pass)

        rollbacks = [e for e in self._repair_events if e.repair_scope == "partial_rollback"]
        false_rollback_rate: Optional[float] = None
        if rollbacks:
            false_count = sum(1 for e in rollbacks if e.was_false_rollback)
            false_rollback_rate = false_count / len(rollbacks)

        successful_repairs = [e for e in self._repair_events if e.succeeded]
        avg_extra_calls_per_fix: Optional[float] = None
        if successful_repairs:
            avg_extra_calls_per_fix = (
                sum(e.extra_llm_calls for e in successful_repairs) / len(successful_repairs)
            )

        avg_rollback_distance: Optional[float] = None
        if rollbacks:
            avg_rollback_distance = sum(e.rollback_distance for e in rollbacks) / len(rollbacks)

        return {
            "first_pass_rate": first_pass_rate,
            "total_repair_events": len(self._repair_events),
            "false_rollback_rate": false_rollback_rate,
            "avg_extra_llm_calls_per_fix": avg_extra_calls_per_fix,
            "avg_rollback_distance": avg_rollback_distance,
            "repair_scope_distribution": self._count_by(
                self._repair_events, lambda e: e.repair_scope
            ),
        }

    # ------------------------------------------------------------------
    # 指标计算：G3 记忆有效性
    # ------------------------------------------------------------------

    def compute_memory_effectiveness(self) -> Dict:
        """
        计算 G3 记忆有效性指标

        功能：
            - 高稳定性条目比例（stability > 0.7）
            - 记忆污染率（revoked 比例）
            - OPEN_LOOP 闭合率

        返回值：
            Dict：记忆有效性统计字典
        """
        if not self._dsl_snapshots:
            return {"snapshots": 0}

        last = self._dsl_snapshots[-1]
        avg_trust = sum(s.memory_trust_level for s in self._dsl_snapshots) / len(self._dsl_snapshots)
        total_revoked = sum(s.revoked_count for s in self._dsl_snapshots)
        total_closed = sum(s.closed_loop_count for s in self._dsl_snapshots)

        high_stability_ratio: Optional[float] = None
        if last.total_entries > 0:
            high_stability_ratio = last.high_stability_count / last.total_entries

        return {
            "snapshots": len(self._dsl_snapshots),
            "final_active_entries": last.total_entries,
            "high_stability_ratio": high_stability_ratio,
            "total_revoked_entries": total_revoked,
            "total_closed_open_loops": total_closed,
            "final_open_loops": last.open_loop_count,
            "avg_memory_trust_level": round(avg_trust, 3),
        }

    # ------------------------------------------------------------------
    # 指标计算：G4 干预遗憾
    # ------------------------------------------------------------------

    def compute_intervention_regret(self) -> Dict:
        """
        计算 G4 干预遗憾指标

        功能：
            - EIV 与实际修复结果的 Pearson 相关系数
            - EIV > 0 时修复成功的比例（EIV 预测准确率）

        返回值：
            Dict：干预遗憾统计字典
        """
        if not self._eiv_records:
            return {"eiv_records": 0}

        eiv_values = [r[0] for r in self._eiv_records]
        success_values = [1.0 if r[1] else 0.0 for r in self._eiv_records]

        correlation: Optional[float] = None
        if len(self._eiv_records) >= 3:
            correlation = self._pearson(eiv_values, success_values)

        positive_eiv = [(eiv, suc) for eiv, suc in self._eiv_records if eiv > 0]
        eiv_accuracy: Optional[float] = None
        if positive_eiv:
            eiv_accuracy = sum(1 for _, suc in positive_eiv if suc) / len(positive_eiv)

        return {
            "eiv_records": len(self._eiv_records),
            "eiv_success_correlation": correlation,
            "eiv_positive_accuracy": eiv_accuracy,
            "avg_eiv": sum(eiv_values) / len(eiv_values) if eiv_values else None,
        }

    # ------------------------------------------------------------------
    # 汇总报告
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        """
        生成完整的评估指标汇总报告

        返回值：
            Dict：包含四组指标的完整报告字典
        """
        return {
            "g1_diagnosis_reliability": self.compute_diagnosis_reliability(),
            "g2_repair_efficiency": self.compute_repair_efficiency(),
            "g3_memory_effectiveness": self.compute_memory_effectiveness(),
            "g4_intervention_regret": self.compute_intervention_regret(),
            "meta": {
                "total_llm_calls": self.total_llm_calls,
                "total_sections": self.total_sections_generated,
                "total_diagnosis_events": len(self._diagnosis_events),
                "total_repair_events": len(self._repair_events),
            },
        }

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _pearson(x: List[float], y: List[float]) -> Optional[float]:
        """
        计算两个列表的 Pearson 相关系数

        参数：
            x: 第一个数值列表
            y: 第二个数值列表

        返回值：
            float 或 None（样本不足或方差为零时）
        """
        n = len(x)
        if n < 3 or len(y) != n:
            return None
        mx = sum(x) / n
        my = sum(y) / n
        num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        denom_x = sum((xi - mx) ** 2 for xi in x) ** 0.5
        denom_y = sum((yi - my) ** 2 for yi in y) ** 0.5
        if denom_x == 0 or denom_y == 0:
            return None
        return num / (denom_x * denom_y)

    @staticmethod
    def _count_by(events: list, key_fn) -> Dict[str, int]:
        """
        按键函数统计事件分布

        参数：
            events: 事件列表
            key_fn: 提取分组键的函数

        返回值：
            Dict[str, int]：键 → 出现次数
        """
        result: Dict[str, int] = {}
        for e in events:
            k = key_fn(e)
            result[k] = result.get(k, 0) + 1
        return result
