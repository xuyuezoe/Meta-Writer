"""
最小责任子图诊断算法：MRSD（Minimal Responsibility Subgraph Diagnosis）

功能：
    给定验证失败事件，寻找能解释失败的最小决策责任子图。
    替代原启发式 DTGDebugger，提供形式化目标函数下的可计算诊断。

    核心流程（五步 BCP）：
        步骤0：约束类型前置过滤（SOFT/STATEFUL 合法更新过滤）
        步骤1：Realization 软短路（CRS 门控）
        步骤2：结构因果路径检测（DTG 路径分析，无 LLM 调用）
        步骤3：反向语义扫描（K=3 次 LLM 预算）
        步骤4：来源判定（无 LLM 调用）
        步骤5：构建 DiagnosisResult + DecodingConfig

依赖：DTGStore、LLMClient、core/diagnosis.py、core/decision.py、core/validation.py
被依赖：OnlineValidator（集成后替换 DTGDebugger 调用）、Orchestrator
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

from ..core.decision import Decision
from ..core.diagnosis import DecodingConfig, DiagnosisResult, ErrorSource, ErrorTier
from ..core.validation import ValidationReport
from ..memory.dtg_store import DTGStore
from ..utils.llm_client import LLMClient


class MRSD:
    """
    最小责任子图诊断器

    功能：
        实现五步 BCP 流程，输出两轴分类（Tier × Source）的诊断结果。
        操作性因果定义：哪些决策节点若被修改，当前失败最可能消失。
        结构因果（路径连通性）优先于纯语义匹配。

    参数：
        dtg_store: DTGStore 实例，提供决策历史和路径检测
        llm_client: LLM 客户端，用于反向语义扫描（最多 K=3 次调用）
        max_llm_budget: 反向语义扫描的 LLM 调用次数上限（默认 3）

    关键实现细节：
        score(G_r) = α|V_r| + β|E_r| + γ·uncertainty(G_r) - δ·explanation_coverage(G_r)
        α=0.1, β=0.05, γ=0.3, δ=1.0（最小化子图大小和不确定性，最大化解释覆盖）
        BCP 是上述目标的有预算贪心近似。
    """

    # 目标函数权重
    _ALPHA = 0.1     # 子图节点数惩罚
    _BETA  = 0.05    # 子图边数惩罚
    _GAMMA = 0.3     # 不确定性惩罚
    _DELTA = 1.0     # 解释覆盖奖励

    # LLM 置信度映射
    _CONF_EXPLICIT_CONFLICT = 0.8
    _CONF_IMPLICIT_OMISSION = 0.5

    # plan_level 触发需要的连续失败次数阈值
    _PLAN_TRIGGER_CONSECUTIVE_FAILURES = 2

    def __init__(
        self,
        dtg_store: DTGStore,
        llm_client: LLMClient,
        max_llm_budget: int = 3,
    ):
        self._dtg = dtg_store
        self._llm = llm_client
        self._max_llm_budget = max_llm_budget
        self.logger = logging.getLogger(__name__)

    def diagnose(
        self,
        report: ValidationReport,
        current_section_id: str,
        section_queue: List[str],
        contamination_risk_score: float,
        consecutive_failures_this_section: int,
        low_trust_dsl_ref_ratio: float,
        last_purge_succeeded: bool,
        intent_from_trusted_dsl: bool,
        recent_section_failure_tiers: List[str],
    ) -> DiagnosisResult:
        """
        对单次验证失败执行 MRSD 诊断

        功能：
            完整执行五步 BCP 流程，返回 DiagnosisResult。

        参数：
            report: OnlineValidator 产出的验证报告
            current_section_id: 当前失败节 ID
            section_queue: 全局节顺序列表
            contamination_risk_score: MetaState.contamination_risk_score（[0,1]）
            consecutive_failures_this_section: 当前节在同一 SectionIntent 下的连续失败次数
            low_trust_dsl_ref_ratio: 当前节引用的低信任 DSL 条目比例
            last_purge_succeeded: 上次 memory_purge 是否有效缓解了失败
            intent_from_trusted_dsl: 当前 Section Intent 是否从高信任 DSL 生成（trust > 0.6）
            recent_section_failure_tiers: 最近节的历史失败层级列表（用于 plan_level 判断）

        返回值：
            DiagnosisResult：包含两轴分类、修复范围、置信度和解码配置
        """
        # 第一步：识别失败维度
        failed_dims = self._identify_failed_dimensions(report)

        # 步骤0：约束类型前置过滤
        if self._is_soft_only_failure(report):
            return self._make_result(
                tier=ErrorTier.REALIZATION,
                source=ErrorSource.INTRINSIC,
                repair_scope="local_rewrite",
                target_section=None,
                causal_subgraph=[],
                confidence=0.9,
                evidence=["仅 SOFT 约束违反，不触发 MRSD"],
                failed_dims=failed_dims,
            )

        # 步骤1：Realization 软短路（低 Coverage 且低污染风险）
        if self._should_realization_short_circuit(report, contamination_risk_score):
            return self._make_result(
                tier=ErrorTier.REALIZATION,
                source=ErrorSource.INTRINSIC,
                repair_scope="local_rewrite",
                target_section=None,
                causal_subgraph=[],
                confidence=0.85,
                evidence=[
                    f"DCAS Coverage 低（score={report.dcas_score:.2f}）",
                    f"CRS={contamination_risk_score:.2f} < 0.3，上游污染风险低",
                ],
                failed_dims=failed_dims,
            )

        # 步骤2：结构因果路径检测（无 LLM 调用）
        structural_candidates = self._detect_structural_responsibility(
            current_section_id, section_queue
        )

        # 步骤3：反向语义扫描（K=3 次 LLM 预算）
        causal_nodes, evidence, node_confidences = self._backward_semantic_scan(
            report=report,
            current_section_id=current_section_id,
            section_queue=section_queue,
            structural_priority_ids=structural_candidates,
        )

        # 步骤4：来源判定
        error_source = self._determine_source(
            causal_nodes=causal_nodes,
            current_section_id=current_section_id,
            section_queue=section_queue,
            report=report,
        )

        # 步骤5：层级判定
        error_tier = self._determine_tier(
            causal_nodes=causal_nodes,
            report=report,
            consecutive_failures=consecutive_failures_this_section,
            failed_dims=failed_dims,
            low_trust_dsl_ref_ratio=low_trust_dsl_ref_ratio,
            last_purge_succeeded=last_purge_succeeded,
            intent_from_trusted_dsl=intent_from_trusted_dsl,
            recent_section_failure_tiers=recent_section_failure_tiers,
            contamination_risk_score=contamination_risk_score,
        )

        # 确定修复范围
        repair_scope, target_section = self._determine_repair_scope(
            tier=error_tier,
            source=error_source,
            causal_nodes=causal_nodes,
            current_section_id=current_section_id,
            section_queue=section_queue,
        )

        # 计算子图整体置信度
        confidence = self._compute_subgraph_confidence(
            causal_nodes=causal_nodes,
            node_confidences=node_confidences,
            failed_dims=failed_dims,
            structural_ids=structural_candidates,
        )

        return self._make_result(
            tier=error_tier,
            source=error_source,
            repair_scope=repair_scope,
            target_section=target_section,
            causal_subgraph=causal_nodes,
            confidence=confidence,
            evidence=evidence,
            failed_dims=failed_dims,
        )

    # ------------------------------------------------------------------
    # 第一阶段：失败维度识别
    # ------------------------------------------------------------------

    def _identify_failed_dimensions(self, report: ValidationReport) -> Set[str]:
        """
        从验证报告中识别失败维度

        功能：
            将验证报告映射到四类固定失败维度，用于 explanation_coverage 计算。

        参数：
            report: 验证报告

        返回值：
            Set[str]：失败维度集合（dim_coverage / dim_consistency / dim_constraint / dim_discourse）
        """
        dims: Set[str] = set()
        if report.dcas_score < 0.5:
            dims.add("dim_coverage")

        for issue in report.issues:
            if issue.type == "alignment" and "consistency" in issue.description.lower():
                dims.add("dim_consistency")
            elif issue.type == "constraint":
                dims.add("dim_constraint")
            elif issue.type == "consistency":
                dims.add("dim_discourse")

        if not dims and not report.passed:
            dims.add("dim_coverage")  # 兜底

        return dims

    # ------------------------------------------------------------------
    # 步骤0：约束前置过滤
    # ------------------------------------------------------------------

    def _is_soft_only_failure(self, report: ValidationReport) -> bool:
        """
        判断是否仅为 SOFT 约束违反（不触发 MRSD）

        参数：
            report: 验证报告

        返回值：
            bool：True 表示仅有 MINOR 级别问题，无需 MRSD
        """
        if report.passed:
            return False
        critical_or_major = [
            i for i in report.issues
            if i.severity in ("critical", "major")
        ]
        return len(critical_or_major) == 0 and len(report.issues) > 0

    # ------------------------------------------------------------------
    # 步骤1：Realization 软短路
    # ------------------------------------------------------------------

    def _should_realization_short_circuit(
        self,
        report: ValidationReport,
        contamination_risk_score: float,
    ) -> bool:
        """
        判断是否触发 Realization 软短路

        关键实现细节：
            软短路条件：DCAS.coverage < 0.5 且 CRS < 0.3。
            若 CRS >= 0.3，不短路，进入步骤2（上游污染嫌疑）。

        参数：
            report: 验证报告
            contamination_risk_score: 上游污染风险分数

        返回值：
            bool：True 表示确定为 realization 错误，直接退出
        """
        low_coverage = report.dcas_score < 0.5
        no_constraint_violation = "dim_constraint" not in self._identify_failed_dimensions(report)
        low_crs = contamination_risk_score < 0.3
        return low_coverage and no_constraint_violation and low_crs

    # ------------------------------------------------------------------
    # 步骤2：结构因果路径检测
    # ------------------------------------------------------------------

    def _detect_structural_responsibility(
        self,
        current_section_id: str,
        section_queue: List[str],
    ) -> Set[str]:
        """
        在 DTG 中检测历史决策节点的结构性责任

        功能：
            检查每个历史决策节点 D 是否在"当前失败节"到"依赖节"的路径上。
            不需要 LLM 调用，使用 DTG 路径连通性判断。

        参数：
            current_section_id: 当前失败节 ID
            section_queue: 全局节顺序列表

        返回值：
            Set[str]：具有结构性责任的决策 ID 集合（优先进入候选集）
        """
        structural_ids: Set[str] = set()

        current_decision_id = self._dtg.section_to_decision.get(current_section_id)
        if not current_decision_id:
            return structural_ids

        current_decision = self._dtg.decision_by_id.get(current_decision_id)
        if not current_decision:
            return structural_ids

        # 获取当前决策的所有引用节
        referenced_section_ids = {sid for sid, _ in current_decision.referenced_sections}

        # 对每个历史决策节点检查路径连通性
        for decision in self._dtg.decision_log:
            if decision.decision_id == current_decision_id:
                continue
            # 若该历史决策生成的内容被当前决策引用，则具有结构性责任
            if decision.target_section in referenced_section_ids:
                structural_ids.add(decision.decision_id)

        return structural_ids

    # ------------------------------------------------------------------
    # 步骤3：反向语义扫描
    # ------------------------------------------------------------------

    def _backward_semantic_scan(
        self,
        report: ValidationReport,
        current_section_id: str,
        section_queue: List[str],
        structural_priority_ids: Set[str],
    ) -> Tuple[List[str], List[str], Dict[str, float]]:
        """
        反向语义扫描候选决策节点，最多使用 K=3 次 LLM 调用

        功能：
            优先扫描具有结构性责任的节点，再扫描时序上最近的节点。
            LLM 判断每个候选节点是否存在 explicit_conflict（置信度 0.8）
            或 implicit_omission（置信度 0.5）。

        参数：
            report: 验证报告（提供失败描述）
            current_section_id: 当前失败节 ID
            section_queue: 全局节顺序列表
            structural_priority_ids: 具有结构性责任的决策 ID 集合（优先扫描）

        返回值：
            Tuple[
                List[str],       # 识别出的责任节点 ID 列表
                List[str],       # 证据字符串列表
                Dict[str, float] # 节点ID → 置信度映射
            ]
        """
        failure_description = self._build_failure_description(report)
        causal_nodes: List[str] = []
        evidence: List[str] = []
        node_confidences: Dict[str, float] = {}
        llm_calls_used = 0

        # 构建候选列表：结构性责任节点优先，然后按时间序倒序
        candidates = self._build_candidate_order(
            current_section_id=current_section_id,
            section_queue=section_queue,
            structural_priority_ids=structural_priority_ids,
        )

        for decision in candidates:
            if llm_calls_used >= self._max_llm_budget:
                break

            conflict_type, conf = self._llm_judge_conflict(decision, failure_description)
            llm_calls_used += 1

            if conflict_type == "explicit_conflict":
                causal_nodes.append(decision.decision_id)
                node_confidences[decision.decision_id] = conf
                evidence.append(
                    f"明确冲突 [{decision.decision_id}] "
                    f"节 {decision.target_section}：{decision.decision[:60]}"
                )
                break  # explicit_conflict 即停止
            elif conflict_type == "implicit_omission":
                causal_nodes.append(decision.decision_id)
                node_confidences[decision.decision_id] = conf
                evidence.append(
                    f"隐式遗漏 [{decision.decision_id}] "
                    f"节 {decision.target_section}：{decision.decision[:60]}"
                )
                # 继续扫描（implicit 不停止）
            # no_conflict 跳过

        return causal_nodes, evidence, node_confidences

    def _build_candidate_order(
        self,
        current_section_id: str,
        section_queue: List[str],
        structural_priority_ids: Set[str],
    ) -> List[Decision]:
        """
        构建反向扫描候选决策节点列表（结构性优先，然后时序倒序）

        参数：
            current_section_id: 当前节 ID
            section_queue: 全局节顺序列表
            structural_priority_ids: 优先候选集

        返回值：
            List[Decision]：有序候选决策列表（不含当前节自身决策）
        """
        current_decision_id = self._dtg.section_to_decision.get(current_section_id)
        structural: List[Decision] = []
        temporal: List[Decision] = []

        for decision in reversed(self._dtg.decision_log):
            if decision.decision_id == current_decision_id:
                continue
            if decision.decision_id in structural_priority_ids:
                structural.append(decision)
            else:
                temporal.append(decision)

        return structural + temporal

    def _llm_judge_conflict(
        self,
        decision: Decision,
        failure_description: str,
    ) -> Tuple[str, float]:
        """
        调用 LLM 判断决策节点与失败描述之间的语义冲突类型

        参数：
            decision: 待判断的决策节点
            failure_description: 当前失败的描述文本

        返回值：
            Tuple[str, float]：(conflict_type, confidence)
                conflict_type: "explicit_conflict" | "implicit_omission" | "no_conflict"
                confidence: 置信度 [0.0, 1.0]
        """
        prompt = (
            "判断以下决策是否是当前验证失败的责任来源。只输出 JSON，不要解释。\n\n"
            f"决策内容：\"{decision.decision}\"\n"
            f"决策推理：\"{decision.reasoning}\"\n"
            f"预期效果：\"{decision.expected_effect}\"\n\n"
            f"当前验证失败描述：\"{failure_description}\"\n\n"
            "输出格式：\n"
            '{"conflict_type": "explicit_conflict"|"implicit_omission"|"no_conflict", "confidence": 0.0~1.0}\n\n'
            "说明：\n"
            "  explicit_conflict：决策内容与失败描述存在直接矛盾\n"
            "  implicit_omission：决策遗漏了本应包含的内容，间接导致失败\n"
            "  no_conflict：与当前失败无关"
        )

        try:
            raw = self._llm.generate(
                prompt,
                temperature=0.0,
                max_tokens=32768,
                log_meta={
                    "component": "MRSD",
                    "section_id": decision.target_section,
                },
            )
            return self._parse_conflict_json(raw)
        except Exception:
            return "no_conflict", 0.0

    def _parse_conflict_json(self, raw: str) -> Tuple[str, float]:
        """
        解析 LLM 冲突判断输出

        参数：
            raw: LLM 原始输出

        返回值：
            Tuple[str, float]：(conflict_type, confidence)
        """
        import json as _json
        import re

        valid_types = {"explicit_conflict", "implicit_omission", "no_conflict"}

        text = raw.strip()
        try:
            obj = _json.loads(text)
            ct = obj.get("conflict_type", "no_conflict")
            conf = float(obj.get("confidence", 0.0))
            if ct in valid_types:
                return ct, min(1.0, max(0.0, conf))
        except Exception:
            pass

        # 正则降级
        m1 = re.search(r'"conflict_type"\s*:\s*"(\w+)"', text)
        m2 = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
        if m1 and m2:
            ct = m1.group(1)
            if ct in valid_types:
                return ct, min(1.0, float(m2.group(1)))

        return "no_conflict", 0.0

    # ------------------------------------------------------------------
    # 步骤4：来源判定
    # ------------------------------------------------------------------

    def _determine_source(
        self,
        causal_nodes: List[str],
        current_section_id: str,
        section_queue: List[str],
        report: ValidationReport,
    ) -> ErrorSource:
        """
        判断错误来源（Intrinsic / Propagated / Ambiguous）

        判断逻辑（无 LLM 调用）：
            causal_nodes 中最远节与当前节的距离 > 1 → propagated
            causal_nodes 仅含当前节决策 → intrinsic
            causal_nodes 为空但一致性失败 → propagated（上游污染）
            causal_nodes 为空 → ambiguous

        参数：
            causal_nodes: 责任子图节点 ID 列表
            current_section_id: 当前节 ID
            section_queue: 全局节顺序列表
            report: 验证报告

        返回值：
            ErrorSource 枚举值
        """
        if not causal_nodes:
            has_consistency_failure = any(
                i.type in ("consistency", "alignment") for i in report.issues
            )
            if has_consistency_failure:
                return ErrorSource.PROPAGATED
            return ErrorSource.AMBIGUOUS

        current_idx = section_queue.index(current_section_id) if current_section_id in section_queue else -1
        current_decision_id = self._dtg.section_to_decision.get(current_section_id)

        max_distance = 0
        for decision_id in causal_nodes:
            decision = self._dtg.decision_by_id.get(decision_id)
            if not decision:
                continue
            if decision.target_section in section_queue:
                node_idx = section_queue.index(decision.target_section)
                distance = abs(current_idx - node_idx)
                max_distance = max(max_distance, distance)

        if max_distance > 1:
            return ErrorSource.PROPAGATED
        elif all(d == current_decision_id for d in causal_nodes):
            return ErrorSource.INTRINSIC
        else:
            return ErrorSource.INTRINSIC

    # ------------------------------------------------------------------
    # 步骤5：层级判定
    # ------------------------------------------------------------------

    def _determine_tier(
        self,
        causal_nodes: List[str],
        report: ValidationReport,
        consecutive_failures: int,
        failed_dims: Set[str],
        low_trust_dsl_ref_ratio: float,
        last_purge_succeeded: bool,
        intent_from_trusted_dsl: bool,
        recent_section_failure_tiers: List[str],
        contamination_risk_score: float,
    ) -> ErrorTier:
        """
        判断错误层级（Plan / Decision / Realization / State）

        判断逻辑（按优先级）：
            plan：满足全部四条操作性条件
            state：decision 合理但引用低信任 DSL，或一致性失败但 decision 无内在冲突
            decision：BCP 发现 explicit_conflict 或 implicit_omission
            realization：兜底（Coverage 低，无上述特征）

        参数：
            causal_nodes: 责任子图节点 ID 列表
            report: 验证报告
            consecutive_failures: 当前节在同一 Section Intent 下的连续失败次数
            failed_dims: 失败维度集合
            low_trust_dsl_ref_ratio: 低信任 DSL 引用比例
            last_purge_succeeded: 上次 memory_purge 是否有效
            intent_from_trusted_dsl: Section Intent 是否从高信任 DSL 生成
            recent_section_failure_tiers: 最近节失败层级列表
            contamination_risk_score: 上游污染风险分数

        返回值：
            ErrorTier 枚举值
        """
        # plan_level 四条件检查
        if self._check_plan_level_conditions(
            consecutive_failures=consecutive_failures,
            failed_dims=failed_dims,
            last_purge_succeeded=last_purge_succeeded,
            intent_from_trusted_dsl=intent_from_trusted_dsl,
        ):
            return ErrorTier.PLAN

        # state_level：decision 合理但 DSL 污染，或一致性失败但 decision 无内在冲突
        has_consistency_fail = any(i.type in ("consistency",) for i in report.issues)
        high_dsl_contamination = low_trust_dsl_ref_ratio > 0.4

        if (has_consistency_fail and not causal_nodes) or high_dsl_contamination:
            return ErrorTier.STATE

        # decision_level：BCP 发现了责任节点
        if causal_nodes:
            return ErrorTier.DECISION

        # realization：兜底
        return ErrorTier.REALIZATION

    def _check_plan_level_conditions(
        self,
        consecutive_failures: int,
        failed_dims: Set[str],
        last_purge_succeeded: bool,
        intent_from_trusted_dsl: bool,
    ) -> bool:
        """
        检查 plan_level 错误的四条操作性条件（须全部满足）

        条件：
            ① 同一 Section Intent 下连续两次低温保守重写仍失败
            ② 失败维度集中在 dim_coverage 和 dim_constraint（目标冲突）
            ③ memory_purge 和 local DSL 修复均不能缓解
            ④ Section Intent 本身从可信 DSL 状态生成（排除 state_level 污染伪装）

        参数：
            consecutive_failures: 当前节连续失败次数
            failed_dims: 失败维度集合
            last_purge_succeeded: 上次 purge 是否有效缓解
            intent_from_trusted_dsl: Section Intent 是否从高信任 DSL 生成

        返回值：
            bool：全部四条满足时返回 True
        """
        cond1 = consecutive_failures >= self._PLAN_TRIGGER_CONSECUTIVE_FAILURES
        cond2 = (
            "dim_coverage" in failed_dims
            and "dim_constraint" in failed_dims
            and len(failed_dims) <= 2
        )
        cond3 = not last_purge_succeeded
        cond4 = intent_from_trusted_dsl

        return cond1 and cond2 and cond3 and cond4

    # ------------------------------------------------------------------
    # 修复范围确定
    # ------------------------------------------------------------------

    def _determine_repair_scope(
        self,
        tier: ErrorTier,
        source: ErrorSource,
        causal_nodes: List[str],
        current_section_id: str,
        section_queue: List[str],
    ) -> Tuple[str, Optional[str]]:
        """
        根据层级和来源确定修复范围及目标节

        修复范围映射：
            realization, intrinsic → local_rewrite
            decision, intrinsic    → local_rewrite
            decision, propagated   → partial_rollback（回退到责任节）
            state                  → memory_purge
            plan                   → local_rewrite（先 revise intent）
            any, ambiguous         → local_rewrite（保守处理）

        参数：
            tier: 错误层级
            source: 错误来源
            causal_nodes: 责任节点 ID 列表
            current_section_id: 当前节 ID
            section_queue: 全局节顺序列表

        返回值：
            Tuple[str, Optional[str]]：(repair_scope, target_section)
        """
        if source == ErrorSource.AMBIGUOUS:
            return "local_rewrite", None

        if tier == ErrorTier.STATE:
            return "memory_purge", None

        if tier == ErrorTier.REALIZATION:
            return "local_rewrite", None

        if tier == ErrorTier.PLAN:
            return "local_rewrite", None  # plan 层由 trigger_section_intent_revision 处理

        if tier == ErrorTier.DECISION:
            if source == ErrorSource.PROPAGATED and causal_nodes:
                target = self._find_rollback_target(causal_nodes, section_queue)
                if target:
                    return "partial_rollback", target
            return "local_rewrite", None

        return "local_rewrite", None

    def _find_rollback_target(
        self,
        causal_nodes: List[str],
        section_queue: List[str],
    ) -> Optional[str]:
        """
        找到最近的责任节作为回退目标

        功能：
            在责任节点中找到 section_queue 中最早（最前）的节，
            回退到该节之前的一节（保留其前驱）。

        参数：
            causal_nodes: 责任节点 ID 列表
            section_queue: 全局节顺序列表

        返回值：
            Optional[str]：回退目标节 ID（回退后保留的最后一节），或 None
        """
        min_idx = len(section_queue)
        for decision_id in causal_nodes:
            decision = self._dtg.decision_by_id.get(decision_id)
            if not decision:
                continue
            if decision.target_section in section_queue:
                idx = section_queue.index(decision.target_section)
                min_idx = min(min_idx, idx)

        if min_idx == 0:
            return None  # 无法回退到第 0 节之前
        if min_idx < len(section_queue):
            return section_queue[min_idx - 1]  # 回退到责任节之前一节
        return None

    # ------------------------------------------------------------------
    # 置信度计算
    # ------------------------------------------------------------------

    def _compute_subgraph_confidence(
        self,
        causal_nodes: List[str],
        node_confidences: Dict[str, float],
        failed_dims: Set[str],
        structural_ids: Set[str],
    ) -> float:
        """
        计算责任子图的整体诊断置信度

        公式：
            base = 节点置信度均值（无节点时 0.5）
            结构性责任节点加成 +0.1
            覆盖所有失败维度加成 +0.1（近似）

        参数：
            causal_nodes: 责任节点 ID 列表
            node_confidences: 节点ID → 置信度映射
            failed_dims: 失败维度集合
            structural_ids: 具有结构性责任的节点 ID 集合

        返回值：
            float：整体置信度 [0.0, 1.0]
        """
        if not causal_nodes:
            return 0.5

        confidences = [node_confidences.get(nid, 0.5) for nid in causal_nodes]
        base = sum(confidences) / len(confidences)

        # 结构性责任加成
        has_structural = bool(set(causal_nodes) & structural_ids)
        structural_bonus = 0.1 if has_structural else 0.0

        # 失败维度覆盖加成
        coverage_bonus = 0.1 if len(failed_dims) <= len(causal_nodes) else 0.0

        return min(1.0, base + structural_bonus + coverage_bonus)

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _build_failure_description(self, report: ValidationReport) -> str:
        """
        将验证报告转换为用于 LLM 判断的失败描述文本

        参数：
            report: 验证报告

        返回值：
            str：失败描述字符串
        """
        parts = [f"DCAS 评分：{report.dcas_score:.2f}"]
        for issue in report.issues:
            if issue.severity in ("critical", "major"):
                parts.append(f"[{issue.type}] {issue.description}")
        for constraint in report.violated_constraints[:3]:
            parts.append(f"违反约束：{constraint}")
        return "；".join(parts)

    def _make_result(
        self,
        tier: ErrorTier,
        source: ErrorSource,
        repair_scope: str,
        target_section: Optional[str],
        causal_subgraph: List[str],
        confidence: float,
        evidence: List[str],
        failed_dims: Set[str],
    ) -> DiagnosisResult:
        """
        构建 DiagnosisResult，自动生成 DecodingConfig 和备选修复

        参数：
            tier: 错误层级
            source: 错误来源
            repair_scope: 修复范围
            target_section: 回退目标节（可为 None）
            causal_subgraph: 责任节点 ID 列表
            confidence: 整体置信度
            evidence: 证据列表
            failed_dims: 失败维度集合

        返回值：
            DiagnosisResult 实例
        """
        decoding_config = DecodingConfig.for_tier(tier)
        alternative_repairs = self._compute_alternatives(tier, source, repair_scope)

        self.logger.info(
            "MRSD 诊断完成：tier=%s source=%s scope=%s conf=%.2f nodes=%d",
            tier.value,
            source.value,
            repair_scope,
            confidence,
            len(causal_subgraph),
        )

        return DiagnosisResult(
            error_tier=tier,
            error_source=source,
            repair_scope=repair_scope,
            target_section=target_section,
            causal_subgraph=causal_subgraph,
            confidence=confidence,
            decoding_config=decoding_config,
            evidence=evidence,
        )

    def _compute_alternatives(
        self,
        tier: ErrorTier,
        source: ErrorSource,
        primary_scope: str,
    ) -> List[Tuple[str, float]]:
        """
        计算备选修复方案及其概率

        参数：
            tier: 错误层级
            source: 错误来源
            primary_scope: 主修复范围

        返回值：
            List[Tuple[str, float]]：[(scope, probability), ...]
        """
        alternatives: List[Tuple[str, float]] = []
        all_scopes = ["local_rewrite", "partial_rollback", "memory_purge"]
        for scope in all_scopes:
            if scope != primary_scope:
                prob = 0.15 if tier == ErrorTier.DECISION else 0.05
                alternatives.append((scope, prob))
        return alternatives
