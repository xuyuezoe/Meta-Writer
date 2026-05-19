"""
最小责任子图诊断算法：MRSD（Minimal Responsibility Subgraph Diagnosis）
内核实现：MRCA（Minimal Repair-oriented Causal Attribution）

功能：
    给定验证失败事件，寻找能解释失败的最小决策责任子图。
    全零 LLM 调用，基于本地 embedding + 图论归因。

    路径 A（PresenceViolation）：五层 MRCA
        Layer 1 — 语义存在概率 p_h(v)：sigmoid 校准 cosine 相似度
        Layer 2 — 语义新引入量 α_h(v)：去除父节点已知贡献
        Layer 3 — 拓扑传播可达性 ρ(v→sf)：反向 DP
        Layer 4 — 节点类型先验 λ(v, τ)：按失败类型加权
        Layer 5 — 责任聚合 + 归一化 → 三分类决策

    路径 B（AbsenceViolation）：规划层审计（三种子情况）

依赖：DTGStore、EmbeddingModel、core/diagnosis.py、core/decision.py、core/validation.py
被依赖：Orchestrator
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from ..core.decision import Decision
from ..core.diagnosis import DecodingConfig, DiagnosisResult, ErrorSource, ErrorTier
from ..core.validation import AbsenceViolation, PresenceViolation, ValidationReport
from ..memory.dtg_store import DTGStore

if TYPE_CHECKING:
    from ..utils.embedding_model import EmbeddingModel


# ── 内部数据类 ──────────────────────────────────────────────────────────────

@dataclass
class RepairDecision:
    """
    MRCA 内部输出，由 diagnose() 桥接为 DiagnosisResult 对外暴露

    字段：
        error_class:      "CURRENT_LOCAL" | "HISTORICAL_ISOLATED" | "HISTORICAL_CONTAGIOUS"
        responsible_node: decision_id 或 intent_node_id
        repair_scope:     需要重生成的 section_id 集合
        instruction:      可选的针对性修复指令
        debug:            调试信息字典
    """
    error_class: str                          # 三分类决策
    responsible_node: Optional[str]           # 最大责任节点 ID
    repair_scope: Set[str]                    # 需要重生成的节集合
    instruction: Optional[str] = None
    debug: Dict = field(default_factory=dict)


# ── 常量 ──────────────────────────────────────────────────────────────────────

EDGE_WEIGHTS: Dict[str, float] = {
    "GENERATES":    0.90,
    "DERIVED_FROM": 0.85,
    "GUIDES":       0.60,
    "REFERENCES":   0.70,   # 默认值，REFERENCES 类型会按 cosine 运行时覆盖
    "DSL_INJECTED": 0.50,   # 默认值，会按 salience 覆盖
}

# 节点类型先验：{τ: weight}
# 行 = 节点类型，列 = 失败类型 (CONSISTENCY, SCOPE, CONTENT_GAP)
NODE_TYPE_PRIOR: Dict[str, Dict[str, float]] = {
    "intent":   {"CONSISTENCY": 0.8, "SCOPE": 1.5, "CONTENT_GAP": 1.3},
    "decision": {"CONSISTENCY": 1.5, "SCOPE": 0.8, "CONTENT_GAP": 1.0},
    "dsl":      {"CONSISTENCY": 1.3, "SCOPE": 1.0, "CONTENT_GAP": 1.2},
    "repair":   {"CONSISTENCY": 1.4, "SCOPE": 0.9, "CONTENT_GAP": 0.8},
}

ETA: float = 0.8       # 历史节责任超出当前节的阈值才回退
THETA_U: float = 0.25  # 传染判断：引用强度阈值
THETA_Q: float = 0.35  # 传染判断：质量压力阈值
THETA_C: float = 0.30  # 传染判断：传染分阈值
M: int = 2             # 传染判断：最小受染节数

# RepairDecision → DiagnosisResult 映射表
_ERROR_CLASS_TO_TIER: Dict[str, ErrorTier] = {
    "CURRENT_LOCAL":         ErrorTier.DECISION,
    "HISTORICAL_ISOLATED":   ErrorTier.DECISION,
    "HISTORICAL_CONTAGIOUS": ErrorTier.STATE,
}
_TAU_TO_TIER: Dict[str, ErrorTier] = {
    "CONSISTENCY":  ErrorTier.STATE,
    "SCOPE":        ErrorTier.STATE,
    "CONTENT_GAP":  ErrorTier.DECISION,
    "METHODOLOGY":  ErrorTier.PLAN,
}
_ERROR_CLASS_TO_SCOPE: Dict[str, str] = {
    "CURRENT_LOCAL":         "local_rewrite",
    "HISTORICAL_ISOLATED":   "partial_rollback",
    "HISTORICAL_CONTAGIOUS": "partial_rollback",
}
_ERROR_CLASS_TO_SOURCE: Dict[str, ErrorSource] = {
    "CURRENT_LOCAL":         ErrorSource.INTRINSIC,
    "HISTORICAL_ISOLATED":   ErrorSource.PROPAGATED,
    "HISTORICAL_CONTAGIOUS": ErrorSource.PROPAGATED,
}

_EPSILON: float = 1e-8


# ── MRSD ────────────────────────────────────────────────────────────────────

class MRSD:
    """
    最小责任子图诊断器（MRCA 内核）

    功能：
        对验证失败执行零 LLM 的五层归因，输出三分类修复决策。
        路径 A（PresenceViolation）→ 五层 MRCA。
        路径 B（AbsenceViolation）→ 规划层审计。

    参数：
        dtg_store:       DTGStore 实例
        embedding_model: EmbeddingModel 实例（五层归因共享）
    """

    def __init__(
        self,
        dtg_store: DTGStore,
        embedding_model: Optional["EmbeddingModel"] = None,
        # 保留向后兼容参数（忽略 LLM）
        llm_client=None,
        max_llm_budget: int = 0,
    ):
        self._dtg = dtg_store
        self._embed = embedding_model
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
        embedding_model: Optional["EmbeddingModel"] = None,
        validator_history: Optional[Dict[str, ValidationReport]] = None,
    ) -> DiagnosisResult:
        """
        对单次验证失败执行 MRCA 诊断

        参数：
            report:                           OnlineValidator 产出的验证报告
            current_section_id:               当前失败节 ID
            section_queue:                    全局节顺序列表
            contamination_risk_score:         MetaState.contamination_risk_score
            consecutive_failures_this_section: 同一 Section Intent 下的连续失败次数
            low_trust_dsl_ref_ratio:          低信任 DSL 引用比例
            last_purge_succeeded:             上次 memory_purge 是否有效
            intent_from_trusted_dsl:          Section Intent 是否从高信任 DSL 生成
            recent_section_failure_tiers:     最近节的历史失败层级列表
            embedding_model:                  可选，覆盖构造时传入的 embedding_model
            validator_history:                历史节验证报告字典 {section_id: report}

        返回：
            DiagnosisResult：包含两轴分类、修复范围、置信度和解码配置
        """
        embed = embedding_model or self._embed
        if validator_history is None:
            validator_history = {}

        # 取第一个 blocking failure（最严重）
        first_failure: Optional[Union[PresenceViolation, AbsenceViolation]] = None
        for f in report.failures:
            first_failure = f
            break

        if first_failure is None:
            # 无 blocking failures（MetaState 门控后 passed=True）
            return self._make_result(
                repair_decision=RepairDecision(
                    error_class="CURRENT_LOCAL",
                    responsible_node=None,
                    repair_scope={current_section_id},
                    debug={"reason": "no_blocking_failure"},
                ),
                report=report,
                current_section_id=current_section_id,
                section_queue=section_queue,
            )

        # 路径分流
        if isinstance(first_failure, PresenceViolation):
            repair_decision = self._presence_attribution(
                failure=first_failure,
                sf_section=current_section_id,
                section_queue=section_queue,
                embed=embed,
                validator_history=validator_history,
                contamination_risk_score=contamination_risk_score,
            )
        else:
            repair_decision = self._absence_attribution(
                failure=first_failure,
                sf_section=current_section_id,
                section_queue=section_queue,
                embed=embed,
                validator_history=validator_history,
            )

        return self._make_result(
            repair_decision=repair_decision,
            report=report,
            current_section_id=current_section_id,
            section_queue=section_queue,
        )

    # ------------------------------------------------------------------
    # 路径 A：PresenceViolation → 五层 MRCA
    # ------------------------------------------------------------------

    def _presence_attribution(
        self,
        failure: PresenceViolation,
        sf_section: str,
        section_queue: List[str],
        embed: Optional["EmbeddingModel"],
        validator_history: Dict[str, ValidationReport],
        contamination_risk_score: float,
    ) -> RepairDecision:
        """路径 A：基于五层 MRCA 计算责任分数"""
        tau = failure.τ
        H = failure.violating_chunks    # 验证器截取的违规内容窗口

        # 第一步：构建归因子图 GF = Anc(sf) ∪ {sf}
        subgraph = self._build_attribution_subgraph(sf_section)
        gf_nodes = subgraph["nodes"]    # List[str]：node_id 列表
        gf_edges = subgraph["edges"]    # List[Dict]：边列表

        if not gf_nodes or not H or embed is None:
            return RepairDecision(
                error_class="CURRENT_LOCAL",
                responsible_node=None,
                repair_scope={sf_section},
                debug={"reason": "empty_subgraph_or_no_embed"},
            )

        # ── Layer 1：语义存在概率 p_h(v) ─────────────────────────────────
        # 对每个节点，计算每个 claim h 存在于该节点内容的校准概率
        node_contents = self._get_node_contents(gf_nodes)
        # shape: {node_id: np.ndarray(H,)}
        p_h_v = self._compute_all_p_h(H, gf_nodes, node_contents, embed)

        # ── Layer 2：语义新引入量 α_F(v) ─────────────────────────────────
        edge_weights = self._compute_edge_weights_map(gf_edges)
        alpha_v = self._compute_alpha_F(gf_nodes, p_h_v, gf_edges, edge_weights)

        # ── Layer 3：拓扑传播可达性 ρ(v→sf) ──────────────────────────────
        sf_node_id = f"content:{sf_section}"
        rho_v = self._compute_reachability(sf_node_id, gf_nodes, gf_edges, edge_weights)

        # ── Layer 4：节点类型先验 λ(v, τ) ────────────────────────────────
        lambda_v = self._compute_node_type_prior(gf_nodes, tau)

        # ── Layer 5：责任聚合 + 归一化 ────────────────────────────────────
        resp_raw = {
            v: alpha_v.get(v, 0.0) * rho_v.get(v, 0.0) * lambda_v.get(v, 1.0)
            for v in gf_nodes
        }
        total = sum(resp_raw.values()) + _EPSILON
        resp_norm = {v: r / total for v, r in resp_raw.items()}

        # Step 6：本节 vs 历史节责任分流
        sf_node_ids = {n for n in gf_nodes if sf_section in n}
        R_cur = sum(resp_norm.get(v, 0.0) for v in sf_node_ids)
        R_hist = sum(resp_norm.get(v, 0.0) for v in gf_nodes if v not in sf_node_ids)

        if R_cur >= ETA * R_hist:
            return RepairDecision(
                error_class="CURRENT_LOCAL",
                responsible_node=max(sf_node_ids, key=lambda v: resp_norm.get(v, 0), default=None),
                repair_scope={sf_section},
                debug={"R_cur": R_cur, "R_hist": R_hist, "resp_norm": resp_norm},
            )

        # Step 7：定位历史责任节
        hist_nodes = {v: resp_norm[v] for v in gf_nodes if v not in sf_node_ids}
        r_node = max(hist_nodes, key=lambda v: hist_nodes[v])
        r_section = self._node_to_section(r_node)

        # Step 8：传染性判断
        return self._contagion_decision(
            r_section=r_section,
            sf_section=sf_section,
            H=H,
            embed=embed,
            section_queue=section_queue,
            validator_history=validator_history,
            r_node=r_node,
            resp_norm=resp_norm,
        )

    # ------------------------------------------------------------------
    # 路径 B：AbsenceViolation → 规划层审计
    # ------------------------------------------------------------------

    def _absence_attribution(
        self,
        failure: AbsenceViolation,
        sf_section: str,
        section_queue: List[str],
        embed: Optional["EmbeddingModel"],
        validator_history: Dict[str, ValidationReport],
    ) -> RepairDecision:
        """
        路径 B：规划层审计三种子情况

        子情况 A：当前节 intent.coverage_requirements 包含 obligation，但执行未覆盖
                 → CURRENT_LOCAL
        子情况 B：历史节 r 的 intent 分配了此 obligation，但未执行
                 → 传染判断
        子情况 C：全局计划从未分配此 obligation
                 → CURRENT_LOCAL（规划层遗漏）
        """
        obligation = failure.obligation
        sf_intent = self._get_intent(sf_section)

        # 子情况 A：当前节规划了但未执行
        if sf_intent and obligation.lower() in " ".join(
            sf_intent.get("coverage_requirements", [])
        ).lower():
            return RepairDecision(
                error_class="CURRENT_LOCAL",
                responsible_node=f"intent:{sf_section}",
                repair_scope={sf_section},
                instruction=f"Must cover the topic: {obligation}",
                debug={"case": "A", "obligation": obligation},
            )

        # 子情况 B：在历史节中找承担此 obligation 的节
        r_section = None
        for sec_id in reversed(section_queue):
            if sec_id == sf_section:
                continue
            intent = self._get_intent(sec_id)
            if not intent:
                continue
            if obligation.lower() in " ".join(
                intent.get("coverage_requirements", [])
            ).lower():
                r_section = sec_id
                break

        if r_section is not None:
            # 历史节有义务但执行遗漏 → 传染判断
            H_obligation = [obligation]
            return self._contagion_decision(
                r_section=r_section,
                sf_section=sf_section,
                H=H_obligation,
                embed=embed,
                section_queue=section_queue,
                validator_history=validator_history,
                r_node=f"intent:{r_section}",
                resp_norm={},
            )

        # 子情况 C：规划层根本遗漏 → CURRENT_LOCAL（不回退，当前节补入）
        return RepairDecision(
            error_class="CURRENT_LOCAL",
            responsible_node=None,
            repair_scope={sf_section},
            instruction=f"Global plan never assigned this topic; include it here: {obligation}",
            debug={"case": "C", "obligation": obligation},
        )

    # ------------------------------------------------------------------
    # Step 8：传染性判断（路径 A/B 共享）
    # ------------------------------------------------------------------

    def _contagion_decision(
        self,
        r_section: str,
        sf_section: str,
        H: List[str],
        embed: Optional["EmbeddingModel"],
        section_queue: List[str],
        validator_history: Dict[str, ValidationReport],
        r_node: str,
        resp_norm: Dict[str, float],
    ) -> RepairDecision:
        """
        传染性判断：判断历史节 r 的错误是否已沿中间节传播

        三维信号：P_j(r)、D_j(r)、S_j(r) → U_j(r) → C(r)
        """
        k_idx = section_queue.index(r_section) if r_section in section_queue else -1
        t_idx = section_queue.index(sf_section) if sf_section in section_queue else len(section_queue)

        if k_idx < 0 or t_idx - k_idx - 1 <= 0:
            # 无中间节，直接返回 HISTORICAL_ISOLATED
            return RepairDecision(
                error_class="HISTORICAL_ISOLATED",
                responsible_node=r_node,
                repair_scope={r_section, sf_section},
                debug={"r_section": r_section, "middle_sections": 0},
            )

        middle_sections = section_queue[k_idx + 1: t_idx]
        infected_count = 0
        contagion_details: Dict[str, float] = {}

        # 构建子图用于可达性计算
        subgraph = self._build_attribution_subgraph(sf_section)
        edge_weights = self._compute_edge_weights_map(subgraph["edges"])

        for j_section in middle_sections:
            # P_j(r)：DTG 路径依赖强度（用 rho 近似）
            rho_all = self._compute_reachability(
                f"content:{j_section}",
                subgraph["nodes"],
                subgraph["edges"],
                edge_weights,
            )
            P_j = rho_all.get(f"content:{r_section}", 0.0)

            # D_j(r)：DSL 引用强度（Phase 3 前用 0，Phase 3 后从 used_by_sections 读取）
            D_j = self._compute_dsl_injection_strength(r_section, j_section)

            # S_j(r)：语义继承强度
            S_j = 0.0
            if embed is not None and H:
                j_content = self._get_section_content(j_section)
                if j_content:
                    h_embeds = embed.embed(H)
                    j_embed = embed.embed([j_content])[0]
                    sims = h_embeds @ j_embed
                    S_j = float(sims.max())

            # U_j(r)：综合引用强度（三路 OR 近似）
            U_j = 1.0 - (1.0 - P_j) * (1.0 - D_j) * (1.0 - S_j)

            # Q_j：质量压力（连续化"勉强过线"信号）
            Q_j = 0.0
            j_report = validator_history.get(j_section)
            if j_report is not None:
                tau_safe = 0.75
                tau_pass = TopicCoverageScorer.THRESHOLD_TCAS_MAJOR if hasattr(
                    self, "_tcas_threshold"
                ) else 0.55
                Q_j = max(0.0, min(1.0, (tau_safe - j_report.score) / (tau_safe - tau_pass + _EPSILON)))

            # C_j：传染分
            C_j = (U_j + Q_j) / 2.0
            contagion_details[j_section] = C_j

            if C_j >= THETA_C:
                infected_count += 1

        is_contagious = infected_count >= M

        if is_contagious:
            return RepairDecision(
                error_class="HISTORICAL_CONTAGIOUS",
                responsible_node=r_node,
                repair_scope=set(section_queue[k_idx:]),
                debug={
                    "r_section": r_section,
                    "infected_count": infected_count,
                    "contagion": contagion_details,
                },
            )
        else:
            return RepairDecision(
                error_class="HISTORICAL_ISOLATED",
                responsible_node=r_node,
                repair_scope={r_section, sf_section},
                debug={
                    "r_section": r_section,
                    "infected_count": infected_count,
                    "contagion": contagion_details,
                },
            )

    # ------------------------------------------------------------------
    # 五层 MRCA 工具方法
    # ------------------------------------------------------------------

    def _compute_all_p_h(
        self,
        H: List[str],
        gf_nodes: List[str],
        node_contents: Dict[str, str],
        embed: "EmbeddingModel",
    ) -> Dict[str, np.ndarray]:
        """
        Layer 1：对所有节点计算每个 claim h 的语义存在概率 p_h(v)

        p_h(v) = σ((max_cos - θ_h) / T_h)
        θ_h = Pct60{cos(E(h), E(v)) | v ∈ GF}（per-claim 动态基准）
        T_h = max_v cos - θ_h + ε（温度）

        返回：{node_id: ndarray(H,)} 每个节点的 H 维概率向量
        """
        if not H or not gf_nodes:
            return {v: np.zeros(len(H)) for v in gf_nodes}

        # 批量编码所有节点内容的窗口（共享计算）
        node_window_embeds: Dict[str, Optional[np.ndarray]] = {}
        for node_id in gf_nodes:
            content = node_contents.get(node_id, "")
            if not content:
                node_window_embeds[node_id] = None
                continue
            windows = self._get_windows(content)
            if windows:
                node_window_embeds[node_id] = embed.embed(windows)  # (W, D)
            else:
                node_window_embeds[node_id] = None

        h_embeds = embed.embed(H)  # (|H|, D)
        result: Dict[str, np.ndarray] = {}

        for h_idx, h_embed in enumerate(h_embeds):
            # 计算所有节点的 max_cos
            max_cos_per_node: Dict[str, float] = {}
            for node_id in gf_nodes:
                w_embeds = node_window_embeds.get(node_id)
                if w_embeds is None:
                    max_cos_per_node[node_id] = 0.0
                else:
                    sims = w_embeds @ h_embed
                    max_cos_per_node[node_id] = float(sims.max())

            max_cos_values = list(max_cos_per_node.values())
            theta_h = float(np.percentile(max_cos_values, 60)) if max_cos_values else 0.0
            global_max = max(max_cos_values) if max_cos_values else 0.0
            T_h = global_max - theta_h + _EPSILON

            for node_id in gf_nodes:
                mc = max_cos_per_node[node_id]
                p = 1.0 / (1.0 + math.exp(-(mc - theta_h) / T_h))
                if node_id not in result:
                    result[node_id] = np.zeros(len(H))
                result[node_id][h_idx] = p

        return result

    def _compute_alpha_F(
        self,
        gf_nodes: List[str],
        p_h_v: Dict[str, np.ndarray],
        gf_edges: List[Dict],
        edge_weights: Dict[Tuple[str, str], float],
    ) -> Dict[str, float]:
        """
        Layer 2：语义新引入量 α_F(v) = max_h α_h(v)

        α_h(v) = max(0, p_h(v) - p̂_h(v))
        p̂_h(v) = 1 - ∏_(u∈Pa(v)) (1 - w_uv * p_h(u))   NB 近似

        返回：{node_id: float} 每个节点的最大跨 claim 新引入量
        """
        parent_map: Dict[str, List[str]] = {v: [] for v in gf_nodes}
        for edge in gf_edges:
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            if src in parent_map and tgt in parent_map:
                parent_map[tgt].append(src)

        alpha_F: Dict[str, float] = {}
        H_len = len(next(iter(p_h_v.values()))) if p_h_v else 0

        for node_id in gf_nodes:
            p_h = p_h_v.get(node_id, np.zeros(H_len))
            parents = parent_map.get(node_id, [])

            if not parents:
                # 根节点：无父节点，α = p_h（全部为新引入）
                alpha_F[node_id] = float(p_h.max()) if len(p_h) > 0 else 0.0
                continue

            # NB 近似：p̂_h(v) = 1 - ∏ (1 - w * p_h(u))
            p_hat = np.zeros(H_len)
            for h_idx in range(H_len):
                prod = 1.0
                for parent_id in parents:
                    w = edge_weights.get((parent_id, node_id), 0.5)
                    p_parent = p_h_v.get(parent_id, np.zeros(H_len))[h_idx]
                    prod *= (1.0 - w * p_parent)
                p_hat[h_idx] = 1.0 - prod

            alpha_h = np.maximum(0.0, p_h - p_hat)
            alpha_F[node_id] = float(alpha_h.max()) if len(alpha_h) > 0 else 0.0

        return alpha_F

    def _compute_reachability(
        self,
        sf_id: str,
        gf_nodes: List[str],
        gf_edges: List[Dict],
        edge_weights: Dict[Tuple[str, str], float],
    ) -> Dict[str, float]:
        """
        Layer 3：反向 DP 计算拓扑传播可达性 ρ(v→sf)

        ρ(sf) = 1
        ρ(v) = 1 - ∏_(v,u)∈E (1 - w_vu * ρ(u))，逆拓扑顺序
        O(|V|+|E|)，无需迭代

        返回：{node_id: float}
        """
        # 构建出边表
        out_edges: Dict[str, List[Tuple[str, float]]] = {v: [] for v in gf_nodes}
        for edge in gf_edges:
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            if src in out_edges and tgt in out_edges:
                w = edge_weights.get((src, tgt), 0.5)
                out_edges[src].append((tgt, w))

        # 拓扑排序（Kahn's algorithm）
        in_degree: Dict[str, int] = {v: 0 for v in gf_nodes}
        for edge in gf_edges:
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            if tgt in in_degree:
                in_degree[tgt] += 1

        from collections import deque
        queue: deque = deque([v for v in gf_nodes if in_degree[v] == 0])
        topo_order: List[str] = []
        in_deg_copy = dict(in_degree)
        while queue:
            node = queue.popleft()
            topo_order.append(node)
            for tgt, _ in out_edges.get(node, []):
                in_deg_copy[tgt] -= 1
                if in_deg_copy[tgt] == 0:
                    queue.append(tgt)

        # 反向 DP（逆拓扑顺序）
        rho: Dict[str, float] = {v: 0.0 for v in gf_nodes}
        rho[sf_id] = 1.0 if sf_id in rho else 0.0

        for node_id in reversed(topo_order):
            if rho.get(node_id, 0.0) == 1.0:
                continue
            prod = 1.0
            for tgt, w in out_edges.get(node_id, []):
                prod *= (1.0 - w * rho.get(tgt, 0.0))
            rho[node_id] = 1.0 - prod

        return rho

    def _compute_node_type_prior(
        self,
        gf_nodes: List[str],
        tau: str,
    ) -> Dict[str, float]:
        """
        Layer 4：节点类型先验 λ(v, τ)

        按节点 ID 前缀推断节点类型，再查 NODE_TYPE_PRIOR 表。
        """
        result: Dict[str, float] = {}
        for node_id in gf_nodes:
            if node_id.startswith("intent:"):
                ntype = "intent"
            elif node_id.startswith("repair:"):
                ntype = "repair"
            elif node_id.startswith("dsl:"):
                ntype = "dsl"
            else:
                ntype = "decision"
            prior_table = NODE_TYPE_PRIOR.get(ntype, {})
            result[node_id] = prior_table.get(tau, 1.0)
        return result

    def _compute_edge_weights_map(
        self,
        gf_edges: List[Dict],
    ) -> Dict[Tuple[str, str], float]:
        """按边类型查表填充权重"""
        weights: Dict[Tuple[str, str], float] = {}
        for edge in gf_edges:
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            etype = edge.get("type", "GENERATES")
            w = EDGE_WEIGHTS.get(etype, 0.5)
            weights[(src, tgt)] = w
        return weights

    # ------------------------------------------------------------------
    # DTG 图查询工具
    # ------------------------------------------------------------------

    def _build_attribution_subgraph(self, sf_section: str) -> Dict:
        """
        构建以 sf_section 为汇点的归因子图 GF = Anc(sf) ∪ {sf}

        返回：{"nodes": List[str], "edges": List[Dict]}
        """
        try:
            exported = self._dtg.export_dtg()
            all_edges = exported.get("edges", [])
            all_nodes: Set[str] = set()

            # 收集祖先节点（BFS 反向）
            reachable: Set[str] = {f"content:{sf_section}"}
            queue = [f"content:{sf_section}"]
            while queue:
                current = queue.pop()
                for edge in all_edges:
                    if edge.get("target") == current:
                        src = edge.get("source", "")
                        if src not in reachable:
                            reachable.add(src)
                            queue.append(src)

            all_nodes = reachable

            # 过滤只含子图内节点的边
            subgraph_edges = [
                e for e in all_edges
                if e.get("source") in all_nodes and e.get("target") in all_nodes
            ]
            return {"nodes": list(all_nodes), "edges": subgraph_edges}
        except Exception as e:
            self.logger.warning("构建归因子图失败：%s", e)
            return {"nodes": [], "edges": []}

    def _get_node_contents(self, node_ids: List[str]) -> Dict[str, str]:
        """从 DTGStore 获取各节点对应的文本内容"""
        contents: Dict[str, str] = {}
        for node_id in node_ids:
            content = ""
            if node_id.startswith("content:"):
                section_id = node_id[len("content:"):]
                decision = None
                dec_id = self._dtg.section_to_decision.get(section_id)
                if dec_id:
                    decision = self._dtg.decision_by_id.get(dec_id)
                if decision:
                    content = decision.decision + " " + decision.reasoning
            elif node_id.startswith("intent:"):
                section_id = node_id[len("intent:"):]
                intent = self._dtg.intent_by_section.get(section_id, {})
                content = intent.get("content", "")
            contents[node_id] = content
        return contents

    def _get_windows(self, text: str, size: int = 256, stride: int = 128) -> List[str]:
        """按词数切分文本为滑动窗口"""
        words = text.split()
        if not words:
            return []
        windows = []
        for start in range(0, len(words), stride):
            window = words[start: start + size]
            if window:
                windows.append(" ".join(window))
            if start + size >= len(words):
                break
        return windows

    def _node_to_section(self, node_id: str) -> str:
        """从节点 ID 提取 section_id"""
        for prefix in ("content:", "intent:", "decision:", "repair:", "dsl:"):
            if node_id.startswith(prefix):
                return node_id[len(prefix):]
        return node_id

    def _get_intent(self, section_id: str) -> Optional[Dict]:
        """获取某节的 intent_node 字典"""
        return self._dtg.intent_by_section.get(section_id)

    def _get_section_content(self, section_id: str) -> str:
        """获取某节的决策文本内容"""
        dec_id = self._dtg.section_to_decision.get(section_id)
        if not dec_id:
            return ""
        decision = self._dtg.decision_by_id.get(dec_id)
        if not decision:
            return ""
        return decision.decision + " " + decision.reasoning

    def _compute_dsl_injection_strength(
        self,
        source_section: str,
        target_section: str,
    ) -> float:
        """
        D_j(r)：DSL 引用强度（Phase 3 前用 0 占位）

        Phase 3 后通过 LedgerEntry.used_by_sections 和 source_node 字段精确计算。
        """
        return 0.0  # Phase 3 前：D_j 恒为 0，传染判断仍可通过 P_j + S_j 两维运作

    # ------------------------------------------------------------------
    # 桥接 RepairDecision → DiagnosisResult
    # ------------------------------------------------------------------

    def _make_result(
        self,
        repair_decision: RepairDecision,
        report: ValidationReport,
        current_section_id: str,
        section_queue: List[str],
    ) -> DiagnosisResult:
        """
        将 RepairDecision 桥接为 DiagnosisResult（Orchestrator 消费格式）

        按 error_class → tier/scope/source 静态映射，
        同时结合 ValidationReport.failures[0].τ 更精确推断 tier。
        """
        error_class = repair_decision.error_class
        tier = _ERROR_CLASS_TO_TIER.get(error_class, ErrorTier.DECISION)

        # 使用失败类型细化 tier
        if report.failures:
            tau = report.failures[0].τ
            tier = _TAU_TO_TIER.get(tau, tier)

        scope = _ERROR_CLASS_TO_SCOPE.get(error_class, "local_rewrite")
        source = _ERROR_CLASS_TO_SOURCE.get(error_class, ErrorSource.INTRINSIC)

        # 确定回退目标节
        target_section = None
        if scope == "partial_rollback" and repair_decision.repair_scope:
            r_sec = self._node_to_section(
                repair_decision.responsible_node or ""
            )
            if r_sec in section_queue:
                r_idx = section_queue.index(r_sec)
                target_section = section_queue[r_idx - 1] if r_idx > 0 else None

        # causal_subgraph：责任节点列表
        causal_subgraph = (
            [repair_decision.responsible_node]
            if repair_decision.responsible_node
            else []
        )

        confidence = self._compute_confidence(repair_decision, report)
        decoding_config = DecodingConfig.for_tier(tier)
        evidence = [
            f"error_class={error_class}",
            f"responsible_node={repair_decision.responsible_node}",
        ]
        if repair_decision.instruction:
            evidence.append(f"instruction={repair_decision.instruction}")

        self.logger.info(
            "MRCA 诊断完成：class=%s tier=%s scope=%s conf=%.2f",
            error_class, tier.value, scope, confidence,
        )

        return DiagnosisResult(
            error_tier=tier,
            error_source=source,
            repair_scope=scope,
            target_section=target_section,
            causal_subgraph=causal_subgraph,
            confidence=confidence,
            decoding_config=decoding_config,
            evidence=evidence,
        )

    def _compute_confidence(
        self,
        repair_decision: RepairDecision,
        report: ValidationReport,
    ) -> float:
        """计算诊断置信度"""
        base = 0.7 if repair_decision.error_class == "CURRENT_LOCAL" else 0.6
        # 有明确责任节点加成
        if repair_decision.responsible_node:
            base += 0.1
        # 多个 failure 时略降低置信度（可能存在多源问题）
        if len(report.failures) > 2:
            base -= 0.05
        return round(min(1.0, max(0.0, base)), 3)
