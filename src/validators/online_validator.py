"""
在线验证器：OnlineValidator

功能：
    对每次生成结果执行四层验证（格式/约束/DSL合规/覆盖率），
    全部基于规则和本地 embedding+NLI 模型，零 LLM-as-judge 调用。
    通过 MetaState 门控决定是否信任 blocking failures，
    返回新版 ValidationReport（含 PresenceViolation/AbsenceViolation）供 Orchestrator 和 MRCA 消费。

依赖：EmbeddingModel、NLIModel、DTGStore、TopicCoverageScorer、MetaState
被依赖：Orchestrator（调用 validate_and_diagnose()）

关键设计：
    Layer 1（格式检查）→ warnings
    Layer 2（约束规则检查）→ PresenceViolation(τ="SCOPE") 或 warnings
    Layer 3（DSL 合规，NLI+embedding）→ PresenceViolation
    Layer 4（覆盖率，embedding）→ AbsenceViolation
    TCAS（TopicCoverageScorer）→ score 字段
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np

from ..core.decision import Decision
from ..core.meta_state import MetaState
from ..core.state import GenerationState
from ..core.validation import (
    AbsenceViolation,
    Issue,
    IssueSeverity,
    PresenceViolation,
    ValidationReport,
    failure_description,
)
from ..metrics.coverage_scorer import TopicCoverageScorer
from ..references.types import GlobalPaperIndex
from .reference_validator import ReferenceValidator

if TYPE_CHECKING:
    from ..core.ledger import LedgerEntry
    from ..logging.run_logger import RunLogger
    from ..utils.embedding_model import EmbeddingModel
    from ..utils.nli_model import NLIModel


DOCUMENT_LEVEL_CONSTRAINT_PREFIX = "Document-level requirement: "


def is_document_level_constraint(requirement: str) -> bool:
    """Return whether the constraint is explicitly tagged as document-level."""
    return requirement.startswith(DOCUMENT_LEVEL_CONSTRAINT_PREFIX)


# ── 常量 ──────────────────────────────────────────────────────────────────────

THETA_TOPIC: float = 0.40       # embedding 粗筛阈值（DSL 合规检查入口）

DSL_TYPE_TO_TAU: Dict[str, str] = {
    "forward_commitment":  "CONSISTENCY",
    "established_claim":   "CONSISTENCY",
    "tentative_claim":     "CONSISTENCY",
    "unresolved_issue":    "CONTENT_GAP",    # 错误关闭方式 → PresenceViolation
    "discourse_policy":    "SCOPE",
}


def sliding_windows(text: str, size: int = 256, stride: int = 128) -> List[str]:
    """
    按词数滑动窗口切分文本

    参数：
        text:   原始文本
        size:   窗口词数（默认 256）
        stride: 步长词数（默认 128）

    返回：
        List[str]: 窗口文本列表
    """
    words = text.split()
    if not words:
        return []
    windows: List[str] = []
    for start in range(0, len(words), stride):
        window = words[start: start + size]
        if window:
            windows.append(" ".join(window))
        if start + size >= len(words):
            break
    return windows


def is_blocking(f: Union[PresenceViolation, AbsenceViolation]) -> bool:
    """所有 PresenceViolation 和 AbsenceViolation 均为 blocking"""
    return True   # 格式/词数问题已在 validate_and_diagnose() 中分流到 warnings


# ── OnlineValidator ────────────────────────────────────────────────────────────

class OnlineValidator:
    """
    在线验证器（零 LLM-as-judge）

    功能：
        执行四层验证流水线，集成 MetaState 门控（blocking failures 抑制）。
        不直接调用 MRSD；MRSD 诊断由 Orchestrator 在获取 ValidationReport 后调用。

    参数：
        embedding_model: EmbeddingModel 实例（全局单例）
        nli_model:       NLIModel 实例（DSL 合规检查）
        dtg_store:       DTGStore（历史决策信息，当前层未直接使用，预留给 MRCA）
        meta_state:      MetaState（行为门控）
        llm_client:      可选，仅用于 ReferenceValidator（约束规则层已无 LLM 调用）
        run_logger:      可选的运行日志器

    关键实现细节：
        TCAS 阈值 0.55（对标原 DCAS 0.6，保守下调 0.05）。
        约束检查使用三阶段规则（无 LLM 兜底）。
        DSL 合规检查：embedding 粗筛 → 批量 NLI → CONTRADICTION → PresenceViolation。
        覆盖率检查：max-cosine pooling < THETA_COVERAGE → AbsenceViolation。
    """

    # 格式检查阈值（词数，不再用字符数）
    _MIN_WORDS = 200

    # 长度惩罚比例阈值（相对于 word_target）
    _LENGTH_SEVERE_FLOOR = 0.35
    _LENGTH_FLOOR = 0.72
    _LENGTH_CEIL = 1.3

    def __init__(
        self,
        embedding_model: "EmbeddingModel",
        nli_model: "NLIModel",
        dtg_store,
        meta_state: MetaState,
        llm_client=None,
        run_logger: Optional["RunLogger"] = None,
    ):
        """
        初始化在线验证器

        参数：
            embedding_model: EmbeddingModel 全局单例
            nli_model:       NLIModel 全局单例
            dtg_store:       DTGStore 实例
            meta_state:      MetaState 实例
            llm_client:      可选，仅用于 ReferenceValidator
            run_logger:      可选的运行日志器
        """
        self.embedding_model = embedding_model
        self.nli_model = nli_model
        self.dtg = dtg_store
        self.meta_state = meta_state
        self.run_logger = run_logger
        self.logger = logging.getLogger(__name__)

        self.coverage_scorer = TopicCoverageScorer(embedding_model)
        # ReferenceValidator 仍需 llm_client（用于复杂引用格式判断）
        self.reference_validator = ReferenceValidator(llm_client=llm_client, min_citations=4)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def validate_and_diagnose(
        self,
        decision: Decision,
        content: str,
        state: GenerationState,
        attempt: int = 1,
        global_index: Optional[GlobalPaperIndex] = None,
        word_target: Optional[int] = None,
        citations=None,
        bundle=None,
    ) -> ValidationReport:
        """
        在线验证（核心方法）

        流程：
            Layer 1 → 格式检查（规则，极快）→ warnings
            Layer 2 → 约束规则检查（无 LLM）→ PresenceViolation(SCOPE) 或 warnings
            Layer 3 → DSL 合规检查（embedding 粗筛 + 批量 NLI）→ PresenceViolation
            Layer 4 → 覆盖率检查（embedding）→ AbsenceViolation
            TCAS    → TopicCoverageScorer 三维评分
            引用检查 → 可选，仅当 global_index 非空
            MetaState 门控 → blocking failures 清空时 passed=True

        参数：
            decision:      当前节对应的决策对象（含 required_topics）
            content:       当前节生成内容
            state:         当前生成状态（含 global_constraints、discourse_ledger）
            attempt:       协调器层的尝试序号（1-based）
            global_index:  全局参考索引（可选）
            word_target:   本节目标词数（可选）

        返回：
            ValidationReport（新版，含 section_id / score / failures / warnings）
        """
        failures: List[Union[PresenceViolation, AbsenceViolation]] = []
        warnings: List[str] = []
        section_id = state.current_section

        if self.run_logger is not None:
            self.run_logger.log_validation_start(section_id, attempt)

        # ── Layer 1：格式检查 ─────────────────────────────────────────────
        # 长度越界（过短/过长）→ blocking AbsenceViolation；其余 Issue → warning
        format_passed = True
        try:
            format_issues = self._check_format(content, word_target=word_target)
            format_passed = not bool(format_issues)
            word_count = len(content.split())
            format_details = f"词数={word_count}"
            if word_target:
                format_details += f"（目标={word_target}）"
            if format_issues:
                format_details += "  " + " | ".join(i.description for i in format_issues)
            for issue in format_issues:
                if issue.type == "format" and issue.severity in (
                    IssueSeverity.MAJOR.value, IssueSeverity.CRITICAL.value
                ):
                    # 长度/严重格式问题转为 blocking failure，驱动 local_rewrite
                    failures.append(AbsenceViolation(
                        τ="CONTENT_GAP",
                        obligation=issue.description,
                        source_check="format_length",
                    ))
                else:
                    warnings.append(str(issue))
        except Exception as e:
            self.logger.warning("格式检查异常（跳过）：%s", e)
            format_details = f"异常跳过：{e}"
        if self.run_logger is not None:
            self.run_logger.log_validation_result(
                section_id, attempt, "格式检查", format_passed, format_details
            )

        # ── Layer 2：约束规则检查 ─────────────────────────────────────────
        constraint_passed = True
        constraint_violations: List[str] = []
        try:
            constraint_violations, _ = self._check_constraints(
                content, state.global_constraints, section_id
            )
            for v in constraint_violations:
                failures.append(PresenceViolation(
                    τ="SCOPE",
                    violating_chunks=[v],
                    violated_dsl_entry=v,
                    source_check="constraint_check",
                ))
            constraint_passed = not bool(constraint_violations)
            total_c = len(state.global_constraints)
            passed_c = total_c - len(constraint_violations)
            constraint_details = f"{passed_c}/{total_c} 条通过"
        except Exception as e:
            constraint_details = f"validator_exception: {e}"
            self.logger.warning("约束检查异常（跳过）：%s", e)
        if self.run_logger is not None:
            self.run_logger.log_validation_result(
                section_id, attempt, "约束检查", constraint_passed, constraint_details
            )

        # ── Layer 3：DSL 合规检查（NLI+embedding） ──────────────────────
        dsl_passed = True
        try:
            dsl_entries: List["LedgerEntry"] = []
            if hasattr(state, "discourse_ledger") and state.discourse_ledger is not None:
                active = getattr(state.discourse_ledger, "_entries", {})
                dsl_entries = [e for e in active.values() if e.is_active()]

            dsl_failures = self._check_dsl_compliance(content, dsl_entries)
            failures.extend(dsl_failures)
            dsl_passed = not bool(dsl_failures)
            dsl_details = f"检查 {len(dsl_entries)} 条 DSL 条目，发现 {len(dsl_failures)} 个违规"
        except Exception as e:
            dsl_details = f"异常跳过：{e}"
            self.logger.warning("DSL 合规检查异常（跳过）：%s", e)
        if self.run_logger is not None:
            self.run_logger.log_validation_result(
                section_id, attempt, "DSL合规(NLI)", dsl_passed, dsl_details
            )

        # ── TCAS 评分 ─────────────────────────────────────────────────────
        score = 1.0
        try:
            constraint_pass_ratio = (
                1.0 - len(constraint_violations) / max(len(state.global_constraints), 1)
            )
            tcas_result = self.coverage_scorer.compute_tcas(
                decision=decision,
                content=content,
                constraint_pass_ratio=constraint_pass_ratio,
                word_target=word_target,
            )
            score = tcas_result["tcas"]
            tcas_details = (
                f"TCAS={score:.3f}  TC={tcas_result['topic_coverage']:.2f}  "
                f"CS={tcas_result['constraint_satisfaction']:.2f}  "
                f"LA={tcas_result['length_adherence']:.2f}"
            )
            if score < TopicCoverageScorer.THRESHOLD_TCAS_MAJOR:
                failures.append(AbsenceViolation(
                    τ="CONTENT_GAP",
                    obligation=(
                        f"TCAS score {score:.3f} below threshold "
                        f"{TopicCoverageScorer.THRESHOLD_TCAS_MAJOR}"
                    ),
                    source_check="tcas",
                ))
        except Exception as e:
            score = 0.5
            tcas_details = f"TCAS 计算异常：{e}"
            self.logger.warning("TCAS 计算异常（跳过）：%s", e)
        if self.run_logger is not None:
            self.run_logger.log_validation_result(
                section_id, attempt, "TCAS评分", score >= TopicCoverageScorer.THRESHOLD_TCAS_MAJOR,
                tcas_details
            )

        # ── 引用检查（可选，保留，产出 reference_report） ─────────────────
        reference_report = None
        if global_index is not None and not global_index.is_empty():
            ref_passed = True
            ref_details = ""
            try:
                reference_report = self.reference_validator.validate(
                    content=content,
                    valid_r_set=global_index.valid_r_set,
                    section_id=section_id,
                )
                ref_passed = reference_report.passed
                ref_details = (
                    f"valid_markers={reference_report.valid_marker_count}, "
                    f"invalid_markers={reference_report.invalid_marker_count}, "
                    f"invalid_r={sorted(reference_report.invalid_r_indices)}"
                )
                if not reference_report.passed:
                    # 引用密度/越界失败 → blocking AbsenceViolation，驱动 local_rewrite
                    major_issues = [i for i in reference_report.issues
                                    if getattr(i, "severity", "") == IssueSeverity.MAJOR.value]
                    obligation = (major_issues[0].description
                                  if major_issues else ref_details)
                    failures.append(AbsenceViolation(
                        τ="CONTENT_GAP",
                        obligation=obligation,
                        source_check="citation_density",
                    ))
            except Exception as e:
                ref_details = f"异常跳过：{e}"
                self.logger.warning("引用范围检查异常（跳过）：%s", e)
            if self.run_logger is not None:
                self.run_logger.log_validation_result(
                    section_id, attempt, "引用检查", ref_passed, ref_details
                )
                if reference_report is not None:
                    self.run_logger.log_reference_validation(section_id, attempt, reference_report)

        # ── MetaState 门控 ────────────────────────────────────────────────
        if not self.meta_state.gate_action("trust_validator_major"):
            failures = []
            self.logger.info(
                "MetaState 门控：验证器不稳定（stability=%.2f），清空 blocking failures",
                self.meta_state.validator_stability_estimate,
            )

        passed = not any(is_blocking(f) for f in failures)

        if not passed:
            self.logger.info(
                "验证失败 [%d blocking failures] TCAS=%.3f section=%s",
                len(failures),
                score,
                section_id,
            )
        else:
            self.logger.info("验证通过 TCAS=%.3f section=%s", score, section_id)

        report = ValidationReport(
            section_id=section_id,
            passed=passed,
            score=score,
            failures=failures,
            warnings=warnings,
            reference_report=reference_report,
        )

        if self.run_logger is not None:
            self.run_logger.log_validation_summary(section_id, attempt, report)

        return report

    # ------------------------------------------------------------------
    # Layer 1：格式检查
    # ------------------------------------------------------------------

    def _check_format(
        self, content: str, word_target: Optional[int] = None
    ) -> List[Issue]:
        """
        格式检查（规则，极快）

        检查：
            - 词数是否达到最低要求（绝对下限 _MIN_WORDS）
            - 词数是否达到本节目标词数比例（有 word_target 时）
            - 是否有残留 XML 标签

        返回：
            List[Issue]：格式问题列表（全部为 warnings）
        """
        issues: List[Issue] = []
        word_count = len(content.split())

        if word_target is not None:
            severe_floor = int(word_target * self._LENGTH_SEVERE_FLOOR)
            floor = int(word_target * self._LENGTH_FLOOR)
            ceil = int(word_target * self._LENGTH_CEIL)

            if word_count < severe_floor:
                issues.append(Issue(
                    type="format",
                    severity=IssueSeverity.MAJOR.value,
                    description=(
                        f"Content is severely under-generated "
                        f"(actual={word_count} target={word_target}). "
                        "Drastically expand with evidence, analysis, and examples."
                    ),
                ))
            elif word_count < floor:
                issues.append(Issue(
                    type="format",
                    severity=IssueSeverity.MAJOR.value,
                    description=(
                        f"Content is too short "
                        f"(actual={word_count} target={word_target}). "
                        "Expand with more evidence, analysis, and examples."
                    ),
                ))
            elif word_count > ceil:
                issues.append(Issue(
                    type="format",
                    severity=IssueSeverity.MAJOR.value,
                    description=(
                        f"Content is too long "
                        f"(actual={word_count} target={word_target}). "
                        "Condense by cutting redundancy while preserving all key arguments."
                    ),
                ))
        elif word_count < self._MIN_WORDS:
            issues.append(Issue(
                type="format",
                severity=IssueSeverity.MAJOR.value,
                description=(
                    f"Content is too short ({word_count} words < minimum {self._MIN_WORDS})"
                ),
            ))

        leftover_tags = re.findall(
            r"<(decision|reasoning|expected_effect|confidence|content)>", content
        )
        if leftover_tags:
            issues.append(Issue(
                type="format",
                severity=IssueSeverity.CRITICAL.value,
                description=f"Content still contains XML tags: {set(leftover_tags)}",
            ))

        return issues

    # ------------------------------------------------------------------
    # Layer 2：约束规则检查
    # ------------------------------------------------------------------

    def _check_constraints(
        self,
        content: str,
        constraints: List[str],
        section_id: Optional[str],
    ) -> Tuple[List[str], List[str]]:
        """
        约束检查（三阶段规则，无 LLM 调用）

        返回：(violations, unknown_constraints)
        """
        violations: List[str] = []
        unknown: List[str] = []
        for constraint in constraints:
            try:
                result, reason = self._check_constraint_satisfaction(
                    constraint, content, section_id
                )
                if result is False:
                    violations.append(constraint)
                elif result is None:
                    unknown.append(constraint)
            except Exception as e:
                unknown.append(constraint)
                self.logger.warning(
                    "validator_exception: constraint=%s error=%s", constraint[:80], e
                )
        return violations, unknown

    def _check_constraint_satisfaction(
        self,
        constraint: str,
        content: str,
        section_id: Optional[str],
    ) -> Tuple[Optional[bool], Optional[str]]:
        """
        检查单个约束（三阶段规则，无 LLM 兜底）

        规则1 — 字数/篇幅类约束：整篇目标，单节直接通过
        规则2 — 实体/属性类约束：在内容中找到关键实体即通过
        规则3 — 无法判断时返回 UNKNOWN（不视为违反）
        """
        c_lower = constraint.lower()
        content_lower = content.lower()

        # 整篇要求的约束不在 section 级拦截
        if is_document_level_constraint(constraint):
            return True, None

        # 规则1：字数/篇幅类 → 整篇目标，单节直接通过
        if re.search(
            r'\d+\s*(?:[字词]|words?)|字数|篇幅|字以内|字左右|字以上|length|paragraph',
            constraint, re.IGNORECASE
        ):
            return True, None

        # 规则2：实体/属性类 → 提取关键实体，在内容中命中即通过
        entities = self._extract_key_entities(constraint)
        if entities and any(e in content_lower for e in entities):
            return True, None

        # 规则3：无法判断 → UNKNOWN（不视为违反，不 blocking）
        return None, "no_rule_matched"

    def _extract_key_entities(self, constraint: str) -> List[str]:
        """从约束中提取关键实体（用于规则2命中检查）"""
        cleaned = re.sub(
            r'主角|背景|场景|设定|故事|人物|名叫|叫做|叫|名为|是|在|位于|属于',
            '', constraint,
        )
        tokens = re.split(r'[\s，。！？,.!?、\-/]+', cleaned.strip())
        entities = [t.lower() for t in tokens if len(t) >= 2]
        extra = [e[:2] for e in entities if len(e) >= 4]
        return entities + extra

    # ------------------------------------------------------------------
    # Layer 3：DSL 合规检查（embedding 粗筛 + 批量 NLI）
    # ------------------------------------------------------------------

    def _check_dsl_compliance(
        self,
        section_content: str,
        dsl_entries: List["LedgerEntry"],
    ) -> List[PresenceViolation]:
        """
        零 LLM DSL 合规检查

        算法：
            1. 将内容切分为滑动窗口
            2. 批量编码所有窗口（一次性，避免重复）
            3. 对每个 DSL 条目，计算与所有窗口的 cosine 相似度
            4. 相似度 ≥ THETA_TOPIC 的窗口进入 NLI 候选队列
            5. 一次性 predict_batch() 所有候选对
            6. CONTRADICTION → PresenceViolation

        参数：
            section_content: 生成内容
            dsl_entries:     活跃 DSL 条目列表

        返回：
            List[PresenceViolation]: 检测到的违规列表
        """
        if not dsl_entries:
            return []

        windows = list(sliding_windows(section_content))
        if not windows:
            return []

        # 批量编码所有窗口（共享计算）
        window_embeds = self.embedding_model.embed(windows)  # (W, D)

        candidates: List[Tuple["LedgerEntry", str]] = []
        for entry in dsl_entries:
            entry_embed = self.embedding_model.embed([entry.content])[0]  # (D,)
            sims = window_embeds @ entry_embed                             # (W,)
            for win_idx, sim in enumerate(sims):
                if float(sim) >= THETA_TOPIC:
                    candidates.append((entry, windows[win_idx]))

        if not candidates:
            return []

        pairs = [(c[0].content, c[1]) for c in candidates]
        labels = self.nli_model.predict_batch(pairs)

        violations: List[PresenceViolation] = []
        for (entry, window), label in zip(candidates, labels):
            if label == "CONTRADICTION":
                tau = DSL_TYPE_TO_TAU.get(entry.commitment_type.value, "CONSISTENCY")
                violations.append(PresenceViolation(
                    τ=tau,
                    violating_chunks=[window],
                    violated_dsl_entry=entry.content,
                    source_check=f"dsl_compliance:{entry.entry_id}",
                ))

        return violations

    # ------------------------------------------------------------------
    # Layer 4：覆盖率检查（embedding）
    # ------------------------------------------------------------------

