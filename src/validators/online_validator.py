"""
在线验证器：OnlineValidator

功能：
    对每次生成结果执行四层验证（格式/约束/对齐度/一致性），
    汇总问题列表，通过 MetaState 门控决定是否信任 MAJOR 级别问题，
    返回 ValidationReport 供 Orchestrator 决策。

依赖：LLMClient、DTGStore、AlignmentScorer、MetaState
被依赖：Orchestrator（调用 validate_and_diagnose()）

关键设计：
    四层验证均支持降级（任一层异常不影响其他层）。
    MetaState.gate_action("trust_validator_major") 返回 False 时，
    MAJOR 级别问题降级为 MINOR（验证器不稳定时减少误报影响）。
"""
from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from examples.benchmark_template import DOCUMENT_LEVEL_CONSTRAINT_PREFIX

from ..core.decision import Decision
from ..core.meta_state import MetaState
from ..core.state import GenerationState
from ..core.validation import Issue, IssueSeverity, ValidationReport
from ..metrics.alignment import AlignmentScorer

if TYPE_CHECKING:
    from ..logging.run_logger import RunLogger


class OnlineValidator:
    """
    在线验证器

    功能：
        执行四层验证流水线，集成 MetaState 门控（MAJOR 级别误报抑制）。
        不直接调用 MRSD；MRSD 诊断由 Orchestrator 在获取 ValidationReport 后调用。

    参数：
        llm_client: LLM 客户端（用于约束 LLM 兜底和一致性检查）
        dtg_store: DTGStore（用于获取历史决策信息）
        alignment_scorer: AlignmentScorer（计算 DCAS）
        meta_state: MetaState（行为门控）

    关键实现细节：
        DCAS 阈值 0.6（MAJOR）/ 0.5（CRITICAL）。
        约束检查使用四阶段规则：字数→情节→实体→LLM兜底。
        一致性检查使用最近 3 节片段，MINOR 级别（减少误报阻断）。
    """

    # 格式检查阈值
    _MIN_CHARS = 20
    _MAX_CHARS = 8000

    # DCAS 阈值
    THRESHOLD_DCAS = 0.6
    THRESHOLD_DCAS_CRITICAL = 0.5

    def __init__(
        self,
        llm_client,
        dtg_store,
        alignment_scorer: AlignmentScorer,
        meta_state: MetaState,
        run_logger: Optional["RunLogger"] = None,
    ):
        """
        初始化在线验证器

        参数：
            llm_client: LLM 客户端（约束 LLM 兜底和一致性检查）
            dtg_store: DTGStore（历史决策信息）
            alignment_scorer: AlignmentScorer（计算 DCAS）
            meta_state: MetaState（行为门控）
            run_logger: 可选的运行日志器，用于记录四层验证逐层结果
        """
        self.llm = llm_client
        self.dtg = dtg_store
        self.alignment_scorer = alignment_scorer
        self.meta_state = meta_state
        self.run_logger = run_logger
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def validate_and_diagnose(
        self,
        decision: Decision,
        content: str,
        state: GenerationState,
        attempt: int = 1,
    ) -> ValidationReport:
        """
        在线验证（核心方法）

        流程：
            1. 格式检查（规则，极快）
            2. 约束检查（规则 + LLM 兜底）
            3. 对齐度检查（DCAS，LLM 评分）
            4. 一致性检查（LLM）
            5. MetaState 门控：gate_action("trust_validator_major") 为 False 时
               将 MAJOR 问题降级为 MINOR
            6. 汇总 blocking issues，确定 passed 状态

        参数：
            decision: 当前节对应的决策对象
            content: 当前节生成内容
            state: 当前生成状态
            attempt: 协调器层的尝试序号（1-based），传给 run_logger

        返回值：
            ValidationReport：包含问题列表、DCAS 分数和约束违反列表

        关键实现细节：
            任一层验证异常不影响其他层执行（各层独立 try-except）。
        """
        issues: List[Issue] = []
        constraint_violations: List[str] = []
        dcas_score = 1.0
        section_id = state.current_section

        # 记录验证开始
        if self.run_logger is not None:
            self.run_logger.log_validation_start(section_id, attempt)

        # 第一层：格式检查
        format_passed = True
        try:
            format_issues = self._check_format(content)
            issues.extend(format_issues)
            format_passed = not bool(format_issues)
            char_count = len(content.strip())
            format_details = f"length={char_count} chars"
            if format_issues:
                format_details += "  " + " | ".join(i.description for i in format_issues)
            self.logger.debug("Format check completed with %d issues", len(format_issues))
        except Exception as e:
            self.logger.warning("Format check failed and was skipped: %s", e)
            format_details = f"skipped_due_to_exception: {e}"
        if self.run_logger is not None:
            self.run_logger.log_validation_result(
                section_id, attempt, "Format", format_passed, format_details
            )

        # 第二层：约束检查
        constraint_passed = True
        constraint_unknowns: List[str] = []
        try:
            constraint_violations, constraint_unknowns = self._check_constraints(
                content,
                state.global_constraints,
                section_id,
            )
            for v in constraint_violations:
                issues.append(Issue(
                    type="constraint",
                    severity=IssueSeverity.MAJOR.value,
                    description=f"Constraint violated: {v}",
                    location=section_id,
                ))
            constraint_passed = not bool(constraint_violations)
            total_c = len(state.global_constraints)
            passed_c = total_c - len(constraint_violations)
            detail_parts = [f"{passed_c}/{total_c} passed"]
            if constraint_unknowns:
                detail_parts.append(f"{len(constraint_unknowns)} unknown")
            constraint_details = ", ".join(detail_parts)
            self.logger.debug(
                "Constraint check completed with %d violations and %d unknown results",
                len(constraint_violations),
                len(constraint_unknowns),
            )
        except Exception as e:
            constraint_details = f"validator_exception: {e}"
            self.logger.warning("Constraint check failed and was skipped: %s", e)
            constraint_unknowns = list(state.global_constraints)
        if self.run_logger is not None:
            self.run_logger.log_validation_result(
                section_id, attempt, "Constraints", constraint_passed, constraint_details
            )
            if constraint_unknowns:
                self.run_logger.log_validation_note(
                    section_id,
                    attempt,
                    f"constraint_unknown={len(constraint_unknowns)}"
                )
        if constraint_unknowns:
            self.logger.warning(
                "constraint_unknown: %d items (section=%s)",
                len(constraint_unknowns),
                section_id,
            )

        # 第三层：对齐度检查（DCAS）
        dcas_passed = True
        dcas_details = ""
        try:
            alignment_result = self.alignment_scorer.compute_dcas(decision, content)
            dcas_score = alignment_result.get("dcas", 1.0)
            dcas_passed = dcas_score >= self.THRESHOLD_DCAS
            dcas_details = f"score={dcas_score:.3f} (threshold={self.THRESHOLD_DCAS})"
            if dcas_score < self.THRESHOLD_DCAS:
                severity = (
                    IssueSeverity.CRITICAL.value
                    if dcas_score < self.THRESHOLD_DCAS_CRITICAL
                    else IssueSeverity.MAJOR.value
                )
                issues.append(Issue(
                    type="alignment",
                    severity=severity,
                    description=(
                        f"Decision-content alignment is too low (DCAS={dcas_score:.3f} < {self.THRESHOLD_DCAS})"
                    ),
                    location=section_id,
                ))
            self.logger.debug("Alignment check completed with DCAS=%.3f", dcas_score)
        except Exception as e:
            self.logger.warning("Alignment check failed and was skipped: %s", e)
            dcas_score = 0.5
            dcas_passed = False
            dcas_details = f"skipped_due_to_exception: {e}"
        if self.run_logger is not None:
            self.run_logger.log_validation_result(
                section_id, attempt, "Alignment (DCAS)", dcas_passed, dcas_details
            )

        # 第四层：一致性检查
        consistency_passed = True
        try:
            consistency_issues = self._check_consistency(decision, content, state)
            issues.extend(consistency_issues)
            # 一致性问题全为 MINOR，不阻断，但记录到日志
            blocking_consistency = [
                i for i in consistency_issues if i.severity != IssueSeverity.MINOR.value
            ]
            consistency_passed = not bool(blocking_consistency)
            dim_count = 4
            fail_count = len(consistency_issues)
            consistency_details = f"{dim_count - fail_count}/{dim_count} dimensions passed"
            self.logger.debug("Consistency check completed with %d issues", len(consistency_issues))
        except Exception as e:
            self.logger.warning("Consistency check failed and was skipped: %s", e)
            consistency_details = f"skipped_due_to_exception: {e}"
        if self.run_logger is not None:
            self.run_logger.log_validation_result(
                section_id, attempt, "Consistency", consistency_passed, consistency_details
            )

        # MetaState 门控：验证器不稳定时，将 MAJOR 降级为 MINOR
        if not self.meta_state.gate_action("trust_validator_major"):
            issues = self._downgrade_major_to_minor(issues)
            self.logger.info(
                "MetaState gate: validator is unstable (stability=%.2f), MAJOR issues downgraded to MINOR",
                self.meta_state.validator_stability_estimate,
            )

        # 汇总：MINOR 不阻断
        blocking_issues = [i for i in issues if i.severity != IssueSeverity.MINOR.value]
        passed = len(blocking_issues) == 0

        if not passed:
            self.logger.info(
                "Validation failed [%d blocking issues] DCAS=%.3f section=%s",
                len(blocking_issues),
                dcas_score,
                section_id,
            )
        else:
            self.logger.info("Validation passed DCAS=%.3f section=%s", dcas_score, section_id)

        report = ValidationReport(
            passed=passed,
            issues=blocking_issues,
            violated_constraints=constraint_violations,
            dcas_score=dcas_score,
        )

        # 记录验证汇总
        if self.run_logger is not None:
            self.run_logger.log_validation_summary(section_id, attempt, report)

        return report

    # ------------------------------------------------------------------
    # 第一层：格式检查
    # ------------------------------------------------------------------

    def _check_format(self, content: str) -> List[Issue]:
        """
        格式检查（规则，极快）

        检查：
            - 内容长度是否在合理范围（20~8000 字符）
            - 是否有残留 XML 标签

        参数：
            content: 生成内容

        返回值：
            List[Issue]：格式问题列表
        """
        issues: List[Issue] = []

        char_count = len(content.strip())
        if char_count < self._MIN_CHARS:
            issues.append(Issue(
                type="format",
                severity=IssueSeverity.MAJOR.value,
                description=f"Content is too short ({char_count} chars < {self._MIN_CHARS})",
            ))
        elif char_count > self._MAX_CHARS:
            issues.append(Issue(
                type="format",
                severity=IssueSeverity.MINOR.value,
                description=f"Content is too long ({char_count} chars > {self._MAX_CHARS})",
            ))

        leftover_tags = re.findall(
            r"<(decision|reasoning|expected_effect|confidence|content)>", content
        )
        if leftover_tags:
            issues.append(Issue(
                type="format",
                severity=IssueSeverity.CRITICAL.value,
                description=f"Content contains leftover XML tags: {set(leftover_tags)}",
            ))

        return issues

    # ------------------------------------------------------------------
    # 第二层：约束检查
    # ------------------------------------------------------------------

    def _check_constraints(
        self,
        content: str,
        constraints: List[str],
        section_id: Optional[str],
    ) -> Tuple[List[str], List[str]]:
        """
        约束检查（四阶段规则 + LLM 兜底）

        参数：
            content: 生成内容
            constraints: 全局约束列表

        返回值：
            List[str]：违反的约束列表
        """
        violations: List[str] = []
        unknown: List[str] = []
        for constraint in constraints:
            try:
                result, reason = self._check_constraint_satisfaction(
                    constraint,
                    content,
                    section_id,
                )
                if result is False:
                    violations.append(constraint)
                elif result is None:
                    unknown.append(constraint)
                    if reason:
                        self.logger.warning(
                            "constraint_unknown[%s]: %s",
                            reason,
                            constraint[:80],
                        )
            except Exception as e:
                unknown.append(constraint)
                self.logger.warning(
                    "validator_exception: constraint=%s error=%s",
                    constraint[:80],
                    e,
                )
        return violations, unknown

    def _check_constraint_satisfaction(
        self,
        constraint: str,
        content: str,
        section_id: Optional[str],
    ) -> Tuple[Optional[bool], Optional[str]]:
        """
        检查单个约束是否满足

        四阶段（规则优先，LLM 兜底）：
            规则1 — 字数/篇幅类约束：整篇目标，单节直接通过
            规则2 — 情节/事件类约束：全文要求，单节直接通过
            规则3 — 实体/属性类约束：在内容中找到关键实体即通过
            规则4 — LLM 兜底：无法判断时返回 UNKNOWN（不再视为通过）

        参数：
            constraint: 约束文本
            content: 生成内容

        返回值：
            Tuple[Optional[bool], Optional[str]]：
                - True/False 表示明确满足/违反
                - None 表示 UNKNOWN，并返回原因关键词
        """
        c_lower = constraint.lower()
        content_lower = content.lower()

        # 规则0：benchmark 适配器显式标记为“整篇要求”的约束不在 section 级拦截
        # 目的：
        #   这些约束本来就是给整篇输出用的，如果在单节即时校验里逐条拦截，
        #   会把“研究范围”“局限性”这类全文锚点错误地压到 sec2/sec3 上。
        if constraint.startswith(DOCUMENT_LEVEL_CONSTRAINT_PREFIX):
            return True, None

        # 规则1：字数/篇幅类 → 整篇目标，单节直接通过
        if re.search(r"\b\d+\s*(?:words?|chars?|characters?)\b|word count|length|within \d+ words|around \d+ words|at least \d+ words", c_lower):
            return True, None

        # 规则2：情节/事件类 → 故事整体要求，单节不强制
        if re.search(r"\b(include|must include|needs to include|contain|contains|appear|appears|occur|occurs|there is|there are|ending|conclusion|must have|should have)\b", c_lower):
            return True, None

        # 规则3：实体/属性类 → 提取关键实体，在内容中命中即通过
        entities = self._extract_key_entities(constraint)
        if entities and any(e in content_lower for e in entities):
            return True, None

        # 规则4：LLM 兜底（无法判断也需显式返回 UNKNOWN）
        prompt = (
            "Decide whether the section content directly satisfies the constraint below. "
            "Answer with true or false only. true means satisfied. false means violated. "
            "Do not explain your answer.\n\n"
            f"Constraint: {constraint}\n"
            f"Content: {content[:400]}\n"
        )
        try:
            response = self.llm.generate(
                prompt,
                temperature=0.1,
                max_tokens=32768,
                strip_think=False,
                allow_think_only_fallback=False,
                log_meta={
                    "component": "Validator.constraint",
                    "section_id": section_id,
                    "constraint": constraint[:80],
                },
            )
            stripped = response.strip()
            if not stripped:
                return None, "llm_empty_visible_output"

            visible = re.sub(r"<think>.*?</think>", "", stripped, flags=re.DOTALL).strip()
            candidates = [text for text in (visible, stripped) if text]

            for text in candidates:
                lowered = text.strip().lower()
                if lowered in {"true", "false"}:
                    return (lowered == "true"), None

                payload = self._extract_json_object(text)
                if payload:
                    data = self._safe_load_json(payload)
                    if isinstance(data, bool):
                        return data, None
                    if isinstance(data, dict):
                        satisfied = data.get("satisfied")
                        if isinstance(satisfied, bool):
                            return satisfied, None

            if "<think>" in stripped.lower() and not visible:
                return None, "llm_think_only"
            return None, "protocol_parse_failure"
        except Exception as e:
            self.logger.warning("validator_exception: constraint=%s error=%s", constraint[:80], e)
            return None, "validator_exception"

    def _extract_key_entities(self, constraint: str) -> List[str]:
        """
        从约束中提取关键实体（用于规则3命中检查）

        参数：
            constraint: 约束文本

        返回值：
            List[str]：关键实体列表（小写）
        """
        cleaned = re.sub(
            r"\b(main character|background|setting|scene|story|character|named|called|known as|is|are|in|at|located|belongs to|must|should|include|contain)\b",
            " ",
            constraint.lower(),
        )
        tokens = re.split(r"[\s,.;:!?()\[\]\"'/-]+", cleaned.strip())
        stopwords = {
            "the", "and", "for", "with", "from", "into", "that", "this", "those", "these",
            "must", "should", "have", "has", "had", "will", "would", "could", "include",
            "contain", "contains", "section", "story", "content", "character", "setting",
        }
        entities = [t for t in tokens if len(t) >= 3 and t not in stopwords]
        return entities

    def _extract_json_object(self, text: str) -> Optional[str]:
        """提取首个 JSON 对象文本，失败返回 None"""
        if not text:
            return None
        stripped = text.strip()
        try:
            json.loads(stripped)
            return stripped
        except Exception:
            pass

        start = stripped.find("{")
        if start == -1:
            return None
        depth = 0
        for idx in range(start, len(stripped)):
            ch = stripped[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return stripped[start:idx + 1]
        return None

    def _safe_load_json(self, payload: str):
        """安全加载 JSON，失败返回 None"""
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None

    def _parse_consistency_flags(self, response: str, keys: List[str]) -> Dict[str, Optional[bool]]:
        """解析一致性判断，返回每个维度的布尔值或 None"""
        result: Dict[str, Optional[bool]] = {k: None for k in keys}
        if not response:
            return result

        payload = self._extract_json_object(response)
        if payload:
            data = self._safe_load_json(payload)
            if isinstance(data, dict):
                for key in keys:
                    result[key] = self._interpret_bool(data.get(key))
                return result

        lowered = response.lower()
        for key in keys:
            idx = lowered.find(key.lower())
            if idx == -1:
                continue
            window = lowered[idx: idx + 80]
            result[key] = self._keyword_to_bool(window)
        return result

    def _interpret_bool(self, value: Optional[object]) -> Optional[bool]:
        """宽松解析多种布尔表达"""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "pass", "passed", "consistent", "coherent", "yes"}:
                return True
            if lowered in {"false", "fail", "failed", "contradiction", "conflict", "duplicate"}:
                return False
        return None

    def _keyword_to_bool(self, text: str) -> Optional[bool]:
        """根据局部关键词推断布尔值"""
        lowered = text.lower()
        if "false" in lowered or "contradiction" in lowered or "conflict" in lowered or "duplicate" in lowered:
            return False
        if "true" in lowered or "consistent" in lowered or "coherent" in lowered or "pass" in lowered:
            return True
        return None

    # ------------------------------------------------------------------
    # 第四层：一致性检查
    # ------------------------------------------------------------------

    def _check_consistency(
        self,
        decision: Decision,
        content: str,
        state: GenerationState,
    ) -> List[Issue]:
        """
        一致性检查（LLM，检查最近 3 节）

        检查维度：
            1. 实体属性一致性（人物、地点、数字前后是否矛盾）
            2. 时间线连贯性（事件顺序是否合理）
            3. 设定一致性（与已生成章节是否冲突）
            4. 叙事推进性（新内容是否在重复已有章节的核心情节动作）

        参数：
            decision: 当前决策对象
            content: 当前节内容
            state: 生成状态

        返回值：
            List[Issue]：一致性问题列表（全部为 MINOR 级别）

        关键实现细节：
            所有四个维度均为 MINOR 级别，不阻断主流程，只记录供统计分析。
            narrative_progress 的 LLM 误判率较高（相邻章节共享主题属正常现象），
            降为 MINOR 可避免频繁触发不必要的重试循环。
            无法判断时视为通过（宁可放行，不误杀）。
        """
        if not state.generated_sections:
            return []

        prev_snippets: List[str] = []
        for sid in state.generated_sections[-3:]:
            snippet = state.section_snippets.get(sid, "")
            if snippet:
                prev_snippets.append(f"[{sid}] {snippet[:200]}")
        if not prev_snippets:
            return []

        context_hint = "\n".join(prev_snippets)
        prompt = (
            "Check whether the new content contains clear contradictions or narrative repetition relative to the prior sections. "
            "Reply with JSON only, for example "
            "{\"entity_consistency\": true, \"timeline_consistency\": true, \"setting_consistency\": true, \"narrative_progress\": true}. "
            "The fields must be entity_consistency, timeline_consistency, setting_consistency, and narrative_progress. "
            "true means pass and false means an issue exists. Do not output markdown code fences or extra text.\n\n"
            f"Prior section snippets:\n{context_hint}\n\n"
            f"New content:\n{content[:500]}"
        )

        issues: List[Issue] = []
        try:
            response = self.llm.generate(
                prompt,
                temperature=0.1,
                max_tokens=32768,
                strip_think=True,
                allow_think_only_fallback=False,
                log_meta={
                    "component": "Validator.consistency",
                    "section_id": state.current_section,
                },
            )
            checks = {
                "entity_consistency":   ("Entity inconsistency", IssueSeverity.MINOR.value),
                "timeline_consistency": ("Timeline inconsistency",   IssueSeverity.MINOR.value),
                "setting_consistency":  ("Setting contradiction",   IssueSeverity.MINOR.value),
                "narrative_progress":   ("Narrative repetition of prior sections", IssueSeverity.MINOR.value),
            }
            flags = self._parse_consistency_flags(response, list(checks.keys()))
            for tag, (label, severity) in checks.items():
                value = flags.get(tag)
                if value is False:
                    issues.append(Issue(
                        type="consistency",
                        severity=severity,
                        description=label,
                        location=state.current_section,
                    ))
        except Exception as e:
            self.logger.warning("Consistency LLM call failed and was skipped: %s", e)

        return issues

    # ------------------------------------------------------------------
    # MetaState 门控
    # ------------------------------------------------------------------

    def _downgrade_major_to_minor(self, issues: List[Issue]) -> List[Issue]:
        """
        将所有 MAJOR 级别问题降级为 MINOR（MetaState 门控后调用）

        功能：
            当验证器不稳定时（validator_stability_estimate < 0.5），
            MAJOR 问题可能是误报，降级为 MINOR 避免触发不必要的修复。
            CRITICAL 级别问题不受影响。

        参数：
            issues: 原始问题列表

        返回值：
            List[Issue]：降级后的问题列表
        """
        result: List[Issue] = []
        for issue in issues:
            if issue.severity == IssueSeverity.MAJOR.value:
                result.append(Issue(
                    type=issue.type,
                    severity=IssueSeverity.MINOR.value,
                    description=issue.description,
                    location=issue.location,
                ))
            else:
                result.append(issue)
        return result

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
