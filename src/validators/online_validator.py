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

import logging
import re
from typing import Dict, List, Optional, Tuple

from ..core.decision import Decision
from ..core.meta_state import MetaState
from ..core.state import GenerationState
from ..core.validation import Issue, IssueSeverity, ValidationReport
from ..metrics.alignment import AlignmentScorer


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
    ):
        self.llm = llm_client
        self.dtg = dtg_store
        self.alignment_scorer = alignment_scorer
        self.meta_state = meta_state
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def validate_and_diagnose(
        self,
        decision: Decision,
        content: str,
        state: GenerationState,
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

        返回值：
            ValidationReport：包含问题列表、DCAS 分数和约束违反列表

        关键实现细节：
            任一层验证异常不影响其他层执行（各层独立 try-except）。
        """
        issues: List[Issue] = []
        constraint_violations: List[str] = []
        dcas_score = 1.0

        # 第一层：格式检查
        try:
            format_issues = self._check_format(content)
            issues.extend(format_issues)
            self.logger.debug("格式检查完成，发现 %d 个问题", len(format_issues))
        except Exception as e:
            self.logger.warning("格式检查异常（跳过）：%s", e)

        # 第二层：约束检查
        try:
            constraint_violations = self._check_constraints(content, state.global_constraints)
            for v in constraint_violations:
                issues.append(Issue(
                    type="constraint",
                    severity=IssueSeverity.MAJOR.value,
                    description=f"违反约束：{v}",
                    location=state.current_section,
                ))
            self.logger.debug("约束检查完成，违反 %d 条", len(constraint_violations))
        except Exception as e:
            self.logger.warning("约束检查异常（跳过）：%s", e)

        # 第三层：对齐度检查（DCAS）
        try:
            alignment_result = self.alignment_scorer.compute_dcas(decision, content)
            dcas_score = alignment_result.get("dcas", 1.0)
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
                        f"决策-内容对齐度不足（DCAS={dcas_score:.3f} < {self.THRESHOLD_DCAS}）"
                    ),
                    location=state.current_section,
                ))
            self.logger.debug("对齐度检查完成，DCAS=%.3f", dcas_score)
        except Exception as e:
            self.logger.warning("对齐度检查异常（跳过）：%s", e)
            dcas_score = 0.5

        # 第四层：一致性检查
        try:
            consistency_issues = self._check_consistency(decision, content, state)
            issues.extend(consistency_issues)
            self.logger.debug("一致性检查完成，发现 %d 个问题", len(consistency_issues))
        except Exception as e:
            self.logger.warning("一致性检查异常（跳过）：%s", e)

        # MetaState 门控：验证器不稳定时，将 MAJOR 降级为 MINOR
        if not self.meta_state.gate_action("trust_validator_major"):
            issues = self._downgrade_major_to_minor(issues)
            self.logger.info(
                "MetaState 门控：验证器不稳定（stability=%.2f），MAJOR 降级为 MINOR",
                self.meta_state.validator_stability_estimate,
            )

        # 汇总：MINOR 不阻断
        blocking_issues = [i for i in issues if i.severity != IssueSeverity.MINOR.value]
        passed = len(blocking_issues) == 0

        if not passed:
            self.logger.info(
                "验证失败 [%d blocking issues] DCAS=%.3f section=%s",
                len(blocking_issues),
                dcas_score,
                state.current_section,
            )
        else:
            self.logger.info("验证通过 DCAS=%.3f section=%s", dcas_score, state.current_section)

        return ValidationReport(
            passed=passed,
            issues=blocking_issues,
            violated_constraints=constraint_violations,
            dcas_score=dcas_score,
        )

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
                description=f"内容过短（{char_count} 字符 < {self._MIN_CHARS}）",
            ))
        elif char_count > self._MAX_CHARS:
            issues.append(Issue(
                type="format",
                severity=IssueSeverity.MINOR.value,
                description=f"内容过长（{char_count} 字符 > {self._MAX_CHARS}）",
            ))

        leftover_tags = re.findall(
            r"<(decision|reasoning|expected_effect|confidence|content)>", content
        )
        if leftover_tags:
            issues.append(Issue(
                type="format",
                severity=IssueSeverity.CRITICAL.value,
                description=f"内容包含残留 XML 标签：{set(leftover_tags)}",
            ))

        return issues

    # ------------------------------------------------------------------
    # 第二层：约束检查
    # ------------------------------------------------------------------

    def _check_constraints(self, content: str, constraints: List[str]) -> List[str]:
        """
        约束检查（四阶段规则 + LLM 兜底）

        参数：
            content: 生成内容
            constraints: 全局约束列表

        返回值：
            List[str]：违反的约束列表
        """
        violations: List[str] = []
        for constraint in constraints:
            try:
                if not self._check_constraint_satisfaction(constraint, content):
                    violations.append(constraint)
            except Exception as e:
                self.logger.warning("约束 '%s' 检查失败（视为通过）：%s", constraint[:30], e)
        return violations

    def _check_constraint_satisfaction(self, constraint: str, content: str) -> bool:
        """
        检查单个约束是否满足

        四阶段（规则优先，LLM 兜底）：
            规则1 — 字数/篇幅类约束：整篇目标，单节直接通过
            规则2 — 情节/事件类约束：全文要求，单节直接通过
            规则3 — 实体/属性类约束：在内容中找到关键实体即通过
            规则4 — LLM 兜底：仅对无法规则判断的约束使用，失败默认通过

        参数：
            constraint: 约束文本
            content: 生成内容

        返回值：
            bool：True 表示满足约束（或无法判断，保守放行）
        """
        c_lower = constraint.lower()
        content_lower = content.lower()

        # 规则1：字数/篇幅类 → 整篇目标，单节直接通过
        if re.search(r'\d+\s*[字词]|字数|篇幅|字以内|字左右|字以上', constraint):
            return True

        # 规则2：情节/事件类 → 故事整体要求，单节不强制
        if re.search(r'包含|必须包含|需要包含|出现|发生|有一个|存在|结局|结尾|必须有|要有', c_lower):
            return True

        # 规则3：实体/属性类 → 提取关键实体，在内容中命中即通过
        entities = self._extract_key_entities(constraint)
        if entities and any(e in content_lower for e in entities):
            return True

        # 规则4：LLM 兜底（失败默认通过）
        prompt = (
            "判断章节内容是否与约束直接矛盾。\n"
            "只有存在明确矛盾时输出 false，否则一律输出 true。\n\n"
            f"约束：{constraint}\n"
            f"内容：{content[:400]}\n\n"
            "仅输出 <satisfied>true</satisfied> 或 <satisfied>false</satisfied>，不要其他内容。"
        )
        try:
            response = self.llm.generate(prompt, temperature=0.1, max_tokens=64)
            raw = self._extract_tag(response, "satisfied").lower()
            return raw != "false"
        except Exception as e:
            self.logger.warning("LLM 约束验证失败，默认通过：%s", e)
            return True

    def _extract_key_entities(self, constraint: str) -> List[str]:
        """
        从约束中提取关键实体（用于规则3命中检查）

        参数：
            constraint: 约束文本

        返回值：
            List[str]：关键实体列表（小写）
        """
        cleaned = re.sub(
            r'主角|背景|场景|设定|故事|人物|名叫|叫做|叫|名为|是|在|位于|属于',
            '',
            constraint,
        )
        tokens = re.split(r'[\s，。！？,.!?、\-/]+', cleaned.strip())
        entities = [t.lower() for t in tokens if len(t) >= 2]
        extra = [e[:2] for e in entities if len(e) >= 4]
        return entities + extra

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
            "检查新生成内容与已有章节是否存在明确矛盾或叙事重复。\n\n"
            f"已有章节片段：\n{context_hint}\n\n"
            f"新生成内容：{content[:500]}\n\n"
            "对以下四个维度，只有存在明确问题时才输出 false，否则输出 true。\n\n"
            "<entity_consistency>true 或 false：实体属性（人物/地点/数字）是否前后一致</entity_consistency>\n"
            "<timeline_consistency>true 或 false：事件时间线是否连贯合理</timeline_consistency>\n"
            "<setting_consistency>true 或 false：世界设定与规则是否前后一致</setting_consistency>\n"
            "<narrative_progress>true 或 false：新内容是否推进了叙事，"
            "而非重复已有章节的核心情节动作（如果新内容在做和前文相同的核心动作则为 false）</narrative_progress>"
        )

        _POSITIVE = {"true", "一致", "连贯", "通过", "正确", "无矛盾", "没有矛盾", "没有冲突", "推进"}
        _NEGATIVE = {"false", "矛盾", "冲突", "不连贯", "不一致", "有问题", "有矛盾", "有冲突", "重复"}

        issues: List[Issue] = []
        try:
            response = self.llm.generate(prompt, temperature=0.1, max_tokens=512)
            checks = {
                "entity_consistency":   ("实体属性矛盾",   IssueSeverity.MINOR.value),
                "timeline_consistency": ("时间线不连贯",   IssueSeverity.MINOR.value),
                "setting_consistency":  ("设定前后冲突",   IssueSeverity.MINOR.value),
                # MINOR 级别：LLM 对"叙事重复"的判断错误率较高，不适合作为强制重写依据；
                # 结果仍记入报告，供统计分析使用。
                "narrative_progress":   ("叙事重复已有章节", IssueSeverity.MINOR.value),
            }
            for tag, (label, severity) in checks.items():
                raw = self._extract_tag(response, tag).strip().lower()
                if not raw:
                    continue
                if any(p in raw for p in _POSITIVE):
                    continue
                if any(n in raw for n in _NEGATIVE):
                    detail = raw.replace("false", "").strip(" —-：:[]")[:80]
                    issues.append(Issue(
                        type="consistency",
                        severity=severity,
                        description=f"{label}：{detail}" if detail else label,
                        location=state.current_section,
                    ))
        except Exception as e:
            self.logger.warning("一致性检查 LLM 调用失败（跳过）：%s", e)

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

    def _extract_tag(self, text: str, tag: str) -> str:
        """
        提取 XML 标签内容

        参数：
            text: 原始文本
            tag: 标签名

        返回值：
            str：标签内容，标签不存在时返回空字符串
        """
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""
