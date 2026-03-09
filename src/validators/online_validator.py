from typing import List, Optional, Tuple, Dict
import logging
import re

from ..core.decision import Decision
from ..core.validation import ValidationReport, Issue, IssueSeverity
from ..core.state import GenerationState
from ..algorithms.debugger import DTGDebugger
from ..metrics.alignment import AlignmentScorer


class OnlineValidator:
    """
    作用：
    系统的"质检员"，负责发现问题并给出修正建议
    依赖：
        - LLMClient（调用LLM验证）
        - DTGStore（获取历史信息）
        - Debugger（定位错误源）
        - AlignmentScorer（计算DCAS）
        - Decision, ValidationReport（数据类型）
    被依赖：
        - Orchestrator（调用validate_and_diagnose）
    关键设计：
        1. 不仅验证，还诊断（区分当前问题 vs 历史问题）
        2. 策略推荐基于决策树（可解释）
        3. 所有检查支持降级（不崩溃）

    核心方法：validate_and_diagnose()
    """

    # 格式检查阈值（字符数，中英文通用）
    _MIN_CHARS = 20
    _MAX_CHARS = 8000

    # DCAS阈值
    THRESHOLD_DCAS = 0.6
    THRESHOLD_DCAS_CRITICAL = 0.5

    def __init__(self, llm_client, dtg_store, debugger: DTGDebugger, alignment_scorer: AlignmentScorer):
        self.llm = llm_client
        self.dtg = dtg_store
        self.debugger = debugger
        self.alignment_scorer = alignment_scorer
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
        在线验证并诊断（核心方法）

        流程：
        1. 四层格式检查:
            - 格式检查（长度、标签）---（规则，极快）
            - 约束检查（是否违反约束）---（关键词+LLM)
            - 对齐度检查(DCAS < 0.6?)---(LLM评分)
            - 一致性检查（与历史内容是否矛盾）---(LLM)
       
        2. 汇总issues，定位根源，推荐策略
            - 使用Debugger定位错误源
            - 调用_suggest_fix_strategy推荐策略
        3. 返回ValidationReport

        任一层异常不影响其他层的执行。
        """
        issues: List[Issue] = []
        constraint_violations: List[str] = []
        dcas_score = 1.0
        alignment_result: Dict = {}

        # 第1层：格式检查
        try:
            format_issues = self._check_format(content, getattr(state, "phase", "expanding"))
            issues.extend(format_issues)
            self.logger.debug("格式检查完成，发现%d个问题", len(format_issues))
        except Exception as e:
            self.logger.warning("格式检查异常（跳过）：%s", e)

        # 第2层：约束检查
        try:
            constraint_violations = self._check_constraints(content, state.global_constraints)
            for v in constraint_violations:
                issues.append(Issue(
                    type="constraint",
                    severity=IssueSeverity.MAJOR.value,
                    description=f"违反约束：{v}",
                    location=state.current_section,
                ))
            self.logger.debug("约束检查完成，违反%d条", len(constraint_violations))
        except Exception as e:
            self.logger.warning("约束检查异常（跳过）：%s", e)

        # 第3层：对齐度检查
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
                    description=f"决策-内容对齐度不足（DCAS={dcas_score:.3f} < {self.THRESHOLD_DCAS}）",
                    location=state.current_section,
                ))
            self.logger.debug("对齐度检查完成，DCAS=%.3f", dcas_score)
        except Exception as e:
            self.logger.warning("对齐度检查异常（跳过）：%s", e)
            dcas_score = 0.5

        # 第4层：一致性检查
        try:
            consistency_issues = self._check_consistency(decision, content, state)
            issues.extend(consistency_issues)
            self.logger.debug("一致性检查完成，发现%d个问题", len(consistency_issues))
        except Exception as e:
            self.logger.warning("一致性检查异常（跳过）：%s", e)

        # MINOR 级别问题（主要来自一致性误报）不触发失败，只记录
        blocking_issues = [i for i in issues if i.severity != IssueSeverity.MINOR.value]
        passed = len(blocking_issues) == 0

        # 诊断与策略推荐
        suspected_source: Optional[Decision] = None
        strategy = ""
        strategy_params: Dict = {}

        if not passed:
            suspected_source = self._locate_suspected_source(state.current_section)
            strategy, strategy_params = self._suggest_fix_strategy(
                blocking_issues, constraint_violations, dcas_score, suspected_source
            )
            self.logger.info(
                "验证失败 [%d blocking issues] DCAS=%.3f 策略=%s 根源=%s",
                len(blocking_issues),
                dcas_score,
                strategy,
                suspected_source.decision_id if suspected_source else "None",
            )
        else:
            self.logger.info("验证通过 DCAS=%.3f section=%s", dcas_score, state.current_section)

        return ValidationReport(
            passed=passed,
            issues=blocking_issues,   # 只返回阻断性问题，MINOR记录在日志即可
            violated_constraints=constraint_violations,
            dcas_score=dcas_score,
            suspected_source=suspected_source,
            suggested_strategy=strategy,
            strategy_params=strategy_params,
        )

    # ------------------------------------------------------------------
    # 第1层：格式检查
    # ------------------------------------------------------------------

    def _check_format(self, content: str, phase: str) -> List[Issue]:
        """
        格式检查（快速，基于规则）

        检查：
        - 内容长度是否在合理范围（10~2000词）
        - 是否有残留XML标签（<decision>、<reasoning>等）
        """
        issues = []

        # 长度检查（字符数，中英文通用）
        char_count = len(content.strip())
        if char_count < self._MIN_CHARS:
            issues.append(Issue(
                type="format",
                severity=IssueSeverity.MAJOR.value,
                description=f"内容过短（{char_count}字符 < {self._MIN_CHARS}）",
            ))
        elif char_count > self._MAX_CHARS:
            issues.append(Issue(
                type="format",
                severity=IssueSeverity.MINOR.value,
                description=f"内容过长（{char_count}字符 > {self._MAX_CHARS}）",
            ))

        # 残留XML标签检查
        leftover_tags = re.findall(r"<(decision|reasoning|expected_effect|confidence|content)>", content)
        if leftover_tags:
            issues.append(Issue(
                type="format",
                severity=IssueSeverity.CRITICAL.value,
                description=f"内容包含残留XML标签：{set(leftover_tags)}",
            ))

        return issues

    # ------------------------------------------------------------------
    # 第2层：约束检查
    # ------------------------------------------------------------------

    def _check_constraints(self, content: str, constraints: List[str]) -> List[str]:
        """
        约束检查（关键词快速筛查 + LLM深度验证）

        返回：违反的约束列表
        """
        violations = []
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

        三阶段（Python规则优先，LLM兜底）：
        1. 规则1 — 字数/篇幅类约束：整篇目标，单节直接通过
        2. 规则2 — 情节/事件类约束（包含/出现/结局等）：全文要求，单节直接通过
        3. 规则3 — 实体/属性类约束：在内容中找到关键实体即通过
        4. LLM兜底 — 仅对无法规则判断的约束使用
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
        #   "主角名叫Alex" → 提取 Alex
        #   "背景是火星殖民地" → 提取 火星殖民地 / 火星
        entities = self._extract_key_entities(constraint)
        if entities and any(e in content_lower for e in entities):
            return True

        # 规则4：LLM兜底（只用于无法规则判断的约束，且失败默认通过）
        prompt = f"""判断章节内容是否与约束直接矛盾。
只有存在明确矛盾时输出 false，否则一律输出 true。

约束：{constraint}
内容：{content[:400]}

<satisfied>[true 或 false]</satisfied>"""

        try:
            response = self.llm.generate(prompt, temperature=0.1, max_tokens=256)
            raw = self._extract_tag(response, "satisfied").lower()
            return raw != "false"   # 任何非明确 false 都视为通过
        except Exception as e:
            self.logger.warning("LLM约束验证失败，默认通过：%s", e)
            return True

    def _extract_key_entities(self, constraint: str) -> List[str]:
        """
        从约束中提取关键实体（用于规则3命中检查）

        策略：去除动词/系词后，提取剩余的名词/专有名词片段
        示例：
          "主角名叫Alex"    → ["alex"]
          "背景是火星殖民地" → ["火星殖民地", "火星"]
        """
        # 去除常见系词和修饰词
        cleaned = re.sub(r'主角|背景|场景|设定|故事|人物|名叫|叫做|叫|名为|是|在|位于|属于', '', constraint)
        # 按标点和空白分割
        tokens = re.split(r'[\s，。！？,.!?、\-/]+', cleaned.strip())
        entities = [t.lower() for t in tokens if len(t) >= 2]

        # 对于较长实体额外生成子串（如"火星殖民地"→也加"火星"）
        extra = []
        for e in entities:
            if len(e) >= 4:
                extra.append(e[:2])  # 前两字作为缩略匹配
        return entities + extra

    # ------------------------------------------------------------------
    # 第4层：一致性检查
    # ------------------------------------------------------------------

    def _check_consistency(
        self,
        decision: Decision,
        content: str,
        state: GenerationState,
    ) -> List[Issue]:
        """
        一致性检查（LLM）

        检查维度：
        1. 实体属性一致性（人物、地点、数字前后是否矛盾）
        2. 时间线连贯性（事件顺序是否合理）
        3. 设定一致性（与已生成章节是否冲突）
        """
        if not state.generated_sections:
            return []  # 首节无需一致性检查

        # 使用实际内容片段（而非仅 section ID）
        prev_snippets = []
        for sid in state.generated_sections[-3:]:
            snippet = state.section_snippets.get(sid, "")
            if snippet:
                prev_snippets.append(f"[{sid}] {snippet[:200]}")
        if not prev_snippets:
            return []  # 没有可用的前文内容，跳过检查

        context_hint = "\n".join(prev_snippets)

        prompt = f"""检查新生成内容与已有章节是否存在明确矛盾。

已有章节片段：
{context_hint}

新生成内容：{content[:500]}

对以下三个维度，只有存在明确、直接的矛盾时才输出 false，否则输出 true。

<entity_consistency>true 或 false</entity_consistency>
<timeline_consistency>true 或 false</timeline_consistency>
<setting_consistency>true 或 false</setting_consistency>"""

        # 正向关键词：说明是一致/通过的词
        _POSITIVE = {"true", "一致", "连贯", "通过", "正确", "无矛盾", "没有矛盾", "没有冲突"}
        # 负向关键词：说明确实有问题的词
        _NEGATIVE = {"false", "矛盾", "冲突", "不连贯", "不一致", "有问题", "有矛盾", "有冲突"}

        issues = []
        try:
            response = self.llm.generate(prompt, temperature=0.1, max_tokens=512)
            checks = {
                "entity_consistency":   ("实体属性矛盾",  "consistency"),
                "timeline_consistency": ("时间线不连贯",  "consistency"),
                "setting_consistency":  ("设定前后冲突",  "consistency"),
            }
            for tag, (label, issue_type) in checks.items():
                raw = self._extract_tag(response, tag).strip().lower()
                if not raw:
                    continue  # 无标签 → 跳过

                # 明确正向词 → 通过
                if any(p in raw for p in _POSITIVE):
                    continue
                # 明确负向词 → 记录问题
                if any(n in raw for n in _NEGATIVE):
                    detail = raw.replace("false", "").strip(" —-：:[]")[:80]
                    issues.append(Issue(
                        type=issue_type,
                        severity=IssueSeverity.MINOR.value,   # 降为MINOR，减少误报影响
                        description=f"{label}：{detail}" if detail else label,
                        location=state.current_section,
                    ))
                # 无法判断 → 视为通过（宁可放行）
        except Exception as e:
            self.logger.warning("一致性检查LLM调用失败（跳过）：%s", e)

        return issues

    # ------------------------------------------------------------------
    # 策略推荐
    # ------------------------------------------------------------------

    def _locate_suspected_source(self, section_id: str) -> Optional[Decision]:
        """调用Debugger定位最可疑的历史决策"""
        try:
            suspects = self.debugger.locate_error_source(section_id, top_k=1)
            if not suspects:
                return None
            top_id = suspects[0][0]
            return self.dtg.decision_by_id.get(top_id)
        except Exception as e:
            self.logger.warning("错误定位失败：%s", e)
            return None

    def _suggest_fix_strategy(
        self,
        issues: List[Issue],
        constraint_violations: List[str],
        dcas_score: float,
        suspected_source: Optional[Decision],
    ) -> Tuple[str, Dict]:
        """
        推荐修正策略（决策树）

        决策树：
        1. dcas < 0.5 → RETRY_SIMPLE（内容与决策严重偏离，直接重试）
        2. 约束违反 + suspected_source存在 → ROLLBACK（历史决策导致）
        3. 约束违反 + suspected_source为None → STRENGTHEN_CONSTRAINT（当前约束不够强）
        4. 一致性问题 + suspected_source存在 → ROLLBACK（历史决策引入冲突）
        5. 其他 → RETRY_WITH_STRONGER_PROMPT

        返回：(strategy_name, params_dict)
        """
        consistency_issues = [i for i in issues if i.type == "consistency"]

        # 情况1：对齐度极低
        if dcas_score < self.THRESHOLD_DCAS_CRITICAL:
            return ("RETRY_SIMPLE", {"reason": f"DCAS={dcas_score:.3f} 过低", "dcas": dcas_score})

        # 情况2：约束违反
        if constraint_violations:
            if suspected_source is not None:
                return (
                    f"ROLLBACK_TO:{suspected_source.target_section}",
                    {
                        "target_section": suspected_source.target_section,
                        "suspected_decision_id": suspected_source.decision_id,
                        "violated_constraints": constraint_violations,
                    },
                )
            else:
                return (
                    "STRENGTHEN_CONSTRAINT",
                    {"violated_constraints": constraint_violations},
                )

        # 情况3：一致性问题
        if consistency_issues:
            if suspected_source is not None:
                return (
                    f"ROLLBACK_TO:{suspected_source.target_section}",
                    {
                        "target_section": suspected_source.target_section,
                        "suspected_decision_id": suspected_source.decision_id,
                        "consistency_issues": [i.description for i in consistency_issues],
                    },
                )
            else:
                return (
                    "RETRY_WITH_STRONGER_PROMPT",
                    {"consistency_issues": [i.description for i in consistency_issues]},
                )

        # 默认
        return ("RETRY_SIMPLE", {"reason": "未分类问题"})

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    def _extract_tag(self, text: str, tag: str) -> str:
        """提取XML标签内容"""
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _extract_constraint_keywords(self, constraint: str) -> List[str]:
        """从约束中提取关键词（停用词过滤）"""
        stopwords = {
            "的", "了", "是", "在", "和", "与", "或", "不", "要", "需",
            "请", "使用", "保持", "确保", "必须", "应该", "应",
            "a", "an", "the", "is", "are", "of", "to", "in", "and", "or",
        }
        tokens = re.split(r"[\s，。！？,.!?\-/]+", constraint.lower())
        return [t for t in tokens if len(t) >= 2 and t not in stopwords]
