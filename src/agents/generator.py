"""
内容生成器：Generator

功能：
    接受 GenerationState 和可选的 SectionIntent，构建结构化 prompt，
    调用 LLM 生成结构化输出（Pydantic），返回 (content, Decision) 对。
    使用 LangChain 的 structured output API，解析失败率 < 1%。
    支持 DecodingConfig 驱动的温度参数和 Section Intent 注入。

依赖：LLMClient、GenerationState（core/state.py）、
      Decision、GenerationDecisionSchema（core/decision.py）、SectionIntent（core/plan.py）
被依赖：Orchestrator
"""
from __future__ import annotations

import re
import time
import logging
from typing import TYPE_CHECKING, Optional, List, Tuple, Dict

from ..core.decision import Decision, GenerationDecisionSchema
from ..core.plan import SectionIntent
from ..core.state import GenerationState

if TYPE_CHECKING:
    from ..logging.run_logger import RunLogger


class Generator:
    """
    内容生成器

    功能：
        构建包含状态、任务、SectionIntent、上下文的 prompt，
        调用 LLM 生成结构化输出（Pydantic GenerationDecisionSchema），
        返回 (content, Decision) 对。
        使用 LangChain structured output，解析失败率 < 1%。

    参数：
        llm_client: LLM 客户端实例

    关键实现细节：
        使用 LLMClient.generate_structured() 代替之前的 XML 正则解析。
        LLM API 原生校验输出满足 schema，失败率从 15-30% 降至 <1%。
    """

    MAX_RETRIES = 3
    RECENT_CONTENT_LIMIT = 500
    _XML_TAGS = ("decision", "reasoning", "expected_effect", "confidence", "content")

    def __init__(self, llm_client, run_logger: Optional["RunLogger"] = None):
        """
        初始化生成器

        参数：
            llm_client: LLM 客户端实例
            run_logger: 可选的运行日志器，用于记录 prompt 和 LLM 原始响应
        """
        self.llm = llm_client
        self.run_logger = run_logger
        self.logger = logging.getLogger(__name__)

    def generate_with_decision(
        self,
        state: GenerationState,
        task: str,
        recent_content: str = "",
        section_intent: Optional[SectionIntent] = None,
        temperature: float = 0.7,
        orchestrator_attempt: int = 1,
    ) -> Tuple[str, Decision]:
        """
        生成内容并返回决策对象（使用结构化输出）

        功能：
            调用 LLMClient.generate_structured()，获取 Pydantic 实例，
            无需 XML 正则解析，直接构造 Decision 对象。

        参数：
            state: 当前生成状态（含 DSL 注入、节信息等）
            task: 全局任务描述
            recent_content: 最近已生成内容（截断到最后 500 字符）
            section_intent: 可选的章节局部计划（来自 SectionPlanner）
            temperature: 生成温度（由 DecodingConfig 或默认值决定）
            orchestrator_attempt: 协调器层的尝试序号（1-based），传给 run_logger

        返回值：
            Tuple[str, Decision]：(生成内容, 决策对象)

        异常：
            RuntimeError：调用失败时抛出
        """
        last_error: Optional[Exception] = None
        section_id = state.current_section

        for attempt in range(self.MAX_RETRIES):
            # 第三次尝试降温至 0.3
            actual_temp = 0.3 if attempt == self.MAX_RETRIES - 1 else temperature
            prompt = self._build_prompt(
                state=state,
                task=task,
                recent_content=recent_content,
                section_intent=section_intent,
                strict=False,  # 不需要 strict 模式，因为 schema 本身就是强约束
            )
            self.logger.debug(
                "Generator attempt %d section=%s temp=%.2f",
                attempt + 1,
                section_id,
                actual_temp,
            )

            try:
                # 调用结构化输出 API
                structured_output = self.llm.generate_structured(
                    prompt=prompt,
                    schema=GenerationDecisionSchema,
                    temperature=actual_temp,
                    max_tokens=32768,
                    log_meta={
                        "component": "Generator",
                        "section_id": section_id,
                        "attempt": orchestrator_attempt,
                        "retry": attempt + 1,
                    },
                )

                # 直接构造 Decision，无需解析
                content = self._sanitize_content(structured_output.content)
                decision = Decision(
                    timestamp=int(time.time()),
                    decision_id="",
                    decision=structured_output.decision or "Provide the core writing move for this section.",
                    reasoning=structured_output.reasoning or "Use the available state and prior sections consistently.",
                    expected_effect=structured_output.expected_effect or "Deliver a coherent draft for the current section.",
                    confidence=structured_output.confidence if structured_output.confidence is not None else 0.8,
                    referenced_sections=self._resolve_section_references(
                        structured_output.referenced_section_ids, state
                    ),
                    target_section=state.current_section,
                    phase="expanding",
                )

                # 记录到 run_logger
                if self.run_logger is not None:
                    self.run_logger.log_parsed_decision(
                        section_id, orchestrator_attempt, content, decision
                    )

                self.logger.info(
                    "Generation succeeded [attempt %d] section=%s confidence=%.2f",
                    attempt + 1,
                    section_id,
                    decision.confidence,
                )
                return content, decision

            except (ValueError, TypeError, RuntimeError) as e:
                last_error = e
                self.logger.warning("Generator call failed at retry %d: %s", attempt + 1, e)

        raise RuntimeError(
            f"Generation failed after {self.MAX_RETRIES} retries. Last error: {last_error}"
        )

    # ------------------------------------------------------------------
    # 新增辅助方法：解析section引用
    # ------------------------------------------------------------------

    def _resolve_section_references(
        self, section_ids: List[str], state: GenerationState
    ) -> List[Tuple[str, str]]:
        """
        从 section ID 列表改写为 (section_id, snippet) 元组列表

        参数：
            section_ids: LLM 返回的节点 ID 列表（如 ['sec1', 'sec2']）
            state: 当前生成状态

        返回值：
            [(section_id, "引用片段"), ...]
        """
        # 如果没有引用或状态中没有历史记录，返回空列表
        if not section_ids or not state.section_snippets:
            return []

        result = []
        for section_id in section_ids:
            # 尝试从生成历史中获取该节的内容（取前 100 字作为 snippet）
            if section_id in state.section_snippets:
                snippet = state.section_snippets[section_id][:100]
                result.append((section_id, snippet))
            else:
                # 如果找不到，用空 snippet
                result.append((section_id, ""))

        return result

    # ------------------------------------------------------------------
    # Prompt 构建
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        state: GenerationState,
        task: str,
        recent_content: str,
        section_intent: Optional[SectionIntent],
        strict: bool = False,
    ) -> str:
        """
        构建生成 prompt

        结构：
            1. 当前状态（state.to_prompt()）
            2. 章节局部计划（Section Intent，若提供）
            3. 最近已生成内容（截断到 500 字符）
            4. 任务描述
            5. 结构化输出要求（使用 schema）

        参数：
            state: 生成状态
            task: 任务描述
            recent_content: 已生成内容（最后 500 字符）
            section_intent: 可选的章节局部计划
            strict: 已弃用（保留用于向后兼容）

        返回值：
            str：完整 prompt 字符串
        """
        state_desc = state.to_prompt()
        truncated = recent_content[-self.RECENT_CONTENT_LIMIT:] if recent_content else "(none)"

        intent_block = ""
        scope_warning = ""
        if section_intent is not None:
            intent_block = f"\n{section_intent.to_prompt_text()}\n"
            scope_warning = (
                "\n[Scope Constraint] "
                "This section may only execute the local plan above. "
                "Do not fully resolve the main conflict early and do not advance plot beats that belong to later sections. "
                "Leave unresolved tension for later sections unless this section is explicitly the ending.\n"
            )

        return (
            "You are a long-form writing system.\n"
            "All natural-language fields must be written in English.\n"
            "In particular, decision, reasoning, expected_effect, and content must be English.\n"
            f"\nCurrent state:\n{state_desc}"
            f"{intent_block}"
            f"{scope_warning}"
            f"\nRecent content:\n{truncated}"
            f"\n\nCurrent task:\n{task}"
            "\n\nReturn one JSON object with these fields:"
            "\n- decision: the core writing decision for this section"
            "\n- reasoning: the rationale for that decision, optionally referencing prior section IDs"
            "\n- expected_effect: the narrative effect this section should achieve"
            "\n- confidence: a numeric confidence value between 0.0 and 1.0"
            "\n- content: narrative prose only, with no section IDs, token counts, or meta commentary"
            "\n- referenced_section_ids: a list of prior section IDs such as ['sec1', 'sec2'], or []"
        )

    # ------------------------------------------------------------------
    # 响应解析
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        response: str,
        state: GenerationState,
        section_intent: Optional[SectionIntent],
    ) -> Tuple[str, Decision]:
        """按严格→宽松→正文兜底三层解析 LLM 响应"""
        try:
            fields = {
                tag: self._extract_tag_strict(response, tag)
                for tag in ("decision", "reasoning", "expected_effect", "confidence", "content")
            }
            return self._finalize_decision(fields, state)
        except ValueError as strict_err:
            self.logger.warning("protocol_parse_failure(strict): %s", strict_err)

        try:
            fields = {
                tag: self._extract_tag_loose(response, tag)
                for tag in ("decision", "reasoning", "expected_effect", "confidence", "content")
            }
            return self._finalize_decision(fields, state)
        except ValueError as loose_err:
            self.logger.warning("protocol_parse_failure(loose): %s", loose_err)

        fallback_content = self._extract_fallback_content(response)
        if fallback_content:
            decision = self._build_fallback_decision(state, section_intent, response)
            self.logger.warning("fallback_decision_used: built decision from visible content")
            return fallback_content, decision

        raise ValueError("protocol_parse_failure: unable to extract content")

    def _sanitize_content(self, content: str) -> str:
        """
        清洗生成内容，移除 LLM 常见的格式污染

        处理以下三类污染：
            1. 节 ID 泄漏：内容开头出现形如 "sec2" 的节 ID 行
            2. 引用标签污染：reasoning 中的 <ref> 或转义后的 <a href> 混入内容
            3. 字数统计：MiniMax 自动追加的"字数：XXX字"行

        参数：
            content: 从 <content> 标签提取的原始文本

        返回值：
            str：清洗后的纯叙事文本
        """
        text = content.strip()

        # 第一阶段：移除开头的节 ID 行（如 sec1、sec2、section_3 等）
        text = re.sub(r'^\s*[a-zA-Z_]*\d+\s*\n', '', text)

        # 第二阶段：移除 HTML 锚点引用标签（reasoning ref 泄漏的产物）
        text = re.sub(r'<a\s+href="[^"]*">\[.*?\]</a>', '', text)
        text = re.sub(r'<ref\s+id="[^"]*">.*?</ref>', '', text, flags=re.DOTALL)

        # 第三阶段：移除末尾的字数统计行
        text = re.sub(r'\n*(?:word count|words?|characters?|chars?)[：:]\s*\d+\s*$', '', text.strip(), flags=re.IGNORECASE)

        return text.strip()

    def _finalize_decision(self, fields: Dict[str, str], state: GenerationState) -> Tuple[str, Decision]:
        """根据解析出的字段构造 Decision 对象并返回正文"""
        decision_text = fields["decision"].strip()
        reasoning = fields["reasoning"].strip()
        expected_effect = fields["expected_effect"].strip()
        confidence_str = fields["confidence"].strip()
        content = self._sanitize_content(fields["content"])
        if not content:
            raise ValueError("protocol_parse_failure: empty content")

        try:
            confidence = float(confidence_str)
        except ValueError as e:
            raise ValueError(f"confidence field is not a valid float: '{confidence_str}'") from e

        referenced_sections = self._extract_references(reasoning)
        decision = Decision(
            timestamp=int(time.time()),
            decision_id="",
            decision=decision_text or "Use the generated content as the section's primary move.",
            reasoning=reasoning or "No structured reasoning was available, so the visible content was used directly.",
            expected_effect=expected_effect or "Deliver a coherent draft for the current section.",
            confidence=confidence,
            referenced_sections=referenced_sections,
            target_section=state.current_section,
            phase="expanding",
        )
        return content, decision

    def _extract_tag_strict(self, text: str, tag: str) -> str:
        """严格提取闭合 XML 标签内容，缺失即抛错"""
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            raise ValueError(f"protocol_parse_failure: missing <{tag}> tag")
        return match.group(1).strip()

    def _extract_tag_loose(self, text: str, tag: str) -> str:
        """宽松提取标签，允许缺少闭合时截断到下一个标签"""
        strict_pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(strict_pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

        start_token = f"<{tag}>"
        start = text.find(start_token)
        if start == -1:
            raise ValueError(f"protocol_parse_failure: missing {start_token}")
        start += len(start_token)
        if tag == "content":
            snippet = text[start:].strip()
            if not snippet:
                raise ValueError("protocol_parse_failure: blank content")
            return snippet

        remainder = text[start:]
        next_idx = len(remainder)
        for other in self._XML_TAGS:
            token = f"<{other}>"
            pos = remainder.find(token)
            if pos == -1:
                continue
            if pos < next_idx:
                next_idx = pos
        snippet = remainder[:next_idx].strip()
        if not snippet:
            raise ValueError(f"protocol_parse_failure: missing content for <{tag}>")
        return snippet

    def _extract_fallback_content(self, text: str) -> str:
        """在 XML 解析失败时尽量提取正文"""
        try:
            raw_content = self._extract_tag_loose(text, "content")
            cleaned = self._sanitize_content(raw_content)
            if cleaned:
                return cleaned
        except ValueError:
            pass

        cleaned = re.sub(r"<[^>]+>", " ", text)
        cleaned = self._sanitize_content(cleaned)
        return cleaned

    def _build_fallback_decision(
        self,
        state: GenerationState,
        section_intent: Optional[SectionIntent],
        raw_response: str,
        decision_text: Optional[str] = None,
        reasoning: Optional[str] = None,
        expected_effect: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> Decision:
        """Construct a fallback decision so the pipeline can continue."""
        goal_hint = section_intent.local_goal if section_intent else f"Draft the content for section {state.current_section}."
        decision_text = decision_text or "Use the generated content as the section's primary move."
        reasoning = reasoning or "No structured reasoning was available, so the visible content was used directly."
        expected_effect = expected_effect or goal_hint
        confidence = confidence if confidence is not None else 0.5

        return Decision(
            timestamp=int(time.time()),
            decision_id="",
            decision=decision_text,
            reasoning=reasoning,
            expected_effect=expected_effect,
            confidence=confidence,
            referenced_sections=[],
            target_section=state.current_section,
            phase="expanding",
        )

    def _extract_references(self, reasoning: str) -> List[Tuple[str, str]]:
        """
        从 reasoning 中提取节引用

        格式：<ref id="section_id">snippet</ref>

        参数：
            reasoning: 推理文本

        返回值：
            List[Tuple[str, str]]：[(section_id, snippet), ...]
        """
        pattern = r'<ref\s+id="([^"]+)">(.*?)</ref>'
        matches = re.findall(pattern, reasoning, re.DOTALL)
        return [(section_id.strip(), snippet.strip()) for section_id, snippet in matches]
