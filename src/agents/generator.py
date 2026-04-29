"""
内容生成器：Generator（新架构）

变更说明：
    - 引用模式：从 [Cx]+citations JSON 改为 [Rx] 简单整数标记
    - 参考文献展示：从 500 字截断改为完整摘要 + 完整 top chunk
    - 移除 _extract_citations()：引用提取由 Orchestrator 后处理代码完成
    - 返回值：(content, Decision)，不再有 citations 列表
    - 接受 List[GlobalPaperEntry]（per-section 相关论文）而非 ReferenceBundle

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
from ..references.types import GlobalPaperEntry

if TYPE_CHECKING:
    from ..logging.run_logger import RunLogger


class Generator:
    """
    内容生成器（新架构）

    功能：
        构建包含状态、任务、SectionIntent、参考文献（全文展示）的 prompt，
        调用 LLM 生成结构化输出（Pydantic GenerationDecisionSchema），
        返回 (content, Decision)。

        LLM 在正文中写 [R1]、[R3] 等标记；代码负责后续提取、验证、渲染。

    参数：
        llm_client: LLM 客户端实例
        run_logger: 可选的运行日志器
    """

    MAX_RETRIES = 3
    RECENT_CONTENT_LIMIT = 500
    _XML_TAGS = ("decision", "reasoning", "expected_effect", "confidence", "content")

    def __init__(self, llm_client, run_logger: Optional["RunLogger"] = None):
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
        section_papers: Optional[List[GlobalPaperEntry]] = None,
        citation_retry_hint: Optional[str] = None,
    ) -> Tuple[str, Decision]:
        """
        生成内容并返回决策对象。

        参数：
            state:               当前生成状态
            task:                全局任务描述
            recent_content:      最近已生成内容（截断到最后 500 字符）
            section_intent:      可选的章节局部计划
            temperature:         生成温度
            orchestrator_attempt: 协调器层尝试序号（1-based）
            section_papers:      当节最相关的 GlobalPaperEntry 列表（可选）
            citation_retry_hint: 上一轮因引用密度不足失败时的强制提示；None 表示首次尝试

        返回值：
            Tuple[str, Decision]：(生成内容含 [Rx] 标记, 决策对象)

        异常：
            RuntimeError：三次重试后仍失败
        """
        last_error: Optional[Exception] = None
        section_id = state.current_section

        for attempt in range(self.MAX_RETRIES):
            actual_temp = 0.3 if attempt == self.MAX_RETRIES - 1 else temperature
            prompt = self._build_prompt(
                state=state,
                task=task,
                recent_content=recent_content,
                section_intent=section_intent,
                section_papers=section_papers,
                citation_retry_hint=citation_retry_hint,
            )
            self.logger.debug(
                "第 %d 次尝试生成，section=%s temp=%.2f papers=%d",
                attempt + 1,
                section_id,
                actual_temp,
                len(section_papers) if section_papers else 0,
            )

            try:
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

                content = self._sanitize_content(structured_output.content)
                decision = Decision(
                    timestamp=int(time.time()),
                    decision_id="",
                    decision=structured_output.decision or "Structured decision provided",
                    reasoning=structured_output.reasoning or "Structured reasoning provided",
                    expected_effect=structured_output.expected_effect or "Deliver the core body text for the current section",
                    confidence=structured_output.confidence if structured_output.confidence is not None else 0.8,
                    referenced_sections=self._resolve_section_references(
                        structured_output.referenced_section_ids, state
                    ),
                    target_section=state.current_section,
                    phase="expanding",
                )

                if self.run_logger is not None:
                    self.run_logger.log_parsed_decision(
                        section_id, orchestrator_attempt, content, decision
                    )

                # 统计正文中 [Rx] 标记数量（仅用于日志）
                rx_count = len(re.findall(r'\[R\d+\]', content))
                self.logger.info(
                    "生成成功 [尝试 %d] section=%s confidence=%.2f rx_markers=%d",
                    attempt + 1,
                    section_id,
                    decision.confidence,
                    rx_count,
                )
                return content, decision

            except (ValueError, TypeError, RuntimeError) as e:
                last_error = e
                self.logger.warning("第 %d 次调用失败：%s", attempt + 1, e)

        raise RuntimeError(
            f"生成失败：经过 {self.MAX_RETRIES} 次重试仍无法获得结构化输出。最后错误：{last_error}"
        )

    # ── 章节引用解析 ──────────────────────────────────────────────────────────

    def _resolve_section_references(
        self, section_ids: List[str], state: GenerationState
    ) -> List[Tuple[str, str]]:
        """从 section ID 列表改写为 (section_id, snippet) 元组列表。"""
        if not section_ids or not state.section_snippets:
            return []
        result = []
        for sid in section_ids:
            snippet = state.section_snippets.get(sid, "")[:100]
            result.append((sid, snippet))
        return result

    # ── Prompt 构建 ──────────────────────────────────────────────────────────

    def _build_prompt(
        self,
        state: GenerationState,
        task: str,
        recent_content: str,
        section_intent: Optional[SectionIntent],
        section_papers: Optional[List[GlobalPaperEntry]] = None,
        citation_retry_hint: Optional[str] = None,
    ) -> str:
        """
        构建生成 prompt。

        结构：
            1. 当前状态（state.to_prompt()）
            2. 章节局部计划（Section Intent，若提供）
            3. 全局参考文献展示（若提供）—— 完整摘要 + 完整 top chunk
            4. 引用使用规范（[Rx] 标记）
            5. 最近已生成内容
            6. 任务描述
            7. 引用重试警告（仅在上一轮因引用密度不足失败时出现，紧贴输出规范前）
            8. JSON 输出要求
        """
        state_desc = state.to_prompt()
        truncated = recent_content[-self.RECENT_CONTENT_LIMIT:] if recent_content else "(none)"

        intent_block = ""
        scope_warning = ""
        word_count_instruction = ""
        if section_intent is not None:
            intent_block = f"\n{section_intent.to_prompt_text()}\n"
            scope_warning = (
                "\n[Scope boundary]"
                " This section must handle only the goals in the section intent above. "
                "Do not fully resolve the main conflict early or advance material that belongs to later sections. "
                "When this section ends, unresolved tension should still remain for later sections.\n"
            )
            if section_intent.word_target is not None:
                word_count_instruction = (
                    f"\n[Length requirement] Write approximately {section_intent.word_target} words "
                    f"for this section. Aim for at least {int(section_intent.word_target * 0.85)} words. "
                    "Expand every key point with concrete examples, evidence, and analysis. "
                    "Do not truncate early.\n"
                )

        reference_block = self._build_reference_block(section_papers)
        citation_instructions = self._build_citation_instructions(section_papers)

        # 引用重试警告：仅在 orchestrator 检测到上一轮引用密度不足时注入，
        # 紧贴 JSON 输出规范正前方，利用 recency 效应最大化约束遵循率。
        citation_warning = (
            f"\n[!!CITATION REQUIREMENT NOT MET IN PREVIOUS ATTEMPT!!]\n"
            f"{citation_retry_hint}\n"
            "You must satisfy this requirement in the current response.\n"
        ) if citation_retry_hint else ""

        return (
            "You are a long-form writing system.\n"
            f"\nCurrent state:\n{state_desc}"
            f"{intent_block}"
            f"{scope_warning}"
            f"{word_count_instruction}"
            f"{reference_block}"
            f"{citation_instructions}"
            f"\nRecent content:\n{truncated}"
            f"\n\nCurrent task:\n{task}"
            f"{citation_warning}"
            "\n\nReturn a JSON object with the following fields:"
            "\n- decision: the core writing decision for this section"
            "\n- reasoning: the reasoning behind that decision, optionally citing earlier section IDs."
            " Do NOT put [Rx] markers here."
            "\n- expected_effect: the effect this section should achieve"
            "\n- confidence: a number between 0.0 and 1.0"
            "\n- content: prose text for the section."
            + (
                " Where a sentence is supported by one of the available references,"
                " append the marker [Rx] (e.g. [R1], [R3]) immediately after that sentence."
                " Aim to cite references actively wherever the evidence is relevant."
                " Do not use any Markdown headings (##, ###, etc.) inside the content."
                " Organize with paragraphs only, not subheadings."
                if reference_block else
                " Do not use any Markdown headings (##, ###, etc.) inside the content."
                " Organize with paragraphs only, not subheadings."
            )
            + "\n- referenced_section_ids: a list of cited earlier section IDs such as [\"sec1\", \"sec2\"]"
        )

    @staticmethod
    def _build_reference_block(papers: Optional[List[GlobalPaperEntry]]) -> str:
        """
        将 per-section 相关论文格式化为 prompt 中的参考文献区块。

        展示内容：
            - 全局编号 [Rx]（LLM 使用该标记）
            - 论文标题
            - 完整摘要
            - BM25 top chunk 完整文本（不截断）
        """
        if not papers:
            return ""

        lines = ["\n== Available References =="]
        lines.append(
            "Each reference below has a unique marker [Rx]. "
            "Read the abstract and excerpt carefully before writing."
        )

        for entry in papers:
            lines.append(f"\n[R{entry.r_index}] {entry.title}")

            # 完整摘要（不截断）
            if entry.abstract:
                lines.append(f"  Abstract: {entry.abstract}")

            # top chunk 完整文本（不截断）
            if entry.top_chunk_text:
                # 去掉 Markdown 标题行前缀使 LLM 聚焦内容
                chunk_body = "\n".join(
                    line for line in entry.top_chunk_text.split("\n")
                    if not line.strip().startswith("#")
                ).strip()
                if chunk_body:
                    lines.append(f"  Key excerpt: {chunk_body}")

        lines.append(
            "\n(Only use the markers [R1]–[R{n}] listed above. "
            "Do not invent any other [Rx] marker. "
            "If no reference fits a sentence, write the sentence without any marker.)".format(
                n=papers[-1].r_index if papers else 0
            )
        )
        return "\n".join(lines)

    @staticmethod
    def _build_citation_instructions(papers: Optional[List[GlobalPaperEntry]]) -> str:
        """生成引用规范说明（仅在有论文时添加）。"""
        if not papers:
            return ""

        valid_labels = ", ".join(f"[R{e.r_index}]" for e in papers)
        return (
            "\n[Citation Instructions]\n"
            f"Valid reference markers for this section: {valid_labels}\n"
            "Rules:\n"
            "  1. After a factual claim that can be supported by one of the references, "
            "append the most relevant [Rx] marker immediately — even if the support is partial.\n"
            "  2. Aim to use AS MANY references as are genuinely relevant. "
            "Actively look for connections between your claims and the available evidence.\n"
            "  3. Do NOT use any marker outside the valid list above.\n"
            "  4. If a claim has no supporting reference at all, write it without any marker.\n"
            "  5. Transition, introductory, and connecting sentences do not need markers.\n"
        )

    # ── 内容清洗 ─────────────────────────────────────────────────────────────

    def _sanitize_content(self, content: str) -> str:
        """
        清洗生成内容，移除常见格式污染。

        处理：
            1. 节 ID 泄漏（开头的 sec2 等行）
            2. HTML 引用标签污染
            3. 末尾字数统计行
            4. CJK 字符（保留 [Rx] 标记）
            5. 擅自插入的 Markdown 标题行
        """
        text = content.strip()
        text = re.sub(r'^\s*[a-zA-Z_]*\d+\s*\n', '', text)
        text = re.sub(r'<a\s+href="[^"]*">\[.*?\]</a>', '', text)
        text = re.sub(r'<ref\s+id="[^"]*">.*?</ref>', '', text, flags=re.DOTALL)
        text = re.sub(
            r'\n*(?:[字字数数]+|word count)[：:]\s*\d+[字]?\s*$',
            '', text.strip(), flags=re.IGNORECASE,
        )
        text = re.sub(r'[一-鿿㐀-䶿＀-￯　-〿]+', ' ', text)
        text = re.sub(r',\s*,', ',', text)
        text = re.sub(r'  +', ' ', text)
        text = re.sub(r'^#{2,}\s+.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    # ── 旧式 XML 解析（保留供降级路径使用） ─────────────────────────────────

    def _finalize_decision(self, fields: Dict[str, str], state: GenerationState) -> Tuple[str, Decision]:
        decision_text = fields["decision"].strip()
        reasoning = fields["reasoning"].strip()
        expected_effect = fields["expected_effect"].strip()
        confidence_str = fields["confidence"].strip()
        content = self._sanitize_content(fields["content"])
        if not content:
            raise ValueError("protocol_parse_failure: content 为空")
        try:
            confidence = float(confidence_str)
        except ValueError as e:
            raise ValueError(f"confidence 字段无法解析：'{confidence_str}'") from e

        decision = Decision(
            timestamp=int(time.time()),
            decision_id="",
            decision=decision_text or "Fallback decision",
            reasoning=reasoning or "No reasoning provided",
            expected_effect=expected_effect or "Deliver core body text",
            confidence=confidence,
            referenced_sections=self._extract_references(reasoning),
            target_section=state.current_section,
            phase="expanding",
        )
        return content, decision

    def _extract_tag_strict(self, text: str, tag: str) -> str:
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            raise ValueError(f"protocol_parse_failure: 缺少 <{tag}> 标签")
        return match.group(1).strip()

    def _extract_tag_loose(self, text: str, tag: str) -> str:
        strict_pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(strict_pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        start_token = f"<{tag}>"
        start = text.find(start_token)
        if start == -1:
            raise ValueError(f"protocol_parse_failure: 缺少 {start_token}")
        start += len(start_token)
        if tag == "content":
            snippet = text[start:].strip()
            if not snippet:
                raise ValueError("protocol_parse_failure: content 空白")
            return snippet
        remainder = text[start:]
        next_idx = len(remainder)
        for other in self._XML_TAGS:
            token = f"<{other}>"
            pos = remainder.find(token)
            if pos != -1 and pos < next_idx:
                next_idx = pos
        snippet = remainder[:next_idx].strip()
        if not snippet:
            raise ValueError(f"protocol_parse_failure: <{tag}> 内容缺失")
        return snippet

    def _extract_fallback_content(self, text: str) -> str:
        try:
            raw = self._extract_tag_loose(text, "content")
            cleaned = self._sanitize_content(raw)
            if cleaned:
                return cleaned
        except ValueError:
            pass
        cleaned = re.sub(r"<[^>]+>", " ", text)
        return self._sanitize_content(cleaned)

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
        goal_hint = section_intent.local_goal if section_intent else f"Complete {state.current_section}"
        return Decision(
            timestamp=int(time.time()),
            decision_id="",
            decision=decision_text or "Structured decision missing",
            reasoning=reasoning or "No structured reasoning",
            expected_effect=expected_effect or goal_hint,
            confidence=confidence if confidence is not None else 0.5,
            referenced_sections=[],
            target_section=state.current_section,
            phase="expanding",
        )

    def _extract_references(self, reasoning: str) -> List[Tuple[str, str]]:
        pattern = r'<ref\s+id="([^"]+)">(.*?)</ref>'
        matches = re.findall(pattern, reasoning, re.DOTALL)
        return [(sid.strip(), snip.strip()) for sid, snip in matches]
