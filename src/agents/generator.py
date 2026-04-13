"""
内容生成器：Generator

功能：
    接受 GenerationState 和可选的 SectionIntent，构建结构化 prompt，
    调用 LLM 生成内容，解析 XML 响应，返回 (content, Decision) 对。
    支持 DecodingConfig 驱动的温度参数和 Section Intent 注入。

依赖：LLMClient、GenerationState（core/state.py）、
      Decision（core/decision.py）、SectionIntent（core/plan.py）
被依赖：Orchestrator
"""
from __future__ import annotations

import re
import time
import logging
from typing import TYPE_CHECKING, Optional, List, Tuple

from ..core.decision import Decision
from ..core.plan import SectionIntent
from ..core.state import GenerationState

if TYPE_CHECKING:
    from ..logging.run_logger import RunLogger


class Generator:
    """
    内容生成器

    功能：
        构建包含状态、任务、SectionIntent、上下文的 prompt，
        调用 LLM 生成内容，解析 XML 格式响应为 (content, Decision)。
        支持最多 3 次重试，逐次加强 prompt 格式约束。

    参数：
        llm_client: LLM 客户端实例

    关键实现细节：
        第 1 次尝试：正常 prompt，使用传入 temperature
        第 2 次尝试：追加格式警告，temperature 不变
        第 3 次尝试：强制格式约束，temperature 降至 0.3（保守生成）
    """

    MAX_RETRIES = 3
    RECENT_CONTENT_LIMIT = 500
    GENERATION_MAX_TOKENS = 4096

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
        生成内容并返回决策对象

        功能：
            三次重试递进机制：
                第 1 次：正常 prompt，使用传入 temperature
                第 2 次：追加格式不合规警告，temperature 不变
                第 3 次：强制格式约束，temperature 降至 0.3

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
            RuntimeError：三次重试后仍无法解析响应
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
                strict=(attempt > 0),
            )
            self.logger.debug(
                "第 %d 次尝试生成，section=%s temp=%.2f",
                attempt + 1,
                section_id,
                actual_temp,
            )

            # 第一阶段：记录 prompt
            if self.run_logger is not None:
                self.run_logger.log_prompt(section_id, orchestrator_attempt, prompt)

            try:
                response = self.llm.generate(
                    prompt,
                    temperature=actual_temp,
                    max_tokens=self.GENERATION_MAX_TOKENS,
                )

                # 第二阶段：记录 LLM 原始响应
                if self.run_logger is not None:
                    self.run_logger.log_llm_raw_response(section_id, orchestrator_attempt, response)

                content, decision = self._parse_response(response, state)

                # 第三阶段：记录解析后的决策和内容
                if self.run_logger is not None:
                    self.run_logger.log_parsed_decision(
                        section_id, orchestrator_attempt, content, decision
                    )

                self.logger.info(
                    "生成成功 [尝试 %d] section=%s confidence=%.2f",
                    attempt + 1,
                    section_id,
                    decision.confidence,
                )
                return content, decision
            except (ValueError, KeyError) as e:
                last_error = e
                self.logger.warning("第 %d 次解析失败：%s", attempt + 1, e)

        raise RuntimeError(
            f"生成失败：经过 {self.MAX_RETRIES} 次重试仍无法解析响应。最后错误：{last_error}"
        )

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
            5. XML 格式输出要求

        参数：
            state: 生成状态
            task: 任务描述
            recent_content: 已生成内容（最后 500 字符）
            section_intent: 可选的章节局部计划
            strict: 是否追加格式警告（第 2/3 次重试时为 True）

        返回值：
            str：完整 prompt 字符串
        """
        state_desc = state.to_prompt()
        truncated = recent_content[-self.RECENT_CONTENT_LIMIT:] if recent_content else "（无）"

        strict_note = (
            "\n【注意】上次输出格式不正确，请严格按照下方 XML 格式输出，不得遗漏任何标签。\n"
            if strict
            else ""
        )

        intent_block = ""
        scope_warning = ""
        if section_intent is not None:
            intent_block = f"\n{section_intent.to_prompt_text()}\n"
            scope_warning = (
                "\n【叙事范围强制约束】"
                "本节只负责上述局部计划中的目标，严禁提前完整解决主要冲突或推进属于后续章节的情节。"
                "本节结束时故事必须仍有未解决的张力留待后续章节处理。\n"
            )

        return (
            "你是一个长文本生成系统。\n"
            f"{strict_note}"
            f"\n当前状态：\n{state_desc}"
            f"{intent_block}"
            f"{scope_warning}"
            f"\n最近内容：\n{truncated}"
            f"\n\n当前任务：\n{task}"
            "\n\n要求：严格按照以下 XML 格式输出，不得添加任何额外说明：\n"
            "<decision>本节的核心写作决策</decision>\n"
            "<reasoning>推理过程，引用前文时使用<ref id=\"节ID\">引用片段</ref>格式</reasoning>\n"
            "<expected_effect>预期达到的叙事效果</expected_effect>\n"
            "<confidence>0.0到1.0之间的置信度数字，如0.8</confidence>\n"
            "<content>纯叙事正文，禁止包含节ID、字数统计、XML标签、引用标签或任何元信息</content>"
        )

    # ------------------------------------------------------------------
    # 响应解析
    # ------------------------------------------------------------------

    def _parse_response(self, response: str, state: GenerationState) -> Tuple[str, Decision]:
        """
        解析 LLM XML 格式响应

        功能：
            提取 decision / reasoning / expected_effect / confidence / content 五个字段，
            解析 reasoning 中的 <ref> 引用标签，构建 Decision 对象。

        参数：
            response: LLM 原始响应文本
            state: 当前生成状态（提供 current_section）

        返回值：
            Tuple[str, Decision]：(内容文本, 决策对象)

        异常：
            ValueError：缺少必要字段或 confidence 无法解析时抛出
        """
        decision_text = self._extract_tag(response, "decision")
        reasoning = self._extract_tag(response, "reasoning")
        expected_effect = self._extract_tag(response, "expected_effect")
        confidence_str = self._extract_tag(response, "confidence")
        content = self._sanitize_content(self._extract_tag(response, "content"))

        try:
            confidence = float(confidence_str.strip())
        except ValueError:
            raise ValueError(f"confidence 字段无法解析为浮点数：'{confidence_str}'")

        referenced_sections = self._extract_references(reasoning)

        decision = Decision(
            timestamp=int(time.time()),
            decision_id="",                 # __post_init__ 自动生成 UUID
            decision=decision_text.strip(),
            reasoning=reasoning.strip(),
            expected_effect=expected_effect.strip(),
            confidence=confidence,
            referenced_sections=referenced_sections,
            target_section=state.current_section,
            phase="expanding",
        )

        return content.strip(), decision

    def _extract_tag(self, text: str, tag: str) -> str:
        """
        从文本中提取 XML 标签内容

        参数：
            text: 原始文本
            tag: 标签名（不含尖括号）

        返回值：
            str：标签内内容

        异常：
            ValueError：标签不存在时抛出
        """
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            if tag == "content":
                opening_match = re.search(r"<content>", text, re.DOTALL)
                if opening_match:
                    trailing_content = text[opening_match.end() :].strip()
                    if trailing_content:
                        # 目的：
                        #   某些长输出会在接近 token 上限时被服务端截断，常见症状是
                        #   `<content>` 已经开始但结尾的 `</content>` 丢失。
                        #   这里优先保住已经生成出来的正文，避免因为最后一个闭合标签
                        #   缺失就整轮重试，拖慢全量 benchmark。
                        return trailing_content
            raise ValueError(f"响应中缺少 <{tag}> 标签")
        return match.group(1)

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
        text = re.sub(r'\n*[字字数数]+[：:]\s*\d+[字]?\s*$', '', text.strip())

        return text.strip()

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
