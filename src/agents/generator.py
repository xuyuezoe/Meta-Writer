import re
import time
import logging
from typing import Tuple, List, Optional

from ..core.state import GenerationState
from ..core.decision import Decision


class Generator:
    """
    生成器 - 负责生成内容并输出结构化Decision

    流程：
      1. 构建prompt（包含状态、任务、上下文）
      2. 调用LLM生成
      3. 解析XML响应
      4. 提取Decision和Content
      5. 重试机制（最多3次）

    还没有做针对llm问询的plan流程，目前是直接问询
    """

    MAX_RETRIES = 3
    RECENT_CONTENT_LIMIT = 500

    def __init__(self, llm_client):
        self.llm = llm_client
        self.logger = logging.getLogger(__name__)

    def generate_with_decision(
        self,
        state: GenerationState,
        task: str,
        recent_content: str = "",
    ) -> Tuple[str, Decision]:
        """
        生成内容并返回决策

        重试机制：最多3次，逐渐加强prompt要求
        第1次：正常prompt
        第2次：强调格式规范
        第3次：强制要求严格遵循格式，否则视为失败
        """
        last_error: Optional[Exception] = None

        for attempt in range(self.MAX_RETRIES):
            prompt = self._build_prompt(state, task, recent_content, strict=(attempt > 0))
            self.logger.debug("第%d次尝试生成，section=%s", attempt + 1, state.current_section)

            try:
                response = self.llm.generate(prompt, temperature=0.7, max_tokens=2048)
                content, decision = self._parse_response(response, state)
                self.logger.info(
                    "生成成功 [尝试%d] section=%s confidence=%.2f",
                    attempt + 1,
                    state.current_section,
                    decision.confidence,
                )
                return content, decision
            except (ValueError, KeyError) as e:
                last_error = e
                self.logger.warning("第%d次解析失败：%s", attempt + 1, e)
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(1)

        raise RuntimeError(
            f"生成失败：经过{self.MAX_RETRIES}次重试仍无法解析响应。最后错误：{last_error}"
        )

    # ------------------------------------------------------------------
    # Prompt构建
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        state: GenerationState,
        task: str,
        recent_content: str,
        strict: bool = False,
    ) -> str:
        """
        构建prompt

        包含：
        1. 状态描述（state.to_prompt()）
        2. 最近内容（截断到500字符）
        3. 当前任务
        4. 输出格式要求（XML标签）
        """
        state_desc = state.to_prompt()
        truncated = recent_content[-self.RECENT_CONTENT_LIMIT:] if recent_content else "（无）"

        strict_note = (
            "\n【注意】上次输出格式不正确，请严格按照下方XML格式输出，不得遗漏任何标签。\n"
            if strict
            else ""
        )

        return f"""你是一个长文本生成系统。
{strict_note}
当前状态：
{state_desc}

最近内容：
{truncated}

当前任务：
{task}

要求：严格按照以下格式输出：
<decision>
[决策描述]
</decision>
<reasoning>
[推理过程，必须包含<ref id="xxx">...</ref>引用]
</reasoning>
<expected_effect>
[预期效果]
</expected_effect>
<confidence>
[0.0-1.0的数字]
</confidence>
<content>
[实际生成的内容]
</content>"""

    # ------------------------------------------------------------------
    # 响应解析
    # ------------------------------------------------------------------

    def _parse_response(self, response: str, state: GenerationState) -> Tuple[str, Decision]:
        """
        解析LLM响应

        提取：
        - <decision>
        - <reasoning>（包含<ref>标签）
        - <expected_effect>
        - <confidence>
        - <content>

        返回：(content, Decision对象)
        """
        decision_text = self._extract_tag(response, "decision")
        reasoning = self._extract_tag(response, "reasoning")
        expected_effect = self._extract_tag(response, "expected_effect")
        confidence_str = self._extract_tag(response, "confidence")
        content = self._extract_tag(response, "content")

        try:
            confidence = float(confidence_str.strip())
        except ValueError:
            raise ValueError(f"confidence字段无法解析为浮点数：'{confidence_str}'")

        referenced_sections = self._extract_references(reasoning)

        decision = Decision(
            timestamp=int(time.time()),
            decision_id="",                        # __post_init__自动生成UUID
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
        从文本中提取XML标签内容，标签缺失时抛出ValueError
        """
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            raise ValueError(f"响应中缺少 <{tag}> 标签")
        return match.group(1)

    def _extract_references(self, reasoning: str) -> List[Tuple[str, str]]:
        """
        从reasoning中提取引用

        格式：<ref id="sec1">snippet</ref>
        返回：[("sec1", "snippet"), ...]
        """
        pattern = r'<ref\s+id="([^"]+)">(.*?)</ref>'
        matches = re.findall(pattern, reasoning, re.DOTALL)
        return [(section_id.strip(), snippet.strip()) for section_id, snippet in matches]
