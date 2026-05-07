"""
HyDE（Hypothetical Document Embeddings 变体）生成器。

核心原理：
    任务描述（写作指令语言）与论文摘要（科学文献语言）处于不同词汇空间，
    BM25 与RAG无法跨越此鸿沟。HyDE 通过让 LLM 先生成一段「假设文档」
    （与语料库同体裁、同词汇空间的文本），再用该文档做 BM25 查询，
    从根本上解决词汇鸿沟问题。

降级策略：
    llm_client 为 None 时静默跳过 LLM 调用，返回 None；
    调用方（HyDERetriever）在收到 None 时自动退化到纯关键词 BM25。

缓存策略：
    实例级内存缓存（session 生命周期），key = sha256(intent + style)[:16]。
    同一任务的全局 HyDE 只调用一次 LLM。
"""
from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from ..utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

# ── Prompt 模板 ────────────────────────────────────────────────────────────────

_TASK_PROMPT = """\
You are a biomedical literature expert. Given a writing task, generate a \
150-word scientific abstract as if it came from one of the key papers \
that a researcher would cite when writing this article. \
Use domain-specific technical terminology as it appears in scientific papers. \
Do not include author names, journal names, or section headers. \
Output only the abstract text.

Task: {task}

Abstract:"""

_SECTION_PROMPT = """\
You are a biomedical literature expert. Generate a 60-word passage using \
the technical vocabulary that would appear in a research paper directly \
relevant to this section of a scholarly review article. \
Output only the passage text, no preamble.

Section title: {section_title}
Section goal: {local_goal}

Passage:"""

_MEMORY_PROMPT = """\
You are summarizing the key technical content that a previous section \
of a long-form article would have discussed, given the following intent. \
Generate a 60-word descriptive passage in the style of academic prose. \
Output only the passage text.

Section intent: {intent}

Passage:"""


# ── HyDEGenerator ─────────────────────────────────────────────────────────────

class HyDEGenerator:
    """
    假设文档生成器（HyDE / Query2Doc 变体）。

    三种生成模式：
        generate_for_task    — 给定写作任务，生成任务级假设摘要（~150词）
        generate_for_section — 给定章节意图，生成章节级假设片段（~60词）
        generate_for_memory  — 【记忆模块接口预留】生成历史内容假设片段

    参数：
        无。llm_client 在每次调用时传入，便于测试时注入 mock。
    """

    def __init__(self) -> None:
        self._cache: Dict[str, str] = {}

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def generate_for_task(
        self,
        task: str,
        llm_client: Optional["LLMClient"],
    ) -> Optional[str]:
        """
        为整个写作任务生成假设摘要，用于全局 BM25 检索。

        参数：
            task:       完整任务描述字符串
            llm_client: LLM 客户端；为 None 时直接返回 None（退化到关键词检索）

        返回：
            假设摘要文本（~150词），LLM 不可用或调用失败时返回 None
        """
        return self._generate(
            key=self._cache_key(task, "task"),
            prompt=_TASK_PROMPT.format(task=task[:1200]),
            label="task",
            llm_client=llm_client,
            max_tokens=2000,
        )

    def generate_for_section(
        self,
        section_title: str,
        local_goal: str,
        llm_client: Optional["LLMClient"],
    ) -> Optional[str]:
        """
        为章节意图生成假设片段，用于章节级 BM25 重排。

        参数：
            section_title: 章节标题
            local_goal:    章节局部目标（SectionIntent.local_goal）
            llm_client:    LLM 客户端；为 None 时返回 None

        返回：
            假设片段文本（~60词），失败时返回 None
        """
        combined = f"{section_title}|{local_goal}"
        return self._generate(
            key=self._cache_key(combined, "section"),
            prompt=_SECTION_PROMPT.format(
                section_title=section_title[:200],
                local_goal=(local_goal or "")[:400],
            ),
            label=f"section:{section_title[:30]}",
            llm_client=llm_client,
            max_tokens=800,
        )

    def generate_for_memory(
        self,
        intent: str,
        llm_client: Optional["LLMClient"],
    ) -> Optional[str]:
        """
        【记忆模块接口 — 预留】

        给定当前章节意图，生成「假设历史片段」，供未来 MemoryRetriever
        对历史生成内容做 BM25 检索时使用。

        接入方式（待 MemoryModule 实现时）：
            memory_retriever = HyDERetriever(memory_corpus, llm_client=llm_client)
            past = memory_retriever.retrieve_from_memory(intent=section_intent.local_goal)

        参数：
            intent:     当前章节意图文本（SectionIntent.local_goal）
            llm_client: LLM 客户端

        返回：
            假设历史片段文本，失败时返回 None
        """
        return self._generate(
            key=self._cache_key(intent, "memory"),
            prompt=_MEMORY_PROMPT.format(intent=intent[:400]),
            label="memory",
            llm_client=llm_client,
            max_tokens=800,
        )

    # ── 内部实现 ──────────────────────────────────────────────────────────────

    def _generate(
        self,
        key: str,
        prompt: str,
        label: str,
        llm_client: Optional["LLMClient"],
        max_tokens: int,
    ) -> Optional[str]:
        """统一生成入口：缓存命中则直接返回，否则调用 LLM。"""
        if key in self._cache:
            logger.debug("HyDEGenerator: cache hit [%s]", label)
            return self._cache[key]

        if llm_client is None:
            logger.debug("HyDEGenerator: llm_client=None，跳过生成 [%s]", label)
            return None

        try:
            # allow_think_only_fallback=True：推理模型若只产出 <think> 内容，
            # llm_client 会提取 think 块内容而非返回空串（见 llm_client.generate 的处理逻辑）
            text = llm_client.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=max_tokens,
                allow_think_only_fallback=True,
                log_meta={"caller": f"HyDEGenerator.{label}"},
            )
            text = text.strip()
            if not text:
                logger.warning("HyDEGenerator: LLM 返回空文本 [%s]", label)
                return None

            self._cache[key] = text
            logger.info(
                "HyDEGenerator: 生成假设文档 [%s] %d词",
                label,
                len(text.split()),
            )
            return text

        except Exception as exc:
            logger.warning("HyDEGenerator: LLM 调用失败 [%s] %s", label, exc)
            return None

    @staticmethod
    def _cache_key(text: str, style: str) -> str:
        return hashlib.sha256(f"{text}|{style}".encode()).hexdigest()[:16]
