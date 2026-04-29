"""
ReferenceRetriever — 全局论文索引构建 + per-section 相关性排序。

设计变更（新架构）：
    - retrieve_global(): 用 task 描述一次性检索 top_global 篇论文，
      分配全局 R1…RN 索引，返回 GlobalPaperIndex。
    - rank_for_section(): 从全局集合里为某节选出最相关的 top_section_k 篇，
      返回 GlobalPaperEntry 列表（保留全局 r_index，不重编号）。
    - 修复 BM25 query：补充噪音词过滤，保留真实域名词。
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

from .corpus import CorpusLoader
from .types import GlobalPaperEntry, GlobalPaperIndex, ReferenceBundle, ReferenceItem

if TYPE_CHECKING:
    from ..core.plan import SectionIntent

logger = logging.getLogger(__name__)


class ReferenceRetriever:
    """
    参数说明：
        corpus:          已加载的 CorpusLoader 实例
        top_global:      全局检索的最大论文数（默认 12）
        top_section_k:   每节展示给 LLM 的最大论文数（默认 8）
        min_score:       BM25 分数最低阈值
    """

    def __init__(
        self,
        corpus: CorpusLoader,
        top_global: int = 12,
        top_section_k: int = 8,
        min_score: float = 0.1,
    ):
        self._corpus = corpus
        self._top_global = top_global
        self._top_section_k = top_section_k
        self._min_score = min_score

    # ── 公开 API ──────────────────────────────────────────────────────────────

    def retrieve_global(self, task: str) -> GlobalPaperIndex:
        """
        用任务描述一次性检索全局论文集合，构建 GlobalPaperIndex。

        流程：
            1. 从 task 提取域名词构建 BM25 query
            2. 检索 top_global × 3 个 chunk 候选
            3. 按论文去重，取每篇得分最高的 chunk
            4. 选 top_global 篇，分配 r_index（1-based）
            5. 为每篇论文填充完整摘要 + top chunk 全文（不截断）
        """
        if self._corpus.paper_count() == 0:
            logger.warning("ReferenceRetriever: 语料库为空，返回空全局索引")
            return GlobalPaperIndex()

        query = self._extract_topic_terms_joined(task)
        if not query:
            logger.warning("ReferenceRetriever: task 提取不到域名词，返回空全局索引")
            return GlobalPaperIndex()

        raw = self._corpus.search(query, top_k=self._top_global * 6)
        raw = [r for r in raw if r["score"] >= self._min_score]

        if not raw:
            logger.info("ReferenceRetriever: 全局检索无结果（min_score=%.2f）", self._min_score)
            return GlobalPaperIndex()

        # 按论文去重，保留每篇最高分 chunk
        best_chunk_per_paper: Dict[str, dict] = {}
        for r in raw:
            pid = r["paper_id"]
            if pid not in best_chunk_per_paper or r["score"] > best_chunk_per_paper[pid]["score"]:
                best_chunk_per_paper[pid] = r

        # 按得分降序取 top_global 篇
        sorted_papers = sorted(
            best_chunk_per_paper.values(),
            key=lambda x: x["score"],
            reverse=True,
        )[: self._top_global]

        entries: List[GlobalPaperEntry] = []
        for rank, chunk_result in enumerate(sorted_papers, start=1):
            pid = chunk_result["paper_id"]
            paper_meta = self._corpus.get_paper(pid) or {}

            # 作者列表
            raw_authors = paper_meta.get("authors", [])
            authors: List[str] = []
            for a in raw_authors:
                if isinstance(a, dict):
                    authors.append(a.get("name", ""))
                elif isinstance(a, str):
                    authors.append(a)

            # 摘要：优先 abstract 字段，退而求其次 index_text
            abstract = paper_meta.get("abstract", "") or paper_meta.get("index_text", "")

            # top chunk 完整文本（不截断）
            top_chunk_text = chunk_result.get("text", "")

            entries.append(GlobalPaperEntry(
                r_index=rank,
                paper_id=pid,
                title=paper_meta.get("title", chunk_result.get("title", "Unknown Title")),
                abstract=abstract,
                top_chunk_text=top_chunk_text,
                authors=authors,
                doi=paper_meta.get("doi", ""),
                retrieval_score=chunk_result["score"],
            ))

        index = GlobalPaperIndex(entries=entries)
        logger.info(
            "ReferenceRetriever.retrieve_global: 检索到 %d 篇论文（query=%s...）",
            len(entries),
            query[:60],
        )
        return index

    def rank_for_section(
        self,
        global_index: GlobalPaperIndex,
        section_title: str,
        section_intent: Optional["SectionIntent"] = None,
        task: str = "",
    ) -> List[GlobalPaperEntry]:
        """
        从全局索引中为某节挑选最相关的 top_section_k 篇。

        策略：
            构建章节专属 query，对全局索引中各论文的标题+摘要做 BM25 内部重排。
            若全局索引条数 ≤ top_section_k，直接返回全部（无需重排）。
        """
        if global_index.is_empty():
            return []

        if len(global_index.entries) <= self._top_section_k:
            return list(global_index.entries)

        # 构建章节 query（保留章节标题词和 local_goal 词）
        query_parts: List[str] = []
        if task:
            query_parts.append(self._extract_topic_terms_joined(task))
        section_terms = self._extract_section_terms(section_title)
        if section_terms:
            query_parts.append(" ".join(section_terms))
        if section_intent is not None and section_intent.local_goal:
            goal_terms = self._extract_section_terms(section_intent.local_goal)
            if goal_terms:
                query_parts.append(" ".join(goal_terms))

        query = " ".join(q for q in query_parts if q)
        if not query:
            return global_index.entries[: self._top_section_k]

        from .corpus import _tokenize
        q_tokens = _tokenize(query)

        # 对每篇论文的标题+摘要做简单 BM25 打分（已有实现直接复用语料库）
        # 简化做法：对标题+摘要拼接字符串做 BM25 逻辑近似 → 用词频交集计数
        def score_entry(entry: GlobalPaperEntry) -> float:
            text = (entry.title + " " + entry.abstract).lower()
            return sum(1 for t in q_tokens if t in text)

        ranked = sorted(global_index.entries, key=score_entry, reverse=True)
        result = ranked[: self._top_section_k]

        logger.info(
            "ReferenceRetriever.rank_for_section: section_title=%s → 选出 %d/%d 篇",
            section_title[:40],
            len(result),
            len(global_index.entries),
        )
        return result

    # ── 内部兼容方法（供 Orchestrator 旧代码路径使用，逐步废弃） ─────────────

    def retrieve(self, query: str, section_id: str) -> ReferenceBundle:
        """保留旧接口签名（仅内部使用），新代码请用 retrieve_global + rank_for_section。"""
        if self._corpus.paper_count() == 0:
            return ReferenceBundle(section_id=section_id, query=query)

        raw = self._corpus.search(query, top_k=self._top_global * 3)
        raw = [r for r in raw if r["score"] >= self._min_score]
        if not raw:
            return ReferenceBundle(section_id=section_id, query=query)

        paper_buckets: Dict[str, List[dict]] = {}
        for r in raw:
            pid = r["paper_id"]
            paper_buckets.setdefault(pid, [])
            if len(paper_buckets[pid]) < 2:
                paper_buckets[pid].append(r)

        sorted_papers = sorted(
            paper_buckets.items(),
            key=lambda kv: kv[1][0]["score"],
            reverse=True,
        )[: self._top_global]

        items: List[ReferenceItem] = []
        rank = 0
        for _pid, chunks in sorted_papers:
            for chunk in chunks:
                items.append(ReferenceItem(
                    paper_id=chunk["paper_id"],
                    title=chunk["title"],
                    chunk_id=chunk["chunk_id"],
                    text=chunk["text"],
                    rank=rank,
                    retrieval_score=chunk["score"],
                ))
                rank += 1

        return ReferenceBundle(section_id=section_id, query=query, items=items)

    # ── Query 构建 ────────────────────────────────────────────────────────────

    @staticmethod
    def build_query(
        task: str,
        section_title: str,
        section_intent: Optional["SectionIntent"] = None,
    ) -> str:
        """向后兼容的 query 构建方法（旧代码路径使用）。"""
        parts: List[str] = []
        topic = ReferenceRetriever._extract_topic_terms_joined(task)
        if topic:
            parts.append(topic)
        section = ReferenceRetriever._extract_section_terms_joined(section_title)
        if section:
            parts.append(section)
        if section_intent is not None and section_intent.local_goal:
            goal = ReferenceRetriever._extract_section_terms_joined(section_intent.local_goal)
            if goal:
                parts.append(goal)
        return " ".join(parts)

    @staticmethod
    def _extract_topic_terms_joined(text: str) -> str:
        return " ".join(ReferenceRetriever._extract_topic_terms(text))

    @staticmethod
    def _extract_section_terms_joined(text: str) -> str:
        return " ".join(ReferenceRetriever._extract_section_terms(text))

    @staticmethod
    def _extract_topic_terms(text: str) -> List[str]:
        """
        从任务描述中提取领域名词，过滤写作指令词汇和介词。

        相比旧版新增过滤词：
            case, report, popular, science, summary, medicine, english,
            article, word, target, length, body（任务描述中的写作规范词）
        """
        import re

        TASK_INSTRUCTION_WORDS = {
            # 写作动作
            "write", "define", "describe", "explain", "outline", "review",
            "organize", "conduct", "compare", "summarize", "present", "introduce",
            "cover", "include", "develop", "close", "open", "begin", "ending",
            "provide", "discuss", "ensure", "keep", "build", "create", "make",
            # 写作结构词
            "section", "article", "paragraph", "word", "piece", "body", "text",
            "form", "long", "scholarly", "rather", "than", "brief", "outline",
            "survey", "paper", "review", "report", "summary", "overview",
            # 写作副词/介词/限定词
            "approximately", "about", "with", "from", "this", "that", "their",
            "these", "while", "around", "main", "line", "least", "natural",
            "within", "each", "also", "both", "more", "most", "some", "must",
            "should", "will", "have", "been", "into", "onto", "using", "style",
            # 新增：任务描述中出现的非领域词
            "case", "popular", "science", "medicine", "english", "target",
            "length", "entire", "fully", "explicitly", "periodically",
            "approximately", "requirement", "constraint", "document",
        }
        # 同时捕获全大写缩写词（ARDS、ECMO、PEEP 等）和普通小写词
        tokens = re.findall(r'\b(?:[A-Z]{2,}|[A-Za-z][a-z]{3,})\b', text)
        return [t for t in tokens if t.lower() not in TASK_INSTRUCTION_WORDS]

    @staticmethod
    def _extract_section_terms(text: str) -> List[str]:
        """
        从章节标题/local_goal 提取词汇，过滤范围比任务描述更窄：
        保留 subgroup、stratification、limitations 等真实域名词。
        """
        import re

        SECTION_INSTRUCTION_WORDS = {
            "write", "define", "describe", "explain", "organize", "conduct",
            "compare", "summarize", "present", "provide", "discuss", "ensure",
            "develop", "close", "open", "include", "cover",
            "section", "paragraph", "piece", "form", "around", "using",
            "with", "from", "this", "that", "their", "these", "each",
            "also", "both", "must", "should", "will", "have", "been",
            "approximately", "about", "within", "while", "main",
        }
        # 同时捕获全大写缩写词（ARDS、ECMO 等）和普通小写词
        tokens = re.findall(r'\b(?:[A-Z]{2,}|[A-Za-z][a-z]{3,})\b', text)
        return [t for t in tokens if t.lower() not in SECTION_INSTRUCTION_WORDS]
