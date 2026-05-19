"""
DenseRetriever — 基于 dense embedding 的统一检索器。

设计原理：
    任务描述与论文摘要之间的词汇鸿沟由 BAAI/bge-m3 的语义向量空间弥合。
    无需假设文档（HyDE）、无需 RRF 融合：单路 dense 检索即可获得语义对齐结果。

    全局检索路：
        任务描述文本 → embed → cosine 相似度 → 去重到 paper 级别 → top_global 篇
    Closing 辅助路：
        固定 limitations/methodology 文本 → embed → cosine → 补充主路未覆盖的论文
    章节级重排：
        section_title + local_goal → embed → cosine 相似度 → 过滤至 global_index → top_section_k 篇

公开 API（与原 HyDERetriever 完全一致，Orchestrator 无需改动）：
    retrieve_global(task)                                  → GlobalPaperIndex
    rank_for_section(global_index, section_title, intent, task) → List[GlobalPaperEntry]

向后兼容别名：
    HyDERetriever = DenseRetriever
    ReferenceRetriever = DenseRetriever

依赖：CorpusLoader（dense embedding 检索）
被依赖：Orchestrator
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .corpus import CorpusLoader
from .types import GlobalPaperEntry, GlobalPaperIndex, ReferenceBundle, ReferenceItem

if TYPE_CHECKING:
    from ..core.plan import SectionIntent

logger = logging.getLogger(__name__)


# ── DenseRetriever ────────────────────────────────────────────────────────────

class DenseRetriever:
    """
    Dense embedding 统一检索器。

    参数：
        corpus:            已加载的 CorpusLoader 实例（含 dense embedding 索引）
        top_global:        全局检索的最大论文数（默认 20）
        top_section_k:     每节展示给写作模型的最大论文数（默认 8）
        top_closing_extra: Closing 辅助路补充的最大论文数（默认 8）

    检索策略：
        主路：任务描述文本直接 embed → cosine 相似度 → top_global 篇
        辅助路：固定 limitations/methodology 文本 embed → 补充主路未覆盖的论文
        章节重排：section_title + local_goal embed → 在 global_index 内重排
    """

    # Closing section 辅助路固定查询文本
    _CLOSING_QUERY = (
        "limitations methodology bias evidence gaps systematic review "
        "clinical trial validation cohort study"
    )

    def __init__(
        self,
        corpus: CorpusLoader,
        top_global: int = 20,
        top_section_k: int = 8,
        top_closing_extra: int = 8,
        # 以下参数保留以兼容旧调用签名（不再使用）
        llm_client: Optional[Any] = None,
        min_score: float = 0.0,
    ) -> None:
        self._corpus = corpus
        self._top_global = top_global
        self._top_section_k = top_section_k
        self._top_closing_extra = top_closing_extra
        self._run_logger: Optional[Any] = None

    def attach_run_logger(self, run_logger: Any) -> None:
        """注入 RunLogger，使检索过程可观测。"""
        self._run_logger = run_logger

    # ── 全局检索 ──────────────────────────────────────────────────────────────

    def retrieve_global(self, task: str) -> GlobalPaperIndex:
        """
        为整个写作任务构建全局论文索引（GlobalPaperIndex）。

        流程：
            第一阶段（主路）：
                1. 用任务描述文本做 dense 检索（corpus.search 内部已做 embedding）
                2. 去重到 paper 级别（每篇保留最高分 chunk）
                3. 取 top_global 篇构建主路论文列表

            第二阶段（Closing 辅助路）：
                4. 固定 limitations/methodology 文本做 dense 检索
                5. 补充主路未覆盖的新论文（最多 top_closing_extra 篇）
        """
        if self._corpus.paper_count() == 0:
            logger.warning("DenseRetriever: 语料库为空，返回空全局索引")
            return GlobalPaperIndex()

        # 第一阶段：主路 dense 检索
        raw = self._corpus.search(task, top_k=self._top_global * 6)
        best_per_paper = _best_chunk_per_paper(raw)
        top_pids = _sort_ids_by_score(best_per_paper)[: self._top_global]

        logger.info(
            "DenseRetriever.retrieve_global（主路）: task=%d字符 → %d paper命中 → top%d",
            len(task), len(best_per_paper), len(top_pids),
        )
        if self._run_logger is not None:
            top3 = top_pids[:3]
            self._run_logger.log_retrieval_event(
                "DENSE_GLOBAL",
                task_chars=len(task),
                paper_hits=len(best_per_paper),
                top3=[
                    f"score={best_per_paper[p]['score']:.4f}  "
                    f"title={self._corpus.get_paper(p) and self._corpus.get_paper(p).get('title','')[:55]}"
                    for p in top3
                ],
            )

        # 构建 GlobalPaperEntry 列表（主路）
        entries: List[GlobalPaperEntry] = []
        for rank, pid in enumerate(top_pids, start=1):
            entry = self._make_entry(pid, best_per_paper[pid], rank)
            entries.append(entry)

        # 第二阶段：Closing 辅助路
        closing_extra_count = 0
        if self._top_closing_extra > 0:
            closing_raw = self._corpus.search(
                self._CLOSING_QUERY,
                top_k=self._top_closing_extra * 4,
            )
            closing_best = _best_chunk_per_paper(closing_raw)
            main_pids = {e.paper_id for e in entries}
            new_closing_ids = [
                pid for pid in _sort_ids_by_score(closing_best)
                if pid not in main_pids
            ][: self._top_closing_extra]

            for pid in new_closing_ids:
                entry = self._make_entry(pid, closing_best[pid], len(entries) + 1)
                entries.append(entry)
                closing_extra_count += 1

        index = GlobalPaperIndex(entries=entries)
        logger.info(
            "DenseRetriever.retrieve_global: 共 %d 篇（主路=%d, closing辅助=%d）",
            len(entries), len(top_pids), closing_extra_count,
        )
        return index

    # ── 章节级重排 ────────────────────────────────────────────────────────────

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
            查询文本 = section_title + " " + local_goal（dense 检索，无 Mini-HyDE）
            检索结果过滤至 global_index 内的 paper_id，按 cosine 相似度重排。

        降级：
            global_index 为空 → 返回 []
            global_index.entries ≤ top_section_k → 直接返回所有条目
        """
        if global_index.is_empty():
            return []

        if len(global_index.entries) <= self._top_section_k:
            return list(global_index.entries)

        entry_map: Dict[str, GlobalPaperEntry] = {e.paper_id: e for e in global_index.entries}
        paper_ids_in_index: set = set(entry_map.keys())

        # 构建章节查询文本
        local_goal = (section_intent.local_goal if section_intent else "") or ""
        section_query = (section_title + " " + local_goal).strip() or task

        # Dense 检索，过滤到 global_index 内
        raw = self._corpus.search(section_query, top_k=len(paper_ids_in_index) * 4)
        best_in_index: Dict[str, float] = {}
        for r in raw:
            pid = r["paper_id"]
            if pid in paper_ids_in_index:
                if pid not in best_in_index or r["score"] > best_in_index[pid]:
                    best_in_index[pid] = r["score"]

        if best_in_index:
            ranked_ids = sorted(best_in_index, key=lambda p: best_in_index[p], reverse=True)
            # 追加未命中的 paper（保证返回完整列表）
            seen = set(ranked_ids)
            for e in global_index.entries:
                if e.paper_id not in seen:
                    ranked_ids.append(e.paper_id)
        else:
            # 无命中：按全局顺序保留
            ranked_ids = [e.paper_id for e in global_index.entries]

        result = [entry_map[pid] for pid in ranked_ids[: self._top_section_k]]
        logger.info(
            "DenseRetriever.rank_for_section: section=%s → %d/%d 篇",
            section_title[:40], len(result), len(global_index.entries),
        )
        if self._run_logger is not None:
            self._run_logger.log_retrieval_event(
                "DENSE_SECTION",
                section=section_title[:60],
                query_chars=len(section_query),
                selected=len(result),
                total_in_index=len(global_index.entries),
                top_papers=[
                    f"[R{e.r_index}] score={e.retrieval_score:.4f}  title={e.title[:55]}"
                    for e in result
                ],
            )
        return result

    # ── 记忆模块接口（预留）──────────────────────────────────────────────────

    def retrieve_from_memory(
        self,
        intent: str,
        memory_corpus: "CorpusLoader",
    ) -> List[dict]:
        """
        【记忆模块接口 — 预留，尚未实现】

        给定当前章节意图，从历史生成内容（memory_corpus）中检索最相关的过去段落。

        Raises:
            NotImplementedError: 待 MemoryModule 接入后实现
        """
        raise NotImplementedError(
            "retrieve_from_memory 接口预留，待 MemoryModule 接入后实现。"
        )

    # ── 向后兼容旧接口 ────────────────────────────────────────────────────────

    def retrieve(self, query: str, section_id: str) -> ReferenceBundle:
        """旧接口兼容层，新代码请用 retrieve_global + rank_for_section。"""
        if self._corpus.paper_count() == 0:
            return ReferenceBundle(section_id=section_id, query=query)

        raw = self._corpus.search(query, top_k=self._top_global * 3)
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
                    paper_id        = chunk["paper_id"],
                    title           = chunk["title"],
                    chunk_id        = chunk["chunk_id"],
                    text            = chunk["text"],
                    rank            = rank,
                    retrieval_score = chunk["score"],
                ))
                rank += 1

        return ReferenceBundle(section_id=section_id, query=query, items=items)

    @staticmethod
    def build_query(
        task: str,
        section_title: str,
        section_intent: Optional["SectionIntent"] = None,
    ) -> str:
        """向后兼容的 query 构建方法（dense 检索不依赖此方法，仅供遗留路径使用）。"""
        parts: List[str] = [task, section_title]
        if section_intent is not None and section_intent.local_goal:
            parts.append(section_intent.local_goal)
        return " ".join(p for p in parts if p)

    # ── 私有辅助 ──────────────────────────────────────────────────────────────

    def _make_entry(
        self,
        pid: str,
        chunk_result: dict,
        rank: int,
    ) -> GlobalPaperEntry:
        """从 paper_id 和最佳 chunk 结果构建 GlobalPaperEntry。"""
        paper_meta = self._corpus.get_paper(pid) or {}
        raw_authors = paper_meta.get("authors", [])
        authors: List[str] = []
        for a in raw_authors:
            if isinstance(a, dict):
                authors.append(a.get("name", ""))
            elif isinstance(a, str):
                authors.append(a)

        abstract = paper_meta.get("abstract", "") or paper_meta.get("index_text", "")
        return GlobalPaperEntry(
            r_index         = rank,
            paper_id        = pid,
            title           = paper_meta.get("title", chunk_result.get("title", "Unknown Title")),
            abstract        = abstract,
            top_chunk_text  = chunk_result.get("text", ""),
            authors         = authors,
            doi             = paper_meta.get("doi", ""),
            retrieval_score = chunk_result["score"],
        )


# ── 向后兼容别名 ───────────────────────────────────────────────────────────────

#: HyDERetriever 是 DenseRetriever 的别名，保持旧代码路径不变。
HyDERetriever = DenseRetriever

#: ReferenceRetriever 是 DenseRetriever 的别名，保持旧代码路径不变。
ReferenceRetriever = DenseRetriever


# ── 私有辅助函数 ───────────────────────────────────────────────────────────────

def _best_chunk_per_paper(raw_results: List[dict]) -> Dict[str, dict]:
    """
    将 chunk 级结果去重到 paper 级别，每篇保留最高 score 的 chunk。

    参数：
        raw_results: corpus.search() 的原始返回列表

    返回：
        Dict[paper_id, best_chunk_dict]
    """
    best: Dict[str, dict] = {}
    for r in raw_results:
        pid = r["paper_id"]
        if pid not in best or r["score"] > best[pid]["score"]:
            best[pid] = r
    return best


def _sort_ids_by_score(best_per_paper: Dict[str, dict]) -> List[str]:
    """按 score 降序返回 paper_id 列表。"""
    return sorted(best_per_paper, key=lambda p: best_per_paper[p]["score"], reverse=True)
