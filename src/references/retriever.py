"""
HyDERetriever — 基于 HyDE + BM25 双路检索 + RRF 融合的统一检索器。

设计原理：
    任务描述（写作指令语言）与论文摘要（科学文献语言）处于不同词汇空间。
    本模块通过两路并行检索 + RRF 融合解决词汇鸿沟问题：

    第一路（关键词 BM25）：从任务描述提取领域词，直接 BM25 检索。
        优点：稳定，对显式术语命中准确。
        缺点：词汇鸿沟严重时召回率低。

    第二路（HyDE BM25）：LLM 生成假设摘要（与论文同语言空间），BM25 检索。
        优点：跨越词汇鸿沟，大幅提升 ACS 等临床任务的召回率。
        缺点：依赖 LLM，有调用延迟。

    RRF 融合：两路排名用 Reciprocal Rank Fusion 合并，兼顾准确率与召回率。
    当 HyDE 失败（LLM 不可用 / 调用出错），系统退化为纯关键词 BM25。

公共 API（与旧 ReferenceRetriever 完全一致，Orchestrator 无需改动）：
    retrieve_global(task) → GlobalPaperIndex
    rank_for_section(global_index, section_title, section_intent, task) → List[GlobalPaperEntry]

记忆模块接口（预留）：
    retrieve_from_memory(intent, memory_corpus) → List[dict]
    此方法在 MemoryModule 接入前会抛出 NotImplementedError。

ReferenceRetriever = HyDERetriever（向后兼容别名）。
"""
from __future__ import annotations

import logging
import re
import string
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .corpus import CorpusLoader
from .fusion import reciprocal_rank_fusion
from .hyde import HyDEGenerator
from .types import GlobalPaperEntry, GlobalPaperIndex, ReferenceBundle, ReferenceItem

if TYPE_CHECKING:
    from ..core.plan import SectionIntent
    from ..utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


# ── HyDERetriever ─────────────────────────────────────────────────────────────

class HyDERetriever:
    """
    统一检索器：HyDE BM25 + 关键词 BM25 + RRF 融合 + Closing 辅助路。

    参数：
        corpus:            已加载的 CorpusLoader 实例（论文语料库）
        llm_client:        LLM 客户端；为 None 时退化为纯关键词 BM25
        top_global:        全局检索的最大论文数（默认 20）
        top_section_k:     每节展示给 LLM 的最大论文数（默认 8）
        min_score:         BM25 分数最低阈值（过滤噪声）
        top_closing_extra: Closing 辅助路额外补充的最大论文数（默认 8）。
                           这些论文针对 limitations/methodology 类终结节检索，
                           补充到全局索引末尾，r_index 接续主路编号。

    三路全局检索设计：
        主路1（关键词 BM25）：提取领域词，直接 BM25 检索主题论文。
        主路2（HyDE BM25）：LLM 生成假设摘要，BM25 检索，弥补词汇鸿沟。
        辅助路3（Closing BM25）：固定 limitations/methodology 关键词检索，
            专为 sec5/sec6/sec7 等 closing sections 提供可引用的方法论文献。
            只补充主路未覆盖的新 paper_id，不重复。
    """

    # 辅助路固定关键词：覆盖 limitations / evidence gaps / methodology 场景
    _CLOSING_QUERY = (
        "limitations methodology bias evidence gaps systematic review "
        "clinical trial validation cohort study"
    )

    def __init__(
        self,
        corpus: CorpusLoader,
        llm_client: Optional["LLMClient"] = None,
        top_global: int = 20,
        top_section_k: int = 8,
        min_score: float = 0.1,
        top_closing_extra: int = 8,
    ) -> None:
        self._corpus = corpus
        self._llm_client = llm_client
        self._top_global = top_global
        self._top_section_k = top_section_k
        self._min_score = min_score
        self._top_closing_extra = top_closing_extra
        self._hyde = HyDEGenerator()
        self._run_logger = None

    def attach_run_logger(self, run_logger: Any) -> None:
        """注入 RunLogger，使检索过程可观测（与 LLMClient.attach_run_logger 模式一致）。"""
        self._run_logger = run_logger

    # ── 全局检索 ──────────────────────────────────────────────────────────────

    def retrieve_global(self, task: str) -> GlobalPaperIndex:
        """
        为整个写作任务构建全局论文索引（GlobalPaperIndex）。

        流程：
            1. 关键词 BM25：提取领域词 → BM25 检索
            2. HyDE BM25：LLM 生成假设摘要 → BM25 检索
            3. 各路结果去重到 paper 级别（保留最高分 chunk）
            4. RRF 融合两路排名，取 top_global 篇为主路
            5. 辅助路（Closing BM25）：固定 limitations/methodology 关键词检索，
               补充主路未覆盖的新论文（最多 top_closing_extra 篇），r_index 接续主路

        降级：
            llm_client=None 或 HyDE 调用失败 → 仅使用关键词 BM25 结果。
            top_closing_extra=0 → 跳过辅助路。
        """
        if self._corpus.paper_count() == 0:
            logger.warning("HyDERetriever: 语料库为空，返回空全局索引")
            return GlobalPaperIndex()

        # 第一路：关键词 BM25
        kw_query = self._extract_topic_terms_joined(task)
        kw_raw: List[dict] = []
        if kw_query:
            kw_raw = self._corpus.search(kw_query, top_k=self._top_global * 6)
            kw_raw = [r for r in kw_raw if r["score"] >= self._min_score]
        logger.info(
            "HyDE第一路(KW-BM25): query='%s...' → %d chunk命中",
            kw_query[:60], len(kw_raw),
        )
        if self._run_logger is not None:
            top3_kw = sorted(kw_raw, key=lambda r: r["score"], reverse=True)[:3]
            self._run_logger.log_retrieval_event(
                "KW_BM25",
                query=kw_query[:120],
                hits=len(kw_raw),
                top3=[
                    f"score={r['score']:.3f}  title={r.get('title','')[:60]}"
                    for r in top3_kw
                ],
            )

        # 第二路：HyDE BM25
        hyde_raw: List[dict] = []
        hypothetical = self._hyde.generate_for_task(task, self._llm_client)
        if hypothetical:
            hyde_raw = self._corpus.search(hypothetical, top_k=self._top_global * 6)
            hyde_raw = [r for r in hyde_raw if r["score"] >= self._min_score]
            logger.info(
                "HyDE第二路(HyDE-BM25): 伪文档(%d词)='%s%s'",
                len(hypothetical.split()),
                hypothetical[:300].replace("\n", " "),
                "..." if len(hypothetical) > 300 else "",
            )
            top3_hyde = sorted(hyde_raw, key=lambda r: r["score"], reverse=True)[:3]
            for i, r in enumerate(top3_hyde, 1):
                logger.info(
                    "  HyDE命中#%d score=%.3f paper=%s title=%s",
                    i, r["score"], r["paper_id"][:8], r.get("title", "")[:60],
                )
            if self._run_logger is not None:
                self._run_logger.log_retrieval_event(
                    "HYDE_BM25",
                    pseudo_doc_words=len(hypothetical.split()),
                    pseudo_doc_preview=hypothetical[:300].replace("\n", " "),
                    hits=len(hyde_raw),
                    top3=[
                        f"score={r['score']:.3f}  title={r.get('title','')[:60]}"
                        for r in top3_hyde
                    ],
                )
        else:
            logger.warning(
                "HyDE第二路: 伪文档生成失败 → 仅使用关键词 BM25（检索质量可能下降）"
            )
            if self._run_logger is not None:
                self._run_logger.log_retrieval_event(
                    "HYDE_BM25",
                    status="FALLBACK — 伪文档生成失败，仅使用关键词 BM25",
                )

        if not kw_raw and not hyde_raw:
            logger.info("HyDERetriever: 两路检索均无结果（min_score=%.2f）", self._min_score)
            return GlobalPaperIndex()

        # 去重到 paper 级别（每篇保留最高分 chunk）
        kw_best  = _best_chunk_per_paper(kw_raw)
        hyde_best = _best_chunk_per_paper(hyde_raw)

        # 构建合并的 chunk 信息表（两路中分数更高的 chunk 优先）
        combined_best: Dict[str, dict] = {}
        for pid, chunk in {**kw_best, **hyde_best}.items():
            if pid not in combined_best or chunk["score"] > combined_best[pid]["score"]:
                combined_best[pid] = chunk

        # RRF 融合两路 paper 排名
        kw_ids   = _sort_ids_by_score(kw_best)
        hyde_ids = _sort_ids_by_score(hyde_best)
        rrf_result = reciprocal_rank_fusion([kw_ids, hyde_ids])
        top_pids = [pid for pid, _ in rrf_result[: self._top_global]]

        # 构建 GlobalPaperEntry 列表
        entries: List[GlobalPaperEntry] = []
        for rank, pid in enumerate(top_pids, start=1):
            chunk_result = combined_best[pid]
            paper_meta   = self._corpus.get_paper(pid) or {}

            raw_authors = paper_meta.get("authors", [])
            authors: List[str] = []
            for a in raw_authors:
                if isinstance(a, dict):
                    authors.append(a.get("name", ""))
                elif isinstance(a, str):
                    authors.append(a)

            abstract = paper_meta.get("abstract", "") or paper_meta.get("index_text", "")

            entries.append(GlobalPaperEntry(
                r_index         = rank,
                paper_id        = pid,
                title           = paper_meta.get("title", chunk_result.get("title", "Unknown Title")),
                abstract        = abstract,
                top_chunk_text  = chunk_result.get("text", ""),
                authors         = authors,
                doi             = paper_meta.get("doi", ""),
                retrieval_score = chunk_result["score"],
            ))

        # 第三路（辅助）：Closing section 专属查询
        # 目标：为 sec5/sec6/sec7（Limitations/Evidence gaps/Future work）补充
        # 主路未覆盖的方法论类文献，使 closing sections 有论文可引用。
        closing_extra_count = 0
        if self._top_closing_extra > 0:
            closing_raw = self._corpus.search(
                self._CLOSING_QUERY,
                top_k=self._top_closing_extra * 4,
            )
            closing_raw = [r for r in closing_raw if r["score"] >= self._min_score]
            closing_best = _best_chunk_per_paper(closing_raw)

            # 只取主路（combined_best）未覆盖的新 paper_id
            main_pids = {e.paper_id for e in entries}
            new_closing_ids = [
                pid for pid in _sort_ids_by_score(closing_best)
                if pid not in main_pids
            ][: self._top_closing_extra]

            for pid in new_closing_ids:
                chunk_result = closing_best[pid]
                paper_meta   = self._corpus.get_paper(pid) or {}

                raw_authors = paper_meta.get("authors", [])
                c_authors: List[str] = []
                for a in raw_authors:
                    if isinstance(a, dict):
                        c_authors.append(a.get("name", ""))
                    elif isinstance(a, str):
                        c_authors.append(a)

                abstract = paper_meta.get("abstract", "") or paper_meta.get("index_text", "")
                entries.append(GlobalPaperEntry(
                    r_index         = len(entries) + 1,
                    paper_id        = pid,
                    title           = paper_meta.get("title", chunk_result.get("title", "Unknown Title")),
                    abstract        = abstract,
                    top_chunk_text  = chunk_result.get("text", ""),
                    authors         = c_authors,
                    doi             = paper_meta.get("doi", ""),
                    retrieval_score = chunk_result["score"],
                ))
                closing_extra_count += 1

        index = GlobalPaperIndex(entries=entries)
        logger.info(
            "HyDERetriever.retrieve_global: %d 篇论文（kw路=%d, hyde路=%d, 融合后=%d, closing辅助=%d）",
            len(entries),
            len(kw_best),
            len(hyde_best),
            len(combined_best),
            closing_extra_count,
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
        从全局索引中为某节挑选最相关的 top_section_k 篇（Mini-HyDE 重排）。

        策略：
            1. 生成章节专属假设片段（Mini-HyDE）
            2. 对全语料库运行 BM25，过滤到 global_index 内的 paper_id
            3. 按章节相关性重新排序，返回 top_section_k 篇

        降级：
            Mini-HyDE 失败 → 退化为关键词词频重排（保留 BM25 逻辑，不再用计数近似）。
        """
        if global_index.is_empty():
            return []

        if len(global_index.entries) <= self._top_section_k:
            return list(global_index.entries)

        # 构建 paper_id → GlobalPaperEntry 映射
        entry_map: Dict[str, GlobalPaperEntry] = {e.paper_id: e for e in global_index.entries}
        paper_ids_in_index = set(entry_map.keys())

        # 第一优先：Mini-HyDE 章节假设片段
        local_goal = (section_intent.local_goal if section_intent else "") or ""
        mini_hypo = self._hyde.generate_for_section(
            section_title=section_title,
            local_goal=local_goal,
            llm_client=self._llm_client,
        )
        if mini_hypo:
            logger.info(
                "Mini-HyDE[%s]: 伪片段(%d词)='%s%s'",
                section_title[:30],
                len(mini_hypo.split()),
                mini_hypo[:150].replace("\n", " "),
                "..." if len(mini_hypo) > 150 else "",
            )
        else:
            logger.warning(
                "Mini-HyDE[%s]: 伪片段生成失败 → 关键词词频回退排序",
                section_title[:40],
            )

        ranked_ids = self._rank_by_query(
            query=mini_hypo,
            paper_ids_in_index=paper_ids_in_index,
            fallback_entries=global_index.entries,
            section_title=section_title,
            local_goal=local_goal,
            task=task,
        )

        result = [entry_map[pid] for pid in ranked_ids[: self._top_section_k]]
        logger.info(
            "HyDERetriever.rank_for_section: section=%s → %d/%d 篇",
            section_title[:40],
            len(result),
            len(global_index.entries),
        )
        if self._run_logger is not None:
            self._run_logger.log_retrieval_event(
                "MINI_HYDE",
                section=section_title[:60],
                pseudo_fragment=mini_hypo[:200].replace("\n", " ") if mini_hypo else "FALLBACK — 生成失败",
                selected=len(result),
                total_in_index=len(global_index.entries),
                top_papers=[
                    f"[R{e.r_index}] score={e.retrieval_score:.3f}  title={e.title[:55]}"
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

        接入方式（待 MemoryModule 实现时）：
            1. MemoryModule 在每节生成后调用 memory_corpus.add_document(section_id, content)
            2. Orchestrator 在生成前调用 retriever.retrieve_from_memory(intent, memory_corpus)
            3. 返回的历史段落注入 section prompt，辅助话语一致性

        参数：
            intent:        当前章节意图（SectionIntent.local_goal）
            memory_corpus: 历史生成内容的 CorpusLoader（支持动态追加的变体）

        返回：
            List[dict]，每项包含 section_id / text / score

        Raises:
            NotImplementedError: 接口预留，待 MemoryModule 接入后实现
        """
        raise NotImplementedError(
            "retrieve_from_memory 接口预留，待 MemoryModule 接入后实现。\n"
            "接入步骤见方法 docstring。"
        )

    # ── 内部兼容方法（旧 ReferenceRetriever 接口，保留供过渡期使用）─────────

    def retrieve(self, query: str, section_id: str) -> ReferenceBundle:
        """旧接口兼容层，新代码请用 retrieve_global + rank_for_section。"""
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
                    paper_id        = chunk["paper_id"],
                    title           = chunk["title"],
                    chunk_id        = chunk["chunk_id"],
                    text            = chunk["text"],
                    rank            = rank,
                    retrieval_score = chunk["score"],
                ))
                rank += 1

        return ReferenceBundle(section_id=section_id, query=query, items=items)

    # ── Query 构建（向后兼容，保留旧静态方法签名）────────────────────────────

    @staticmethod
    def build_query(
        task: str,
        section_title: str,
        section_intent: Optional["SectionIntent"] = None,
    ) -> str:
        """向后兼容的 query 构建方法（旧代码路径使用）。"""
        parts: List[str] = []
        topic = HyDERetriever._extract_topic_terms_joined(task)
        if topic:
            parts.append(topic)
        section = HyDERetriever._extract_section_terms_joined(section_title)
        if section:
            parts.append(section)
        if section_intent is not None and section_intent.local_goal:
            goal = HyDERetriever._extract_section_terms_joined(section_intent.local_goal)
            if goal:
                parts.append(goal)
        return " ".join(parts)

    # ── 私有辅助方法 ──────────────────────────────────────────────────────────

    def _rank_by_query(
        self,
        query: Optional[str],
        paper_ids_in_index: set,
        fallback_entries: List[GlobalPaperEntry],
        section_title: str,
        local_goal: str,
        task: str,
    ) -> List[str]:
        """
        对给定 query 做 BM25，过滤到 global_index 内的论文，返回排好序的 paper_id 列表。
        query 为 None 时退化为关键词词频重排。
        """
        if query:
            raw = self._corpus.search(query, top_k=len(paper_ids_in_index) * 4)
            best_in_index: Dict[str, float] = {}
            for r in raw:
                pid = r["paper_id"]
                if pid in paper_ids_in_index:
                    if pid not in best_in_index or r["score"] > best_in_index[pid]:
                        best_in_index[pid] = r["score"]

            if best_in_index:
                ranked = sorted(best_in_index.items(), key=lambda kv: kv[1], reverse=True)
                ranked_ids = [pid for pid, _ in ranked]
                # 将未命中的 paper_id 按全局顺序追加（保证返回完整列表）
                seen = set(ranked_ids)
                for e in fallback_entries:
                    if e.paper_id not in seen:
                        ranked_ids.append(e.paper_id)
                return ranked_ids

        # query 为 None 或 BM25 无命中：按章节/任务关键词词频重排（BM25 近似）
        return self._keyword_rank_fallback(fallback_entries, section_title, local_goal, task)

    def _keyword_rank_fallback(
        self,
        entries: List[GlobalPaperEntry],
        section_title: str,
        local_goal: str,
        task: str,
    ) -> List[str]:
        """
        关键词词频回退排序（Mini-HyDE 失败时使用）。
        对标题+摘要做 BM25 近似（词频加权，非计数），比旧版 token 计数近似更准确。
        """
        from .corpus import _tokenize

        query_parts: List[str] = []
        if task:
            query_parts.append(self._extract_topic_terms_joined(task))
        query_parts.append(" ".join(self._extract_section_terms(section_title)))
        if local_goal:
            query_parts.append(" ".join(self._extract_section_terms(local_goal)))

        q_tokens = _tokenize(" ".join(q for q in query_parts if q))
        if not q_tokens:
            return [e.paper_id for e in entries]

        # 词频加权：每个 token 在文档中出现一次计 1 分
        def score_entry(entry: GlobalPaperEntry) -> float:
            text = (entry.title + " " + entry.abstract).lower()
            return sum(text.count(t) for t in q_tokens)

        return [e.paper_id for e in sorted(entries, key=score_entry, reverse=True)]

    # ── 静态 Query 辅助（保留原有逻辑，供两处复用）──────────────────────────

    @staticmethod
    def _extract_topic_terms_joined(text: str) -> str:
        return " ".join(HyDERetriever._extract_topic_terms(text))

    @staticmethod
    def _extract_section_terms_joined(text: str) -> str:
        return " ".join(HyDERetriever._extract_section_terms(text))

    @staticmethod
    def _extract_topic_terms(text: str) -> List[str]:
        """
        从任务描述中提取领域名词，过滤写作指令词汇和介词。
        """
        TASK_INSTRUCTION_WORDS = {
            "write", "define", "describe", "explain", "outline", "review",
            "organize", "conduct", "compare", "summarize", "present", "introduce",
            "cover", "include", "develop", "close", "open", "begin", "ending",
            "provide", "discuss", "ensure", "keep", "build", "create", "make",
            "section", "article", "paragraph", "word", "piece", "body", "text",
            "form", "long", "scholarly", "rather", "than", "brief", "outline",
            "survey", "paper", "review", "report", "summary", "overview",
            "approximately", "about", "with", "from", "this", "that", "their",
            "these", "while", "around", "main", "line", "least", "natural",
            "within", "each", "also", "both", "more", "most", "some", "must",
            "should", "will", "have", "been", "into", "onto", "using", "style",
            "case", "popular", "science", "medicine", "english", "target",
            "length", "entire", "fully", "explicitly", "periodically",
            "approximately", "requirement", "constraint", "document",
        }
        tokens = re.findall(r'\b(?:[A-Z]{2,}|[A-Za-z][a-z]{3,})\b', text)
        return [t for t in tokens if t.lower() not in TASK_INSTRUCTION_WORDS]

    @staticmethod
    def _extract_section_terms(text: str) -> List[str]:
        """
        从章节标题/local_goal 提取词汇，保留真实域名词。
        """
        SECTION_INSTRUCTION_WORDS = {
            "write", "define", "describe", "explain", "organize", "conduct",
            "compare", "summarize", "present", "provide", "discuss", "ensure",
            "develop", "close", "open", "include", "cover",
            "section", "paragraph", "piece", "form", "around", "using",
            "with", "from", "this", "that", "their", "these", "each",
            "also", "both", "must", "should", "will", "have", "been",
            "approximately", "about", "within", "while", "main",
        }
        tokens = re.findall(r'\b(?:[A-Z]{2,}|[A-Za-z][a-z]{3,})\b', text)
        return [t for t in tokens if t.lower() not in SECTION_INSTRUCTION_WORDS]


# ── 向后兼容别名 ───────────────────────────────────────────────────────────────

#: ReferenceRetriever 是 HyDERetriever 的别名，保持旧代码路径不变。
ReferenceRetriever = HyDERetriever


# ── 私有辅助函数 ───────────────────────────────────────────────────────────────

def _best_chunk_per_paper(raw_results: List[dict]) -> Dict[str, dict]:
    """
    将 chunk 级别的 BM25 结果去重到 paper 级别，保留每篇最高分 chunk。

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
    return [
        pid for pid in sorted(
            best_per_paper,
            key=lambda p: best_per_paper[p]["score"],
            reverse=True,
        )
    ]
