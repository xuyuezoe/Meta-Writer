"""
CorpusLoader — 加载本地 Markdown 论文语料库，提供 dense embedding 检索。

检索原理：
    每个 chunk 以「abstract_text + chunk_text」拼接后由 BAAI/bge-m3 编码；
    向量 L2-normalization 后，查询时对 query 做同样编码，
    用矩阵内积（= cosine similarity）一次性计算全库相似度并取 top-k。

优势（对比 BM25）：
    语义泛化，无词汇鸿沟；一路检索，无 RRF 融合开销；零外部依赖（纯 numpy）。

依赖：EmbeddingModel（src/utils/embedding_model.py）
被依赖：retriever.py（HyDERetriever）
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from .loaders.markdown_loader import load_corpus_dir

if TYPE_CHECKING:
    from ..utils.embedding_model import EmbeddingModel

logger = logging.getLogger(__name__)


class CorpusLoader:
    """
    论文语料库加载器（dense embedding 检索）

    参数：
        corpus_dir:  Markdown 论文目录路径
        embed_model: EmbeddingModel 实例（全局单例，用于批量编码 + 查询编码）

    公开接口：
        load()           → 解析语料，批量计算并存储 chunk 向量（幂等）
        search(q, top_k) → 返回 top-k 语义最近邻 chunk 结果
        list_papers()    → 返回所有论文元数据列表
        get_paper(pid)   → 按 paper_id 获取论文元数据
        get_chunk(cid)   → 按 chunk_id 获取 chunk
        paper_count()    → 已加载论文数
        chunk_count()    → 已加载 chunk 数
    """

    def __init__(self, corpus_dir: str, embed_model: "EmbeddingModel") -> None:
        self._corpus_dir = corpus_dir
        self._embed = embed_model
        self._papers: Dict[str, dict] = {}           # paper_id → paper dict
        self._chunks: Dict[str, dict] = {}            # chunk_id → chunk dict
        self._embeddings: Optional[np.ndarray] = None # (N, D) L2-normalized
        self._index_entries: List[Tuple[str, str]] = []  # [(paper_id, chunk_id)]
        self._loaded = False

    # ── 公开 API ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        解析所有 Markdown 论文文件，批量计算 dense embeddings。幂等：重复调用无效。

        编码策略：
            每个 chunk 的编码文本 = abstract_text + " " + chunk_text。
            摘要信息为 chunk 提供全局领域上下文，改善跨 chunk 的语义一致性。
        """
        if self._loaded:
            return

        papers = load_corpus_dir(self._corpus_dir)
        if not papers:
            logger.warning("CorpusLoader: %s 目录下未找到论文", self._corpus_dir)
            self._loaded = True
            return

        texts_to_embed: List[str] = []

        for paper in papers:
            pid = paper["paper_id"]
            self._papers[pid] = paper
            abstract_text = paper.get("index_text", "") or paper.get("abstract", "")

            for chunk in paper.get("chunks", []):
                cid = chunk["chunk_id"]
                self._chunks[cid] = {**chunk, "paper_id": pid, "title": paper["title"]}

                # 编码单元：摘要 + chunk 文本（摘要提供领域语义背景）
                entry_text = abstract_text + " " + chunk.get("text", "")
                texts_to_embed.append(entry_text.strip())
                self._index_entries.append((pid, cid))

        if texts_to_embed:
            # 批量编码，EmbeddingModel 内部已做 L2-normalize
            self._embeddings = self._embed.embed(texts_to_embed)  # (N, D)

        self._loaded = True
        logger.info(
            "CorpusLoader: 已索引 %d 篇论文 / %d 个 chunk（dense embedding）",
            len(self._papers),
            len(self._chunks),
        )

    def search(self, query: str, top_k: int = 20) -> List[dict]:
        """
        返回与 query 语义最近邻的 top_k 个 chunk 结果。

        参数：
            query:  查询字符串（自由文本，无需分词）
            top_k:  返回结果数量上限

        返回：
            List[dict]，每项包含 paper_id / title / chunk_id / section_title / text / score

        算法：
            1. 对 query 做 L2-normalized embedding → q_emb (1, D)
            2. self._embeddings @ q_emb[0] → (N,) cosine similarity（因向量已归一化）
            3. argsort 取 top_k，score 为 cosine 相似度值
        """
        if not self._loaded:
            self.load()
        if self._embeddings is None or not self._index_entries:
            return []
        if not query.strip():
            return []

        # 第一阶段：编码查询
        q_emb = self._embed.embed([query])  # (1, D)

        # 第二阶段：批量计算 cosine similarity
        sims: np.ndarray = self._embeddings @ q_emb[0]  # (N,)

        # 第三阶段：取 top_k
        k = min(top_k, len(sims))
        top_indices = np.argpartition(sims, -k)[-k:]
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

        results: List[dict] = []
        for idx in top_indices:
            pid, cid = self._index_entries[int(idx)]
            chunk = self._chunks.get(cid, {})
            paper = self._papers.get(pid, {})
            results.append({
                "paper_id":     pid,
                "title":        paper.get("title", ""),
                "chunk_id":     cid,
                "section_title": chunk.get("section_title", ""),
                "text":         chunk.get("text", ""),
                "score":        float(sims[idx]),
            })
        return results

    # ── Lookup 辅助 ───────────────────────────────────────────────────────────

    def list_papers(self) -> List[dict]:
        """返回所有已加载论文的元数据列表（不含 chunks）。"""
        if not self._loaded:
            self.load()
        return [
            {
                "paper_id":      p["paper_id"],
                "title":         p["title"],
                "abstract":      p.get("abstract", "")[:300],
                "subject_areas": p.get("subject_areas", []),
            }
            for p in self._papers.values()
        ]

    def get_paper(self, paper_id: str) -> Optional[dict]:
        """按 paper_id 返回论文元数据（含 chunks）。"""
        if not self._loaded:
            self.load()
        return self._papers.get(paper_id)

    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        """按 chunk_id 返回 chunk 数据。"""
        if not self._loaded:
            self.load()
        return self._chunks.get(chunk_id)

    def paper_count(self) -> int:
        if not self._loaded:
            self.load()
        return len(self._papers)

    def chunk_count(self) -> int:
        if not self._loaded:
            self.load()
        return len(self._chunks)
