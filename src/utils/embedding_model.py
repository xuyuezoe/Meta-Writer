"""
本地 Embedding 模型封装：EmbeddingModel

功能：
    使用 sentence-transformers 加载 BAAI/bge-m3，为文本批量生成 L2 归一化 embedding。
    全局单例使用，确保所有模块（检索/验证/MRCA）共享同一 embedding 空间。

选型依据：
    BAAI/bge-m3 支持 100+ 语言，MTEB 榜单顶级，完全本地推理。
    与生成 LLM（GPT/Qwen/Claude）完全解耦，保证对比实验的控制变量唯一性。

依赖：sentence-transformers
被依赖：OnlineValidator、TopicCoverageScorer、MRSD、CorpusLoader、HyDERetriever
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    本地 Embedding 模型（BAAI/bge-m3）

    参数：
        model_name: sentence-transformers 支持的模型名称（默认 BAAI/bge-m3）

    关键实现细节：
        embed() 返回 L2 归一化向量，shape (N, D)，可直接 @ 运算得到 cosine 相似度。
        首次加载时打印进度日志（模型约 2.2 GB）。
    """

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("正在加载 embedding 模型：%s（device=%s）", model_name, device)
        self._model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        logger.info("Embedding 模型加载完成：%s（device=%s）", model_name, device)

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        批量编码文本，返回 L2 归一化向量矩阵

        参数：
            texts: 待编码文本列表（长度 N）

        返回：
            np.ndarray shape (N, D)，L2 归一化，可直接 @ 运算计算 cosine 相似度
        """
        if not texts:
            return np.zeros((0, self._model.get_sentence_embedding_dimension()), dtype=np.float32)
        return self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def similarity(self, a: str, b: str) -> float:
        """
        计算两段文本的语义相似度

        参数：
            a: 文本 A
            b: 文本 B

        返回：
            float: cosine 相似度 [0, 1]（归一化后等价于点积）
        """
        vecs = self.embed([a, b])
        return float(np.dot(vecs[0], vecs[1]))
