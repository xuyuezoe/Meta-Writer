"""
LightRAG 相似度服务封装。
提供统一 compute_similarity(text1, text2) 接口给 SupportSubsetBuilder 使用。
"""

import asyncio
import threading

import numpy as np


class LightRAGSimilarityService:
    def __init__(self, rag):
        self.rag = rag

    def _run_awaitable_sync(self, awaitable):
        """
        在同步代码中执行 async coroutine。
        如果当前没有 running event loop，直接 asyncio.run。
        如果当前已有 running event loop，则开一个新线程运行独立 event loop。
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(awaitable)

        result_box = {}
        error_box = {}

        def runner():
            try:
                result_box["result"] = asyncio.run(awaitable)
            except Exception as e:
                error_box["error"] = e

        thread = threading.Thread(target=runner)
        thread.start()
        thread.join()

        if "error" in error_box:
            raise error_box["error"]
        return result_box.get("result")

    def _call_embedding(self, text: str):
        raw = self.rag.embedding_func(text)
        if hasattr(raw, "__await__"):
            raw = self._run_awaitable_sync(raw)
        return raw

    def _normalize_embedding(self, emb):
        """
        将 embedding 转成 list[float]。
        """
        if hasattr(emb, "__await__"):
            raise ValueError("embedding_func returned coroutine; please use sync embedding")

        if isinstance(emb, dict) and "data" in emb:
            return emb["data"][0]["embedding"]

        if isinstance(emb, np.ndarray):
            if emb.ndim == 2 and emb.shape[0] == 1:
                return emb[0].tolist()
            return emb.tolist()

        if isinstance(emb, list):
            if emb and isinstance(emb[0], list):
                return emb[0]
            return emb

        normalized = list(emb)
        if normalized and isinstance(normalized[0], list):
            return normalized[0]
        return normalized

    def is_duplicate(self, text1: str, text2: str, threshold: float = 0.95) -> bool:
        """
        判断两段文本是否语义重复。
        """
        return self.compute_similarity(text1, text2) >= threshold

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        使用 LightRAG embedding 计算相似度（cosine similarity）。
        """
        # 防御空文本
        if not text1.strip() or not text2.strip():
            return 0.0

        # 调用 embedding。逐个调用、逐个执行，避免遗留未 await 的 coroutine。
        emb1 = self._normalize_embedding(self._call_embedding(text1))
        emb2 = self._normalize_embedding(self._call_embedding(text2))

        # 计算 cosine similarity
        v1 = np.array(emb1)
        v2 = np.array(emb2)

        denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
        if denom == 0:
            return 0.0

        return float(np.dot(v1, v2) / denom)
