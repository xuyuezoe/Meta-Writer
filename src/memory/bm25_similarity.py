"""
BM25 风格文本相似度服务。

这个实现用于替代 LightRAGSimilarityService 的相似度接口：
- 不依赖 LightRAG
- 不调用 embedding
- 不调用 OpenAI embedding
- 不新增任何外部依赖
"""

import math
import re
from collections import Counter
from typing import List


class BM25SimilarityService:
    """
    使用纯 Python 实现的简化 BM25-like 相似度服务。

    说明：
        标准 BM25 通常需要一组文档来计算 IDF。这里的调用场景是
        compute_similarity(query, text)，即只把 text 当作单个 document。
        因此本实现采用单文档 BM25-like 打分，再用“理想匹配分”做归一化，
        让结果稳定落在 0~1 区间。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def tokenize(self, text: str) -> List[str]:
        """
        简单分词：
        1. 统一转小写
        2. 英文和数字 token 原样保留
        3. 中文连续片段保留原始片段，并额外生成中文 bigram
        4. 返回 token list

        示例：
            "急性冠脉综合征" 会生成
            ["急性冠脉综合征", "急性", "性冠", "冠脉", "脉综", "综合", "合征"]
        """
        if not text:
            return []

        normalized = text.lower()
        raw_tokens = re.findall(r"[a-z]+|\d+|[\u4e00-\u9fff]+", normalized)
        tokens: List[str] = []

        for token in raw_tokens:
            tokens.append(token)

            # 中文连续片段如果长度大于 1，同时加入相邻双字 token。
            # 这能让中文短语在没有外部分词库的情况下获得更好的召回。
            if re.fullmatch(r"[\u4e00-\u9fff]+", token) and len(token) > 1:
                tokens.extend(token[index:index + 2] for index in range(len(token) - 1))

        return tokens

    def _term_score(self, term_freq: int, doc_len: int, avg_doc_len: float, idf: float) -> float:
        """
        计算单个 term 的 BM25-like 分数。

        在单文档场景下 avg_doc_len 取当前 document 长度，用于保留 BM25
        的长度归一化形式。
        """
        if term_freq <= 0 or doc_len <= 0 or avg_doc_len <= 0:
            return 0.0

        length_norm = 1.0 - self.b + self.b * (doc_len / avg_doc_len)
        denominator = term_freq + self.k1 * length_norm
        if denominator <= 0:
            return 0.0

        return idf * (term_freq * (self.k1 + 1.0)) / denominator

    def compute_similarity(self, query: str, text: str) -> float:
        """
        计算 query 与 text 的 BM25-like 相似度。

        实现方式：
        1. 将 text 视为唯一 document
        2. 用 query tokens 去匹配 document tokens
        3. 累加 BM25-like 分数
        4. 用 query 在“理想 document”中的最高可得分归一化，返回 0~1

        空文本或无有效 token 时返回 0.0。
        """
        query_tokens = self.tokenize(query)
        doc_tokens = self.tokenize(text)

        if not query_tokens or not doc_tokens:
            return 0.0

        query_counts = Counter(query_tokens)
        doc_counts = Counter(doc_tokens)

        doc_len = len(doc_tokens)
        avg_doc_len = float(doc_len)

        # 单文档场景下，出现在 document 中的 term 使用同一个平滑 IDF。
        # 公式来自 BM25 常见 IDF 形式的平滑版本，避免负数和零值。
        single_doc_idf = math.log(1.0 + (1.0 - 1.0 + 0.5) / (1.0 + 0.5))

        raw_score = 0.0
        max_score = 0.0

        for term, query_freq in query_counts.items():
            # query_freq 作为权重，保留重复词对相似度的影响。
            raw_score += query_freq * self._term_score(
                doc_counts.get(term, 0),
                doc_len,
                avg_doc_len,
                single_doc_idf,
            )

            # 理想分数：假设 document 中至少以 query 中的频次完整包含该 term。
            # 归一化后，完全相同或高度覆盖的文本会接近 1.0。
            ideal_term_freq = query_freq
            max_score += query_freq * self._term_score(
                ideal_term_freq,
                doc_len,
                avg_doc_len,
                single_doc_idf,
            )

        if max_score <= 0:
            return 0.0

        normalized_score = raw_score / max_score
        return max(0.0, min(1.0, normalized_score))

    def is_duplicate(self, text1: str, text2: str, threshold: float = 0.95) -> bool:
        """
        判断两段文本是否重复。

        这里直接复用 compute_similarity(text1, text2)，达到阈值即认为重复。
        """
        return self.compute_similarity(text1, text2) >= threshold
