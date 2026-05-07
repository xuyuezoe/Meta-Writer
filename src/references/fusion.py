"""
Reciprocal Rank Fusion（RRF）实现。

原理：
    将多路独立排名列表融合为一个统一排名，使用排名而非分数融合，
    规避了不同检索路径之间分数量纲不可比的问题。

公式：
    score(d) = Σᵢ  1 / (k + rankᵢ(d))

    （k=60足够大以弱化顶部绝对优势，但不至于让尾部论文乘虚而入。）

适用场景：
    - 关键词 BM25 + HyDE BM25 双路融合
    - 未来扩展为三路（加入 embedding 检索）时直接支持
"""
from __future__ import annotations

from typing import Dict, List, Tuple


def reciprocal_rank_fusion(
    ranked_lists: List[List[str]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    合并多个已排序的 paper_id 列表，返回融合后的排序结果。

    参数：
        ranked_lists: 每个元素是一路 BM25 检索结果的 paper_id 列表，
                      列表内部已按相关性降序排列（index 0 = 最相关）。
                      空列表或 None 列表会被安全跳过。
        k:            RRF 平滑参数，默认 60。

    返回：
        List[Tuple[paper_id, rrf_score]]，按 rrf_score 降序排列。
        paper_id 不重复（已去重）。

    注意：
        - 不出现在某路结果中的文档不会因"缺席"被惩罚（直接跳过）。
        - 只出现在一路中的文档仍可进入最终排名（分数 = 1/(k+rank)）。
    """
    scores: Dict[str, float] = {}

    for ranked in ranked_lists:
        if not ranked:
            continue
        for rank_zero_based, paper_id in enumerate(ranked):
            rank_one_based = rank_zero_based + 1
            scores[paper_id] = scores.get(paper_id, 0.0) + 1.0 / (k + rank_one_based)

    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
