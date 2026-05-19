"""
主题覆盖评分器：TopicCoverageScorer（TCAS）

功能：
    替代 LLM-as-judge DCAS，实现零 LLM 调用的三维可计算评分。
    三个维度均使用本地 embedding + 规则计算，与生成 LLM 完全解耦。

三维评分：
    TopicCoverage（权重 0.5）：max-cosine pooling，阈值 θ=0.65
    ConstraintSatisfaction（权重 0.3）：直接传入约束通过比率
    LengthAdherence（权重 0.2）：词数比值映射到 [0, 1]

学术形式化：
    TCAS = 0.5·TC + 0.3·CS + 0.2·LA

依赖：EmbeddingModel（来自 src/utils/embedding_model.py）
被依赖：OnlineValidator
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from ..core.decision import Decision
    from ..utils.embedding_model import EmbeddingModel


def _split_sentences(text: str) -> List[str]:
    """按句子边界切分文本，过滤空句"""
    parts = re.split(r"[.!?。！？]+", text)
    return [p.strip() for p in parts if p.strip()]


class TopicCoverageScorer:
    """
    TCAS：主题覆盖评分器

    参数：
        embedding_model: EmbeddingModel 实例（全局单例）

    关键实现细节：
        TopicCoverage：对每个 required_topic，计算与内容各句的最大 cosine 相似度；
                       ≥ THETA_TOPIC_COVERAGE 认定已覆盖；取所有 topic 覆盖率的均值。
        ConstraintSatisfaction：直接接受外部传入的通过比率（0~1）。
        LengthAdherence：词数比值 clip 后线性映射。
        TCAS < THRESHOLD_TCAS_MAJOR 触发 MAJOR（对标原 DCAS 阈值 0.6，保守下调 0.05）。
    """

    THETA_TOPIC_COVERAGE: float = 0.40    # 与 online_validator.THETA_COVERAGE 对齐
    THRESHOLD_TCAS_MAJOR: float = 0.55

    # 词数比例阈值（相对于 word_target）
    _WORD_FLOOR: float = 0.72
    _WORD_CEIL: float = 1.30

    def __init__(self, embedding_model: "EmbeddingModel"):
        self._embed = embedding_model

    def compute_tcas(
        self,
        decision: "Decision",
        content: str,
        constraint_pass_ratio: float,
        word_target: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        计算 TCAS 总分及三个子维度分数

        参数：
            decision:              当前节的决策对象（提供 required_topics）
            content:               当前节生成内容
            constraint_pass_ratio: 约束通过比率 [0, 1]（由验证器外部计算）
            word_target:           本节目标词数（可选）

        返回：
            {
                "topic_coverage":          float,  # 主题覆盖率 [0, 1]
                "constraint_satisfaction": float,  # 约束满足率 [0, 1]
                "length_adherence":        float,  # 词数符合度 [0, 1]
                "tcas":                    float,  # TCAS 总分 [0, 1]
                "covered_topics":          List[str],  # 已覆盖的 topic
                "missing_topics":          List[str],  # 未覆盖的 topic
            }
        """
        # 第一阶段：主题覆盖率
        topic_coverage, covered_topics, missing_topics = self._compute_topic_coverage(
            required_topics=decision.required_topics,
            content=content,
        )

        # 第二阶段：约束满足率（外部传入，直接使用）
        constraint_satisfaction = float(np.clip(constraint_pass_ratio, 0.0, 1.0))

        # 第三阶段：词数符合度
        length_adherence = self._compute_length_adherence(content, word_target)

        # 第四阶段：TCAS 加权求和
        tcas = (
            0.5 * topic_coverage
            + 0.3 * constraint_satisfaction
            + 0.2 * length_adherence
        )

        return {
            "topic_coverage":          round(topic_coverage, 4),
            "constraint_satisfaction": round(constraint_satisfaction, 4),
            "length_adherence":        round(length_adherence, 4),
            "tcas":                    round(float(tcas), 4),
            "covered_topics":          covered_topics,
            "missing_topics":          missing_topics,
        }

    def _compute_topic_coverage(
        self,
        required_topics: List[str],
        content: str,
    ) -> tuple:
        """
        计算主题覆盖率

        算法：
            1. 将内容按句子边界切分为句子列表
            2. 批量编码所有句子（避免重复编码）
            3. 对每个 required_topic，计算其 embedding 与所有句子 embedding 的最大 cosine 相似度
            4. ≥ THETA_TOPIC_COVERAGE 认为已覆盖

        参数：
            required_topics: 必须覆盖的主题短语列表
            content:         生成内容

        返回：
            (coverage_ratio, covered_topics, missing_topics)
        """
        if not required_topics:
            return 1.0, [], []

        sentences = _split_sentences(content)
        if not sentences:
            return 0.0, [], list(required_topics)

        # 批量编码句子（共享计算）
        sent_embeds = self._embed.embed(sentences)  # (S, D)

        covered: List[str] = []
        missing: List[str] = []

        # 批量编码所有 topic
        topic_embeds = self._embed.embed(required_topics)  # (T, D)

        for i, topic in enumerate(required_topics):
            # max cosine: (S, D) @ (D,) → (S,)
            sims = sent_embeds @ topic_embeds[i]
            max_sim = float(sims.max()) if len(sims) > 0 else 0.0
            if max_sim >= self.THETA_TOPIC_COVERAGE:
                covered.append(topic)
            else:
                missing.append(topic)

        coverage_ratio = len(covered) / len(required_topics)
        return coverage_ratio, covered, missing

    def _compute_length_adherence(
        self,
        content: str,
        word_target: Optional[int],
    ) -> float:
        """
        计算词数符合度，将词数比值映射到 [0, 1]

        参数：
            content:     生成内容
            word_target: 目标词数（为 None 时返回 1.0）

        返回：
            float: 词数符合度 [0, 1]
        """
        if word_target is None or word_target <= 0:
            return 1.0

        word_count = len(content.split())
        ratio = word_count / word_target

        floor = self._WORD_FLOOR
        ceil_ = self._WORD_CEIL

        if ratio < floor:
            # 低于下限：线性衰减，最低 0
            return max(0.0, ratio / floor)
        elif ratio > ceil_:
            # 超过上限：线性衰减，最低 0
            excess = ratio - ceil_
            return max(0.0, 1.0 - excess)
        else:
            return 1.0
