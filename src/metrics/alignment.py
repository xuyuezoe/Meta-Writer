"""
评估决策与内容的对齐程度（DCAS），用于在线验证。

依赖：LLMClient、Decision
被依赖：OnlineValidator
关键设计：LLM-as-judge，三个子维度独立计算，失败降级为0.5
"""
from typing import Dict
import logging
import re

from ..core.decision import Decision


# DCAS 加权系数
_W_COVERAGE      = 0.4
_W_CONSISTENCY   = 0.3
_W_EFFECTIVENESS = 0.3

# LLM评分失败时的降级默认值
_FALLBACK_SCORE = 0.5


class AlignmentScorer:
    """
    Decision-Content Alignment Score (DCAS) 评分器

    评估决策与内容的对齐度：
    - Coverage（0.4）：意图覆盖度 — decision的意图是否在content中实现
    - Consistency（0.3）：逻辑一致性 — reasoning是否与content一致
    - Effectiveness（0.3）：效果达成度 — expected_effect是否达到

    公式：DCAS = 0.4·Coverage + 0.3·Consistency + 0.3·Effectiveness
    """

    def __init__(self, llm_client):
        self.llm = llm_client
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def compute_dcas(self, decision: Decision, content: str) -> Dict[str, float]:
        """
        计算DCAS
      
        DCAS = 0.4 * Coverage + 0.3 * Consistency + 0.3 * Effectiveness
      
        返回：
            {
            'coverage': float,      # 意图覆盖度
            'consistency': float,   # 逻辑一致性
            'effectiveness': float, # 效果达成度
            'dcas': float          # 加权总分
            }

        任一子维度LLM调用失败时，该维度降级为0.5，不中断整体评分。
        """
        coverage      = self._compute_coverage(decision, content)
        consistency   = self._compute_consistency(decision, content)
        effectiveness = self._compute_effectiveness(decision, content)

        dcas = (
            coverage      * _W_COVERAGE +
            consistency   * _W_CONSISTENCY +
            effectiveness * _W_EFFECTIVENESS
        )

        result = {
            "coverage":      round(coverage,      3),
            "consistency":   round(consistency,   3),
            "effectiveness": round(effectiveness, 3),
            "dcas":          round(dcas,          3),
        }

        self.logger.info(
            "DCAS计算完成: coverage=%.3f consistency=%.3f effectiveness=%.3f → dcas=%.3f",
            coverage, consistency, effectiveness, dcas,
        )
        return result

    # ------------------------------------------------------------------
    # 子维度评分
    # ------------------------------------------------------------------

    def _compute_coverage(self, decision: Decision, content: str) -> float:
        """
        计算意图覆盖度

        方法：用LLM判断decision中的意图是否在content中实现
        Prompt:
        Decision中说要做X，Content是否做了？
        打分0.0-1.0
        评分标准：1.0=完整实现 0.8=大部分 0.6=部分 0.0=完全未实现
        """
        prompt = f"""任务：判断content是否完成了decision中的意图。

Decision: {decision.decision}
Content: {content}

输出格式：
<coverage_score>
[0.0到1.0的浮点数]
</coverage_score>

评分标准：
1.0 = 所有意图完整实现
0.8 = 大部分实现
0.6 = 部分实现
0.0 = 完全未实现"""

        return self._call_llm_for_score(prompt, "coverage_score", "意图覆盖度")

    def _compute_consistency(self, decision: Decision, content: str) -> float:
        """
        计算逻辑一致性

        方法：判断decision的reasoning中所描述的逻辑前提是否与content保持一致
        """
        prompt = f"""任务：判断content的逻辑与decision的推理过程是否一致。

Reasoning: {decision.reasoning}
Content: {content}

输出格式：
<consistency_score>
[0.0到1.0的浮点数]
</consistency_score>

评分标准：
1.0 = 逻辑完全一致，无矛盾
0.8 = 基本一致，存在小偏差
0.6 = 部分一致，有明显偏离
0.0 = 逻辑严重矛盾"""

        return self._call_llm_for_score(prompt, "consistency_score", "逻辑一致性")

    def _compute_effectiveness(self, decision: Decision, content: str) -> float:
        """
        计算效果达成度

        方法：判断content是否达到decision中声明的expected_effect
        """
        prompt = f"""任务：判断content是否达到了decision预期的效果。

Expected Effect: {decision.expected_effect}
Content: {content}

输出格式：
<effectiveness_score>
[0.0到1.0的浮点数]
</effectiveness_score>

评分标准：
1.0 = 预期效果完全达到
0.8 = 大部分达到
0.6 = 部分达到
0.0 = 完全未达到"""

        return self._call_llm_for_score(prompt, "effectiveness_score", "效果达成度")

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _call_llm_for_score(self, prompt: str, tag: str, dimension: str) -> float:
        """
        调用LLM并解析评分标签，失败时返回降级默认值

        :param prompt: 发送给LLM的完整prompt
        :param tag: 要解析的XML标签名
        :param dimension: 维度名称（仅用于日志）
        :return: [0.0, 1.0] 范围内的评分
        """
        try:
            response = self.llm.generate(prompt, temperature=0.3, max_tokens=1024)
            raw = self._extract_tag(response, tag)
            if not raw:
                raise ValueError(f"响应中缺少 <{tag}> 标签")
            score = float(raw)
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"评分越界：{score}")
            return score
        except Exception as e:
            self.logger.warning("%s 评分失败，降级为%.1f：%s", dimension, _FALLBACK_SCORE, e)
            return _FALLBACK_SCORE

    def _extract_tag(self, text: str, tag: str) -> str:
        """提取XML标签内容"""
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""
