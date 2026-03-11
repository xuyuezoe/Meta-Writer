"""
评估决策与内容的对齐程度（DCAS），用于在线验证。

依赖：LLMClient、Decision
被依赖：OnlineValidator
关键设计：LLM-as-judge，三个子维度在单次 LLM 调用中同时评分，
         输出格式为 JSON（避免 XML 示例值被模型当作最终答案）。
         任一维度解析失败时独立降级为 0.65（高于验证阈值 0.6），不阻断流程。
"""
import json
import re
from typing import Dict, Tuple
import logging

from ..core.decision import Decision


# DCAS 加权系数
_W_COVERAGE      = 0.4
_W_CONSISTENCY   = 0.3
_W_EFFECTIVENESS = 0.3

# LLM 评分失败时的降级默认值
# 设为 0.65（高于 OnlineValidator.THRESHOLD_DCAS = 0.6），确保解析失败时不触发 MAJOR，
# 不因格式解析问题阻断生成流程；警告日志仍然输出供调试。
_FALLBACK_SCORE = 0.65


class AlignmentScorer:
    """
    Decision-Content Alignment Score (DCAS) 评分器

    评估决策与内容的对齐度：
    - Coverage（0.4）：意图覆盖度 — decision 的意图是否在 content 中实现
    - Consistency（0.3）：逻辑一致性 — reasoning 是否与 content 一致
    - Effectiveness（0.3）：效果达成度 — expected_effect 是否达到

    公式：DCAS = 0.4·Coverage + 0.3·Consistency + 0.3·Effectiveness

    关键实现细节：
        三个维度合并为单次 LLM 调用，输出格式为 JSON。
        使用中文描述词作为 JSON 值的占位符（而非数字示例），避免模型将格式示例
        当作最终答案直接返回。JSON 解析失败时降级为字段名正则匹配，
        再失败时独立降级为 0.5。
    """

    def __init__(self, llm_client):
        self.llm = llm_client
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def compute_dcas(self, decision: Decision, content: str) -> Dict[str, float]:
        """
        计算 DCAS（单次 LLM 调用，三维度批量评分）

        功能：
            构建包含三个评分维度的 JSON 格式 prompt，一次调用获取所有评分。
            任一维度解析失败时独立降级为 0.5，整体评分不中断。

        参数：
            decision: 当前节的决策对象（含 decision/reasoning/expected_effect）
            content: 当前节生成的文本内容

        返回：
            {
            'coverage': float,      # 意图覆盖度 [0, 1]
            'consistency': float,   # 逻辑一致性 [0, 1]
            'effectiveness': float, # 效果达成度 [0, 1]
            'dcas': float           # 加权总分 [0, 1]
            }
        """
        coverage, consistency, effectiveness = self._compute_all_dimensions(decision, content)

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
    # 批量评分（单次 LLM 调用）
    # ------------------------------------------------------------------

    def _compute_all_dimensions(
        self,
        decision: Decision,
        content: str,
    ) -> Tuple[float, float, float]:
        """
        单次 LLM 调用同时评估三个 DCAS 子维度

        功能：
            构建 JSON 输出格式的批量评分 prompt。
            JSON 值位置使用中文描述词（如"覆盖度评分"）而非数字示例，
            确保模型理解这是需要填写实际评分的位置，而非已给出的答案。

            解析优先级：JSON 整体解析 → 字段正则匹配 → 维度独立降级

        参数：
            decision: 决策对象
            content: 生成内容

        返回值：
            Tuple[float, float, float]：(coverage, consistency, effectiveness)
        """
        prompt = (
            "你是一个叙事质量评估助手，请评估生成内容的质量。\n\n"
            f"写作决策：{decision.decision}\n"
            f"推理过程：{decision.reasoning}\n"
            f"预期效果：{decision.expected_effect}\n"
            f"实际内容：{content}\n\n"
            "请评估以下三个维度（评分范围 0.0 到 1.0）：\n"
            "1. coverage_score（意图覆盖度）：decision 的意图在 content 中实现的程度\n"
            "   1.0=完整实现，0.8=大部分，0.6=部分实现，0.0=完全未实现\n"
            "2. consistency_score（逻辑一致性）：reasoning 的逻辑与 content 是否一致\n"
            "   1.0=完全一致，0.8=基本一致，0.6=部分一致，0.0=严重矛盾\n"
            "3. effectiveness_score（效果达成度）：content 是否达到了 expected_effect\n"
            "   1.0=完全达到，0.8=大部分，0.6=部分达到，0.0=完全未达到\n\n"
            "仅输出以下 JSON，将三个中文描述词替换为你的实际评分数字，不要输出任何其他内容：\n"
            "{\"coverage_score\": 覆盖度评分, \"consistency_score\": 一致性评分, "
            "\"effectiveness_score\": 效果评分}"
        )

        try:
            # strip_think=False：MiniMax M2.5 等推理模型会将结构化评分结果放在 <think> 块内，
            # 保留原始输出（含 <think> 块）以便 _parse_json_field 从中提取评分
            response = self.llm.generate(prompt, temperature=0.3, max_tokens=512, strip_think=False)
        except Exception as e:
            self.logger.warning("DCAS 批量评分 LLM 调用失败，全部降级：%s", e)
            return _FALLBACK_SCORE, _FALLBACK_SCORE, _FALLBACK_SCORE

        coverage      = self._parse_json_field(response, "coverage_score",      "意图覆盖度")
        consistency   = self._parse_json_field(response, "consistency_score",   "逻辑一致性")
        effectiveness = self._parse_json_field(response, "effectiveness_score", "效果达成度")

        return coverage, consistency, effectiveness

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _parse_json_field(self, response: str, field: str, dimension: str) -> float:
        """
        从 LLM JSON 响应中提取指定字段的评分值

        解析优先级：
            第零阶段：从 <think> 块内提取内容，合并为搜索文本（针对 MiniMax M2.5 等推理模型）
            第一阶段：清理代码块标记后整体 JSON 解析
            第二阶段：字段名正则匹配（兼容非标准 JSON 输出）
            第三阶段：降级返回 _FALLBACK_SCORE

        关键实现细节：
            MiniMax M2.5 等推理模型常将结构化评分 JSON 置于 <think> 块内，
            可见输出仅剩确认语（如"好的，以上是我的评分"）。
            本方法在 strip_think=False 模式下接收含 <think> 块的完整响应，
            将 <think> 内容提取后与主响应文本合并进行搜索。

        参数：
            response: LLM 完整响应文本（含 <think> 块）
            field: JSON 字段名（如 "coverage_score"）
            dimension: 维度名称（仅用于日志）

        返回值：
            float：[0.0, 1.0] 范围内的评分，解析失败时返回 _FALLBACK_SCORE
        """
        try:
            # 第零阶段：提取 <think> 块内容，与原文合并作为搜索文本
            # 推理模型（MiniMax M2.5、DeepSeek-R1 等）将 JSON 评分置于 <think> 块中
            think_contents = re.findall(r"<think>(.*?)</think>", response, re.DOTALL)
            search_text = response + "\n" + "\n".join(think_contents)

            # 第一阶段：整体 JSON 解析（去除可能的 markdown 代码块标记）
            cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", search_text).strip()
            # 提取第一个 JSON 对象
            json_match = re.search(r"\{[^{}]+\}", cleaned, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                val = data.get(field)
                if val is not None:
                    score = float(val)
                    if 0.0 <= score <= 1.0:
                        return score

            # 第二阶段：字段名正则匹配
            # 匹配 "field_name": 0.8 或 "field_name":0.8 或 field_name: 0.8
            pattern = rf'"{re.escape(field)}"\s*:\s*([0-9]*\.?[0-9]+)'
            m = re.search(pattern, search_text)
            if m:
                score = float(m.group(1))
                if 0.0 <= score <= 1.0:
                    return score

            raise ValueError(f"无法从响应中提取字段 '{field}'")

        except Exception as e:
            self.logger.warning("%s 评分解析失败，降级为 %.1f：%s", dimension, _FALLBACK_SCORE, e)
            return _FALLBACK_SCORE
