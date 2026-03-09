from typing import List, Dict, Tuple, Optional
import logging
import time

from ..memory.dtg_store import DTGStore
from ..core.decision import Decision


class DTGDebugger:
    """
    DTG调试器
    基于DTG进行错误定位和约束分析

    核心算法：
    1. locate_error_source：基于启发式评分定位错误源
    2. analyze_constraint_coverage：分析约束覆盖情况

    关键设计：
    1. locate_error_source是启发式算法（不保证100%准确）
    2. 基于三个启发式指标：低信心、孤立、时间近
    3. analyze_constraint_coverage是辅助功能
    """

    # 评分权重
    _W_CONFIDENCE = 0.4
    _W_REFERENCES = 0.3
    _W_TIME       = 0.3

    def __init__(self, dtg_store: DTGStore):
        self.dtg = dtg_store
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # 错误定位
    # ------------------------------------------------------------------

    def locate_error_source(
        self,
        problematic_section: str,
        top_k: int = 3,
    ) -> List[Tuple[str, float, str]]:
        """
        定位错误来源（核心算法）

        算法：基于三个因素的启发式评分
        1. 置信度低 → 嫌疑高（权重40%）：score += (1 - confidence) * 0.4
        2. 引用少   → 嫌疑高（权重30%）：score += 1/(ref_count+1) * 0.3
        3. 时间距离近→嫌疑高（权重30%）：score += (1 - normalized_dist) * 0.3

        返回：[(decision_id, suspicion_score, reason), ...] 按嫌疑度降序，取top-k=3
        """
        chain = self.dtg.trace_decision_chain(problematic_section)
        if not chain:
            self.logger.warning("locate_error_source: section '%s' 无决策链", problematic_section)
            return []

        # 以最新决策时间戳为基准
        problem_timestamp = chain[-1].timestamp
        timestamps = [d.timestamp for d in chain]
        max_time_distance = max(abs(problem_timestamp - t) for t in timestamps) or 1

        scored: List[Tuple[str, float, str]] = []
        for decision in chain:
            # 因素1：置信度低 → 嫌疑高
            conf_score = (1.0 - decision.confidence) * self._W_CONFIDENCE

            # 因素2：引用少 → 嫌疑高（孤立决策更可疑）
            ref_score = (1.0 / (decision.get_reference_count() + 1)) * self._W_REFERENCES

            # 因素3：时间距离近 → 嫌疑高
            time_dist = abs(decision.timestamp - problem_timestamp)
            normalized_dist = time_dist / max_time_distance
            time_score = (1.0 - normalized_dist) * self._W_TIME

            total = conf_score + ref_score + time_score

            reason = (
                f"置信度={decision.confidence:.2f}(+{conf_score:.3f}) "
                f"引用数={decision.get_reference_count()}(+{ref_score:.3f}) "
                f"时间距离={time_dist}s(+{time_score:.3f})"
            )
            scored.append((decision.decision_id, round(total, 4), reason))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # 约束覆盖分析
    # ------------------------------------------------------------------

    def analyze_constraint_coverage(
        self,
        constraints: List[str],
    ) -> Dict[str, Dict]:
        """
        分析约束覆盖情况

        返回：
        {
            constraint: {
                'satisfied': bool,
                'related_decisions': [decision_ids],
                'coverage_score': float   # 0.0 ~ 1.0
            }
        }

        覆盖分数 = 关键词命中率，>= 0.5 视为已满足
        """
        result: Dict[str, Dict] = {}

        for constraint in constraints:
            related = self._find_decisions_mentioning(constraint)
            keywords = self._extract_keywords(constraint)

            if not keywords:
                coverage_score = 0.0
            else:
                # 统计在所有相关决策文本中命中的关键词比例
                all_text = " ".join(
                    f"{d.decision} {d.reasoning} {d.expected_effect}"
                    for d in related
                ).lower()
                hit = sum(1 for kw in keywords if kw in all_text)
                coverage_score = round(hit / len(keywords), 3)

            result[constraint] = {
                "satisfied": coverage_score >= 0.5,
                "related_decisions": [d.decision_id for d in related],
                "coverage_score": coverage_score,
            }

            self.logger.debug(
                "约束覆盖: '%s' score=%.3f satisfied=%s",
                constraint[:40],
                coverage_score,
                result[constraint]["satisfied"],
            )

        return result

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _find_decisions_mentioning(self, constraint: str) -> List[Decision]:
        """
        找到提到某约束的决策

        方法：对 decision、reasoning、expected_effect 字段进行关键词匹配
        """
        keywords = self._extract_keywords(constraint)
        if not keywords:
            return []

        matched = []
        for decision in self.dtg.decision_log:
            text = f"{decision.decision} {decision.reasoning} {decision.expected_effect}".lower()
            if any(kw in text for kw in keywords):
                matched.append(decision)
        return matched

    def _extract_keywords(self, constraint: str) -> List[str]:
        """
        从约束中提取关键词

        策略：去除常见停用词，保留长度>=2的词（中英文均适用）
        """
        stopwords = {
            "的", "了", "是", "在", "和", "与", "或", "不", "要", "需",
            "请", "使用", "保持", "确保", "必须", "应该", "应",
            "a", "an", "the", "is", "are", "of", "to", "in", "and", "or",
        }
        # 简单分词：按空格、标点拆分
        import re
        tokens = re.split(r"[\s，。！？,.!?\-/]+", constraint.lower())
        return [t for t in tokens if len(t) >= 2 and t not in stopwords]
