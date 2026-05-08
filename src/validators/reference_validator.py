"""
ReferenceValidator（新架构）— 纯代码 [Rx] 范围验证，零 LLM 调用。

新架构变更：
    - 移除 claim_support LLM 检查（Check 4）
    - 移除 unsupported_critical_claims LLM 检查（Check 5）
    - 移除 citation_existence / chunk_locality 检查（Check 2/3）
    - 唯一验证：扫描正文中的 [Rx] 标记，检查 x 是否在 valid_r_set 中

设计原则：
    - 越界标记仅报 MINOR（记录，不阻断生成）
    - 后处理阶段会机械删除所有越界标记，无需重新生成
    - 验证速度极快（纯正则），不消耗任何 LLM token
"""
from __future__ import annotations

import logging
import re
from typing import List, Set

from ..core.validation import Issue, IssueSeverity
from ..references.types import SectionReferenceReport

logger = logging.getLogger(__name__)

_RX_MARKER_RE = re.compile(r"\[R(\d+)\]")


class ReferenceValidator:
    """
    纯代码引用标记验证器。

    参数
    ----
    llm_client:
        保留参数（向后兼容），新架构中不使用。
    min_citations:
        每节正文要求的最低合法引用数。当全局索引非空但正文合法引用数不足时，
        报 MAJOR 问题（阻断）以强制 LLM 重写并补充引用。默认 2。
    """

    def __init__(self, llm_client=None, min_citations: int = 2):
        self._min_citations = min_citations

    def validate(
        self,
        content: str,
        valid_r_set: Set[int],
        section_id: str,
        # 以下参数保留签名向后兼容，新架构中不使用
        citations=None,
        bundle=None,
    ) -> SectionReferenceReport:
        """
        验证正文中的 [Rx] 标记是否在全局合法范围内。

        流程：
            1. 正则提取所有 [Rx] 标记
            2. 分为合法（x ∈ valid_r_set）和越界（x ∉ valid_r_set）两组
            3. 越界标记记录为 MINOR issue（不阻断）

        参数
        ----
        content:       章节正文文本
        valid_r_set:   全局合法 r_index 集合（如 {1,2,3,...,12}）
        section_id:    当前节 ID（用于日志和 issue location）
        citations:     废弃（向后兼容，忽略）
        bundle:        废弃（向后兼容，忽略）

        返回
        ----
        SectionReferenceReport
        """
        issues: List[Issue] = []

        raw_markers = _RX_MARKER_RE.findall(content)
        all_r_indices = [int(m) for m in raw_markers]

        valid_found: List[int] = []
        invalid_found: List[int] = []

        for idx in all_r_indices:
            if idx in valid_r_set:
                valid_found.append(idx)
            else:
                invalid_found.append(idx)

        invalid_set = set(invalid_found)

        for idx in sorted(invalid_set):
            issues.append(Issue(
                type="reference",
                severity=IssueSeverity.MINOR.value,
                description=(
                    f"[R{idx}] 标记越界：r_index={idx} 不在全局索引范围 "
                    f"{{1..{max(valid_r_set) if valid_r_set else 0}}} 内，"
                    "后处理将自动删除此标记"
                ),
                location=section_id,
            ))

        # 引用密度检查：全局索引非空时要求至少 min_citations 个合法引用
        # 设计：valid_r_set 非空 → 确实有可用参考文献 → 零引用是 LLM 的主动放弃
        passed = True
        if valid_r_set and len(valid_found) < self._min_citations:
            issues.append(Issue(
                type="reference",
                severity=IssueSeverity.MAJOR.value,
                description=(
                    f"引用密度不足：本节仅有 {len(valid_found)} 个合法引用标记，"
                    f"要求至少 {self._min_citations} 个。"
                    "请在相关论据处主动补充 [Rx] 标记。"
                ),
                location=section_id,
            ))
            passed = False

        report = SectionReferenceReport(
            passed=passed,
            issues=issues,
            invalid_r_indices=invalid_set,
            valid_marker_count=len(valid_found),
            invalid_marker_count=len(invalid_found),
        )

        logger.info(
            "ReferenceValidator: section=%s valid_markers=%d invalid_markers=%d invalid_r=%s",
            section_id,
            len(valid_found),
            len(invalid_found),
            sorted(invalid_set) if invalid_set else "[]",
        )
        return report
