from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Union
from enum import Enum

if TYPE_CHECKING:
    from ..references.types import SectionReferenceReport

"""
验证数据结构：PresenceViolation、AbsenceViolation、ValidationReport

作用：
    封装验证结果，生成检测报告，传递诊断信息。
    零 LLM 验证路径产出的失败对象由 MRCA 直接消费，无需 LLM 解析。

依赖：无
被依赖：OnlineValidator（生成）、Orchestrator（读取）、MRSD（消费 failures）
"""


class IssueSeverity(Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


@dataclass
class Issue:
    """单个格式/约束问题的描述（用于 warnings 产出）"""
    type: str          # "format", "constraint"
    severity: str      # IssueSeverity 的值
    description: str
    location: str = ""

    def __str__(self) -> str:
        loc = f" @ {self.location}" if self.location else ""
        return f"[{self.severity.upper()}][{self.type}]{loc}: {self.description}"


@dataclass
class PresenceViolation:
    """
    写了不该写的内容（DSL 合规检查触发）

    字段来源：
        τ                 ← 由被违反的 DSL entry 类型静态决定
        violating_chunks  ← NLI 检测时直接截取的内容窗口
        violated_dsl_entry ← 被违反的 DSL 条目文本
        source_check      ← 格式 "dsl_compliance:{entry_id}" 或 "constraint_check"

    MRCA 消费方式：
        violating_chunks 直接用于 embedding 检索，无需 LLM 提取。
    """
    τ: str                          # "CONSISTENCY" | "SCOPE" | "CONTENT_GAP"
    violating_chunks: List[str]     # 违规内容窗口
    violated_dsl_entry: str         # 被违反的 DSL 条目文本
    source_check: str               # 触发此失败的检查 ID


@dataclass
class AbsenceViolation:
    """
    应写未写的内容（覆盖率检查触发）

    字段来源：
        τ          ← 由检查类型静态决定
        obligation ← 从 required_topics 或 DSL entry 直接读取
        source_check ← 格式 "coverage:{topic_id}" 或 "tcas"

    MRCA 消费方式：
        obligation 直接用于规划层审计，无需 LLM 提取。
    """
    τ: str                  # "CONTENT_GAP" | "METHODOLOGY"
    obligation: str         # 缺失义务文本
    source_check: str       # 触发此失败的检查 ID


@dataclass
class ValidationReport:
    """
    验证报告

    包含验证结果、blocking failure 列表和 warnings
    """
    section_id: str
    passed: bool
    score: float                    # TCAS 总分（替代原 dcas_score）
    failures: List[Union[PresenceViolation, AbsenceViolation]]
    warnings: List[str]             # 非 blocking 问题（格式/词数等）
    reference_report: Optional["SectionReferenceReport"] = None

    def __str__(self) -> str:
        """格式化输出（用于日志）"""
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"ValidationReport [{status}] TCAS={self.score:.3f} section={self.section_id}"]

        if self.failures:
            lines.append(f"  blocking failures（{len(self.failures)}条）：")
            for f in self.failures:
                lines.append(f"    [{f.τ}] {failure_description(f)[:80]}")

        if self.warnings:
            lines.append(f"  warnings（{len(self.warnings)}条）：")
            for w in self.warnings[:3]:
                lines.append(f"    - {w[:80]}")

        return "\n".join(lines)

    def has_critical(self) -> bool:
        """是否存在 CONSISTENCY 或 SCOPE 类型的严重 failure"""
        return any(f.τ in ("CONSISTENCY", "SCOPE") for f in self.failures)


def failure_description(f: Union[PresenceViolation, AbsenceViolation]) -> str:
    """
    从 FailureObject 提取可读描述字符串（供日志和 Orchestrator 使用）

    参数：
        f: PresenceViolation 或 AbsenceViolation 实例

    返回：
        str: 简洁描述
    """
    if isinstance(f, PresenceViolation):
        return f"Violated DSL: {f.violated_dsl_entry[:60]}"
    if isinstance(f, AbsenceViolation):
        return f"Missing: {f.obligation[:60]}"
    return str(f)
