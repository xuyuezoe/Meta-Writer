from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum

"""
作用：
  封装验证结果，生成检测报告，传递诊断信息
依赖：
  Decision（suspected_source字段）
被依赖：
  OnlineValidator（生成这个报告）
  Orchestrator（使用这个报告做决策）
"""
class IssueSeverity(Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


@dataclass
class Issue:
    """单个问题的描述"""
    type: str          # "format", "constraint", "alignment", "consistency"
    severity: str      # IssueSeverity的值
    description: str
    location: str = ""

    def __str__(self) -> str:
        loc = f" @ {self.location}" if self.location else ""
        return f"[{self.severity.upper()}][{self.type}]{loc}: {self.description}"


@dataclass
class ValidationReport:
    """
    验证报告

    包含验证结果、问题列表和修正建议
    """
    passed: bool
    issues: List[Issue]
    violated_constraints: List[str] = field(default_factory=list)
    dcas_score: float = 1.0
    suspected_source: Optional[object] = None   # Decision，避免循环导入用object
    suggested_strategy: str = ""
    strategy_params: Dict = field(default_factory=dict)

    def __str__(self) -> str:
        """格式化输出（用于日志）"""
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"ValidationReport [{status}] DCAS={self.dcas_score:.3f}"]

        if self.issues:
            lines.append(f"  问题列表（{len(self.issues)}条）：")
            for issue in self.issues:
                lines.append(f"    {issue}")

        if self.violated_constraints:
            lines.append(f"  违反约束：")
            for c in self.violated_constraints:
                lines.append(f"    - {c}")

        if self.suspected_source:
            src_id = getattr(self.suspected_source, "decision_id", str(self.suspected_source))
            lines.append(f"  疑似根源决策：{src_id}")

        if self.suggested_strategy:
            lines.append(f"  建议策略：{self.suggested_strategy}")
            if self.strategy_params:
                lines.append(f"  策略参数：{self.strategy_params}")

        return "\n".join(lines)

    def critical_issues(self) -> List[Issue]:
        """返回所有CRITICAL级别问题"""
        return [i for i in self.issues if i.severity == IssueSeverity.CRITICAL.value]

    def has_critical(self) -> bool:
        """是否存在严重问题"""
        return bool(self.critical_issues())
