"""
生成状态数据类：GenerationState

功能：
    管理当前生成过程的状态信息，支持显式状态注入（Explicit State Management）。
    包含全局约束、DSL 注入上下文、节内容摘要等，作为 Generator prompt 的核心输入。

依赖：无
被依赖：Generator（读取 to_prompt()）、SectionPlanner（读取 section_summaries）、
        OnlineValidator（读取 section_snippets）、Orchestrator（维护状态）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class GenerationState:
    """
    生成状态

    功能：
        维护生成上下文，通过 to_prompt() 实现显式状态注入。
        让 LLM 知道"当前在哪里、已经做了什么、有哪些约束"。

    参数：
        current_section: 当前生成的节 ID
        progress: 整体进度 [0.0, 1.0]
        global_constraints: 全局约束列表（IMMUTABLE 类型，始终注入）
        pending_goals: 待完成目标列表
        outline: 全局大纲 {section_id: section_title}
        generated_sections: 已成功生成的节 ID 列表（按顺序）
        flagged_issues: 已标记的问题（上下文警告）
        section_snippets: 已生成内容的短片段 {section_id: 前300字}（用于一致性检查）
        section_summaries: 已生成内容的摘要 {section_id: 摘要}（用于 SectionPlanner）
        dsl_injection: DSL 上下文注入文本（当 strengthen_dsl_injection=True 时使用）

    关键实现细节：
        section_snippets 用于一致性检查（OnlineValidator）。
        section_summaries 用于 SectionPlanner 生成 SectionIntent 的上下文。
        dsl_injection 由 DiscourseLedger 计算后由 Orchestrator 写入，
        在 strengthen_dsl_injection=True 时注入 prompt。
    """

    # 当前状态
    current_section: str
    progress: float  # 0.0-1.0

    # 约束与目标
    global_constraints: List[str] = field(default_factory=list)
    pending_goals: List[str] = field(default_factory=list)

    # 内容结构
    outline: Dict[str, str] = field(default_factory=dict)
    generated_sections: List[str] = field(default_factory=list)

    # 问题标记
    flagged_issues: List[str] = field(default_factory=list)

    # 已生成内容（短片段，用于一致性检查）
    section_snippets: Dict[str, str] = field(default_factory=dict)

    # 已生成内容摘要（较长，用于 SectionPlanner）
    section_summaries: Dict[str, str] = field(default_factory=dict)

    # DSL 注入上下文（strengthen_dsl_injection=True 时由 Orchestrator 写入）
    dsl_injection: str = ""

    def to_prompt(self) -> str:
        """
        将当前状态转换为自然语言描述（显式状态注入的核心）

        功能：
            格式化当前状态，注入 global_constraints、pending_goals、
            generated_sections、flagged_issues 和 DSL 上下文（若有）。

        返回值：
            str：格式化的状态描述字符串
        """
        lines = [
            "## Current Generation State",
            f"- Current section: {self.current_section}",
            f"- Overall progress: {self.progress:.0%} "
            f"({len(self.generated_sections)}/{len(self.outline)} sections completed)",
        ]

        if self.global_constraints:
            lines.append("\n## Global Constraints")
            for c in self.global_constraints:
                lines.append(f"- {c}")

        if self.dsl_injection:
            lines.append("\n## Discourse State (Must Be Respected)")
            lines.append(self.dsl_injection)

        if self.pending_goals:
            lines.append("\n## Pending Goals")
            for g in self.pending_goals:
                lines.append(f"- {g}")

        if self.generated_sections:
            lines.append("\n## Completed Sections")
            lines.append(", ".join(self.generated_sections))

        if self.flagged_issues:
            lines.append("\n## Flagged Issues")
            for issue in self.flagged_issues:
                lines.append(f"- {issue}")

        return "\n".join(lines)

    def update_progress(self) -> None:
        """
        根据已生成节数更新整体进度

        关键实现细节：
            使用 outline 总节数作为分母；outline 为空时进度不更新。
        """
        if self.outline:
            self.progress = len(self.generated_sections) / len(self.outline)
