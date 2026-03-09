from dataclasses import dataclass, field
from typing import List, Dict

"""
    功能：
    管理当前生成过程的状态信息
    核心内容：
    class GenerationState:
        - current_section: 当前生成哪个section
        - progress: 进度（0.0-1.0）
        - global_constraints: 全局约束列表
        - outline: 大纲（section_id → 描述）
        - generated_sections: 已生成的section列表
        - flagged_issues: 标记的问题
        
    关键方法：
        - to_prompt(): 转换为自然语言描述
        （这是显式状态注入的核心！）
        - update_progress(): 更新进度
    作用：
    1. 维护生成上下文
    2. 支持显式状态注入（Explicit State Management）
    3. 让LLM知道"当前在哪里，已经做了什么"
"""
@dataclass
class GenerationState:
    """
    生成状态 - 简化版

    存储当前生成的状态信息，用于显式状态注入
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

    # 已生成内容摘要（用于一致性检查，section_id -> 前300字片段）
    section_snippets: Dict[str, str] = field(default_factory=dict)

    def to_prompt(self) -> str:
        """
        转换为自然语言状态描述（显式状态注入的核心）

        返回格式化的状态描述，用于注入到prompt中
        """
        lines = []

        lines.append(f"## 当前生成状态")
        lines.append(f"- 当前章节：{self.current_section}")
        lines.append(f"- 整体进度：{self.progress:.0%}（已完成 {len(self.generated_sections)}/{len(self.outline)} 节）")

        if self.global_constraints:
            lines.append("\n## 全局约束")
            for c in self.global_constraints:
                lines.append(f"- {c}")

        if self.pending_goals:
            lines.append("\n## 待完成目标")
            for g in self.pending_goals:
                lines.append(f"- {g}")

        if self.generated_sections:
            lines.append("\n## 已生成章节")
            lines.append("、".join(self.generated_sections))

        if self.flagged_issues:
            lines.append("\n## 已标记问题（需注意）")
            for issue in self.flagged_issues:
                lines.append(f"- {issue}")

        return "\n".join(lines)

    def update_progress(self):
        """更新进度"""
        if self.outline:
            self.progress = len(self.generated_sections) / len(self.outline)
