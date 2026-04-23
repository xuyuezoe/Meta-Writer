"""
规划层数据类：SectionIntent、PlanState

功能：
    定义可修订规划层的数据结构。
    SectionIntent 是每节生成前动态生成的局部计划，进入 DTG 追踪。
    PlanState 管理全局大纲和 Section Intent 的生命周期。

依赖：无
被依赖：SectionPlanner（生成）、Orchestrator（集成）、DTGStore（追踪）
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SectionIntent:
    """
    章节局部计划 - 每节生成前的规划产物

    功能：
        在 Task Plan + 当前 DSL + 已完成内容的基础上，为当前节制定局部计划。
        成为 Generator 的输入约束，实现"决定做什么"和"执行"的职责分离。
        作为 intent_node 进入 DTG 追踪，支持 plan_level 错误的诊断。

    参数：
        intent_id: 唯一标识（UUID）
        section_id: 对应节 ID
        local_goal: 本节需要实现的具体目标（≤50字）
        scope_boundary: 本节明确不应涉及的内容，防止 LLM 越界解决后续冲突
        open_loops_to_advance: 本节应推进的未闭合线索列表
        commitments_to_maintain: 本节必须维护的承诺列表
        risks_to_avoid: 本节需避免的高风险冲突列表
        success_criteria: 本节通过验证的最低标准（1-2条）
        source_dsl_entry_ids: 生成此 Section Intent 时引用的 DSL 条目 ID
        dsl_trust_at_generation: 生成时 DSL 的整体可信度
        created_at: 创建时间戳
        revision_count: 已被修订的次数

    关键实现细节：
        scope_boundary 是解决叙事不推进问题的关键约束字段，明确禁止本节提前解决后续冲突。
        source_dsl_entry_ids 用于 plan_level 诊断时判断是否被低 trust DSL 污染。
        trust_level_at_generation 用于排除"state_level 污染伪装成 plan_level"的情况。
    """
    intent_id: str
    section_id: str
    local_goal: str
    scope_boundary: str
    open_loops_to_advance: List[str]
    commitments_to_maintain: List[str]
    risks_to_avoid: List[str]
    success_criteria: List[str]
    source_dsl_entry_ids: List[str]
    dsl_trust_at_generation: float
    created_at: int
    revision_count: int = 0

    @classmethod
    def create(
        cls,
        section_id: str,
        local_goal: str,
        scope_boundary: str,
        open_loops_to_advance: List[str],
        commitments_to_maintain: List[str],
        risks_to_avoid: List[str],
        success_criteria: List[str],
        source_dsl_entry_ids: List[str],
        dsl_trust_at_generation: float,
    ) -> "SectionIntent":
        """工厂方法：创建新的 Section Intent"""
        return cls(
            intent_id=str(uuid.uuid4()),
            section_id=section_id,
            local_goal=local_goal,
            scope_boundary=scope_boundary,
            open_loops_to_advance=open_loops_to_advance,
            commitments_to_maintain=commitments_to_maintain,
            risks_to_avoid=risks_to_avoid,
            success_criteria=success_criteria,
            source_dsl_entry_ids=source_dsl_entry_ids,
            dsl_trust_at_generation=dsl_trust_at_generation,
            created_at=int(time.time()),
        )

    def to_prompt_text(self) -> str:
        """将 Section Intent 转换为可注入 prompt 的自然语言描述"""
        lines = [f"## Section Intent ({self.section_id})"]
        lines.append(f"**Goal**: {self.local_goal}")

        if self.scope_boundary:
            lines.append(f"\n**Do not cross this boundary**: {self.scope_boundary}")

        if self.open_loops_to_advance:
            lines.append("\n**Open loops to advance**:")
            for item in self.open_loops_to_advance:
                lines.append(f"- {item}")

        if self.commitments_to_maintain:
            lines.append("\n**Commitments to maintain**:")
            for item in self.commitments_to_maintain:
                lines.append(f"- {item}")

        if self.risks_to_avoid:
            lines.append("\n**Risks to avoid**:")
            for item in self.risks_to_avoid:
                lines.append(f"- {item}")

        if self.success_criteria:
            lines.append("\n**Minimum success criteria**:")
            for item in self.success_criteria:
                lines.append(f"- {item}")

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """序列化为字典（用于 DTG intent_node 存储）"""
        return {
            "intent_id":                self.intent_id,
            "section_id":               self.section_id,
            "local_goal":               self.local_goal,
            "scope_boundary":           self.scope_boundary,
            "open_loops_to_advance":    self.open_loops_to_advance,
            "commitments_to_maintain":  self.commitments_to_maintain,
            "risks_to_avoid":           self.risks_to_avoid,
            "success_criteria":         self.success_criteria,
            "source_dsl_entry_ids":     self.source_dsl_entry_ids,
            "dsl_trust_at_generation":  round(self.dsl_trust_at_generation, 3),
            "created_at":               self.created_at,
            "revision_count":           self.revision_count,
        }


@dataclass
class PlanState:
    """
    规划状态 - 管理全局大纲和 Section Intent 的生命周期

    功能：
        维护 Task Plan（全局大纲）和各节 Section Intent 的映射关系。
        支持 Plan Revision（当 MRSD 连续发现 plan_level 错误时触发）。
        为 plan_level 错误诊断提供可操作的诊断对象。

    参数：
        global_outline: 全局大纲 {section_id: section_title}
        section_intents: 已生成的 Section Intent {section_id: SectionIntent}
        plan_confidence: 对当前大纲结构的整体信心 [0, 1]
        revised_sections: 曾被修订的节 ID 列表
        plan_revision_history: 修订历史记录列表

    关键实现细节：
        section_intents 存储 SectionIntent 对象，intent_id 对应 DTG 中的 intent_node_id
        plan_level 错误触发条件：同一节连续 2 次低温保守重写仍失败
    """
    global_outline: Dict[str, str]
    section_intents: Dict[str, SectionIntent] = field(default_factory=dict)
    plan_confidence: float = 1.0
    revised_sections: List[str] = field(default_factory=list)
    plan_revision_history: List[Dict] = field(default_factory=list)

    def add_intent(self, intent: SectionIntent) -> None:
        """记录某节的 Section Intent"""
        self.section_intents[intent.section_id] = intent

    def get_intent(self, section_id: str) -> Optional[SectionIntent]:
        """获取某节的 Section Intent"""
        return self.section_intents.get(section_id)

    def revise_intent(
        self,
        section_id: str,
        new_intent: SectionIntent,
        reason: str,
    ) -> None:
        """
        修订某节的 Section Intent，记录修订历史

        参数：
            section_id: 被修订的节 ID
            new_intent: 新的 Section Intent
            reason: 修订原因
        """
        old_intent = self.section_intents.get(section_id)
        new_intent.revision_count = (old_intent.revision_count + 1) if old_intent else 1
        self.section_intents[section_id] = new_intent
        if section_id not in self.revised_sections:
            self.revised_sections.append(section_id)
        self.plan_revision_history.append({
            "section_id":    section_id,
            "reason":        reason,
            "old_intent_id": old_intent.intent_id if old_intent else None,
            "new_intent_id": new_intent.intent_id,
            "at":            int(time.time()),
        })

    def rollback_intents_from(self, section_id: str, section_queue: List[str]) -> None:
        """
        回退时清除指定节及其之后的 Section Intent

        参数：
            section_id: 回退起始节 ID
            section_queue: 全局节顺序列表
        """
        if section_id not in section_queue:
            return
        start_idx = section_queue.index(section_id)
        for sid in section_queue[start_idx:]:
            self.section_intents.pop(sid, None)
