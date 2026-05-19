"""
话语状态账本核心数据类：CommitmentType、ConstraintType、LedgerEntry、EntryRelation

功能：
    定义 DiscourseLedger 的基础数据结构。
    LedgerEntry 是带完整生命周期管理的账本条目，不是简单的约束字符串。

说明：
    CommitmentType 问的是"这个话语承诺是什么性质的东西"

    ESTABLISHED_CLAIM  → 已确立的学术事实或命题，后续不可矛盾
    FORWARD_COMMITMENT → 对后续节的明确预告（"下节将讨论..."）
    UNRESOLVED_ISSUE   → 尚未解决的研究问题或争议点
    TENTATIVE_CLAIM    → 推测性主张，应保持暂定态度
    DISCOURSE_POLICY   → 全局话语规约（写作风格、引用格式等）

    ConstraintType 问的是"这个承诺对后续生成有多强的约束力"

    IMMUTABLE → 任务规范给定的硬约束，生成过程不可覆盖
    STATEFUL  → 可随生成推进合法更新的动态状态
    SOFT      → 风格偏好或非核心约束，不作为强回退依据

    两个维度正交可任意组合。

依赖：无
被依赖：DiscourseLedger、CommitmentExtractor、OnlineValidator
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class CommitmentType(Enum):
    """
    话语承诺类型（五类，对后续节的约束力不同）

    关键实现细节：
        ESTABLISHED_CLAIM  → 已确立命题，按 salience 显著性筛选注入
        FORWARD_COMMITMENT → 后文必须履行的预告，全部注入
        UNRESOLVED_ISSUE   → 越久未解决越显著，全部注入
        TENTATIVE_CLAIM    → 推测性主张，不注入刚性约束
        DISCOURSE_POLICY   → 全局话语规约，全部注入
    """
    ESTABLISHED_CLAIM  = "established_claim"
    FORWARD_COMMITMENT = "forward_commitment"
    UNRESOLVED_ISSUE   = "unresolved_issue"
    TENTATIVE_CLAIM    = "tentative_claim"
    DISCOURSE_POLICY   = "discourse_policy"


class ConstraintType(Enum):
    """
    约束强度类型（三级分类）

    关键实现细节：
        IMMUTABLE → 任务规范给定的硬约束，生成过程不可覆盖
        STATEFUL  → 可随生成推进合法更新的动态状态
        SOFT      → 风格偏好或非核心约束，不作为强回退依据
    """
    IMMUTABLE = "immutable"
    STATEFUL  = "stateful"
    SOFT      = "soft"


@dataclass
class LedgerEntry:
    """
    账本条目 - 带完整生命周期管理的承诺对象

    功能：
        存储从生成内容中提取并经过承诺判定的话语状态。
        不是简单的事实列表，而是带可信度、关系支持、修订历史的状态节点。

    参数：
        entry_id: 唯一标识（UUID）
        commitment_type: 承诺对象类型（CommitmentType 枚举）
        content: 约束内容描述（自然语言）
        constraint_type: 约束强度（ConstraintType 枚举）
        source_section: 引入此条目的节 ID
        source_decision_id: 引入时对应的决策 ID
        introduced_at: 引入时间戳
        support_span: 支持此条目的节 ID 列表（随后续节更新）
        stability_score: 经过多少节未被冲突 [0.0, 1.0]
        trust_level: 综合可信度（stability + 关系支持度）
        is_resolved: OPEN_LOOP 是否已被闭合
        resolved_in: 闭合此线索的节 ID
        revoked_by: 若已被撤销，记录撤销节 ID
        revision_history: 历次修订记录

    关键实现细节：
        salience_score 不在此处存储，由 DiscourseLedger.compute_salience() 运行时计算
        trust_level 初始由承诺判定的 constraint_strength 决定，后续随 stability 更新
    """
    entry_id: str
    commitment_type: CommitmentType
    content: str
    constraint_type: ConstraintType
    source_section: str
    source_decision_id: str
    introduced_at: int

    support_span: List[str] = field(default_factory=list)
    stability_score: float = 1.0
    trust_level: float = 1.0

    is_resolved: bool = False
    resolved_in: Optional[str] = None
    revoked_by: Optional[str] = None
    revision_history: List[Dict] = field(default_factory=list)

    # Phase 3：DSL 溯源字段
    # source_node: 在 DTG 中生成此条目的节点 ID（decision_id 或 intent_node_id）
    # used_by_sections: 记录哪些节在生成时注入了此条目（溯源 D_j(r) 传播强度）
    source_node: str = ""
    used_by_sections: List[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        commitment_type: CommitmentType,
        content: str,
        constraint_type: ConstraintType,
        source_section: str,
        source_decision_id: str,
        trust_level: float = 1.0,
    ) -> "LedgerEntry":
        """工厂方法：创建新账本条目"""
        return cls(
            entry_id=str(uuid.uuid4()),
            commitment_type=commitment_type,
            content=content,
            constraint_type=constraint_type,
            source_section=source_section,
            source_decision_id=source_decision_id,
            introduced_at=int(time.time()),
            trust_level=trust_level,
        )

    def revoke(self, revoked_by_section: str) -> None:
        """撤销此条目，记录撤销节"""
        self.revoked_by = revoked_by_section
        self.revision_history.append({
            "action": "revoked",
            "by_section": revoked_by_section,
            "at": int(time.time()),
        })

    def update_stability(self, passed_sections: int) -> None:
        """根据经过的无冲突节数更新稳定性得分"""
        #这里的问题：passed_sections 只是时间流逝，并不追踪这些节里有没有人主动引用或验证过这个条目。所以理论上来说，一个条目存了 5 节、无人问津，和一个条目被 5 节持续引用且通过验证，在这个公式里得分完全一样。这是一个设计上的简化，用时间代替了真正的"被验证次数"。
        self.stability_score = min(1.0, 0.5 + passed_sections * 0.1)
        #随着节的推进，实证稳定性逐渐主导，初始的约束类型权重衰减。设计意图是：时间证明比先验判断更重要
        self.trust_level = (self.stability_score * 0.7 + self.trust_level * 0.3)

    def is_active(self) -> bool:
        """是否仍为活跃状态（未被撤销且未过期解决）"""
        return self.revoked_by is None and not (
            self.commitment_type == CommitmentType.UNRESOLVED_ISSUE and self.is_resolved
        )

    def to_dict(self) -> Dict:
        """序列化为字典"""
        return {
            "entry_id":          self.entry_id,
            "commitment_type":   self.commitment_type.value,
            "content":           self.content,
            "constraint_type":   self.constraint_type.value,
            "source_section":    self.source_section,
            "source_decision_id": self.source_decision_id,
            "introduced_at":     self.introduced_at,
            "support_span":      self.support_span,
            "stability_score":   round(self.stability_score, 3),
            "trust_level":       round(self.trust_level, 3),
            "is_resolved":       self.is_resolved,
            "resolved_in":       self.resolved_in,
            "revoked_by":        self.revoked_by,
            "source_node":       self.source_node,
            "used_by_sections":  self.used_by_sections,
        }


@dataclass
class EntryRelation:
    """
    账本条目间的结构关系

    功能：
        构成 DSL 的关系层，使账本从列表升级为图结构。
        三类关系涌现出级联撤销、冲突传播检测、OPEN_LOOP 自动闭合等能力。

    参数：
        source_id: 来源条目 ID
        target_id: 目标条目 ID
        relation_type: 关系类型（supports / conflicts / resolves）
        confidence: 关系置信度 [0, 1]

    关键实现细节：
        supports(A, B)：A 为 B 提供依据；A 被 revoke 时 B 降低 trust_level
        conflicts(A, B)：A 与 B 互斥；写入新条目时检测冲突
        resolves(A, B)：A 闭合了 OPEN_LOOP B；自动标记 B.is_resolved = True
    """
    source_id: str
    target_id: str
    relation_type: str   # "supports" | "conflicts" | "resolves"
    confidence: float

    def to_dict(self) -> Dict:
        """序列化为字典"""
        return {
            "source_id":     self.source_id,
            "target_id":     self.target_id,
            "relation_type": self.relation_type,
            "confidence":    round(self.confidence, 3),
        }
