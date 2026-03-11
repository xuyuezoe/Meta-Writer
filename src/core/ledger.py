"""
话语状态账本核心数据类：CommitmentType、ConstraintType、LedgerEntry、EntryRelation

功能：
    定义 DiscourseLedger 的基础数据结构。
    LedgerEntry 是带完整生命周期管理的账本条目，不是简单的约束字符串。

说明：
    ommitmentType 问的是"这个承诺是什么性质的东西"                                                                                                                                                                                                                     
                                                            
    fact         → 已发生、已确立的事实，判断"这个设定在叙事上是否应该被视为不可变"（"主角在火星"）
    commitment   → 对后文的明确预告（"他决定明天去找Sarah"）
    open_loop    → 悬而未决的线索（"神秘信号的来源"）
    hypothesis   → 角色的主观猜测（"他怀疑是蓄意破坏"）
    style_policy → 全局风格约定（"第三人称限知视角"）

    ConstraintType 问的是"这个承诺对后续生成有多强的约束力"

    immutable → 用户明确给定，不可被后续生成覆盖
    stateful  → 可以随情节推进合法更新（状态会变化）
    soft      → 尽量满足，但可以酌情调整

    两个维度是正交的，可以任意组合。例如"主角名叫Alex"是 fact + immutable；"主角目前在基地"是 fact + stateful（位置会变）；"故事节奏尽量紧凑"是 style_policy + soft。
    这两个维度的标签怎么打，完全依赖 LLM 的判断，而这个判断没有任何校验机制。LLM 决定一个事实是 immutable 还是 stateful，直接决定了它的初始 trust_level，进而影响它被注入的优先级、被撤销时造成的级联损失大小、MetaState 的 CRS 计算。

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
    承诺对象类型（五类，对后续节的约束力不同）

    关键实现细节：
        FACT         → 对后续不可矛盾，按 salience显著性筛选注入
        COMMITMENT   → 后文必须履行，全部注入
        OPEN_LOOP    → 越久未闭合越显著，全部注入
        HYPOTHESIS   → 不注入 prompt（角色猜测，不刚性约束）
        STYLE_POLICY → 全局有效，全部注入
    """
    FACT         = "fact"
    COMMITMENT   = "commitment"
    OPEN_LOOP    = "open_loop"
    HYPOTHESIS   = "hypothesis"
    STYLE_POLICY = "style_policy"


class ConstraintType(Enum):
    """
    约束强度类型（三级分类）

    关键实现细节：
        IMMUTABLE → 叙事上不可逆的设定，任何生成决策不可与之矛盾
        STATEFUL  → 可随情节推进合法演变的状态，更新须前后连贯
        SOFT      → 风格偏好或非核心细节，不作为强回退依据
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
            self.commitment_type == CommitmentType.OPEN_LOOP and self.is_resolved
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
