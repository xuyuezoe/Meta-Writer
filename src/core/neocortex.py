"""
新皮层知识数据结构：NeocortexItem

功能：
    定义新皮层知识的数据结构，用于存储和管理知识项。

依赖：无
被依赖：未来可能由其他模块使用
"""

from dataclasses import dataclass
from typing import Dict, Any
import uuid


@dataclass
class NeocortexItem:
    """
    新皮层知识项 - 存储单个知识条目

    功能：
        存储知识内容及其元数据，支持序列化和反序列化。

    字段：
        item_id: 唯一标识符
        content: 知识内容
        source_id: 来源节点ID
        activation_level: 激活水平 [0.0, 1.0]
        memory_state: 记忆状态（active/dormant/archived）
        is_static: 是否为静态知识
        created_section: 创建该知识的章节ID
    """

    item_id: str
    content: str
    source_id: str
    activation_level: float = 0.6
    memory_state: str = "active"
    is_static: bool = False
    created_section: str = ""

    @classmethod
    def create(
        cls,
        content: str,
        source_id: str,
        activation_level: float = 0.6,
        memory_state: str = "active",
        is_static: bool = False,
        created_section: str = "",
    ) -> "NeocortexItem":
        """工厂方法：创建新知识项"""
        return cls(
            item_id=str(uuid.uuid4()),
            content=content,
            source_id=source_id,
            activation_level=activation_level,
            memory_state=memory_state,
            is_static=is_static,
            created_section=created_section,
        )

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "item_id": self.item_id,
            "content": self.content,
            "source_id": self.source_id,
            "activation_level": round(self.activation_level, 3),
            "memory_state": self.memory_state,
            "is_static": self.is_static,
            "created_section": self.created_section,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NeocortexItem":
        """从字典反序列化"""
        return cls(
            item_id=data.get("item_id", data.get("id", str(uuid.uuid4()))),
            content=data.get("content", ""),
            source_id=data.get("source_id", ""),
            activation_level=data.get("activation_level", 0.6),
            memory_state=data.get("memory_state", "active"),
            is_static=data.get("is_static", False),
            created_section=data.get("created_section", ""),
        )