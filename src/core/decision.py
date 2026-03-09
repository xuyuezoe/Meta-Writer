from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import time
import uuid
"""
    功能：
    定义Decision数据类，记录每次生成的决策信息
    核心内容：
    class Decision:
        - decision: 决策描述（做什么）
        - reasoning: 推理过程（为什么）
        - referenced_sections: 引用关系（依赖什么）
        - target_section: 生成目标（生成什么）
        - confidence: 置信度（有多确定）
        
    关键方法：
        - to_dict() / from_dict(): 序列化
        - get_dependency_edges(): 获取依赖边
        - get_reference_count(): 获取引用数
    作用：
    DTG的核心节点类型，记录"为什么这样生成"
"""

@dataclass
class Decision:
    """
    决策记录 - DTG的核心节点

    记录每次生成的决策信息，包括：
    - 决策内容（做什么）
    - 推理过程（为什么）
    - 依赖关系（引用了什么）
    - 元信息（置信度、时间戳等）
    """

    # 基础信息
    timestamp: int
    decision_id: str

    # 决策内容
    decision: str              # 决策描述
    reasoning: str             # 推理过程
    expected_effect: str       # 预期效果
    confidence: float          # 置信度 [0, 1]

    # 依赖关系（DTG的关键）
    referenced_sections: List[Tuple[str, str]]  # [(section_id, snippet)]
    target_section: str        # 生成的目标section

    # 元数据
    phase: str = "expanding"

    def __post_init__(self):
        """验证数据有效性"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence必须在[0,1]范围内，当前值：{self.confidence}")
        if not self.decision_id:
            self.decision_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = int(time.time())

    def to_dict(self) -> Dict:
        """序列化为字典（用于JSON保存）"""
        return {
            "timestamp": self.timestamp,
            "decision_id": self.decision_id,
            "decision": self.decision,
            "reasoning": self.reasoning,
            "expected_effect": self.expected_effect,
            "confidence": self.confidence,
            "referenced_sections": [list(ref) for ref in self.referenced_sections],
            "target_section": self.target_section,
            "phase": self.phase,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Decision":
        """从字典反序列化"""
        return cls(
            timestamp=data["timestamp"],
            decision_id=data["decision_id"],
            decision=data["decision"],
            reasoning=data["reasoning"],
            expected_effect=data["expected_effect"],
            confidence=data["confidence"],
            referenced_sections=[tuple(ref) for ref in data["referenced_sections"]],
            target_section=data["target_section"],
            phase=data.get("phase", "expanding"),
        )

    def get_dependency_edges(self) -> List[Tuple[str, str]]:
        """
        获取依赖边（用于构建DTG）
        返回：[(referenced_section_id, target_section_id), ...]
        """
        return [(section_id, self.target_section) for section_id, _ in self.referenced_sections]

    def get_reference_count(self) -> int:
        """获取引用数（用于错误定位算法）"""
        return len(self.referenced_sections)
