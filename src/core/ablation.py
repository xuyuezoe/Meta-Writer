"""
消融配置核心类型：AblationConfig

功能：
    以单个冻结（不可变）数据类显式承载所有消融开关，供 Orchestrator 及其
    子组件在运行时按开关分支行为。

设计动机（第一性原理）：
    消融实验的本质是"在保持其余一切不变的前提下，关闭某一个机制，观测系统
    性能变化"。因此消融开关具备三个内在要求：
        1. 显式且不可变：一次运行的配置一旦确定，运行中不应被悄悄修改（frozen）。
        2. 可序列化：每次运行必须把"关闭了什么"完整写入产物（as_dict）。
        3. 默认零变化：默认全部 False，等价完整系统，保证未启用消融时
           系统行为与改造前逐字节一致。

依赖方向说明：
    本类型置于 src/core 而非 experiments 包，是为了让 Orchestrator（src 内）
    引用它时无需反向依赖实验脚手架。实验层（experiments/config/ablation.py）
    在此基础上附加预设注册表（PRESETS）与命名词汇，依赖方向为 experiments → src。

字段与 ablation_study_plan.md 的对应：
    no_correction      → A0  移除全部在线修正（验证失败直接接受，不诊断不修复）
    no_dsl             → A1  移除话语状态账本（不提取承诺、注入恒空）
    no_mrsd            → A2  移除诊断（验证失败固定 local_rewrite，不归因）
    no_metastate       → A3  移除元认知门控（gate_action 恒放行）
    no_planner         → A4  移除分节规划（退化为最简 SectionIntent）
    no_dsl_relations   → A6  移除 DSL 关系层（跳过 process_pending_relations）
    no_memory_purge    → A7  移除三层修复中的 memory_purge（强制降级 partial_rollback）

    注意：原方案 A5（No HyDE）已失效——HyDE 双路检索已从代码移除，当前为单路
    dense embedding，无对应可关闭挂钩点，故本配置不含该字段。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Dict


@dataclass(frozen=True)
class AblationConfig:
    """
    消融开关冻结配置

    功能：
        以一个不可变对象表达"本次运行关闭了哪些机制"。
        默认全部 False，即等价完整系统（Full）。

    参数：
        no_correction:     A0 移除全部在线修正机制（验证失败直接接受降级内容）
        no_dsl:            A1 移除话语状态账本（不提取承诺、DSL 注入恒为空）
        no_mrsd:           A2 移除 MRSD 诊断（验证失败固定走 local_rewrite）
        no_metastate:      A3 移除元认知门控（MetaState.gate_action 恒返回 True）
        no_planner:        A4 移除分节意图规划（使用最简 SectionIntent）
        no_dsl_relations:  A6 移除 DSL 关系层（跳过 process_pending_relations）
        no_memory_purge:   A7 移除三层修复中的 memory_purge（强制降级 partial_rollback）

    关键实现细节：
        - frozen=True 保证实例创建后不可变，杜绝运行期被意外修改。
        - is_full() 用于在 Orchestrator 中短路所有消融分支，确保 Full 路径与
          改造前完全一致。
    """

    no_correction: bool = False
    no_dsl: bool = False
    no_mrsd: bool = False
    no_metastate: bool = False
    no_planner: bool = False
    no_dsl_relations: bool = False
    no_memory_purge: bool = False

    @classmethod
    def full(cls) -> "AblationConfig":
        """
        构造完整系统配置（所有机制启用）

        返回值：
            AblationConfig：全部开关为 False 的配置实例
        """
        return cls()

    def is_full(self) -> bool:
        """
        判断是否为完整系统（无任何消融）

        返回值：
            bool：所有开关均为 False 时返回 True
        """
        return not any(getattr(self, f.name) for f in fields(self))

    @property
    def method_label(self) -> str:
        """
        生成稳定的方法标识符（用于运行命名与结果聚合分组）

        返回值：
            str：完整系统返回 "full"；否则返回所有被关闭机制按声明顺序的拼接，
                 如 "no_dsl"、"no_dsl+no_mrsd"。

        关键实现细节：
            字段顺序固定（按 dataclass 声明顺序），保证同一配置始终产生同一标签，
            使聚合分组具备确定性。
        """
        if self.is_full():
            return "full"
        active = [f.name for f in fields(self) if getattr(self, f.name)]
        return "+".join(active)

    def as_dict(self) -> Dict[str, bool]:
        """
        序列化为字典（写入运行 provenance）

        返回值：
            Dict[str, bool]：字段名到开关值的完整映射
        """
        return asdict(self)
