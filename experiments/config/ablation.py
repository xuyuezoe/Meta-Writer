"""
消融预设注册表（实验层）

功能：
    在 src.core.ablation.AblationConfig 核心类型之上，附加实验层的预设注册表
    （PRESETS）与按名构造接口（from_preset），并提供非法名称的显式报错。

设计动机（第一性原理）：
    - 核心数据类型（AblationConfig）属于 src，供 Orchestrator 直接消费，
      不应承载"实验有哪些命名变体"这类实验词汇。
    - 预设集合（a0~a7）是论文中明确定义的有限集合，属于实验层词汇，
      故在本模块固定下来。非法名称必须显式抛错（禁止静默回退到 Full）。

依赖方向：experiments → src（正确方向）。
"""
from __future__ import annotations

from typing import Dict

from src.core.ablation import AblationConfig

__all__ = [
    "ABLATION_PRESETS",
    "AblationConfig",
    "AblationConfigError",
    "from_preset",
]


class AblationConfigError(ValueError):
    """
    消融配置非法异常

    设计目的：
        把"未知的消融预设名称"显式抛出，而非静默回退到 Full 配置，
        避免实验在错误配置下悄悄跑完。
    """


# ----------------------------------------------------------------------
# 预设注册表：论文中明确定义的消融变体集合
# ----------------------------------------------------------------------
#
# 设计说明：
#     每个预设对应 ablation_study_plan.md §5.2 矩阵中的一行。
#     "full" 是完整系统上界；"a0"~"a7"（去除已失效的 a5）是逐机制关闭的变体。
#     单一职责：每个消融预设只关闭一个机制，保证归因清晰。
ABLATION_PRESETS: Dict[str, AblationConfig] = {
    "full": AblationConfig.full(),
    "a0_no_correction": AblationConfig(no_correction=True),
    "a1_no_dsl": AblationConfig(no_dsl=True),
    "a2_no_mrsd": AblationConfig(no_mrsd=True),
    "a3_no_metastate": AblationConfig(no_metastate=True),
    "a4_no_planner": AblationConfig(no_planner=True),
    "a6_no_dsl_relations": AblationConfig(no_dsl_relations=True),
    "a7_no_memory_purge": AblationConfig(no_memory_purge=True),
}


def from_preset(preset_name: str) -> AblationConfig:
    """
    根据预设名称构造消融配置

    功能：
        从 ABLATION_PRESETS 注册表查表构造，名称非法时显式抛错。

    参数：
        preset_name: 预设名称（如 "full"、"a1_no_dsl"），大小写不敏感。

    返回值：
        AblationConfig：对应的消融配置实例

    异常：
        AblationConfigError：当 preset_name 不在注册表中时抛出，附带全部合法名称。
    """
    normalized = preset_name.strip().lower()
    if normalized not in ABLATION_PRESETS:
        valid_names = ", ".join(sorted(ABLATION_PRESETS.keys()))
        raise AblationConfigError(
            f"[消融配置非法] 未知的消融预设名称: '{preset_name}'。"
            f"合法名称为: {valid_names}"
        )
    return ABLATION_PRESETS[normalized]
