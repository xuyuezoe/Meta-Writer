"""
实验配置层（L0）

功能：
    定义实验所需的全部纯数据结构，无任何副作用。
    这些结构是 runners（执行层）与 analysis（分析层）共同消费的契约。

导出：
    AblationConfig        — 消融开关冻结配置 + PRESETS 注册表
    AblationConfigError   — 非法消融配置异常
    BackboneConfig        — 骨干模型配置（model/base_url/api_key 来源）
    BackboneRegistryError — 非法骨干模型异常
    RunContext            — 单次运行的四元组上下文（命名 + provenance）
"""
from __future__ import annotations

from .ablation import (
    ABLATION_PRESETS,
    AblationConfig,
    AblationConfigError,
    from_preset,
)
from .backbone import (
    BACKBONE_TEMPLATES,
    BackboneConfig,
    BackboneRegistryError,
)
from .run_context import RunContext

__all__ = [
    "ABLATION_PRESETS",
    "AblationConfig",
    "AblationConfigError",
    "from_preset",
    "BACKBONE_TEMPLATES",
    "BackboneConfig",
    "BackboneRegistryError",
    "RunContext",
]
