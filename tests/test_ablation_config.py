"""
AblationConfig 单元测试

验证目标：
    1. 默认配置等价完整系统（is_full 为 True，所有开关 False）
    2. 每个预设只关闭对应的一个机制（单一职责）
    3. method_label 稳定且可区分 full 与各消融
    4. 非法预设名显式抛错（禁止兜底回退）
    5. as_dict 完整序列化全部字段（provenance 完整性）
    6. frozen 不可变性
"""
from __future__ import annotations

import dataclasses

import pytest

from experiments.config.ablation import (
    ABLATION_PRESETS,
    AblationConfig,
    AblationConfigError,
    from_preset,
)


def test_default_is_full() -> None:
    """默认构造应等价完整系统"""
    config = AblationConfig()
    assert config.is_full() is True
    assert config.method_label == "full"
    assert all(value is False for value in config.as_dict().values())


def test_full_factory_matches_default() -> None:
    """full() 工厂方法应与默认构造一致"""
    assert AblationConfig.full() == AblationConfig()


def test_each_preset_disables_exactly_one_mechanism() -> None:
    """除 full 外，每个预设应且仅应关闭一个机制"""
    for name, config in ABLATION_PRESETS.items():
        enabled_switches = [k for k, v in config.as_dict().items() if v]
        if name == "full":
            assert enabled_switches == []
        else:
            assert len(enabled_switches) == 1, (
                f"预设 {name} 应只关闭一个机制，实际关闭了 {enabled_switches}"
            )


def test_from_preset_known_names() -> None:
    """已知预设名应正确查表"""
    assert from_preset("full").is_full()
    assert from_preset("a1_no_dsl").no_dsl is True
    assert from_preset("a7_no_memory_purge").no_memory_purge is True


def test_from_preset_is_case_insensitive_and_trims() -> None:
    """预设名应大小写不敏感并去除首尾空白"""
    assert from_preset("  A1_NO_DSL ").no_dsl is True


def test_from_preset_unknown_raises() -> None:
    """未知预设名必须显式抛错，且不静默回退"""
    with pytest.raises(AblationConfigError) as exc_info:
        from_preset("a5_no_hyde")  # 已删除的消融
    # 错误信息应列出合法名称，便于排查
    assert "a1_no_dsl" in str(exc_info.value)


def test_method_label_for_single_ablation() -> None:
    """单机制消融的标签应为该字段名"""
    assert AblationConfig(no_dsl=True).method_label == "no_dsl"
    assert AblationConfig(no_mrsd=True).method_label == "no_mrsd"


def test_method_label_for_combined_ablation_is_ordered() -> None:
    """组合消融的标签应按字段声明顺序拼接，保证确定性"""
    config = AblationConfig(no_mrsd=True, no_dsl=True)
    # no_dsl 在 no_mrsd 之前声明，故应排在前
    assert config.method_label == "no_dsl+no_mrsd"


def test_as_dict_contains_all_seven_switches() -> None:
    """as_dict 应包含全部七个开关字段（A5 已删除，不应出现 no_retrieval）"""
    keys = set(AblationConfig().as_dict().keys())
    expected = {
        "no_correction",
        "no_dsl",
        "no_mrsd",
        "no_metastate",
        "no_planner",
        "no_dsl_relations",
        "no_memory_purge",
    }
    assert keys == expected
    assert "no_retrieval" not in keys


def test_config_is_frozen() -> None:
    """配置应不可变（frozen）"""
    config = AblationConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.no_dsl = True  # type: ignore[misc]
