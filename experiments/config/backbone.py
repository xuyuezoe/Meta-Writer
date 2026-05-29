"""
骨干模型配置：BackboneConfig

功能：
    为 EXP-III（骨干模型泛化性验证）提供统一的模型配置抽象。
    把"用哪个模型、打哪个端点、用哪个密钥"收敛为一个可解析、可序列化的对象。

设计动机（第一性原理）：
    框架泛化性实验的本质是"固定框架，替换底座模型，观测增益是否稳定"。
    因此模型配置必须满足：
        1. 与代码解耦：切换模型不应改动任何业务代码，只换配置（LLMClient
           已支持任意 model/base_url，故此处只需提供配置来源）。
        2. 密钥不入库：api_key 一律从环境变量读取，配置中只登记环境变量名，
           真实密钥由用户在 baseline_env 中填写，绝不硬编码进仓库。
        3. 可枚举：五个骨干模型是论文中固定的集合，以模版注册表固定下来。

与 ablation_study_plan.md §4.2 的对应：
    minimax / gpt-4o / claude-sonnet / deepseek-v3 / qwen2.5-72b
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional


class BackboneRegistryError(ValueError):
    """
    骨干模型配置非法异常

    设计目的：
        未知模型别名、缺失的环境变量等错误均显式抛出，
        而非静默回退到默认模型，避免实验在错误底座上跑完。
    """


@dataclass(frozen=True)
class BackboneConfig:
    """
    骨干模型配置模版

    功能：
        描述一个骨干模型的"逻辑别名 + 真实模型名 + 端点环境变量 + 密钥环境变量"，
        并提供 resolve() 把环境变量解析为 LLMClient 可直接消费的运行时参数。

    参数：
        alias:           逻辑别名（实验中稳定引用的短名，如 "gpt-4o"），
                         用于运行命名与结果聚合，不随真实模型版本变化。
        model_env:       存放真实模型名的环境变量名（如 "GPT4O_MODEL"）。
                         留空待用户在 baseline_env 中填写真实模型字符串。
        base_url_env:    存放 API 端点的环境变量名（如 "GPT4O_BASE_URL"）。
        api_key_env:     存放 API 密钥的环境变量名（如 "GPT4O_API_KEY"）。
        default_model:   当 model_env 未设置时使用的占位真实模型名（模版默认值）。
                         真实实验前应由用户通过环境变量覆盖。

    关键实现细节：
        - 一切真实值来自环境变量，本类只登记"去哪里取值"，不存储任何密钥。
        - resolve() 在 api_key 缺失时显式抛错（禁止用空密钥发起请求）。
    """

    alias: str
    model_env: str
    base_url_env: str
    api_key_env: str
    default_model: str = ""

    def resolve(self, env: Optional[Dict[str, str]] = None) -> "ResolvedBackbone":
        """
        将配置模版解析为运行时参数

        功能：
            从环境变量读取真实 model / base_url / api_key，组装成可直接
            传入 LLMClient 的解析结果。

        参数：
            env: 环境变量映射；默认使用 os.environ。注入 env 便于测试。

        返回值：
            ResolvedBackbone：含 alias / model / base_url / api_key 的运行时配置

        异常：
            BackboneRegistryError：当 api_key 为空（未配置密钥）时抛出。
        """
        source = dict(os.environ) if env is None else dict(env)

        model = source.get(self.model_env, "").strip() or self.default_model
        base_url = source.get(self.base_url_env, "").strip() or None
        api_key = source.get(self.api_key_env, "").strip()

        if not model:
            raise BackboneRegistryError(
                f"[骨干模型配置非法] 别名 '{self.alias}' 的真实模型名为空。"
                f"请在 baseline_env 中设置环境变量 {self.model_env}，"
                f"或为该模版提供 default_model。"
            )
        if not api_key:
            raise BackboneRegistryError(
                f"[骨干模型配置非法] 别名 '{self.alias}' 的 API 密钥为空。"
                f"请在 baseline_env 中设置环境变量 {self.api_key_env}。"
            )

        return ResolvedBackbone(
            alias=self.alias,
            model=model,
            base_url=base_url,
            api_key=api_key,
        )

    def as_dict(self) -> Dict[str, str]:
        """序列化为字典（写入 provenance，仅记录环境变量名，不含密钥值）"""
        return {
            "alias": self.alias,
            "model_env": self.model_env,
            "base_url_env": self.base_url_env,
            "api_key_env": self.api_key_env,
            "default_model": self.default_model,
        }


@dataclass(frozen=True)
class ResolvedBackbone:
    """
    解析后的骨干模型运行时配置

    功能：
        承载从环境变量解析出的真实参数，供 LLMClient 直接构造。
        api_key 仅在内存中流转，序列化时一律脱敏。

    参数：
        alias:    逻辑别名（用于命名与聚合）
        model:    真实模型名（传给 LLMClient.model）
        base_url: API 端点（传给 LLMClient.base_url，可为 None）
        api_key:  API 密钥（传给 LLMClient.api_key）
    """

    alias: str
    model: str
    base_url: Optional[str]
    api_key: str

    def as_provenance(self) -> Dict[str, Optional[str]]:
        """
        序列化为可落盘的 provenance（密钥脱敏）

        返回值：
            Dict：含 alias / model / base_url，以及 api_key_present 布尔标记，
                  绝不写入密钥明文。
        """
        return {
            "alias": self.alias,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_present": bool(self.api_key),
        }


# ----------------------------------------------------------------------
# 五骨干模型模版注册表（EXP-III）
# ----------------------------------------------------------------------
#
# 设计说明：
#     真实模型名、端点、密钥全部留待用户在 baseline_env 中通过环境变量填写。
#     此处只固定"逻辑别名 → 环境变量名"的映射，作为可复现的契约。
#     default_model 仅为占位提示，真实实验须由环境变量覆盖。
BACKBONE_TEMPLATES: Dict[str, BackboneConfig] = {
    # 复用项目根 .env 的 legacy 三元组（API_KEY/MODEL/BASE_URL），
    # 便于直接用现有配置跑实验（无需另填 baseline_env）。run 命名标签为 "default"。
    "default": BackboneConfig(
        alias="default",
        model_env="MODEL",
        base_url_env="BASE_URL",
        api_key_env="API_KEY",
        default_model="",
    ),
    "minimax": BackboneConfig(
        alias="minimax",
        model_env="MINIMAX_MODEL",
        base_url_env="MINIMAX_BASE_URL",
        api_key_env="MINIMAX_API_KEY",
        default_model="MiniMax-M2.5",
    ),
    "gpt-4o": BackboneConfig(
        alias="gpt-4o",
        model_env="GPT4O_MODEL",
        base_url_env="GPT4O_BASE_URL",
        api_key_env="GPT4O_API_KEY",
        default_model="gpt-4o",
    ),
    "claude-sonnet": BackboneConfig(
        alias="claude-sonnet",
        model_env="CLAUDE_SONNET_MODEL",
        base_url_env="CLAUDE_SONNET_BASE_URL",
        api_key_env="CLAUDE_SONNET_API_KEY",
        default_model="claude-sonnet-4-6",
    ),
    "deepseek-v3": BackboneConfig(
        alias="deepseek-v3",
        model_env="DEEPSEEK_V3_MODEL",
        base_url_env="DEEPSEEK_V3_BASE_URL",
        api_key_env="DEEPSEEK_V3_API_KEY",
        default_model="deepseek-v3",
    ),
    "qwen2.5-72b": BackboneConfig(
        alias="qwen2.5-72b",
        model_env="QWEN25_72B_MODEL",
        base_url_env="QWEN25_72B_BASE_URL",
        api_key_env="QWEN25_72B_API_KEY",
        default_model="qwen2.5-72b-instruct",
    ),
}


def get_backbone(alias: str) -> BackboneConfig:
    """
    按别名查表获取骨干模型模版

    参数：
        alias: 逻辑别名（如 "gpt-4o"）

    返回值：
        BackboneConfig：对应的模型配置模版

    异常：
        BackboneRegistryError：当别名不在注册表中时抛出，附带全部合法别名。
    """
    normalized = alias.strip().lower()
    if normalized not in BACKBONE_TEMPLATES:
        valid_aliases = ", ".join(sorted(BACKBONE_TEMPLATES.keys()))
        raise BackboneRegistryError(
            f"[骨干模型配置非法] 未知的骨干模型别名: '{alias}'。"
            f"合法别名为: {valid_aliases}"
        )
    return BACKBONE_TEMPLATES[normalized]
