"""
对比系统注册表：get_baseline_adapter

功能：
    按系统名称构造对应的对比系统适配器，统一执行层的接入入口。

设计动机：
    执行层（runners）只需提供系统名，无需关心是进程内（Direct-LLM）还是
    子进程（外部系统）。非法名称显式抛错（禁止兜底）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .autosurvey_adapter import AutoSurveyAdapter
from .base_adapter import BaselineAdapter, BaselineAdapterError
from .direct_llm import DirectLLMAdapter
from .lira_adapter import LiRAAdapter
from .paperorchestra_adapter import PaperOrchestraAdapter
from .surveyforge_adapter import SurveyForgeAdapter

# 已知对比系统名称集合（含别名）
KNOWN_BASELINES = (
    "direct-llm",
    "autosurvey",
    "lira",
    "surveyforge",
    "paperorchestra",
)

# 外部系统名称 → 适配器类
_EXTERNAL_ADAPTERS = {
    "autosurvey": AutoSurveyAdapter,
    "lira": LiRAAdapter,
    "surveyforge": SurveyForgeAdapter,
    "paperorchestra": PaperOrchestraAdapter,
}


def get_baseline_adapter(
    system_name: str,
    *,
    llm_client: Optional[Any] = None,
    envs_dir: Optional[Path] = None,
) -> BaselineAdapter:
    """
    按系统名构造对比系统适配器

    参数：
        system_name: 系统名称（KNOWN_BASELINES 之一，大小写不敏感）。
        llm_client:  Direct-LLM 所需的 LLM 客户端；仅 "direct-llm" 需要。
        envs_dir:    外部系统 env 配置目录（默认 baselines/envs）。

    返回值：
        BaselineAdapter：对应系统的适配器实例

    异常：
        BaselineAdapterError：系统名未知，或 direct-llm 缺少 llm_client 时抛出。
    """
    normalized = system_name.strip().lower()

    if normalized == "direct-llm":
        if llm_client is None:
            raise BaselineAdapterError(
                "[对比系统构造失败] direct-llm 需要 llm_client，但未提供。"
            )
        return DirectLLMAdapter(llm_client)

    adapter_cls = _EXTERNAL_ADAPTERS.get(normalized)
    if adapter_cls is None:
        valid = ", ".join(KNOWN_BASELINES)
        raise BaselineAdapterError(
            f"[对比系统未知] 不支持的系统名: '{system_name}'。合法名称: {valid}"
        )
    return adapter_cls(envs_dir=envs_dir)
