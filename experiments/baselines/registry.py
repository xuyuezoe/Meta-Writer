"""Registry for baseline adapters."""

from __future__ import annotations

from pathlib import Path

from .base_adapter import BaselineAdapterError
from .direct_llm import DirectLLMAdapter
from .subprocess_adapter import SubprocessBaselineAdapter

EXTERNAL_BASELINES = {
    "autosurvey": "https://github.com/AutoSurveys/AutoSurvey",
    "lira": "https://github.com/LiRA-benchmark/LiRA",
    "surveyforge": "https://github.com/surveyforge/surveyforge",
    "paperorchestra": "https://github.com/paper-orchestra/paper-orchestra",
}


def get_baseline_adapter(
    system_name: str,
    *,
    llm_client=None,
    envs_dir: str | Path | None = None,
):
    """Return the adapter for one comparison system."""

    normalized = system_name.strip().lower()
    if normalized == "direct-llm":
        if llm_client is None:
            raise BaselineAdapterError("direct-llm requires an llm_client")
        return DirectLLMAdapter(llm_client)

    repo_url = EXTERNAL_BASELINES.get(normalized)
    if repo_url is None:
        raise BaselineAdapterError(f"unknown baseline system: {system_name}")
    return SubprocessBaselineAdapter(
        normalized,
        repo_url,
        envs_dir=envs_dir,
    )
