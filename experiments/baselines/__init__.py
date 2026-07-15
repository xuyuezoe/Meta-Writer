"""Baseline adapter package used by experiment runners."""

from __future__ import annotations

from .base_adapter import BaselineAdapterError, BaselineResult, BaselineTask
from .direct_llm import DirectLLMAdapter
from .registry import get_baseline_adapter
from .subprocess_adapter import SubprocessBaselineAdapter

__all__ = [
    "BaselineAdapterError",
    "BaselineResult",
    "BaselineTask",
    "DirectLLMAdapter",
    "SubprocessBaselineAdapter",
    "get_baseline_adapter",
]
