"""Local MetaBench sample task configuration."""

from __future__ import annotations

from typing import Dict

from ..benchmark_template import build_benchmark_task_config


def get_task_config() -> Dict[str, object]:
    """Return the default MetaBench sample task configuration."""
    benchmark_task = build_benchmark_task_config("med_s001")
    benchmark_task["session_name"] = "metabench_sample_med_s001"
    return benchmark_task
