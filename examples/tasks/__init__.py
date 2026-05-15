"""Task registry for the local example bundle."""

from __future__ import annotations

from typing import Callable

from .argumentative_essay import get_task_config as _argumentative_essay
from .metabench_pmc12440783 import get_task_config as _metabench_pmc12440783
from .metabench_sample import get_task_config as _metabench_sample
from .scifi_story import get_task_config as _scifi_story

TaskFactory = Callable[[], dict[str, object]]

TASK_REGISTRY: dict[str, TaskFactory] = {
    "scifi_story": _scifi_story,
    "argumentative_essay": _argumentative_essay,
    "metabench_sample": _metabench_sample,
    "metabench_pmc12440783": _metabench_pmc12440783,
}


def _extract_reference_task_id(task_factory: TaskFactory) -> str | None:
    config = task_factory()
    reference = config.get("reference")
    if not isinstance(reference, dict):
        return None

    raw_task_id = reference.get("task_id")
    if not isinstance(raw_task_id, str):
        return None

    task_id = raw_task_id.strip()
    return task_id or None


TASK_ID_REGISTRY: dict[str, str] = {}
for _task_name, _task_factory in TASK_REGISTRY.items():
    _task_id = _extract_reference_task_id(_task_factory)
    if _task_id is None:
        continue
    if _task_id in TASK_ID_REGISTRY:
        raise ValueError(f"Duplicate task_id registered for MetaBench tasks: {_task_id}")
    TASK_ID_REGISTRY[_task_id] = _task_name

META_BENCH_TASK_NAMES = sorted(TASK_ID_REGISTRY.values())

__all__ = [
    "META_BENCH_TASK_NAMES",
    "TASK_ID_REGISTRY",
    "TASK_REGISTRY",
]
