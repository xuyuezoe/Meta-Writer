"""Task registry for the local example bundle."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Callable

from .argumentative_essay import get_task_config as _argumentative_essay
from .metabench_pmc12440783 import get_task_config as _metabench_pmc12440783
from .metabench_sample import get_task_config as _metabench_sample
from .scifi_story import get_task_config as _scifi_story

TaskFactory = Callable[[], dict[str, object]]

REVIEW_MEDIUM_MIN_BODY_WORDS = 3500
REVIEW_LONG_MIN_BODY_WORDS = 5000


def _load_generated_task_exclusions() -> set[str]:
    repo_root = Path(__file__).resolve().parents[2]
    policy_path = repo_root / "data_sample" / "experiments" / "task_registry_policy.json"
    if not policy_path.exists():
        return set()

    try:
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except Exception:
        return set()

    excluded = payload.get("excluded_task_names")
    if not isinstance(excluded, list):
        return set()

    return {
        str(task_name).strip()
        for task_name in excluded
        if str(task_name).strip()
    }


GENERATED_TASK_EXCLUSIONS = _load_generated_task_exclusions()

TASK_REGISTRY: dict[str, TaskFactory] = {
    "scifi_story": _scifi_story,
    "argumentative_essay": _argumentative_essay,
    "metabench_sample": _metabench_sample,
    "metabench_pmc12440783": _metabench_pmc12440783,
}


def _register_generated_metabench_tasks() -> None:
    tasks_dir = Path(__file__).resolve().parent
    for module_path in sorted(tasks_dir.glob("metabench_pmc*.py")):
        module_name = module_path.stem
        if module_name in TASK_REGISTRY:
            continue
        if module_name in GENERATED_TASK_EXCLUSIONS:
            continue
        module = importlib.import_module(f"{__name__}.{module_name}")
        task_factory = getattr(module, "get_task_config", None)
        if callable(task_factory):
            TASK_REGISTRY[module_name] = task_factory


_register_generated_metabench_tasks()


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


def _extract_body_target_words(task_factory: TaskFactory) -> int | None:
    config = task_factory()
    reference = config.get("reference")
    if not isinstance(reference, dict):
        return None

    constraints = reference.get("constraints")
    if not isinstance(constraints, dict):
        return None

    raw_value = constraints.get("body_target_words")
    if not isinstance(raw_value, int):
        return None

    return raw_value if raw_value > 0 else None


TASK_ID_REGISTRY: dict[str, str] = {}
for _task_name, _task_factory in TASK_REGISTRY.items():
    _task_id = _extract_reference_task_id(_task_factory)
    if _task_id is None:
        continue
    if _task_id in TASK_ID_REGISTRY:
        raise ValueError(f"Duplicate task_id registered for MetaBench tasks: {_task_id}")
    TASK_ID_REGISTRY[_task_id] = _task_name

META_BENCH_ALL_TASK_NAMES = sorted(
    task_name
    for task_name in TASK_ID_REGISTRY.values()
    if task_name.startswith("metabench_pmc")
)

META_BENCH_MEDIUM_TASK_NAMES = sorted(
    task_name
    for task_name in META_BENCH_ALL_TASK_NAMES
    if (
        (body_words := _extract_body_target_words(TASK_REGISTRY[task_name])) is not None
        and REVIEW_MEDIUM_MIN_BODY_WORDS <= body_words < REVIEW_LONG_MIN_BODY_WORDS
    )
)

META_BENCH_LONG_TASK_NAMES = sorted(
    task_name
    for task_name in META_BENCH_ALL_TASK_NAMES
    if (
        (body_words := _extract_body_target_words(TASK_REGISTRY[task_name])) is not None
        and body_words >= REVIEW_LONG_MIN_BODY_WORDS
    )
)

META_BENCH_TASK_TIERS = {
    "medium": META_BENCH_MEDIUM_TASK_NAMES,
    "long": META_BENCH_LONG_TASK_NAMES,
    "benchmark": sorted(META_BENCH_MEDIUM_TASK_NAMES + META_BENCH_LONG_TASK_NAMES),
    "all": META_BENCH_ALL_TASK_NAMES,
}

# Batch/default MetaBench runs target medium and long real-review PMC tasks.
# Keep short PMC tasks and metabench_sample registered for explicit runs.
META_BENCH_TASK_NAMES = list(META_BENCH_TASK_TIERS["benchmark"])

__all__ = [
    "META_BENCH_ALL_TASK_NAMES",
    "META_BENCH_LONG_TASK_NAMES",
    "META_BENCH_MEDIUM_TASK_NAMES",
    "META_BENCH_TASK_NAMES",
    "META_BENCH_TASK_TIERS",
    "REVIEW_LONG_MIN_BODY_WORDS",
    "REVIEW_MEDIUM_MIN_BODY_WORDS",
    "TASK_ID_REGISTRY",
    "TASK_REGISTRY",
]
