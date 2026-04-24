"""Task registry for the local example bundle."""

from __future__ import annotations

from typing import Callable, Dict

from .argumentative_essay import get_task_config as _argumentative_essay
from .metabench_sample import get_task_config as _metabench_sample
from .scifi_story import get_task_config as _scifi_story
from .survey_paper import get_task_config as _survey_paper
from ..benchmark_template import build_benchmark_task_config, list_benchmark_task_ids


def _build_benchmark_task_factory(task_id: str) -> Callable[[], Dict[str, object]]:
    """Build a task factory so each benchmark sample behaves like a normal task."""

    def _task_factory() -> Dict[str, object]:
        return build_benchmark_task_config(task_id)

    return _task_factory


TASK_REGISTRY: Dict[str, Callable[[], Dict[str, object]]] = {
    "scifi_story": _scifi_story,
    "argumentative_essay": _argumentative_essay,
    "survey_paper": _survey_paper,
    "metabench_sample": _metabench_sample,
}

for _benchmark_task_id in list_benchmark_task_ids():
    TASK_REGISTRY[f"metabench_{_benchmark_task_id}"] = _build_benchmark_task_factory(
        _benchmark_task_id
    )
