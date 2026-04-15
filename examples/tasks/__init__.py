"""任务注册表。

新增任务步骤：
  1. 在 `examples/tasks/` 下创建新文件，并实现 `get_task_config()`
  2. 在此处 import 并注册到 `TASK_REGISTRY`
  3. 如需正式 benchmark 样本入口，优先通过 `benchmark_template` 动态注册
"""

from __future__ import annotations

from typing import Callable, Dict

from .argumentative_essay import get_task_config as _argumentative_essay
from .metabench_sample import get_task_config as _metabench_sample
from .scifi_story import get_task_config as _scifi_story
from .survey_paper import get_task_config as _survey_paper
from ..benchmark_template import build_benchmark_task_config, list_benchmark_task_ids


def _build_benchmark_task_factory(task_id: str) -> Callable[[], Dict[str, object]]:
    """
    构造 benchmark 任务工厂函数。

    设计目的：
        让每个 benchmark 样本都能在任务注册表里表现成一个普通任务入口，
        这样 `main.py --task-id ...` 与主流程无需再维护两套任务加载逻辑。
    """

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
