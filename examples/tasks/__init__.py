"""任务注册表。

新增任务步骤：
  1. 在 examples/tasks/ 下创建新文件，实现 get_task_config() 函数
  2. 在此处 import 并注册到 TASK_REGISTRY
  3. 在 main.py 中将 TASK_NAME 改为新任务名
"""

from __future__ import annotations

from typing import Callable, Dict

from .scifi_story import get_task_config as _scifi_story
from .argumentative_essay import get_task_config as _argumentative_essay
from .metabench_sample import get_task_config as _metabench_sample
from ..benchmark_template import build_benchmark_task_config, list_benchmark_task_ids


def _build_benchmark_task_factory(task_id: str) -> Callable[[], Dict[str, object]]:
    """构造 benchmark 任务工厂函数。

    参数：
        task_id: benchmark 样本 ID。

    返回值：
        Callable[[], Dict[str, object]]：延迟加载的任务配置函数。

    关键实现细节：
        使用闭包保留 task_id，使注册表可直接暴露多个 bench 样本入口。
    """

    def _task_factory() -> Dict[str, object]:
        return build_benchmark_task_config(task_id)

    return _task_factory


TASK_REGISTRY: Dict[str, Callable] = {
    "scifi_story": _scifi_story,
    "argumentative_essay": _argumentative_essay,
    "metabench_sample": _metabench_sample,
}

for _benchmark_task_id in list_benchmark_task_ids():
    TASK_REGISTRY[f"metabench_{_benchmark_task_id}"] = _build_benchmark_task_factory(
        _benchmark_task_id
    )
