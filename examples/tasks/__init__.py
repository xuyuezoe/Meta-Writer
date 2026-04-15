"""任务注册表。

新增任务步骤：
  1. 在 `examples/tasks/` 下创建新文件，实现 `get_task_config()` 函数
  2. 在此处 import 并注册到 `TASK_REGISTRY`
  3. 通过 `main.py` 的 `--task` 或 `TASK_NAME` 环境变量选择任务
"""

from __future__ import annotations

from typing import Callable, Dict

from .argumentative_essay import get_task_config as _argumentative_essay
from .scifi_story import get_task_config as _scifi_story
from .survey_paper import get_task_config as _survey_paper


TASK_REGISTRY: Dict[str, Callable[[], Dict[str, object]]] = {
    "scifi_story": _scifi_story,
    "argumentative_essay": _argumentative_essay,
    "survey_paper": _survey_paper,
}
