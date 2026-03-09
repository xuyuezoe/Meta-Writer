"""
任务注册表

新增任务步骤：
  1. 在 examples/tasks/ 下创建新文件，实现 get_task_config() 函数
  2. 在此处 import 并注册到 TASK_REGISTRY
  3. 在 main.py 中将 TASK_NAME 改为新任务名
"""
from typing import Dict, Callable

from .scifi_story import get_task_config as _scifi_story
from .argumentative_essay import get_task_config as _argumentative_essay

TASK_REGISTRY: Dict[str, Callable] = {
    "scifi_story":          _scifi_story,
    "argumentative_essay":  _argumentative_essay,
}
