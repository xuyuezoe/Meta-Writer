"""任务配置：MetaBench 本地样本。

功能：
    将仓库内整理后的 benchmark 样本接入 Meta-Writer 任务注册表，
    复用 `examples/benchmark_template.py` 中的加载接口。
"""

from __future__ import annotations

from typing import Dict

from ..benchmark_template import build_benchmark_task_config


def get_task_config() -> Dict[str, object]:
    """返回 MetaBench 样本任务配置。

    参数：
        无。

    返回值：
        Dict[str, object]：包含 Meta-Writer 生成所需的任务、约束、大纲和会话名。

    关键实现细节：
        当前固定接入 `med_s001` 样本，保证医学 bench example 接口可直接运行。
    """
    benchmark_task = build_benchmark_task_config("med_s001")
    benchmark_task["session_name"] = "metabench_sample_med_s001"
    return benchmark_task
