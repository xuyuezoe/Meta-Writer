"""
实验执行层（L2 runners）

功能：
    把配置层（config）与系统实现（src orchestrator / baselines）组装为可执行的
    单次运行与矩阵批跑。

子模块：
    run_metawriter — 跑 MetaWriter（Full 或某消融变体）单任务
    run_baseline   — 跑对比系统（Direct-LLM 或外部系统）单任务
    batch_driver   — 矩阵驱动：任务 × 方法 × 模型 × 重复，支持断点续跑与预算监控
"""
from __future__ import annotations

from .run_baseline import run_baseline_task
from .run_metawriter import run_metawriter_task

__all__ = [
    "run_baseline_task",
    "run_metawriter_task",
]
