"""
对比系统适配器包（baselines）

功能：
    为 EXP-II（系统横向对比）提供统一的对比系统接入层，把外部已发表系统
    （AutoSurvey / LiRA / SurveyForge / PaperOrchestra）与内部 Direct-LLM 基线
    统一到同一 I/O 契约下，使它们与 MetaWriter 在完全相同的任务、语料、评估口径
    下可比。

设计动机（第一性原理）：
    公平对比的本质是"输入相同、评估相同，只有系统不同"。因此适配器层必须：
        1. 统一输入：把 MetaBench 任务规范化为 BaselineTask（与各系统无关）。
        2. 统一输出：所有系统都产出 (final_text, 运行统计)，再用同一个
           evaluate_meta_bench 评估，杜绝评估口径漂移。
        3. 环境隔离：外部系统依赖各异，通过子进程 + 独立解释器调用，
           不把它们的依赖污染主环境。
        4. 显式失败：环境未配置 / 输出缺失时显式抛错，绝不静默产出空文本。

接入约定：
    - Direct-LLM：进程内直接用 LLMClient 单次生成，无外部依赖。
    - 四个外部系统：通过 envs/<system>.json 声明 repo 路径、解释器、命令模版，
      经标准化 input.json 桥接调用，读取 output.txt 作为最终文本。
"""
from __future__ import annotations

from .base_adapter import (
    BaselineAdapter,
    BaselineAdapterError,
    BaselineResult,
    BaselineTask,
)
from .registry import KNOWN_BASELINES, get_baseline_adapter

__all__ = [
    "BaselineAdapter",
    "BaselineAdapterError",
    "BaselineResult",
    "BaselineTask",
    "KNOWN_BASELINES",
    "get_baseline_adapter",
]
