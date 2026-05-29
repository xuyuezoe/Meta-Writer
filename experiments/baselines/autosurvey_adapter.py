"""
AutoSurvey 对比系统适配器（S1）

对应论文：Zhu et al., 2024 —— 大规模检索 + LLM 分段生成综述。
官方仓库：https://github.com/AutoSurveys/AutoSurvey

接入方式：
    通过 SubprocessBaselineAdapter 的标准桥接契约驱动。AutoSurvey 的真实 CLI 与
    依赖由 envs/autosurvey.json 声明（repo 路径、独立解释器、桥接脚本），
    本类仅固定系统标识与仓库地址。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .subprocess_adapter import SubprocessBaselineAdapter

REPO_URL = "https://github.com/AutoSurveys/AutoSurvey"


class AutoSurveyAdapter(SubprocessBaselineAdapter):
    """AutoSurvey 适配器：固定 system_name 与 repo_url，复用子进程桥接逻辑。"""

    def __init__(self, *, envs_dir: Optional[Path] = None) -> None:
        super().__init__(system_name="autosurvey", repo_url=REPO_URL, envs_dir=envs_dir)
