"""
SurveyForge 对比系统适配器（S3）

对应论文：SurveyForge, 2024 —— 结构化综述生成框架。
官方仓库：https://github.com/Alpha-Innovator/SurveyForge

接入方式：
    通过 SubprocessBaselineAdapter 的标准桥接契约驱动。SurveyForge 的真实 CLI 与
    依赖由 envs/surveyforge.json 声明，本类仅固定系统标识与仓库地址。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .subprocess_adapter import SubprocessBaselineAdapter

REPO_URL = "https://github.com/Alpha-Innovator/SurveyForge"


class SurveyForgeAdapter(SubprocessBaselineAdapter):
    """SurveyForge 适配器：固定 system_name 与 repo_url，复用子进程桥接逻辑。"""

    def __init__(self, *, envs_dir: Optional[Path] = None) -> None:
        super().__init__(system_name="surveyforge", repo_url=REPO_URL, envs_dir=envs_dir)
