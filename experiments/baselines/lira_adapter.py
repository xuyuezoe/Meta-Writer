"""
LiRA 对比系统适配器（S2）

对应论文：LiRA, 2024 —— 文献检索增强的综述自动生成。
官方仓库：https://github.com/lira-workflow/auto-review-writing

接入方式：
    通过 SubprocessBaselineAdapter 的标准桥接契约驱动。LiRA 的真实 CLI 与依赖
    由 envs/lira.json 声明，本类仅固定系统标识与仓库地址。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .subprocess_adapter import SubprocessBaselineAdapter

REPO_URL = "https://github.com/lira-workflow/auto-review-writing"


class LiRAAdapter(SubprocessBaselineAdapter):
    """LiRA 适配器：固定 system_name 与 repo_url，复用子进程桥接逻辑。"""

    def __init__(self, *, envs_dir: Optional[Path] = None) -> None:
        super().__init__(system_name="lira", repo_url=REPO_URL, envs_dir=envs_dir)
