"""
外部对比系统子进程适配器：SubprocessBaselineAdapter

功能：
    以"子进程 + 独立解释器 + 标准化桥接"的方式驱动外部已发表系统
    （AutoSurvey / LiRA / SurveyForge / PaperOrchestra），实现环境隔离的真实接入。

设计动机（第一性原理）：
    外部系统各有独立、互相冲突的依赖，绝不能塞进主环境。唯一稳健的接入方式是
    进程边界隔离：
        1. 契约标准化：本适配器把任务写为标准 input.json，约定外部系统的桥接脚本
           读取它、驱动其自身管线、把最终文本写到 output.txt。
        2. 环境隔离：用该系统自己的解释器（独立 venv）在其 repo 目录内运行桥接脚本。
        3. 配置外置：repo 路径、解释器、桥接脚本、超时等声明在 envs/<system>.json，
           由用户在拥有各 repo 时填写；API 密钥统一从 baseline_env 注入子进程环境。
        4. 显式失败：配置缺失、子进程非零退出、输出缺失/为空，一律显式抛错。

桥接契约（外部 repo 侧需提供的 bridge 脚本）：
    输入：--input <path>  指向本适配器写出的标准 input.json
    输出：--output <path> 桥接脚本须把最终综述文本写入该路径（UTF-8 纯文本）
    可选：--stats <path>  桥接脚本可写出 {"total_tokens":int,"request_count":int}
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_adapter import (
    BaselineAdapter,
    BaselineAdapterError,
    BaselineResult,
    BaselineTask,
)

try:
    # python-dotenv 已是项目依赖（main.py 使用）；用于把 baseline_env 注入子进程
    from dotenv import dotenv_values
except Exception:  # pragma: no cover - 依赖缺失时给出明确指引
    dotenv_values = None  # type: ignore[assignment]


# envs 目录（与本文件同级）
_ENVS_DIR = Path(__file__).resolve().parent / "envs"
# baseline_env 统一 API 配置文件（用户填写）
_BASELINE_ENV_FILE = _ENVS_DIR / "baseline_env.env"


class SubprocessBaselineAdapter(BaselineAdapter):
    """
    外部系统子进程适配器

    功能：
        读取 envs/<system>.json 配置，写标准 input.json，调用桥接脚本，读 output.txt。

    参数：
        system_name: 系统标识（同时是 envs/<system_name>.json 的文件名主干）。
        repo_url:    该系统的官方仓库地址（仅作 provenance 记录与报错提示）。
        envs_dir:    envs 目录路径（默认与本模块同级，便于测试注入）。
    """

    def __init__(
        self,
        system_name: str,
        repo_url: str,
        *,
        envs_dir: Optional[Path] = None,
    ) -> None:
        self.system_name = system_name
        self.repo_url = repo_url
        self._envs_dir = Path(envs_dir) if envs_dir is not None else _ENVS_DIR

    # ------------------------------------------------------------------
    # 配置加载
    # ------------------------------------------------------------------

    def _config_path(self) -> Path:
        """该系统的 env 配置文件路径"""
        return self._envs_dir / f"{self.system_name}.json"

    def _load_config(self) -> Dict[str, Any]:
        """
        加载并校验该系统的 env 配置

        返回值：
            Dict：env 配置（含 repo_path / python_executable / bridge_script 等）

        异常：
            BaselineAdapterError：配置文件缺失、JSON 非法，或必填项为空时抛出。
        """
        config_path = self._config_path()
        if not config_path.exists():
            raise BaselineAdapterError(
                f"[对比系统未配置] 缺少配置文件 {config_path}。"
                f"请参照 envs/README.md 为 {self.system_name}（{self.repo_url}）填写配置。"
            )
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise BaselineAdapterError(
                f"[对比系统配置非法] {config_path} 不是合法 JSON: {exc}"
            ) from exc

        required = ("repo_path", "python_executable", "bridge_script")
        for key in required:
            value = config.get(key)
            if not isinstance(value, str) or not value.strip():
                raise BaselineAdapterError(
                    f"[对比系统配置不完整] {config_path} 缺少必填项 '{key}'。"
                    f"请填写 {self.system_name} 的 repo 路径 / 独立解释器 / 桥接脚本路径。"
                )
        return config

    def _load_baseline_env(self, env_file: Optional[Path] = None) -> Dict[str, str]:
        """
        加载 baseline_env 中的 API 配置（注入子进程环境）

        参数：
            env_file: baseline_env 文件路径（默认 envs/baseline_env.env）。

        返回值：
            Dict[str, str]：环境变量键值对；文件不存在时返回空字典（API 可能由
                            os.environ 直接提供，故缺失不报错，但会在子进程缺密钥时
                            由桥接脚本自身报错）。

        异常：
            BaselineAdapterError：当 python-dotenv 不可用时抛出（无法解析 env 文件）。
        """
        target = env_file if env_file is not None else _BASELINE_ENV_FILE
        if not target.exists():
            return {}
        if dotenv_values is None:
            raise BaselineAdapterError(
                "[对比系统依赖缺失] 需要 python-dotenv 解析 baseline_env，请先安装。"
            )
        values = dotenv_values(str(target))
        return {k: v for k, v in values.items() if v is not None}

    # ------------------------------------------------------------------
    # 运行
    # ------------------------------------------------------------------

    def run(self, task: BaselineTask, *, work_dir: str) -> BaselineResult:
        """
        驱动外部系统生成综述

        参数：
            task:     规范化对比任务
            work_dir: 工作目录（写入 input.json，读取 output.txt）

        返回值：
            BaselineResult：含最终文本与（若桥接提供的）token 统计

        异常：
            BaselineAdapterError：配置缺失 / 子进程失败 / 输出缺失或为空时抛出。
        """
        config = self._load_config()
        work_path = Path(work_dir)
        work_path.mkdir(parents=True, exist_ok=True)

        input_file = work_path / "input.json"
        output_file = work_path / "output.txt"
        stats_file = work_path / "output_stats.json"

        input_file.write_text(
            json.dumps(task.to_bridge_payload(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # 清理可能残留的旧输出，避免误读上一次结果
        for stale in (output_file, stats_file):
            if stale.exists():
                stale.unlink()

        command = self._build_command(config, input_file, output_file, stats_file)
        sub_env = self._build_subprocess_env(config)
        timeout_seconds = int(config.get("timeout_seconds", 3600))

        start = time.time()
        try:
            completed = subprocess.run(
                command,
                cwd=config["repo_path"],
                env=sub_env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise BaselineAdapterError(
                f"[对比系统启动失败] 无法执行命令 {command!r}（解释器或脚本路径不存在）: {exc}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise BaselineAdapterError(
                f"[对比系统超时] {self.system_name} 超过 {timeout_seconds}s 未完成: {exc}"
            ) from exc
        wall_time = time.time() - start

        if completed.returncode != 0:
            stderr_tail = (completed.stderr or "")[-2000:]
            raise BaselineAdapterError(
                f"[对比系统非零退出] {self.system_name} 退出码 {completed.returncode}。"
                f"stderr 尾部:\n{stderr_tail}"
            )

        text = self._read_output_text(output_file)
        stats = self._read_output_stats(stats_file)

        return BaselineResult(
            system=self.system_name,
            final_text=text,
            total_tokens=stats.get("total_tokens"),
            request_count=stats.get("request_count"),
            wall_time_seconds=round(wall_time, 3),
            status="completed",
            extra={
                "repo_url": self.repo_url,
                "repo_path": config["repo_path"],
                "command": command,
                "stdout_tail": (completed.stdout or "")[-1000:],
            },
        )

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _build_command(
        self,
        config: Dict[str, Any],
        input_file: Path,
        output_file: Path,
        stats_file: Path,
    ) -> List[str]:
        """
        构造子进程命令

        参数：
            config:      env 配置
            input_file:  标准输入 JSON 路径
            output_file: 期望的输出文本路径
            stats_file:  期望的统计 JSON 路径（可选产物）

        返回值：
            List[str]：完整命令 argv
        """
        command: List[str] = [
            config["python_executable"],
            config["bridge_script"],
            "--input", str(input_file.resolve()),
            "--output", str(output_file.resolve()),
            "--stats", str(stats_file.resolve()),
        ]
        extra_args = config.get("extra_args")
        if isinstance(extra_args, list):
            command.extend(str(arg) for arg in extra_args)
        return command

    def _build_subprocess_env(self, config: Dict[str, Any]) -> Dict[str, str]:
        """
        构造子进程环境变量

        功能：
            以 os.environ 为基底，叠加 baseline_env 的 API 配置，
            供外部系统读取密钥/端点。

        参数：
            config: env 配置（可含 env_file 覆盖 baseline_env 路径）

        返回值：
            Dict[str, str]：子进程环境变量
        """
        import os

        sub_env = dict(os.environ)
        env_file_value = config.get("env_file")
        env_file = Path(env_file_value) if isinstance(env_file_value, str) and env_file_value.strip() else None
        sub_env.update(self._load_baseline_env(env_file))
        return sub_env

    def _read_output_text(self, output_file: Path) -> str:
        """
        读取并校验输出文本

        异常：
            BaselineAdapterError：输出文件缺失或内容为空时抛出。
        """
        if not output_file.exists():
            raise BaselineAdapterError(
                f"[对比系统无输出] {self.system_name} 未生成 {output_file}。"
                f"请检查桥接脚本是否按契约写出 --output 指定的文本文件。"
            )
        text = output_file.read_text(encoding="utf-8").strip()
        if not text:
            raise BaselineAdapterError(
                f"[对比系统输出为空] {self.system_name} 的 {output_file} 为空文本"
            )
        return text

    def _read_output_stats(self, stats_file: Path) -> Dict[str, Any]:
        """
        读取可选的统计 JSON

        返回值：
            Dict：含 total_tokens / request_count；文件缺失或非法时返回空字典
                  （统计为可选产物，缺失不影响主流程，但会被映射为 None）。
        """
        if not stats_file.exists():
            return {}
        try:
            stats = json.loads(stats_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        if not isinstance(stats, dict):
            return {}
        return stats
