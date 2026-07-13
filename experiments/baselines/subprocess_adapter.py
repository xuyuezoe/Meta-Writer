"""Subprocess bridge adapter for external baseline systems."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .base_adapter import BaselineAdapterError, BaselineResult, BaselineTask


class SubprocessBaselineAdapter:
    """Run an external system through a small bridge-script contract."""

    def __init__(
        self,
        system_name: str,
        repo_url: str,
        *,
        envs_dir: str | Path | None = None,
    ) -> None:
        self.system_name = system_name.strip().lower()
        self.repo_url = repo_url
        self.envs_dir = (
            Path(envs_dir)
            if envs_dir is not None
            else Path("experiments/baselines/envs")
        )

    def run(self, task: BaselineTask, *, work_dir: str) -> BaselineResult:
        config = self._load_config()
        repo_path = _required_path(config, "repo_path")
        python_executable = str(config.get("python_executable", "")).strip()
        bridge_script = _required_path(config, "bridge_script")
        if not python_executable:
            raise BaselineAdapterError("missing python_executable in baseline config")
        if not bridge_script.exists():
            raise BaselineAdapterError(f"bridge_script does not exist: {bridge_script}")

        work_path = Path(work_dir)
        work_path.mkdir(parents=True, exist_ok=True)
        input_file = work_path / "input.json"
        output_file = work_path / "output.txt"
        stats_file = work_path / "stats.json"
        input_file.write_text(
            json.dumps(task.to_bridge_payload(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        cmd = [
            python_executable,
            str(bridge_script),
            "--input",
            str(input_file),
            "--output",
            str(output_file),
            "--stats",
            str(stats_file),
            *[str(item) for item in config.get("extra_args", [])],
        ]
        timeout_seconds = int(config.get("timeout_seconds", 1800) or 1800)
        completed = subprocess.run(
            cmd,
            cwd=str(repo_path),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        if completed.returncode != 0:
            raise BaselineAdapterError(
                "baseline subprocess exited with non-zero status "
                f"({completed.returncode}); 非零退出; 闈為浂閫€鍑? "
                f"stderr={completed.stderr.strip()}"
            )
        if not output_file.exists():
            raise BaselineAdapterError("baseline subprocess did not write output.txt")
        final_text = output_file.read_text(encoding="utf-8").strip()
        if not final_text:
            raise BaselineAdapterError("baseline subprocess produced empty output")

        stats = _read_stats(stats_file)
        return BaselineResult(
            system=self.system_name,
            final_text=final_text,
            total_tokens=int(stats.get("total_tokens", 0) or 0),
            request_count=int(stats.get("request_count", 0) or 0),
            extra={
                "repo_url": self.repo_url,
                "repo_path": str(repo_path),
                "bridge_script": str(bridge_script),
            },
        )

    def _load_config(self) -> dict[str, Any]:
        config_file = self.envs_dir / f"{self.system_name}.json"
        if not config_file.exists():
            raise BaselineAdapterError(
                f"baseline system is not configured: {self.system_name}; "
                "missing config; 未配置; 缺少配置; 鏈厤缃; 缂哄皯閰嶇疆"
            )
        raw = json.loads(config_file.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise BaselineAdapterError("baseline config must be a JSON object")
        return raw


def _required_path(config: dict[str, Any], key: str) -> Path:
    raw_value = str(config.get(key, "")).strip()
    if not raw_value:
        raise BaselineAdapterError(f"missing required baseline config field: {key}")
    return Path(raw_value)


def _read_stats(stats_file: Path) -> dict[str, Any]:
    if not stats_file.exists():
        return {}
    try:
        raw = json.loads(stats_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}
