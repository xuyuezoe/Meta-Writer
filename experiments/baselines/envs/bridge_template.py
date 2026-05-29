"""
对比系统桥接脚本模版（bridge_template）

功能：
    本模版示范"标准契约 ↔ 某外部综述系统私有 API"之间的翻译层。
    请将本文件复制进各对比系统的 repo，并把 TODO 处替换为该系统的真实调用。

契约（由 MetaWriter 的 SubprocessBaselineAdapter 约定）：
    输入：--input  <path>  指向标准 input.json
    输出：--output <path>  须写入最终综述全文（UTF-8 纯文本）
    可选：--stats  <path>  可写入 {"total_tokens": int, "request_count": int}

运行环境：
    本脚本在对比系统自己的虚拟环境内、以其 repo 为工作目录被调用，
    因此可直接 import 该系统的模块。API 密钥/端点从环境变量读取
    （由 baseline_env.env 注入）。

设计要求：
    - 不得静默失败：任何缺字段、调用失败都应抛异常并以非零码退出，
      由上层适配器捕获 stderr 并显式报错。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


def _parse_args() -> argparse.Namespace:
    """解析标准契约参数"""
    parser = argparse.ArgumentParser(description="MetaWriter baseline bridge")
    parser.add_argument("--input", required=True, help="标准 input.json 路径")
    parser.add_argument("--output", required=True, help="最终综述文本输出路径")
    parser.add_argument("--stats", required=False, help="可选 token 统计 JSON 输出路径")
    return parser.parse_args()


def _load_task(input_path: str) -> Dict[str, Any]:
    """读取标准任务输入"""
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"input.json 顶层应为对象，实际为 {type(payload)}")
    return payload


def _run_target_system(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    调用目标对比系统生成综述（每个系统在此处接入自身真实管线）

    参数：
        task: 标准任务输入（含 topic / task_description / constraints / outline /
              corpus_dir / target_words）

    返回值：
        Dict：至少含 "final_text"；可选含 "total_tokens" / "request_count"

    TODO（接入者填写）：
        用 task 中的字段构造该系统所需输入，调用其管线，取回最终综述文本。
        例如（伪代码）：
            from autosurvey.pipeline import run_survey
            text = run_survey(
                topic=task["topic"],
                outline=task["outline"],
                corpus_dir=task["corpus_dir"],
                target_words=task["target_words"],
            )
            return {"final_text": text, "total_tokens": ..., "request_count": ...}
    """
    raise NotImplementedError(
        "请在桥接脚本中接入目标对比系统的真实生成管线（见本函数 docstring 的 TODO）。"
    )


def main() -> int:
    """桥接主流程：读输入 → 跑系统 → 写输出/统计"""
    args = _parse_args()
    task = _load_task(args.input)

    result = _run_target_system(task)
    final_text = result.get("final_text")
    if not isinstance(final_text, str) or not final_text.strip():
        raise ValueError("目标系统未产出非空 final_text")

    Path(args.output).write_text(final_text, encoding="utf-8")

    if args.stats:
        stats = {
            "total_tokens": result.get("total_tokens"),
            "request_count": result.get("request_count"),
        }
        Path(args.stats).write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
