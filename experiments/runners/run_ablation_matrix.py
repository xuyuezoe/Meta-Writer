"""
消融实验矩阵批跑 CLI（命令行入口）

功能：
    把 batch_driver 的函数式矩阵驱动（任务 × 消融预设 × 重复）封装为一个
    可直接 `python3 experiments/runners/run_ablation_matrix.py ...` 调用的命令行入口。
    供不熟悉代码的执行者只通过命令行即可跑完整套消融实验，无需写 Python。

设计动机：
    batch_driver.run_matrix / build_ablation_matrix 仅提供函数接口，没有 CLI。
    本脚本只做"参数解析 → 任务集解析 → 构建矩阵 → 调用 run_matrix → 打印汇总"，
    不引入任何新逻辑，保证与既有运行器口径完全一致。

典型用法：
    # 1) 用根目录 .env 的模型，在 3 个指定任务上跑全部消融预设，各跑 1 次
    python3 experiments/runners/run_ablation_matrix.py \
        --tasks med_pmc12440783,med_pmc12852008,med_pmc12855103 \
        --model default --runs 1

    # 2) 在全部 50 个基准任务上跑全部消融预设，各跑 3 次（完整消融实验）
    python3 experiments/runners/run_ablation_matrix.py --tasks all --runs 3

    # 3) 只跑 full 与 a1_no_dsl 两个变体做快速验证
    python3 experiments/runners/run_ablation_matrix.py \
        --tasks med_pmc12440783 --presets full,a1_no_dsl
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

# 将项目根目录加入 import 路径，使脚本可被直接执行（python3 路径/本文件）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

from experiments.config.ablation import ABLATION_PRESETS
from experiments.runners.batch_driver import build_ablation_matrix, run_matrix


def _resolve_task_ids(tasks_arg: str) -> List[str]:
    """
    解析 --tasks 参数为具体任务 ID 列表

    参数：
        tasks_arg: "all" / "eval40" / "eval20" / 逗号分隔的 task_id 列表

    返回值：
        List[str]：任务 ID 列表

    异常：
        ValueError：任务集关键字非法、或显式列表为空时抛出。
    """
    normalized = tasks_arg.strip().lower()

    if normalized == "all":
        # 全部已注册的 MetaBench 基准任务（task_id → task_name 映射的键）
        from examples.tasks import TASK_ID_REGISTRY

        return sorted(TASK_ID_REGISTRY.keys())

    if normalized == "eval40":
        # 论文 40 任务主评估集（按词数分层确定性抽样；若任务池某层不足会显式报错）
        from experiments.config.eval_subset import build_eval_40

        return build_eval_40()

    if normalized == "eval20":
        from experiments.config.eval_subset import build_eval_20

        return build_eval_20()

    explicit = [item.strip() for item in tasks_arg.split(",") if item.strip()]
    if not explicit:
        raise ValueError(
            "[参数非法] --tasks 为空。请给出 all / eval40 / eval20，"
            "或逗号分隔的 task_id（如 med_pmc12440783,med_pmc12852008）。"
        )
    return explicit


def _resolve_presets(presets_arg: str) -> List[str]:
    """
    解析 --presets 参数为消融预设名列表

    参数：
        presets_arg: "all" 或逗号分隔的预设名（如 full,a1_no_dsl）

    返回值：
        List[str]：预设名列表

    异常：
        ValueError：出现未注册的预设名时抛出，附带全部合法名称。
    """
    if presets_arg.strip().lower() == "all":
        # 固定顺序：full 在先，其余按注册表声明顺序
        return list(ABLATION_PRESETS.keys())

    requested = [item.strip() for item in presets_arg.split(",") if item.strip()]
    invalid = [name for name in requested if name not in ABLATION_PRESETS]
    if invalid:
        valid_names = ", ".join(sorted(ABLATION_PRESETS.keys()))
        raise ValueError(
            f"[参数非法] 未知消融预设: {invalid}。合法预设为: {valid_names}"
        )
    return requested


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="消融实验矩阵批跑：任务 × 消融预设 × 重复",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help="任务集：all（全部50基准任务）/ eval40 / eval20 / 逗号分隔的 task_id。默认 all。",
    )
    parser.add_argument(
        "--presets",
        type=str,
        default="all",
        help="消融预设：all（全部8个变体）或逗号分隔（如 full,a1_no_dsl）。默认 all。",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="default",
        help="骨干模型别名：default（用根 .env 的 API_KEY/MODEL/BASE_URL）/ minimax / gpt-4o / "
        "claude-sonnet / deepseek-v3 / qwen2.5-72b。默认 default。",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="每个（任务×预设）组合的重复次数（论文报告均值±标准差时用 3）。默认 1。",
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default="./experiments_out/runs",
        help="运行产物根目录。默认 ./experiments_out/runs。",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=None,
        help="累计 token 上限，超过即停止（防止超支）。默认不限。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="强制重跑：忽略已存在的 summary.json（默认开启断点续跑，跳过已完成单元）。",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="任一运行单元失败立即中断整批（默认故障隔离，失败被记录后继续）。",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # 载入环境变量：根 .env（default 别名用）+ 五骨干模型 baseline_env（若存在）
    load_dotenv(override=False)
    baseline_env_file = _PROJECT_ROOT / "experiments/baselines/envs/baseline_env.env"
    if baseline_env_file.exists():
        load_dotenv(dotenv_path=str(baseline_env_file), override=False)

    task_ids = _resolve_task_ids(args.tasks)
    preset_names = _resolve_presets(args.presets)

    specs = build_ablation_matrix(
        task_ids=task_ids,
        preset_names=preset_names,
        model=args.model,
        runs=args.runs,
    )

    print("=" * 60)
    print("消融实验矩阵批跑")
    print("=" * 60)
    print(f"  任务数:     {len(task_ids)}")
    print(f"  消融预设:   {preset_names}")
    print(f"  骨干模型:   {args.model}")
    print(f"  重复次数:   {args.runs}")
    print(f"  运行单元总数: {len(specs)}")
    print(f"  产物根目录: {args.root_dir}")
    print(f"  断点续跑:   {'否（overwrite）' if args.overwrite else '是'}")
    print("=" * 60)

    report = run_matrix(
        specs,
        root_dir=args.root_dir,
        token_budget=args.token_budget,
        overwrite=bool(args.overwrite),
        fail_fast=bool(args.fail_fast),
    )

    counts = report["counts"]
    print("\n" + "=" * 60)
    print("批跑完成")
    print("=" * 60)
    print(f"  完成:       {counts['completed']}")
    print(f"  跳过(续跑): {counts['skipped']}")
    print(f"  失败:       {counts['failed']}")
    print(f"  预算停止:   {counts['stopped_budget']}")
    print(f"  累计 token: {report['total_tokens']:,}")
    print(f"  进度清单:   {report['progress_file']}")

    if counts["failed"] > 0:
        print("\n部分运行单元失败，请查看进度清单中的 traceback。")
        sys.exit(1)


if __name__ == "__main__":
    main()
