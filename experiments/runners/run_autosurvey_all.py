"""
全量 50 任务顺序驱动：在更新后的系统里重测 autosurvey baseline。

设计要点（第一性原理）：
    1. 进程内顺序执行：dotenv 与各类 import 仅加载一次，避免 50× 解释器/SDK 冷启动开销；
       同时对中转 API 保持单并发（参考 lira 打爆中转的教训，绝不并行轰炸）。
    2. 故障隔离：单任务异常不中断整批，记录到进度文件后继续下一个。
    3. 断点续跑：进度文件记录已成功的 task_id；重启时自动跳过，已成功者不重复生成。
    4. 显式产物：每完成一个任务即把指标追加进度文件并 flush，过程可观测。

用法：
    python experiments/runners/run_autosurvey_all.py
        [--system autosurvey] [--model minimax]
        [--progress experiments_out/manifests/autosurvey_all_progress.json]
        [--force]   # 忽略进度文件，全部强制重跑
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _load_task_ids() -> List[str]:
    """从基准任务注册表政策文件解析 50 个任务 ID（metabench_pmcXXX -> med_pmcXXX）。"""
    policy_file = _PROJECT_ROOT / "data_sample" / "experiments" / "task_registry_policy.json"
    policy = json.loads(policy_file.read_text(encoding="utf-8"))
    names = policy["benchmark_task_names"]
    return [name.replace("metabench_", "med_") for name in names]


def _load_progress(progress_file: Path) -> Dict[str, Any]:
    if progress_file.exists():
        return json.loads(progress_file.read_text(encoding="utf-8"))
    return {"completed": {}, "failed": {}}


def _save_progress(progress_file: Path, progress: Dict[str, Any]) -> None:
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    progress_file.write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="顺序重测 autosurvey 全量任务")
    parser.add_argument("--system", default="autosurvey")
    parser.add_argument("--model", default="minimax")
    parser.add_argument(
        "--progress",
        default="experiments_out/manifests/autosurvey_all_progress.json",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # 配置加载（与 main.py 实验模式一致）：根 .env + baseline_env（不覆盖已有环境）
    from dotenv import load_dotenv

    load_dotenv(override=False)
    baseline_env = _PROJECT_ROOT / "experiments" / "baselines" / "envs" / "baseline_env.env"
    if baseline_env.exists():
        load_dotenv(dotenv_path=str(baseline_env), override=False)

    # ── 子进程环境硬化（根因修复，确定性而非依赖桥接 setdefault）──────────────
    # 桥接子进程通过 os.environ 继承本进程环境。先前批量失败的根因是：
    #   1) 瞬时 GPU 显存被其他进程占满 → SentenceTransformer 加载触发联网回退；
    #   2) 联网时 httpx 发现 socks:// 代理但缺 socksio → ImportError 退出码 1。
    # 三道硬化（覆盖式，不用 setdefault，避免被继承环境压制）：
    #   - CUDA_VISIBLE_DEVICES="" → 嵌入模型固定 CPU，彻底摆脱 GPU 争用（生成走 API 不吃 GPU）；
    #   - HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE=1 → 嵌入模型纯离线读缓存，不发任何 HF 网络请求；
    #   - 清除 socks:// 代理 → 即便意外联网也不会撞 httpx socks 崩溃（API 走 http 代理仍可用）。
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    for _proxy_var in ("ALL_PROXY", "all_proxy"):
        if os.environ.get(_proxy_var, "").startswith("socks://"):
            os.environ.pop(_proxy_var, None)

    from experiments.runners import run_baseline_task

    task_ids = _load_task_ids()
    progress_file = Path(args.progress)
    progress = _load_progress(progress_file)
    if args.force:
        progress = {"completed": {}, "failed": {}}

    total = len(task_ids)
    print(f"[驱动] 系统={args.system} 模型标签={args.model} 任务数={total}", flush=True)

    for index, task_id in enumerate(task_ids, start=1):
        if task_id in progress["completed"]:
            print(f"[{index}/{total}] {task_id} 已完成，跳过。", flush=True)
            continue

        print(f"[{index}/{total}] {task_id} 开始生成 + 评估 ...", flush=True)
        start = time.time()
        # 每任务最多尝试 3 次：吸收瞬时抖动（如偶发网络/资源争用），
        # 连续失败才判定为真失败并隔离。
        max_attempts = 3
        summary = None
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                summary = run_baseline_task(
                    task_id=task_id,
                    system_name=args.system,
                    model_label=args.model,
                    run_id="r1",
                    root_dir="./experiments_out/runs",
                    overwrite=True,
                )
                break
            except Exception as exc:  # noqa: BLE001 — 显式记录每次尝试，便于诊断
                last_exc = exc
                print(
                    f"[{index}/{total}] {task_id} 第 {attempt}/{max_attempts} 次尝试失败: {exc!r}",
                    flush=True,
                )
                time.sleep(5)

        if summary is None:  # 故障隔离：连续失败，记录后继续下一个任务
            elapsed = round(time.time() - start, 1)
            progress["failed"][task_id] = {"error": repr(last_exc), "elapsed_s": elapsed}
            progress["completed"].pop(task_id, None)
            _save_progress(progress_file, progress)
            print(f"[{index}/{total}] {task_id} 最终失败（{elapsed}s）: {last_exc!r}", flush=True)
            continue

        elapsed = round(time.time() - start, 1)
        scores = summary.get("meta_bench_scores", {})
        progress["completed"][task_id] = {
            "status": summary.get("status"),
            "word_count": summary.get("word_count"),
            "elapsed_s": elapsed,
            "meta_bench_scores": scores,
        }
        progress["failed"].pop(task_id, None)
        _save_progress(progress_file, progress)
        print(
            f"[{index}/{total}] {task_id} 完成（{elapsed}s, words={summary.get('word_count')}, "
            f"status={summary.get('status')}）",
            flush=True,
        )

    done = len(progress["completed"])
    failed = len(progress["failed"])
    print(f"\n[驱动结束] 成功 {done}/{total}，失败 {failed}。", flush=True)
    if progress["failed"]:
        print(f"失败任务: {sorted(progress['failed'])}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
