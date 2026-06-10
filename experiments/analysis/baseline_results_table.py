"""
汇总对比系统在全部任务上的全部指标，并算每系统均分。

读取 experiments_out/runs/<task>__<system>__minimax__r1/summary.json：
    - 7 个标准指标取自 meta_bench_scores
    - source_fidelity 取自 source_fidelity_eval_side（评测侧忠实解析法补算）
输出：每系统的 任务×指标 全表 + 每系统在所有任务上的指标均分（缺失/失败计为 N/A，不计入均值）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

METRICS = [
    ("entity_consistency_score", "ECS"),
    ("proxy_hit_rate", "PHR"),
    ("length_score", "LEN"),
    ("completion_rate", "CMP"),
    ("source_fidelity", "SF"),
    ("section_distribution", "SD"),
    ("citation_count", "CD"),
    ("source_balance", "SB"),
]


def _fmt(value: Optional[float]) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "—"


def _read_metrics(bundle: Path) -> Optional[Dict[str, Optional[float]]]:
    summary_file = bundle / "summary.json"
    if not summary_file.exists():
        return None
    data = json.loads(summary_file.read_text(encoding="utf-8"))
    scores = data.get("meta_bench_scores", {})
    sf = data.get("source_fidelity_eval_side", {}).get("source_fidelity")
    out: Dict[str, Optional[float]] = {}
    for key, _ in METRICS:
        out[key] = sf if key == "source_fidelity" else scores.get(key)
    return out


def run(systems: List[str], task_ids: List[str], root_dir: str) -> None:
    root = Path(root_dir)
    # 每系统的指标列值（用于算均分）
    per_system_cols: Dict[str, Dict[str, List[float]]] = {
        s: {key: [] for key, _ in METRICS} for s in systems
    }
    per_system_done: Dict[str, int] = {s: 0 for s in systems}

    for system in systems:
        print(f"\n### {system} —— 任务 × 指标")
        print("| task | " + " | ".join(sn for _, sn in METRICS) + " |")
        print("|" + "---|" * (len(METRICS) + 1))
        for task in task_ids:
            bundle = root / f"{task}__{system}__minimax__r1"
            metrics = _read_metrics(bundle)
            if metrics is None:
                row = " | ".join("—" for _ in METRICS)
                print(f"| {task} | {row} |  (失败/缺失)")
                continue
            per_system_done[system] += 1
            cells = []
            for key, _ in METRICS:
                v = metrics[key]
                cells.append(_fmt(v))
                if isinstance(v, (int, float)):
                    per_system_cols[system][key].append(float(v))
            print(f"| {task} | " + " | ".join(cells) + " |")

    # 均分汇总
    print("\n\n### 每系统在所有任务上的指标均分（N/A 不计入；括号为有效样本数）")
    print("| system | done | " + " | ".join(sn for _, sn in METRICS) + " |")
    print("|" + "---|" * (len(METRICS) + 2))
    for system in systems:
        cells = []
        for key, _ in METRICS:
            vals = per_system_cols[system][key]
            cells.append(f"{mean(vals):.3f}({len(vals)})" if vals else "—(0)")
        print(
            f"| {system} | {per_system_done[system]}/{len(task_ids)} | "
            + " | ".join(cells)
            + " |"
        )

    print("\n指标缩写：ECS 必需术语覆盖 / PHR proxy 点级覆盖 / LEN 长度 / CMP 章节完整 / "
          "SF 源忠实度(评测侧补算) / SD 引用分布 / CD 引用密度 / SB 来源均衡")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 baseline 全任务全指标 + 均分")
    parser.add_argument("--systems", nargs="+", default=["autosurvey", "surveyforge"])
    parser.add_argument("--task-id", nargs="+", required=True)
    parser.add_argument("--root-dir", default="./experiments_out/runs")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.systems, args.task_id, args.root_dir)
