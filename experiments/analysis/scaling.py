"""
长度 Scaling 分析：scaling

功能：
    计算 DSL 增益（Full 与 No-DSL 的指标差）随文档目标长度的变化曲线（EXP-V.1）。
    预期结论：差值随目标词数增大而单调增大，证明 MetaWriter 的价值集中在长文本。

设计动机：
    "增益随长度 scaling"是论文的核心论点之一。本模块把按任务配对的 Δ 指标
    归入词数档位，输出可直接绘图/制表的逐档统计。
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from .aggregate import paired_series_by_task
from .stats import mean_std


def compute_scaling_curve(
    summaries: Sequence[Mapping[str, Any]],
    task_word_targets: Mapping[str, int],
    *,
    metric: str,
    method_a: str = "full",
    method_b: str = "no_dsl",
    bins: Sequence[int],
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    计算指标差随目标词数的 scaling 曲线

    参数：
        summaries:         运行摘要序列
        task_word_targets: 任务 ID → 目标总词数（用于分档）
        metric:            指标名（如 "constraint_violation_rate"）
        method_a:          方法 A（默认 "full"）
        method_b:          方法 B（默认 "no_dsl"）
        bins:              词数档位中心序列（如 (3000,4500,...,12800)）
        model:             可选骨干模型过滤

    返回值：
        List[Dict]：每档含 bin_center / n_tasks / mean_delta / std_delta，
            其中 delta = (method_a 值 - method_b 值) 的按任务配对差。

    关键实现细节：
        - 先用 paired_series_by_task 取得按任务对齐的 a、b 值。
        - 每个任务的 Δ = a - b，按其目标词数就近归入最接近的档位中心。
        - 缺少目标词数的任务被跳过（无法分档），不静默归入任意档。
    """
    a_values, b_values, task_ids = paired_series_by_task(
        summaries,
        metric=metric,
        method_a=method_a,
        method_b=method_b,
        model=model,
    )

    # 每档收集 Δ
    bucket: Dict[int, List[float]] = {center: [] for center in bins}
    for a, b, task_id in zip(a_values, b_values, task_ids):
        target = task_word_targets.get(task_id)
        if not isinstance(target, (int, float)):
            continue
        nearest = min(bins, key=lambda center: abs(int(target) - center))
        bucket[nearest].append(a - b)

    curve: List[Dict[str, Any]] = []
    for center in bins:
        deltas = bucket[center]
        if deltas:
            mean_delta, std_delta = mean_std(deltas)
        else:
            mean_delta, std_delta = 0.0, 0.0
        curve.append({
            "bin_center": center,
            "n_tasks": len(deltas),
            "mean_delta": round(mean_delta, 6),
            "std_delta": round(std_delta, 6),
        })
    return curve
