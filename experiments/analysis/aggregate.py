"""
结果聚合：aggregate

功能：
    扫描 experiments_out/runs 下所有运行的 summary.json，聚合为：
        1. 长表（每个 run × 每个指标一行）——便于通用分组与导出。
        2. 分组均值±标准差表——论文主表的数据来源。
        3. 配对序列——供 stats 做配对显著性检验。

设计动机：
    summary.json 是各运行器产出的唯一标准数据源。聚合层只读不写运行产物，
    把"分散的单次结果"还原为"方法 × 模型 × 指标"的可比结构。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .stats import mean_std


def load_run_summaries(root_dir: str = "./experiments_out/runs") -> List[Dict[str, Any]]:
    """
    扫描并加载所有运行摘要

    参数：
        root_dir: 运行产物根目录

    返回值：
        List[Dict]：所有可解析的 summary.json 内容；目录不存在时返回空列表。

    关键实现细节：
        逐个 <run>/summary.json 读取；JSON 非法的文件被跳过并不静默吞掉——
        会附在返回项的同名 _parse_error 中以便排查（此处选择跳过损坏文件，
        因为聚合应对部分损坏鲁棒，但保留可观测性）。
    """
    root = Path(root_dir)
    if not root.exists():
        return []

    summaries: List[Dict[str, Any]] = []
    for summary_file in sorted(root.glob("*/summary.json")):
        try:
            data = json.loads(summary_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            summaries.append({
                "_parse_error": str(exc),
                "_source": str(summary_file),
            })
            continue
        if isinstance(data, dict):
            data.setdefault("_source", str(summary_file))
            summaries.append(data)
    return summaries


def _iter_metrics(summary: Mapping[str, Any]) -> Dict[str, float]:
    """
    从单个摘要提取所有数值指标（MetaBench 分数 + 效率指标）

    返回值：
        Dict[str, float]：指标名 → 值。包含 meta_bench_scores 全部指标，
            以及派生效率指标 total_tokens / word_count / tokens_per_word。
    """
    metrics: Dict[str, float] = {}

    scores = summary.get("meta_bench_scores")
    if isinstance(scores, Mapping):
        for name, value in scores.items():
            if isinstance(value, (int, float)):
                metrics[str(name)] = float(value)

    word_count = summary.get("word_count")
    if isinstance(word_count, (int, float)):
        metrics["word_count"] = float(word_count)

    llm_stats = summary.get("llm_stats")
    if isinstance(llm_stats, Mapping):
        total_tokens = llm_stats.get("total_tokens")
        if isinstance(total_tokens, (int, float)):
            metrics["total_tokens"] = float(total_tokens)
            if isinstance(word_count, (int, float)) and word_count:
                metrics["tokens_per_word"] = float(total_tokens) / float(word_count)

    return metrics


def to_long_rows(summaries: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """
    把摘要列表展开为长表行

    参数：
        summaries: 运行摘要序列

    返回值：
        List[Dict]：每行含 task_id / method / model / run_id / status / metric / value
    """
    rows: List[Dict[str, Any]] = []
    for summary in summaries:
        if "_parse_error" in summary:
            continue
        base = {
            "task_id": summary.get("task_id", ""),
            "method": summary.get("method", ""),
            "model": summary.get("model", ""),
            "run_id": summary.get("run_id", ""),
            "status": summary.get("status", ""),
        }
        for metric_name, value in _iter_metrics(summary).items():
            row = dict(base)
            row["metric"] = metric_name
            row["value"] = value
            rows.append(row)
    return rows


def aggregate_mean_std(
    long_rows: Sequence[Mapping[str, Any]],
    *,
    group_keys: Sequence[str] = ("method", "model", "metric"),
) -> List[Dict[str, Any]]:
    """
    按指定键分组聚合均值±标准差

    参数:
        long_rows:  长表行
        group_keys: 分组键（默认按方法×模型×指标聚合，跨 run 与 task）

    返回值：
        List[Dict]：每组含分组键 + mean / std / n，按分组键排序。
    """
    buckets: Dict[Tuple[Any, ...], List[float]] = {}
    for row in long_rows:
        key = tuple(row.get(k, "") for k in group_keys)
        value = row.get("value")
        if isinstance(value, (int, float)):
            buckets.setdefault(key, []).append(float(value))

    aggregated: List[Dict[str, Any]] = []
    for key in sorted(buckets.keys(), key=lambda k: tuple(str(x) for x in k)):
        values = buckets[key]
        mean, std = mean_std(values)
        record = {k: key[i] for i, k in enumerate(group_keys)}
        record["mean"] = round(mean, 6)
        record["std"] = round(std, 6)
        record["n"] = len(values)
        aggregated.append(record)
    return aggregated


def paired_series_by_task(
    summaries: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    method_a: str,
    method_b: str,
    model: Optional[str] = None,
) -> Tuple[List[float], List[float], List[str]]:
    """
    构建按任务配对的两方法指标序列（供配对检验）

    功能：
        对每个任务，分别对 method_a / method_b 在该任务上的多次运行取均值，
        然后只保留两方法都出现的任务，按 task_id 对齐。

    参数：
        metric:   指标名（如 "constraint_violation_rate"）
        method_a: 方法 A（如 "full"）
        method_b: 方法 B（如 "no_dsl" 或 "autosurvey"）
        model:    可选骨干模型过滤（None 表示不限）

    返回值：
        Tuple[List[float], List[float], List[str]]：(A 值, B 值, 对齐的 task_id)
    """
    def _collect(method: str) -> Dict[str, List[float]]:
        per_task: Dict[str, List[float]] = {}
        for summary in summaries:
            if summary.get("method") != method:
                continue
            if model is not None and summary.get("model") != model:
                continue
            metrics = _iter_metrics(summary)
            if metric not in metrics:
                continue
            task_id = str(summary.get("task_id", ""))
            per_task.setdefault(task_id, []).append(metrics[metric])
        return per_task

    a_by_task = _collect(method_a)
    b_by_task = _collect(method_b)

    common_tasks = sorted(set(a_by_task.keys()) & set(b_by_task.keys()))
    a_values: List[float] = []
    b_values: List[float] = []
    for task_id in common_tasks:
        a_values.append(sum(a_by_task[task_id]) / len(a_by_task[task_id]))
        b_values.append(sum(b_by_task[task_id]) / len(b_by_task[task_id]))
    return a_values, b_values, common_tasks


def write_csv(rows: Sequence[Mapping[str, Any]], path: str) -> Path:
    """
    将行数据写为 CSV

    参数：
        rows: 行序列（每行为字典，键作为列）
        path: 输出 CSV 路径

    返回值：
        Path：写出的文件路径

    异常：
        ValueError：rows 为空时抛出（无法推断列）。
    """
    if not rows:
        raise ValueError("[聚合导出失败] 无数据行可写入 CSV")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 列顺序：以首行键为基准，合并所有行可能出现的额外键
    fieldnames: List[str] = list(rows[0].keys())
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return out_path
