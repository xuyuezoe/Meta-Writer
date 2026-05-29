"""
分析层单元测试（L3 analysis）

验证目标：
    1. stats：Bonferroni 阈值、Wilcoxon（与 scipy 交叉验证）、Cohen's d、mean_std
    2. aggregate：长表展开、分组均值±标准差、按任务配对序列、CSV 导出
    3. scaling：Δ 指标按词数分档
    4. tables：LaTeX 渲染、显著性标记、下划线转义
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from experiments.analysis import aggregate, scaling, stats, tables


# ── stats ─────────────────────────────────────────────────────────────

def test_bonferroni_alpha() -> None:
    assert stats.bonferroni_alpha(7) == pytest.approx(0.05 / 7)
    assert stats.bonferroni_alpha(5, alpha=0.1) == pytest.approx(0.02)
    with pytest.raises(stats.StatsError):
        stats.bonferroni_alpha(0)


def test_mean_std() -> None:
    mean, std = stats.mean_std([1.0, 2.0, 3.0])
    assert mean == pytest.approx(2.0)
    assert std == pytest.approx(1.0)  # 样本标准差 ddof=1
    # 单样本标准差为 0
    assert stats.mean_std([5.0]) == (5.0, 0.0)


def test_cohens_d_paired_direction() -> None:
    x = [0.9, 0.8, 0.95, 0.85]
    y = [0.3, 0.25, 0.35, 0.28]
    d = stats.cohens_d_paired(x, y)
    assert d > 0  # x 系统性高于 y
    # 差值近似常数 → 标准差极小 → |d| 很大
    assert abs(d) > 2.0


def test_wilcoxon_matches_scipy() -> None:
    """与 scipy 正态近似交叉验证 p 值与统计量（scipy 仅用于测试校验）"""
    scipy_stats = pytest.importorskip("scipy.stats")
    x = [0.92, 0.88, 0.95, 0.90, 0.87, 0.93, 0.91, 0.89, 0.94, 0.86]
    y = [0.80, 0.79, 0.85, 0.78, 0.82, 0.81, 0.83, 0.77, 0.84, 0.76]

    mine = stats.wilcoxon_signed_rank(x, y)
    ref = scipy_stats.wilcoxon(x, y, correction=True, method="approx")

    assert mine.statistic == pytest.approx(float(ref.statistic), abs=1e-6)
    assert mine.p_value == pytest.approx(float(ref.pvalue), abs=1e-3)


def test_wilcoxon_all_zero_diffs() -> None:
    """全零差应返回 p=1（无方向信息）"""
    res = stats.wilcoxon_signed_rank([1.0, 2.0], [1.0, 2.0])
    assert res.n_effective == 0
    assert res.p_value == 1.0


def test_paired_comparison_significant() -> None:
    x = [0.92, 0.88, 0.95, 0.90, 0.87, 0.93, 0.91, 0.89, 0.94, 0.86]
    y = [0.60, 0.59, 0.65, 0.58, 0.62, 0.61, 0.63, 0.57, 0.64, 0.56]
    result = stats.paired_comparison(x, y, num_comparisons=7)
    assert result.mean_diff > 0
    assert result.alpha_corrected == pytest.approx(0.05 / 7)
    assert result.significant is True


# ── aggregate ─────────────────────────────────────────────────────────

def _synthetic_summaries() -> list[dict]:
    """构造跨方法×任务×run 的合成摘要"""
    summaries: list[dict] = []
    data = {
        "full": {"med_s001": [0.03, 0.04], "med_s002": [0.02, 0.03]},
        "no_dsl": {"med_s001": [0.10, 0.12], "med_s002": [0.09, 0.11]},
    }
    for method, per_task in data.items():
        for task_id, values in per_task.items():
            for i, v in enumerate(values, start=1):
                summaries.append({
                    "task_id": task_id,
                    "method": method,
                    "model": "minimax",
                    "run_id": f"r{i}",
                    "status": "completed",
                    "word_count": 4000,
                    "llm_stats": {"total_tokens": 100000, "request_count": 50},
                    "meta_bench_scores": {"constraint_violation_rate": v},
                })
    return summaries


def test_to_long_rows_and_aggregate() -> None:
    summaries = _synthetic_summaries()
    rows = aggregate.to_long_rows(summaries)
    # 每个摘要展开为多指标行（cvr + word_count + total_tokens + tokens_per_word）
    cvr_rows = [r for r in rows if r["metric"] == "constraint_violation_rate"]
    assert len(cvr_rows) == 8  # 2 方法 × 2 任务 × 2 run

    agg = aggregate.aggregate_mean_std(rows)
    full_cvr = [
        r for r in agg
        if r["method"] == "full" and r["metric"] == "constraint_violation_rate"
    ][0]
    assert full_cvr["n"] == 4
    assert full_cvr["mean"] == pytest.approx((0.03 + 0.04 + 0.02 + 0.03) / 4)


def test_paired_series_by_task() -> None:
    summaries = _synthetic_summaries()
    a, b, tasks = aggregate.paired_series_by_task(
        summaries,
        metric="constraint_violation_rate",
        method_a="full",
        method_b="no_dsl",
        model="minimax",
    )
    assert tasks == ["med_s001", "med_s002"]
    # full 的每任务均值应低于 no_dsl
    assert all(av < bv for av, bv in zip(a, b))


def test_write_csv(tmp_path: Path) -> None:
    rows = [{"method": "full", "metric": "cvr", "mean": 0.03, "std": 0.01, "n": 4}]
    out = aggregate.write_csv(rows, str(tmp_path / "agg.csv"))
    assert out.exists()
    with out.open(encoding="utf-8") as handle:
        parsed = list(csv.DictReader(handle))
    assert parsed[0]["method"] == "full"
    assert parsed[0]["mean"] == "0.03"


def test_write_csv_empty_raises() -> None:
    with pytest.raises(ValueError):
        aggregate.write_csv([], "/tmp/should_not_exist.csv")


# ── scaling ───────────────────────────────────────────────────────────

def test_compute_scaling_curve() -> None:
    # 两个任务：短任务 Δ 小，长任务 Δ 大 → 验证按词数分档
    summaries = [
        {"task_id": "short", "method": "full", "model": "m", "run_id": "r1",
         "status": "completed", "meta_bench_scores": {"cvr": 0.05}},
        {"task_id": "short", "method": "no_dsl", "model": "m", "run_id": "r1",
         "status": "completed", "meta_bench_scores": {"cvr": 0.06}},
        {"task_id": "long", "method": "full", "model": "m", "run_id": "r1",
         "status": "completed", "meta_bench_scores": {"cvr": 0.03}},
        {"task_id": "long", "method": "no_dsl", "model": "m", "run_id": "r1",
         "status": "completed", "meta_bench_scores": {"cvr": 0.15}},
    ]
    word_targets = {"short": 3000, "long": 12000}
    curve = scaling.compute_scaling_curve(
        summaries,
        word_targets,
        metric="cvr",
        method_a="no_dsl",  # 注意：此处算 no_dsl - full，使长任务 Δ 更大为正
        method_b="full",
        bins=(3000, 12000),
    )
    by_center = {row["bin_center"]: row for row in curve}
    assert by_center[3000]["mean_delta"] == pytest.approx(0.01, abs=1e-6)
    assert by_center[12000]["mean_delta"] == pytest.approx(0.12, abs=1e-6)
    # 长任务的 Δ 显著大于短任务（scaling 趋势）
    assert by_center[12000]["mean_delta"] > by_center[3000]["mean_delta"]


# ── tables ────────────────────────────────────────────────────────────

def test_significance_marker() -> None:
    assert tables.significance_marker(0.005) == "**"
    assert tables.significance_marker(0.03) == "*"
    assert tables.significance_marker(0.2) == ""


def test_render_results_table_structure() -> None:
    aggregated = [
        {"method": "full", "model": "minimax", "metric": "cvr", "mean": 0.032, "std": 0.007, "n": 3},
        {"method": "no_dsl", "model": "minimax", "metric": "cvr", "mean": 0.105, "std": 0.011, "n": 3},
    ]
    latex = tables.render_results_table(
        aggregated,
        methods=["full", "no_dsl"],
        metrics=["cvr"],
        model="minimax",
        comparisons={("no_dsl", "cvr"): 0.005},
    )
    assert r"\begin{tabular}" in latex
    assert "0.032" in latex
    # no_dsl 应转义下划线，并带强显著标记
    assert r"no\_dsl" in latex
    assert "**" in latex


def test_render_results_table_missing_cell() -> None:
    """缺失组合应渲染为 --，不静默填 0"""
    latex = tables.render_results_table(
        [{"method": "full", "model": "m", "metric": "cvr", "mean": 0.03, "std": 0.01, "n": 3}],
        methods=["full", "autosurvey"],
        metrics=["cvr"],
        model="m",
    )
    assert "--" in latex
