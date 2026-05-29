"""
RunContext 与 eval_subset 单元测试

验证目标（RunContext）：
    1. run_prefix 命名规范正确
    2. bundle_dir / artifact_path 路径派生正确
    3. provenance 含完整四元组与时间戳
    4. parse_prefix 与 run_prefix 互逆
    5. 非法分量（含下划线 / 空 / 非法字符）显式抛错

验证目标（eval_subset）：
    6. 分层判定边界正确
    7. 分层抽样确定性（同种子同结果）
    8. 容量不足显式抛错
"""
from __future__ import annotations

from pathlib import Path

import pytest

from experiments.config.eval_subset import (
    EVAL_40_COUNTS,
    EvalSubsetError,
    TaskPoolEntry,
    classify_length,
    deterministic_halve,
    stratified_sample,
)
from experiments.config.run_context import RunContext, RunContextError


# ── RunContext ────────────────────────────────────────────────────────

def test_run_prefix_naming() -> None:
    """run_prefix 应为 {task}__{method}__{model}__{run}（双下划线分隔）"""
    ctx = RunContext(task_id="med_s017", method="full", model="minimax", run_id="r1")
    assert ctx.run_prefix == "med_s017__full__minimax__r1"


def test_bundle_dir_and_artifact_path() -> None:
    """产物目录与产物路径应正确派生"""
    ctx = RunContext(
        task_id="med_s001",
        method="no_dsl",
        model="gpt-4o",
        run_id="r2",
        root_dir="/tmp/runs",
    )
    assert ctx.bundle_dir == Path("/tmp/runs/med_s001__no_dsl__gpt-4o__r2")
    assert ctx.artifact_path("summary.json") == Path(
        "/tmp/runs/med_s001__no_dsl__gpt-4o__r2/summary.json"
    )


def test_provenance_contains_quadruple() -> None:
    """provenance 应含完整四元组与附加 details"""
    ctx = RunContext(task_id="med_s017", method="full", model="minimax", run_id="r1")
    record = ctx.provenance(extra={"no_dsl": False})
    assert record["task_id"] == "med_s017"
    assert record["method"] == "full"
    assert record["model"] == "minimax"
    assert record["run_id"] == "r1"
    assert record["run_prefix"] == "med_s017__full__minimax__r1"
    assert "created_at" in record
    assert record["details"] == {"no_dsl": False}


def test_parse_prefix_roundtrip() -> None:
    """parse_prefix 应与 run_prefix 互逆"""
    ctx = RunContext(task_id="med_s017", method="full", model="gpt-4o", run_id="r3")
    parsed = RunContext.parse_prefix(ctx.run_prefix)
    assert parsed.task_id == "med_s017"
    assert parsed.method == "full"
    assert parsed.model == "gpt-4o"
    assert parsed.run_id == "r3"


def test_single_underscore_in_component_allowed() -> None:
    """分量内含单下划线应被允许（task_id/method 本身含单下划线）"""
    ctx = RunContext(task_id="med_s017", method="no_dsl", model="minimax", run_id="r1")
    assert ctx.run_prefix == "med_s017__no_dsl__minimax__r1"
    # 反解析应还原原始分量
    parsed = RunContext.parse_prefix(ctx.run_prefix)
    assert parsed.task_id == "med_s017"
    assert parsed.method == "no_dsl"


def test_double_underscore_in_component_rejected() -> None:
    """分量内含双下划线应被拒绝（它是字段分隔符）"""
    with pytest.raises(RunContextError):
        RunContext(task_id="med__s017", method="full", model="minimax", run_id="r1")


def test_empty_component_rejected() -> None:
    """空分量应被拒绝"""
    with pytest.raises(RunContextError):
        RunContext(task_id="", method="full", model="minimax", run_id="r1")


def test_parse_prefix_wrong_arity_raises() -> None:
    """非四段前缀应抛错"""
    with pytest.raises(RunContextError):
        RunContext.parse_prefix("only_three_parts")


# ── eval_subset ───────────────────────────────────────────────────────

def test_classify_length_boundaries() -> None:
    """长度分层边界判定应符合 §2.2 定义"""
    assert classify_length(3000) == "short"
    assert classify_length(5000) == "short"
    assert classify_length(5001) == "medium"
    assert classify_length(8000) == "medium"
    assert classify_length(8001) == "long"
    assert classify_length(12800) == "long"


def _synthetic_pool() -> list[TaskPoolEntry]:
    """构造一个满足 40 任务配额的合成任务池"""
    pool: list[TaskPoolEntry] = []
    # Short 20 个、Medium 15 个、Long 20 个（均超过配额，便于测试抽样）
    for i in range(20):
        pool.append(TaskPoolEntry(task_id=f"short_{i:03d}", target_words=4000))
    for i in range(15):
        pool.append(TaskPoolEntry(task_id=f"medium_{i:03d}", target_words=6500))
    for i in range(20):
        pool.append(TaskPoolEntry(task_id=f"long_{i:03d}", target_words=10000))
    return pool


def test_stratified_sample_is_deterministic() -> None:
    """同种子同输入应得到完全相同的抽样结果"""
    pool = _synthetic_pool()
    first = stratified_sample(pool, EVAL_40_COUNTS, seed=42)
    second = stratified_sample(pool, EVAL_40_COUNTS, seed=42)
    assert first == second
    assert len(first) == 40


def test_stratified_sample_respects_quota() -> None:
    """抽样结果应满足各分层配额"""
    pool = _synthetic_pool()
    selected = set(stratified_sample(pool, EVAL_40_COUNTS, seed=42))
    short = {t for t in selected if t.startswith("short_")}
    medium = {t for t in selected if t.startswith("medium_")}
    long = {t for t in selected if t.startswith("long_")}
    assert len(short) == 15
    assert len(medium) == 10
    assert len(long) == 15


def test_stratified_sample_insufficient_raises() -> None:
    """分层容量不足应显式抛错"""
    tiny_pool = [TaskPoolEntry(task_id=f"short_{i}", target_words=4000) for i in range(3)]
    with pytest.raises(EvalSubsetError):
        stratified_sample(tiny_pool, EVAL_40_COUNTS, seed=42)


def test_deterministic_halve() -> None:
    """确定性取半应稳定且取半数"""
    ids = [f"t_{i:03d}" for i in range(40)]
    first = deterministic_halve(ids, seed=42)
    second = deterministic_halve(ids, seed=42)
    assert first == second
    assert len(first) == 20
