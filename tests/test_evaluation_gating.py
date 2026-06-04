"""
评估门控单元测试（方案一）

验证目标：
    1. degraded 运行同样产出真实 metric_scores，并标记 generation_degraded=True
    2. completed 运行标记 generation_degraded=False
    3. 缺失 reference 时返回 None（不评估）

设计背景：
    旧实现对 degraded 运行返回 {"status": "skipped"}，导致只要有 1 节降级
    就丢失全部指标，消融实验无法量化差异。方案一解耦"评估完成"与"生成降级"，
    始终评分并显式标记。本测试锁定该行为，防止回归。
"""
from __future__ import annotations

import meta_bench
from experiments.runners._common import (
    evaluate_if_possible,
    extract_ordered_metric_scores,
)


def _fake_evaluate_meta_bench(final_text, reference):
    """构造一个固定的 completed 评估结果，隔离对真实 LLM judge 的依赖"""
    return {
        "status": "completed",
        "framework": "meta_bench",
        "task_id": reference.get("task_id"),
        "metric_scores": {
            "entity_consistency_score": 0.8,
            "length_score": 0.6,
        },
        "dimensions": {},
    }


def test_degraded_run_still_scored_and_marked(monkeypatch) -> None:
    """degraded：仍产出真实分数，generation_degraded=True，评估自身 status 仍为 completed"""
    monkeypatch.setattr(meta_bench, "evaluate_meta_bench", _fake_evaluate_meta_bench)

    evaluation = evaluate_if_possible("text", {"task_id": "t1"}, run_status="degraded")

    assert evaluation is not None
    assert evaluation["status"] == "completed"        # 评估本身完成
    assert evaluation["generation_degraded"] is True  # 生成降级被正交标记
    assert evaluation["run_status"] == "degraded"

    scores = extract_ordered_metric_scores(evaluation)
    # 关键断言：不再为空字典
    assert scores["entity_consistency_score"] == 0.8
    assert scores["length_score"] == 0.6


def test_completed_run_marked_not_degraded(monkeypatch) -> None:
    """completed：generation_degraded=False，分数正常提取"""
    monkeypatch.setattr(meta_bench, "evaluate_meta_bench", _fake_evaluate_meta_bench)

    evaluation = evaluate_if_possible("text", {"task_id": "t1"}, run_status="completed")

    assert evaluation["generation_degraded"] is False
    assert evaluation["run_status"] == "completed"
    assert extract_ordered_metric_scores(evaluation)["entity_consistency_score"] == 0.8


def test_no_reference_returns_none(monkeypatch) -> None:
    """缺失 reference：返回 None，不触发评估"""
    monkeypatch.setattr(meta_bench, "evaluate_meta_bench", _fake_evaluate_meta_bench)

    assert evaluate_if_possible("text", None, run_status="completed") is None
