"""
执行层单元测试（L2 runners）

验证目标：
    1. 矩阵构建器产出正确数量与结构的运行单元
    2. run_matrix：断点续跑、token 预算停机、单元故障隔离、进度清单落盘
    3. run_baseline_task（Direct-LLM）端到端：摘要结构正确、可续跑
"""
from __future__ import annotations

from pathlib import Path

import pytest

from experiments.config.backbone import ResolvedBackbone
from experiments.runners import batch_driver
from experiments.runners.batch_driver import (
    RunSpec,
    build_ablation_matrix,
    build_backbone_matrix,
    build_comparison_matrix,
    run_matrix,
)


# ── 矩阵构建器 ────────────────────────────────────────────────────────

def test_build_ablation_matrix_count() -> None:
    specs = build_ablation_matrix(["t1", "t2"], ["full", "a1_no_dsl"], "minimax", 3)
    assert len(specs) == 2 * 2 * 3
    assert all(s.kind == "metawriter" for s in specs)
    assert {s.run_id for s in specs} == {"r1", "r2", "r3"}


def test_build_comparison_matrix_count() -> None:
    specs = build_comparison_matrix(["t1"], ["direct-llm", "autosurvey"], "gpt-4o", 2)
    assert len(specs) == 1 * 2 * 2
    assert all(s.kind == "baseline" for s in specs)


def test_build_backbone_matrix_includes_full_and_direct() -> None:
    specs = build_backbone_matrix(["t1"], ["minimax", "gpt-4o"], 1)
    kinds = sorted((s.kind, s.method) for s in specs)
    assert ("metawriter", "full") in kinds
    assert ("baseline", "direct-llm") in kinds
    # 每模型 2 个单元（full + direct-llm）
    assert len(specs) == 2 * 2


# ── run_matrix：续跑/预算/故障 ───────────────────────────────────────

def test_run_matrix_resume_and_failure_isolation(monkeypatch, tmp_path: Path) -> None:
    """模拟分派：第二个单元失败，其余完成，故障被隔离且整批继续"""
    calls: list[str] = []

    def _fake_dispatch(spec, *, root_dir, overwrite, envs_dir, backbone_cache):
        calls.append(spec.task_id)
        if spec.task_id == "t2":
            raise RuntimeError("boom")
        return {"status": "completed", "llm_stats": {"total_tokens": 100}, "meta_bench_scores": {}}

    monkeypatch.setattr(batch_driver, "_dispatch", _fake_dispatch)

    specs = [
        RunSpec("metawriter", "t1", "full", "minimax", "r1"),
        RunSpec("metawriter", "t2", "full", "minimax", "r1"),
        RunSpec("metawriter", "t3", "full", "minimax", "r1"),
    ]
    report = run_matrix(
        specs,
        root_dir=str(tmp_path / "runs"),
        manifest_dir=str(tmp_path / "manifests"),
    )
    assert report["counts"]["completed"] == 2
    assert report["counts"]["failed"] == 1
    assert report["total_tokens"] == 200
    assert calls == ["t1", "t2", "t3"]
    # 进度清单应已落盘
    assert (tmp_path / "manifests" / "matrix_progress.json").exists()


def test_run_matrix_token_budget_stops(monkeypatch, tmp_path: Path) -> None:
    """累计 token 超预算后应停止后续单元"""

    def _fake_dispatch(spec, *, root_dir, overwrite, envs_dir, backbone_cache):
        return {"status": "completed", "llm_stats": {"total_tokens": 1000}, "meta_bench_scores": {}}

    monkeypatch.setattr(batch_driver, "_dispatch", _fake_dispatch)

    specs = [RunSpec("metawriter", f"t{i}", "full", "minimax", "r1") for i in range(5)]
    report = run_matrix(
        specs,
        root_dir=str(tmp_path / "runs"),
        manifest_dir=str(tmp_path / "manifests"),
        token_budget=2500,
    )
    # 第 1、2 单元各 1000（累计 2000 < 2500 才进入第 3）；第 3 后累计 3000 ≥ 2500 → 第 4 前停机
    assert report["counts"]["completed"] == 3
    assert report["counts"]["stopped_budget"] == 1
    assert report["total_tokens"] == 3000


def test_run_matrix_fail_fast_raises(monkeypatch, tmp_path: Path) -> None:
    def _fake_dispatch(spec, *, root_dir, overwrite, envs_dir, backbone_cache):
        raise RuntimeError("boom")

    monkeypatch.setattr(batch_driver, "_dispatch", _fake_dispatch)
    specs = [RunSpec("metawriter", "t1", "full", "minimax", "r1")]
    with pytest.raises(RuntimeError):
        run_matrix(
            specs,
            root_dir=str(tmp_path / "runs"),
            manifest_dir=str(tmp_path / "manifests"),
            fail_fast=True,
        )


# ── run_baseline_task（Direct-LLM）端到端 ────────────────────────────

class _FakeLLM:
    def generate(self, *, prompt, temperature, max_tokens, log_meta=None):
        return "A complete generated review about the topic. " * 20

    def get_statistics(self):
        return {"total_tokens": 5000, "request_count": 1}


def test_run_baseline_direct_llm_end_to_end(monkeypatch, tmp_path: Path) -> None:
    """Direct-LLM 端到端：用 fake client + 跳过重型评估，校验摘要结构与续跑"""
    from experiments.runners import run_baseline as rb

    # 用 fake client 替代真实 LLMClient，避免触网
    monkeypatch.setattr(rb, "build_llm_client", lambda resolved: _FakeLLM())
    # 跳过重型 MetaBench 评估（嵌入/NLI 模型），聚焦运行器结构
    monkeypatch.setattr(rb, "evaluate_if_possible", lambda text, ref, *, run_status: None)

    resolved = ResolvedBackbone(alias="minimax", model="MiniMax-M2.5", base_url=None, api_key="k")
    summary = rb.run_baseline_task(
        task_id="med_s001",
        system_name="direct-llm",
        model_label="minimax",
        resolved_backbone=resolved,
        run_id="r1",
        root_dir=str(tmp_path / "runs"),
    )

    assert summary["task_id"] == "med_s001"
    assert summary["method"] == "direct-llm"
    assert summary["model"] == "minimax"
    assert summary["status"] == "completed"
    assert summary["llm_stats"]["total_tokens"] == 5000
    assert summary["word_count"] > 0

    # 产物落盘
    bundle = tmp_path / "runs" / "med_s001__direct-llm__minimax__r1"
    assert (bundle / "summary.json").exists()
    assert (bundle / "text.txt").exists()

    # 续跑：第二次调用应复用既有摘要（fake client 不应再次被调用也不报错）
    summary2 = rb.run_baseline_task(
        task_id="med_s001",
        system_name="direct-llm",
        model_label="minimax",
        resolved_backbone=resolved,
        run_id="r1",
        root_dir=str(tmp_path / "runs"),
    )
    assert summary2["provenance"]["run_prefix"] == summary["provenance"]["run_prefix"]
