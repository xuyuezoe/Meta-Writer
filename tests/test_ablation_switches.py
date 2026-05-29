"""
消融开关行为单元测试（L1 改造验证）

验证目标：
    逐个确认每个消融开关在对应挂钩点确实改变了系统行为，
    且完整系统（Full）下行为不受影响。

    A1 No DSL          → _update_dsl_injection 注入恒空；_on_section_success 不提取承诺
    A2 No MRSD         → _fixed_local_rewrite_diagnosis 返回 local_rewrite
    A3 No MetaState    → gate_action 无条件放行（即便预算耗尽）
    A4 No SectionPlanner→ _plan_section 返回最简意图且不调用 SectionPlanner
    A6 No DSL Relations → _on_section_success 不调用 process_pending_relations

设计说明：
    通过 __new__ 绕过 __init__ 构造轻量 orchestrator，注入 SimpleNamespace 假件，
    以隔离单一挂钩点的行为，不触网、不依赖真实 LLM。
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from src.core.ablation import AblationConfig
from src.core.meta_state import MetaState
from src.core.plan import PlanState
from src.core.state import GenerationState
from src.orchestrator_v2 import SelfCorrectingOrchestrator


# ── A3 No MetaState：门控 ────────────────────────────────────────────

def test_metastate_gate_blocks_when_budget_exhausted() -> None:
    """完整系统下，预算耗尽应拒绝回退"""
    ms = MetaState(remaining_rollback_budget=0)
    assert ms.gate_action("allow_rollback") is False


def test_no_metastate_gate_always_allows() -> None:
    """A3：no_metastate 时门控无条件放行，即便预算耗尽、EIV 为负"""
    ms = MetaState(
        no_metastate=True,
        remaining_rollback_budget=0,
        expected_intervention_value=-5.0,
        validator_stability_estimate=0.0,
    )
    assert ms.gate_action("allow_rollback") is True
    assert ms.gate_action("trust_validator_major") is True
    assert ms.gate_action("strengthen_dsl_injection") is True


# ── 轻量 orchestrator 构造 ──────────────────────────────────────────

def _make_orchestrator(ablation: AblationConfig) -> SelfCorrectingOrchestrator:
    """通过 __new__ 构造仅含必要假件的 orchestrator"""
    orch = SelfCorrectingOrchestrator.__new__(SelfCorrectingOrchestrator)
    orch.ablation = ablation
    orch.logger = logging.getLogger("test.ablation")
    orch.run_logger = None
    return orch


# ── A1 No DSL：注入与承诺提取 ───────────────────────────────────────

def test_no_dsl_injection_is_empty() -> None:
    """A1：no_dsl 时 dsl_injection 恒为空，且不查询 DSL 条目"""
    orch = _make_orchestrator(AblationConfig(no_dsl=True))

    called = {"get_injectable": False}

    def _fail_get_injectable(**_kwargs):
        called["get_injectable"] = True
        return []

    orch.dsl = SimpleNamespace(get_injectable_entries=_fail_get_injectable)
    state = GenerationState(current_section="s1", progress=0.0, outline={"s1": "Intro"})
    state.dsl_injection = "stale"

    orch._update_dsl_injection(state, "s1", ["s1"], 0)

    assert state.dsl_injection == ""
    # A1 应短路，不应触碰 DSL 条目查询
    assert called["get_injectable"] is False


def test_full_dsl_injection_queries_entries() -> None:
    """完整系统下应正常查询并格式化 DSL 条目"""
    orch = _make_orchestrator(AblationConfig.full())
    orch.dtg = SimpleNamespace(decision_log=[])
    orch.dsl = SimpleNamespace(get_injectable_entries=lambda **_: [])
    state = GenerationState(current_section="s1", progress=0.0, outline={"s1": "Intro"})

    orch._update_dsl_injection(state, "s1", ["s1"], 0)
    # 无可注入条目时为空字符串，但路径已走查询分支（不报错即说明分支正确）
    assert state.dsl_injection == ""


# ── A2 No MRSD：固定诊断 ────────────────────────────────────────────

def test_fixed_local_rewrite_diagnosis() -> None:
    """A2：固定诊断应指向 local_rewrite，无回退目标、不触发重规划"""
    orch = _make_orchestrator(AblationConfig(no_mrsd=True))
    diag = orch._fixed_local_rewrite_diagnosis()
    assert diag.repair_scope == "local_rewrite"
    assert diag.target_section is None
    assert diag.should_rollback() is False
    assert diag.decoding_config.trigger_section_intent_revision is False


# ── A4 No SectionPlanner：最简意图 ──────────────────────────────────

def test_no_planner_returns_minimal_intent_without_calling_planner() -> None:
    """A4：no_planner 时返回最简意图，且不调用 SectionPlanner.plan_section"""
    orch = _make_orchestrator(AblationConfig(no_planner=True))

    def _fail_plan(**_kwargs):
        raise AssertionError("no_planner 下不应调用 SectionPlanner")

    orch.section_planner = SimpleNamespace(plan_section=_fail_plan)
    orch.dsl = SimpleNamespace(compute_memory_trust_level=lambda: 0.9)
    plan_state = PlanState(global_outline={"s1": "Intro"})

    intent = orch._plan_section(
        section_id="s1",
        section_title="Introduction",
        task="write a review",
        plan_state=plan_state,
    )
    assert intent.section_id == "s1"
    assert intent.coverage_requirements == []
    assert intent.commitments_to_maintain == []
    assert intent.word_target is None


# ── A6 No DSL Relations：跳过关系处理 ───────────────────────────────

def _make_success_orchestrator(ablation: AblationConfig, *, extract_calls, relation_calls):
    """构造可观测承诺提取与关系处理调用的 orchestrator"""
    orch = _make_orchestrator(ablation)

    def _extract(**_kwargs):
        extract_calls.append(1)
        return []

    def _process_relations(**_kwargs):
        relation_calls.append(1)
        return {}

    orch.dtg = SimpleNamespace(add_decision=lambda decision: None)
    orch.correction_log = SimpleNamespace(add_success=lambda section_id, attempts: None)
    orch.commitment_extractor = SimpleNamespace(extract=_extract)
    orch.dsl = SimpleNamespace(
        process_pending_relations=_process_relations,
        update_entry_stability=lambda section_id, generated: None,
        compute_memory_trust_level=lambda: 1.0,
        get_active_entries=lambda: [],
        get_open_loops=lambda: [],
        add_entry=lambda entry: None,
    )
    orch.meta_state = SimpleNamespace(
        memory_trust_level=1.0,
        update_contamination_risk=lambda **_: None,
    )
    orch.metric_collector = SimpleNamespace(record_dsl_snapshot=lambda **_: None)
    orch._compute_low_trust_ratio = lambda section_id: 0.0
    orch._log_postprocess_skipped = lambda section_id: None
    orch._print_success = lambda section_id, attempts, tcas: None
    orch._log_dsl_relation_stats = lambda section_id, n, stats: None
    return orch


def _run_success(orch) -> None:
    from src.core.decision import Decision

    state = GenerationState(current_section="s1", progress=0.0, outline={"s1": "Intro"})
    decision = Decision(
        timestamp=1,
        decision_id="d1",
        decision="write",
        reasoning="r",
        expected_effect="e",
        confidence=0.8,
        referenced_sections=[],
        target_section="s1",
    )
    orch._on_section_success(
        section_id="s1",
        content="Some content.",
        decision=decision,
        state=state,
        generated_content={},
        section_queue=["s1"],
        plan_state=PlanState(global_outline={"s1": "Intro"}),
        attempt=0,
        tcas=0.9,
    )


def test_no_dsl_relations_skips_relation_processing_but_keeps_extraction() -> None:
    """A6：跳过关系处理，但仍提取承诺（A6 只关闭关系层）"""
    extract_calls: list[int] = []
    relation_calls: list[int] = []
    orch = _make_success_orchestrator(
        AblationConfig(no_dsl_relations=True),
        extract_calls=extract_calls,
        relation_calls=relation_calls,
    )
    _run_success(orch)
    assert extract_calls == [1], "A6 仍应提取承诺"
    assert relation_calls == [], "A6 应跳过关系处理"


def test_no_dsl_skips_both_extraction_and_relations() -> None:
    """A1：既不提取承诺，也不处理关系"""
    extract_calls: list[int] = []
    relation_calls: list[int] = []
    orch = _make_success_orchestrator(
        AblationConfig(no_dsl=True),
        extract_calls=extract_calls,
        relation_calls=relation_calls,
    )
    _run_success(orch)
    assert extract_calls == [], "A1 不应提取承诺"
    assert relation_calls == [], "A1 不应处理关系"


def test_full_does_both() -> None:
    """完整系统应同时提取承诺并处理关系"""
    extract_calls: list[int] = []
    relation_calls: list[int] = []
    orch = _make_success_orchestrator(
        AblationConfig.full(),
        extract_calls=extract_calls,
        relation_calls=relation_calls,
    )
    _run_success(orch)
    assert extract_calls == [1]
    assert relation_calls == [1]
