from __future__ import annotations

import logging
import sys
import types
import unittest
from types import SimpleNamespace

sys.modules.setdefault("openai", types.ModuleType("openai"))

from src.core.decision import Decision
from src.core.ledger import CommitmentType, ConstraintType, LedgerEntry
from src.core.plan import PlanState
from src.core.state import GenerationState
from src.orchestrator_v2 import SelfCorrectingOrchestrator


class FakeDSL:
    def __init__(self) -> None:
        self.added_entries: list[LedgerEntry] = []
        self.process_calls: list[tuple[str | None, int, int, float]] = []
        self.stability_updates: list[tuple[str, list[str]]] = []

    def add_entry(self, entry: LedgerEntry) -> None:
        self.added_entries.append(entry)

    def process_pending_relations(
        self,
        section_id: str | None = None,
        max_pairs: int = 0,
        batch_size: int = 0,
        confidence_threshold: float = 0.0,
    ):
        self.process_calls.append((section_id, max_pairs, batch_size, confidence_threshold))
        return {
            "raw_pairs_checked": 2,
            "pairs_dedup_skipped": 0,
            "pairs_prefilter_none": 1,
            "pairs_cache_hit": 0,
            "pairs_enqueued": 1,
            "pairs_sent_to_llm": 1,
            "pairs_none_llm": 0,
            "pairs_supports": 1,
            "pairs_conflicts": 0,
            "pairs_resolves": 0,
            "remaining_queue": 0,
            "time_cost_ms": 12,
        }

    def update_entry_stability(self, section_id: str, generated_sections: list[str]) -> None:
        self.stability_updates.append((section_id, list(generated_sections)))

    def compute_memory_trust_level(self) -> float:
        return 0.82

    def get_active_entries(self):
        return list(self.added_entries)

    def get_open_loops(self):
        return [e for e in self.added_entries if e.commitment_type == CommitmentType.OPEN_LOOP]


class OrchestratorDslRelationTests(unittest.TestCase):
    def _entry(self, entry_id: str, commitment_type: CommitmentType) -> LedgerEntry:
        return LedgerEntry(
            entry_id=entry_id,
            commitment_type=commitment_type,
            content=f"{entry_id} content about Captain Alice and the northern tunnel",
            constraint_type=ConstraintType.STATEFUL,
            source_section="sec1",
            source_decision_id="decision-1",
            introduced_at=100,
        )

    def test_on_section_success_processes_relations_once_after_all_entries_added(self) -> None:
        orchestrator = SelfCorrectingOrchestrator.__new__(SelfCorrectingOrchestrator)
        orchestrator.logger = logging.getLogger("test.orchestrator.dsl")
        orchestrator.dtg = SimpleNamespace(add_decision=lambda decision: None)
        orchestrator.correction_log = SimpleNamespace(add_success=lambda section_id, attempts: None)
        orchestrator.commitment_extractor = SimpleNamespace(
            extract=lambda **_: [
                self._entry("entry1", CommitmentType.COMMITMENT),
                self._entry("entry2", CommitmentType.OPEN_LOOP),
            ]
        )
        orchestrator.dsl = FakeDSL()
        orchestrator.meta_state = SimpleNamespace(
            memory_trust_level=1.0,
            update_contamination_risk=lambda **kwargs: None,
        )
        orchestrator.metric_collector = SimpleNamespace(record_dsl_snapshot=lambda **kwargs: None)
        orchestrator.run_logger = None
        orchestrator._compute_low_trust_ratio = lambda section_id: 0.0
        orchestrator._log_postprocess_skipped = lambda section_id: None
        orchestrator._print_success = lambda section_id, attempts, dcas: None

        state = GenerationState(current_section="sec1", progress=0.0, outline={"sec1": "Intro"})
        generated_content: dict[str, str] = {}
        plan_state = PlanState(global_outline={"sec1": "Intro"})
        decision = Decision(
            timestamp=1,
            decision_id="decision-1",
            decision="write intro",
            reasoning="reason",
            expected_effect="effect",
            confidence=0.8,
            referenced_sections=[],
            target_section="sec1",
        )

        orchestrator._on_section_success(
            section_id="sec1",
            content="Captain Alice enters the northern tunnel.",
            decision=decision,
            state=state,
            generated_content=generated_content,
            section_queue=["sec1"],
            plan_state=plan_state,
            attempt=0,
            dcas=0.93,
        )

        self.assertEqual([entry.entry_id for entry in orchestrator.dsl.added_entries], ["entry1", "entry2"])
        self.assertEqual(
            orchestrator.dsl.process_calls,
            [(
                "sec1",
                SelfCorrectingOrchestrator.DSL_RELATION_MAX_PAIRS_PER_SECTION,
                SelfCorrectingOrchestrator.DSL_RELATION_BATCH_SIZE,
                SelfCorrectingOrchestrator.DSL_RELATION_MIN_CONFIDENCE,
            )],
        )
        self.assertEqual(orchestrator.dsl.stability_updates, [("sec1", ["sec1"])])
        self.assertAlmostEqual(orchestrator.meta_state.memory_trust_level, 0.82)


if __name__ == "__main__":
    unittest.main()
