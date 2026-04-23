from __future__ import annotations

import json
import unittest

from src.core.ledger import CommitmentType, ConstraintType, LedgerEntry
from src.memory.discourse_ledger import DiscourseLedger


class FakeLLM:
    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or []
        self.prompts: list[str] = []

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 0,
        **kwargs,
    ) -> str:
        self.prompts.append(prompt)
        if self.responses:
            return self.responses.pop(0)
        return "[]"


class DiscourseLedgerTests(unittest.TestCase):
    def _entry(
        self,
        entry_id: str,
        commitment_type: CommitmentType,
        content: str,
        source_section: str,
        introduced_at: int,
    ) -> LedgerEntry:
        return LedgerEntry(
            entry_id=entry_id,
            commitment_type=commitment_type,
            content=content,
            constraint_type=ConstraintType.STATEFUL,
            source_section=source_section,
            source_decision_id=f"decision_{entry_id}",
            introduced_at=introduced_at,
        )

    def test_overlap_ratio_uses_intersection_over_min_size(self) -> None:
        ratio = DiscourseLedger._overlap_ratio({"低资源环境", "真实世界研究"}, {"低资源环境"})
        self.assertEqual(ratio, 1.0)

    def test_default_gate_threshold_is_tuned_for_runtime_recall(self) -> None:
        ledger = DiscourseLedger()
        self.assertEqual(ledger.candidate_score_threshold, 1.0)

    def test_gate_relation_pair_keeps_medium_overlap_commitment_to_open_loop_in_verify_mode(self) -> None:
        ledger = DiscourseLedger(candidate_score_threshold=1.5)
        new = self._entry(
            "new",
            CommitmentType.COMMITMENT,
            "针对低资源环境开展真实世界研究以补强证据链",
            "s2",
            110,
        )
        old = self._entry(
            "old",
            CommitmentType.OPEN_LOOP,
            "低资源环境真实世界研究证据仍待补强",
            "s1",
            100,
        )

        keep, gate_score, signals = ledger._gate_relation_pair(new, old)

        self.assertTrue(keep)
        self.assertGreaterEqual(gate_score, 1.5)
        self.assertGreater(signals["term_overlap_score"], 0.0)
        self.assertEqual(signals["type_bonus"], 1.0)

    def test_gate_relation_pair_keeps_low_overlap_commitment_to_open_loop_pair_at_runtime_threshold(self) -> None:
        ledger = DiscourseLedger(candidate_score_threshold=1.0)
        new = self._entry(
            "new",
            CommitmentType.COMMITMENT,
            "适应证边界的界定应更加明确地标注证据等级与适用条件",
            "s2",
            110,
        )
        old = self._entry(
            "old",
            CommitmentType.OPEN_LOOP,
            "适应证边界与证据稀疏条件下的适用范围仍待澄清",
            "s1",
            100,
        )

        keep, gate_score, signals = ledger._gate_relation_pair(new, old)

        self.assertTrue(keep)
        self.assertGreaterEqual(signals["term_overlap_score"], 0.1)
        self.assertGreaterEqual(signals["term_score"], 0.5)
        self.assertEqual(signals["type_bonus"], 1.0)
        self.assertGreaterEqual(gate_score, 1.0)

    def test_gate_relation_pair_drops_weak_commitment_to_commitment_pair(self) -> None:
        ledger = DiscourseLedger(candidate_score_threshold=1.0)
        new = self._entry(
            "new",
            CommitmentType.COMMITMENT,
            "适应证边界的界定应更加明确地标注证据等级与适用条件",
            "s2",
            110,
        )
        old = self._entry(
            "old",
            CommitmentType.COMMITMENT,
            "适应证边界需要在证据稀疏条件下明确标注适用范围",
            "s1",
            100,
        )

        keep, gate_score, signals = ledger._gate_relation_pair(new, old)

        self.assertFalse(keep)
        self.assertEqual(signals["term_score"], 0.5)
        self.assertEqual(gate_score, 1.0)

    def test_gate_relation_pair_drops_double_short_noise(self) -> None:
        ledger = DiscourseLedger()
        new = self._entry("new", CommitmentType.COMMITMENT, "待补充", "s2", 110)
        old = self._entry("old", CommitmentType.OPEN_LOOP, "见前文", "s1", 100)

        keep, gate_score, signals = ledger._gate_relation_pair(new, old)

        self.assertFalse(keep)
        self.assertEqual(gate_score, 0.0)
        self.assertEqual(signals["hard_drop"], 1.0)

    def test_add_entry_enqueues_pair_without_calling_llm(self) -> None:
        llm = FakeLLM()
        ledger = DiscourseLedger(llm_client=llm, candidate_score_threshold=1.5)
        old = self._entry(
            "old",
            CommitmentType.OPEN_LOOP,
            "低资源环境真实世界研究证据仍待补强",
            "s1",
            100,
        )
        new = self._entry(
            "new",
            CommitmentType.COMMITMENT,
            "针对低资源环境开展真实世界研究以补强证据链",
            "s2",
            110,
        )

        ledger.add_entry(old)
        ledger.add_entry(new)

        self.assertEqual(llm.prompts, [])
        self.assertEqual(ledger._pending_relation_pairs, [("new", "old")])
        self.assertEqual(ledger._relation_stats_window["raw_pairs_checked"], 1)
        self.assertEqual(ledger._relation_stats_window["pairs_enqueued"], 1)

    def test_add_entry_drop_writes_none_prefilter_cache(self) -> None:
        ledger = DiscourseLedger(candidate_score_threshold=2.0)
        old = self._entry("old", CommitmentType.COMMITMENT, "MRI", "s1", 100)
        new = self._entry("new", CommitmentType.COMMITMENT, "RCT", "s2", 110)

        ledger.add_entry(old)
        ledger.add_entry(new)

        key = ledger._make_pair_key("old", "new")
        self.assertEqual(ledger._relation_result_cache[key], ("none_prefilter", 1.0))
        self.assertEqual(ledger._pending_relation_pairs, [])

    def test_process_pending_relations_writes_edges_and_resolves_open_loop(self) -> None:
        llm = FakeLLM(
            [
                json.dumps(
                    [
                        {
                            "source_id": "commitment",
                            "target_id": "loop",
                            "relation_type": "resolves",
                            "confidence": 0.91,
                        }
                    ]
                )
            ]
        )
        ledger = DiscourseLedger(llm_client=llm, candidate_score_threshold=1.5)
        commitment = self._entry(
            "commitment",
            CommitmentType.COMMITMENT,
            "Captain Alice finally reveals the signal source in the control room",
            "s2",
            120,
        )
        loop = self._entry(
            "loop",
            CommitmentType.OPEN_LOOP,
            "What is the source of the strange signal in the control room?",
            "s1",
            100,
        )

        ledger.add_entry(loop)
        ledger.add_entry(commitment)
        stats = ledger.process_pending_relations(section_id="s2", max_pairs=8, batch_size=4, confidence_threshold=0.5)

        self.assertEqual(stats["pairs_sent_to_llm"], 1)
        self.assertEqual(stats["pairs_resolves"], 1)
        self.assertTrue(loop.is_resolved)
        self.assertEqual(loop.resolved_in, "s2")
        self.assertEqual(ledger._relation_result_cache[("commitment", "loop")], ("resolves", 0.91))

    def test_low_confidence_relation_is_cached_but_not_applied(self) -> None:
        llm = FakeLLM(
            [
                json.dumps(
                    [
                        {
                            "source_id": "b",
                            "target_id": "a",
                            "relation_type": "supports",
                            "confidence": 0.2,
                        }
                    ]
                )
            ]
        )
        ledger = DiscourseLedger(llm_client=llm, candidate_score_threshold=1.5)
        a = self._entry("a", CommitmentType.COMMITMENT, "低资源环境真实世界研究路径需要补强", "s1", 100)
        b = self._entry("b", CommitmentType.COMMITMENT, "针对低资源环境开展真实世界研究以补强路径", "s2", 110)

        ledger.add_entry(a)
        ledger.add_entry(b)
        stats = ledger.process_pending_relations(section_id="s2", confidence_threshold=0.5)

        self.assertEqual(stats["pairs_supports"], 0)
        self.assertEqual(ledger._relation_result_cache[("a", "b")], ("supports", 0.2))
        self.assertEqual(ledger._relations["a"], [])

    def test_process_pending_relations_with_no_llm_clears_queue(self) -> None:
        ledger = DiscourseLedger(llm_client=None, candidate_score_threshold=1.5)
        old = self._entry(
            "old",
            CommitmentType.COMMITMENT,
            "Captain Alice hides the map in the observatory chamber",
            "s1",
            100,
        )
        new = self._entry(
            "new",
            CommitmentType.OPEN_LOOP,
            "Why did Captain Alice hide the map in the observatory chamber?",
            "s2",
            110,
        )

        ledger.add_entry(old)
        ledger.add_entry(new)
        stats = ledger.process_pending_relations(section_id="s2")

        self.assertEqual(stats["processed"], 0)
        self.assertEqual(stats["remaining_queue"], 0)
        self.assertEqual(ledger._pending_relation_pairs, [])

    def test_rollback_cleans_pending_pairs_and_cache(self) -> None:
        llm = FakeLLM(
            [
                json.dumps(
                    [
                        {
                            "source_id": "a",
                            "target_id": "b",
                            "relation_type": "supports",
                            "confidence": 0.7,
                        }
                    ]
                )
            ]
        )
        ledger = DiscourseLedger(llm_client=llm, candidate_score_threshold=1.5)
        a = self._entry("a", CommitmentType.COMMITMENT, "Alice guards the west gate tonight", "s1", 100)
        b = self._entry("b", CommitmentType.COMMITMENT, "Alice guards the west gate to protect the archive", "s2", 110)
        c = self._entry("c", CommitmentType.OPEN_LOOP, "Will Alice keep guarding the west gate tomorrow?", "s3", 120)

        ledger.add_entry(a)
        ledger.add_entry(b)
        ledger.process_pending_relations(section_id="s2")
        ledger.add_entry(c)

        self.assertTrue(ledger._pending_relation_pairs)
        self.assertTrue(ledger._relation_result_cache)

        ledger.rollback_to_section("s1", ["s1", "s2", "s3"])

        self.assertNotIn("b", ledger._entries)
        self.assertNotIn("c", ledger._entries)
        self.assertEqual(ledger._pending_relation_pairs, [])
        self.assertEqual(ledger._pending_relation_keys, set())
        self.assertEqual(ledger._relation_result_cache, {})

    def test_batch_parse_failure_is_recorded_as_none_llm(self) -> None:
        llm = FakeLLM(["not-json"])
        ledger = DiscourseLedger(llm_client=llm, candidate_score_threshold=1.5)
        old = self._entry(
            "old",
            CommitmentType.COMMITMENT,
            "Captain Alice tracks the signal inside the northern tunnel",
            "s1",
            100,
        )
        new = self._entry(
            "new",
            CommitmentType.OPEN_LOOP,
            "Will Captain Alice solve the signal mystery inside the northern tunnel?",
            "s2",
            110,
        )

        ledger.add_entry(old)
        ledger.add_entry(new)
        stats = ledger.process_pending_relations(section_id="s2")

        self.assertEqual(stats["pairs_none_llm"], 1)
        self.assertEqual(ledger._relation_result_cache[("new", "old")], ("none_llm", 0.0))

    def test_generic_phrase_penalty_lowers_priority(self) -> None:
        ledger = DiscourseLedger(candidate_score_threshold=1.0)
        generic = self._entry(
            "generic",
            CommitmentType.COMMITMENT,
            "后续章节将对这些问题做详细比较并提供参考框架",
            "s1",
            100,
        )
        specific = self._entry(
            "specific",
            CommitmentType.COMMITMENT,
            "适应证边界需要在证据稀疏条件下明确标注适用范围",
            "s1",
            101,
        )
        loop = self._entry(
            "loop",
            CommitmentType.OPEN_LOOP,
            "适应证边界与证据稀疏条件下的适用范围仍待澄清",
            "s2",
            110,
        )

        ledger.add_entry(generic)
        ledger.add_entry(specific)
        ledger.add_entry(loop)

        generic_priority = ledger._score_pending_pair_priority("generic", "loop")
        specific_priority = ledger._score_pending_pair_priority("specific", "loop")
        self.assertLess(generic_priority, specific_priority)


if __name__ == "__main__":
    unittest.main()
