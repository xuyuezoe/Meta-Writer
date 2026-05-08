from __future__ import annotations

import sys
import types
import unittest

sys.modules.setdefault("openai", types.ModuleType("openai"))

from src.agents.generator import Generator
from src.agents.section_planner import SectionPlanner
from src.core.decision import Decision
from src.core.meta_state import MetaState
from src.core.plan import SectionIntent
from src.core.state import GenerationState
from src.memory.commitment_extractor import CommitmentExtractor
from src.memory.discourse_ledger import DiscourseLedger
from src.metrics.alignment import AlignmentScorer
from src.validators.online_validator import OnlineValidator


class FakeLLM:
    def __init__(self, response: str = "[]") -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        return self.response


class FakeDTG:
    def add_intent_node(self, **kwargs) -> None:
        self.last_intent = kwargs


class FakeAlignment:
    def compute_dcas(self, decision: Decision, content: str):
        return {"dcas": 0.9}


class EnglishOnlyMigrationTests(unittest.TestCase):
    def test_generator_prompt_is_english_only(self) -> None:
        generator = Generator(llm_client=None)
        state = GenerationState(
            current_section="sec2",
            progress=0.5,
            global_constraints=["Use English only."],
            pending_goals=["Explain the causal mechanism."],
            outline={"sec1": "Intro", "sec2": "Mechanism"},
            generated_sections=["sec1"],
            flagged_issues=["Avoid repeating the introduction."],
            dsl_injection="- Keep the biomarker definition stable.",
        )
        intent = SectionIntent.create(
            section_id="sec2",
            local_goal="Explain the mechanism behind the biomarker shift.",
            scope_boundary="Do not resolve the final clinical recommendation yet.",
            open_loops_to_advance=["Why the biomarker rises in resistant cases"],
            commitments_to_maintain=["The cohort remains low-resource and retrospective"],
            risks_to_avoid=["Do not claim causal certainty without evidence"],
            success_criteria=["The mechanism is clearer and the main conflict remains open"],
            source_dsl_entry_ids=[],
            dsl_trust_at_generation=0.9,
        )

        prompt = generator._build_prompt(state, "Write the mechanism section.", "Recent English content.", intent)

        self.assertIn("Current state:", prompt)
        self.assertIn("Recent content:", prompt)
        self.assertIn("Current task:", prompt)
        self.assertIn("Return a JSON object with the following fields:", prompt)
        self.assertNotIn("当前", prompt)
        self.assertNotIn("本节", prompt)

    def test_section_planner_default_and_prompt_are_english(self) -> None:
        planner = SectionPlanner(FakeLLM("{}"), FakeDTG())

        prompt = planner._build_prompt(
            section_title="Discuss the evidence gap",
            task_description="Write an English review of low-resource diagnostics.",
            dsl_context="- Evidence quality is uneven.",
            section_summaries="sec1: The introduction defines the review scope.",
        )
        default_intent = planner._build_default_intent("sec3", [], 0.8)

        self.assertIn("Create a local plan", prompt)
        self.assertIn("Output format (strict JSON):", prompt)
        self.assertEqual(default_intent.local_goal, "Complete the content for section sec3")
        self.assertIn("does not violate major constraints", default_intent.success_criteria[0])
        self.assertNotIn("完成", default_intent.local_goal)

    def test_section_planner_parses_relaxed_json_and_commitment_alias(self) -> None:
        planner = SectionPlanner(FakeLLM("{}"), FakeDTG())
        raw = (
            '{"local_goal":"Define Alzheimer\\\'s disease in the older-adult care context.",'
            '"scope_boundary":"Do not introduce treatment protocols yet.",'
            '"open_loops_to_advance":["Clarify diagnostic boundary questions"],'
            '"commitments_to_preserve":["Keep the review scholarly and English-only"],'
            '"risks_to_avoid":["Do not resolve later controversies"],'
            '"success_criteria":["The conceptual frame is clear."]}'
        )

        intent = planner._parse_intent(raw, "sec1", [], 0.9)

        self.assertEqual(
            intent.local_goal,
            "Define Alzheimer's disease in the older-adult care context.",
        )
        self.assertEqual(
            intent.commitments_to_maintain,
            ["Keep the review scholarly and English-only"],
        )
        self.assertEqual(
            intent.open_loops_to_advance,
            ["Clarify diagnostic boundary questions"],
        )

    def test_commitment_extractor_prompt_requires_english_content(self) -> None:
        extractor = CommitmentExtractor(FakeLLM())
        prompt = extractor._build_prompt(
            "The trial leaves the dosing schedule unresolved.",
            "The prior section defined the patient cohort.",
        )

        self.assertIn("All content values must be in English.", prompt)
        self.assertIn("Current section content:", prompt)
        self.assertNotIn("当前节内容", prompt)

    def test_discourse_ledger_relation_prompt_and_overlap_are_english_first(self) -> None:
        ledger = DiscourseLedger()
        source = type("Entry", (), {"commitment_type": type("Type", (), {"value": "commitment"})(), "content": "Clarify the dosing schedule for low-resource clinics."})()
        target = type("Entry", (), {"commitment_type": type("Type", (), {"value": "open_loop"})(), "content": "The dosing schedule for low-resource clinics remains unclear."})()
        ledger._entries["a"] = source
        ledger._entries["b"] = target

        prompt = ledger._build_batch_relation_prompt([("a", "b")])
        overlap = ledger._compute_term_overlap_score(source.content, target.content)

        self.assertIn("Determine the relation for each DSL entry pair below.", prompt)
        self.assertIn("Return a JSON array only.", prompt)
        self.assertGreater(overlap, 0.1)
        self.assertNotIn("判断以下", prompt)

    def test_state_prompt_text_is_english(self) -> None:
        state = GenerationState(
            current_section="sec1",
            progress=0.25,
            outline={"sec1": "Intro", "sec2": "Discussion"},
            generated_sections=["sec0"],
            global_constraints=["Use English only."],
            pending_goals=["Introduce the problem scope."],
            flagged_issues=["Do not overclaim efficacy."],
        )

        prompt_text = state.to_prompt()

        self.assertIn("## Current Generation State", prompt_text)
        self.assertIn("## Global Constraints", prompt_text)
        self.assertIn("## Pending Goals", prompt_text)
        self.assertNotIn("当前生成状态", prompt_text)

    def test_alignment_prompt_and_validator_outputs_are_english(self) -> None:
        llm = FakeLLM('{"coverage_score": 0.8, "consistency_score": 0.7, "effectiveness_score": 0.9}')
        scorer = AlignmentScorer(llm)
        decision = Decision(
            timestamp=1,
            decision_id="d1",
            decision="Explain the mechanism.",
            reasoning="The section should connect the biomarker shift to drug resistance.",
            expected_effect="The mechanism becomes clearer.",
            confidence=0.9,
            referenced_sections=[],
            target_section="sec2",
        )

        result = scorer.compute_dcas(decision, "The content explains how resistance changes the biomarker pathway.")

        self.assertEqual(result["score_source"], "parsed")
        self.assertIn("You are a narrative quality evaluator.", llm.prompts[0])
        self.assertNotIn("写作决策", llm.prompts[0])

        validator_llm = FakeLLM("true")
        validator = OnlineValidator(
            llm_client=validator_llm,
            dtg_store=None,
            alignment_scorer=FakeAlignment(),
            meta_state=MetaState(),
        )

        issues = validator._check_format("short")
        satisfied, reason = validator._check_constraint_satisfaction(
            "Avoid overstating certainty.",
            "The wording remains measured and cautious.",
            "sec2",
        )

        self.assertEqual(issues[0].description, "Content is too short (1 words < minimum 200)")
        self.assertTrue(satisfied)
        self.assertIsNone(reason)
        self.assertIn("Reply with only true or false.", validator_llm.prompts[0])
        self.assertNotIn("判断章节内容", validator_llm.prompts[0])


if __name__ == "__main__":
    unittest.main()
