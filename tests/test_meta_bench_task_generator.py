import unittest
from unittest.mock import patch

from meta_bench.schemas import TaskSpec
from meta_bench.section_budget import build_section_budget
from meta_bench.task_generator import (
    build_generation_constraints,
    build_main_task_config,
    build_must_include,
    generate_outline,
    generate_proxy_questions,
    generate_review_task,
)


class TaskGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.spec = TaskSpec(
            task_id="med_s001",
            topic="acute coronary syndrome",
            domain="cardiovascular medicine",
            target_words=4200,
            expected_sections=7,
            practice_context="adult inpatient",
            organizer="classification framework",
            focus_points=["mechanism", "hemodynamics", "evidence integration"],
        )

    def test_build_must_include_deduplicates_required_keywords(self):
        keywords = build_must_include(self.spec)

        self.assertIn("scope", keywords)
        self.assertIn("acute coronary syndrome", keywords)
        self.assertIn("classification framework", keywords)
        self.assertIn("adult inpatient", keywords)
        self.assertEqual(len(keywords), len(set(keyword.casefold() for keyword in keywords)))

    def test_generate_outline_has_expected_section_count_and_roles(self):
        outline = generate_outline(self.spec)

        self.assertEqual(len(outline), 7)
        self.assertIn("Scope", outline["sec1"])
        self.assertIn("Organizing framework", outline["sec2"])
        self.assertIn("Limitations", outline["sec6"])
        self.assertIn("Future work", outline["sec7"])

    def test_generate_proxy_questions_extend_from_topic(self):
        questions = generate_proxy_questions(self.spec)

        self.assertEqual(len(questions), 5)
        self.assertEqual(questions[0].qid, "q_scope")
        self.assertIn("acute coronary syndrome", questions[0].question)
        self.assertIn("classification framework", questions[1].question)
        self.assertTrue(any("hemodynamics" in point for point in questions[2].required_points))

    def test_generate_review_task_returns_prompt_reference_and_proxy_questions(self):
        task = generate_review_task(self.spec)

        self.assertEqual(task.task_id, "med_s001")
        self.assertIn("4200-word", task.prompt)
        self.assertEqual(task.constraints["required_length_words"], 4200)
        self.assertEqual(task.constraints["body_target_words"], 4200)
        self.assertEqual(task.constraints["total_target_words"], 4200)
        self.assertIn("section_word_targets", task.constraints)
        self.assertIn("section_roles", task.constraints)
        self.assertIn("required_sections", task.constraints)
        self.assertEqual(set(task.constraints["section_word_targets"]), set(task.outline))
        self.assertEqual(sum(task.constraints["section_word_targets"].values()), 4200)
        self.assertEqual(len(task.outline), 7)
        self.assertEqual(len(task.reference["proxy_questions"]), 5)

    def test_generation_constraints_are_runtime_list_strings(self):
        constraints = build_generation_constraints(self.spec)

        self.assertIsInstance(constraints, list)
        self.assertTrue(all(isinstance(item, str) for item in constraints))
        self.assertTrue(
            any(
                "4200 words overall, excluding the final references list" in item
                for item in constraints
            )
        )

    def test_build_main_task_config_matches_main_runtime_shape(self):
        config = build_main_task_config(self.spec)

        self.assertIn("task", config)
        self.assertIn("constraints", config)
        self.assertIn("outline", config)
        self.assertIn("session_name", config)
        self.assertIn("reference", config)
        self.assertIsInstance(config["task"], str)
        self.assertIsInstance(config["constraints"], list)
        self.assertIsInstance(config["outline"], dict)
        self.assertIsInstance(config["session_name"], str)
        self.assertIsInstance(config["reference"], dict)
        reference_constraints = config["reference"]["constraints"]
        self.assertEqual(reference_constraints["body_target_words"], 4200)
        self.assertEqual(set(reference_constraints["section_word_targets"]), set(config["outline"]))
        self.assertEqual(sum(reference_constraints["section_word_targets"].values()), 4200)

    def test_build_main_task_config_accepts_optional_corpus_dir(self):
        config = build_main_task_config(self.spec, corpus_dir="./data_sample/med_papers")

        self.assertEqual(config["corpus_dir"], "./data_sample/med_papers")

    def test_body_target_words_override_changes_prompt_and_section_budgets(self):
        spec = TaskSpec(
            task_id="med_override",
            topic="chemokines in alopecia areata",
            domain="immunodermatology",
            target_words=6083,
            body_target_words=2916,
            expected_sections=6,
            practice_context="clinical dermatology decision-making",
            organizer="chemokine pathway framework",
            focus_points=[
                "Th1-associated chemokines",
                "Th2-associated chemokines",
                "blood and skin biomarker patterns",
                "therapeutic implications",
            ],
        )

        task = generate_review_task(spec)

        self.assertIn("2916-word", task.prompt)
        self.assertEqual(task.constraints["required_length_words"], 2916)
        self.assertEqual(task.constraints["body_target_words"], 2916)
        self.assertEqual(task.constraints["total_target_words"], 6083)
        self.assertEqual(sum(task.constraints["section_word_targets"].values()), 2916)
        self.assertEqual(
            task.constraints["required_sections"],
            ["introduction", "main_body", "limitations_gaps"],
        )

    def test_outline_override_recomputes_section_budgets_from_override(self):
        outline_override = {
            "sec1": "Scope, disease context, and chemokine terminology in alopecia areata",
            "sec2": "Chemokine-pathway framework and hair-follicle immune privilege collapse",
            "sec3": "Evidence base and measurement strategies across blood and skin studies",
            "sec4": "Chemokine signatures in alopecia areata across Th1, Th2, and related pathways",
            "sec5": "Biomarker value and therapeutic implications for clinical dermatology",
            "sec6": "Limitations, heterogeneity, and future research priorities",
        }
        spec = TaskSpec(
            task_id="med_override_outline",
            topic="chemokines in alopecia areata",
            domain="immunodermatology",
            target_words=6083,
            body_target_words=2916,
            expected_sections=6,
            practice_context="clinical dermatology decision-making",
            organizer="chemokine pathway framework",
            focus_points=[
                "Th1-associated chemokines",
                "Th2-associated chemokines",
                "blood and skin biomarker patterns",
                "therapeutic implications",
            ],
        )

        config = build_main_task_config(spec, outline_override=outline_override)

        reference_constraints = config["reference"]["constraints"]
        self.assertEqual(config["outline"], outline_override)
        self.assertEqual(config["reference"]["outline"], outline_override)
        self.assertEqual(reference_constraints["section_roles"]["sec2"], "evidence_synthesis")
        self.assertEqual(reference_constraints["section_roles"]["sec5"], "discussion")
        self.assertEqual(
            reference_constraints["required_sections"],
            ["introduction", "main_body", "limitations_gaps"],
        )
        self.assertEqual(sum(reference_constraints["section_word_targets"].values()), 2916)
        self.assertGreater(reference_constraints["section_word_targets"]["sec2"], 300)
        self.assertGreater(
            reference_constraints["section_word_targets"]["sec2"],
            reference_constraints["section_word_targets"]["sec6"],
        )

    def test_true_taxonomy_titles_still_keep_taxonomy_role(self):
        outline_override = {
            "sec1": "Scope and terminology",
            "sec2": "Classification framework, nomenclature, and phenotype categories",
            "sec3": "Evidence synthesis across cohorts and trials",
            "sec4": "Limitations and future research priorities",
        }
        spec = TaskSpec(
            task_id="taxonomy_guardrail",
            topic="immune phenotypes in chronic disease",
            domain="translational medicine",
            target_words=2400,
            body_target_words=2400,
            expected_sections=4,
            practice_context="specialist clinical decision-making",
            organizer="classification framework",
            focus_points=["phenotype categories", "evidence synthesis"],
        )

        config = build_main_task_config(spec, outline_override=outline_override)
        reference_constraints = config["reference"]["constraints"]

        self.assertEqual(reference_constraints["section_roles"]["sec2"], "taxonomy")

    def test_six_slot_priors_can_override_role_budget_for_six_section_tasks(self):
        outline_override = {
            "sec1": "Scope, disease context, and chemokine terminology in alopecia areata",
            "sec2": "Chemokine-pathway framework and hair-follicle immune privilege collapse",
            "sec3": "Evidence base and measurement strategies across blood and skin studies",
            "sec4": "Chemokine signatures in alopecia areata across Th1, Th2, and related pathways",
            "sec5": "Biomarker value and therapeutic implications for clinical dermatology",
            "sec6": "Limitations, heterogeneity, and future research priorities",
        }
        mocked_priors = {
            "scope_context": 0.22,
            "framework_mechanism": 0.16,
            "evidence_methods": 0.18,
            "findings_synthesis": 0.19,
            "implications_discussion": 0.16,
            "limitations_future": 0.09,
        }

        with patch("meta_bench.section_budget.load_six_slot_priors", return_value=mocked_priors):
            budget = build_section_budget(
                task_id="six_slot_budget",
                body_target_words=3000,
                outline=outline_override,
            )

        self.assertEqual(sum(budget.section_word_targets.values()), 3000)
        self.assertEqual(budget.section_word_targets["sec1"], 660)
        self.assertEqual(budget.section_word_targets["sec2"], 480)
        self.assertEqual(budget.section_word_targets["sec3"], 540)
        self.assertEqual(budget.section_word_targets["sec4"], 570)
        self.assertEqual(budget.section_word_targets["sec5"], 480)
        self.assertEqual(budget.section_word_targets["sec6"], 270)
        self.assertEqual(budget.role_base_shares, {})

    def test_outline_titles_are_clean_ascii_punctuation(self):
        outline = generate_outline(self.spec)

        for title in outline.values():
            self.assertNotIn("闁", title)
            self.assertNotIn("閳", title)
            self.assertNotIn("鏂", title)


if __name__ == "__main__":
    unittest.main()
