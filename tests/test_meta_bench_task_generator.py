import unittest

from meta_bench.task_generator import (
    build_generation_constraints,
    build_main_task_config,
    build_must_include,
    generate_outline,
    generate_proxy_questions,
    generate_review_task,
)
from meta_bench.schemas import TaskSpec


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
        self.assertEqual(len(task.outline), 7)
        self.assertEqual(len(task.reference["proxy_questions"]), 5)

    def test_generation_constraints_are_runtime_list_strings(self):
        constraints = build_generation_constraints(self.spec)

        self.assertIsInstance(constraints, list)
        self.assertTrue(all(isinstance(item, str) for item in constraints))
        self.assertTrue(any("4200 words" in item for item in constraints))

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

    def test_build_main_task_config_accepts_optional_corpus_dir(self):
        config = build_main_task_config(self.spec, corpus_dir="./data_sample/med_papers")

        self.assertEqual(config["corpus_dir"], "./data_sample/med_papers")

    def test_outline_titles_are_clean_ascii_punctuation(self):
        outline = generate_outline(self.spec)

        for title in outline.values():
            self.assertNotIn("鈥", title)
            self.assertNotIn("бк", title)


if __name__ == "__main__":
    unittest.main()
