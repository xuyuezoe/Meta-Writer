from __future__ import annotations

from pathlib import Path
import re
import unittest

from examples.benchmark_template import (
    DOCUMENT_LEVEL_CONSTRAINT_PREFIX,
    _extract_paragraph_blocks,
    build_benchmark_task_config,
    evaluate_output,
)


class BenchmarkTemplateTests(unittest.TestCase):
    def test_benchmark_constraints_are_marked_as_document_level(self) -> None:
        config = build_benchmark_task_config("med_s001")
        constraints = config["constraints"]

        self.assertIsInstance(constraints, list)
        self.assertTrue(constraints)
        self.assertTrue(
            all(
                isinstance(item, str)
                and item.startswith(DOCUMENT_LEVEL_CONSTRAINT_PREFIX)
                for item in constraints
            )
        )
        self.assertIn("language", config)
        self.assertEqual(config["language"], "en")

    def test_benchmark_task_is_english(self) -> None:
        config = build_benchmark_task_config("med_s001")
        task = config["task"]
        reference = config["reference"]

        self.assertIsInstance(task, str)
        self.assertNotRegex(task, r"[\u4e00-\u9fff]")
        self.assertIn("English", task)
        self.assertIsInstance(reference, dict)
        self.assertEqual(reference["language"], "en")
        self.assertTrue(
            all(
                not re.search(r"[\u4e00-\u9fff]", item)
                for item in reference["constraints"]["must_include"]
            )
        )

    def test_benchmark_source_assets_are_english(self) -> None:
        samples_path = Path("metabench/examples/samples.jsonl")
        samples_text = samples_path.read_text(encoding="utf-8")

        self.assertNotRegex(samples_text, r"[\u4e00-\u9fff]")

    def test_extract_paragraph_blocks_skips_markdown_wrappers(self) -> None:
        text = (
            "## Section One\n\n"
            "First paragraph.\n\n"
            "---\n\n"
            "## Section Two\n\n"
            "Second paragraph.\n\n"
            "Third paragraph."
        )

        self.assertEqual(
            _extract_paragraph_blocks(text),
            ["First paragraph.", "Second paragraph.", "Third paragraph."],
        )

    def test_benchmark_reference_keeps_once_keywords(self) -> None:
        config = build_benchmark_task_config("med_s001")
        reference = config["reference"]

        self.assertIsInstance(reference, dict)
        self.assertIn("once_keywords", reference["constraints"])
        self.assertTrue(reference["constraints"]["once_keywords"])

    def test_evaluate_output_penalizes_large_length_shortfall(self) -> None:
        config = build_benchmark_task_config("med_s001")
        reference = config["reference"]
        constraints = reference["constraints"]
        must_include = constraints["must_include"]
        expected_blocks = constraints["expected_blocks"]

        paragraph = (
            "This review covers "
            + ", ".join(must_include)
            + ". "
            + "analysis " * 40
        ).strip()
        generated_text = "\n\n".join(paragraph for _ in range(expected_blocks))

        evaluation = evaluate_output(generated_text, reference)
        diagnostics = evaluation["diagnostics"]

        self.assertEqual(evaluation["entity_consistency_score"], 1.0)
        self.assertLess(evaluation["length_score"], 0.5)
        self.assertGreater(evaluation["constraint_violation_rate"], 0.5)
        self.assertFalse(diagnostics["length_within_tolerance"])
        self.assertLess(
            diagnostics["response_word_count"],
            diagnostics["required_length_words"],
        )


if __name__ == "__main__":
    unittest.main()
