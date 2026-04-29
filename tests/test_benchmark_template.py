from __future__ import annotations

import copy
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

    def test_benchmark_outline_uses_review_sections_with_word_budgets(self) -> None:
        config = build_benchmark_task_config("med_s003")
        outline = config["outline"]

        self.assertIsInstance(outline, dict)
        self.assertGreaterEqual(len(outline), 6)
        self.assertTrue(
            all(
                isinstance(title, str)
                and "target about" in title
                and "words" in title
                for title in outline.values()
            )
        )

    def test_short_benchmark_outline_keeps_closing_section(self) -> None:
        config = build_benchmark_task_config("med_s381")
        outline_titles = list(config["outline"].values())

        self.assertGreaterEqual(len(outline_titles), 4)
        self.assertIn("Limitations", outline_titles[-1])
        self.assertIn("future work", outline_titles[-1])

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

    def test_evaluate_output_accepts_soft_keyword_and_range_fallback_hits(self) -> None:
        config = build_benchmark_task_config("med_s001")
        reference = config["reference"]
        paragraphs = [
            "Cardiovascular medicine and acute coronary syndrome are introduced with mechanism, adult inpatient care, and hemodynamics in view.",
            "The classification scheme frames a comparison of diagnostic and therapeutic priorities.",
            "The mechanism and hemodynamics evidence are compared across adult inpatient pathways.",
            "A further comparison integrates cardiovascular medicine evidence with practical inpatient constraints.",
            "The review turns to limitations and uncertainties in the evidence base.",
            "Evidence gaps remain visible when adult inpatient pathways are compared across settings.",
            "Future research should refine the agenda while the discussion boundaries remain explicit.",
        ]

        evaluation = evaluate_output("\n\n".join(paragraphs), reference)
        diagnostics = evaluation["diagnostics"]

        self.assertIn("scope", evaluation["diagnostics"]["matched_keywords"])
        self.assertIn("scope", diagnostics["range_keyword_global_fallback_hits"])
        self.assertNotIn("scope", diagnostics["missing_range_keywords"])
        self.assertIn("future work", diagnostics["range_keyword_hits"])

    def test_evaluate_output_accepts_partial_long_domain_phrase(self) -> None:
        config = build_benchmark_task_config("med_s381")
        reference = config["reference"]
        paragraphs = [
            "The scope of perioperative optimization is framed within perioperative medicine and low-resource settings.",
            "The controversy focus concerns whether prognosis tools and risk assessment can travel across settings.",
            "The evidence compares limitations in risk assessment and implementation feasibility.",
            "Future work should address evidence gaps and practical constraints.",
        ]

        evaluation = evaluate_output("\n\n".join(paragraphs), reference)

        self.assertIn(
            "surgery and perioperative medicine",
            evaluation["diagnostics"]["matched_keywords"],
        )

    def test_evaluate_output_uses_semantic_coverage_beyond_raw_keywords(self) -> None:
        config = build_benchmark_task_config("med_s381")
        reference = copy.deepcopy(config["reference"])
        reference["constraints"]["must_include"].extend(
            [
                "specialized surgical systems",
                "perioperative delivery networks",
                "resource stewardship model",
                "implementation maturity index",
            ]
        )
        paragraphs = [
            (
                "The review boundary is defined around perioperative optimization "
                "in resource-limited services, with prognosis and risk assessment "
                "used to frame the core debate."
            ),
            (
                "The discussion compares evidence conflicts across care settings "
                "and explains why implementation feasibility matters more than "
                "copying protocols from high-resource systems."
            ),
            (
                "The closing section describes evidence gaps, uncertainties, "
                "and future research directions for practical perioperative medicine."
            ),
        ]
        generated_text = "\n\n".join(paragraphs)

        evaluation = evaluate_output(generated_text, reference)
        diagnostics = evaluation["diagnostics"]

        self.assertLess(
            diagnostics["raw_keyword_coverage"],
            diagnostics["semantic_coverage_score"],
        )
        self.assertEqual(
            evaluation["entity_consistency_score"],
            diagnostics["semantic_coverage_score"],
        )
        self.assertIn("specialized surgical systems", diagnostics["missing_keywords"])
        self.assertGreater(diagnostics["proxy_hit_rate"], 0.0)
        self.assertLess(
            diagnostics["semantic_violation_rate"],
            1.0 - diagnostics["semantic_coverage_score"],
        )


if __name__ == "__main__":
    unittest.main()
