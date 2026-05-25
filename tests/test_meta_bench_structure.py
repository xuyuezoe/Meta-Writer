import unittest

from meta_bench.structure import (
    classify_functional_sections,
    compute_length_score_from_ratio,
    count_length_units,
    evaluate_structure_dimension,
    parse_content_sections,
    score_completion_rate,
    score_length,
)


class LengthScoreTests(unittest.TestCase):
    def test_count_length_units_counts_english_tokens_only(self):
        self.assertEqual(count_length_units("acute coronary syndrome 心血管"), 3)

    def test_score_length_uses_required_length_as_baseline(self):
        result = score_length("one two three four five", 10)

        self.assertEqual(result.response_word_count, 5)
        self.assertEqual(result.required_length_words, 10)
        self.assertEqual(result.length_ratio, 0.5)
        self.assertAlmostEqual(result.score, 0.4)

    def test_length_score_piecewise_reasonable_zone(self):
        self.assertEqual(compute_length_score_from_ratio(1.0), 1.0)
        self.assertAlmostEqual(compute_length_score_from_ratio(1.2), 0.9)
        self.assertAlmostEqual(compute_length_score_from_ratio(1.25), 0.875)

    def test_evaluate_structure_dimension_extracts_required_length(self):
        result = evaluate_structure_dimension(
            "one two three four five",
            {"constraints": {"required_length_words": 5, "expected_sections": 1}},
        )

        self.assertEqual(result["dimension"], "structure")
        self.assertEqual(result["scores"]["length_score"], 1.0)
        self.assertEqual(result["scores"]["completion_rate"], 0.0)
        self.assertEqual(result["diagnostics"]["length"]["length_ratio"], 1.0)


class CompletionRateTests(unittest.TestCase):
    def test_parse_content_sections_ignores_title_abstract_and_references(self):
        text = """# Article Title

## Abstract

Abstract text.

## Introduction

Intro text.

## Evidence synthesis

Evidence text.

## References

1. Ref.
"""

        sections = parse_content_sections(text)

        self.assertEqual([section.heading for section in sections], ["Introduction", "Evidence synthesis"])
        self.assertEqual(len(sections), 2)

    def test_score_completion_rate_counts_non_empty_content_sections(self):
        text = """## Introduction

This introduction defines the scope and background context for the article.

## Methods


## Discussion

Evidence from studies is compared in the main discussion body.
"""

        result = score_completion_rate(text, expected_section_count=4)

        self.assertEqual(result.parsed_section_count, 3)
        self.assertEqual(result.complete_section_count, 2)
        self.assertAlmostEqual(result.score, 2 / 3)

    def test_completion_rate_scores_functional_slots_not_heading_count(self):
        text = """## One

Text.

## Two

Text.

## Three

Text.
"""

        result = score_completion_rate(text, expected_section_count=2)

        self.assertAlmostEqual(result.score, 1 / 3)
        self.assertEqual(result.complete_section_count, 1)

    def test_classify_functional_sections_uses_tail_paragraph_fallback_with_signals(self):
        text = """## Discussion

The article opens with background context and defines the scope of the disease area for clinicians, researchers, and health systems.

Evidence from studies is compared and synthesized across patient groups.

Overall, these findings show important limitations and future work for research, including better evidence synthesis, clearer outcome definitions, and more careful study design.
"""

        slots = classify_functional_sections(text)

        self.assertEqual(slots["introduction"].source, "fallback_first_paragraph")
        self.assertEqual(slots["main_body"].source, "explicit_heading")
        self.assertEqual(slots["conclusion"].source, "fallback_tail_paragraph")
        self.assertIn("overall", slots["conclusion"].matched_signals)

    def test_tail_fallback_requires_functional_signal(self):
        text = """## Discussion

The article opens with background context and defines the scope of the disease area for clinicians, researchers, and health systems.

Evidence from studies is compared and synthesized across patient groups.

This paragraph reports another detail about the same comparison.
"""

        slots = classify_functional_sections(text)

        self.assertFalse(slots["conclusion"].filled)

    def test_classify_functional_sections_recognizes_descriptive_intro_and_future_headings(self):
        text = """## Scope, terminology, and practice context

This section defines the scope and background context for the review.

## Future work and research agenda

Future work should address limitations, open questions, and research priorities.
"""

        slots = classify_functional_sections(text)

        self.assertEqual(slots["introduction"].source, "explicit_heading")
        self.assertEqual(slots["conclusion"].source, "explicit_heading")
        self.assertTrue(slots["conclusion"].filled)

    def test_evaluate_structure_dimension_extracts_expected_sections(self):
        text = """## Introduction

This introduction defines the scope and background context.

## Evidence

Evidence from studies is compared and synthesized across groups.
"""
        result = evaluate_structure_dimension(
            text,
            {
                "constraints": {
                    "required_length_words": 4,
                    "expected_sections": 4,
                }
            },
        )

        self.assertAlmostEqual(result["scores"]["completion_rate"], 2 / 3)
        self.assertEqual(
            result["diagnostics"]["completion"]["expected_section_count"],
            3,
        )


if __name__ == "__main__":
    unittest.main()
