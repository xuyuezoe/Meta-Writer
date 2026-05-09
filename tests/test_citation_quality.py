from __future__ import annotations

import unittest

from examples.citation_quality import evaluate_citation_quality
from examples.benchmark_template import evaluate_output


class CitationQualityFlagSplitTests(unittest.TestCase):
    def test_text_inferred_citation_quality_scores_main_output_style(self) -> None:
        text = """## Scope and background

Acute coronary syndrome mechanisms are heterogeneous and involve plaque rupture, plaque erosion, and thrombotic activation [1].

---

## Evidence synthesis

Mechanistic reviews and clinical cohorts describe inflammatory signaling and platelet activation as recurring explanatory axes [1][2]. These studies also show that microvascular dysfunction can shape downstream injury patterns [2].

---

## Limitations and future work

Future work should compare mechanisms across subgroups while avoiding overgeneralized causal claims from single-source evidence [2].

---

## References

[1] Smith A. Plaque disruption mechanisms in acute coronary syndrome.
[2] Jones B. Microvascular dysfunction and platelet activation in coronary disease.
"""

        result = evaluate_citation_quality(text, None)

        self.assertEqual(result["status"], "evaluated")
        self.assertEqual(result["diagnostics"]["citation_event_count"], 5)
        self.assertEqual(result["diagnostics"]["unique_source_count"], 2)
        self.assertEqual(
            result["diagnostics"]["citation_quality_input"]["config_source"],
            "text_inferred",
        )
        self.assertEqual(len(result["section_scores"]), 3)

    def test_benchmark_evaluator_includes_text_inferred_citation_quality(self) -> None:
        text = """## Scope and background

Acute coronary syndrome reviews distinguish plaque rupture from plaque erosion [1].

---

## Evidence synthesis

Microvascular dysfunction and platelet activation recur across clinical discussions [1][2].

---

## Limitations and future work

Future work should compare subgroups more carefully [2].

---

## References

[1] Smith A. Plaque disruption mechanisms in acute coronary syndrome.
[2] Jones B. Microvascular dysfunction in coronary disease.
"""
        reference = {
            "constraints": {
                "required_length_words": 50,
                "must_include": ["acute coronary syndrome"],
                "once_keywords": ["acute coronary syndrome"],
                "expected_blocks": 3,
                "range_keywords": [],
                "periodic_keywords": [],
            },
            "proxy_questions": [{"qid": "q1", "question": "topic?", "answer": "plaque"}],
            "checklist": ["acute coronary syndrome"],
        }

        result = evaluate_output(text, reference)

        self.assertIn("citation_quality", result)
        self.assertEqual(result["citation_quality"]["status"], "evaluated")
        self.assertEqual(
            result["citation_quality"]["diagnostics"]["citation_quality_input"][
                "config_source"
            ],
            "text_inferred",
        )

    def test_review_low_citation_count_only_hits_coverage_not_overcitation(self) -> None:
        result = evaluate_citation_quality(
            "alpha beta gamma delta epsilon",
            {
                "paper_type": "review",
                "section_map": [
                    {
                        "section_id": "sec1",
                        "type": "Taxonomy",
                        "word_count": 1400,
                    }
                ],
                "citation_manifest": [
                    {
                        "citation_id": "C001",
                        "source_id": "paper_a",
                        "section_id": "sec1",
                        "claim_span": "taxonomy observation",
                        "source_excerpt": "taxonomy observation",
                        "source_type": "mechanistic",
                        "claim_role": "mechanism",
                        "claim_strength": 1,
                        "evidence_strength": 1,
                    },
                    {
                        "citation_id": "C002",
                        "source_id": "paper_b",
                        "section_id": "sec1",
                        "claim_span": "second taxonomy observation",
                        "source_excerpt": "second taxonomy observation",
                        "source_type": "mechanistic",
                        "claim_role": "mechanism",
                        "claim_strength": 1,
                        "evidence_strength": 1,
                    },
                ],
                "judge_flags": {"sections": {"sec1": {}}},
            },
        )

        penalties = result["section_scores"][0]["penalties"]
        metric_breakdown = result["diagnostics"]["metric_breakdown"]
        self.assertEqual(penalties["expected_min_citations"], 4.0)
        self.assertAlmostEqual(penalties["coverage"], 0.5, places=4)
        self.assertAlmostEqual(penalties["citation_density"], 1.4286, places=4)
        self.assertEqual(penalties["overcitation"], 0.0)
        self.assertIn("overcitation", result["penalties"])
        self.assertNotIn("density", result["penalties"])
        self.assertIn("overcitation", metric_breakdown)
        self.assertNotIn("density", metric_breakdown)
        self.assertAlmostEqual(
            sum(item["weight"] for item in metric_breakdown.values()),
            1.0,
            places=4,
        )

    def test_review_true_citation_pileup_hits_overcitation(self) -> None:
        result = evaluate_citation_quality(
            "alpha beta gamma delta epsilon",
            {
                "paper_type": "review",
                "section_map": [
                    {
                        "section_id": "sec1",
                        "type": "Evidence Synthesis",
                        "word_count": 400,
                    }
                ],
                "citation_manifest": [
                    {
                        "citation_id": f"C{i:03d}",
                        "source_id": f"paper_{i}",
                        "section_id": "sec1",
                        "claim_span": f"claim {i}",
                        "source_excerpt": f"excerpt {i}",
                        "source_type": "mechanistic",
                        "claim_role": "mechanism",
                        "claim_strength": 1,
                        "evidence_strength": 1,
                    }
                    for i in range(1, 14)
                ],
                "judge_flags": {"sections": {"sec1": {}}},
            },
        )

        penalties = result["section_scores"][0]["penalties"]
        self.assertEqual(penalties["coverage"], 0.0)
        self.assertEqual(penalties["overcitation"], 0.2)
        self.assertAlmostEqual(penalties["citation_density"], 13.0, places=4)

    def test_source_balance_replaces_separate_source_diversity_and_dominance_scoring(self) -> None:
        result = evaluate_citation_quality(
            "alpha beta gamma delta epsilon",
            {
                "paper_type": "review",
                "section_map": [
                    {
                        "section_id": "sec1",
                        "type": "Evidence Synthesis",
                        "word_count": 1000,
                    }
                ],
                "citation_manifest": [
                    {
                        "citation_id": f"C{i:03d}",
                        "source_id": "paper_a" if i <= 4 else f"paper_{i}",
                        "section_id": "sec1",
                        "claim_span": f"claim {i}",
                        "source_excerpt": f"excerpt {i}",
                        "source_type": "mechanistic",
                        "claim_role": "mechanism",
                        "claim_strength": 1,
                        "evidence_strength": 1,
                    }
                    for i in range(1, 7)
                ],
                "judge_flags": {"sections": {"sec1": {}}},
            },
        )

        penalties = result["penalties"]
        metric_breakdown = result["diagnostics"]["metric_breakdown"]
        source_details = result["diagnostics"]["source_balance_details"]
        section = result["section_scores"][0]
        section_penalties = section["penalties"]

        self.assertEqual(penalties["source_balance"], 1.0)
        self.assertIn("source_balance", metric_breakdown)
        self.assertNotIn("source_diversity", metric_breakdown)
        self.assertNotIn("source_dominance", metric_breakdown)
        self.assertEqual(source_details["source_diversity_penalty"], 0.2)
        self.assertEqual(source_details["source_dominance_penalty"], 1.0)
        self.assertEqual(section_penalties["source_dominance"], 1.0)
        self.assertNotEqual(section["score"], 0.95)

    def test_review_source_balance_requires_minimum_repeated_source_count(self) -> None:
        result = evaluate_citation_quality(
            "alpha beta gamma delta epsilon",
            {
                "paper_type": "review",
                "section_map": [
                    {
                        "section_id": "sec1",
                        "type": "Evidence Synthesis",
                        "word_count": 1000,
                    }
                ],
                "citation_manifest": [
                    {
                        "citation_id": f"C{i:03d}",
                        "source_id": "paper_a" if i <= 3 else f"paper_{i}",
                        "section_id": "sec1",
                        "claim_span": f"claim {i}",
                        "source_excerpt": f"excerpt {i}",
                        "source_type": "mechanistic",
                        "claim_role": "mechanism",
                        "claim_strength": 1,
                        "evidence_strength": 1,
                    }
                    for i in range(1, 14)
                ],
                "judge_flags": {"sections": {"sec1": {}}},
            },
        )

        source_details = result["diagnostics"]["source_balance_details"]
        self.assertEqual(source_details["source_diversity_penalty"], 0.0)
        self.assertEqual(source_details["source_dominance_penalty"], 0.0)
        self.assertEqual(result["penalties"]["source_balance"], 0.0)

    def test_wrong_source_alignment_does_not_count_as_granularity_unsupported(self) -> None:
        result = evaluate_citation_quality(
            "alpha beta gamma",
            {
                "paper_type": "review",
                "section_map": [
                    {
                        "section_id": "sec1",
                        "type": "Evidence Synthesis",
                        "word_count": 350,
                    }
                ],
                "citation_manifest": [
                    {
                        "citation_id": "C001",
                        "source_id": "paper_a",
                        "section_id": "sec1",
                        "claim_span": "mechanistic observation",
                        "source_excerpt": "mechanistic observation",
                        "source_type": "mechanistic",
                        "claim_role": "mechanism",
                        "claim_strength": 1,
                        "evidence_strength": 1,
                    }
                ],
                "judge_flags": {
                    "sections": {
                        "sec1": {
                            "unsupported_high_claims": 0,
                            "wrong_source_alignment_events": 2,
                            "citation_pileups": 0,
                            "paragraph_tail_only_events": 0,
                        }
                    }
                },
            },
        )

        section = result["section_scores"][0]
        self.assertEqual(section["penalties"]["unsupported_high_claims"], 0.0)
        self.assertEqual(section["penalties"]["wrong_source_alignment_events"], 2.0)
        self.assertEqual(section["penalties"]["citation_granularity"], 0.0)
        self.assertEqual(section["penalties"]["claim_source_match"], 1.0)

    def test_diagnostics_expose_aggregated_judge_flags_separately(self) -> None:
        result = evaluate_citation_quality(
            "alpha beta gamma",
            {
                "paper_type": "review",
                "section_map": [
                    {
                        "section_id": "sec1",
                        "type": "Evidence Synthesis",
                        "word_count": 350,
                    }
                ],
                "citation_manifest": [
                    {
                        "citation_id": "C001",
                        "source_id": "paper_a",
                        "section_id": "sec1",
                        "claim_span": "mechanistic observation",
                        "source_excerpt": "mechanistic observation",
                        "source_type": "mechanistic",
                        "claim_role": "mechanism",
                        "claim_strength": 1,
                        "evidence_strength": 1,
                    }
                ],
                "judge_flags": {
                    "sections": {
                        "sec1": {
                            "unsupported_high_claims": 3,
                            "judge_declared_unsupported_high_claims": 2,
                            "retrieval_unsupported_alignment_events": 1,
                            "wrong_source_alignment_events": 2,
                            "weak_match_alignment_events": 1,
                            "citation_pileups": 1,
                            "paragraph_tail_only_events": 0,
                        }
                    }
                },
            },
        )

        diagnostics = result["diagnostics"]
        self.assertEqual(
            diagnostics["judge_aggregated_flags"],
            {
                "unsupported_high_claims": 3.0,
                "judge_declared_unsupported_high_claims": 2.0,
                "weighted_judge_declared_unsupported_high_claims": 2.0,
                "retrieval_unsupported_alignment_events": 1.0,
                "wrong_source_alignment_events": 2.0,
                "weak_match_alignment_events": 1.0,
                "citation_pileups": 1.0,
                "paragraph_tail_only_events": 0.0,
            },
        )
        self.assertEqual(diagnostics["flagged_events"]["unsupported_high_claim_events"], [])
        self.assertIn("judge_aggregated_flags", diagnostics["flag_semantics"])

    def test_weak_match_counts_as_partial_match_penalty_not_unsupported(self) -> None:
        result = evaluate_citation_quality(
            "alpha beta gamma delta",
            {
                "paper_type": "review",
                "section_map": [
                    {
                        "section_id": "sec1",
                        "type": "Evidence Synthesis",
                        "word_count": 400,
                    }
                ],
                "citation_manifest": [
                    {
                        "citation_id": "C001",
                        "source_id": "paper_a",
                        "section_id": "sec1",
                        "claim_span": "mechanistic observation",
                        "source_excerpt": "mechanistic observation",
                        "source_type": "mechanistic",
                        "claim_role": "mechanism",
                        "claim_strength": 1,
                        "evidence_strength": 1,
                    },
                    {
                        "citation_id": "C002",
                        "source_id": "paper_b",
                        "section_id": "sec1",
                        "claim_span": "second mechanistic observation",
                        "source_excerpt": "second mechanistic observation",
                        "source_type": "mechanistic",
                        "claim_role": "mechanism",
                        "claim_strength": 1,
                        "evidence_strength": 1,
                    },
                ],
                "judge_flags": {
                    "sections": {
                        "sec1": {
                            "unsupported_high_claims": 0,
                            "wrong_source_alignment_events": 0,
                            "weak_match_alignment_events": 2,
                            "citation_pileups": 0,
                            "paragraph_tail_only_events": 0,
                        }
                    }
                },
            },
        )

        section = result["section_scores"][0]
        self.assertEqual(section["penalties"]["unsupported_high_claims"], 0.0)
        self.assertEqual(section["penalties"]["weak_match_alignment_events"], 2.0)
        self.assertEqual(section["penalties"]["citation_granularity"], 0.0)
        self.assertEqual(section["penalties"]["claim_source_match"], 0.35)

    def test_role_mismatch_is_discounted_when_alignment_flags_also_exist(self) -> None:
        result = evaluate_citation_quality(
            "alpha beta gamma delta",
            {
                "paper_type": "review",
                "section_map": [
                    {
                        "section_id": "sec1",
                        "type": "Evidence Synthesis",
                        "word_count": 400,
                    }
                ],
                "citation_manifest": [
                    {
                        "citation_id": "C001",
                        "source_id": "paper_a",
                        "section_id": "sec1",
                        "claim_span": "quantitative recommendation",
                        "source_excerpt": "animal mechanism",
                        "source_type": "mechanistic",
                        "claim_role": "recommendation",
                        "claim_strength": 2,
                        "evidence_strength": 1,
                        "role_mismatch": True,
                    }
                ],
                "judge_flags": {
                    "sections": {
                        "sec1": {
                            "unsupported_high_claims": 0,
                            "wrong_source_alignment_events": 1,
                            "weak_match_alignment_events": 1,
                            "citation_pileups": 0,
                            "paragraph_tail_only_events": 0,
                        }
                    }
                },
            },
        )

        section = result["section_scores"][0]
        self.assertEqual(section["penalties"]["wrong_source_alignment_events"], 1.0)
        self.assertEqual(section["penalties"]["weak_match_alignment_events"], 1.0)
        self.assertEqual(section["penalties"]["claim_source_match"], 1.0)

    def test_unsupported_penalty_separates_judge_declared_and_retrieval_sources(self) -> None:
        result = evaluate_citation_quality(
            "alpha beta gamma delta",
            {
                "paper_type": "review",
                "section_map": [
                    {
                        "section_id": "sec1",
                        "type": "Evidence Synthesis",
                        "word_count": 400,
                    }
                ],
                "citation_manifest": [],
                "judge_flags": {
                    "sections": {
                        "sec1": {
                            "unsupported_high_claims": 5,
                            "judge_declared_unsupported_high_claims": 3,
                            "retrieval_unsupported_alignment_events": 2,
                            "wrong_source_alignment_events": 0,
                            "weak_match_alignment_events": 0,
                            "citation_pileups": 0,
                            "paragraph_tail_only_events": 0,
                        }
                    }
                },
            },
        )

        section = result["section_scores"][0]
        self.assertEqual(section["penalties"]["unsupported_high_claims"], 5.0)
        self.assertEqual(section["penalties"]["judge_declared_unsupported_high_claims"], 3.0)
        self.assertEqual(section["penalties"]["retrieval_unsupported_alignment_events"], 2.0)
        self.assertEqual(
            result["diagnostics"]["judge_aggregated_flags"]["judge_declared_unsupported_high_claims"],
            3.0,
        )
        self.assertEqual(
            result["diagnostics"]["judge_aggregated_flags"]["retrieval_unsupported_alignment_events"],
            2.0,
        )

    def test_zero_judge_declared_does_not_double_count_retrieval_unsupported(self) -> None:
        result = evaluate_citation_quality(
            "alpha beta gamma delta",
            {
                "paper_type": "review",
                "section_map": [
                    {
                        "section_id": "sec1",
                        "type": "Evidence Synthesis",
                        "word_count": 400,
                    }
                ],
                "citation_manifest": [
                    {
                        "citation_id": "C001",
                        "source_id": "paper_a",
                        "section_id": "sec1",
                        "claim_span": "mechanistic observation",
                        "source_excerpt": "mechanistic observation",
                        "source_type": "mechanistic",
                        "claim_role": "mechanism",
                        "claim_strength": 1,
                        "evidence_strength": 1,
                    }
                ],
                "judge_flags": {
                    "sections": {
                        "sec1": {
                            "unsupported_high_claims": 2,
                            "judge_declared_unsupported_high_claims": 0,
                            "retrieval_unsupported_alignment_events": 2,
                            "wrong_source_alignment_events": 0,
                            "weak_match_alignment_events": 0,
                            "citation_pileups": 0,
                            "paragraph_tail_only_events": 0,
                        }
                    }
                },
            },
        )

        section = result["section_scores"][0]
        penalties = section["penalties"]
        self.assertEqual(penalties["unsupported_high_claims"], 2.0)
        self.assertEqual(penalties["judge_declared_unsupported_high_claims"], 0.0)
        self.assertEqual(penalties["retrieval_unsupported_alignment_events"], 2.0)
        self.assertEqual(penalties["granularity_judge_unsupported_component"], 0.0)
        self.assertEqual(penalties["granularity_retrieval_unsupported_component"], 1.0)
        self.assertEqual(penalties["citation_granularity"], 1.0)
        self.assertEqual(
            result["diagnostics"]["judge_aggregated_flags"][
                "judge_declared_unsupported_high_claims"
            ],
            0.0,
        )

    def test_granularity_components_are_reported_separately(self) -> None:
        result = evaluate_citation_quality(
            "alpha beta gamma delta",
            {
                "paper_type": "review",
                "section_map": [
                    {
                        "section_id": "sec1",
                        "type": "Evidence Synthesis",
                        "word_count": 400,
                    }
                ],
                "citation_manifest": [
                    {
                        "citation_id": "C001",
                        "source_id": "paper_a",
                        "section_id": "sec1",
                        "claim_span": "mechanistic observation",
                        "source_excerpt": "mechanistic observation",
                        "source_type": "mechanistic",
                        "claim_role": "mechanism",
                        "claim_strength": 1,
                        "evidence_strength": 1,
                    },
                    {
                        "citation_id": "C002",
                        "source_id": "paper_b",
                        "section_id": "sec1",
                        "claim_span": "second mechanistic observation",
                        "source_excerpt": "second mechanistic observation",
                        "source_type": "mechanistic",
                        "claim_role": "mechanism",
                        "claim_strength": 1,
                        "evidence_strength": 1,
                    },
                ],
                "judge_flags": {
                    "sections": {
                        "sec1": {
                            "unsupported_high_claims": 2,
                            "judge_declared_unsupported_high_claims": 1,
                            "retrieval_unsupported_alignment_events": 1,
                            "wrong_source_alignment_events": 2,
                            "weak_match_alignment_events": 2,
                            "citation_pileups": 1,
                            "paragraph_tail_only_events": 1,
                        }
                    }
                },
            },
        )

        section = result["section_scores"][0]
        penalties = section["penalties"]
        details = result["diagnostics"]["citation_granularity_details"]

        self.assertAlmostEqual(
            penalties["granularity_judge_unsupported_component"], 0.25, places=4
        )
        self.assertAlmostEqual(
            penalties["granularity_retrieval_unsupported_component"], 0.25, places=4
        )
        self.assertAlmostEqual(penalties["granularity_pileup_component"], 0.15, places=4)
        self.assertAlmostEqual(penalties["granularity_tail_only_component"], 0.1, places=4)
        self.assertAlmostEqual(penalties["citation_granularity_uncapped"], 0.75, places=4)
        self.assertAlmostEqual(penalties["citation_granularity"], 0.75, places=4)
        self.assertEqual(penalties["wrong_source_alignment_events"], 2.0)
        self.assertEqual(penalties["weak_match_alignment_events"], 2.0)
        self.assertIn("boundary_note", details)
        self.assertEqual(
            details["weighted_components"],
            {
                "judge_unsupported": 0.25,
                "retrieval_unsupported": 0.25,
                "pileup": 0.15,
                "tail_only": 0.1,
                "uncapped_sum": 0.75,
                "capped_penalty": 0.75,
            },
        )

    def test_granularity_uncapped_component_explains_section_cap(self) -> None:
        result = evaluate_citation_quality(
            "alpha beta gamma delta",
            {
                "paper_type": "review",
                "section_map": [
                    {
                        "section_id": "sec1",
                        "type": "Evidence Synthesis",
                        "word_count": 400,
                    }
                ],
                "citation_manifest": [
                    {
                        "citation_id": "C001",
                        "source_id": "paper_a",
                        "section_id": "sec1",
                        "claim_span": "mechanistic observation",
                        "source_excerpt": "mechanistic observation",
                        "source_type": "mechanistic",
                        "claim_role": "mechanism",
                        "claim_strength": 1,
                        "evidence_strength": 1,
                    }
                ],
                "judge_flags": {
                    "sections": {
                        "sec1": {
                            "unsupported_high_claims": 5,
                            "judge_declared_unsupported_high_claims": 3,
                            "retrieval_unsupported_alignment_events": 2,
                            "wrong_source_alignment_events": 0,
                            "weak_match_alignment_events": 0,
                            "citation_pileups": 1,
                            "paragraph_tail_only_events": 1,
                        }
                    }
                },
            },
        )

        penalties = result["section_scores"][0]["penalties"]
        details = result["diagnostics"]["citation_granularity_details"]
        self.assertEqual(penalties["citation_granularity"], 1.0)
        self.assertAlmostEqual(penalties["citation_granularity_uncapped"], 3.0, places=4)
        self.assertEqual(details["weighted_components"]["capped_penalty"], 1.0)
        self.assertAlmostEqual(details["weighted_components"]["uncapped_sum"], 3.0, places=4)

    def test_soft_sections_discount_judge_declared_unsupported_in_granularity(self) -> None:
        result = evaluate_citation_quality(
            "alpha beta gamma delta",
            {
                "paper_type": "review",
                "section_map": [
                    {
                        "section_id": "sec1",
                        "type": "Introduction",
                        "word_count": 400,
                    }
                ],
                "citation_manifest": [
                    {
                        "citation_id": "C001",
                        "source_id": "paper_a",
                        "section_id": "sec1",
                        "claim_span": "introductory context",
                        "source_excerpt": "introductory context",
                        "source_type": "mechanistic",
                        "claim_role": "background",
                        "claim_strength": 0,
                        "evidence_strength": 1,
                    }
                ],
                "judge_flags": {
                    "sections": {
                        "sec1": {
                            "unsupported_high_claims": 3,
                            "judge_declared_unsupported_high_claims": 3,
                            "retrieval_unsupported_alignment_events": 0,
                            "wrong_source_alignment_events": 0,
                            "weak_match_alignment_events": 0,
                            "citation_pileups": 0,
                            "paragraph_tail_only_events": 0,
                        }
                    }
                },
            },
        )

        section = result["section_scores"][0]
        self.assertEqual(section["penalties"]["judge_declared_unsupported_high_claims"], 3.0)
        self.assertEqual(section["penalties"]["weighted_judge_declared_unsupported_high_claims"], 1.8)
        self.assertEqual(section["penalties"]["citation_granularity"], 0.9)


if __name__ == "__main__":
    unittest.main()
