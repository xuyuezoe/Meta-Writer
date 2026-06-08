import unittest

from meta_bench.citation import (
    CITATION_COUNT_THRESHOLDS,
    SOURCE_BALANCE_THRESHOLDS,
    CitationChunk,
    CitationEvent,
    classify_chunk_role,
    evaluate_citation_dimension,
    infer_claim_strength,
    infer_evidence_strength,
    score_citation_count,
    score_section_distribution,
    score_source_balance,
    score_source_fidelity,
    split_into_seven_sections,
)


class ChunkRoleClassifierTests(unittest.TestCase):
    def test_limitations_title_beats_evidence_keyword(self):
        chunk = CitationChunk(
            chunk_id="sec6",
            index=6,
            title="Limitations and evidence gaps",
            body=(
                "Bias, uncertainty, and heterogeneity limit interpretation. "
                "Future studies should address evidence gaps in larger cohorts."
            ),
            word_count=18,
        )

        result = classify_chunk_role(chunk, total_chunks=7)

        self.assertEqual(result.canonical_type, "limitations_gaps")
        self.assertIn("limitations->limitations_gaps", result.matched_signals["title"])
        self.assertIn("evidence->evidence_synthesis", result.matched_signals["title"])
        self.assertGreater(
            result.scores_by_type["limitations_gaps"],
            result.scores_by_type["evidence_synthesis"],
        )

    def test_classifier_uses_position_and_body_signals(self):
        chunk = CitationChunk(
            chunk_id="sec2",
            index=2,
            title="Clinical groups",
            body=(
                "Patients can be classified into phenotypes and subtypes using "
                "a practical framework that separates presentation and risk."
            ),
            word_count=20,
        )

        result = classify_chunk_role(chunk, total_chunks=6)

        self.assertEqual(result.canonical_type, "taxonomy")
        self.assertIn("second_chunk:taxonomy", result.matched_signals["position"])
        self.assertTrue(
            any(signal.endswith("->taxonomy") for signal in result.matched_signals["body"])
        )

    def test_classifier_marks_close_scores_low_confidence(self):
        chunk = CitationChunk(
            chunk_id="sec3",
            index=3,
            title="Clinical implications",
            body=(
                "The discussion interprets findings and implications for practice. "
                "The section also compares evidence from studies."
            ),
            word_count=18,
        )

        result = classify_chunk_role(chunk, total_chunks=5)

        self.assertTrue(result.low_confidence)
        self.assertGreater(result.confidence, 0.0)

    def test_benchmark_range_keywords_add_hint(self):
        chunk = CitationChunk(
            chunk_id="sec7",
            index=7,
            title="Research agenda",
            body="Future work should resolve uncertainty and open questions.",
            word_count=9,
        )
        reference = {
            "constraints": {
                "range_keywords": {
                    "future work": "7",
                }
            }
        }

        result = classify_chunk_role(chunk, total_chunks=7, reference=reference)

        self.assertEqual(result.canonical_type, "limitations_gaps")
        self.assertIn(
            "range_keyword:future work->limitations_gaps",
            result.matched_signals["benchmark_hint"],
        )


class SourceFidelityTests(unittest.TestCase):
    def test_strength_inference_uses_metadata_and_text_cues(self):
        event = CitationEvent(
            citation_id="C1",
            chunk_id="sec1",
            source_id="S1",
            source_type="observational",
            claim_role="unknown",
            claim_span="This treatment should improve outcomes in practice.",
            source_excerpt="A retrospective cohort study reported an association.",
        )

        self.assertEqual(infer_claim_strength(event), 4)
        self.assertEqual(infer_evidence_strength(event), 2)

    def test_score_source_fidelity_penalizes_overclaiming(self):
        final_text = """## Introduction

This background section defines the scope.

## Evidence synthesis

Trials and cohorts are compared here.
"""
        citation_manifest = [
            {
                "citation_id": "C1",
                "section_id": "sec2",
                "source_id": "mechanism_only",
                "source_type": "mechanistic",
                "claim_role": "recommendation",
            },
            {
                "citation_id": "C2",
                "section_id": "sec2",
                "source_id": "meta_review",
                "source_type": "meta_analysis",
                "claim_role": "evidence_synthesis",
            },
        ]

        result = score_source_fidelity(final_text, citation_manifest)

        self.assertEqual(result.citation_count, 2)
        self.assertEqual(result.overclaim_count, 1)
        self.assertEqual(result.severe_overclaim_count, 1)
        self.assertAlmostEqual(result.penalty, 0.4)
        self.assertAlmostEqual(result.score, 0.6)

    def test_evaluate_citation_dimension_extracts_manifest_and_chunks(self):
        final_text = "Plain text fallback body."
        reference = {
            "chunk_map": [
                {
                    "chunk_id": "sec1",
                    "title": "Limitations and evidence gaps",
                    "text": "Bias and uncertainty create evidence gaps.",
                }
            ],
            "citation_manifest": [
                {
                    "citation_id": "C1",
                    "chunk_id": "sec1",
                    "source_id": "cohort",
                    "source_type": "cohort",
                    "claim_role": "comparison",
                }
            ],
        }

        result = evaluate_citation_dimension(final_text, reference)

        self.assertEqual(result["dimension"], "citation")
        self.assertEqual(result["scores"]["source_fidelity"], 1.0)
        diagnostics = result["diagnostics"]["source_fidelity"]
        self.assertEqual(
            diagnostics["chunk_classifications"][0]["canonical_type"],
            "limitations_gaps",
        )

    def test_evaluate_citation_dimension_skips_without_manifest(self):
        result = evaluate_citation_dimension(
            "Some final text.",
            {"constraints": {}},
        )

        self.assertEqual(result["scores"], {"citation_count": 0.0, "source_balance": 0.0})
        self.assertIn("citation_count", result["diagnostics"])
        self.assertIn("source_balance", result["diagnostics"])
        self.assertEqual(
            result["diagnostics"]["source_fidelity"]["reason"],
            "missing_citation_manifest",
        )


class SectionDistributionTests(unittest.TestCase):
    def test_split_into_seven_sections_uses_article_slots(self):
        text = """# Review title

## Abstract

Abstract evidence [1].

## Introduction

Background context [1].

## Evidence synthesis

Main evidence [2,3].

## Conclusion

Overall implications.

## References

1. Ref one.
2. Ref two.
3. Ref three.
"""

        sections = split_into_seven_sections(text)

        self.assertTrue(sections["title"].text_present)
        self.assertTrue(sections["abstract"].text_present)
        self.assertTrue(sections["introduction"].text_present)
        self.assertTrue(sections["main_body"].text_present)
        self.assertTrue(sections["conclusion"].text_present)
        self.assertTrue(sections["references"].text_present)

    def test_section_distribution_scores_reasonable_distribution(self):
        final_text = "Generated article body."
        citation_manifest = [
            *[
                {"citation_id": f"A{i}", "section_key": "abstract"}
                for i in range(1, 9)
            ],
            *[
                {"citation_id": f"I{i}", "section_key": "introduction"}
                for i in range(1, 11)
            ],
            *[
                {"citation_id": f"M{i}", "section_key": "main_body"}
                for i in range(1, 58)
            ],
            *[
                {"citation_id": f"C{i}", "section_key": "conclusion"}
                for i in range(1, 3)
            ],
        ]

        result = score_section_distribution(final_text, citation_manifest)

        self.assertEqual(result.status, "evaluated")
        self.assertGreaterEqual(result.score, 0.9)
        self.assertEqual(result.section_counts["main_body"], 57)
        self.assertAlmostEqual(result.section_shares["main_body"], 57 / 77, places=4)
        self.assertEqual(result.scheme, "seven_section_fallback")

    def test_section_distribution_penalizes_main_body_underuse(self):
        final_text = "Generated article body."
        citation_manifest = [
            *[
                {"citation_id": f"A{i}", "section_key": "abstract"}
                for i in range(1, 21)
            ],
            *[
                {"citation_id": f"I{i}", "section_key": "introduction"}
                for i in range(1, 21)
            ],
            *[
                {"citation_id": f"M{i}", "section_key": "main_body"}
                for i in range(1, 11)
            ],
        ]

        result = score_section_distribution(final_text, citation_manifest)

        self.assertEqual(result.status, "evaluated")
        self.assertLess(result.score, 0.8)
        self.assertEqual(result.section_penalties["main_body"], 1.0)

    def test_section_distribution_prefers_six_slot_task_layout(self):
        final_text = "Generated article body."
        citation_manifest = [
            *[
                {"citation_id": f"S1_{i}", "chunk_id": "sec1", "section_key": "introduction"}
                for i in range(1, 11)
            ],
            *[
                {"citation_id": f"S2_{i}", "chunk_id": "sec2", "section_key": "main_body"}
                for i in range(1, 21)
            ],
            *[
                {"citation_id": f"S3_{i}", "chunk_id": "sec3", "section_key": "main_body"}
                for i in range(1, 21)
            ],
            *[
                {"citation_id": f"S4_{i}", "chunk_id": "sec4", "section_key": "main_body"}
                for i in range(1, 21)
            ],
            *[
                {"citation_id": f"S5_{i}", "chunk_id": "sec5", "section_key": "main_body"}
                for i in range(1, 21)
            ],
            *[
                {"citation_id": f"S6_{i}", "chunk_id": "sec6", "section_key": "conclusion"}
                for i in range(1, 9)
            ],
        ]
        reference = {
            "outline": {
                "sec1": "Scope, disease context, and chemokine terminology in alopecia areata",
                "sec2": "Chemokine-pathway framework and hair-follicle immune privilege collapse",
                "sec3": "Evidence base and measurement strategies across blood and skin studies",
                "sec4": "Chemokine signatures in alopecia areata across Th1, Th2, and related pathways",
                "sec5": "Biomarker value and therapeutic implications for clinical dermatology",
                "sec6": "Limitations, heterogeneity, and future research priorities",
            }
        }

        result = score_section_distribution(
            final_text,
            citation_manifest,
            reference=reference,
        )

        self.assertEqual(result.status, "evaluated")
        self.assertEqual(result.scheme, "six_slot_task")
        self.assertGreaterEqual(result.score, 0.9)
        self.assertIsNotNone(result.prior_metadata)
        self.assertIn("prior_version", result.prior_metadata)
        self.assertEqual(
            result.prior_metadata["threshold_source"],
            "data_sample/six_slot_citation_priors.json",
        )
        self.assertEqual(
            list(result.section_counts.keys()),
            [
                "scope_context",
                "framework_mechanism",
                "evidence_methods",
                "findings_synthesis",
                "implications_discussion",
                "limitations_future",
            ],
        )
        self.assertAlmostEqual(result.section_shares["scope_context"], 10 / 98, places=4)

    def test_section_distribution_low_citation_count_is_not_evaluated(self):
        result = score_section_distribution(
            "Short text [1].",
            [{"citation_id": "C1", "section_key": "main_body"}],
        )

        self.assertIsNone(result.score)
        self.assertEqual(result.reason, "low_citation_count")

    def test_evaluate_citation_dimension_includes_section_distribution(self):
        final_text = """## Introduction

Background context.

## Main body

Evidence synthesis.
"""
        reference = {
            "citation_manifest": [
                {"citation_id": "C1", "section_key": "introduction"},
                {"citation_id": "C2", "section_key": "main_body"},
                {"citation_id": "C3", "section_key": "main_body"},
                {"citation_id": "C4", "section_key": "main_body"},
                {"citation_id": "C5", "section_key": "main_body"},
                {"citation_id": "C6", "section_key": "main_body"},
            ]
        }

        result = evaluate_citation_dimension(final_text, reference)

        self.assertIn("section_distribution", result["diagnostics"])
        self.assertIn("section_distribution", result["scores"])

    def test_section_distribution_diagnostics_include_six_slot_prior_metadata(self):
        final_text = "Generated article body."
        citation_manifest = [
            *[
                {"citation_id": f"S1_{i}", "chunk_id": "sec1", "section_key": "introduction"}
                for i in range(1, 11)
            ],
            *[
                {"citation_id": f"S2_{i}", "chunk_id": "sec2", "section_key": "main_body"}
                for i in range(1, 21)
            ],
            *[
                {"citation_id": f"S3_{i}", "chunk_id": "sec3", "section_key": "main_body"}
                for i in range(1, 21)
            ],
            *[
                {"citation_id": f"S4_{i}", "chunk_id": "sec4", "section_key": "main_body"}
                for i in range(1, 21)
            ],
            *[
                {"citation_id": f"S5_{i}", "chunk_id": "sec5", "section_key": "main_body"}
                for i in range(1, 21)
            ],
            *[
                {"citation_id": f"S6_{i}", "chunk_id": "sec6", "section_key": "conclusion"}
                for i in range(1, 9)
            ],
        ]
        reference = {
            "outline": {
                "sec1": "Scope, disease context, and chemokine terminology in alopecia areata",
                "sec2": "Chemokine-pathway framework and hair-follicle immune privilege collapse",
                "sec3": "Evidence base and measurement strategies across blood and skin studies",
                "sec4": "Chemokine signatures in alopecia areata across Th1, Th2, and related pathways",
                "sec5": "Biomarker value and therapeutic implications for clinical dermatology",
                "sec6": "Limitations, heterogeneity, and future research priorities",
            },
            "citation_manifest": citation_manifest,
        }

        result = evaluate_citation_dimension(final_text, reference)

        diagnostics = result["diagnostics"]["section_distribution"]
        self.assertEqual(diagnostics["scheme"], "six_slot_task")
        self.assertIn("prior_metadata", diagnostics)
        self.assertIn("prior_version", diagnostics["prior_metadata"])
        self.assertEqual(
            diagnostics["prior_metadata"]["threshold_source"],
            "data_sample/six_slot_citation_priors.json",
        )


class CitationCountScoreTests(unittest.TestCase):
    def test_citation_count_thresholds_are_ordered(self):
        self.assertGreater(CITATION_COUNT_THRESHOLDS["soft_lower"], 0.0)
        self.assertLess(
            CITATION_COUNT_THRESHOLDS["soft_lower"],
            CITATION_COUNT_THRESHOLDS["soft_upper"],
        )
        self.assertLess(
            CITATION_COUNT_THRESHOLDS["hard_lower"],
            CITATION_COUNT_THRESHOLDS["soft_lower"],
        )
        self.assertGreater(
            CITATION_COUNT_THRESHOLDS["hard_upper"],
            CITATION_COUNT_THRESHOLDS["soft_upper"],
        )

    def test_citation_count_gives_full_score_inside_threshold_band(self):
        final_text = (
            "# Test review\n\n"
            "## Introduction\n\n"
            + (
                "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
                "omicron pi rho sigma tau upsilon phi chi psi omega background mechanism "
                "clinical evidence integration hemodynamics patient outcome comparison [1]. "
                * 2
            )
            + "\n\n## Evidence synthesis\n\n"
            + (
                "Therapy pathway cohort subgroup response biomarker physiology severity risk "
                "stratification treatment pathway comparative analysis evidence synthesis "
                "clinical pattern diagnostic trajectory management implication literature "
                "signal uncertainty trial context review framework application [2]. "
                * 2
            )
            + "\n\n## References\n\n1. Ref one.\n2. Ref two.\n"
        )

        result = score_citation_count(final_text)

        self.assertEqual(result.source, "text_regex")
        self.assertEqual(result.penalty, 0.0)
        self.assertEqual(result.score, 1.0)

    def test_citation_count_penalizes_sparse_density(self):
        final_text = (
            "# Test review\n\n"
            "## Introduction\n\n"
            + ("alpha beta gamma delta epsilon zeta eta theta iota kappa " * 80)
            + "[1].\n\n## References\n\n1. Ref one.\n"
        )

        result = score_citation_count(final_text)

        self.assertLess(result.citations_per_1000_words, CITATION_COUNT_THRESHOLDS["soft_lower"])
        self.assertGreater(result.penalty, 0.0)
        self.assertLess(result.score, 1.0)

    def test_citation_count_penalizes_dense_density(self):
        final_text = (
            "# Test review\n\n"
            "## Introduction\n\n"
            + ("alpha [1]. " * 40)
            + "\n\n## References\n\n1. Ref one.\n"
        )

        result = score_citation_count(final_text)

        self.assertGreater(result.citations_per_1000_words, CITATION_COUNT_THRESHOLDS["soft_upper"])
        self.assertGreater(result.penalty, 0.0)
        self.assertLess(result.score, 1.0)

    def test_citation_count_text_regex_fallback_excludes_references(self):
        final_text = """# Test review

## Introduction

Background evidence [1].

## Evidence synthesis

The reviewed cohorts report associations [2,3].

## References

1. Ref one.
2. Ref two.
3. Ref three.
"""

        result = score_citation_count(final_text)

        self.assertEqual(result.source, "text_regex")
        self.assertEqual(result.citation_count_without_references, 3)
        self.assertEqual(result.section_counts["references"], 0)
        self.assertGreater(result.citations_per_1000_words, 0)

    def test_evaluate_citation_dimension_includes_citation_count(self):
        final_text = """# Test review

## Introduction

Background evidence [1].

## Evidence synthesis

The reviewed cohorts report associations [2,3].

## References

1. Ref one.
2. Ref two.
3. Ref three.
"""
        result = evaluate_citation_dimension(final_text, {"constraints": {}})

        self.assertIn("citation_count", result["scores"])
        self.assertIn("citation_count", result["diagnostics"])
        self.assertEqual(result["diagnostics"]["citation_count"]["source"], "text_regex")


class SourceBalanceScoreTests(unittest.TestCase):
    def test_source_balance_thresholds_are_ordered(self):
        self.assertGreaterEqual(SOURCE_BALANCE_THRESHOLDS["soft_upper"], 0.0)
        self.assertLess(
            SOURCE_BALANCE_THRESHOLDS["soft_upper"],
            SOURCE_BALANCE_THRESHOLDS["hard_upper"],
        )
        self.assertLessEqual(SOURCE_BALANCE_THRESHOLDS["hard_upper"], 1.0)

    def test_source_balance_full_score_for_even_mix(self):
        result = score_source_balance(
            "Generated article text.",
            [
                {"citation_id": "C1", "source_id": "S1"},
                {"citation_id": "C2", "source_id": "S2"},
                {"citation_id": "C3", "source_id": "S3"},
                {"citation_id": "C4", "source_id": "S4"},
                {"citation_id": "C5", "source_id": "S5"},
                {"citation_id": "C6", "source_id": "S6"},
            ],
        )

        self.assertEqual(result.source, "citation_manifest")
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.penalty, 0.0)
        self.assertEqual(result.max_single_source_share, 0.1667)

    def test_source_balance_penalizes_high_share(self):
        result = score_source_balance(
            "Generated article text.",
            [
                {"citation_id": "C1", "source_id": "S1"},
                {"citation_id": "C2", "source_id": "S1"},
                {"citation_id": "C3", "source_id": "S1"},
                {"citation_id": "C4", "source_id": "S1"},
                {"citation_id": "C5", "source_id": "S2"},
            ],
        )

        self.assertGreater(result.max_single_source_share, SOURCE_BALANCE_THRESHOLDS["soft_upper"])
        self.assertGreater(result.penalty, 0.0)
        self.assertLess(result.score, 1.0)

    def test_source_balance_manifest_uses_source_id_counts(self):
        result = score_source_balance(
            "Generated article text.",
            [
                {"citation_id": "C1", "source_id": "S1"},
                {"citation_id": "C2", "source_id": "S1"},
                {"citation_id": "C3", "source_id": "S2"},
                {"citation_id": "C4", "source_id": "S3"},
            ],
        )

        self.assertEqual(result.source, "citation_manifest")
        self.assertEqual(result.total_citations, 4)
        self.assertEqual(result.max_single_source_count, 2)
        self.assertEqual(result.dominant_source_id, "S1")
        self.assertEqual(result.max_single_source_share, 0.5)

    def test_evaluate_citation_dimension_includes_source_balance(self):
        final_text = """# Test review

## Introduction

Background evidence [1].

## Evidence synthesis

The reviewed cohorts report associations [2,3].

## References

1. Ref one.
2. Ref two.
3. Ref three.
"""
        result = evaluate_citation_dimension(final_text, {"constraints": {}})

        self.assertIn("source_balance", result["scores"])
        self.assertIn("source_balance", result["diagnostics"])


if __name__ == "__main__":
    unittest.main()
