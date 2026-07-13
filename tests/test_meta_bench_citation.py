import unittest

from meta_bench.citation import (
    CITATION_COUNT_THRESHOLDS,
    ClaimCitationQualityDecision,
    CitationChunk,
    CitationEvent,
    CitationQualityDecision,
    build_citation_quality_judge_prompt,
    classify_chunk_role,
    event_source_fidelity_penalty,
    evaluate_citation_dimension,
    estimate_citation_quality_claim_count,
    infer_claim_strength,
    infer_evidence_strength,
    score_citation_count,
    score_citation_quality_f1,
    score_section_distribution,
    score_source_balance,
    score_source_fidelity,
    split_into_seven_sections,
)


class StaticCitationQualityJudge:
    def __init__(self, supported_by_id):
        self.supported_by_id = supported_by_id

    def judge_citation_quality(self, *, claim, source_excerpt, source_id, citation_id):
        supported = self.supported_by_id.get(citation_id, False)
        return CitationQualityDecision(
            citation_id=citation_id,
            claim_key=claim.casefold() or citation_id,
            source_id=source_id,
            supported=supported,
            necessary=supported,
            rationale="static test decision",
        )


class StaticClaimCitationQualityJudge:
    def __init__(self, supported_claims):
        self.supported_claims = supported_claims

    def judge_claim_citation_quality(self, *, claim, sources):
        supported = self.supported_claims.get(claim, False)
        citation_ids = [str(source.get("citation_id", "")) for source in sources]
        return ClaimCitationQualityDecision(
            claim_key=claim.casefold(),
            claim=claim,
            citation_ids=citation_ids,
            source_ids=[str(source.get("source_id", "")) for source in sources],
            supported=supported,
            necessary_citation_ids=citation_ids if supported else [],
            rationale="static claim-level test decision",
        )


class RecordingClaimCitationQualityJudge:
    def __init__(self):
        self.calls = []

    def judge_claim_citation_quality(self, *, claim, sources):
        self.calls.append({"claim": claim, "sources": sources})
        citation_ids = [str(source.get("citation_id", "")) for source in sources]
        return ClaimCitationQualityDecision(
            claim_key=claim.casefold(),
            claim=claim,
            citation_ids=citation_ids,
            source_ids=[str(source.get("source_id", "")) for source in sources],
            supported=True,
            necessary_citation_ids=list(dict.fromkeys(citation_ids)),
            rationale="recorded",
        )


class FailingCitationQualityJudge:
    def judge_claim_citation_quality(self, *, claim, sources):
        raise AssertionError("procedural claims should not be judged")


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
    def test_estimate_citation_quality_claim_count_counts_body_sentences(self):
        text = """# Review title

This intervention reduced symptom burden in randomized clinical trials. It also improved functional status across multiple patient-reported outcome measures.

[1]

Short.

References
"""

        self.assertEqual(estimate_citation_quality_claim_count(text), 2)

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

    def test_evidence_text_cues_raise_strength_by_one_level(self):
        event = CitationEvent(
            citation_id="C1",
            chunk_id="sec1",
            source_id="S1",
            source_type="mechanistic",
            claim_role="unknown",
            claim_span="The evidence suggests a pathway.",
            source_excerpt="A meta-analysis of randomized trials reported benefit.",
        )

        self.assertEqual(infer_evidence_strength(event), 2)

    def test_one_level_strength_gap_gets_mild_penalty(self):
        event = CitationEvent(
            citation_id="C1",
            chunk_id="sec1",
            source_id="S1",
            source_type="cohort",
            claim_role="effect",
            claim_span="The exposure improves outcomes.",
            source_excerpt="",
        )

        self.assertAlmostEqual(event_source_fidelity_penalty(event), 0.2)

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
        self.assertAlmostEqual(result.penalty, 0.45)
        self.assertAlmostEqual(result.score, 0.55)

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
        diagnostics = result["diagnostics"]["source_fidelity_legacy"]
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
            result["diagnostics"]["citation_quality_f1"]["reason"],
            "missing_citation_manifest",
        )

    def test_score_citation_quality_f1_uses_lira_scaled_recall(self):
        final_text = "Generated article text."
        citation_manifest = [
            {
                "citation_id": "C1",
                "source_id": "S1",
                "claim_span": "Claim one",
                "source_excerpt": "Evidence for claim one.",
            },
            {
                "citation_id": "C2",
                "source_id": "S2",
                "claim_span": "Claim two",
                "source_excerpt": "Unrelated evidence.",
            },
        ]

        result = score_citation_quality_f1(
            final_text,
            citation_manifest,
            StaticCitationQualityJudge({"C1": True, "C2": False}),
        )

        self.assertEqual(result.claim_count, 2)
        self.assertEqual(result.citation_count, 2)
        self.assertAlmostEqual(result.precision, 0.5)
        self.assertAlmostEqual(result.recall, 0.5)
        self.assertLess(result.scaled_recall, result.recall)

    def test_score_citation_quality_f1_accepts_benchmark_level_claim_count(self):
        final_text = "Generated article text."
        citation_manifest = [
            {
                "citation_id": "C1",
                "source_id": "S1",
                "claim_span": "Claim one",
                "source_excerpt": "Evidence for claim one.",
            },
            {
                "citation_id": "C2",
                "source_id": "S2",
                "claim_span": "Claim two",
                "source_excerpt": "Evidence for claim two.",
            },
        ]

        result = score_citation_quality_f1(
            final_text,
            citation_manifest,
            StaticCitationQualityJudge({"C1": True, "C2": True}),
            scaling_claim_count=100,
        )

        self.assertEqual(result.claim_count, 2)
        self.assertEqual(result.scaling_claim_count, 100)
        self.assertAlmostEqual(result.scaling, 0.6321, places=4)

    def test_citation_quality_prompt_uses_supporting_citation_ids(self):
        prompt = build_citation_quality_judge_prompt(
            claim="Claim one",
            sources=[
                {
                    "citation_id": "C1",
                    "source_id": "S1",
                    "source_excerpt": "Evidence.",
                }
            ],
        )

        self.assertIn("supporting_citation_ids", prompt)
        self.assertIn("related/supportive", prompt)
        self.assertNotIn('"necessary_citation_ids"', prompt)

    def test_score_citation_quality_f1_selects_claim_relevant_reference_chunks(self):
        final_text = "Generated article text."
        irrelevant = " ".join(
            f"Background sentence {index} about unrelated metabolism."
            for index in range(80)
        )
        relevant = (
            "Interleukin 17 blockade improved psoriasis severity in randomized "
            "clinical trials and reduced inflammatory skin lesions."
        )
        citation_manifest = [
            {
                "citation_id": "C1",
                "source_id": "S1",
                "claim_span": "Interleukin 17 blockade improved psoriasis severity",
                "source_excerpt": f"{irrelevant} {relevant}",
            }
        ]
        judge = RecordingClaimCitationQualityJudge()

        result = score_citation_quality_f1(
            final_text,
            citation_manifest,
            judge,
        )

        self.assertEqual(result.supported_claim_count, 1)
        self.assertTrue(judge.calls)
        joined_sources = "\n".join(
            source["source_excerpt"] for source in judge.calls[0]["sources"]
        )
        self.assertIn("Interleukin 17 blockade", joined_sources)

    def test_score_citation_quality_f1_excludes_review_procedure_claims(self):
        final_text = "Generated article text."
        citation_manifest = [
            {
                "citation_id": "C1",
                "source_id": "S1",
                "claim_span": (
                    "Boolean operators combined these pillars to maximize "
                    "sensitivity in the literature search strategy"
                ),
                "source_excerpt": "Biomedical source text.",
            }
        ]

        result = score_citation_quality_f1(
            final_text,
            citation_manifest,
            FailingCitationQualityJudge(),
        )

        self.assertEqual(result.claim_count, 0)
        self.assertEqual(result.citation_count, 0)
        self.assertEqual(result.excluded_claim_count, 1)
        self.assertEqual(result.excluded_citation_count, 1)
        self.assertEqual(result.score, 0.0)

    def test_score_citation_quality_f1_excludes_review_framing_claims(self):
        final_text = "Generated article text."
        citation_manifest = [
            {
                "citation_id": "C1",
                "source_id": "S1",
                "claim_span": (
                    "This review aims to systematically synthesize the available "
                    "clinical evidence for evidence-based clinical decision-making"
                ),
                "source_excerpt": "Biomedical source text.",
            }
        ]

        result = score_citation_quality_f1(
            final_text,
            citation_manifest,
            FailingCitationQualityJudge(),
        )

        self.assertEqual(result.claim_count, 0)
        self.assertEqual(result.excluded_claim_count, 1)
        self.assertEqual(result.excluded_citation_count, 1)

    def test_score_citation_quality_f1_excludes_numeric_fragments(self):
        final_text = "Generated article text."
        citation_manifest = [
            {
                "citation_id": "C1",
                "source_id": "S1",
                "claim_span": "62",
                "source_excerpt": "Biomedical source text.",
            }
        ]

        result = score_citation_quality_f1(
            final_text,
            citation_manifest,
            FailingCitationQualityJudge(),
        )

        self.assertEqual(result.claim_count, 0)
        self.assertEqual(result.excluded_claim_count, 1)
        self.assertEqual(result.excluded_citation_count, 1)

    def test_score_citation_quality_f1_judges_reference_set_per_claim(self):
        final_text = "Generated article text."
        citation_manifest = [
            {
                "citation_id": "C1",
                "source_id": "S1",
                "claim_span": "Combined claim",
                "source_excerpt": "Evidence for the first part.",
            },
            {
                "citation_id": "C2",
                "source_id": "S2",
                "claim_span": "Combined claim",
                "source_excerpt": "Evidence for the second part.",
            },
        ]

        result = score_citation_quality_f1(
            final_text,
            citation_manifest,
            StaticClaimCitationQualityJudge({"Combined claim": True}),
        )

        self.assertEqual(result.claim_count, 1)
        self.assertEqual(result.citation_count, 2)
        self.assertEqual(result.supported_claim_count, 1)
        self.assertEqual(result.necessary_citation_count, 2)
        self.assertAlmostEqual(result.precision, 1.0)
        self.assertAlmostEqual(result.recall, 1.0)
        self.assertEqual(len(result.claim_decisions), 1)

    def test_score_citation_quality_f1_splits_obvious_multi_sentence_claims(self):
        final_text = "Generated article text."
        citation_manifest = [
            {
                "citation_id": "C1",
                "source_id": "S1",
                "claim_span": (
                    "The first intervention improved symptom scores in adults. "
                    "The second intervention reduced follow-up hospitalizations."
                ),
                "source_excerpt": "Evidence for both intervention effects.",
            },
        ]

        result = score_citation_quality_f1(
            final_text,
            citation_manifest,
            StaticClaimCitationQualityJudge(
                {
                    "The first intervention improved symptom scores in adults.": True,
                    "The second intervention reduced follow-up hospitalizations.": True,
                }
            ),
        )

        self.assertEqual(result.claim_count, 2)
        self.assertEqual(result.citation_count, 2)
        self.assertEqual(result.supported_claim_count, 2)
        self.assertEqual(result.necessary_citation_count, 2)

    def test_evaluate_citation_dimension_includes_citation_quality_with_judge(self):
        final_text = "Generated article text."
        reference = {
            "citation_quality_scaling_claim_count": 2,
            "citation_quality_scaling_claim_count_source": "article_sentences",
            "citation_manifest": [
                {
                    "citation_id": "C1",
                    "source_id": "S1",
                    "claim_span": "Claim one",
                    "source_excerpt": "Evidence for claim one.",
                }
            ]
        }

        result = evaluate_citation_dimension(
            final_text,
            reference,
            citation_quality_judge=StaticCitationQualityJudge({"C1": True}),
        )

        self.assertIn("citation_quality_f1", result["scores"])
        self.assertIn("citation_quality_f1", result["diagnostics"])
        self.assertEqual(
            result["diagnostics"]["citation_quality_f1"][
                "scaling_claim_count_source"
            ],
            "article_sentences",
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

    def test_citation_count_thresholds_use_central_empirical_band(self):
        self.assertAlmostEqual(CITATION_COUNT_THRESHOLDS["soft_lower"], 21.780305082175012)
        self.assertAlmostEqual(CITATION_COUNT_THRESHOLDS["soft_upper"], 53.88519249312971)
        self.assertAlmostEqual(CITATION_COUNT_THRESHOLDS["hard_lower"], 15.047717849932427)
        self.assertAlmostEqual(CITATION_COUNT_THRESHOLDS["hard_upper"], 71.44435519935337)

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
        self.assertEqual(result.method, "normalized_hhi")
        self.assertAlmostEqual(result.concentration_hhi, 1 / 6, places=4)

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

        self.assertEqual(result.method, "normalized_hhi")
        self.assertGreater(result.penalty, 0.0)
        self.assertLess(result.score, 1.0)
        self.assertAlmostEqual(result.concentration_hhi, 0.68)

    def test_source_balance_single_source_scores_zero(self):
        result = score_source_balance(
            "Generated article text.",
            [
                {"citation_id": "C1", "source_id": "S1"},
                {"citation_id": "C2", "source_id": "S1"},
            ],
        )

        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.penalty, 1.0)
        self.assertEqual(result.cited_source_count, 1)

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
