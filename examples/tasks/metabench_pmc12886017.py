"""MetaBench task derived from review article PMC12886017."""

from __future__ import annotations

from typing import Dict

from meta_bench import TaskSpec, build_main_task_config


CUSTOM_OUTLINE = {
    "sec1": "Scope, terminology, and practice context",
    "sec2": "Mechanistic and organizing framework",
    "sec3": "Evidence base, methods, and measurement strategy",
    "sec4": "Findings and cross-study synthesis",
    "sec5": "Clinical implications and interpretive discussion",
    "sec6": "Limitations, heterogeneity, and future research priorities"
}


def get_task_config() -> Dict[str, object]:
    """Return the PMC12886017-derived MetaBench task configuration."""

    spec = TaskSpec(
        task_id="med_pmc12886017",
        topic="Diagnostic accuracy of machine learning for endometriosis",
        domain="clinical evidence synthesis",
        target_words=2842,
        body_target_words=920,
        expected_sections=6,
        practice_context="evidence-based clinical decision-making",
        organizer="systematic evidence synthesis framework",
        focus_points=[
        "Correction note",
        "Diagnostic accuracy of machine learning"
],
        extra_must_include=[
        "ML",
        "AUC",
        "CI",
        "CRD42024605113"
],
    )
    config = build_main_task_config(
        spec,
        session_name="metabench_pmc12886017",
        corpus_dir="./data_sample/med_papers_review_augmented_strict",
        outline_override=dict(CUSTOM_OUTLINE),
    )

    reference = config["reference"]
    constraints = reference["constraints"]
    constraints["section_word_targets"] = {
        "sec1": 571,
        "sec2": 46,
        "sec3": 56,
        "sec4": 59,
        "sec5": 48,
        "sec6": 140
}
    constraints["section_budget_trace"] = {
        "source": "real_article_six_slot_alignment",
        "source_article_path": "external:medical_reviews_300_ready/PMC12886017.md",
        "raw_article_section_count": 3,
        "raw_article_outline": {
                "sec1": "Introduction",
                "sec2": "Conclusions",
                "sec3": "Correction note"
        },
        "raw_article_section_word_targets": {
                "sec1": 408,
                "sec2": 100,
                "sec3": 17
        },
        "six_slot_order": [
                "scope_context",
                "framework_mechanism",
                "evidence_methods",
                "findings_synthesis",
                "implications_discussion",
                "limitations_future"
        ],
        "six_slot_outline": {
                "sec1": "Scope, terminology, and practice context",
                "sec2": "Mechanistic and organizing framework",
                "sec3": "Evidence base, methods, and measurement strategy",
                "sec4": "Findings and cross-study synthesis",
                "sec5": "Clinical implications and interpretive discussion",
                "sec6": "Limitations, heterogeneity, and future research priorities"
        },
        "raw_six_slot_word_targets": {
                "sec1": 408.0,
                "sec2": 2.375,
                "sec3": 6.125,
                "sec4": 6.125,
                "sec5": 2.375,
                "sec6": 100.0
        },
        "actual_section_word_targets": {
                "sec1": 326.0069,
                "sec2": 26.0035,
                "sec3": 31.7754,
                "sec4": 33.933,
                "sec5": 27.3776,
                "sec6": 79.9037
        },
        "slot_prior_floor_factor": 0.35,
        "scaled_to_body_target_words": 920
}
    constraints["section_prior_scheme"] = "real_article_six_slot_aligned"
    constraints["six_slot_prior_version"] = '52bb45a3787b'

    return config
