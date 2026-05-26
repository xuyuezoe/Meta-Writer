"""MetaBench task derived from review article PMC12901347."""

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
    """Return the PMC12901347-derived MetaBench task configuration."""

    spec = TaskSpec(
        task_id="med_pmc12901347",
        topic="Association between glycated hemoglobin variability and risk of diabetic kidney disease and diabetic retinopathy in diabetic patients",
        domain="clinical evidence synthesis",
        target_words=8773,
        body_target_words=3993,
        expected_sections=6,
        practice_context="evidence-based clinical decision-making",
        organizer="systematic evidence synthesis framework",
        focus_points=[
        "Association between glycated hemoglobin variability and risk of diabetic kidney disease and diabetic retinopathy in diabetic patients"
],
        extra_must_include=[
        "OR",
        "HR",
        "SD",
        "T1DM"
],
    )
    config = build_main_task_config(
        spec,
        session_name="metabench_pmc12901347",
        corpus_dir="./data_sample/med_papers_review_augmented_strict",
        outline_override=dict(CUSTOM_OUTLINE),
    )

    reference = config["reference"]
    constraints = reference["constraints"]
    constraints["section_word_targets"] = {
        "sec1": 1264,
        "sec2": 331,
        "sec3": 402,
        "sec4": 430,
        "sec5": 358,
        "sec6": 1208
}
    constraints["section_budget_trace"] = {
        "source": "real_article_six_slot_alignment",
        "source_article_path": "external:medical_reviews_300_ready/PMC12901347.md",
        "raw_article_section_count": 3,
        "raw_article_outline": {
                "sec1": "Introduction",
                "sec2": "Discussion",
                "sec3": "Conclusion"
        },
        "raw_article_section_word_targets": {
                "sec1": 1053,
                "sec2": 2222,
                "sec3": 51
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
                "sec1": 1053.0,
                "sec2": 276.0991,
                "sec3": 335.0801,
                "sec4": 357.9674,
                "sec5": 297.8535,
                "sec6": 1006.0
        },
        "actual_section_word_targets": {
                "sec1": 1053.0,
                "sec2": 276.0991,
                "sec3": 335.0801,
                "sec4": 357.9674,
                "sec5": 297.8535,
                "sec6": 1006.0
        },
        "slot_prior_floor_factor": 0.35,
        "scaled_to_body_target_words": 3993
}
    constraints["section_prior_scheme"] = "real_article_six_slot_aligned"
    constraints["six_slot_prior_version"] = '52bb45a3787b'

    return config
