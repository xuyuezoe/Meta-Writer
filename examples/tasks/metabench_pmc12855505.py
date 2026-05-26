"""MetaBench task derived from review article PMC12855505."""

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
    """Return the PMC12855505-derived MetaBench task configuration."""

    spec = TaskSpec(
        task_id="med_pmc12855505",
        topic="Does early pregnancy exposure to macrolide antibiotics lead to major birth defects? A systematic review and meta-analysis",
        domain="clinical evidence synthesis",
        target_words=4685,
        body_target_words=2889,
        expected_sections=6,
        practice_context="evidence-based clinical decision-making",
        organizer="systematic evidence synthesis framework",
        focus_points=[
        "Does early pregnancy exposure to macrolide antibiotics lead to major birth defects? A systematic review and meta-analysis"
],
        extra_must_include=[
        "STATA",
        "OR",
        "CI"
],
    )
    config = build_main_task_config(
        spec,
        session_name="metabench_pmc12855505",
        corpus_dir="./data_sample/med_papers_review_augmented_strict",
        outline_override=dict(CUSTOM_OUTLINE),
    )

    reference = config["reference"]
    constraints = reference["constraints"]
    constraints["section_word_targets"] = {
        "sec1": 486,
        "sec2": 148,
        "sec3": 181,
        "sec4": 194,
        "sec5": 156,
        "sec6": 1724
}
    constraints["section_budget_trace"] = {
        "source": "real_article_six_slot_alignment",
        "source_article_path": "external:medical_reviews_300_ready/PMC12855505.md",
        "raw_article_section_count": 3,
        "raw_article_outline": {
                "sec1": "Introduction",
                "sec2": "Discussion",
                "sec3": "Conclusion"
        },
        "raw_article_section_word_targets": {
                "sec1": 511,
                "sec2": 1919,
                "sec3": 86
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
                "sec1": 511.0,
                "sec2": 25.7419,
                "sec3": 66.3871,
                "sec4": 66.3871,
                "sec5": 33.4839,
                "sec6": 1813.0
        },
        "actual_section_word_targets": {
                "sec1": 423.1616,
                "sec2": 129.1522,
                "sec3": 157.8196,
                "sec4": 168.5356,
                "sec5": 135.977,
                "sec6": 1501.354
        },
        "slot_prior_floor_factor": 0.35,
        "scaled_to_body_target_words": 2889
}
    constraints["section_prior_scheme"] = "real_article_six_slot_aligned"
    constraints["six_slot_prior_version"] = '52bb45a3787b'

    return config
