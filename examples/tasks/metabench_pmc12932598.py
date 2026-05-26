"""MetaBench task derived from review article PMC12932598."""

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
    """Return the PMC12932598-derived MetaBench task configuration."""

    spec = TaskSpec(
        task_id="med_pmc12932598",
        topic="Association between vaccination and myasthenia gravis",
        domain="clinical evidence synthesis",
        target_words=2441,
        body_target_words=1159,
        expected_sections=6,
        practice_context="evidence-based clinical decision-making",
        organizer="systematic evidence synthesis framework",
        focus_points=[
        "Methods and analysis",
        "Association between vaccination and myasthenia gravis"
],
        extra_must_include=[
        "MG",
        "CNKI",
        "VIP",
        "NOS"
],
    )
    config = build_main_task_config(
        spec,
        session_name="metabench_pmc12932598",
        corpus_dir="./data_sample/med_papers_review_augmented_strict",
        outline_override=dict(CUSTOM_OUTLINE),
    )

    reference = config["reference"]
    constraints = reference["constraints"]
    constraints["section_word_targets"] = {
        "sec1": 398,
        "sec2": 102,
        "sec3": 121,
        "sec4": 133,
        "sec5": 110,
        "sec6": 295
}
    constraints["section_budget_trace"] = {
        "source": "real_article_six_slot_alignment",
        "source_article_path": "external:medical_reviews_300_ready/PMC12932598.md",
        "raw_article_section_count": 4,
        "raw_article_outline": {
                "sec1": "Introduction",
                "sec2": "Methods and analysis",
                "sec3": "Discussion",
                "sec4": "Conclusions"
        },
        "raw_article_section_word_targets": {
                "sec1": 270,
                "sec2": 38,
                "sec3": 278,
                "sec4": 200
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
                "sec1": 270.0,
                "sec2": 68.7707,
                "sec3": 82.2946,
                "sec4": 90.3945,
                "sec5": 74.5402,
                "sec6": 200.0
        },
        "actual_section_word_targets": {
                "sec1": 270.0,
                "sec2": 68.7707,
                "sec3": 82.2946,
                "sec4": 90.3945,
                "sec5": 74.5402,
                "sec6": 200.0
        },
        "slot_prior_floor_factor": 0.35,
        "scaled_to_body_target_words": 1159
}
    constraints["section_prior_scheme"] = "real_article_six_slot_aligned"
    constraints["six_slot_prior_version"] = '52bb45a3787b'

    return config
