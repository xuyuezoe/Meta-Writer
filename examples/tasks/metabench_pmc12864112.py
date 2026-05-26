"""MetaBench task derived from review article PMC12864112."""

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
    """Return the PMC12864112-derived MetaBench task configuration."""

    spec = TaskSpec(
        task_id="med_pmc12864112",
        topic="Immune checkpoint inhibitors-induced thyroid dysfunction improves the prognosis of patients with lung cancer: a meta-analysis and systematic review",
        domain="clinical evidence synthesis",
        target_words=4362,
        body_target_words=2046,
        expected_sections=6,
        practice_context="evidence-based clinical decision-making",
        organizer="systematic evidence synthesis framework",
        focus_points=[
        "immune checkpoint inhibitor",
        "lung cancer",
        "meta-analysis",
        "prognosis"
],
        extra_must_include=[
        "ICI",
        "CNKI",
        "ICI-",
        "OS"
],
    )
    config = build_main_task_config(
        spec,
        session_name="metabench_pmc12864112",
        corpus_dir="./data_sample/med_papers_review_augmented_strict",
        outline_override=dict(CUSTOM_OUTLINE),
    )

    reference = config["reference"]
    constraints = reference["constraints"]
    constraints["section_word_targets"] = {
        "sec1": 703,
        "sec2": 246,
        "sec3": 279,
        "sec4": 296,
        "sec5": 245,
        "sec6": 277
}
    constraints["section_budget_trace"] = {
        "source": "real_article_six_slot_alignment",
        "source_article_path": "external:medical_reviews_300_ready/PMC12864112.md",
        "raw_article_section_count": 4,
        "raw_article_outline": {
                "sec1": "Introduction",
                "sec2": "Materials and methods",
                "sec3": "Discussion",
                "sec4": "Conclusion"
        },
        "raw_article_section_word_targets": {
                "sec1": 538,
                "sec2": 27,
                "sec3": 918,
                "sec4": 83
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
                "sec1": 538.0,
                "sec2": 188.2976,
                "sec3": 213.7591,
                "sec4": 226.1222,
                "sec5": 187.821,
                "sec6": 212.0
        },
        "actual_section_word_targets": {
                "sec1": 538.0,
                "sec2": 188.2976,
                "sec3": 213.7591,
                "sec4": 226.1222,
                "sec5": 187.821,
                "sec6": 212.0
        },
        "slot_prior_floor_factor": 0.35,
        "scaled_to_body_target_words": 2046
}
    constraints["section_prior_scheme"] = "real_article_six_slot_aligned"
    constraints["six_slot_prior_version"] = '52bb45a3787b'

    return config
