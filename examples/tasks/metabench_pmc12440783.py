"""MetaBench task derived from review article PMC12440783."""

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
    """Return the PMC12440783-derived MetaBench task configuration."""

    spec = TaskSpec(
        task_id="med_pmc12440783",
        topic="chemokines in alopecia areata: recruiting immune cells toward the hair follicle",
        domain="clinical evidence synthesis",
        target_words=6083,
        body_target_words=2916,
        expected_sections=6,
        practice_context="evidence-based clinical decision-making",
        organizer="systematic evidence synthesis framework",
        focus_points=[
        "chemokines in alopecia areata",
        "recruiting immune cells toward the hair follicle"
],
        extra_must_include=[
        "AA",
        "CXCL9",
        "CXCL10",
        "CCL5"
],
    )
    config = build_main_task_config(
        spec,
        session_name="metabench_pmc12440783",
        corpus_dir="./data_sample/med_papers_review_augmented_strict",
        outline_override=dict(CUSTOM_OUTLINE),
    )

    reference = config["reference"]
    constraints = reference["constraints"]
    constraints["section_word_targets"] = {
        "sec1": 654,
        "sec2": 377,
        "sec3": 457,
        "sec4": 525,
        "sec5": 377,
        "sec6": 526
}
    constraints["section_budget_trace"] = {
        "source": "real_article_six_slot_alignment",
        "source_article_path": "external:medical_reviews_300_ready/PMC12440783.md",
        "raw_article_section_count": 5,
        "raw_article_outline": {
                "sec1": "Introduction",
                "sec2": "Materials and methods",
                "sec3": "Results",
                "sec4": "Discussion",
                "sec5": "Conclusion"
        },
        "raw_article_section_word_targets": {
                "sec1": 571,
                "sec2": 417,
                "sec3": 265,
                "sec4": 1076,
                "sec5": 216
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
                "sec1": 571.0,
                "sec2": 329.2199,
                "sec3": 399.093,
                "sec4": 457.9986,
                "sec5": 328.6885,
                "sec6": 459.0
        },
        "actual_section_word_targets": {
                "sec1": 571.0,
                "sec2": 329.2199,
                "sec3": 399.093,
                "sec4": 457.9986,
                "sec5": 328.6885,
                "sec6": 459.0
        },
        "slot_prior_floor_factor": 0.35,
        "scaled_to_body_target_words": 2916
}
    constraints["section_prior_scheme"] = "real_article_six_slot_aligned"
    constraints["six_slot_prior_version"] = '52bb45a3787b'

    return config
