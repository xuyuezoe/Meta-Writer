"""MetaBench task derived from review article PMC12875926."""

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
    """Return the PMC12875926-derived MetaBench task configuration."""

    spec = TaskSpec(
        task_id="med_pmc12875926",
        topic="Impact of real-time continuous glucose monitoring on glycaemic control in adults with type 2 diabetes",
        domain="clinical evidence synthesis",
        target_words=4269,
        body_target_words=2114,
        expected_sections=6,
        practice_context="evidence-based clinical decision-making",
        organizer="systematic evidence synthesis framework",
        focus_points=[
        "glycaemic control",
        "real-time continuous glucose monitoring",
        "self-management of diabetes",
        "self-monitoring blood glucose"
],
        extra_must_include=[
        "CINAHL",
        "CRD42025625444"
],
    )
    config = build_main_task_config(
        spec,
        session_name="metabench_pmc12875926",
        corpus_dir="./data_sample/med_papers_review_augmented_strict",
        outline_override=dict(CUSTOM_OUTLINE),
    )

    reference = config["reference"]
    constraints = reference["constraints"]
    constraints["section_word_targets"] = {
        "sec1": 485,
        "sec2": 227,
        "sec3": 276,
        "sec4": 299,
        "sec5": 227,
        "sec6": 600
}
    constraints["section_budget_trace"] = {
        "source": "real_article_six_slot_alignment",
        "source_article_path": "external:medical_reviews_300_ready/PMC12875926.md",
        "raw_article_section_count": 5,
        "raw_article_outline": {
                "sec1": "Introduction",
                "sec2": "Results",
                "sec3": "Discussion",
                "sec4": "Strengths and limitations",
                "sec5": "Conclusion"
        },
        "raw_article_section_word_targets": {
                "sec1": 396,
                "sec2": 168,
                "sec3": 870,
                "sec4": 158,
                "sec5": 134
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
                "sec1": 396.0,
                "sec2": 185.0112,
                "sec3": 225.8068,
                "sec4": 243.7758,
                "sec5": 185.4062,
                "sec6": 490.0
        },
        "actual_section_word_targets": {
                "sec1": 396.0,
                "sec2": 185.0112,
                "sec3": 225.8068,
                "sec4": 243.7758,
                "sec5": 185.4062,
                "sec6": 490.0
        },
        "slot_prior_floor_factor": 0.35,
        "scaled_to_body_target_words": 2114
}
    constraints["section_prior_scheme"] = "real_article_six_slot_aligned"
    constraints["six_slot_prior_version"] = '52bb45a3787b'

    return config
