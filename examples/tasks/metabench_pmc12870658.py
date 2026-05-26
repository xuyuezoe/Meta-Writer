"""MetaBench task derived from review article PMC12870658."""

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
    """Return the PMC12870658-derived MetaBench task configuration."""

    spec = TaskSpec(
        task_id="med_pmc12870658",
        topic="Octreotide long-acting release in the treatment of autosomal dominant polycystic kidney disease",
        domain="clinical evidence synthesis",
        target_words=3594,
        body_target_words=2048,
        expected_sections=6,
        practice_context="evidence-based clinical decision-making",
        organizer="systematic evidence synthesis framework",
        focus_points=[
        "Study limitation",
        "Octreotide long-acting release in the treatment of autosomal dominant polycystic kidney disease"
],
        extra_must_include=[
        "ADPKD",
        "LAR",
        "GFR",
        "SMD"
],
    )
    config = build_main_task_config(
        spec,
        session_name="metabench_pmc12870658",
        corpus_dir="./data_sample/med_papers_review_augmented_strict",
        outline_override=dict(CUSTOM_OUTLINE),
    )

    reference = config["reference"]
    constraints = reference["constraints"]
    constraints["section_word_targets"] = {
        "sec1": 439,
        "sec2": 110,
        "sec3": 134,
        "sec4": 143,
        "sec5": 115,
        "sec6": 1107
}
    constraints["section_budget_trace"] = {
        "source": "real_article_six_slot_alignment",
        "source_article_path": "external:medical_reviews_300_ready/PMC12870658.md",
        "raw_article_section_count": 4,
        "raw_article_outline": {
                "sec1": "Introduction",
                "sec2": "Discussion",
                "sec3": "Study limitation",
                "sec4": "Conclusion"
        },
        "raw_article_section_word_targets": {
                "sec1": 406,
                "sec2": 1063,
                "sec3": 102,
                "sec4": 65
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
                "sec1": 406.0,
                "sec2": 27.753,
                "sec3": 71.5736,
                "sec4": 75.747,
                "sec5": 31.9264,
                "sec6": 1023.0
        },
        "actual_section_word_targets": {
                "sec1": 350.7983,
                "sec2": 87.6235,
                "sec3": 107.073,
                "sec4": 114.3433,
                "sec5": 92.2538,
                "sec6": 883.9081
        },
        "slot_prior_floor_factor": 0.35,
        "scaled_to_body_target_words": 2048
}
    constraints["section_prior_scheme"] = "real_article_six_slot_aligned"
    constraints["six_slot_prior_version"] = '52bb45a3787b'

    return config
