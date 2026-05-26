"""MetaBench task derived from review article PMC12875953."""

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
    """Return the PMC12875953-derived MetaBench task configuration."""

    spec = TaskSpec(
        task_id="med_pmc12875953",
        topic="Evaluating cardiac echocardiographic changes with levothyroxine in hypothyroid patients",
        domain="clinical evidence synthesis",
        target_words=6154,
        body_target_words=4300,
        expected_sections=6,
        practice_context="evidence-based clinical decision-making",
        organizer="systematic evidence synthesis framework",
        focus_points=[
        "Limitations",
        "Evaluating cardiac echocardiographic changes with levothyroxine in hypothyroid patients"
],
        extra_must_include=[
        "L-T4",
        "LV",
        "MD",
        "CI"
],
    )
    config = build_main_task_config(
        spec,
        session_name="metabench_pmc12875953",
        corpus_dir="./data_sample/med_papers_review_augmented_strict",
        outline_override=dict(CUSTOM_OUTLINE),
    )

    reference = config["reference"]
    constraints = reference["constraints"]
    constraints["section_word_targets"] = {
        "sec1": 460,
        "sec2": 523,
        "sec3": 685,
        "sec4": 725,
        "sec5": 577,
        "sec6": 1330
}
    constraints["section_budget_trace"] = {
        "source": "real_article_six_slot_alignment",
        "source_article_path": "external:medical_reviews_300_ready/PMC12875953.md",
        "raw_article_section_count": 5,
        "raw_article_outline": {
                "sec1": "Introduction",
                "sec2": "Methods",
                "sec3": "Discussion",
                "sec4": "Limitations",
                "sec5": "Conclusion"
        },
        "raw_article_section_word_targets": {
                "sec1": 393,
                "sec2": 38,
                "sec3": 2928,
                "sec4": 219,
                "sec5": 92
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
                "sec1": 393.0,
                "sec2": 446.136,
                "sec3": 584.2082,
                "sec4": 618.7783,
                "sec5": 492.8775,
                "sec6": 1135.0
        },
        "actual_section_word_targets": {
                "sec1": 393.0,
                "sec2": 446.136,
                "sec3": 584.2082,
                "sec4": 618.7783,
                "sec5": 492.8775,
                "sec6": 1135.0
        },
        "slot_prior_floor_factor": 0.35,
        "scaled_to_body_target_words": 4300
}
    constraints["section_prior_scheme"] = "real_article_six_slot_aligned"
    constraints["six_slot_prior_version"] = '52bb45a3787b'

    return config
