"""MetaBench task derived from review article PMC12909487."""

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
    """Return the PMC12909487-derived MetaBench task configuration."""

    spec = TaskSpec(
        task_id="med_pmc12909487",
        topic="Cardiovascular diseases and risk factors associated with sudden cardiac death in amateur athletes: a scoping review",
        domain="clinical evidence synthesis",
        target_words=3818,
        body_target_words=2842,
        expected_sections=6,
        practice_context="evidence-based clinical decision-making",
        organizer="evidence comparison framework",
        focus_points=[
        "amateur athletes",
        "cardiovascular diseases",
        "risk factors",
        "sudden cardiac arrest"
],
        extra_must_include=[
        "SCD",
        "PRISMA-",
        "JBI",
        "PCC"
],
    )
    config = build_main_task_config(
        spec,
        session_name="metabench_pmc12909487",
        corpus_dir="./data_sample/med_papers_review_augmented_strict",
        outline_override=dict(CUSTOM_OUTLINE),
    )

    reference = config["reference"]
    constraints = reference["constraints"]
    constraints["section_word_targets"] = {
        "sec1": 529,
        "sec2": 419,
        "sec3": 526,
        "sec4": 560,
        "sec5": 434,
        "sec6": 374
}
    constraints["section_budget_trace"] = {
        "source": "real_article_six_slot_alignment",
        "source_article_path": "external:medical_reviews_300_ready/PMC12909487.md",
        "raw_article_section_count": 5,
        "raw_article_outline": {
                "sec1": "Introduction",
                "sec2": "Materials and methods",
                "sec3": "Results",
                "sec4": "Discussion",
                "sec5": "Conclusion"
        },
        "raw_article_section_word_targets": {
                "sec1": 451,
                "sec2": 207,
                "sec3": 442,
                "sec4": 1245,
                "sec5": 79
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
                "sec1": 451.0,
                "sec2": 356.9979,
                "sec3": 449.0335,
                "sec4": 477.6869,
                "sec5": 370.2817,
                "sec6": 319.0
        },
        "actual_section_word_targets": {
                "sec1": 451.0,
                "sec2": 356.9979,
                "sec3": 449.0335,
                "sec4": 477.6869,
                "sec5": 370.2817,
                "sec6": 319.0
        },
        "slot_prior_floor_factor": 0.35,
        "scaled_to_body_target_words": 2842
}
    constraints["section_prior_scheme"] = "real_article_six_slot_aligned"
    constraints["six_slot_prior_version"] = '52bb45a3787b'

    return config
