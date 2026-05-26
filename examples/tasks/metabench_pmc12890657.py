"""MetaBench task derived from review article PMC12890657."""

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
    """Return the PMC12890657-derived MetaBench task configuration."""

    spec = TaskSpec(
        task_id="med_pmc12890657",
        topic="Inflammatory bowel disease and the risk of all caused or specific fracture: a meta-epidemiologic study",
        domain="clinical evidence synthesis",
        target_words=2542,
        body_target_words=923,
        expected_sections=6,
        practice_context="evidence-based clinical decision-making",
        organizer="evidence comparison framework",
        focus_points=[
        "cohort study",
        "Crohn’s disease",
        "fracture",
        "inflammatory bowel disease"
],
        extra_must_include=[
        "IBD",
        "PRISMA",
        "RR",
        "CI"
],
    )
    config = build_main_task_config(
        spec,
        session_name="metabench_pmc12890657",
        corpus_dir="./data_sample/med_papers_review_augmented_strict",
        outline_override=dict(CUSTOM_OUTLINE),
    )

    reference = config["reference"]
    constraints = reference["constraints"]
    constraints["section_word_targets"] = {
        "sec1": 618,
        "sec2": 46,
        "sec3": 56,
        "sec4": 60,
        "sec5": 48,
        "sec6": 95
}
    constraints["section_budget_trace"] = {
        "source": "real_article_six_slot_alignment",
        "source_article_path": "external:medical_reviews_300_ready/PMC12890657.md",
        "raw_article_section_count": 3,
        "raw_article_outline": {
                "sec1": "Introduction",
                "sec2": "Methods",
                "sec3": "Conclusion"
        },
        "raw_article_section_word_targets": {
                "sec1": 444,
                "sec2": 20,
                "sec3": 68
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
                "sec1": 444.0,
                "sec2": 2.6815,
                "sec3": 7.7218,
                "sec4": 6.9153,
                "sec5": 2.6815,
                "sec6": 68.0
        },
        "actual_section_word_targets": {
                "sec1": 356.2562,
                "sec2": 26.4604,
                "sec3": 32.3337,
                "sec4": 34.5292,
                "sec5": 27.8587,
                "sec6": 54.5618
        },
        "slot_prior_floor_factor": 0.35,
        "scaled_to_body_target_words": 923
}
    constraints["section_prior_scheme"] = "real_article_six_slot_aligned"
    constraints["six_slot_prior_version"] = '52bb45a3787b'

    return config
