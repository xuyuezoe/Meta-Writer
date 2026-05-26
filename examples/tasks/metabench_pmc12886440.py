"""MetaBench task derived from review article PMC12886440."""

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
    """Return the PMC12886440-derived MetaBench task configuration."""

    spec = TaskSpec(
        task_id="med_pmc12886440",
        topic="The impact of music intervention during emergency suturing on patients’ pain and anxiety",
        domain="clinical evidence synthesis",
        target_words=3029,
        body_target_words=1797,
        expected_sections=6,
        practice_context="evidence-based clinical decision-making",
        organizer="systematic evidence synthesis framework",
        focus_points=[
        "The impact of music intervention during emergency suturing on patients’ pain and anxiety"
],
        extra_must_include=[
        "GRADE",
        "SMD",
        "CI"
],
    )
    config = build_main_task_config(
        spec,
        session_name="metabench_pmc12886440",
        corpus_dir="./data_sample/med_papers_review_augmented_strict",
        outline_override=dict(CUSTOM_OUTLINE),
    )

    reference = config["reference"]
    constraints = reference["constraints"]
    constraints["section_word_targets"] = {
        "sec1": 374,
        "sec2": 174,
        "sec3": 216,
        "sec4": 220,
        "sec5": 171,
        "sec6": 642
}
    constraints["section_budget_trace"] = {
        "source": "real_article_six_slot_alignment",
        "source_article_path": "external:medical_reviews_300_ready/PMC12886440.md",
        "raw_article_section_count": 3,
        "raw_article_outline": {
                "sec1": "Introduction",
                "sec2": "Discussion",
                "sec3": "Conclusion"
        },
        "raw_article_section_word_targets": {
                "sec1": 317,
                "sec2": 1023,
                "sec3": 183
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
                "sec1": 317.0,
                "sec2": 147.496,
                "sec3": 182.9798,
                "sec4": 186.5069,
                "sec5": 145.0172,
                "sec6": 544.0
        },
        "actual_section_word_targets": {
                "sec1": 317.0,
                "sec2": 147.496,
                "sec3": 182.9798,
                "sec4": 186.5069,
                "sec5": 145.0172,
                "sec6": 544.0
        },
        "slot_prior_floor_factor": 0.35,
        "scaled_to_body_target_words": 1797
}
    constraints["section_prior_scheme"] = "real_article_six_slot_aligned"
    constraints["six_slot_prior_version"] = '52bb45a3787b'

    return config
