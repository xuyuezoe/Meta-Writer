"""MetaBench task derived from review article PMC12894325."""

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
    """Return the PMC12894325-derived MetaBench task configuration."""

    spec = TaskSpec(
        task_id="med_pmc12894325",
        topic="Efficacy and safety of first-line immunotherapy and targeted therapy in advanced HCC: a network meta-analysis with subgroup analysis based on HBV and HCV infection",
        domain="clinical evidence synthesis",
        target_words=6018,
        body_target_words=3980,
        expected_sections=6,
        practice_context="evidence-based clinical decision-making",
        organizer="systematic evidence synthesis framework",
        focus_points=[
        "Efficacy and safety of first-line immunotherapy and targeted therapy in advanced HCC",
        "a network meta-analysis with subgroup analysis based on HBV and HCV infection"
],
        extra_must_include=[
        "HCC",
        "HBV",
        "HCV",
        "RCT"
],
    )
    config = build_main_task_config(
        spec,
        session_name="metabench_pmc12894325",
        corpus_dir="./data_sample/med_papers_review_augmented_strict",
        outline_override=dict(CUSTOM_OUTLINE),
    )

    reference = config["reference"]
    constraints = reference["constraints"]
    constraints["section_word_targets"] = {
        "sec1": 864,
        "sec2": 557,
        "sec3": 650,
        "sec4": 708,
        "sec5": 529,
        "sec6": 672
}
    constraints["section_budget_trace"] = {
        "source": "real_article_six_slot_alignment",
        "source_article_path": "external:medical_reviews_300_ready/PMC12894325.md",
        "raw_article_section_count": 3,
        "raw_article_outline": {
                "sec1": "Introduction",
                "sec2": "Materials and methods",
                "sec3": "Discussion"
        },
        "raw_article_section_word_targets": {
                "sec1": 660,
                "sec2": 199,
                "sec3": 2180
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
                "sec1": 660.0,
                "sec2": 425.4095,
                "sec3": 496.2732,
                "sec4": 540.5188,
                "sec5": 403.7985,
                "sec6": 513.0
        },
        "actual_section_word_targets": {
                "sec1": 660.0,
                "sec2": 425.4095,
                "sec3": 496.2732,
                "sec4": 540.5188,
                "sec5": 403.7985,
                "sec6": 513.0
        },
        "slot_prior_floor_factor": 0.35,
        "scaled_to_body_target_words": 3980
}
    constraints["section_prior_scheme"] = "real_article_six_slot_aligned"
    constraints["six_slot_prior_version"] = '52bb45a3787b'

    return config
