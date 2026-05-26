"""MetaBench task derived from review article PMC12883763."""

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
    """Return the PMC12883763-derived MetaBench task configuration."""

    spec = TaskSpec(
        task_id="med_pmc12883763",
        topic="Analysis of evolution of the policy framework and governance mechanisms and their influence on the institutionalisation process of integrated community case management in Burkina Faso between 2010 and 2024: a scoping review",
        domain="clinical evidence synthesis",
        target_words=10987,
        body_target_words=5350,
        expected_sections=6,
        practice_context="evidence-based clinical decision-making",
        organizer="mechanism and evidence framework",
        focus_points=[
        "Burkina Faso",
        "childhood diseases",
        "community dynamics",
        "institutionalisation"
],
        extra_must_include=[
        "READ",
        "HIV",
        "AIDS"
],
    )
    config = build_main_task_config(
        spec,
        session_name="metabench_pmc12883763",
        corpus_dir="./data_sample/med_papers_review_augmented_strict",
        outline_override=dict(CUSTOM_OUTLINE),
    )

    reference = config["reference"]
    constraints = reference["constraints"]
    constraints["section_word_targets"] = {
        "sec1": 647,
        "sec2": 690,
        "sec3": 803,
        "sec4": 856,
        "sec5": 688,
        "sec6": 1666
}
    constraints["section_budget_trace"] = {
        "source": "real_article_six_slot_alignment",
        "source_article_path": "external:medical_reviews_300_ready/PMC12883763.md",
        "raw_article_section_count": 5,
        "raw_article_outline": {
                "sec1": "Introduction",
                "sec2": "Methods",
                "sec3": "Results",
                "sec4": "Discussion",
                "sec5": "Conclusion"
        },
        "raw_article_section_word_targets": {
                "sec1": 600,
                "sec2": 275,
                "sec3": 1978,
                "sec4": 1730,
                "sec5": 382
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
                "sec1": 600.0,
                "sec2": 640.7312,
                "sec3": 744.748,
                "sec4": 794.5956,
                "sec5": 638.9252,
                "sec6": 1546.0
        },
        "actual_section_word_targets": {
                "sec1": 600.0,
                "sec2": 640.7312,
                "sec3": 744.748,
                "sec4": 794.5956,
                "sec5": 638.9252,
                "sec6": 1546.0
        },
        "slot_prior_floor_factor": 0.35,
        "scaled_to_body_target_words": 5350
}
    constraints["section_prior_scheme"] = "real_article_six_slot_aligned"
    constraints["six_slot_prior_version"] = '52bb45a3787b'

    return config
