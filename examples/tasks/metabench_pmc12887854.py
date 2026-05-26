"""MetaBench task derived from review article PMC12887854."""

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
    """Return the PMC12887854-derived MetaBench task configuration."""

    spec = TaskSpec(
        task_id="med_pmc12887854",
        topic="The effectiveness of exercise interventions on muscle strength and balance function in pre-frail older adults: a systematic review and Bayesian network meta-analysis",
        domain="clinical evidence synthesis",
        target_words=4895,
        body_target_words=2527,
        expected_sections=6,
        practice_context="evidence-based clinical decision-making",
        organizer="systematic evidence synthesis framework",
        focus_points=[
        "balance function",
        "Bayesian network meta-analysis",
        "exercise interventions",
        "muscle strength"
],
        extra_must_include=[
        "SPPB",
        "TUG",
        "MD",
        "CI"
],
    )
    config = build_main_task_config(
        spec,
        session_name="metabench_pmc12887854",
        corpus_dir="./data_sample/med_papers_review_augmented_strict",
        outline_override=dict(CUSTOM_OUTLINE),
    )

    reference = config["reference"]
    constraints = reference["constraints"]
    constraints["section_word_targets"] = {
        "sec1": 700,
        "sec2": 141,
        "sec3": 173,
        "sec4": 184,
        "sec5": 149,
        "sec6": 1180
}
    constraints["section_budget_trace"] = {
        "source": "real_article_six_slot_alignment",
        "source_article_path": "external:medical_reviews_300_ready/PMC12887854.md",
        "raw_article_section_count": 4,
        "raw_article_outline": {
                "sec1": "Introduction",
                "sec2": "Methods",
                "sec3": "Discussion",
                "sec4": "Conclusion"
        },
        "raw_article_section_word_targets": {
                "sec1": 634,
                "sec2": 40,
                "sec3": 1256,
                "sec4": 134
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
                "sec1": 634.0,
                "sec2": 76.7757,
                "sec3": 98.2521,
                "sec4": 103.5061,
                "sec5": 83.4661,
                "sec6": 1068.0
        },
        "actual_section_word_targets": {
                "sec1": 571.9437,
                "sec2": 115.4196,
                "sec3": 141.0388,
                "sec4": 150.6154,
                "sec5": 121.5188,
                "sec6": 963.4636
        },
        "slot_prior_floor_factor": 0.35,
        "scaled_to_body_target_words": 2527
}
    constraints["section_prior_scheme"] = "real_article_six_slot_aligned"
    constraints["six_slot_prior_version"] = '52bb45a3787b'

    return config
