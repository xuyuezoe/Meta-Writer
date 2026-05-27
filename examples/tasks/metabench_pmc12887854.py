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
        target_words=7468,
        body_target_words=5100,
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
    for key in ('section_budget_trace', 'six_slot_prior_version'):
        constraints.pop(key, None)

    return config
