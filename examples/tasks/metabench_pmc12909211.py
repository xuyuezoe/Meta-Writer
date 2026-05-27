"""MetaBench task derived from review article PMC12909211."""

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
    """Return the PMC12909211-derived MetaBench task configuration."""

    spec = TaskSpec(
        task_id="med_pmc12909211",
        topic="The correlation between traditional Chinese medicine constitution and prediabetes",
        domain="clinical evidence synthesis",
        target_words=7554,
        body_target_words=3600,
        expected_sections=6,
        practice_context="evidence-based clinical decision-making",
        organizer="systematic evidence synthesis framework",
        focus_points=[
        "The correlation between traditional Chinese medicine constitution and prediabetes"
],
        extra_must_include=[
        "TCM",
        "PDC",
        "BC",
        "YIDC"
],
    )
    config = build_main_task_config(
        spec,
        session_name="metabench_pmc12909211",
        corpus_dir="./data_sample/med_papers_review_augmented_strict",
        outline_override=dict(CUSTOM_OUTLINE),
    )

    reference = config["reference"]
    constraints = reference["constraints"]
    for key in ('section_budget_trace', 'six_slot_prior_version'):
        constraints.pop(key, None)

    return config
