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
    for key in ('section_budget_trace', 'six_slot_prior_version'):
        constraints.pop(key, None)

    return config
