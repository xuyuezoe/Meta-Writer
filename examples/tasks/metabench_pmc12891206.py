"""MetaBench task derived from review article PMC12891206."""

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
    """Return the PMC12891206-derived MetaBench task configuration."""

    spec = TaskSpec(
        task_id="med_pmc12891206",
        topic="Interventions to enhance in-home taking medication among older adults with multimorbidity/polypharmacy",
        domain="clinical evidence synthesis",
        target_words=10755,
        body_target_words=7000,
        expected_sections=6,
        practice_context="evidence-based clinical decision-making",
        organizer="systematic evidence synthesis framework",
        focus_points=[
        "aged",
        "drug therapy",
        "frail older adults",
        "home environment"
],
        extra_must_include=[
        "PROSPERO",
        "CRD42024513056",
        "PRISMA",
        "CINAHL"
],
    )
    config = build_main_task_config(
        spec,
        session_name="metabench_pmc12891206",
        corpus_dir="./data_sample/med_papers_review_augmented_strict",
        outline_override=dict(CUSTOM_OUTLINE),
    )

    reference = config["reference"]
    constraints = reference["constraints"]
    for key in ('section_budget_trace', 'six_slot_prior_version'):
        constraints.pop(key, None)

    return config
