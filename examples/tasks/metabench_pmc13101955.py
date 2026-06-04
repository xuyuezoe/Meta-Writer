"""MetaBench task derived from review article PMC13101955."""

from __future__ import annotations

from typing import Dict

from meta_bench import TaskSpec, build_main_task_config


CUSTOM_OUTLINE = {
    "sec1": "Scope, terminology, and practice context",
    "sec2": "Mechanistic and organizing framework",
    "sec3": "Evidence base, methods, and measurement strategy",
    "sec4": "Findings and cross-study synthesis",
    "sec5": "Clinical implications and interpretive discussion",
    "sec6": "Limitations, heterogeneity, and future research priorities",
}


def get_task_config() -> Dict[str, object]:
    """Return the PMC13101955-derived MetaBench task configuration."""

    spec = TaskSpec(
        task_id="med_pmc13101955",
        topic="Gestational Diabetes and Long\u2010Term Risk of Maternal Kidney Disease: Systematic Review and Meta\u2010Analysis of Population Base Cohort Studies",
        domain="clinical evidence synthesis",
        target_words=7589,
        body_target_words=5400,
        expected_sections=6,
        practice_context="evidence-based clinical decision-making",
        organizer="systematic evidence synthesis framework",
        focus_points=[
                "chronic kidney disease",
                "gestational diabetes",
                "maternal",
                "renal outcomes",
        ],
        extra_must_include=[
                "GDM",
                "CKD",
                "MEDLINE",
                "EMBASE",
        ],
    )
    config = build_main_task_config(
        spec,
        session_name="metabench_pmc13101955",
        corpus_dir="./data_sample/med_papers_review_augmented_strict",
        outline_override=dict(CUSTOM_OUTLINE),
    )

    reference = config["reference"]
    constraints = reference["constraints"]
    for key in ('section_budget_trace', 'six_slot_prior_version'):
        constraints.pop(key, None)

    return config
