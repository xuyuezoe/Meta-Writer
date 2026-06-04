"""MetaBench task derived from review article PMC13021469."""

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
    """Return the PMC13021469-derived MetaBench task configuration."""

    spec = TaskSpec(
        task_id="med_pmc13021469",
        topic="Metabolic-associated steatotic liver disease in children and adolescents: a scoping review and narrative synthesis of epidemiology, risk factors, and screening approaches with emerging implications for sub-Saharan Africa",
        domain="clinical evidence synthesis",
        target_words=9191,
        body_target_words=6238,
        expected_sections=6,
        practice_context="evidence-based clinical decision-making",
        organizer="evidence comparison framework",
        focus_points=[
                "Associated Data",
                "Metabolic-associated steatotic liver disease in children and adolescents",
                "a scoping review and narrative synthesis of epidemiology, risk factors, and screening approaches with emerging implications",
                "sub-Saharan Africa",
        ],
        extra_must_include=[
                "MASLD",
                "SSA",
                "PRISMA-",
                "JBI",
        ],
    )
    config = build_main_task_config(
        spec,
        session_name="metabench_pmc13021469",
        corpus_dir="./data_sample/med_papers_review_augmented_strict",
        outline_override=dict(CUSTOM_OUTLINE),
    )

    reference = config["reference"]
    constraints = reference["constraints"]
    for key in ('section_budget_trace', 'six_slot_prior_version'):
        constraints.pop(key, None)

    return config
