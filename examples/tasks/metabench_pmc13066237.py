"""MetaBench task derived from review article PMC13066237."""

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
    """Return the PMC13066237-derived MetaBench task configuration."""

    spec = TaskSpec(
        task_id="med_pmc13066237",
        topic="Clinicopathological and molecular characteristics associated with pathological complete response in neoadjuvant immunotherapy for breast cancer",
        domain="clinical evidence synthesis",
        target_words=6762,
        body_target_words=4600,
        expected_sections=6,
        practice_context="evidence-based clinical decision-making",
        organizer="evidence comparison framework",
        focus_points=[
                "Method",
                "Clinicopathological and molecular characteristics associated with pathological complete response in neoadjuvant immunotherapy",
                "breast cancer",
        ],
        extra_must_include=[
                "ICI",
                "EMBASE",
                "MEDLINE",
                "ICI-",
        ],
    )
    config = build_main_task_config(
        spec,
        session_name="metabench_pmc13066237",
        corpus_dir="./data_sample/med_papers_review_augmented_strict",
        outline_override=dict(CUSTOM_OUTLINE),
    )

    reference = config["reference"]
    constraints = reference["constraints"]
    for key in ('section_budget_trace', 'six_slot_prior_version'):
        constraints.pop(key, None)

    return config
