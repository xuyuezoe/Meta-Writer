"""MetaBench task derived from review article PMC12864541."""

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
    """Return the PMC12864541-derived MetaBench task configuration."""

    spec = TaskSpec(
        task_id="med_pmc12864541",
        topic="Hoffman’s Exercise for Breastfeeding Support Among Postnatal Mothers With Nipple Defects: A Scoping Review and Exploratory Meta‐Analysis",
        domain="clinical evidence synthesis",
        target_words=9128,
        body_target_words=7600,
        expected_sections=6,
        practice_context="evidence-based clinical decision-making",
        organizer="mechanism and evidence framework",
        focus_points=[
        "2. Current Evidence",
        "4. Results—Primary Outcome",
        "5. Results—Secondary Outcome",
        "6. Mechanism of Action"
],
        extra_must_include=[],
    )
    config = build_main_task_config(
        spec,
        session_name="metabench_pmc12864541",
        corpus_dir="./data_sample/med_papers_review_augmented_strict",
        outline_override=dict(CUSTOM_OUTLINE),
    )

    reference = config["reference"]
    constraints = reference["constraints"]
    for key in ('section_budget_trace', 'six_slot_prior_version'):
        constraints.pop(key, None)

    return config
