"""MetaBench task derived from review article PMC12440783."""

from __future__ import annotations

from typing import Dict

from meta_bench import TaskSpec, build_main_task_config


CUSTOM_OUTLINE = {
    "sec1": "Scope, disease context, and chemokine terminology in alopecia areata",
    "sec2": "Chemokine-pathway framework and hair-follicle immune privilege collapse",
    "sec3": "Evidence base and measurement strategies across blood and skin studies",
    "sec4": "Chemokine signatures in alopecia areata across Th1, Th2, and related pathways",
    "sec5": "Biomarker value and therapeutic implications for clinical dermatology",
    "sec6": "Limitations, heterogeneity, and future research priorities",
}


def get_task_config() -> Dict[str, object]:
    """Return the PMC12440783-derived MetaBench task configuration."""

    spec = TaskSpec(
        task_id="med_pmc12440783",
        topic="chemokines in alopecia areata",
        domain="immunodermatology",
        target_words=6083,
        body_target_words=2916,
        expected_sections=len(CUSTOM_OUTLINE),
        practice_context="clinical dermatology decision-making",
        organizer="chemokine pathway framework",
        focus_points=[
            "Th1-associated chemokines",
            "Th2-associated chemokines",
            "blood and skin biomarker patterns",
            "therapeutic implications",
        ],
        extra_must_include=[
            "meta-analysis",
            "immune privilege",
            "dupilumab",
        ],
    )
    config = build_main_task_config(
        spec,
        session_name="metabench_pmc12440783",
        corpus_dir="./data_sample/med_papers",
        outline_override=dict(CUSTOM_OUTLINE),
    )

    return config
