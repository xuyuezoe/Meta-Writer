"""MetaBench task derived from review article PMC12855459."""

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
    """Return the PMC12855459-derived MetaBench task configuration."""

    spec = TaskSpec(
        task_id="med_pmc12855459",
        topic="The effects of virtual reality-based interventions on cognitive function, depressive symptoms, and daily functioning in older adults with mild cognitive impairment: a systematic review and meta-analysis of randomized controlled trials",
        domain="clinical evidence synthesis",
        target_words=4884,
        body_target_words=2269,
        expected_sections=6,
        practice_context="evidence-based clinical decision-making",
        organizer="systematic evidence synthesis framework",
        focus_points=[
        "virtual reality",
        "mild cognitive impairment",
        "exergaming",
        "cognition"
],
        extra_must_include=[
        "VR",
        "MCI",
        "GRADE",
        "SMD"
],
    )
    config = build_main_task_config(
        spec,
        session_name="metabench_pmc12855459",
        corpus_dir="./data_sample/med_papers_review_augmented_strict",
        outline_override=dict(CUSTOM_OUTLINE),
    )

    reference = config["reference"]
    constraints = reference["constraints"]
    constraints["section_word_targets"] = {
        "sec1": 547,
        "sec2": 277,
        "sec3": 334,
        "sec4": 355,
        "sec5": 284,
        "sec6": 472
}
    constraints["section_budget_trace"] = {
        "source": "real_article_six_slot_alignment",
        "source_article_path": "external:medical_reviews_300_ready/PMC12855459.md",
        "raw_article_section_count": 3,
        "raw_article_outline": {
                "sec1": "Introduction",
                "sec2": "Discussion",
                "sec3": "Conclusion"
        },
        "raw_article_section_word_targets": {
                "sec1": 465,
                "sec2": 1311,
                "sec3": 151
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
                "sec1": 465.0,
                "sec2": 235.0502,
                "sec3": 283.5994,
                "sec4": 301.4564,
                "sec5": 240.894,
                "sec6": 401.0
        },
        "actual_section_word_targets": {
                "sec1": 465.0,
                "sec2": 235.0502,
                "sec3": 283.5994,
                "sec4": 301.4564,
                "sec5": 240.894,
                "sec6": 401.0
        },
        "slot_prior_floor_factor": 0.35,
        "scaled_to_body_target_words": 2269
}
    constraints["section_prior_scheme"] = "real_article_six_slot_aligned"
    constraints["six_slot_prior_version"] = '52bb45a3787b'

    return config
