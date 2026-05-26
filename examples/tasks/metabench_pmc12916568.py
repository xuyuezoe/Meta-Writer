"""MetaBench task derived from review article PMC12916568."""

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
    """Return the PMC12916568-derived MetaBench task configuration."""

    spec = TaskSpec(
        task_id="med_pmc12916568",
        topic="Using serious games for cardiopulmonary resuscitation training: a meta-analysis and systematic review",
        domain="biomedical evidence synthesis",
        target_words=4020,
        body_target_words=2683,
        expected_sections=6,
        practice_context="biomedical research interpretation",
        organizer="systematic evidence synthesis framework",
        focus_points=[
        "cardiopulmonary resuscitation",
        "education",
        "meta-analysis",
        "serious games"
],
        extra_must_include=[
        "CPR",
        "SMD",
        "CI"
],
    )
    config = build_main_task_config(
        spec,
        session_name="metabench_pmc12916568",
        corpus_dir="./data_sample/med_papers_review_augmented_strict",
        outline_override=dict(CUSTOM_OUTLINE),
    )

    reference = config["reference"]
    constraints = reference["constraints"]
    constraints["section_word_targets"] = {
        "sec1": 549,
        "sec2": 162,
        "sec3": 198,
        "sec4": 212,
        "sec5": 171,
        "sec6": 1391
}
    constraints["section_budget_trace"] = {
        "source": "real_article_six_slot_alignment",
        "source_article_path": "external:medical_reviews_300_ready/PMC12916568.md",
        "raw_article_section_count": 4,
        "raw_article_outline": {
                "sec1": "Introduction",
                "sec2": "Methods",
                "sec3": "Discussion",
                "sec4": "Conclusion"
        },
        "raw_article_section_word_targets": {
                "sec1": 489,
                "sec2": 589,
                "sec3": 1017,
                "sec4": 235
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
                "sec1": 489.0,
                "sec2": 128.2349,
                "sec3": 172.7679,
                "sec4": 164.6721,
                "sec5": 137.3252,
                "sec6": 1238.0
        },
        "actual_section_word_targets": {
                "sec1": 477.0313,
                "sec2": 140.8963,
                "sec3": 172.1706,
                "sec4": 183.861,
                "sec5": 148.3418,
                "sec6": 1207.6989
        },
        "slot_prior_floor_factor": 0.35,
        "scaled_to_body_target_words": 2683
}
    constraints["section_prior_scheme"] = "real_article_six_slot_aligned"
    constraints["six_slot_prior_version"] = '52bb45a3787b'

    return config
