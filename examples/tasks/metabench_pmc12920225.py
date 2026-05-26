"""MetaBench task derived from review article PMC12920225."""

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
    """Return the PMC12920225-derived MetaBench task configuration."""

    spec = TaskSpec(
        task_id="med_pmc12920225",
        topic="Dupilumab treatment outcomes in bullous pemphigoid: a systematic review and single-arm meta-analysis",
        domain="clinical evidence synthesis",
        target_words=3542,
        body_target_words=1845,
        expected_sections=6,
        practice_context="evidence-based clinical decision-making",
        organizer="systematic evidence synthesis framework",
        focus_points=[
        "Dupilumab treatment outcomes in bullous pemphigoid",
        "a systematic review and single-arm meta-analysis"
],
        extra_must_include=[
        "BP",
        "CI",
        "PROSPERO",
        "CRD420251048550"
],
    )
    config = build_main_task_config(
        spec,
        session_name="metabench_pmc12920225",
        corpus_dir="./data_sample/med_papers_review_augmented_strict",
        outline_override=dict(CUSTOM_OUTLINE),
    )

    reference = config["reference"]
    constraints = reference["constraints"]
    constraints["section_word_targets"] = {
        "sec1": 485,
        "sec2": 165,
        "sec3": 203,
        "sec4": 218,
        "sec5": 182,
        "sec6": 592
}
    constraints["section_budget_trace"] = {
        "source": "real_article_six_slot_alignment",
        "source_article_path": "external:medical_reviews_300_ready/PMC12920225.md",
        "raw_article_section_count": 4,
        "raw_article_outline": {
                "sec1": "Introduction",
                "sec2": "Methods",
                "sec3": "Discussion",
                "sec4": "Conclusion"
        },
        "raw_article_section_word_targets": {
                "sec1": 395,
                "sec2": 27,
                "sec3": 1043,
                "sec4": 38
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
                "sec1": 395.0,
                "sec2": 134.4876,
                "sec3": 165.0421,
                "sec4": 177.8677,
                "sec5": 148.6025,
                "sec6": 482.0
        },
        "actual_section_word_targets": {
                "sec1": 395.0,
                "sec2": 134.4876,
                "sec3": 165.0421,
                "sec4": 177.8677,
                "sec5": 148.6025,
                "sec6": 482.0
        },
        "slot_prior_floor_factor": 0.35,
        "scaled_to_body_target_words": 1845
}
    constraints["section_prior_scheme"] = "real_article_six_slot_aligned"
    constraints["six_slot_prior_version"] = '52bb45a3787b'

    return config
