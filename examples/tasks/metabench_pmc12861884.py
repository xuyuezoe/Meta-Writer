"""MetaBench task derived from review article PMC12861884."""

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
    """Return the PMC12861884-derived MetaBench task configuration."""

    spec = TaskSpec(
        task_id="med_pmc12861884",
        topic="Association between prediabetes and the risk of atrial fibrillation",
        domain="clinical evidence synthesis",
        target_words=3753,
        body_target_words=1737,
        expected_sections=6,
        practice_context="evidence-based clinical decision-making",
        organizer="systematic evidence synthesis framework",
        focus_points=[
        "atrial fibrillation",
        "meta-analysis",
        "prediabetes",
        "risk factors"
],
        extra_must_include=[
        "AF",
        "CRD420251233423"
],
    )
    config = build_main_task_config(
        spec,
        session_name="metabench_pmc12861884",
        corpus_dir="./data_sample/med_papers_review_augmented_strict",
        outline_override=dict(CUSTOM_OUTLINE),
    )

    reference = config["reference"]
    constraints = reference["constraints"]
    constraints["section_word_targets"] = {
        "sec1": 620,
        "sec2": 94,
        "sec3": 116,
        "sec4": 123,
        "sec5": 100,
        "sec6": 684
}
    constraints["section_budget_trace"] = {
        "source": "real_article_six_slot_alignment",
        "source_article_path": "external:medical_reviews_300_ready/PMC12861884.md",
        "raw_article_section_count": 4,
        "raw_article_outline": {
                "sec1": "Introduction",
                "sec2": "Materials and methods",
                "sec3": "Discussion",
                "sec4": "Conclusion"
        },
        "raw_article_section_word_targets": {
                "sec1": 573,
                "sec2": 45,
                "sec3": 675,
                "sec4": 118
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
                "sec1": 573.0,
                "sec2": 33.5515,
                "sec3": 55.0393,
                "sec4": 79.7019,
                "sec5": 37.7072,
                "sec6": 632.0
        },
        "actual_section_word_targets": {
                "sec1": 503.5625,
                "sec2": 76.8655,
                "sec3": 93.9271,
                "sec4": 100.3048,
                "sec5": 80.9274,
                "sec6": 555.4128
        },
        "slot_prior_floor_factor": 0.35,
        "scaled_to_body_target_words": 1737
}
    constraints["section_prior_scheme"] = "real_article_six_slot_aligned"
    constraints["six_slot_prior_version"] = '52bb45a3787b'

    return config
