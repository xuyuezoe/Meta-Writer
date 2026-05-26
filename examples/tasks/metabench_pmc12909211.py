"""MetaBench task derived from review article PMC12909211."""

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
    """Return the PMC12909211-derived MetaBench task configuration."""

    spec = TaskSpec(
        task_id="med_pmc12909211",
        topic="The correlation between traditional Chinese medicine constitution and prediabetes",
        domain="clinical evidence synthesis",
        target_words=5312,
        body_target_words=1358,
        expected_sections=6,
        practice_context="evidence-based clinical decision-making",
        organizer="systematic evidence synthesis framework",
        focus_points=[
        "The correlation between traditional Chinese medicine constitution and prediabetes"
],
        extra_must_include=[
        "TCM",
        "PDC",
        "BC",
        "YIDC"
],
    )
    config = build_main_task_config(
        spec,
        session_name="metabench_pmc12909211",
        corpus_dir="./data_sample/med_papers_review_augmented_strict",
        outline_override=dict(CUSTOM_OUTLINE),
    )

    reference = config["reference"]
    constraints = reference["constraints"]
    constraints["section_word_targets"] = {
        "sec1": 862,
        "sec2": 74,
        "sec3": 90,
        "sec4": 96,
        "sec5": 78,
        "sec6": 158
}
    constraints["section_budget_trace"] = {
        "source": "real_article_six_slot_alignment",
        "source_article_path": "external:medical_reviews_300_ready/PMC12909211.md",
        "raw_article_section_count": 3,
        "raw_article_outline": {
                "sec1": "Introduction",
                "sec2": "Discussion",
                "sec3": "Conclusion"
        },
        "raw_article_section_word_targets": {
                "sec1": 639,
                "sec2": 127,
                "sec3": 117
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
                "sec1": 639.0,
                "sec2": 16.0561,
                "sec3": 43.8222,
                "sec4": 43.8222,
                "sec5": 23.2994,
                "sec6": 117.0
        },
        "actual_section_word_targets": {
                "sec1": 560.4966,
                "sec2": 48.0108,
                "sec3": 58.6675,
                "sec4": 62.6511,
                "sec5": 50.5478,
                "sec6": 102.6261
        },
        "slot_prior_floor_factor": 0.35,
        "scaled_to_body_target_words": 1358
}
    constraints["section_prior_scheme"] = "real_article_six_slot_aligned"
    constraints["six_slot_prior_version"] = '52bb45a3787b'

    return config
