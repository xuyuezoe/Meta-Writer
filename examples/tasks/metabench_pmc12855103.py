"""MetaBench task derived from review article PMC12855103."""

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
    """Return the PMC12855103-derived MetaBench task configuration."""

    spec = TaskSpec(
        task_id="med_pmc12855103",
        topic="Association between statin use and the risk of colorectal cancer in patients with inflammatory bowel disease",
        domain="clinical evidence synthesis",
        target_words=4584,
        body_target_words=2297,
        expected_sections=6,
        practice_context="evidence-based clinical decision-making",
        organizer="systematic evidence synthesis framework",
        focus_points=[
        "colorectal cancer",
        "incidence",
        "inflammatory bowel disease",
        "meta-analysis"
],
        extra_must_include=[
        "IBD",
        "CRC",
        "RR",
        "CI"
],
    )
    config = build_main_task_config(
        spec,
        session_name="metabench_pmc12855103",
        corpus_dir="./data_sample/med_papers_review_augmented_strict",
        outline_override=dict(CUSTOM_OUTLINE),
    )

    reference = config["reference"]
    constraints = reference["constraints"]
    constraints["section_word_targets"] = {
        "sec1": 544,
        "sec2": 133,
        "sec3": 162,
        "sec4": 173,
        "sec5": 140,
        "sec6": 1145
}
    constraints["section_budget_trace"] = {
        "source": "real_article_six_slot_alignment",
        "source_article_path": "external:medical_reviews_300_ready/PMC12855103.md",
        "raw_article_section_count": 4,
        "raw_article_outline": {
                "sec1": "Introduction",
                "sec2": "Methods",
                "sec3": "Discussion",
                "sec4": "Conclusions"
        },
        "raw_article_section_word_targets": {
                "sec1": 490,
                "sec2": 80,
                "sec3": 1280,
                "sec4": 77
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
                "sec1": 490.0,
                "sec2": 88.6302,
                "sec3": 113.3246,
                "sec4": 122.583,
                "sec5": 82.4621,
                "sec6": 1030.0
        },
        "actual_section_word_targets": {
                "sec1": 456.8002,
                "sec2": 111.357,
                "sec3": 136.0745,
                "sec4": 145.3141,
                "sec5": 117.2415,
                "sec6": 960.2126
        },
        "slot_prior_floor_factor": 0.35,
        "scaled_to_body_target_words": 2297
}
    constraints["section_prior_scheme"] = "real_article_six_slot_aligned"
    constraints["six_slot_prior_version"] = '52bb45a3787b'

    return config
