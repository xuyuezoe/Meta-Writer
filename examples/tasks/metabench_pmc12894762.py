"""MetaBench task derived from review article PMC12894762."""

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
    """Return the PMC12894762-derived MetaBench task configuration."""

    spec = TaskSpec(
        task_id="med_pmc12894762",
        topic="Targeting the gut microbiome for type 2 diabetes management: a scoping review of systematic reviews and meta-analyses",
        domain="clinical evidence synthesis",
        target_words=6677,
        body_target_words=1998,
        expected_sections=6,
        practice_context="evidence-based clinical decision-making",
        organizer="systematic evidence synthesis framework",
        focus_points=[
        "Strengths and limitations",
        "Targeting the gut microbiome",
        "type 2 diabetes management",
        "a scoping review of systematic reviews and meta-analyses"
],
        extra_must_include=[
        "GM",
        "T2DM",
        "PRISMA",
        "AMSTAR2"
],
    )
    config = build_main_task_config(
        spec,
        session_name="metabench_pmc12894762",
        corpus_dir="./data_sample/med_papers_review_augmented_strict",
        outline_override=dict(CUSTOM_OUTLINE),
    )

    reference = config["reference"]
    constraints = reference["constraints"]
    constraints["section_word_targets"] = {
        "sec1": 1100,
        "sec2": 102,
        "sec3": 125,
        "sec4": 134,
        "sec5": 108,
        "sec6": 429
}
    constraints["section_budget_trace"] = {
        "source": "real_article_six_slot_alignment",
        "source_article_path": "external:medical_reviews_300_ready/PMC12894762.md",
        "raw_article_section_count": 5,
        "raw_article_outline": {
                "sec1": "Introduction",
                "sec2": "Materials and methods",
                "sec3": "Discussion",
                "sec4": "Strengths and limitations",
                "sec5": "Conclusion"
        },
        "raw_article_section_word_targets": {
                "sec1": 1068,
                "sec2": 118,
                "sec3": 177,
                "sec4": 126,
                "sec5": 114
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
                "sec1": 1068.0,
                "sec2": 23.6683,
                "sec3": 41.6294,
                "sec4": 32.1962,
                "sec5": 20.5061,
                "sec6": 417.0
        },
        "actual_section_word_targets": {
                "sec1": 882.4432,
                "sec2": 82.1022,
                "sec3": 100.3261,
                "sec4": 107.1383,
                "sec5": 86.4408,
                "sec6": 344.5494
        },
        "slot_prior_floor_factor": 0.35,
        "scaled_to_body_target_words": 1998
}
    constraints["section_prior_scheme"] = "real_article_six_slot_aligned"
    constraints["six_slot_prior_version"] = '52bb45a3787b'

    return config
