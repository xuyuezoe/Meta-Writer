"""MetaBench task derived from review article PMC12909180."""

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
    """Return the PMC12909180-derived MetaBench task configuration."""

    spec = TaskSpec(
        task_id="med_pmc12909180",
        topic="Anti-phospholipid antibodies as a risk factor for renal injury in patients with systemic lupus erythematosus: a comprehensive analysis",
        domain="clinical evidence synthesis",
        target_words=6054,
        body_target_words=2349,
        expected_sections=6,
        practice_context="evidence-based clinical decision-making",
        organizer="evidence comparison framework",
        focus_points=[
        "anti-phospholipid antibodies",
        "anti-phospholipid syndrome",
        "lupus anticoagulant",
        "renal injury"
],
        extra_must_include=[
        "SLE",
        "OR",
        "LA",
        "PT"
],
    )
    config = build_main_task_config(
        spec,
        session_name="metabench_pmc12909180",
        corpus_dir="./data_sample/med_papers_review_augmented_strict",
        outline_override=dict(CUSTOM_OUTLINE),
    )

    reference = config["reference"]
    constraints = reference["constraints"]
    constraints["section_word_targets"] = {
        "sec1": 628,
        "sec2": 193,
        "sec3": 234,
        "sec4": 280,
        "sec5": 227,
        "sec6": 787
}
    constraints["section_budget_trace"] = {
        "source": "real_article_six_slot_alignment",
        "source_article_path": "external:medical_reviews_300_ready/PMC12909180.md",
        "raw_article_section_count": 3,
        "raw_article_outline": {
                "sec1": "Introduction",
                "sec2": "Discussion",
                "sec3": "Conclusions"
        },
        "raw_article_section_word_targets": {
                "sec1": 509,
                "sec2": 1358,
                "sec3": 38
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
                "sec1": 509.0,
                "sec2": 156.8656,
                "sec3": 189.4849,
                "sec4": 227.0998,
                "sec5": 184.5498,
                "sec6": 638.0
        },
        "actual_section_word_targets": {
                "sec1": 509.0,
                "sec2": 156.8656,
                "sec3": 189.4849,
                "sec4": 227.0998,
                "sec5": 184.5498,
                "sec6": 638.0
        },
        "slot_prior_floor_factor": 0.35,
        "scaled_to_body_target_words": 2349
}
    constraints["section_prior_scheme"] = "real_article_six_slot_aligned"
    constraints["six_slot_prior_version"] = '52bb45a3787b'

    return config
