"""MetaBench task derived from review article PMC12891206."""

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
    """Return the PMC12891206-derived MetaBench task configuration."""

    spec = TaskSpec(
        task_id="med_pmc12891206",
        topic="Interventions to enhance in-home taking medication among older adults with multimorbidity/polypharmacy",
        domain="clinical evidence synthesis",
        target_words=5390,
        body_target_words=1635,
        expected_sections=6,
        practice_context="evidence-based clinical decision-making",
        organizer="systematic evidence synthesis framework",
        focus_points=[
        "aged",
        "drug therapy",
        "frail older adults",
        "home environment"
],
        extra_must_include=[
        "PROSPERO",
        "CRD42024513056",
        "PRISMA",
        "CINAHL"
],
    )
    config = build_main_task_config(
        spec,
        session_name="metabench_pmc12891206",
        corpus_dir="./data_sample/med_papers_review_augmented_strict",
        outline_override=dict(CUSTOM_OUTLINE),
    )

    reference = config["reference"]
    constraints = reference["constraints"]
    constraints["section_word_targets"] = {
        "sec1": 434,
        "sec2": 94,
        "sec3": 114,
        "sec4": 122,
        "sec5": 98,
        "sec6": 773
}
    constraints["section_budget_trace"] = {
        "source": "real_article_six_slot_alignment",
        "source_article_path": "external:medical_reviews_300_ready/PMC12891206.md",
        "raw_article_section_count": 4,
        "raw_article_outline": {
                "sec1": "Introduction",
                "sec2": "Materials and methods",
                "sec3": "Results",
                "sec4": "Discussion"
        },
        "raw_article_section_word_targets": {
                "sec1": 329,
                "sec2": 30,
                "sec3": 66,
                "sec4": 720
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
                "sec1": 329.0,
                "sec2": 49.6103,
                "sec3": 63.4034,
                "sec4": 67.7906,
                "sec5": 49.1958,
                "sec6": 586.0
        },
        "actual_section_word_targets": {
                "sec1": 303.7816,
                "sec2": 65.5355,
                "sec3": 80.0822,
                "sec4": 85.5198,
                "sec5": 68.9987,
                "sec6": 541.0821
        },
        "slot_prior_floor_factor": 0.35,
        "scaled_to_body_target_words": 1635
}
    constraints["section_prior_scheme"] = "real_article_six_slot_aligned"
    constraints["six_slot_prior_version"] = '52bb45a3787b'

    return config
