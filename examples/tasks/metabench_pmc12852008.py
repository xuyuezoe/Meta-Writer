"""MetaBench task derived from review article PMC12852008."""

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
    """Return the PMC12852008-derived MetaBench task configuration."""

    spec = TaskSpec(
        task_id="med_pmc12852008",
        topic="Inflammatory biomarker response to GLP-1 receptor agonists versus other glucose-lowering medications in patients with type 2 diabetes",
        domain="clinical evidence synthesis",
        target_words=5068,
        body_target_words=2256,
        expected_sections=6,
        practice_context="evidence-based clinical decision-making",
        organizer="systematic evidence synthesis framework",
        focus_points=[
        "Inflammatory biomarker response to GLP-1 receptor agonists",
        "other glucose-lowering medications",
        "type 2 diabetes"
],
        extra_must_include=[
        "GLP-1",
        "T2D",
        "C-",
        "CRP"
],
    )
    config = build_main_task_config(
        spec,
        session_name="metabench_pmc12852008",
        corpus_dir="./data_sample/med_papers_review_augmented_strict",
        outline_override=dict(CUSTOM_OUTLINE),
    )

    reference = config["reference"]
    constraints = reference["constraints"]
    constraints["section_word_targets"] = {
        "sec1": 835,
        "sec2": 305,
        "sec3": 353,
        "sec4": 374,
        "sec5": 300,
        "sec6": 89
}
    constraints["section_budget_trace"] = {
        "source": "real_article_six_slot_alignment",
        "source_article_path": "external:medical_reviews_300_ready/PMC12852008.md",
        "raw_article_section_count": 3,
        "raw_article_outline": {
                "sec1": "Introduction",
                "sec2": "Methodology",
                "sec3": "Discussion"
        },
        "raw_article_section_word_targets": {
                "sec1": 612,
                "sec2": 58,
                "sec3": 984
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
                "sec1": 612.0,
                "sec2": 223.7124,
                "sec3": 258.6984,
                "sec4": 274.5663,
                "sec5": 220.0229,
                "sec6": 65.0
        },
        "actual_section_word_targets": {
                "sec1": 612.0,
                "sec2": 223.7124,
                "sec3": 258.6984,
                "sec4": 274.5663,
                "sec5": 220.0229,
                "sec6": 65.0
        },
        "slot_prior_floor_factor": 0.35,
        "scaled_to_body_target_words": 2256
}
    constraints["section_prior_scheme"] = "real_article_six_slot_aligned"
    constraints["six_slot_prior_version"] = '52bb45a3787b'

    return config
