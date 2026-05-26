"""MetaBench task derived from review article PMC12864541."""

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
    """Return the PMC12864541-derived MetaBench task configuration."""

    spec = TaskSpec(
        task_id="med_pmc12864541",
        topic="Hoffman’s Exercise for Breastfeeding Support Among Postnatal Mothers With Nipple Defects: A Scoping Review and Exploratory Meta‐Analysis",
        domain="clinical evidence synthesis",
        target_words=5377,
        body_target_words=3849,
        expected_sections=6,
        practice_context="evidence-based clinical decision-making",
        organizer="mechanism and evidence framework",
        focus_points=[
        "2. Current Evidence",
        "4. Results—Primary Outcome",
        "5. Results—Secondary Outcome",
        "6. Mechanism of Action"
],
        extra_must_include=[],
    )
    config = build_main_task_config(
        spec,
        session_name="metabench_pmc12864541",
        corpus_dir="./data_sample/med_papers_review_augmented_strict",
        outline_override=dict(CUSTOM_OUTLINE),
    )

    reference = config["reference"]
    constraints = reference["constraints"]
    constraints["section_word_targets"] = {
        "sec1": 295,
        "sec2": 353,
        "sec3": 445,
        "sec4": 438,
        "sec5": 344,
        "sec6": 1974
}
    constraints["section_budget_trace"] = {
        "source": "real_article_six_slot_alignment",
        "source_article_path": "external:medical_reviews_300_ready/PMC12864541.md",
        "raw_article_section_count": 10,
        "raw_article_outline": {
                "sec1": "1. Introduction",
                "sec2": "2. Current Evidence",
                "sec3": "3. Methods",
                "sec4": "4. Results—Primary Outcome",
                "sec5": "5. Results—Secondary Outcome",
                "sec6": "6. Mechanism of Action",
                "sec7": "7. Clinical Interpretation of the Findings",
                "sec8": "8. Recommendations for Future Studies",
                "sec9": "9. Recommended Sample Size for Future Studies",
                "sec10": "10. Conclusions"
        },
        "raw_article_section_word_targets": {
                "sec1": 264,
                "sec2": 293,
                "sec3": 906,
                "sec4": 398,
                "sec5": 235,
                "sec6": 380,
                "sec7": 542,
                "sec8": 89,
                "sec9": 255,
                "sec10": 78
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
                "sec1": 264.0,
                "sec2": 315.9843,
                "sec3": 397.592,
                "sec4": 391.0606,
                "sec5": 307.3631,
                "sec6": 1764.0
        },
        "actual_section_word_targets": {
                "sec1": 264.0,
                "sec2": 315.9843,
                "sec3": 397.592,
                "sec4": 391.0606,
                "sec5": 307.3631,
                "sec6": 1764.0
        },
        "slot_prior_floor_factor": 0.35,
        "scaled_to_body_target_words": 3849
}
    constraints["section_prior_scheme"] = "real_article_six_slot_aligned"
    constraints["six_slot_prior_version"] = '52bb45a3787b'

    return config
