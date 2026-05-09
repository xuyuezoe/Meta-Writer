"""Citation quality scoring for benchmark outputs.

This module is intentionally model-free. It can consume structured citation
metadata when a benchmark sample provides it, and otherwise falls back to the
generated paper's inline citation markers and References section. The benchmark
assumes citation validity has already been checked elsewhere; this scorer only
evaluates full-document citation use quality.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union, cast


JsonDict = Dict[str, object]
WEAK_MATCH_MISMATCH_WEIGHT = 0.35
ROLE_MISMATCH_WHEN_ALIGNMENT_EXISTS_WEIGHT = 0.25
SOFT_UNSUPPORTED_SECTION_TYPES = frozenset({"introduction", "limitations_gaps", "conclusion"})
SOFT_UNSUPPORTED_SECTION_WEIGHT = 0.6


REVIEW_DISTRIBUTION_BOUNDS: Dict[str, Tuple[float, float]] = {
    "introduction": (0.15, 0.30),
    "taxonomy": (0.20, 0.35),
    "evidence_synthesis": (0.35, 0.60),
    "limitations_gaps": (0.10, 0.25),
    "conclusion": (0.00, 0.10),
}

RESEARCH_DISTRIBUTION_BOUNDS: Dict[str, Tuple[float, float]] = {
    "introduction": (0.25, 0.40),
    "related_work": (0.15, 0.30),
    "methods": (0.05, 0.20),
    "results": (0.00, 0.10),
    "discussion": (0.25, 0.45),
    "limitations": (0.05, 0.15),
    "conclusion": (0.00, 0.05),
}

RESEARCH_REQUIRED_ROLES = {
    "research_gap",
    "gap",
    "external_method",
    "external_tool",
    "method",
    "dataset",
    "measurement",
    "measurement_scale",
    "reporting_standard",
    "prior_result_comparison",
    "guideline_or_protocol",
}


@dataclass
class CitationEvent:
    citation_id: str
    source_id: str
    section_id: str
    source_type: str = "unknown"
    claim_role: str = "unknown"
    claim_span: str = ""
    source_excerpt: str = ""
    claim_strength: Optional[int] = None
    evidence_strength: Optional[int] = None
    role_mismatch: Optional[bool] = None
    fidelity_penalty: Optional[float] = None
    unsupported_high_claim: bool = False
    citation_pileup: bool = False
    paragraph_tail_only: bool = False


@dataclass
class SectionStats:
    section_id: str
    section_type: str
    canonical_type: str
    word_count: int
    citations: List[CitationEvent] = field(default_factory=list)
    source_counts: Counter[str] = field(default_factory=Counter)
    penalties: Dict[str, float] = field(default_factory=dict)

    @property
    def citation_count(self) -> int:
        return len(self.citations)


def evaluate_citation_quality(
    final_text: str,
    citation_quality_config: object,
) -> Dict[str, object]:
    """Evaluate citation quality from benchmark metadata or generated text.

    Args:
        final_text: Generated full text.
        citation_quality_config: Optional dict with paper_type, section_map,
            citation_manifest, and judge_flags. When missing, the scorer parses
            the generated text's section headings, inline citation markers, and
            References list.

    Returns:
        A serializable dictionary with citation-quality score and diagnostics.
    """
    if citation_quality_config is None:
        config: JsonDict = _infer_citation_quality_config_from_text(final_text)
    elif not isinstance(citation_quality_config, dict):
        raise TypeError("citation_quality 必须是字典")
    else:
        config = cast(JsonDict, citation_quality_config)

    config_source = str(config.get("config_source", "structured_manifest"))
    parse_warnings = _as_list(config.get("parse_warnings"))
    if parse_warnings:
        config.setdefault("metadata", {})
        metadata = config["metadata"]
        if isinstance(metadata, dict):
            metadata["parse_warnings"] = parse_warnings

    manifest_object = config.get("citation_manifest")
    if manifest_object is None:
        inferred_config = _infer_citation_quality_config_from_text(final_text)
        inferred_config["paper_type"] = config.get(
            "paper_type", inferred_config.get("paper_type", "review")
        )
        if isinstance(config.get("section_map"), list):
            inferred_config["section_map"] = config["section_map"]
        if isinstance(config.get("judge_flags"), dict):
            inferred_config["judge_flags"] = config["judge_flags"]
        config = inferred_config
        config_source = str(config.get("config_source", "text_inferred"))
        manifest_object = config.get("citation_manifest")

    if not isinstance(manifest_object, list):
        raise TypeError("citation_quality.citation_manifest 必须是列表")


    citations = _normalize_citations(manifest_object)
    paper_type = _normalize_paper_type(config.get("paper_type"))
    judge_flags = _as_dict(config.get("judge_flags"))
    sections = _normalize_sections(config.get("section_map"), final_text, citations)

    _attach_citations_to_sections(sections, citations)

    if paper_type == "research":
        return _evaluate_research(sections, citations, config, judge_flags)
    return _evaluate_review(sections, citations, judge_flags, config)


def _evaluate_review(
    sections: List[SectionStats],
    citations: List[CitationEvent],
    judge_flags: Mapping[str, object],
    config: Mapping[str, object],
) -> Dict[str, object]:
    total_citations = len(citations)
    total_words = sum(max(0, section.word_count) for section in sections)
    section_weights = _word_weights(sections)

    p0_coverage = _review_coverage_penalty(sections, section_weights)
    p0_overcitation = _review_overcitation_penalty(sections, section_weights)
    p_source_diversity = _source_diversity_penalty(citations, "review")
    p0_source_dominance = _source_dominance_penalty(sections, citations, "review")
    p_source_balance = _source_balance_penalty(
        p_source_diversity,
        p0_source_dominance,
    )
    (
        p_section_distribution,
        distribution_section_penalties,
        distribution_diagnostics,
    ) = _section_distribution_penalty(
        sections,
        total_citations,
        REVIEW_DISTRIBUTION_BOUNDS,
        "review",
    )
    for section in sections:
        section.penalties["section_distribution"] = distribution_section_penalties.get(
            section.section_id, 0.0
        )
    p0_claim_source_match = _claim_source_match_penalty(sections, "review", judge_flags)
    p0_source_fidelity = _source_fidelity_penalty(sections)
    p0_citation_granularity = _citation_granularity_penalty(
        sections, judge_flags
    )

    for section in sections:
        section_score = _clamp01(
            1.0
            - section.penalties.get("coverage", 0.0) * 0.30
            - section.penalties.get("overcitation", 0.0) * 0.10
            - section.penalties.get("claim_source_match", 0.0) * 0.15
            - section.penalties.get("source_fidelity", 0.0) * 0.20
            - section.penalties.get("citation_granularity", 0.0) * 0.15
        )
        section.penalties["section_score"] = section_score

    penalties = {
        "coverage": p0_coverage,
        "overcitation": p0_overcitation,
        "source_balance": p_source_balance,
        "section_distribution": p_section_distribution,
        "claim_source_match": p0_claim_source_match,
        "source_fidelity": p0_source_fidelity,
        "citation_granularity": p0_citation_granularity,
    }
    metric_weights = _normalize_metric_weights(
        {
            "coverage": 0.15,
            "overcitation": 0.05,
            "source_balance": 0.15,
            "section_distribution": 0.15,
            "claim_source_match": 0.10,
            "source_fidelity": 0.10,
            "citation_granularity": 0.10,
        }
    )
    score = _score_from_penalties(penalties, metric_weights)

    return _build_result(
        paper_type="review",
        score=score,
        penalties=penalties,
        sections=sections,
        citations=citations,
        total_words=total_words,
        metric_weights=metric_weights,
        extra_diagnostics={
            "citation_quality_input": _citation_quality_input_diagnostics(config),
            "section_distribution_details": distribution_diagnostics,
            "source_balance_details": {
                "source_diversity_penalty": round(p_source_diversity, 4),
                "source_dominance_penalty": round(p0_source_dominance, 4),
                "source_balance_rule": "max(source_diversity_penalty, source_dominance_penalty)",
            },
            "citation_granularity_details": _citation_granularity_component_summary(
                sections
            ),
        },
    )


def _evaluate_research(
    sections: List[SectionStats],
    citations: List[CitationEvent],
    config: Mapping[str, object],
    judge_flags: Mapping[str, object],
) -> Dict[str, object]:
    total_citations = len(citations)
    total_words = sum(max(0, section.word_count) for section in sections)

    p_required_coverage, section_required = _required_coverage_penalty(
        config, judge_flags
    )
    for section in sections:
        section.penalties["required_coverage"] = section_required.get(
            section.section_id, 0.0
        )

    (
        p_section_distribution,
        distribution_section_penalties,
        distribution_diagnostics,
    ) = _section_distribution_penalty(
        sections,
        total_citations,
        RESEARCH_DISTRIBUTION_BOUNDS,
        "research",
    )
    for section in sections:
        section.penalties["section_distribution"] = distribution_section_penalties.get(
            section.section_id, 0.0
        )

    p_source_diversity = _source_diversity_penalty(citations, "research")
    p0_source_dominance = _source_dominance_penalty(sections, citations, "research")
    p_source_balance = _source_balance_penalty(
        p_source_diversity,
        p0_source_dominance,
    )
    p0_claim_source_match = _claim_source_match_penalty(sections, "research", judge_flags)
    p0_source_fidelity = _source_fidelity_penalty(sections)
    p0_citation_granularity = _citation_granularity_penalty(
        sections, judge_flags
    )
    p_conclusion_novelty, conclusion_sources = _conclusion_novelty_penalty(
        sections, judge_flags
    )
    for section in sections:
        if section.canonical_type == "conclusion":
            section.penalties["conclusion_novelty"] = p_conclusion_novelty

    for section in sections:
        section_score = _clamp01(
            1.0
            - section.penalties.get("required_coverage", 0.0) * 0.25
            - section.penalties.get("section_distribution", 0.0) * 0.10
            - section.penalties.get("claim_source_match", 0.0) * 0.15
            - section.penalties.get("source_fidelity", 0.0) * 0.20
            - section.penalties.get("citation_granularity", 0.0) * 0.10
            - section.penalties.get("conclusion_novelty", 0.0) * 0.15
        )
        section.penalties["section_score"] = section_score

    penalties = {
        "required_coverage": p_required_coverage,
        "section_distribution": p_section_distribution,
        "source_balance": p_source_balance,
        "claim_source_match": p0_claim_source_match,
        "source_fidelity": p0_source_fidelity,
        "citation_granularity": p0_citation_granularity,
        "conclusion_novelty": p_conclusion_novelty,
    }
    metric_weights = _normalize_metric_weights(
        {
            "required_coverage": 0.20,
            "section_distribution": 0.20,
            "source_balance": 0.15,
            "claim_source_match": 0.15,
            "source_fidelity": 0.15,
            "citation_granularity": 0.05,
            "conclusion_novelty": 0.05,
        }
    )
    score = _score_from_penalties(penalties, metric_weights)

    result = _build_result(
        paper_type="research",
        score=score,
        penalties=penalties,
        sections=sections,
        citations=citations,
        total_words=total_words,
        metric_weights=metric_weights,
        extra_diagnostics={
            "citation_quality_input": _citation_quality_input_diagnostics(config),
            "section_distribution_details": distribution_diagnostics,
            "source_balance_details": {
                "source_diversity_penalty": round(p_source_diversity, 4),
                "source_dominance_penalty": round(p0_source_dominance, 4),
                "source_balance_rule": "max(source_diversity_penalty, source_dominance_penalty)",
            },
            "citation_granularity_details": _citation_granularity_component_summary(
                sections
            ),
            "new_conclusion_sources": conclusion_sources,
        },
    )
    return result


def _review_coverage_penalty(
    sections: List[SectionStats], section_weights: Mapping[str, float]
) -> float:
    penalties: List[Tuple[float, float]] = []
    for section in sections:
        expected_min = max(1, math.ceil(max(0, section.word_count) / 350))
        coverage_ratio = min(1.0, section.citation_count / expected_min)
        penalty = 1.0 - coverage_ratio
        section.penalties["coverage"] = penalty
        section.penalties["expected_min_citations"] = float(expected_min)
        penalties.append((penalty, section_weights.get(section.section_id, 0.0)))
    return _weighted_average(penalties)


def _review_overcitation_penalty(
    sections: List[SectionStats], section_weights: Mapping[str, float]
) -> float:
    penalties: List[Tuple[float, float]] = []
    for section in sections:
        density = section.citation_count / max(1.0, section.word_count / 1000.0)
        if density <= 12.0:
            penalty = 0.0
        elif density <= 18.0:
            penalty = 0.2
        else:
            penalty = 0.6
        section.penalties["overcitation"] = penalty
        section.penalties["citation_density"] = density
        penalties.append((penalty, section_weights.get(section.section_id, 0.0)))
    return _weighted_average(penalties)


def _source_diversity_penalty(citations: List[CitationEvent], paper_type: str) -> float:
    total = len(citations)
    if total == 0:
        return 1.0
    ratio = len({event.source_id for event in citations}) / total
    if paper_type == "research":
        if ratio >= 0.70:
            return 0.0
        if ratio >= 0.50:
            return 0.2
        if ratio >= 0.30:
            return 0.6
        return 1.0

    if ratio >= 0.65:
        return 0.0
    if ratio >= 0.45:
        return 0.2
    if ratio >= 0.20:
        return 0.7
    return 1.0


def _source_balance_penalty(
    source_diversity_penalty: float,
    source_dominance_penalty: float,
) -> float:
    return max(source_diversity_penalty, source_dominance_penalty)


def _source_dominance_penalty(
    sections: List[SectionStats], citations: List[CitationEvent], paper_type: str
) -> float:
    total = len(citations)
    if total == 0:
        for section in sections:
            section.penalties["source_dominance"] = 0.0
        return 0.0

    source_counts = Counter(event.source_id for event in citations)
    max_source_count = max(source_counts.values(), default=0)
    global_dominance = max_source_count / total

    if paper_type == "research":
        if max_source_count >= 6 or (max_source_count >= 4 and global_dominance > 0.30):
            global_penalty = 1.0
        elif max_source_count >= 4 and global_dominance > 0.20:
            global_penalty = 0.6
        else:
            global_penalty = 0.0
    else:
        if max_source_count >= 8 or (max_source_count >= 4 and global_dominance > 0.18):
            global_penalty = 1.0
        elif max_source_count >= 5 and global_dominance > 0.12:
            global_penalty = 0.6
        else:
            global_penalty = 0.0

    section_weights = _word_weights(sections)
    section_penalties: List[Tuple[float, float]] = []
    for section in sections:
        if section.citation_count == 0:
            penalty = 0.0
        else:
            section_max = max(section.source_counts.values(), default=0)
            section_dominance = section_max / section.citation_count
            if section.citation_count >= 4 and section_dominance > 0.65:
                penalty = 1.0
            elif section.citation_count >= 4 and section_dominance > 0.50:
                penalty = 0.6
            else:
                penalty = 0.0
        section.penalties["source_dominance"] = penalty
        section_penalties.append((penalty, section_weights.get(section.section_id, 0.0)))

    return max(global_penalty, _weighted_average(section_penalties))


def _section_distribution_penalty(
    sections: List[SectionStats],
    total_citations: int,
    bounds: Mapping[str, Tuple[float, float]],
    paper_type: str,
) -> Tuple[float, Dict[str, float], Dict[str, object]]:
    role_citation_counts: Dict[str, int] = defaultdict(int)
    section_penalties: Dict[str, float] = {}
    diagnostics: Dict[str, object] = {}

    for section in sections:
        role = _distribution_role(section.canonical_type, paper_type)
        if role in bounds:
            role_citation_counts[role] += section.citation_count

    role_penalties: Dict[str, float] = {}
    for role, (lower_bound, upper_bound) in bounds.items():
        if not any(
            _distribution_role(section.canonical_type, paper_type) == role
            for section in sections
        ):
            continue
        share = (
            role_citation_counts.get(role, 0) / total_citations
            if total_citations > 0
            else 0.0
        )
        role_penalty = max(0.0, lower_bound - share) + max(0.0, share - upper_bound)
        role_penalties[role] = role_penalty
        for section in sections:
            if _distribution_role(section.canonical_type, paper_type) == role:
                section_penalties[section.section_id] = role_penalty

    base_penalty = min(1.0, sum(role_penalties.values()))

    if paper_type != "research":
        diagnostics["role_penalties"] = _round_dict(role_penalties)
        return base_penalty, section_penalties, diagnostics

    role_shares = {
        role: (
            role_citation_counts.get(role, 0) / total_citations
            if total_citations > 0
            else 0.0
        )
        for role in role_citation_counts
    }
    results_share = role_shares.get("results", 0.0)
    conclusion_share = role_shares.get("conclusion", 0.0)
    discussion_words = sum(
        section.word_count for section in sections if section.canonical_type == "discussion"
    )
    discussion_citations = role_citation_counts.get("discussion", 0)

    if results_share > 0.20:
        p_results_overcitation = 1.0
    elif results_share > 0.10:
        p_results_overcitation = 0.5
    else:
        p_results_overcitation = 0.0

    p_discussion_missing = (
        1.0 if discussion_words > 150 and discussion_citations == 0 else 0.0
    )
    p_conclusion_new_evidence = 0.6 if conclusion_share > 0.10 else 0.0

    special_penalty = max(
        p_results_overcitation,
        p_discussion_missing,
        p_conclusion_new_evidence,
    )
    final_penalty = max(base_penalty, special_penalty)
    diagnostics.update(
        {
            "role_penalties": _round_dict(role_penalties),
            "results_overcitation": p_results_overcitation,
            "discussion_missing": p_discussion_missing,
            "conclusion_new_evidence_share": p_conclusion_new_evidence,
            "role_shares": _round_dict(role_shares),
        }
    )
    return final_penalty, section_penalties, diagnostics


def _claim_source_match_penalty(
    sections: List[SectionStats], paper_type: str, judge_flags: Mapping[str, object]
) -> float:
    section_weights = _word_weights(sections)
    flag_sections = _extract_section_flags(judge_flags)
    penalties: List[Tuple[float, float]] = []
    for section in sections:
        wrong_source_count = _as_count(
            flag_sections.get(section.section_id, {}).get("wrong_source_alignment_events")
        )
        weak_match_count = _as_count(
            flag_sections.get(section.section_id, {}).get("weak_match_alignment_events")
        )
        if section.citation_count == 0:
            weighted_mismatch = wrong_source_count + WEAK_MATCH_MISMATCH_WEIGHT * weak_match_count
            penalty = 1.0 if weighted_mismatch > 0 else 0.0
        else:
            alignment_signal = wrong_source_count + weak_match_count
            mismatch_count = sum(
                1 for event in section.citations if _is_mismatched(event, paper_type)
            )
            mismatch_weight = (
                ROLE_MISMATCH_WHEN_ALIGNMENT_EXISTS_WEIGHT if alignment_signal > 0 else 1.0
            )
            weighted_mismatch = (
                mismatch_count * mismatch_weight
                + wrong_source_count
                + WEAK_MATCH_MISMATCH_WEIGHT * weak_match_count
            )
            penalty = weighted_mismatch / section.citation_count
            penalty = min(1.0, penalty)
        section.penalties["claim_source_match"] = penalty
        section.penalties["wrong_source_alignment_events"] = float(wrong_source_count)
        section.penalties["weak_match_alignment_events"] = float(weak_match_count)
        penalties.append((penalty, section_weights.get(section.section_id, 0.0)))
    return _weighted_average(penalties)


def _source_fidelity_penalty(sections: List[SectionStats]) -> float:
    section_weights = _word_weights(sections)
    penalties: List[Tuple[float, float]] = []
    for section in sections:
        if section.citation_count == 0:
            penalty = 0.0
        else:
            penalty = sum(_event_fidelity_penalty(event) for event in section.citations)
            penalty /= section.citation_count
        section.penalties["source_fidelity"] = penalty
        penalties.append((penalty, section_weights.get(section.section_id, 0.0)))
    return _weighted_average(penalties)


def _citation_granularity_penalty(
    sections: List[SectionStats], judge_flags: Mapping[str, object]
) -> float:
    section_weights = _word_weights(sections)
    flag_sections = _extract_section_flags(judge_flags)
    penalties: List[Tuple[float, float]] = []

    for section in sections:
        flags = flag_sections.get(section.section_id, {})
        judge_declared_unsupported = _as_count(
            flags.get("judge_declared_unsupported_high_claims")
        )
        if (
            "judge_declared_unsupported_high_claims" not in flags
            and "retrieval_unsupported_alignment_events" not in flags
        ):
            judge_declared_unsupported = _as_count(flags.get("unsupported_high_claims"))
        retrieval_unsupported = _as_count(
            flags.get("retrieval_unsupported_alignment_events")
        )
        event_level_unsupported = sum(
            1 for event in section.citations if event.unsupported_high_claim
        )
        section_type_weight = (
            SOFT_UNSUPPORTED_SECTION_WEIGHT
            if section.canonical_type in SOFT_UNSUPPORTED_SECTION_TYPES
            else 1.0
        )
        weighted_judge_declared_unsupported = judge_declared_unsupported * section_type_weight
        unsupported = max(
            judge_declared_unsupported + retrieval_unsupported,
            event_level_unsupported,
        )
        pileups = max(
            _as_count(flags.get("citation_pileups")),
            sum(1 for event in section.citations if event.citation_pileup),
        )
        tail_only = max(
            _as_count(flags.get("paragraph_tail_only_events")),
            sum(1 for event in section.citations if event.paragraph_tail_only),
        )

        denominator = max(1, section.citation_count)
        judge_component = 0.5 * weighted_judge_declared_unsupported / denominator
        retrieval_component = 0.5 * retrieval_unsupported / denominator
        pileup_component = 0.3 * pileups / denominator
        tail_only_component = 0.2 * tail_only / denominator
        uncapped_penalty = (
            judge_component
            + retrieval_component
            + pileup_component
            + tail_only_component
        )
        penalty = min(1.0, uncapped_penalty)
        section.penalties["citation_granularity"] = penalty
        section.penalties["citation_granularity_uncapped"] = uncapped_penalty
        section.penalties["granularity_judge_unsupported_component"] = judge_component
        section.penalties["granularity_retrieval_unsupported_component"] = (
            retrieval_component
        )
        section.penalties["granularity_pileup_component"] = pileup_component
        section.penalties["granularity_tail_only_component"] = tail_only_component
        section.penalties["unsupported_high_claims"] = float(unsupported)
        section.penalties["judge_declared_unsupported_high_claims"] = float(
            judge_declared_unsupported
        )
        section.penalties["weighted_judge_declared_unsupported_high_claims"] = float(
            weighted_judge_declared_unsupported
        )
        section.penalties["retrieval_unsupported_alignment_events"] = float(
            retrieval_unsupported
        )
        section.penalties["wrong_source_alignment_events"] = float(
            _as_count(flags.get("wrong_source_alignment_events"))
        )
        section.penalties["weak_match_alignment_events"] = float(
            _as_count(flags.get("weak_match_alignment_events"))
        )
        section.penalties["citation_pileups"] = float(pileups)
        section.penalties["paragraph_tail_only_events"] = float(tail_only)
        penalties.append((penalty, section_weights.get(section.section_id, 0.0)))

    return _weighted_average(penalties)


def _citation_granularity_component_summary(
    sections: List[SectionStats],
) -> Dict[str, object]:
    section_weights = _word_weights(sections)
    component_keys = {
        "judge_unsupported": "granularity_judge_unsupported_component",
        "retrieval_unsupported": "granularity_retrieval_unsupported_component",
        "pileup": "granularity_pileup_component",
        "tail_only": "granularity_tail_only_component",
        "uncapped_sum": "citation_granularity_uncapped",
        "capped_penalty": "citation_granularity",
    }
    weighted_components: Dict[str, float] = {}
    for label, key in component_keys.items():
        weighted_components[label] = _weighted_mean_unbounded(
            (
                (
                    float(section.penalties.get(key, 0.0)),
                    section_weights.get(section.section_id, 0.0),
                )
                for section in sections
            )
        )

    return {
        "component_rule": (
            "citation_granularity = min(1, "
            "0.5*judge_unsupported + 0.5*retrieval_unsupported "
            "+ 0.3*pileup + 0.2*tail_only, normalized by section citation count)"
        ),
        "weighted_components": _round_dict(weighted_components),
        "boundary_note": (
            "wrong_source and weak_match are scored under claim_source_match; "
            "citation_granularity only scores unsupported strong claims and "
            "citation placement style issues."
        ),
        "cap_note": (
            "uncapped_sum can exceed capped_penalty because each section-level "
            "citation_granularity penalty is capped at 1.0."
        ),
    }


def _weighted_mean_unbounded(values: Iterable[Tuple[float, float]]) -> float:
    total = 0.0
    weight_sum = 0.0
    for value, weight in values:
        total += float(value) * max(0.0, weight)
        weight_sum += max(0.0, weight)
    if weight_sum <= 0:
        return 0.0
    return total / weight_sum


def _normalize_metric_weights(
    metric_weights: Mapping[str, float],
) -> Dict[str, float]:
    total = sum(max(0.0, float(weight)) for weight in metric_weights.values())
    if total <= 0.0:
        return {metric: 0.0 for metric in metric_weights}
    return {
        metric: max(0.0, float(weight)) / total
        for metric, weight in metric_weights.items()
    }


def _score_from_penalties(
    penalties: Mapping[str, float],
    metric_weights: Mapping[str, float],
) -> float:
    deduction = 0.0
    for metric, weight in metric_weights.items():
        deduction += _clamp01(float(penalties.get(metric, 0.0))) * max(0.0, weight)
    return _clamp01(1.0 - deduction)


def _required_coverage_penalty(
    config: Mapping[str, object],
    judge_flags: Mapping[str, object],
) -> Tuple[float, Dict[str, float]]:
    required_object = config.get("required_claims", judge_flags.get("required_claims"))
    if required_object is None:
        return 0.0, {}

    if isinstance(required_object, dict):
        total = int(required_object.get("total", 0))
        cited = int(required_object.get("cited", 0))
        penalty = 0.0 if total <= 0 else 1.0 - min(1.0, cited / total)
        return penalty, {}

    if not isinstance(required_object, list):
        raise TypeError("required_claims 必须是列表或字典")

    total = 0
    cited = 0
    by_section_total: Counter[str] = Counter()
    by_section_cited: Counter[str] = Counter()
    for item in required_object:
        if not isinstance(item, dict):
            continue
        role = _normalize_token(item.get("claim_role", item.get("type", "")))
        if role and role not in RESEARCH_REQUIRED_ROLES:
            continue
        total += 1
        section_id = str(item.get("section_id", ""))
        if section_id:
            by_section_total[section_id] += 1
        is_cited = _truthy(item.get("cited")) or bool(item.get("citation_id"))
        if is_cited:
            cited += 1
            if section_id:
                by_section_cited[section_id] += 1

    if total == 0:
        return 0.0, {}

    section_penalties = {
        section_id: 1.0 - min(1.0, by_section_cited[section_id] / count)
        for section_id, count in by_section_total.items()
        if count > 0
    }
    return 1.0 - min(1.0, cited / total), section_penalties


def _conclusion_novelty_penalty(
    sections: List[SectionStats], judge_flags: Mapping[str, object]
) -> Tuple[float, List[str]]:
    explicit_sources = judge_flags.get("new_conclusion_sources")
    if isinstance(explicit_sources, list):
        sources = [str(source) for source in explicit_sources]
        return _conclusion_novelty_from_count(len(sources)), sources

    seen_sources: set[str] = set()
    new_sources: set[str] = set()
    in_conclusion = False
    for section in sections:
        if section.canonical_type == "conclusion":
            in_conclusion = True
            for event in section.citations:
                if event.source_id not in seen_sources:
                    new_sources.add(event.source_id)
        elif not in_conclusion:
            seen_sources.update(event.source_id for event in section.citations)

    sources = sorted(new_sources)
    return _conclusion_novelty_from_count(len(sources)), sources


def _conclusion_novelty_from_count(count: int) -> float:
    if count <= 0:
        return 0.0
    if count == 1:
        return 0.4
    return 1.0


def _infer_citation_quality_config_from_text(final_text: str) -> JsonDict:
    sections = _split_text_sections(final_text)
    references = _parse_reference_list(final_text)
    reference_ids = {ref_id for ref_id, _ref_text in references}
    citation_manifest: List[JsonDict] = []
    judge_sections: Dict[str, Dict[str, int]] = {}
    parse_warnings: List[str] = []

    for section in sections:
        markers = _extract_inline_citation_numbers(section["body"])
        pileups = _count_citation_pileups(section["body"])
        tail_only = _count_paragraph_tail_only_events(section["body"])
        invalid_markers = [
            marker for marker in markers if reference_ids and marker not in reference_ids
        ]
        if invalid_markers:
            parse_warnings.append(
                f"{section['section_id']} has inline citations not found in References: "
                f"{sorted(set(invalid_markers))}"
            )
        judge_sections[str(section["section_id"])] = {
            "citation_pileups": pileups,
            "paragraph_tail_only_events": tail_only,
            "invalid_reference_markers": len(invalid_markers),
        }

        for marker in markers:
            if reference_ids and marker not in reference_ids:
                continue
            citation_manifest.append(
                {
                    "citation_id": f"T{len(citation_manifest) + 1:03d}",
                    "source_id": f"ref_{marker}",
                    "section_id": section["section_id"],
                    "source_type": "unknown",
                    "claim_role": _default_claim_role_for_section(
                        str(section["section_type"])
                    ),
                    "claim_span": _sentence_around_marker(section["body"], marker),
                    "source_excerpt": _reference_excerpt(marker, references),
                    "citation_pileup": False,
                    "paragraph_tail_only": False,
                }
            )

    if not sections:
        sections = [
            {
                "section_id": "sec1",
                "section_type": "other",
                "word_count": _count_words(_strip_references_section(final_text)),
                "body": _strip_references_section(final_text),
            }
        ]

    return {
        "paper_type": "review",
        "config_source": "text_inferred",
        "section_map": [
            {
                "section_id": section["section_id"],
                "type": section["section_type"],
                "word_count": section["word_count"],
            }
            for section in sections
        ],
        "citation_manifest": citation_manifest,
        "judge_flags": {"sections": judge_sections},
        "metadata": {
            "config_source": "text_inferred",
            "reference_count": len(reference_ids),
            "inline_citation_marker_count": len(citation_manifest),
            "parse_warnings": parse_warnings,
            "parse_note": (
                "Citation quality was inferred from generated inline citation "
                "markers and the References section; claim/source semantic "
                "fidelity fields are only scored when structured metadata exists."
            ),
        },
    }


def _split_text_sections(final_text: str) -> List[JsonDict]:
    body_text = _strip_references_section(final_text).strip()
    heading_pattern = re.compile(r"^#{1,3}\s+(.+?)\s*$", re.MULTILINE)
    matches = [
        match
        for match in heading_pattern.finditer(body_text)
        if _normalize_token(match.group(1)) not in {"references", "bibliography"}
    ]

    sections: List[JsonDict] = []
    if matches:
        for index, match in enumerate(matches, start=1):
            start = match.end()
            end = matches[index].start() if index < len(matches) else len(body_text)
            title = match.group(1).strip()
            body = body_text[start:end].strip()
            body = re.sub(r"^---\s*$", "", body, flags=re.MULTILINE).strip()
            sections.append(
                {
                    "section_id": f"sec{index}",
                    "section_type": title,
                    "word_count": _count_words(body),
                    "body": body,
                }
            )
        return sections

    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", body_text) if item.strip()]
    if not paragraphs and body_text:
        paragraphs = [body_text]
    for index, paragraph in enumerate(paragraphs, start=1):
        sections.append(
            {
                "section_id": f"sec{index}",
                "section_type": _fallback_section_type(index, len(paragraphs)),
                "word_count": _count_words(paragraph),
                "body": paragraph,
            }
        )
    return sections


def _strip_references_section(final_text: str) -> str:
    match = re.search(
        r"(?im)^\s*#{1,3}\s+(references|bibliography|参考文献)\s*$",
        final_text,
    )
    if match:
        return final_text[: match.start()]
    return final_text


def _parse_reference_list(final_text: str) -> List[Tuple[str, str]]:
    match = re.search(
        r"(?im)^\s*#{1,3}\s+(references|bibliography|参考文献)\s*$",
        final_text,
    )
    if not match:
        return []
    reference_text = final_text[match.end() :]
    references: List[Tuple[str, str]] = []
    for line in reference_text.splitlines():
        item_match = re.match(r"\s*\[(\d+)\]\s*(.+?)\s*$", line)
        if item_match:
            references.append((item_match.group(1), item_match.group(2).strip()))
    return references


def _extract_inline_citation_numbers(section_body: str) -> List[str]:
    return [match.group(1) for match in re.finditer(r"\[(\d+)\]", section_body)]


def _count_citation_pileups(section_body: str) -> int:
    return len(re.findall(r"(?:\[(?:\d+)\]\s*){4,}", section_body))


def _count_paragraph_tail_only_events(section_body: str) -> int:
    count = 0
    for paragraph in re.split(r"\n\s*\n", section_body.strip()):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        marker_count = len(_extract_inline_citation_numbers(paragraph))
        if marker_count == 1 and re.search(r"\[\d+\]\s*[.!?。！？]?\s*$", paragraph):
            count += 1
    return count


def _sentence_around_marker(section_body: str, marker: str) -> str:
    marker_match = re.search(rf"\[{re.escape(marker)}\]", section_body)
    if not marker_match:
        return ""
    start = max(
        section_body.rfind(".", 0, marker_match.start()),
        section_body.rfind("?", 0, marker_match.start()),
        section_body.rfind("!", 0, marker_match.start()),
        section_body.rfind("。", 0, marker_match.start()),
        section_body.rfind("？", 0, marker_match.start()),
        section_body.rfind("！", 0, marker_match.start()),
    )
    end_candidates = [
        pos
        for pos in (
            section_body.find(".", marker_match.end()),
            section_body.find("?", marker_match.end()),
            section_body.find("!", marker_match.end()),
            section_body.find("。", marker_match.end()),
            section_body.find("？", marker_match.end()),
            section_body.find("！", marker_match.end()),
        )
        if pos != -1
    ]
    end = min(end_candidates) + 1 if end_candidates else len(section_body)
    return re.sub(r"\s+", " ", section_body[start + 1 : end]).strip()


def _reference_excerpt(marker: str, references: List[Tuple[str, str]]) -> str:
    for ref_id, ref_text in references:
        if ref_id == marker:
            return ref_text[:400]
    return ""


def _default_claim_role_for_section(section_type: str) -> str:
    canonical = _canonical_section_type(section_type)
    if canonical == "introduction":
        return "background"
    if canonical == "taxonomy":
        return "definition"
    if canonical in {"limitations", "limitations_gaps", "conclusion"}:
        return "gap"
    if canonical in {"methods", "results"}:
        return "prior_result_comparison"
    return "evidence_synthesis"


def _fallback_section_type(index: int, total: int) -> str:
    if index == 1:
        return "Introduction"
    if index == total:
        return "Conclusion"
    if index == max(1, total - 1):
        return "Limitations"
    return "Evidence Synthesis"


def _normalize_citations(raw_citations: Iterable[object]) -> List[CitationEvent]:
    citations: List[CitationEvent] = []
    for index, item in enumerate(raw_citations, start=1):
        if not isinstance(item, dict):
            raise TypeError("citation_manifest 中每个 citation event 必须是字典")
        event = cast(JsonDict, item)
        citation_id = str(event.get("citation_id", f"C{index:03d}"))
        citations.append(
            CitationEvent(
                citation_id=citation_id,
                source_id=str(event.get("source_id", "unknown")),
                section_id=str(event.get("section_id", "sec1")),
                source_type=_normalize_token(event.get("source_type", "unknown")),
                claim_role=_normalize_token(event.get("claim_role", "unknown")),
                claim_span=str(event.get("claim_span", "")),
                source_excerpt=str(event.get("source_excerpt", "")),
                claim_strength=_optional_int(event.get("claim_strength")),
                evidence_strength=_optional_int(event.get("evidence_strength")),
                role_mismatch=_optional_bool(
                    event.get("role_mismatch", event.get("mismatch"))
                ),
                fidelity_penalty=_optional_float(
                    event.get("fidelity_penalty", event.get("overclaim_penalty"))
                ),
                unsupported_high_claim=_truthy(event.get("unsupported_high_claim")),
                citation_pileup=_truthy(event.get("citation_pileup")),
                paragraph_tail_only=_truthy(event.get("paragraph_tail_only")),
            )
        )
    return citations


def _normalize_sections(
    raw_sections: object, final_text: str, citations: List[CitationEvent]
) -> List[SectionStats]:
    sections: List[SectionStats] = []
    total_words = _count_words(final_text)

    if isinstance(raw_sections, list):
        for index, item in enumerate(raw_sections, start=1):
            if not isinstance(item, dict):
                raise TypeError("section_map 中每个 section 必须是字典")
            raw_section = cast(JsonDict, item)
            section_id = str(raw_section.get("section_id", f"sec{index}"))
            section_type = str(
                raw_section.get("type", raw_section.get("section_type", "other"))
            )
            word_count = _section_word_count(raw_section)
            sections.append(
                SectionStats(
                    section_id=section_id,
                    section_type=section_type,
                    canonical_type=_canonical_section_type(section_type),
                    word_count=word_count,
                )
            )

    if not sections:
        citation_section_ids = sorted({event.section_id for event in citations})
        if not citation_section_ids:
            citation_section_ids = ["sec1"]
        estimated_words = max(1, math.ceil(total_words / len(citation_section_ids)))
        sections = [
            SectionStats(
                section_id=section_id,
                section_type="other",
                canonical_type="other",
                word_count=estimated_words,
            )
            for section_id in citation_section_ids
        ]

    missing_section_ids = {
        event.section_id for event in citations
    } - {section.section_id for section in sections}
    for section_id in sorted(missing_section_ids):
        sections.append(
            SectionStats(
                section_id=section_id,
                section_type="other",
                canonical_type="other",
                word_count=1,
            )
        )

    if all(section.word_count <= 0 for section in sections):
        estimated_words = max(1, math.ceil(total_words / len(sections)))
        for section in sections:
            section.word_count = estimated_words
    else:
        for section in sections:
            if section.word_count <= 0:
                section.word_count = 1

    return sections


def _attach_citations_to_sections(
    sections: List[SectionStats], citations: List[CitationEvent]
) -> None:
    section_by_id = {section.section_id: section for section in sections}
    for event in citations:
        section = section_by_id.get(event.section_id)
        if section is None:
            continue
        section.citations.append(event)
        section.source_counts[event.source_id] += 1


def _section_word_count(raw_section: Mapping[str, object]) -> int:
    for key in ("word_count", "words"):
        value = raw_section.get(key)
        if isinstance(value, (int, float, str)) and str(value).strip() != "":
            return max(0, int(float(value)))
    text = raw_section.get("text")
    if isinstance(text, str) and text.strip():
        return _count_words(text)
    return 0


def _is_mismatched(event: CitationEvent, paper_type: str) -> bool:
    if event.role_mismatch is not None:
        return event.role_mismatch

    source_type = event.source_type
    claim_role = event.claim_role
    recommendation_roles = {"recommendation", "clinical_recommendation", "practice"}
    effect_roles = {"effect", "clinical_effect", "efficacy", "recommendation"}
    result_roles = {"result", "specific_result", "original_result", "measurement"}
    mechanism_roles = {"mechanism", "causal_mechanism"}

    if source_type in {"mechanistic", "preclinical", "animal", "in_vitro"}:
        if claim_role in recommendation_roles:
            return True
    if source_type in {"method", "dataset", "software"}:
        if claim_role in effect_roles:
            return True
    if source_type == "review":
        if claim_role in result_roles:
            return True
    if source_type == "guideline":
        if claim_role in mechanism_roles:
            return True
    if paper_type == "research":
        if claim_role in {"result", "results"} and source_type not in {"self", "unknown"}:
            return True
    return False


def _event_fidelity_penalty(event: CitationEvent) -> float:
    if event.fidelity_penalty is not None:
        return _clamp01(event.fidelity_penalty)

    claim_strength = event.claim_strength
    evidence_strength = event.evidence_strength
    if claim_strength is None:
        claim_strength = _infer_claim_strength(event.claim_role)
    if evidence_strength is None:
        evidence_strength = _infer_evidence_strength(event.source_type)

    delta = claim_strength - evidence_strength
    if delta <= 1:
        return 0.0
    if delta == 2:
        return 0.4
    return 0.8


def _infer_claim_strength(claim_role: str) -> int:
    if claim_role in {"background", "definition"}:
        return 0
    if claim_role in {"gap", "mechanism", "interpretation"}:
        return 1
    if claim_role in {"comparison", "prior_result", "evidence_synthesis"}:
        return 2
    if claim_role in {"effect", "clinical_effect", "causal"}:
        return 3
    if claim_role in {"recommendation", "clinical_recommendation", "practice"}:
        return 4
    if claim_role in {"universal_claim", "definitive"}:
        return 5
    return 1


def _infer_evidence_strength(source_type: str) -> int:
    if source_type in {"expert", "opinion", "narrative"}:
        return 0
    if source_type in {"mechanistic", "preclinical", "animal", "in_vitro"}:
        return 1
    if source_type in {"observational", "retrospective", "case_control", "cohort"}:
        return 2
    if source_type in {"trial", "rct", "prospective", "benchmark"}:
        return 3
    if source_type in {"systematic_review", "meta_analysis", "guideline", "consensus"}:
        return 4
    return 1


def _extract_section_flags(
    judge_flags: Mapping[str, object]
) -> Dict[str, Mapping[str, object]]:
    section_flags = judge_flags.get("sections", judge_flags.get("section_flags"))
    if isinstance(section_flags, dict):
        normalized: Dict[str, Mapping[str, object]] = {}
        for section_id, value in section_flags.items():
            if isinstance(value, dict):
                normalized[str(section_id)] = cast(Mapping[str, object], value)
        return normalized
    return {}


def _normalize_paper_type(raw_value: object) -> str:
    value = _normalize_token(raw_value or "review")
    if value in {"research", "original_research", "empirical", "study"}:
        return "research"
    return "review"


def _canonical_section_type(section_type: str) -> str:
    value = _normalize_token(section_type)
    if "intro" in value or "scope" in value or value in {"background"}:
        return "introduction"
    if "related" in value or value in {"prior_work", "literature_review"}:
        return "related_work"
    if "taxonom" in value or "framework" in value or "classification" in value:
        return "taxonomy"
    if "limitation" in value:
        return "limitations"
    if "gap" in value or "future" in value:
        return "limitations_gaps"
    if "conclusion" in value or "summary" in value:
        return "conclusion"
    if "discussion" in value:
        return "discussion"
    if value.startswith(("method", "material")) or value in {"methods", "materials"}:
        return "methods"
    if "evidence" in value or "synthesis" in value or value in {"analysis"}:
        return "evidence_synthesis"
    if value.startswith(("result", "finding")) or value in {"results", "findings"}:
        return "results"
    return "other"


def _distribution_role(canonical_type: str, paper_type: str) -> str:
    if paper_type == "review" and canonical_type == "limitations":
        return "limitations_gaps"
    return canonical_type


def _count_words(text: str) -> int:
    latin_tokens = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    return len(latin_tokens) + len(cjk_chars)


def _word_weights(sections: List[SectionStats]) -> Dict[str, float]:
    total_words = sum(max(0, section.word_count) for section in sections)
    if total_words <= 0:
        weight = 1.0 / len(sections) if sections else 0.0
        return {section.section_id: weight for section in sections}
    return {
        section.section_id: max(0, section.word_count) / total_words
        for section in sections
    }


def _weighted_average(values: Iterable[Tuple[float, float]]) -> float:
    total = 0.0
    weight_sum = 0.0
    for value, weight in values:
        total += _clamp01(value) * max(0.0, weight)
        weight_sum += max(0.0, weight)
    if weight_sum <= 0:
        return 0.0
    return _clamp01(total / weight_sum)


def _build_result(
    *,
    paper_type: str,
    score: float,
    penalties: Mapping[str, float],
    sections: List[SectionStats],
    citations: List[CitationEvent],
    total_words: int,
    metric_weights: Mapping[str, float],
    extra_diagnostics: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    top_sources = Counter(event.source_id for event in citations).most_common(5)
    diagnostics = _build_diagnostics(
        paper_type=paper_type,
        penalties=penalties,
        metric_weights=metric_weights,
        sections=sections,
        citations=citations,
        total_words=total_words,
        top_sources=top_sources,
        extra_diagnostics=extra_diagnostics,
    )
    return {
        "status": "evaluated",
        "paper_type": paper_type,
        "score": round(score, 4),
        "grade": _grade(score),
        "rubric": {
            "penalty_semantics": "0 is best, 1 is worst",
            "grade_thresholds": {
                "publication_ready": 0.88,
                "acceptable_with_minor_revision": 0.75,
                "major_revision_needed": 0.60,
            },
        },
        "penalties": _round_dict(penalties),
        "section_scores": [
            {
                "section_id": section.section_id,
                "type": section.section_type,
                "canonical_type": section.canonical_type,
                "word_count": section.word_count,
                "citation_count": section.citation_count,
                "score": round(section.penalties.get("section_score", 1.0), 4),
                "penalties": _round_dict(
                    {
                        key: value
                        for key, value in section.penalties.items()
                        if key != "section_score"
                    }
                ),
            }
            for section in sections
        ],
        "diagnostics": diagnostics,
    }


def _build_diagnostics(
    *,
    paper_type: str,
    penalties: Mapping[str, float],
    metric_weights: Mapping[str, float],
    sections: List[SectionStats],
    citations: List[CitationEvent],
    total_words: int,
    top_sources: List[Tuple[str, int]],
    extra_diagnostics: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    aggregated_flag_counts = _aggregate_section_penalty_counts(sections)
    diagnostics: Dict[str, object] = {
        "paper_type": paper_type,
        "citation_event_count": len(citations),
        "unique_source_count": len({event.source_id for event in citations}),
        "total_word_count": total_words,
        "top_repeated_sources": [
            {"source_id": source_id, "count": count}
            for source_id, count in top_sources
        ],
        "metric_breakdown": _metric_breakdown(penalties, metric_weights),
        "section_overview": _section_overview(sections),
        "citation_events": _citation_event_diagnostics(citations),
        "flagged_events": _flagged_event_diagnostics(citations),
        "judge_aggregated_flags": aggregated_flag_counts,
        "flag_semantics": {
            "citation_events_and_flagged_events": (
                "Only retained citation_manifest events that survived judge filtering."
            ),
            "judge_aggregated_flags": (
                "Section-level judge counts, including unsupported or filtered items "
                "that may not appear as retained citation events."
            ),
        },
    }
    if extra_diagnostics:
        diagnostics.update(extra_diagnostics)
    return diagnostics


def _citation_quality_input_diagnostics(config: Mapping[str, object]) -> Dict[str, object]:
    metadata = config.get("metadata")
    metadata_dict = metadata if isinstance(metadata, dict) else {}
    return {
        "config_source": str(
            config.get("config_source", metadata_dict.get("config_source", "structured_manifest"))
        ),
        "reference_count": metadata_dict.get("reference_count"),
        "inline_citation_marker_count": metadata_dict.get("inline_citation_marker_count"),
        "parse_warnings": list(metadata_dict.get("parse_warnings", []))
        if isinstance(metadata_dict.get("parse_warnings"), list)
        else [],
        "parse_note": metadata_dict.get("parse_note"),
    }


def _metric_breakdown(
    penalties: Mapping[str, float],
    metric_weights: Mapping[str, float],
) -> Dict[str, Dict[str, object]]:
    breakdown: Dict[str, Dict[str, object]] = {}
    for metric, penalty in penalties.items():
        weight = float(metric_weights.get(metric, 0.0))
        breakdown[metric] = {
            "penalty": round(float(penalty), 4),
            "weight": round(weight, 4),
            "weighted_deduction": round(float(penalty) * weight, 4),
        }
    return breakdown


def _section_overview(sections: List[SectionStats]) -> List[Dict[str, object]]:
    overview: List[Dict[str, object]] = []
    for section in sections:
        top_sources = section.source_counts.most_common(5)
        overview.append(
            {
                "section_id": section.section_id,
                "type": section.section_type,
                "canonical_type": section.canonical_type,
                "word_count": section.word_count,
                "citation_count": section.citation_count,
                "unique_source_count": len(section.source_counts),
                "top_sources": [
                    {"source_id": source_id, "count": count}
                    for source_id, count in top_sources
                ],
                "penalties": _round_dict(section.penalties),
            }
        )
    return overview


def _aggregate_section_penalty_counts(
    sections: List[SectionStats],
) -> Dict[str, float]:
    totals = {
        "unsupported_high_claims": 0.0,
        "judge_declared_unsupported_high_claims": 0.0,
        "weighted_judge_declared_unsupported_high_claims": 0.0,
        "retrieval_unsupported_alignment_events": 0.0,
        "wrong_source_alignment_events": 0.0,
        "weak_match_alignment_events": 0.0,
        "citation_pileups": 0.0,
        "paragraph_tail_only_events": 0.0,
    }
    for section in sections:
        for key in totals:
            totals[key] += float(section.penalties.get(key, 0.0))
    return _round_dict(totals)


def _citation_event_diagnostics(
    citations: List[CitationEvent],
) -> List[Dict[str, object]]:
    return [
        {
            "citation_id": event.citation_id,
            "section_id": event.section_id,
            "source_id": event.source_id,
            "source_type": event.source_type,
            "claim_role": event.claim_role,
            "claim_strength": event.claim_strength,
            "evidence_strength": event.evidence_strength,
            "role_mismatch": event.role_mismatch,
            "fidelity_penalty": round(_event_fidelity_penalty(event), 4),
            "unsupported_high_claim": event.unsupported_high_claim,
            "citation_pileup": event.citation_pileup,
            "paragraph_tail_only": event.paragraph_tail_only,
            "claim_span": event.claim_span,
            "source_excerpt": event.source_excerpt,
        }
        for event in citations
    ]


def _flagged_event_diagnostics(
    citations: List[CitationEvent],
) -> Dict[str, List[Dict[str, object]]]:
    return {
        "role_mismatch_events": [
            _flagged_event_record(event)
            for event in citations
            if _truthy(event.role_mismatch)
        ],
        "overclaim_events": [
            _flagged_event_record(event)
            for event in citations
            if _event_fidelity_penalty(event) > 0.0
        ],
        "unsupported_high_claim_events": [
            _flagged_event_record(event)
            for event in citations
            if event.unsupported_high_claim
        ],
        "citation_pileup_events": [
            _flagged_event_record(event)
            for event in citations
            if event.citation_pileup
        ],
        "paragraph_tail_only_events": [
            _flagged_event_record(event)
            for event in citations
            if event.paragraph_tail_only
        ],
    }


def _flagged_event_record(event: CitationEvent) -> Dict[str, object]:
    return {
        "citation_id": event.citation_id,
        "section_id": event.section_id,
        "source_id": event.source_id,
        "source_type": event.source_type,
        "claim_role": event.claim_role,
        "claim_strength": event.claim_strength,
        "evidence_strength": event.evidence_strength,
        "fidelity_penalty": round(_event_fidelity_penalty(event), 4),
        "claim_span": event.claim_span,
    }


def _grade(score: float) -> str:
    if score >= 0.88:
        return "publication_ready"
    if score >= 0.75:
        return "acceptable_with_minor_revision"
    if score >= 0.60:
        return "major_revision_needed"
    return "citation_use_unreliable"


def _skipped(reason: str) -> Dict[str, object]:
    return {"status": "skipped", "reason": reason}


def _round_dict(values: Mapping[str, Union[float, int]]) -> Dict[str, float]:
    return {key: round(float(value), 4) for key, value in values.items()}


def _as_dict(value: object) -> Mapping[str, object]:
    if isinstance(value, dict):
        return cast(Mapping[str, object], value)
    return {}


def _as_list(value: object) -> List[object]:
    if isinstance(value, list):
        return list(value)
    return []


def _optional_int(value: object) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: object) -> Optional[bool]:
    if value is None:
        return None
    return _truthy(value)


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return False


def _as_count(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return max(0, int(value))
    if isinstance(value, list):
        return len(value)
    return 0


def _normalize_token(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
