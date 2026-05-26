"""Six-slot section priors derived from real medical review articles."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import mean, median
from typing import Mapping, Sequence

from .citation import CitationChunk, classify_chunk_roles, split_into_seven_sections
from .structure import (
    ABSTRACT_HEADINGS,
    CONCLUSION_HEADINGS,
    INTRODUCTION_HEADINGS,
    LIMITATIONS_HEADINGS,
    KEYWORD_HEADINGS,
    NON_BODY_HEADINGS,
    REFERENCE_HEADINGS,
    count_length_units,
    normalize_heading,
    parse_markdown_sections,
)


SIX_SLOT_ORDER = (
    "scope_context",
    "framework_mechanism",
    "evidence_methods",
    "findings_synthesis",
    "implications_discussion",
    "limitations_future",
)

SIX_SLOT_PRIORS_PATH = (
    Path(__file__).resolve().parent.parent / "data_sample" / "six_slot_section_priors.json"
)
SIX_SLOT_CITATION_PRIORS_PATH = (
    Path(__file__).resolve().parent.parent / "data_sample" / "six_slot_citation_priors.json"
)
PRIOR_VERSION_HASH_LENGTH = 12

SCOPE_TERMS = (
    "scope",
    "background",
    "introduction",
    "context",
    "terminology",
    "overview",
    "disease context",
)

FRAMEWORK_TERMS = (
    "framework",
    "pathway",
    "pathogenesis",
    "mechanism",
    "mechanisms",
    "immune privilege",
    "classification",
    "taxonomy",
    "organizing",
)

EVIDENCE_METHODS_TERMS = (
    "methods",
    "method",
    "materials and methods",
    "study design",
    "search strategy",
    "eligibility",
    "measurement",
    "measurements",
    "assay",
    "assays",
    "blood and skin",
    "serum",
    "plasma",
    "biopsy",
)

FINDINGS_TERMS = (
    "results",
    "findings",
    "signatures",
    "signature",
    "patterns",
    "pattern",
    "profiles",
    "profile",
    "associations",
    "evidence base",
    "evidence synthesis",
    "meta-analysis",
    "meta analysis",
)

IMPLICATIONS_TERMS = (
    "discussion",
    "implication",
    "implications",
    "clinical",
    "therapeutic",
    "therapy",
    "treatment",
    "management",
    "biomarker value",
    "decision-making",
)

LIMITATIONS_TERMS = (
    "limitations",
    "limitation",
    "heterogeneity",
    "future",
    "future work",
    "future research",
    "research priorities",
    "research agenda",
    "conclusion",
    "conclusions",
    "summary",
    "gaps",
)

BODY_SLOT_ORDER = (
    "framework_mechanism",
    "evidence_methods",
    "findings_synthesis",
    "implications_discussion",
)

BODY_SLOT_ANCHORS: dict[str, float] = {
    "framework_mechanism": 0.125,
    "evidence_methods": 0.375,
    "findings_synthesis": 0.625,
    "implications_discussion": 0.875,
}

BODY_SLOT_CUE_TERMS: dict[str, tuple[str, ...]] = {
    "framework_mechanism": (
        "pathway",
        "mechanism",
        "mechanisms",
        "framework",
        "axis",
        "cascade",
        "classification",
    ),
    "evidence_methods": (
        "methods",
        "assay",
        "measurement",
        "measurements",
        "cohort",
        "trial",
        "serum",
        "plasma",
        "biopsy",
        "protocol",
        "screening",
    ),
    "findings_synthesis": (
        "results",
        "findings",
        "elevated",
        "associated",
        "pattern",
        "profile",
        "comparison",
        "compare",
        "synthesis",
        "evidence",
    ),
    "implications_discussion": (
        "clinical",
        "therapeutic",
        "management",
        "practice",
        "implications",
        "future",
        "limitations",
        "uncertainty",
        "decision-making",
    ),
}

BODY_SLOT_DISTANCE_WEIGHT = 2.1
BODY_SLOT_BASE_HEIGHT = 1.12
BODY_SLOT_MIN_SCORE = 0.01
BODY_SLOT_CUE_WEIGHT = 0.05

EXTRA_METADATA_TERMS = (
    "funding",
    "data availability",
    "author contributions",
    "conflict of interest",
    "generative ai statement",
    "publisher",
    "supplementary material",
)


@dataclass(frozen=True)
class SixSlotStats:
    """Aggregate share statistics for one six-slot template position."""

    mean_share: float
    median_share: float
    p25_share: float
    p75_share: float
    mean_words: float
    median_words: float
    articles_with_content: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "mean_share": round(self.mean_share, 6),
            "median_share": round(self.median_share, 6),
            "p25_share": round(self.p25_share, 6),
            "p75_share": round(self.p75_share, 6),
            "mean_words": round(self.mean_words, 2),
            "median_words": round(self.median_words, 2),
            "articles_with_content": self.articles_with_content,
        }


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    normalized = " ".join(str(text).casefold().split())
    return any(" ".join(term.casefold().split()) in normalized for term in terms)


def _paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n+", str(text)) if part.strip()]


def _normalized_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text).casefold()))


def _term_score(text: str, terms: Sequence[str]) -> float:
    normalized_text = " ".join(str(text).casefold().split())
    normalized_word_set = _normalized_words(normalized_text)
    score = 0.0
    for term in terms:
        normalized_term = " ".join(str(term).casefold().split())
        if " " in normalized_term:
            if normalized_term in normalized_text:
                score += 1.0
        elif normalized_term in normalized_word_set:
            score += 1.0
    return score


def _soft_assign_main_body_slots(main_body_text: str) -> dict[str, float]:
    paragraphs = _paragraphs(main_body_text)
    paragraph_words = [count_length_units(paragraph) for paragraph in paragraphs]
    return _soft_assign_weighted_paragraphs(list(zip(paragraphs, paragraph_words)))


def _soft_assign_weighted_paragraphs(
    paragraph_records: Sequence[tuple[str, int]],
) -> dict[str, float]:
    paragraphs = [paragraph for paragraph, _word_count in paragraph_records if paragraph]
    paragraph_words = [word_count for _paragraph, word_count in paragraph_records if word_count > 0]
    total_body_words = sum(paragraph_words)
    counts = {slot: 0.0 for slot in BODY_SLOT_ORDER}
    if total_body_words <= 0:
        return counts

    cumulative_words = 0
    for paragraph, word_count in zip(paragraphs, paragraph_words):
        if word_count <= 0:
            continue
        paragraph_center = (cumulative_words + word_count / 2.0) / total_body_words
        raw_scores: dict[str, float] = {}
        for slot in BODY_SLOT_ORDER:
            position_score = max(
                BODY_SLOT_MIN_SCORE,
                BODY_SLOT_BASE_HEIGHT
                - abs(paragraph_center - BODY_SLOT_ANCHORS[slot]) * BODY_SLOT_DISTANCE_WEIGHT,
            )
            cue_score = BODY_SLOT_CUE_WEIGHT * _term_score(
                paragraph,
                BODY_SLOT_CUE_TERMS[slot],
            )
            raw_scores[slot] = position_score + cue_score
        score_total = sum(raw_scores.values())
        if score_total <= 0:
            counts["findings_synthesis"] += word_count
        else:
            for slot, score in raw_scores.items():
                counts[slot] += word_count * score / score_total
        cumulative_words += word_count
    return counts


def derive_article_six_slot_word_counts(
    markdown_text: str,
) -> dict[str, float]:
    """Project one review article into the shared six-slot template.

    For real-review tasks we want the slot counts to respect the article's
    actual body structure rather than a coarse intro/main_body/conclusion
    reduction. We therefore classify paragraph-level content chunks in
    sequence, while still reusing the same deterministic chunk-role and
    six-slot mappers used elsewhere in MetaBench.
    """

    slot_word_counts = {slot: 0.0 for slot in SIX_SLOT_ORDER}
    retained_body_records: list[tuple[str, int]] = []

    for section in parse_markdown_sections(markdown_text):
        if section.is_title:
            continue
        normalized_heading = normalize_heading(section.heading)
        if normalized_heading in ABSTRACT_HEADINGS:
            continue
        if normalized_heading in KEYWORD_HEADINGS:
            continue
        if normalized_heading in REFERENCE_HEADINGS:
            continue
        if _is_metadata_heading(normalized_heading):
            continue

        if normalized_heading in INTRODUCTION_HEADINGS:
            slot_word_counts["scope_context"] += float(count_length_units(section.body))
            continue

        if normalized_heading in CONCLUSION_HEADINGS or normalized_heading in LIMITATIONS_HEADINGS:
            slot_word_counts["limitations_future"] += float(count_length_units(section.body))
            continue

        for paragraph in _paragraphs(section.body):
            word_count = count_length_units(paragraph)
            if word_count <= 0:
                continue

            scoring_text = f"{section.heading}\n{paragraph}"
            normalized_scoring_text = " ".join(scoring_text.casefold().split())
            if _contains_any(normalized_scoring_text, LIMITATIONS_TERMS):
                slot_word_counts["limitations_future"] += float(word_count)
                continue

            retained_body_records.append((scoring_text, word_count))

    if retained_body_records:
        middle_counts = _soft_assign_weighted_paragraphs(retained_body_records)
        for slot in BODY_SLOT_ORDER:
            slot_word_counts[slot] += float(middle_counts.get(slot, 0.0))

    return slot_word_counts


def derive_template_six_slot_word_counts(
    markdown_text: str,
) -> dict[str, float]:
    """Project one review article into six template slots for corpus priors.

    This keeps the original template-oriented reduction used to estimate the
    shared six-slot prior: introduction and conclusion map directly to slots 1
    and 6, and the remaining main body is softly distributed across the middle
    four slots.
    """

    sections = split_into_seven_sections(markdown_text)
    slot_word_counts = {slot: 0.0 for slot in SIX_SLOT_ORDER}
    slot_word_counts["scope_context"] = float(sections["introduction"].word_count)
    slot_word_counts["limitations_future"] = float(sections["conclusion"].word_count)
    middle_counts = _soft_assign_main_body_slots(sections["main_body"].text)
    for slot in BODY_SLOT_ORDER:
        slot_word_counts[slot] = float(middle_counts.get(slot, 0.0))
    return slot_word_counts


def _citation_marker_count(text: str) -> int:
    """Count numeric reference mentions in a review body fragment.

    The source review corpus uses compact numeric citations such as ``( 1 )``,
    ``(1, 2)`` and ``[1-3]``.  We count source mentions rather than bracket
    groups, while ignoring obvious years such as ``(2025)``.
    """

    count = 0
    for match in re.finditer(r"(?:\[\s*([0-9][0-9,\s;–—-]*)\s*\]|\(\s*([0-9][0-9,\s;–—-]*)\s*\))", text):
        content = match.group(1) or match.group(2) or ""
        for start, end in re.findall(r"(\d+)(?:\s*[–—-]\s*(\d+))?", content):
            start_value = int(start)
            if start_value > 300:
                continue
            if end:
                end_value = int(end)
                if end_value > 300 or end_value < start_value:
                    continue
                count += min(end_value - start_value + 1, 50)
            else:
                count += 1
    return count


def _soft_assign_citation_paragraphs(
    paragraph_records: Sequence[tuple[str, int, int]],
) -> dict[str, float]:
    """Softly assign paragraph citation counts across the four middle slots."""

    records = [
        (paragraph, word_count, citation_count)
        for paragraph, word_count, citation_count in paragraph_records
        if paragraph and word_count > 0 and citation_count > 0
    ]
    total_body_words = sum(word_count for _paragraph, word_count, _citation_count in records)
    counts = {slot: 0.0 for slot in BODY_SLOT_ORDER}
    if total_body_words <= 0:
        return counts

    cumulative_words = 0
    for paragraph, word_count, citation_count in records:
        paragraph_center = (cumulative_words + word_count / 2.0) / total_body_words
        raw_scores: dict[str, float] = {}
        for slot in BODY_SLOT_ORDER:
            position_score = max(
                BODY_SLOT_MIN_SCORE,
                BODY_SLOT_BASE_HEIGHT
                - abs(paragraph_center - BODY_SLOT_ANCHORS[slot]) * BODY_SLOT_DISTANCE_WEIGHT,
            )
            cue_score = BODY_SLOT_CUE_WEIGHT * _term_score(
                paragraph,
                BODY_SLOT_CUE_TERMS[slot],
            )
            raw_scores[slot] = position_score + cue_score
        score_total = sum(raw_scores.values())
        if score_total <= 0:
            counts["findings_synthesis"] += citation_count
        else:
            for slot, score in raw_scores.items():
                counts[slot] += citation_count * score / score_total
        cumulative_words += word_count
    return counts


def derive_article_six_slot_citation_counts(
    markdown_text: str,
) -> dict[str, float]:
    """Project one review article's body citation mentions into six slots."""

    slot_citation_counts = {slot: 0.0 for slot in SIX_SLOT_ORDER}
    retained_middle_records: list[tuple[str, int, int]] = []

    for section in parse_markdown_sections(markdown_text):
        if section.is_title:
            continue
        normalized_heading = normalize_heading(section.heading)
        if normalized_heading in ABSTRACT_HEADINGS:
            continue
        if normalized_heading in KEYWORD_HEADINGS:
            continue
        if normalized_heading in REFERENCE_HEADINGS:
            continue
        if _is_metadata_heading(normalized_heading):
            continue

        section_citations = _citation_marker_count(section.body)
        if section_citations <= 0:
            continue

        if normalized_heading in INTRODUCTION_HEADINGS:
            slot_citation_counts["scope_context"] += float(section_citations)
            continue

        if normalized_heading in CONCLUSION_HEADINGS or normalized_heading in LIMITATIONS_HEADINGS:
            slot_citation_counts["limitations_future"] += float(section_citations)
            continue

        if normalized_heading in EVIDENCE_METHODS_TERMS or _contains_any(normalized_heading, EVIDENCE_METHODS_TERMS):
            slot_citation_counts["evidence_methods"] += float(section_citations)
            continue

        if normalized_heading in FINDINGS_TERMS or _contains_any(normalized_heading, FINDINGS_TERMS):
            slot_citation_counts["findings_synthesis"] += float(section_citations)
            continue

        if normalized_heading in IMPLICATIONS_TERMS or _contains_any(normalized_heading, IMPLICATIONS_TERMS):
            slot_citation_counts["implications_discussion"] += float(section_citations)
            continue

        if normalized_heading in FRAMEWORK_TERMS or _contains_any(normalized_heading, FRAMEWORK_TERMS):
            slot_citation_counts["framework_mechanism"] += float(section_citations)
            continue

        for paragraph in _paragraphs(section.body):
            citation_count = _citation_marker_count(paragraph)
            word_count = count_length_units(paragraph)
            if citation_count <= 0 or word_count <= 0:
                continue
            scoring_text = f"{section.heading}\n{paragraph}"
            if _contains_any(scoring_text, LIMITATIONS_TERMS):
                slot_citation_counts["limitations_future"] += float(citation_count)
                continue
            retained_middle_records.append((scoring_text, word_count, citation_count))

    if retained_middle_records:
        middle_counts = _soft_assign_citation_paragraphs(retained_middle_records)
        for slot in BODY_SLOT_ORDER:
            slot_citation_counts[slot] += float(middle_counts.get(slot, 0.0))

    return slot_citation_counts


def derive_template_six_slot_citation_counts(
    markdown_text: str,
) -> dict[str, float]:
    """Project one review article's citations into the shared six-slot template."""

    sections = split_into_seven_sections(markdown_text)
    slot_citation_counts = {slot: 0.0 for slot in SIX_SLOT_ORDER}
    slot_citation_counts["scope_context"] = float(
        _citation_marker_count(sections["introduction"].text)
    )
    slot_citation_counts["limitations_future"] = float(
        _citation_marker_count(sections["conclusion"].text)
    )

    middle_records: list[tuple[str, int, int]] = []
    for paragraph in _paragraphs(sections["main_body"].text):
        word_count = count_length_units(paragraph)
        citation_count = _citation_marker_count(paragraph)
        if word_count <= 0 or citation_count <= 0:
            continue
        middle_records.append((paragraph, word_count, citation_count))

    middle_counts = _soft_assign_citation_paragraphs(middle_records)
    for slot in BODY_SLOT_ORDER:
        slot_citation_counts[slot] = float(middle_counts.get(slot, 0.0))

    return slot_citation_counts


def _is_metadata_heading(heading: str) -> bool:
    normalized = " ".join(str(heading).casefold().replace("’", "'").split())
    if normalized in NON_BODY_HEADINGS:
        return True
    return any(term in normalized for term in EXTRA_METADATA_TERMS)


def _non_metadata_sections(markdown_text: str) -> list[CitationChunk]:
    sections = parse_markdown_sections(markdown_text)
    chunks: list[CitationChunk] = []
    for section in sections:
        if section.is_title:
            continue
        heading = normalize_heading(section.heading)
        if heading in ABSTRACT_HEADINGS:
            continue
        if heading in KEYWORD_HEADINGS:
            continue
        if heading in REFERENCE_HEADINGS:
            continue
        if _is_metadata_heading(heading):
            continue
        chunks.append(
            CitationChunk(
                chunk_id=f"sec{len(chunks) + 1}",
                index=len(chunks) + 1,
                title=section.heading,
                body=section.body,
                word_count=count_length_units(section.body),
            )
        )
    return chunks


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def classify_chunk_to_six_slot(
    chunk: CitationChunk,
    *,
    total_chunks: int,
    canonical_type: str | None = None,
) -> str:
    """Map one content chunk into the six-slot review template."""

    text = f"{chunk.title}\n{chunk.body}"
    title = chunk.title
    normalized_title = " ".join(title.casefold().replace("’", "'").split())
    index = chunk.index
    role = canonical_type or "other"

    if role == "introduction" or index == 1 or _contains_any(normalized_title, SCOPE_TERMS):
        return "scope_context"
    if role in {"limitations_gaps", "conclusion"} or _contains_any(normalized_title, LIMITATIONS_TERMS):
        return "limitations_future"
    if role in {"methods"} or _contains_any(normalized_title, EVIDENCE_METHODS_TERMS):
        return "evidence_methods"
    if role in {"results"} or _contains_any(normalized_title, FINDINGS_TERMS):
        return "findings_synthesis"
    if role in {"discussion"} or _contains_any(normalized_title, IMPLICATIONS_TERMS):
        return "implications_discussion"
    if role in {"taxonomy"} or (index == 2 and _contains_any(normalized_title, FRAMEWORK_TERMS)):
        return "framework_mechanism"
    if _contains_any(normalized_title, FRAMEWORK_TERMS):
        if index <= 2:
            return "framework_mechanism"
        return "findings_synthesis"
    if role == "evidence_synthesis":
        if total_chunks >= 6 and index == 2:
            return "framework_mechanism"
        if total_chunks >= 6 and index == 3:
            return "evidence_methods"
        if total_chunks >= 6 and index == total_chunks - 2:
            return "findings_synthesis"
        if total_chunks >= 6 and index == total_chunks - 1:
            return "implications_discussion"
        if index <= max(2, total_chunks // 2):
            return "evidence_methods"
        return "findings_synthesis"

    if index == total_chunks:
        return "limitations_future"
    if total_chunks >= 6 and index == 2:
        return "framework_mechanism"
    if total_chunks >= 6 and index == 3:
        return "evidence_methods"
    if total_chunks >= 6 and index == total_chunks - 2:
        return "findings_synthesis"
    if index == total_chunks - 1:
        return "implications_discussion"
    if index <= max(3, total_chunks // 2):
        return "evidence_methods"
    return "findings_synthesis"


def classify_outline_to_six_slots(
    outline_items: Sequence[tuple[str, str]],
) -> dict[str, str]:
    """Map an outline into the six-slot template when it matches the expected shape."""

    if len(outline_items) != len(SIX_SLOT_ORDER):
        return {}

    chunks = [
        CitationChunk(
            chunk_id=section_id,
            index=index,
            title=title,
            body=title,
            word_count=max(1, len(title.split())),
        )
        for index, (section_id, title) in enumerate(outline_items, start=1)
    ]
    classifications = classify_chunk_roles(chunks, reference={"outline": dict(outline_items)})
    slot_by_section: dict[str, str] = {}
    for chunk, classification in zip(chunks, classifications):
        slot_by_section[chunk.chunk_id] = classify_chunk_to_six_slot(
            chunk,
            total_chunks=len(chunks),
            canonical_type=classification.canonical_type,
        )

    slots_in_order = [slot_by_section[section_id] for section_id, _title in outline_items]
    if len(set(slots_in_order)) != len(SIX_SLOT_ORDER):
        return {}
    if tuple(slots_in_order) != SIX_SLOT_ORDER:
        return {}
    return slot_by_section


def derive_six_slot_priors_from_corpus(
    corpus_dir: Path,
) -> dict[str, object]:
    """Derive six-slot share priors from a directory of real review articles."""

    article_paths = sorted(path for path in corpus_dir.glob("PMC*.md") if path.is_file())
    per_slot_shares = {slot: [] for slot in SIX_SLOT_ORDER}
    per_slot_words = {slot: [] for slot in SIX_SLOT_ORDER}
    mapped_article_count = 0

    for article_path in article_paths:
        markdown_text = article_path.read_text(encoding="utf-8")
        slot_word_counts = derive_template_six_slot_word_counts(markdown_text)

        total_words = sum(slot_word_counts.values())
        if total_words <= 0:
            continue

        mapped_article_count += 1
        for slot in SIX_SLOT_ORDER:
            words = slot_word_counts[slot]
            share = words / total_words
            per_slot_shares[slot].append(share)
            per_slot_words[slot].append(words)

    slot_stats = {
        slot: SixSlotStats(
            mean_share=mean(per_slot_shares[slot]) if per_slot_shares[slot] else 0.0,
            median_share=median(per_slot_shares[slot]) if per_slot_shares[slot] else 0.0,
            p25_share=_percentile(per_slot_shares[slot], 0.25),
            p75_share=_percentile(per_slot_shares[slot], 0.75),
            mean_words=mean(per_slot_words[slot]) if per_slot_words[slot] else 0.0,
            median_words=median(per_slot_words[slot]) if per_slot_words[slot] else 0.0,
            articles_with_content=sum(1 for item in per_slot_words[slot] if item > 0),
        )
        for slot in SIX_SLOT_ORDER
    }
    normalized_mean_shares = {
        slot: slot_stats[slot].mean_share
        for slot in SIX_SLOT_ORDER
    }
    total_mean_share = sum(normalized_mean_shares.values())
    if total_mean_share > 0:
        normalized_mean_shares = {
            slot: normalized_mean_shares[slot] / total_mean_share
            for slot in SIX_SLOT_ORDER
        }

    return {
        "source_corpus_dir": str(corpus_dir),
        "article_count": len(article_paths),
        "mapped_article_count": mapped_article_count,
        "slot_order": list(SIX_SLOT_ORDER),
        "normalized_mean_shares": {
            slot: round(share, 6)
            for slot, share in normalized_mean_shares.items()
        },
        "slot_stats": {
            slot: stats.to_dict()
            for slot, stats in slot_stats.items()
        },
        "method": {
            "unit": "per-article normalized body-word share",
            "aggregation": "mean share across mapped articles, then renormalized to sum to 1.0",
            "notes": [
                "Each article is first reduced with the same deterministic introduction/main_body/conclusion parser used elsewhere in MetaBench.",
                "The introduction and conclusion map directly to slots 1 and 6.",
                "Paragraphs inside the parsed main_body are softly distributed across the four middle slots using relative position plus light lexical cues, yielding a corpus-linked six-slot prior without an LLM.",
            ],
        },
    }


def _build_slot_share_stats(
    *,
    per_slot_shares: Mapping[str, Sequence[float]],
    per_slot_units: Mapping[str, Sequence[float]],
    unit_label: str,
) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = {}
    for slot in SIX_SLOT_ORDER:
        shares = list(per_slot_shares.get(slot, []))
        units = list(per_slot_units.get(slot, []))
        stats[slot] = {
            "mean_share": round(mean(shares), 6) if shares else 0.0,
            "median_share": round(median(shares), 6) if shares else 0.0,
            "p25_share": round(_percentile(shares, 0.25), 6),
            "p75_share": round(_percentile(shares, 0.75), 6),
            f"mean_{unit_label}": round(mean(units), 2) if units else 0.0,
            f"median_{unit_label}": round(median(units), 2) if units else 0.0,
            f"articles_with_{unit_label}": sum(1 for item in units if item > 0),
        }
    return stats


def derive_six_slot_citation_priors_from_corpus(
    corpus_dir: Path,
) -> dict[str, object]:
    """Derive six-slot citation-distribution priors from real review articles."""

    article_paths = sorted(path for path in corpus_dir.glob("PMC*.md") if path.is_file())
    per_slot_shares = {slot: [] for slot in SIX_SLOT_ORDER}
    per_slot_citations = {slot: [] for slot in SIX_SLOT_ORDER}
    mapped_article_count = 0
    total_citation_mentions = 0

    for article_path in article_paths:
        markdown_text = article_path.read_text(encoding="utf-8", errors="ignore")
        slot_citation_counts = derive_template_six_slot_citation_counts(markdown_text)

        total_citations = sum(slot_citation_counts.values())
        if total_citations <= 0:
            continue

        mapped_article_count += 1
        total_citation_mentions += int(round(total_citations))
        for slot in SIX_SLOT_ORDER:
            citations = float(slot_citation_counts[slot])
            share = citations / total_citations
            per_slot_shares[slot].append(share)
            per_slot_citations[slot].append(citations)

    slot_stats = _build_slot_share_stats(
        per_slot_shares=per_slot_shares,
        per_slot_units=per_slot_citations,
        unit_label="citations",
    )
    normalized_mean_shares = {
        slot: float(slot_stats[slot]["mean_share"])
        for slot in SIX_SLOT_ORDER
    }
    total_mean_share = sum(normalized_mean_shares.values())
    if total_mean_share > 0:
        normalized_mean_shares = {
            slot: normalized_mean_shares[slot] / total_mean_share
            for slot in SIX_SLOT_ORDER
        }

    return {
        "source_corpus_dir": str(corpus_dir),
        "article_count": len(article_paths),
        "mapped_article_count": mapped_article_count,
        "total_citation_mentions": total_citation_mentions,
        "slot_order": list(SIX_SLOT_ORDER),
        "normalized_mean_shares": {
            slot: round(share, 6)
            for slot, share in normalized_mean_shares.items()
        },
        "slot_stats": slot_stats,
        "method": {
            "unit": "per-article normalized body citation mention share",
            "aggregation": "mean citation share across mapped articles, then renormalized to sum to 1.0",
            "notes": [
                "Numeric body citation markers such as (1), (1, 2), and [1-3] are counted as reference mentions.",
                "Reference-list, abstract, keyword, funding, conflict, and other metadata sections are excluded.",
                "The same deterministic introduction/main_body/conclusion reduction used for the six-slot word priors is reused here.",
                "Introduction and conclusion map directly to slots 1 and 6; citation-bearing paragraphs inside the parsed main_body are softly distributed across the middle four slots using relative position plus light lexical cues.",
            ],
        },
    }


@lru_cache(maxsize=1)
def load_six_slot_prior_payload(
    path: Path | None = None,
) -> dict[str, object] | None:
    """Load the raw six-slot prior payload from disk."""

    priors_path = path or SIX_SLOT_PRIORS_PATH
    if not priors_path.exists():
        return None
    payload = json.loads(priors_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return None
    return dict(payload)


@lru_cache(maxsize=1)
def load_six_slot_citation_prior_payload(
    path: Path | None = None,
) -> dict[str, object] | None:
    """Load the raw six-slot citation prior payload from disk."""

    priors_path = path or SIX_SLOT_CITATION_PRIORS_PATH
    if not priors_path.exists():
        return None
    payload = json.loads(priors_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return None
    return dict(payload)


def _build_prior_version_payload(
    payload: Mapping[str, object],
) -> dict[str, object]:
    return {
        "article_count": payload.get("article_count"),
        "mapped_article_count": payload.get("mapped_article_count"),
        "slot_order": payload.get("slot_order"),
        "normalized_mean_shares": payload.get("normalized_mean_shares"),
        "slot_stats": payload.get("slot_stats"),
    }


def _compute_prior_version(payload: Mapping[str, object]) -> str:
    canonical_payload = _build_prior_version_payload(payload)
    encoded = json.dumps(
        canonical_payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:PRIOR_VERSION_HASH_LENGTH]


@lru_cache(maxsize=1)
def load_six_slot_prior_metadata(
    path: Path | None = None,
) -> dict[str, object] | None:
    """Load stable metadata describing the active six-slot priors."""

    payload = load_six_slot_prior_payload(path)
    if payload is None:
        return None

    raw_slot_order = payload.get("slot_order")
    if not isinstance(raw_slot_order, Sequence):
        return None

    slot_order = [str(slot).strip() for slot in raw_slot_order if str(slot).strip()]
    if slot_order != list(SIX_SLOT_ORDER):
        return None

    return {
        "prior_version": _compute_prior_version(payload),
        "source_corpus_dir": str(payload.get("source_corpus_dir", "")),
        "article_count": int(payload.get("article_count", 0) or 0),
        "mapped_article_count": int(payload.get("mapped_article_count", 0) or 0),
        "slot_order": slot_order,
    }


@lru_cache(maxsize=1)
def load_six_slot_citation_prior_metadata(
    path: Path | None = None,
) -> dict[str, object] | None:
    """Load stable metadata describing the active six-slot citation priors."""

    payload = load_six_slot_citation_prior_payload(path)
    if payload is None:
        return None

    raw_slot_order = payload.get("slot_order")
    if not isinstance(raw_slot_order, Sequence):
        return None

    slot_order = [str(slot).strip() for slot in raw_slot_order if str(slot).strip()]
    if slot_order != list(SIX_SLOT_ORDER):
        return None

    return {
        "prior_version": _compute_prior_version(payload),
        "source_corpus_dir": str(payload.get("source_corpus_dir", "")),
        "article_count": int(payload.get("article_count", 0) or 0),
        "mapped_article_count": int(payload.get("mapped_article_count", 0) or 0),
        "total_citation_mentions": int(payload.get("total_citation_mentions", 0) or 0),
        "slot_order": slot_order,
    }


@lru_cache(maxsize=1)
def load_six_slot_priors(
    path: Path | None = None,
) -> dict[str, float] | None:
    """Load normalized six-slot priors if they exist and validate them."""

    payload = load_six_slot_prior_payload(path)
    if payload is None:
        return None
    raw_shares = payload.get("normalized_mean_shares")
    if not isinstance(raw_shares, Mapping):
        return None
    shares: dict[str, float] = {}
    for slot in SIX_SLOT_ORDER:
        raw_value = raw_shares.get(slot)
        if not isinstance(raw_value, (int, float)):
            return None
        shares[slot] = float(raw_value)
    total = sum(shares.values())
    if total <= 0:
        return None
    return {
        slot: shares[slot] / total
        for slot in SIX_SLOT_ORDER
    }
