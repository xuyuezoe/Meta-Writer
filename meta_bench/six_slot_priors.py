"""Six-slot section priors derived from real medical review articles."""

from __future__ import annotations

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
        sections = split_into_seven_sections(markdown_text)
        slot_word_counts = {slot: 0.0 for slot in SIX_SLOT_ORDER}
        slot_word_counts["scope_context"] = float(sections["introduction"].word_count)
        slot_word_counts["limitations_future"] = float(sections["conclusion"].word_count)
        middle_counts = _soft_assign_main_body_slots(sections["main_body"].text)
        for slot in BODY_SLOT_ORDER:
            slot_word_counts[slot] = float(middle_counts.get(slot, 0.0))

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


@lru_cache(maxsize=1)
def load_six_slot_priors(
    path: Path | None = None,
) -> dict[str, float] | None:
    """Load normalized six-slot priors if they exist and validate them."""

    priors_path = path or SIX_SLOT_PRIORS_PATH
    if not priors_path.exists():
        return None
    payload = json.loads(priors_path.read_text(encoding="utf-8"))
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
