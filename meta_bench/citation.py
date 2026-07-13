"""Citation-dimension scores for the clean MetaBench evaluator."""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Iterable, Literal, Mapping, Protocol, Sequence, cast

from .structure import count_length_units, parse_markdown_sections


CHUNK_TYPES = [
    "introduction",
    "taxonomy",
    "evidence_synthesis",
    "limitations_gaps",
    "conclusion",
    "methods",
    "results",
    "discussion",
    "other",
]

LOW_CONFIDENCE_MARGIN = 0.15
BODY_SIGNAL_WORD_LIMIT = 220
BODY_SIGNAL_SENTENCE_LIMIT = 3
MIN_CITATIONS_FOR_DISTRIBUTION = 5
MAX_CITATION_QUALITY_CHUNK_CHARS = 1200
MAX_CITATION_QUALITY_CHUNKS_PER_SOURCE = 3
MIN_CITATION_QUALITY_SUBCLAIM_WORDS = 6
MIN_CITATION_QUALITY_ARTICLE_CLAIM_WORDS = 8
CITATION_COUNT_THRESHOLDS = {
    # Calibrated from medical_reviews_300_ready/article_summary.csv using
    # whole-body citation density: body citation events per 1,000 body words.
    # Full-score band uses the central 70% (p15-p85); the linear penalty
    # band uses the wider p5-p95 interval.
    "soft_lower": 21.780305082175012,
    "soft_upper": 53.88519249312971,
    "hard_lower": 15.047717849932427,
    "hard_upper": 71.44435519935337,
}
SEVEN_SECTION_KEYS = [
    "title",
    "abstract",
    "keywords",
    "introduction",
    "main_body",
    "conclusion",
    "references",
]

SECTION_DISTRIBUTION_THRESHOLDS: dict[str, dict[str, float]] = {
    "title": {"soft_lower": 0.0, "soft_upper": 0.005, "hard_lower": 0.0, "hard_upper": 0.02, "weight": 0.05},
    "abstract": {"soft_lower": 0.0, "soft_upper": 0.30, "hard_lower": 0.0, "hard_upper": 0.45, "weight": 0.10},
    "keywords": {"soft_lower": 0.0, "soft_upper": 0.002, "hard_lower": 0.0, "hard_upper": 0.01, "weight": 0.05},
    "introduction": {"soft_lower": 0.03, "soft_upper": 0.30, "hard_lower": 0.0, "hard_upper": 0.40, "weight": 0.20},
    "main_body": {"soft_lower": 0.45, "soft_upper": 0.90, "hard_lower": 0.30, "hard_upper": 0.97, "weight": 0.45},
    "conclusion": {"soft_lower": 0.0, "soft_upper": 0.05, "hard_lower": 0.0, "hard_upper": 0.12, "weight": 0.10},
    "references": {"soft_lower": 0.0, "soft_upper": 0.0, "hard_lower": 0.0, "hard_upper": 0.01, "weight": 0.05},
}

REFERENCE_HEADINGS = {
    "reference",
    "references",
    "bibliography",
    "literature cited",
    "works cited",
}
CONCLUSION_HEADINGS = {
    "conclusion",
    "conclusion and implication",
    "conclusion and implications",
    "conclusions",
    "summary",
    "key messages",
    "implications",
    "interpretation",
    "conclusions and implications",
    "clinical implications",
    "future directions",
    "future direction",
    "concluding remarks",
    "concluding remark",
    "concluding remarks and key take aways",
    "concluding remarks and key takeaways",
    "concluding remarks and key points",
    "concluding remarks and future directions",
    "key take aways",
    "key takeaways",
    "perspective",
    "perspectives",
}
TAIL_CONCLUSION_SOURCE_HEADINGS = {"discussion", "results", "main results"}
INTRODUCTION_HEADINGS = {"introduction"}
NON_MAIN_HEADINGS = {
    *REFERENCE_HEADINGS,
    *CONCLUSION_HEADINGS,
    "abstract",
    "keywords",
    "keyword",
    "acknowledgement",
    "acknowledgements",
    "acknowledgment",
    "acknowledgments",
    "funding",
    "funding statement",
    "author contributions",
    "authors' contributions",
    "data availability",
    "data availability statement",
    "conflict of interest",
    "conflicts of interest",
    "ethics statement",
    "supplementary material",
    "publisher's note",
    "associated data",
    "generative ai statement",
}
CITATION_NON_CONTENT_HEADINGS = NON_MAIN_HEADINGS - CONCLUSION_HEADINGS

BRACKET_CITATION_RE = re.compile(r"\[(.*?)\]")
PAREN_CITATION_RE = re.compile(r"\((.*?)\)")
NUMERIC_TOKEN_RE = re.compile(r"\d+(?:\s*[-–—]\s*\d+)?")
AUTHOR_YEAR_CITATION_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z'`\-]+(?:\s+[A-Z][A-Za-z'`\-]+)*|"
    r"[A-Z][A-Za-z'`\-]+\s+et\s+al\.)\s*,?\s*(?:\(|\[)?(19|20)\d{2}(?:\)|\])?"
)
MULTI_AUTHOR_YEAR_RE = re.compile(
    r"\b[A-Z][A-Za-z'`\-]+\s+et\s+al\.\s*,?\s*(19|20)\d{2}"
)
PLAIN_NUMERIC_CITATION_RE = re.compile(
    r"(?:(?<=\s)|(?<=,)|(?<=;)|(?<=\()|(?<=\.)|^)(\d{1,3})(?=\s*(?:,|\.|;|\)|$))"
)
YEAR_PREFIXED_CITATION_RE = re.compile(r"\b(?:19|20)\d{2}\s*,\s*(\d{1,3})(?=\b)")

TITLE_TERMS: dict[str, tuple[tuple[str, float], ...]] = {
    "introduction": (
        ("scope", 1.0),
        ("intro", 1.0),
        ("introduction", 1.0),
        ("background", 1.0),
        ("context", 1.0),
        ("terminology", 1.0),
        ("definition", 1.0),
    ),
    "taxonomy": (
        ("framework", 1.0),
        ("taxonomy", 1.0),
        ("classification", 1.0),
        ("classifications", 1.0),
        ("organizing", 1.0),
        ("category", 1.0),
        ("categories", 1.0),
        ("subtype", 1.0),
        ("phenotype", 1.0),
    ),
    "evidence_synthesis": (
        ("evidence", 1.0),
        ("synthesis", 1.0),
        ("findings", 1.0),
        ("studies", 1.0),
        ("literature", 1.0),
        ("meta-analysis", 1.0),
    ),
    "limitations_gaps": (
        ("limitation", 1.2),
        ("limitations", 1.2),
        ("bias", 1.2),
        ("uncertainty", 1.2),
        ("constraint", 1.2),
        ("constraints", 1.2),
        ("gap", 1.2),
        ("gaps", 1.2),
        ("future", 1.2),
        ("research agenda", 1.2),
        ("open question", 1.2),
        ("open questions", 1.2),
    ),
    "conclusion": (
        ("conclusion", 1.0),
        ("conclusions", 1.0),
        ("summary", 1.0),
        ("key messages", 1.0),
        ("takeaways", 1.0),
        ("concluding", 1.0),
    ),
    "methods": (
        ("method", 1.0),
        ("methods", 1.0),
        ("protocol", 1.0),
        ("design", 1.0),
        ("materials", 1.0),
    ),
    "results": (
        ("result", 1.0),
        ("results", 1.0),
        ("outcomes", 1.0),
        ("findings", 0.8),
    ),
    "discussion": (
        ("discussion", 1.0),
        ("interpretation", 1.0),
        ("implications", 0.8),
        ("clinical implications", 1.0),
    ),
}

BODY_TERMS: dict[str, tuple[tuple[str, float], ...]] = {
    "introduction": (
        ("define", 1.0),
        ("defined", 1.0),
        ("definition", 1.0),
        ("scope", 1.0),
        ("terminology", 1.0),
        ("context", 1.0),
        ("overview", 1.0),
        ("background", 1.0),
        ("aim", 0.8),
        ("objective", 0.8),
    ),
    "taxonomy": (
        ("classify", 1.0),
        ("classification", 1.0),
        ("category", 1.0),
        ("categories", 1.0),
        ("framework", 1.0),
        ("phenotype", 1.0),
        ("subtype", 1.0),
        ("subtypes", 1.0),
        ("typology", 1.0),
        ("cluster", 0.8),
    ),
    "evidence_synthesis": (
        ("trial", 1.0),
        ("trials", 1.0),
        ("cohort", 1.0),
        ("evidence", 1.0),
        ("finding", 1.0),
        ("findings", 1.0),
        ("compare", 1.0),
        ("comparison", 1.0),
        ("synthesize", 1.0),
        ("synthesis", 1.0),
        ("studies", 1.0),
        ("meta-analysis", 1.0),
    ),
    "limitations_gaps": (
        ("limitation", 1.2),
        ("limitations", 1.2),
        ("bias", 1.2),
        ("uncertainty", 1.2),
        ("heterogeneity", 1.2),
        ("evidence gap", 1.2),
        ("evidence gaps", 1.2),
        ("gap", 1.2),
        ("gaps", 1.2),
        ("future studies", 1.2),
        ("future work", 1.2),
        ("future research", 1.2),
        ("research agenda", 1.2),
        ("open question", 1.2),
        ("open questions", 1.2),
    ),
    "conclusion": (
        ("conclusion", 1.0),
        ("conclusions", 1.0),
        ("in summary", 1.0),
        ("overall", 1.0),
        ("taken together", 1.0),
        ("key message", 1.0),
        ("key messages", 1.0),
        ("implication", 0.8),
        ("implications", 0.8),
    ),
    "methods": (
        ("method", 1.0),
        ("methods", 1.0),
        ("design", 1.0),
        ("protocol", 1.0),
        ("dataset", 1.0),
        ("criteria", 1.0),
        ("inclusion", 1.0),
        ("statistical", 1.0),
        ("analysis", 0.8),
    ),
    "results": (
        ("result", 1.0),
        ("results", 1.0),
        ("outcome", 1.0),
        ("outcomes", 1.0),
        ("table", 0.8),
        ("figure", 0.8),
        ("estimate", 0.8),
        ("significant", 0.8),
        ("confidence interval", 1.0),
    ),
    "discussion": (
        ("interpret", 1.0),
        ("interpretation", 1.0),
        ("discuss", 1.0),
        ("discussion", 1.0),
        ("implication", 0.8),
        ("implications", 0.8),
        ("clinical implications", 1.0),
    ),
}

CLAIM_ROLE_STRENGTH = {
    "background": 0,
    "definition": 0,
    "gap": 1,
    "mechanism": 1,
    "interpretation": 1,
    "comparison": 2,
    "prior_result": 2,
    "evidence_synthesis": 2,
    "effect": 3,
    "clinical_effect": 3,
    "causal": 3,
    "recommendation": 4,
    "clinical_recommendation": 4,
    "practice": 4,
    "universal_claim": 5,
    "definitive": 5,
}

SOURCE_TYPE_STRENGTH = {
    "expert": 0,
    "opinion": 0,
    "narrative": 0,
    "mechanistic": 1,
    "preclinical": 1,
    "animal": 1,
    "in_vitro": 1,
    "observational": 2,
    "retrospective": 2,
    "case_control": 2,
    "cohort": 2,
    "trial": 3,
    "rct": 3,
    "prospective": 3,
    "benchmark": 3,
    "systematic_review": 4,
    "meta_analysis": 4,
    "guideline": 4,
    "consensus": 4,
}

CLAIM_TEXT_CUES: tuple[tuple[str, int], ...] = (
    ("always", 5),
    ("proves", 5),
    ("definitively", 5),
    ("established that", 5),
    ("must", 4),
    ("should", 4),
    ("recommend", 4),
    ("clinical practice", 4),
    ("causes", 3),
    ("causal", 3),
    ("leads to", 3),
    ("improves", 3),
    ("reduces", 3),
    ("effective", 3),
    ("efficacy", 3),
    ("associated with", 2),
    ("compared with", 2),
    ("higher", 2),
    ("lower", 2),
    ("suggests", 1),
    ("may", 1),
    ("possible", 1),
    ("uncertain", 1),
    ("gap", 1),
)

EVIDENCE_TEXT_CUES: tuple[tuple[str, int], ...] = (
    ("meta-analysis", 4),
    ("meta analysis", 4),
    ("systematic review", 4),
    ("guideline", 4),
    ("consensus", 4),
    ("randomized", 3),
    ("randomised", 3),
    ("rct", 3),
    ("trial", 3),
    ("prospective", 3),
    ("cohort", 2),
    ("observational", 2),
    ("retrospective", 2),
    ("case-control", 2),
    ("case control", 2),
    ("mechanistic", 1),
    ("animal", 1),
    ("in vitro", 1),
    ("expert opinion", 0),
    ("narrative", 0),
)


@dataclass(frozen=True)
class CitationChunk:
    """A citation-evaluation chunk parsed from final text or reference data."""

    chunk_id: str
    index: int
    title: str
    body: str
    word_count: int


@dataclass(frozen=True)
class ChunkRoleClassification:
    """Deterministic role classification for one citation chunk."""

    chunk_id: str
    title: str
    canonical_type: str
    confidence: float
    low_confidence: bool
    matched_signals: dict[str, list[str]]
    scores_by_type: dict[str, float]
    component_scores: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "title": self.title,
            "canonical_type": self.canonical_type,
            "confidence": self.confidence,
            "low_confidence": self.low_confidence,
            "matched_signals": self.matched_signals,
            "scores_by_type": self.scores_by_type,
            "component_scores": self.component_scores,
        }


@dataclass(frozen=True)
class CitationEvent:
    """Normalized citation event for source-fidelity scoring."""

    citation_id: str
    chunk_id: str
    source_id: str
    source_type: str
    claim_role: str
    claim_span: str
    source_excerpt: str
    claim_strength: int | None = None
    evidence_strength: int | None = None
    fidelity_penalty: float | None = None


@dataclass(frozen=True)
class SourceFidelityEventDecision:
    """Source-fidelity decision for one citation event."""

    citation_id: str
    chunk_id: str
    source_id: str
    chunk_type: str
    claim_role: str
    source_type: str
    claim_strength: int
    evidence_strength: int
    strength_delta: int
    penalty: float
    score: float
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return {
            "citation_id": self.citation_id,
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "chunk_type": self.chunk_type,
            "claim_role": self.claim_role,
            "source_type": self.source_type,
            "claim_strength": self.claim_strength,
            "evidence_strength": self.evidence_strength,
            "strength_delta": self.strength_delta,
            "penalty": self.penalty,
            "score": self.score,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class SourceFidelityResult:
    """Aggregated source-fidelity score and diagnostics."""

    score: float
    penalty: float
    citation_count: int
    overclaim_count: int
    severe_overclaim_count: int
    chunk_classifications: list[ChunkRoleClassification]
    event_decisions: list[SourceFidelityEventDecision]
    chunk_scores: dict[str, dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "source_fidelity": self.score,
            "source_fidelity_penalty": self.penalty,
            "citation_count": self.citation_count,
            "overclaim_count": self.overclaim_count,
            "severe_overclaim_count": self.severe_overclaim_count,
            "chunk_classifications": [
                classification.to_dict()
                for classification in self.chunk_classifications
            ],
            "event_decisions": [
                decision.to_dict() for decision in self.event_decisions
            ],
            "chunk_scores": self.chunk_scores,
        }


@dataclass(frozen=True)
class CitationQualityDecision:
    """Citation contribution decision for one citation event."""

    citation_id: str
    claim_key: str
    source_id: str
    supported: bool
    necessary: bool
    rationale: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "citation_id": self.citation_id,
            "claim_key": self.claim_key,
            "source_id": self.source_id,
            "supported": self.supported,
            "necessary": self.necessary,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class ClaimCitationQualityDecision:
    """LiRA-style claim-level citation-quality decision."""

    claim_key: str
    claim: str
    citation_ids: list[str]
    source_ids: list[str]
    supported: bool
    necessary_citation_ids: list[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_key": self.claim_key,
            "claim": self.claim,
            "citation_ids": self.citation_ids,
            "source_ids": self.source_ids,
            "supported": self.supported,
            "supporting_citation_ids": self.necessary_citation_ids,
            "necessary_citation_ids": self.necessary_citation_ids,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class CitationQualityF1Result:
    """LiRA citation quality F1 with recall scaling."""

    score: float
    precision: float
    recall: float
    scaled_recall: float
    scaling: float
    scaling_claim_count: float
    claim_count: int
    citation_count: int
    supported_claim_count: int
    necessary_citation_count: int
    excluded_claim_count: int
    excluded_citation_count: int
    excluded_claims: list[dict[str, object]]
    decisions: list[CitationQualityDecision]
    claim_decisions: list[ClaimCitationQualityDecision] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "citation_quality_f1": self.score,
            "precision": self.precision,
            "recall": self.recall,
            "scaled_recall": self.scaled_recall,
            "scaling": self.scaling,
            "scaling_claim_count": self.scaling_claim_count,
            "claim_count": self.claim_count,
            "citation_count": self.citation_count,
            "supported_claim_count": self.supported_claim_count,
            "supporting_citation_count": self.necessary_citation_count,
            "necessary_citation_count": self.necessary_citation_count,
            "excluded_claim_count": self.excluded_claim_count,
            "excluded_citation_count": self.excluded_citation_count,
            "excluded_claims": self.excluded_claims,
            "claim_decisions": [
                decision.to_dict() for decision in self.claim_decisions
            ],
            "decisions": [decision.to_dict() for decision in self.decisions],
            "formula": "2 * precision * scaled_recall / (precision + scaled_recall)",
            "scaling_formula": "1 - exp(-0.01 * scaling_claim_count)",
        }


class CitationQualityJudge(Protocol):
    """Judge interface for LiRA-style claim-citation alignment."""

    def judge_claim_citation_quality(
        self,
        *,
        claim: str,
        sources: Sequence[Mapping[str, str]],
    ) -> ClaimCitationQualityDecision:
        """Return whether a cited reference set supports one claim."""


@dataclass(frozen=True)
class SevenSection:
    """One section in the seven-section citation-distribution layout."""

    section_key: str
    text: str
    word_count: int
    text_present: bool


@dataclass(frozen=True)
class CitationCountResult:
    """Deterministic citation-count result."""

    total_citations: int
    reference_counts: dict[str, int]


@dataclass(frozen=True)
class CitationCountScoreResult:
    """Citation-density score and diagnostics."""

    score: float
    penalty: float
    word_count_without_references: int
    citation_count_without_references: int
    citations_per_1000_words: float
    thresholds: dict[str, float]
    source: str
    section_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "citation_count": self.score,
            "citation_count_penalty": self.penalty,
            "word_count_without_references": self.word_count_without_references,
            "citation_count_without_references": self.citation_count_without_references,
            "citations_per_1000_words": self.citations_per_1000_words,
            "thresholds": self.thresholds,
            "source": self.source,
            "section_counts": self.section_counts,
        }


@dataclass(frozen=True)
class SourceBalanceResult:
    """Single-source concentration score and diagnostics."""

    score: float
    penalty: float
    total_citations: int
    max_single_source_count: int
    max_single_source_share: float
    dominant_source_id: str
    concentration_hhi: float
    cited_source_count: int
    method: str
    source: str
    source_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "source_balance": self.score,
            "source_balance_penalty": self.penalty,
            "total_citations": self.total_citations,
            "max_single_source_count": self.max_single_source_count,
            "max_single_source_share": self.max_single_source_share,
            "dominant_source_id": self.dominant_source_id,
            "concentration_hhi": self.concentration_hhi,
            "cited_source_count": self.cited_source_count,
            "method": self.method,
            "source": self.source,
            "source_counts": self.source_counts,
        }


@dataclass(frozen=True)
class SectionDistributionResult:
    """Citation distribution score and diagnostics."""

    score: float | None
    penalty: float | None
    status: str
    total_citations: int
    section_counts: dict[str, int]
    section_shares: dict[str, float]
    section_penalties: dict[str, float]
    thresholds: dict[str, dict[str, float]]
    source: str
    scheme: str = "seven_section"
    reason: str = ""
    prior_metadata: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        output: dict[str, object] = {
            "section_distribution": self.score,
            "section_distribution_penalty": self.penalty,
            "status": self.status,
            "total_citations": self.total_citations,
            "section_counts": self.section_counts,
            "section_shares": self.section_shares,
            "section_penalties": self.section_penalties,
            "thresholds": self.thresholds,
            "source": self.source,
            "scheme": self.scheme,
        }
        if self.reason:
            output["reason"] = self.reason
        if self.prior_metadata:
            output["prior_metadata"] = self.prior_metadata
        return output


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).casefold().strip()


def _normalize_token(value: object) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").casefold()).strip("_")
    return normalized or "unknown"


def _contains_term(text: str, term: str) -> bool:
    normalized_text = _normalize_text(text)
    normalized_term = _normalize_text(term)
    if not normalized_term:
        return False
    if " " in normalized_term or "-" in normalized_term:
        return normalized_term in normalized_text
    return re.search(rf"\b{re.escape(normalized_term)}\b", normalized_text) is not None


def normalize_section_heading(heading: str) -> str:
    """Normalize a heading for seven-section citation distribution."""

    value = re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", heading.strip())
    return re.sub(r"\s+", " ", value).casefold()


def _join_text(existing: str, new_text: str) -> str:
    if not new_text.strip():
        return existing
    if not existing.strip():
        return new_text.strip()
    return f"{existing.strip()}\n\n{new_text.strip()}"


def _paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def _split_first_paragraph(text: str) -> tuple[str, str]:
    paragraphs = _paragraphs(text)
    if len(paragraphs) <= 1:
        return text.strip(), ""
    return paragraphs[0], "\n\n".join(paragraphs[1:])


def _split_last_paragraph(text: str) -> tuple[str, str]:
    paragraphs = _paragraphs(text)
    if len(paragraphs) <= 1:
        return text.strip(), ""
    return "\n\n".join(paragraphs[:-1]), paragraphs[-1]


def split_into_seven_sections(final_text: str) -> dict[str, SevenSection]:
    """Split generated text into title/abstract/keywords/intro/body/conclusion/references."""

    raw_sections = parse_markdown_sections(final_text)
    texts = {section_key: "" for section_key in SEVEN_SECTION_KEYS}
    content_sections: list[tuple[str, str, str]] = []

    for section in raw_sections:
        heading = section.heading.strip()
        body = section.body.strip()
        normalized_heading = normalize_section_heading(heading)
        if section.is_title and not texts["title"]:
            texts["title"] = heading
            continue
        if heading and not texts["title"]:
            texts["title"] = heading
            continue

        if normalized_heading == "abstract":
            texts["abstract"] = _join_text(texts["abstract"], body)
        elif normalized_heading in {"keywords", "keyword"}:
            texts["keywords"] = _join_text(texts["keywords"], body)
        elif normalized_heading in INTRODUCTION_HEADINGS:
            introduction_text, overflow_text = _split_first_paragraph(body)
            texts["introduction"] = _join_text(texts["introduction"], introduction_text)
            texts["main_body"] = _join_text(texts["main_body"], overflow_text)
            content_sections.append(("introduction", normalized_heading, body))
        elif normalized_heading in CONCLUSION_HEADINGS:
            texts["conclusion"] = _join_text(texts["conclusion"], body)
            content_sections.append(("conclusion", normalized_heading, body))
        elif normalized_heading in REFERENCE_HEADINGS:
            texts["references"] = _join_text(texts["references"], body)
        elif normalized_heading and normalized_heading not in NON_MAIN_HEADINGS:
            texts["main_body"] = _join_text(texts["main_body"], body)
            content_sections.append(("main_body", normalized_heading, body))

    if not raw_sections:
        texts["main_body"] = final_text.strip()

    if not texts["introduction"].strip() and texts["main_body"].strip():
        introduction_text, overflow_text = _split_first_paragraph(texts["main_body"])
        texts["introduction"] = introduction_text
        texts["main_body"] = overflow_text

    if not texts["conclusion"].strip():
        for _section_key, normalized_heading, body in reversed(content_sections):
            if normalized_heading in TAIL_CONCLUSION_SOURCE_HEADINGS and body.strip():
                paragraphs = _paragraphs(body)
                if paragraphs:
                    texts["conclusion"] = paragraphs[-1]
                    break

    if not texts["conclusion"].strip() and texts["main_body"].strip():
        main_text, conclusion_text = _split_last_paragraph(texts["main_body"])
        texts["main_body"] = main_text
        texts["conclusion"] = conclusion_text

    return {
        section_key: SevenSection(
            section_key=section_key,
            text=texts[section_key],
            word_count=count_length_units(texts[section_key]),
            text_present=bool(texts[section_key].strip()),
        )
        for section_key in SEVEN_SECTION_KEYS
    }


def expand_numeric_citation_tokens(inner_text: str) -> list[int]:
    """Expand numeric citation tokens such as 3, 5-7."""

    if not re.fullmatch(r"[\d\s,;\-–—]+", inner_text.strip()):
        return []
    results: list[int] = []
    for token in NUMERIC_TOKEN_RE.findall(inner_text):
        compact = re.sub(r"\s+", "", token)
        if any(marker in compact for marker in ("-", "–", "—")):
            parts = re.split(r"[-–—]", compact)
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                start = int(parts[0])
                end = int(parts[1])
                if start <= end and end - start <= 500:
                    results.extend(range(start, end + 1))
        elif compact.isdigit():
            results.append(int(compact))
    return results


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _count_author_year_citations(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    protected_spans: set[tuple[int, int]] = set()
    for match in MULTI_AUTHOR_YEAR_RE.finditer(text):
        protected_spans.add(match.span())
        counts[f"ay:{_normalize_space(match.group(0).casefold())}"] += 1
    for match in AUTHOR_YEAR_CITATION_RE.finditer(text):
        if match.span() in protected_spans:
            continue
        snippet = match.group(0)
        lowered = snippet.casefold()
        if "organisation" in lowered and not lowered.startswith("world health organisation"):
            continue
        counts[f"ay:{_normalize_space(lowered)}"] += 1
    return counts


def _count_plain_numeric_citations(
    text: str,
    max_reference_number: int | None,
) -> Counter[str]:
    if not max_reference_number or max_reference_number <= 0:
        return Counter()
    counts: Counter[str] = Counter()
    seen_spans: set[tuple[int, int]] = set()
    for match in YEAR_PREFIXED_CITATION_RE.finditer(text):
        number = int(match.group(1))
        if 1 <= number <= max_reference_number:
            counts[f"plain:{number}"] += 1
            seen_spans.add(match.span(1))
    for match in PLAIN_NUMERIC_CITATION_RE.finditer(text):
        if match.span(1) in seen_spans:
            continue
        number = int(match.group(1))
        if 1 <= number <= max_reference_number:
            counts[f"plain:{number}"] += 1
    return counts


def count_reference_items(section_text: str) -> int:
    """Count reference-list items in a references section."""

    if not section_text.strip():
        return 0
    paragraphs = _paragraphs(section_text)
    if paragraphs:
        return len(paragraphs)
    return len([line.strip() for line in section_text.splitlines() if line.strip()])


def count_citations_in_text(
    text: str,
    max_reference_number: int | None = None,
) -> CitationCountResult:
    """Count in-text citation events using deterministic regex rules."""

    counts: Counter[str] = Counter()
    for match in BRACKET_CITATION_RE.finditer(text):
        for ref in expand_numeric_citation_tokens(match.group(1)):
            counts[f"num:{ref}"] += 1
    for match in PAREN_CITATION_RE.finditer(text):
        inner = match.group(1)
        if any(
            marker in inner.casefold()
            for marker in ["ci", "%", "p <", "p>", "or=", "smd", "auc", "i 2"]
        ):
            continue
        for ref in expand_numeric_citation_tokens(inner):
            counts[f"num:{ref}"] += 1

    counts.update(_count_author_year_citations(text))
    counts.update(_count_plain_numeric_citations(text, max_reference_number))
    return CitationCountResult(
        total_citations=sum(counts.values()),
        reference_counts=dict(counts),
    )


def _preview_body(body: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", body.strip())
    sentence_preview = " ".join(
        sentence for sentence in sentences[:BODY_SIGNAL_SENTENCE_LIMIT] if sentence
    )
    words = re.findall(r"[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)*", body)
    word_preview = " ".join(words[:BODY_SIGNAL_WORD_LIMIT])
    if sentence_preview and count_length_units(sentence_preview) <= BODY_SIGNAL_WORD_LIMIT:
        return sentence_preview
    return word_preview


def _term_scores(
    text: str,
    term_table: Mapping[str, tuple[tuple[str, float], ...]],
) -> tuple[dict[str, float], dict[str, list[str]]]:
    scores = {chunk_type: 0.0 for chunk_type in CHUNK_TYPES}
    matched = {chunk_type: [] for chunk_type in CHUNK_TYPES}
    for chunk_type, terms in term_table.items():
        for term, weight in terms:
            if _contains_term(text, term):
                scores[chunk_type] += weight
                matched[chunk_type].append(term)
    return scores, matched


def _position_signal(index: int, total: int) -> tuple[dict[str, float], list[str]]:
    scores = {chunk_type: 0.0 for chunk_type in CHUNK_TYPES}
    signals: list[str] = []
    if total <= 0:
        return scores, signals

    if index == 1:
        scores["introduction"] += 1.0
        signals.append("first_chunk:introduction")
    if index == 2 and total >= 4:
        scores["taxonomy"] += 1.0
        signals.append("second_chunk:taxonomy")
    if 1 < index < total:
        if not (index == 2 and total >= 4) and not (index == total - 1):
            scores["evidence_synthesis"] += 1.0
            signals.append("middle_chunk:evidence_synthesis")
    if total >= 3 and index == total - 1:
        scores["limitations_gaps"] += 1.0
        signals.append("penultimate_chunk:limitations_gaps")
    if index == total:
        scores["conclusion"] += 1.0
        scores["limitations_gaps"] += 0.7
        signals.append("last_chunk:conclusion")
        signals.append("last_chunk:limitations_gaps")
    return scores, signals


def _keyword_type(keyword: str) -> str | None:
    keyword_text = _normalize_text(keyword)
    for chunk_type, terms in TITLE_TERMS.items():
        if any(_normalize_text(term) in keyword_text for term, _ in terms):
            return chunk_type
    for chunk_type, terms in BODY_TERMS.items():
        if any(_normalize_text(term) in keyword_text for term, _ in terms):
            return chunk_type
    return None


def _parse_index_range(raw_value: object) -> set[int]:
    if isinstance(raw_value, int):
        return {raw_value}
    if isinstance(raw_value, float):
        return {int(raw_value)}
    if isinstance(raw_value, (list, tuple, set)):
        indexes: set[int] = set()
        for item in raw_value:
            indexes.update(_parse_index_range(item))
        return indexes
    text = str(raw_value or "").strip()
    if not text:
        return set()
    match = re.fullmatch(r"(\d+)\s*[-:]\s*(\d+)", text)
    if match:
        start = int(match.group(1))
        end = int(match.group(2))
        if start > end:
            start, end = end, start
        return set(range(start, end + 1))
    numbers = [int(item) for item in re.findall(r"\d+", text)]
    return set(numbers)


def _iter_range_keyword_specs(reference: Mapping[str, object]) -> Iterable[tuple[str, set[int]]]:
    constraints = reference.get("constraints")
    if not isinstance(constraints, Mapping):
        return []
    raw_range_keywords = constraints.get("range_keywords")
    if raw_range_keywords is None:
        return []

    specs: list[tuple[str, set[int]]] = []
    if isinstance(raw_range_keywords, Mapping):
        for keyword, range_value in raw_range_keywords.items():
            specs.append((str(keyword), _parse_index_range(range_value)))
        return specs

    if isinstance(raw_range_keywords, list):
        for item in raw_range_keywords:
            if isinstance(item, Mapping):
                keyword = str(
                    item.get("keyword", item.get("term", item.get("phrase", "")))
                ).strip()
                raw_range = item.get("range", item.get("sections", item.get("positions")))
                if not keyword:
                    continue
                specs.append((keyword, _parse_index_range(raw_range)))
        return specs

    return specs


def _benchmark_hint_signal(
    chunk: CitationChunk,
    reference: Mapping[str, object] | None,
) -> tuple[dict[str, float], list[str]]:
    scores = {chunk_type: 0.0 for chunk_type in CHUNK_TYPES}
    signals: list[str] = []
    if reference is None:
        return scores, signals

    chunk_text = f"{chunk.title}\n{_preview_body(chunk.body)}"
    for keyword, indexes in _iter_range_keyword_specs(reference):
        if chunk.index not in indexes:
            continue
        chunk_type = _keyword_type(keyword)
        if chunk_type is None:
            continue
        if _contains_term(chunk_text, keyword):
            scores[chunk_type] += 1.0
            signals.append(f"range_keyword:{keyword}->{chunk_type}")

    outline = reference.get("outline")
    if isinstance(outline, Mapping):
        outline_text = outline.get(chunk.chunk_id)
        if outline_text is None:
            outline_text = outline.get(f"sec{chunk.index}")
        if outline_text is not None:
            outline_scores, outline_matches = _term_scores(str(outline_text), TITLE_TERMS)
            for chunk_type, score in outline_scores.items():
                if score <= 0:
                    continue
                scores[chunk_type] += min(1.0, score)
                for term in outline_matches.get(chunk_type, []):
                    signals.append(f"outline:{term}->{chunk_type}")

    return scores, signals


def _combine_component_scores(
    title_scores: Mapping[str, float],
    position_scores: Mapping[str, float],
    body_scores: Mapping[str, float],
    benchmark_scores: Mapping[str, float],
) -> dict[str, float]:
    return {
        chunk_type: (
            0.35 * title_scores.get(chunk_type, 0.0)
            + 0.25 * position_scores.get(chunk_type, 0.0)
            + 0.25 * body_scores.get(chunk_type, 0.0)
            + 0.15 * benchmark_scores.get(chunk_type, 0.0)
        )
        for chunk_type in CHUNK_TYPES
    }


def classify_chunk_role(
    chunk: CitationChunk,
    *,
    total_chunks: int,
    reference: Mapping[str, object] | None = None,
) -> ChunkRoleClassification:
    """Classify one citation chunk with deterministic multi-signal scoring."""

    title_scores, title_matches = _term_scores(chunk.title, TITLE_TERMS)
    position_scores, position_matches = _position_signal(chunk.index, total_chunks)
    body_scores, body_matches = _term_scores(_preview_body(chunk.body), BODY_TERMS)
    benchmark_scores, benchmark_matches = _benchmark_hint_signal(chunk, reference)
    scores_by_type = _combine_component_scores(
        title_scores,
        position_scores,
        body_scores,
        benchmark_scores,
    )

    ranked = sorted(
        scores_by_type.items(),
        key=lambda item: (item[1], item[0] != "other"),
        reverse=True,
    )
    top_type, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    total_score = sum(score for score in scores_by_type.values() if score > 0)
    if top_score <= 0:
        top_type = "other"
        confidence = 0.0
        low_confidence = True
    else:
        confidence = top_score / total_score if total_score > 0 else 0.0
        low_confidence = (top_score - second_score) < LOW_CONFIDENCE_MARGIN

    matched_signals = {
        "title": [
            f"{term}->{chunk_type}"
            for chunk_type, terms in title_matches.items()
            for term in terms
        ],
        "position": position_matches,
        "body": [
            f"{term}->{chunk_type}"
            for chunk_type, terms in body_matches.items()
            for term in terms
        ],
        "benchmark_hint": benchmark_matches,
    }

    component_scores = {
        "title": dict(title_scores),
        "position": dict(position_scores),
        "body": dict(body_scores),
        "benchmark_hint": dict(benchmark_scores),
    }
    return ChunkRoleClassification(
        chunk_id=chunk.chunk_id,
        title=chunk.title,
        canonical_type=top_type,
        confidence=round(confidence, 4),
        low_confidence=low_confidence,
        matched_signals=matched_signals,
        scores_by_type={key: round(value, 4) for key, value in scores_by_type.items()},
        component_scores={
            component: {
                key: round(value, 4) for key, value in scores.items()
            }
            for component, scores in component_scores.items()
        },
    )


def classify_chunk_roles(
    chunks: Sequence[CitationChunk],
    reference: Mapping[str, object] | None = None,
) -> list[ChunkRoleClassification]:
    """Classify all citation chunks."""

    total_chunks = len(chunks)
    return [
        classify_chunk_role(chunk, total_chunks=total_chunks, reference=reference)
        for chunk in chunks
    ]


def parse_citation_chunks(
    final_text: str,
    raw_chunks: object | None = None,
) -> list[CitationChunk]:
    """Parse citation chunks from reference data or Markdown final text."""

    if raw_chunks is not None:
        chunks = _normalize_raw_chunks(raw_chunks)
        if chunks:
            return chunks

    markdown_sections = parse_markdown_sections(final_text)
    chunks: list[CitationChunk] = []
    for section in markdown_sections:
        if section.is_title:
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
    if chunks:
        return chunks

    return [
        CitationChunk(
            chunk_id="sec1",
            index=1,
            title="",
            body=final_text,
            word_count=count_length_units(final_text),
        )
    ]


def _normalize_raw_chunks(raw_chunks: object) -> list[CitationChunk]:
    if not isinstance(raw_chunks, list):
        raise TypeError("chunk_map must be a list")
    chunks: list[CitationChunk] = []
    for index, raw_item in enumerate(raw_chunks, start=1):
        if not isinstance(raw_item, Mapping):
            raise TypeError("each chunk_map item must be a mapping")
        item = cast(Mapping[str, object], raw_item)
        chunk_id = str(
            item.get("chunk_id", item.get("section_id", f"sec{index}"))
        ).strip() or f"sec{index}"
        title = str(
            item.get(
                "title",
                item.get("heading", item.get("section_type", item.get("type", ""))),
            )
        ).strip()
        body = str(item.get("text", item.get("body", ""))).strip()
        raw_word_count = item.get("word_count", item.get("words"))
        if isinstance(raw_word_count, (int, float, str)) and str(raw_word_count).strip():
            word_count = max(0, int(float(raw_word_count)))
        else:
            word_count = count_length_units(body)
        chunks.append(
            CitationChunk(
                chunk_id=chunk_id,
                index=index,
                title=title,
                body=body,
                word_count=word_count,
            )
        )
    return chunks


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_citation_events(raw_citations: Iterable[object]) -> list[CitationEvent]:
    """Normalize raw citation-manifest dictionaries."""

    citations: list[CitationEvent] = []
    for index, raw_item in enumerate(raw_citations, start=1):
        if not isinstance(raw_item, Mapping):
            raise TypeError("each citation event must be a mapping")
        item = cast(Mapping[str, object], raw_item)
        chunk_id = str(
            item.get("chunk_id", item.get("section_id", item.get("section", "sec1")))
        ).strip() or "sec1"
        citations.append(
            CitationEvent(
                citation_id=str(item.get("citation_id", f"C{index:03d}")),
                chunk_id=chunk_id,
                source_id=str(item.get("source_id", "unknown")),
                source_type=_normalize_token(item.get("source_type", "unknown")),
                claim_role=_normalize_token(item.get("claim_role", "unknown")),
                claim_span=str(item.get("claim_span", "")),
                source_excerpt=str(item.get("source_excerpt", "")),
                claim_strength=_optional_int(item.get("claim_strength")),
                evidence_strength=_optional_int(item.get("evidence_strength")),
                fidelity_penalty=_optional_float(
                    item.get("fidelity_penalty", item.get("overclaim_penalty"))
                ),
            )
        )
    return citations


class LLMCitationQualityJudge:
    """API-agnostic LiRA-style citation-quality judge wrapper."""

    def __init__(self, complete: Callable[[str], str]):
        self._complete = complete

    def judge_claim_citation_quality(
        self,
        *,
        claim: str,
        sources: Sequence[Mapping[str, str]],
    ) -> ClaimCitationQualityDecision:
        prompt = build_citation_quality_judge_prompt(
            claim=claim,
            sources=sources,
        )
        raw_response = self._complete(prompt)
        return parse_citation_quality_judge_response(
            raw_response,
            claim=claim,
            sources=sources,
        )

    def judge_citation_quality(
        self,
        *,
        claim: str,
        source_excerpt: str,
        source_id: str,
        citation_id: str,
    ) -> CitationQualityDecision:
        """Backward-compatible single-citation wrapper."""

        claim_decision = self.judge_claim_citation_quality(
            claim=claim,
            sources=[
                {
                    "citation_id": citation_id,
                    "source_id": source_id,
                    "source_excerpt": source_excerpt,
                }
            ],
        )
        necessary = citation_id in set(claim_decision.necessary_citation_ids)
        return CitationQualityDecision(
            citation_id=citation_id,
            claim_key=claim_decision.claim_key,
            source_id=source_id,
            supported=necessary,
            necessary=necessary,
            rationale=claim_decision.rationale,
        )


def create_default_citation_quality_judge(
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 4096,
    load_env: bool = True,
) -> LLMCitationQualityJudge:
    """Create the default LLM judge for LiRA-style citation quality."""

    if load_env:
        from dotenv import load_dotenv

        load_dotenv(override=True)

    resolved_api_key = api_key or os.getenv("API_KEY")
    if not resolved_api_key:
        raise EnvironmentError("API_KEY is required for citation quality judging")

    resolved_model = (
        model
        or os.getenv("META_BENCH_CITATION_QUALITY_JUDGE_MODEL")
        or os.getenv("META_BENCH_PROXY_JUDGE_MODEL")
        or os.getenv("MODEL")
        or "glm-5"
    )
    resolved_base_url = base_url if base_url is not None else os.getenv("BASE_URL")

    from src.utils.llm_client import LLMClient

    client = LLMClient(
        api_key=resolved_api_key,
        model=resolved_model,
        base_url=resolved_base_url,
    )

    def complete(prompt: str) -> str:
        return client.generate(
            prompt,
            temperature=0.0,
            max_tokens=max_tokens,
            strip_think=True,
        )

    judge = LLMCitationQualityJudge(complete)
    setattr(judge, "llm_client", client)
    setattr(judge, "model", resolved_model)
    return judge


def build_citation_quality_judge_prompt(
    *,
    claim: str,
    sources: Sequence[Mapping[str, str]],
) -> str:
    """Build a constrained LiRA-style claim-citation alignment prompt."""

    source_blocks: list[str] = []
    for index, source in enumerate(sources, start=1):
        citation_id = str(source.get("citation_id", f"C{index:03d}"))
        source_id = str(source.get("source_id", "unknown"))
        source_excerpt = str(source.get("source_excerpt", ""))
        source_blocks.append(
            f"[{index}] citation_id: {citation_id}\n"
            f"source_id: {source_id}\n"
            f"excerpt:\n{source_excerpt}"
        )
    formatted_sources = "\n\n".join(source_blocks)

    return (
        "You are a citation-quality judge for a scientific literature review. "
        "Decide whether the cited reference set supports the generated claim.\n\n"
        "Use natural-language inference over only the provided excerpts. Judge "
        "the core scientific assertion of the claim, not every rhetorical "
        "modifier or broad framing phrase. Do not require exact wording. Mark "
        "supported=true when the cited excerpts, considered together, provide "
        "reasonable evidence for the central assertion that a domain expert "
        "would accept in a review. A single excerpt does not need to support the "
        "whole claim by itself. Mark supported=false when the evidence is "
        "contradictory, unrelated, or misses the central assertion. Mark a "
        "citation as supporting when it provides relevant evidence for the "
        "claim, either alone or together with other cited excerpts. Do not "
        "require the citation to be uniquely necessary; LiRA-style precision "
        "counts whether cited references are related/supportive rather than "
        "whether they are irreplaceable.\n\n"
        "Return strict JSON with this schema:\n"
        '{\n'
        '  "supported": true,\n'
        '  "supporting_citation_ids": ["C1", "C2"],\n'
        '  "rationale": "short explanation"\n'
        '}\n\n'
        f"Generated claim:\n{claim}\n\n"
        f"Cited source excerpts:\n{formatted_sources}"
    )


def parse_citation_quality_judge_response(
    raw_response: str,
    *,
    claim: str,
    sources: Sequence[Mapping[str, str]],
) -> ClaimCitationQualityDecision:
    json_text = _extract_json_object(raw_response)
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError("citation quality judge response must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("citation quality judge response must decode to an object")
    supported = bool(parsed.get("supported", False))
    citation_ids = [str(source.get("citation_id", "")) for source in sources]
    citation_ids = [citation_id for citation_id in citation_ids if citation_id]
    source_ids = [str(source.get("source_id", "")) for source in sources]
    source_ids = [source_id for source_id in source_ids if source_id]
    valid_citation_ids = set(citation_ids)
    source_to_citation_ids: dict[str, list[str]] = defaultdict(list)
    for source in sources:
        citation_id = str(source.get("citation_id", "")).strip()
        source_id = str(source.get("source_id", "")).strip()
        if citation_id and source_id:
            source_to_citation_ids[source_id].append(citation_id)

    raw_necessary = (
        parsed.get("supporting_citation_ids")
        or parsed.get("related_citation_ids")
        or parsed.get("necessary_citation_ids")
        or []
    )
    necessary_ids: list[str] = []
    if isinstance(raw_necessary, str):
        raw_necessary = [raw_necessary]
    if isinstance(raw_necessary, Sequence):
        for raw_id in raw_necessary:
            citation_id = str(raw_id).strip()
            if citation_id in valid_citation_ids and citation_id not in necessary_ids:
                necessary_ids.append(citation_id)

    raw_source_ids = (
        parsed.get("supporting_source_ids")
        or parsed.get("related_source_ids")
        or parsed.get("necessary_source_ids")
        or []
    )
    if isinstance(raw_source_ids, str):
        raw_source_ids = [raw_source_ids]
    if isinstance(raw_source_ids, Sequence):
        for raw_source_id in raw_source_ids:
            for citation_id in source_to_citation_ids.get(str(raw_source_id).strip(), []):
                if citation_id not in necessary_ids:
                    necessary_ids.append(citation_id)

    if (
        supported
        and not necessary_ids
        and bool(parsed.get("supporting", parsed.get("necessary", supported)))
    ):
        necessary_ids = list(citation_ids)

    return ClaimCitationQualityDecision(
        claim_key=_claim_key(claim, citation_ids[0] if citation_ids else ""),
        claim=claim,
        citation_ids=citation_ids,
        source_ids=source_ids,
        supported=supported,
        necessary_citation_ids=necessary_ids,
        rationale=str(parsed.get("rationale", "")),
    )


def _extract_json_object(raw_response: str) -> str:
    text = raw_response.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _claim_key(claim: str, citation_id: str) -> str:
    normalized = re.sub(r"\s+", " ", claim.casefold()).strip()
    return normalized or citation_id


def _word_count_for_citation_quality(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", str(text or "")))


def estimate_citation_quality_claim_count(text: str) -> int:
    """Estimate generated-article claim count for LiRA recall scaling.

    LiRA defines ``nclaims`` as the average number of claims in generated
    articles. This lightweight estimator counts claim-like body sentences
    rather than citation-manifest entries, which only cover cited claims.
    """

    if not isinstance(text, str) or not text.strip():
        return 0

    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith("#")
            or re.fullmatch(r"\[[0-9,\- ]+\]", stripped)
        ):
            continue
        normalized_line = re.sub(r"\s+", " ", stripped.casefold()).strip()
        if normalized_line.startswith(("references", "bibliography")):
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", stripped):
            normalized_sentence = re.sub(
                r"\s+", " ", sentence.casefold()
            ).strip()
            if not normalized_sentence:
                continue
            if normalized_sentence.startswith(("references", "bibliography")):
                continue
            if (
                _word_count_for_citation_quality(sentence)
                >= MIN_CITATION_QUALITY_ARTICLE_CLAIM_WORDS
            ):
                count += 1
    return count


def _split_citation_quality_claim(claim: str) -> list[str]:
    """Split only obvious multi-claim spans into LiRA-style atomic claims."""

    normalized = re.sub(r"\s+", " ", str(claim or "")).strip()
    if not normalized:
        return [""]

    sentence_parts = [
        part.strip(" -")
        for part in re.split(r"(?<=[.!?])\s+", normalized)
        if part.strip(" -")
    ]
    if len(sentence_parts) > 1 and all(
        _word_count_for_citation_quality(part) >= MIN_CITATION_QUALITY_SUBCLAIM_WORDS
        for part in sentence_parts
    ):
        return sentence_parts

    semicolon_parts = [
        part.strip(" -")
        for part in re.split(r"\s*;\s*", normalized)
        if part.strip(" -")
    ]
    if len(semicolon_parts) > 1 and all(
        _word_count_for_citation_quality(part) >= MIN_CITATION_QUALITY_SUBCLAIM_WORDS
        for part in semicolon_parts
    ):
        return semicolon_parts

    return [normalized]


def _citation_quality_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text).casefold())
        if len(token) > 2
    }


def _split_citation_quality_excerpt(text: str) -> list[str]:
    excerpt = re.sub(r"\s+", " ", str(text or "")).strip()
    if not excerpt:
        return [""]
    if len(excerpt) <= MAX_CITATION_QUALITY_CHUNK_CHARS:
        return [excerpt]

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", excerpt)
        if sentence.strip()
    ]
    if not sentences:
        return [
            excerpt[index : index + MAX_CITATION_QUALITY_CHUNK_CHARS].strip()
            for index in range(0, len(excerpt), MAX_CITATION_QUALITY_CHUNK_CHARS)
        ]

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not current:
            current = sentence
            continue
        if len(current) + 1 + len(sentence) <= MAX_CITATION_QUALITY_CHUNK_CHARS:
            current = f"{current} {sentence}"
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def _select_citation_quality_chunks(claim: str, excerpt: str) -> list[str]:
    chunks = _split_citation_quality_excerpt(excerpt)
    if len(chunks) <= MAX_CITATION_QUALITY_CHUNKS_PER_SOURCE:
        return chunks

    claim_tokens = _citation_quality_tokens(claim)
    ranked: list[tuple[float, int, str]] = []
    for index, chunk in enumerate(chunks):
        chunk_tokens = _citation_quality_tokens(chunk)
        overlap = len(claim_tokens & chunk_tokens)
        normalized_overlap = overlap / max(1, len(claim_tokens))
        ranked.append((normalized_overlap, -index, chunk))
    selected = sorted(
        ranked,
        key=lambda item: (item[0], item[1]),
        reverse=True,
    )[:MAX_CITATION_QUALITY_CHUNKS_PER_SOURCE]
    return [chunk for _, _, chunk in sorted(selected, key=lambda item: -item[1])]


def _citation_quality_sources_for_claim(
    claim: str,
    claim_events: Sequence[CitationEvent],
) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    for event in claim_events:
        chunks = _select_citation_quality_chunks(claim, event.source_excerpt)
        for chunk_index, chunk in enumerate(chunks, start=1):
            source_excerpt = chunk
            if len(chunks) > 1:
                source_excerpt = f"Evidence chunk {chunk_index}/{len(chunks)}:\n{chunk}"
            sources.append(
                {
                    "citation_id": event.citation_id,
                    "source_id": event.source_id,
                    "source_excerpt": source_excerpt,
                }
            )
    return sources


PROCEDURAL_REVIEW_CLAIM_TERMS = (
    "boolean operator",
    "search strategy",
    "search string",
    "database search",
    "literature search",
    "inclusion criteria",
    "exclusion criteria",
    "eligibility criteria",
    "studies were required",
    "studies were excluded",
    "data extraction",
    "quality assessment",
    "risk of bias",
    "screening process",
    "narrative synthesis",
    "meta-analytical pooling",
    "meta analytical pooling",
    "analytical strategy",
    "systematic approach",
    "systematic evidence synthesis",
    "transparent, reproducible",
    "methodologically rigorous",
    "cross-study comparisons",
    "this review aims",
    "this review aimed",
    "this review seeks",
    "this scoping review",
    "this systematic review",
    "this meta-analysis",
    "this section details",
    "this section describes",
    "the methodological architecture",
    "designed to rigorously capture",
    "capture, evaluate, and synthesize",
    "transparently aggregating",
    "evaluating, and synthesizing",
    "studies were included if",
    "study selection",
    "study designs were considered eligible",
    "trial designs were considered eligible",
    "both parallel-group and crossover trial designs",
    "primary outcomes were",
    "secondary outcomes were",
    "search was restricted",
    "language publications",
    "from database inception",
    "supplementary search techniques",
    "reference lists",
    "bibliographies of included",
    "manually screened",
    "additional records",
    "organizing endpoints",
    "upcoming literature analysis",
)


def _is_procedural_review_claim(claim: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(claim).casefold()).strip()
    if not normalized:
        return False
    tokens = re.findall(r"[a-z0-9]+", normalized)
    if tokens and sum(token.isdigit() for token in tokens) / len(tokens) >= 0.5:
        return True
    if len(tokens) <= 3 and any(token.isdigit() for token in tokens):
        return True
    return any(term in normalized for term in PROCEDURAL_REVIEW_CLAIM_TERMS)


def infer_claim_strength(event: CitationEvent) -> int:
    """Infer claim strength deterministically from event metadata and text cues."""

    if event.claim_strength is not None:
        return int(_clamp(event.claim_strength, 0, 5))
    strength = CLAIM_ROLE_STRENGTH.get(event.claim_role, 1)
    claim_text = event.claim_span
    for cue, cue_strength in CLAIM_TEXT_CUES:
        if _contains_term(claim_text, cue):
            strength = max(strength, cue_strength)
    return int(_clamp(strength, 0, 5))


def infer_evidence_strength(event: CitationEvent) -> int:
    """Infer evidence strength deterministically from source metadata and excerpt."""

    if event.evidence_strength is not None:
        return int(_clamp(event.evidence_strength, 0, 4))
    base_strength = SOURCE_TYPE_STRENGTH.get(event.source_type, 1)
    strength = base_strength
    evidence_text = event.source_excerpt
    for cue, cue_strength in EVIDENCE_TEXT_CUES:
        if _contains_term(evidence_text, cue):
            strength = max(strength, min(cue_strength, base_strength + 1))
    return int(_clamp(strength, 0, 4))


def event_source_fidelity_penalty(event: CitationEvent) -> float:
    """Return the overclaim penalty for one citation event."""

    if event.fidelity_penalty is not None:
        return round(_clamp(event.fidelity_penalty, 0.0, 1.0), 4)

    claim_strength = infer_claim_strength(event)
    evidence_strength = infer_evidence_strength(event)
    delta = claim_strength - evidence_strength
    if delta <= 0:
        return 0.0
    if delta == 1:
        return 0.2
    if delta == 2:
        return 0.55
    return 0.9


def _event_rationale(delta: int, penalty: float) -> str:
    if penalty <= 0:
        return "claim strength does not exceed evidence strength"
    if delta == 1:
        return "claim is one level stronger than the cited evidence"
    if delta == 2:
        return "claim is two levels stronger than the cited evidence"
    return "claim is at least three levels stronger than the cited evidence"


def score_source_fidelity(
    final_text: str,
    citation_manifest: Sequence[object],
    *,
    reference: Mapping[str, object] | None = None,
    chunk_map: object | None = None,
) -> SourceFidelityResult:
    """Score whether cited claims avoid over-extrapolating from evidence."""

    if not isinstance(final_text, str) or final_text.strip() == "":
        raise ValueError("final_text must be a non-empty string")
    if not citation_manifest:
        raise ValueError("citation_manifest must contain at least one citation event")

    chunks = parse_citation_chunks(final_text, chunk_map)
    classifications = classify_chunk_roles(chunks, reference)
    chunk_type_by_id = {
        classification.chunk_id: classification.canonical_type
        for classification in classifications
    }
    citations = normalize_citation_events(citation_manifest)
    decisions: list[SourceFidelityEventDecision] = []
    penalties: list[float] = []

    for event in citations:
        claim_strength = infer_claim_strength(event)
        evidence_strength = infer_evidence_strength(event)
        delta = claim_strength - evidence_strength
        penalty = event_source_fidelity_penalty(event)
        penalties.append(penalty)
        decisions.append(
            SourceFidelityEventDecision(
                citation_id=event.citation_id,
                chunk_id=event.chunk_id,
                source_id=event.source_id,
                chunk_type=chunk_type_by_id.get(event.chunk_id, "other"),
                claim_role=event.claim_role,
                source_type=event.source_type,
                claim_strength=claim_strength,
                evidence_strength=evidence_strength,
                strength_delta=delta,
                penalty=penalty,
                score=round(1.0 - penalty, 4),
                rationale=_event_rationale(delta, penalty),
            )
        )

    average_penalty = sum(penalties) / len(penalties)
    chunk_scores = _aggregate_chunk_scores(decisions)
    return SourceFidelityResult(
        score=round(1.0 - average_penalty, 4),
        penalty=round(average_penalty, 4),
        citation_count=len(decisions),
        overclaim_count=sum(1 for decision in decisions if decision.penalty > 0),
        severe_overclaim_count=sum(1 for decision in decisions if decision.penalty >= 0.8),
        chunk_classifications=classifications,
        event_decisions=decisions,
        chunk_scores=chunk_scores,
    )


def score_citation_quality_f1(
    final_text: str,
    citation_manifest: Sequence[object],
    judge: CitationQualityJudge,
    *,
    scaling_k: float = 0.01,
    scaling_claim_count: float | None = None,
) -> CitationQualityF1Result:
    """Score LiRA-style citation quality F1 with recall scaling.

    LiRA evaluates claim-citation alignment as a balance of citation recall and
    precision, with recall scaled by ``1 - exp(-k * n_claims)``.
    """

    if not isinstance(final_text, str) or final_text.strip() == "":
        raise ValueError("final_text must be a non-empty string")
    if not citation_manifest:
        raise ValueError("citation_manifest must contain at least one citation event")

    events = normalize_citation_events(citation_manifest)
    events_by_claim: dict[str, list[CitationEvent]] = defaultdict(list)
    for event in events:
        subclaims = _split_citation_quality_claim(event.claim_span)
        for subclaim_index, subclaim in enumerate(subclaims, start=1):
            event_for_claim = (
                event
                if len(subclaims) == 1
                else CitationEvent(
                    citation_id=event.citation_id,
                    chunk_id=event.chunk_id,
                    source_id=event.source_id,
                    claim_span=subclaim,
                    source_excerpt=event.source_excerpt,
                    claim_role=event.claim_role,
                    source_type=event.source_type,
                    claim_strength=event.claim_strength,
                    evidence_strength=event.evidence_strength,
                    fidelity_penalty=event.fidelity_penalty,
                )
            )
            key_suffix = (
                event.citation_id
                if len(subclaims) == 1
                else f"{event.citation_id}#subclaim{subclaim_index}"
            )
            events_by_claim[_claim_key(subclaim, key_suffix)].append(event_for_claim)

    claim_decisions: list[ClaimCitationQualityDecision] = []
    decisions: list[CitationQualityDecision] = []
    excluded_claims: list[dict[str, object]] = []
    for claim_key, claim_events in events_by_claim.items():
        claim = claim_events[0].claim_span
        if _is_procedural_review_claim(claim):
            excluded_claims.append(
                {
                    "claim_key": claim_key,
                    "claim": claim,
                    "citation_ids": [event.citation_id for event in claim_events],
                    "source_ids": list(
                        dict.fromkeys(event.source_id for event in claim_events)
                    ),
                    "reason": "procedural_review_claim",
                }
            )
            continue
        sources = _citation_quality_sources_for_claim(claim, claim_events)
        claim_decision = _judge_claim_citation_quality(
            judge=judge,
            claim=claim,
            claim_key=claim_key,
            events=claim_events,
            sources=sources,
        )
        claim_decisions.append(claim_decision)
        necessary_ids = set(claim_decision.necessary_citation_ids)
        for event in claim_events:
            necessary = event.citation_id in necessary_ids
            decisions.append(
                CitationQualityDecision(
                    citation_id=event.citation_id,
                    claim_key=claim_key,
                    source_id=event.source_id,
                    supported=necessary,
                    necessary=necessary,
                    rationale=claim_decision.rationale,
                )
            )

    claim_count = len(events_by_claim) - len(excluded_claims)
    citation_count = len(decisions)
    supported_claim_count = sum(1 for decision in claim_decisions if decision.supported)
    necessary_citation_count = sum(1 for decision in decisions if decision.necessary)
    recall = supported_claim_count / claim_count if claim_count else 0.0
    precision = necessary_citation_count / citation_count if citation_count else 0.0
    active_scaling_claim_count = (
        float(scaling_claim_count)
        if scaling_claim_count is not None
        else float(claim_count)
    )
    if active_scaling_claim_count < 0:
        raise ValueError("scaling_claim_count must be non-negative")
    scaling = 1.0 - math.exp(-scaling_k * active_scaling_claim_count)
    scaled_recall = recall * scaling
    if precision + scaled_recall <= 0:
        score = 0.0
    else:
        score = 2 * precision * scaled_recall / (precision + scaled_recall)

    return CitationQualityF1Result(
        score=round(score, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        scaled_recall=round(scaled_recall, 4),
        scaling=round(scaling, 4),
        scaling_claim_count=round(active_scaling_claim_count, 4),
        claim_count=claim_count,
        citation_count=citation_count,
        supported_claim_count=supported_claim_count,
        necessary_citation_count=necessary_citation_count,
        excluded_claim_count=len(excluded_claims),
        excluded_citation_count=sum(
            len(item.get("citation_ids", [])) for item in excluded_claims
        ),
        excluded_claims=excluded_claims,
        decisions=decisions,
        claim_decisions=claim_decisions,
    )


def _judge_claim_citation_quality(
    *,
    judge: CitationQualityJudge,
    claim: str,
    claim_key: str,
    events: Sequence[CitationEvent],
    sources: Sequence[Mapping[str, str]],
) -> ClaimCitationQualityDecision:
    grouped_judge = getattr(judge, "judge_claim_citation_quality", None)
    if callable(grouped_judge):
        try:
            decision = grouped_judge(claim=claim, sources=sources)
        except Exception as exc:
            return ClaimCitationQualityDecision(
                claim_key=claim_key,
                claim=claim,
                citation_ids=[event.citation_id for event in events],
                source_ids=[event.source_id for event in events],
                supported=False,
                necessary_citation_ids=[],
                rationale=f"judge_error: {exc}",
            )
        valid_ids = {event.citation_id for event in events}
        necessary_ids = [
            citation_id
            for citation_id in decision.necessary_citation_ids
            if citation_id in valid_ids
        ]
        return ClaimCitationQualityDecision(
            claim_key=claim_key,
            claim=claim,
            citation_ids=[event.citation_id for event in events],
            source_ids=[event.source_id for event in events],
            supported=decision.supported,
            necessary_citation_ids=necessary_ids,
            rationale=decision.rationale,
        )

    single_judge = getattr(judge, "judge_citation_quality", None)
    if not callable(single_judge):
        raise TypeError(
            "citation quality judge must implement judge_claim_citation_quality"
        )

    event_decisions: list[CitationQualityDecision] = []
    for event in events:
        event_decisions.append(
            single_judge(
                claim=event.claim_span,
                source_excerpt=event.source_excerpt,
                source_id=event.source_id,
                citation_id=event.citation_id,
            )
        )
    necessary_ids = [
        decision.citation_id for decision in event_decisions if decision.necessary
    ]
    return ClaimCitationQualityDecision(
        claim_key=claim_key,
        claim=claim,
        citation_ids=[event.citation_id for event in events],
        source_ids=[event.source_id for event in events],
        supported=any(decision.supported for decision in event_decisions),
        necessary_citation_ids=necessary_ids,
        rationale="; ".join(
            decision.rationale for decision in event_decisions if decision.rationale
        ),
    )


def _aggregate_chunk_scores(
    decisions: Sequence[SourceFidelityEventDecision],
) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[SourceFidelityEventDecision]] = defaultdict(list)
    for decision in decisions:
        grouped[decision.chunk_id].append(decision)

    output: dict[str, dict[str, object]] = {}
    for chunk_id, chunk_decisions in grouped.items():
        average_penalty = sum(item.penalty for item in chunk_decisions) / len(chunk_decisions)
        output[chunk_id] = {
            "source_fidelity": round(1.0 - average_penalty, 4),
            "source_fidelity_penalty": round(average_penalty, 4),
            "citation_count": len(chunk_decisions),
            "overclaim_count": sum(1 for item in chunk_decisions if item.penalty > 0),
            "severe_overclaim_count": sum(
                1 for item in chunk_decisions if item.penalty >= 0.8
            ),
        }
    return output


def _seven_section_key_from_value(value: object) -> str | None:
    normalized = _normalize_token(value)
    aliases = {
        "title": "title",
        "abstract": "abstract",
        "keywords": "keywords",
        "keyword": "keywords",
        "introduction": "introduction",
        "intro": "introduction",
        "background": "introduction",
        "main_body": "main_body",
        "body": "main_body",
        "evidence": "main_body",
        "evidence_synthesis": "main_body",
        "taxonomy": "main_body",
        "framework": "main_body",
        "methods": "main_body",
        "results": "main_body",
        "discussion": "main_body",
        "limitations_gaps": "conclusion",
        "limitations": "conclusion",
        "gaps": "conclusion",
        "conclusion": "conclusion",
        "conclusions": "conclusion",
        "summary": "conclusion",
        "future_work": "conclusion",
        "references": "references",
        "reference": "references",
        "bibliography": "references",
    }
    if normalized in aliases:
        return aliases[normalized]

    return None


def _six_slot_key_from_value(value: object) -> str | None:
    normalized = _normalize_token(value)
    aliases = {
        "scope_context": "scope_context",
        "scope": "scope_context",
        "intro": "scope_context",
        "introduction": "scope_context",
        "framework_mechanism": "framework_mechanism",
        "framework": "framework_mechanism",
        "mechanism": "framework_mechanism",
        "evidence_methods": "evidence_methods",
        "methods": "evidence_methods",
        "findings_synthesis": "findings_synthesis",
        "findings": "findings_synthesis",
        "synthesis": "findings_synthesis",
        "implications_discussion": "implications_discussion",
        "discussion": "implications_discussion",
        "implications": "implications_discussion",
        "limitations_future": "limitations_future",
        "limitations_gaps": "limitations_future",
        "limitations": "limitations_future",
        "conclusion": "limitations_future",
        "future_work": "limitations_future",
        "future": "limitations_future",
    }
    return aliases.get(normalized)


def _extract_outline_items(
    reference: Mapping[str, object] | None,
) -> list[tuple[str, str]]:
    if reference is None:
        return []
    raw_outline = reference.get("outline")
    if not isinstance(raw_outline, Mapping):
        return []
    outline_items: list[tuple[str, str]] = []
    for raw_section_id, raw_title in raw_outline.items():
        section_id = str(raw_section_id).strip()
        if not section_id:
            continue
        outline_items.append((section_id, str(raw_title).strip()))
    return outline_items


@lru_cache(maxsize=1)
def _load_six_slot_section_distribution_thresholds() -> dict[str, dict[str, float]] | None:
    from .six_slot_priors import (
        SIX_SLOT_ORDER,
        load_six_slot_citation_prior_payload,
    )

    payload = load_six_slot_citation_prior_payload()
    if payload is None:
        return None
    raw_stats = payload.get("slot_stats")
    raw_mean_shares = payload.get("normalized_mean_shares")
    if not isinstance(raw_stats, Mapping) or not isinstance(raw_mean_shares, Mapping):
        return None

    thresholds: dict[str, dict[str, float]] = {}
    for slot in SIX_SLOT_ORDER:
        slot_stats = raw_stats.get(slot)
        raw_weight = raw_mean_shares.get(slot)
        if not isinstance(slot_stats, Mapping) or not isinstance(raw_weight, (int, float)):
            return None
        p25 = float(slot_stats.get("p25_share", 0.0))
        p75 = float(slot_stats.get("p75_share", 0.0))
        iqr = max(0.0, p75 - p25)
        soft_pad = 0.25 * iqr
        hard_lower = max(0.0, p25 - 1.5 * iqr)
        hard_upper = min(1.0, p75 + 1.5 * iqr)
        thresholds[slot] = {
            "soft_lower": round(max(0.0, p25 - soft_pad), 6),
            "soft_upper": round(min(1.0, p75 + soft_pad), 6),
            "hard_lower": round(min(hard_lower, p25), 6),
            "hard_upper": round(max(hard_upper, p75), 6),
            "weight": round(float(raw_weight), 6),
        }
    return thresholds


@lru_cache(maxsize=1)
def _load_six_slot_section_distribution_metadata() -> dict[str, object] | None:
    from .six_slot_priors import load_six_slot_citation_prior_metadata

    prior_metadata = load_six_slot_citation_prior_metadata()
    thresholds = _load_six_slot_section_distribution_thresholds()
    if prior_metadata is None or thresholds is None:
        return None
    return {
        "prior_version": prior_metadata.get("prior_version"),
        "source_corpus_dir": prior_metadata.get("source_corpus_dir"),
        "article_count": prior_metadata.get("article_count"),
        "mapped_article_count": prior_metadata.get("mapped_article_count"),
        "slot_order": prior_metadata.get("slot_order"),
        "threshold_source": "data_sample/six_slot_citation_priors.json",
        "threshold_method": "soft=(p25-0.25*IQR, p75+0.25*IQR), hard=(p25-1.5*IQR, p75+1.5*IQR)",
    }


def _resolve_six_slot_section_mapping(
    reference: Mapping[str, object] | None,
    chunk_map: object | None,
) -> tuple[list[str], dict[str, str], dict[str, dict[str, float]]] | None:
    from .six_slot_priors import SIX_SLOT_ORDER, classify_outline_to_six_slots

    outline_items = _extract_outline_items(reference)
    if not outline_items:
        return None

    slot_by_section = classify_outline_to_six_slots(outline_items)
    if not slot_by_section:
        return None

    thresholds = _load_six_slot_section_distribution_thresholds()
    if thresholds is None:
        return None

    mapping: dict[str, str] = dict(slot_by_section)
    if isinstance(chunk_map, list):
        for raw_item in chunk_map:
            if not isinstance(raw_item, Mapping):
                continue
            item = cast(Mapping[str, object], raw_item)
            section_id = str(item.get("section_id", "")).strip()
            chunk_id = str(item.get("chunk_id", "")).strip()
            slot = slot_by_section.get(section_id)
            if slot is None:
                continue
            if section_id:
                mapping[section_id] = slot
            if chunk_id:
                mapping[chunk_id] = slot

    return list(SIX_SLOT_ORDER), mapping, thresholds


def _section_key_by_chunk_id(
    reference: Mapping[str, object] | None,
    chunk_map: object | None,
) -> dict[str, str]:
    raw_chunks = chunk_map
    if raw_chunks is None and reference is not None:
        raw_chunks = _extract_chunk_map(reference)
    if not isinstance(raw_chunks, list):
        return {}

    mapping: dict[str, str] = {}
    for index, raw_item in enumerate(raw_chunks, start=1):
        if not isinstance(raw_item, Mapping):
            continue
        item = cast(Mapping[str, object], raw_item)
        chunk_id = str(
            item.get("chunk_id", item.get("section_id", f"sec{index}"))
        ).strip() or f"sec{index}"
        section_key = None
        for key in ("section_key", "seven_section", "section_group", "slot"):
            if key in item:
                section_key = _seven_section_key_from_value(item.get(key))
                if section_key:
                    break
        if section_key is None:
            section_key = _seven_section_key_from_value(
                item.get("type", item.get("section_type", item.get("title", "")))
            )
        if section_key is not None:
            mapping[chunk_id] = section_key
    return mapping


def _citation_counts_from_manifest(
    citation_manifest: Sequence[object],
    *,
    reference: Mapping[str, object] | None = None,
    chunk_map: object | None = None,
) -> dict[str, int] | None:
    chunk_mapping = _section_key_by_chunk_id(reference, chunk_map)
    counts = {section_key: 0 for section_key in SEVEN_SECTION_KEYS}
    mapped_count = 0
    for raw_item in citation_manifest:
        if not isinstance(raw_item, Mapping):
            raise TypeError("each citation event must be a mapping")
        item = cast(Mapping[str, object], raw_item)
        chunk_id = str(
            item.get("chunk_id", item.get("section_id", item.get("section", "")))
        ).strip()
        section_key = None
        for key in ("section_key", "seven_section", "section_group", "slot"):
            if key in item:
                section_key = _seven_section_key_from_value(item.get(key))
                if section_key:
                    break
        if section_key is None:
            section_key = chunk_mapping.get(chunk_id)
        if section_key is None and chunk_id:
            section_key = _seven_section_key_from_value(chunk_id)
        if section_key is None and item.get("section_id") is not None:
            section_key = _seven_section_key_from_value(item.get("section_id"))
        if section_key is None:
            continue
        mapped_count += 1
        counts[section_key] += 1
    if mapped_count == 0:
        return None
    return counts


def _citation_counts_from_manifest_by_keys(
    citation_manifest: Sequence[object],
    *,
    valid_keys: Sequence[str],
    mapping: Mapping[str, str] | None = None,
    key_resolver: Callable[[object], str | None] | None = None,
    prefer_mapping: bool = False,
) -> dict[str, int] | None:
    counts = {section_key: 0 for section_key in valid_keys}
    mapped_count = 0
    for raw_item in citation_manifest:
        if not isinstance(raw_item, Mapping):
            raise TypeError("each citation event must be a mapping")
        item = cast(Mapping[str, object], raw_item)
        chunk_id = str(
            item.get("chunk_id", item.get("section_id", item.get("section", "")))
        ).strip()

        section_key = None
        if prefer_mapping and mapping and chunk_id:
            section_key = mapping.get(chunk_id)

        if section_key is None and key_resolver is not None:
            for key in ("slot", "section_key", "seven_section", "section_group"):
                if key in item:
                    resolved = key_resolver(item.get(key))
                    if resolved in counts:
                        section_key = resolved
                        break

        if section_key is None and mapping and chunk_id:
            section_key = mapping.get(chunk_id)
        if section_key is None and mapping and item.get("section_id") is not None:
            section_key = mapping.get(str(item.get("section_id")).strip())
        if section_key is None and key_resolver is not None and chunk_id:
            resolved = key_resolver(chunk_id)
            if resolved in counts:
                section_key = resolved
        if section_key is None:
            continue
        mapped_count += 1
        counts[section_key] += 1

    if mapped_count == 0:
        return None
    return counts


def _content_sections_for_distribution(final_text: str) -> list[str]:
    content_sections: list[str] = []
    for section in parse_markdown_sections(final_text):
        if section.is_title:
            continue
        normalized_heading = normalize_section_heading(section.heading)
        if normalized_heading in REFERENCE_HEADINGS:
            continue
        if normalized_heading in {"abstract", "keywords", "keyword"}:
            continue
        if normalized_heading in CITATION_NON_CONTENT_HEADINGS:
            continue
        if section.body.strip():
            content_sections.append(section.body)
    return content_sections


def _citation_counts_from_six_slot_text(
    final_text: str,
    *,
    slot_order: Sequence[str],
) -> dict[str, int] | None:
    content_sections = _content_sections_for_distribution(final_text)
    if len(content_sections) != len(slot_order):
        return None

    reference_item_count = count_reference_items(
        split_into_seven_sections(final_text)["references"].text
    )
    counts = {slot: 0 for slot in slot_order}
    for slot, text in zip(slot_order, content_sections):
        counts[slot] = count_citations_in_text(
            text,
            max_reference_number=reference_item_count,
        ).total_citations
    return counts


def _citation_counts_from_text(final_text: str) -> dict[str, int]:
    sections = split_into_seven_sections(final_text)
    reference_item_count = count_reference_items(sections["references"].text)
    counts = {section_key: 0 for section_key in SEVEN_SECTION_KEYS}
    for section_key, section in sections.items():
        if section_key == "references":
            counts[section_key] = 0
            continue
        counts[section_key] = count_citations_in_text(
            section.text,
            max_reference_number=reference_item_count,
        ).total_citations
    return counts


def _reference_counts_from_manifest(
    citation_manifest: Sequence[object],
) -> dict[str, int] | None:
    counts: Counter[str] = Counter()
    for index, raw_item in enumerate(citation_manifest, start=1):
        if not isinstance(raw_item, Mapping):
            raise TypeError("each citation event must be a mapping")
        item = cast(Mapping[str, object], raw_item)
        source_id = str(
            item.get("source_id", item.get("reference_id", item.get("ref_id", "")))
        ).strip()
        if not source_id:
            source_id = f"citation_event:{index}"
        counts[source_id] += 1
    if not counts:
        return None
    return dict(counts)


def _reference_counts_from_text(final_text: str) -> dict[str, int]:
    sections = split_into_seven_sections(final_text)
    reference_item_count = count_reference_items(sections["references"].text)
    counts: Counter[str] = Counter()
    for section_key, section in sections.items():
        if section_key == "references":
            continue
        citation_result = count_citations_in_text(
            section.text,
            max_reference_number=reference_item_count,
        )
        counts.update(citation_result.reference_counts)
    return dict(counts)


def _non_reference_word_count(final_text: str) -> int:
    sections = split_into_seven_sections(final_text)
    return sum(
        section.word_count
        for section_key, section in sections.items()
        if section_key != "references"
    )


def _citation_count_penalty(density: float) -> float:
    soft_lower = CITATION_COUNT_THRESHOLDS["soft_lower"]
    soft_upper = CITATION_COUNT_THRESHOLDS["soft_upper"]
    hard_lower = CITATION_COUNT_THRESHOLDS["hard_lower"]
    hard_upper = CITATION_COUNT_THRESHOLDS["hard_upper"]

    if soft_lower <= density <= soft_upper:
        return 0.0
    if density < soft_lower:
        denominator = soft_lower - hard_lower
        if denominator <= 0:
            return 1.0
        return _clamp((soft_lower - density) / denominator, 0.0, 1.0)
    denominator = hard_upper - soft_upper
    if denominator <= 0:
        return 1.0
    return _clamp((density - soft_upper) / denominator, 0.0, 1.0)


def score_citation_count(
    final_text: str,
    citation_manifest: Sequence[object] | None = None,
    *,
    reference: Mapping[str, object] | None = None,
    chunk_map: object | None = None,
) -> CitationCountScoreResult:
    """Score whether the whole-text citation density is reasonable."""

    if not isinstance(final_text, str) or final_text.strip() == "":
        raise ValueError("final_text must be a non-empty string")

    word_count = _non_reference_word_count(final_text)
    if word_count <= 0:
        raise ValueError("final_text contains no countable non-reference words")

    if citation_manifest:
        manifest_counts = _citation_counts_from_manifest(
            citation_manifest,
            reference=reference,
            chunk_map=chunk_map,
        )
        if manifest_counts is not None:
            section_counts = manifest_counts
            source = "citation_manifest"
        else:
            section_counts = _citation_counts_from_text(final_text)
            source = "text_regex_manifest_unmapped"
    else:
        section_counts = _citation_counts_from_text(final_text)
        source = "text_regex"

    citation_count = sum(
        section_counts.get(section_key, 0)
        for section_key in SEVEN_SECTION_KEYS
        if section_key != "references"
    )
    density = citation_count * 1000.0 / word_count
    penalty = _citation_count_penalty(density)
    return CitationCountScoreResult(
        score=round(1.0 - penalty, 4),
        penalty=round(penalty, 4),
        word_count_without_references=word_count,
        citation_count_without_references=citation_count,
        citations_per_1000_words=round(density, 4),
        thresholds=CITATION_COUNT_THRESHOLDS,
        source=source,
        section_counts=section_counts,
    )


def _normalized_source_balance_score(source_counts: Mapping[str, int]) -> tuple[float, float]:
    """Return normalized Simpson diversity and raw HHI concentration.

    HHI=sum(p_i^2) is 1.0 when all citations use one source and approaches
    1/K when citations are evenly distributed across K cited sources. The
    normalized score maps the attainable [1/K, 1] range to [1, 0].
    """

    counts = [count for count in source_counts.values() if count > 0]
    total = sum(counts)
    cited_source_count = len(counts)
    if total <= 0 or cited_source_count <= 1:
        return 0.0, 1.0

    hhi = sum((count / total) ** 2 for count in counts)
    max_diversity = 1.0 - (1.0 / cited_source_count)
    if max_diversity <= 0:
        return 0.0, hhi
    score = (1.0 - hhi) / max_diversity
    return _clamp(score, 0.0, 1.0), hhi


def score_source_balance(
    final_text: str,
    citation_manifest: Sequence[object] | None = None,
) -> SourceBalanceResult:
    """Score whether citations are over-concentrated in one source."""

    if not isinstance(final_text, str) or final_text.strip() == "":
        raise ValueError("final_text must be a non-empty string")

    if citation_manifest:
        source_counts = _reference_counts_from_manifest(citation_manifest)
        source = "citation_manifest" if source_counts is not None else "text_regex"
    else:
        source_counts = None
        source = "text_regex"
    if source_counts is None:
        source_counts = _reference_counts_from_text(final_text)

    total_citations = sum(source_counts.values())
    if total_citations <= 0:
        max_count = 0
        max_share = 0.0
        dominant_source_id = ""
        score = 0.0
        hhi = 1.0
        cited_source_count = 0
    else:
        dominant_source_id, max_count = max(
            source_counts.items(),
            key=lambda item: (item[1], item[0]),
        )
        max_share = max_count / total_citations
        score, hhi = _normalized_source_balance_score(source_counts)
        cited_source_count = sum(1 for count in source_counts.values() if count > 0)
    penalty = 1.0 - score

    return SourceBalanceResult(
        score=round(score, 4),
        penalty=round(penalty, 4),
        total_citations=total_citations,
        max_single_source_count=max_count,
        max_single_source_share=round(max_share, 4),
        dominant_source_id=dominant_source_id,
        concentration_hhi=round(hhi, 4),
        cited_source_count=cited_source_count,
        method="normalized_hhi",
        source=source,
        source_counts=source_counts,
    )


def _section_distribution_penalty(
    section_key: str,
    share: float,
    thresholds: Mapping[str, Mapping[str, float]],
) -> float:
    threshold = thresholds[section_key]
    soft_lower = threshold["soft_lower"]
    soft_upper = threshold["soft_upper"]
    hard_lower = threshold["hard_lower"]
    hard_upper = threshold["hard_upper"]

    if soft_lower <= share <= soft_upper:
        return 0.0
    if share < soft_lower:
        denominator = soft_lower - hard_lower
        if denominator <= 0:
            return 1.0
        return _clamp((soft_lower - share) / denominator, 0.0, 1.0)
    denominator = hard_upper - soft_upper
    if denominator <= 0:
        return 1.0
    return _clamp((share - soft_upper) / denominator, 0.0, 1.0)


def _score_section_distribution_from_counts(
    *,
    section_counts: dict[str, int],
    thresholds: dict[str, dict[str, float]],
    source: str,
    scheme: str,
    min_citations: int,
    prior_metadata: dict[str, object] | None = None,
) -> SectionDistributionResult:
    counted_keys = [
        section_key
        for section_key in thresholds
        if section_key != "references"
    ]
    total_citations = sum(section_counts.get(section_key, 0) for section_key in counted_keys)
    if total_citations < min_citations:
        section_shares = {section_key: 0.0 for section_key in thresholds}
        if total_citations > 0:
            section_shares = {
                section_key: round(
                    section_counts.get(section_key, 0) / total_citations,
                    4,
                )
                for section_key in thresholds
            }
        return SectionDistributionResult(
            score=None,
            penalty=None,
            status="not_evaluated",
            total_citations=total_citations,
            section_counts=section_counts,
            section_shares=section_shares,
            section_penalties={},
            thresholds=thresholds,
            source=source,
            scheme=scheme,
            reason="low_citation_count",
            prior_metadata=prior_metadata,
        )

    raw_shares = {
        section_key: section_counts.get(section_key, 0) / total_citations
        for section_key in thresholds
    }
    section_penalties = {
        section_key: _section_distribution_penalty(section_key, share, thresholds)
        for section_key, share in raw_shares.items()
    }
    weighted_penalty = sum(
        section_penalties[section_key] * thresholds[section_key]["weight"]
        for section_key in thresholds
    )
    return SectionDistributionResult(
        score=round(1.0 - weighted_penalty, 4),
        penalty=round(weighted_penalty, 4),
        status="evaluated",
        total_citations=total_citations,
        section_counts=section_counts,
        section_shares={
            section_key: round(share, 4)
            for section_key, share in raw_shares.items()
        },
        section_penalties={
            section_key: round(penalty, 4)
            for section_key, penalty in section_penalties.items()
        },
        thresholds=thresholds,
        source=source,
        scheme=scheme,
        prior_metadata=prior_metadata,
    )


def score_section_distribution(
    final_text: str,
    citation_manifest: Sequence[object] | None = None,
    *,
    reference: Mapping[str, object] | None = None,
    chunk_map: object | None = None,
    min_citations: int = MIN_CITATIONS_FOR_DISTRIBUTION,
) -> SectionDistributionResult:
    """Score whether citations are reasonably distributed across the task layout."""

    if not isinstance(final_text, str) or final_text.strip() == "":
        raise ValueError("final_text must be a non-empty string")
    if min_citations <= 0:
        raise ValueError("min_citations must be positive")

    six_slot_layout = _resolve_six_slot_section_mapping(reference, chunk_map)
    if six_slot_layout is not None:
        slot_order, slot_mapping, six_slot_thresholds = six_slot_layout
        prior_metadata = _load_six_slot_section_distribution_metadata()
        if citation_manifest:
            manifest_counts = _citation_counts_from_manifest_by_keys(
                citation_manifest,
                valid_keys=slot_order,
                mapping=slot_mapping,
                key_resolver=_six_slot_key_from_value,
                prefer_mapping=True,
            )
            if manifest_counts is not None:
                return _score_section_distribution_from_counts(
                    section_counts=manifest_counts,
                    thresholds=six_slot_thresholds,
                    source="citation_manifest",
                    scheme="six_slot_task",
                    min_citations=min_citations,
                    prior_metadata=prior_metadata,
                )

            text_counts = _citation_counts_from_six_slot_text(
                final_text,
                slot_order=slot_order,
            )
            if text_counts is not None:
                return _score_section_distribution_from_counts(
                    section_counts=text_counts,
                    thresholds=six_slot_thresholds,
                    source="text_regex_manifest_unmapped",
                    scheme="six_slot_task",
                    min_citations=min_citations,
                    prior_metadata=prior_metadata,
                )
        else:
            text_counts = _citation_counts_from_six_slot_text(
                final_text,
                slot_order=slot_order,
            )
            if text_counts is not None:
                return _score_section_distribution_from_counts(
                    section_counts=text_counts,
                    thresholds=six_slot_thresholds,
                    source="text_regex",
                    scheme="six_slot_task",
                    min_citations=min_citations,
                    prior_metadata=prior_metadata,
                )

    if citation_manifest:
        manifest_counts = _citation_counts_from_manifest(
            citation_manifest,
            reference=reference,
            chunk_map=chunk_map,
        )
        if manifest_counts is not None:
            return _score_section_distribution_from_counts(
                section_counts=manifest_counts,
                thresholds=SECTION_DISTRIBUTION_THRESHOLDS,
                source="citation_manifest",
                scheme="seven_section_fallback",
                min_citations=min_citations,
            )
        section_counts = _citation_counts_from_text(final_text)
        source = "text_regex_manifest_unmapped"
    else:
        section_counts = _citation_counts_from_text(final_text)
        source = "text_regex"

    return _score_section_distribution_from_counts(
        section_counts=section_counts,
        thresholds=SECTION_DISTRIBUTION_THRESHOLDS,
        source=source,
        scheme="seven_section_fallback",
        min_citations=min_citations,
    )


def _extract_citation_manifest(reference: Mapping[str, object]) -> list[object]:
    raw_manifest = reference.get("citation_manifest")
    if raw_manifest is None:
        raw_manifest = reference.get("citations")
    if raw_manifest is None:
        return []
    if not isinstance(raw_manifest, list):
        raise TypeError("reference.citation_manifest must be a list")
    return list(raw_manifest)


def _extract_chunk_map(reference: Mapping[str, object]) -> object | None:
    for key in ("chunk_map", "section_map", "sections"):
        raw_chunks = reference.get(key)
        if raw_chunks is not None:
            return raw_chunks
    return None


def evaluate_citation_dimension(
    final_text: str,
    reference: Mapping[str, object],
    *,
    citation_quality_judge: CitationQualityJudge | Literal["default"] | None = None,
) -> dict[str, object]:
    """Evaluate citation-dimension scores currently implemented in MetaBench."""

    citation_manifest = _extract_citation_manifest(reference)
    chunk_map = _extract_chunk_map(reference)
    citation_count_result = score_citation_count(
        final_text=final_text,
        citation_manifest=citation_manifest,
        reference=reference,
        chunk_map=chunk_map,
    )
    source_balance_result = score_source_balance(
        final_text=final_text,
        citation_manifest=citation_manifest,
    )
    section_distribution_result = score_section_distribution(
        final_text=final_text,
        citation_manifest=citation_manifest,
        reference=reference,
        chunk_map=chunk_map,
    )
    if not citation_manifest:
        return {
            "dimension": "citation",
            "scores": {
                "citation_count": citation_count_result.score,
                "source_balance": source_balance_result.score,
                **(
                    {"section_distribution": section_distribution_result.score}
                    if section_distribution_result.score is not None
                    else {}
                )
            },
            "diagnostics": {
                "citation_count": citation_count_result.to_dict(),
                "source_balance": source_balance_result.to_dict(),
                "section_distribution": section_distribution_result.to_dict(),
                "citation_quality_f1": {
                    "status": "not_evaluated",
                    "reason": "missing_citation_manifest",
                }
            },
        }

    source_fidelity_result = score_source_fidelity(
        final_text=final_text,
        citation_manifest=citation_manifest,
        reference=reference,
        chunk_map=chunk_map,
    )
    quality_result: CitationQualityF1Result | None = None
    quality_error: str | None = None
    if citation_quality_judge == "default":
        try:
            active_quality_judge: CitationQualityJudge | None = (
                create_default_citation_quality_judge()
            )
        except Exception as exc:
            active_quality_judge = None
            quality_error = str(exc)
    else:
        active_quality_judge = citation_quality_judge

    if active_quality_judge is not None:
        try:
            raw_scaling_claim_count = reference.get(
                "citation_quality_scaling_claim_count"
            )
            scaling_claim_count = (
                float(raw_scaling_claim_count)
                if raw_scaling_claim_count is not None
                else None
            )
            quality_result = score_citation_quality_f1(
                final_text=final_text,
                citation_manifest=citation_manifest,
                judge=active_quality_judge,
                scaling_claim_count=scaling_claim_count,
            )
        except Exception as exc:
            quality_error = str(exc)

    quality_scores = (
        {"citation_quality_f1": quality_result.score}
        if quality_result is not None
        else {}
    )
    quality_diagnostics: dict[str, object]
    if quality_result is not None:
        quality_diagnostics = quality_result.to_dict()
        if isinstance(reference, Mapping) and reference.get(
            "citation_quality_scaling_claim_count_source"
        ):
            quality_diagnostics["scaling_claim_count_source"] = str(
                reference.get("citation_quality_scaling_claim_count_source")
            )
    else:
        quality_diagnostics = {
            "status": "not_evaluated",
            "reason": "citation_quality_judge_not_available",
            **({"error": quality_error} if quality_error else {}),
        }
    return {
        "dimension": "citation",
        "scores": {
            **quality_scores,
            "citation_count": citation_count_result.score,
            "source_balance": source_balance_result.score,
            **(
                {"section_distribution": section_distribution_result.score}
                if section_distribution_result.score is not None
                else {}
            ),
        },
        "diagnostics": {
            "citation_count": citation_count_result.to_dict(),
            "source_balance": source_balance_result.to_dict(),
            "section_distribution": section_distribution_result.to_dict(),
            "citation_quality_f1": quality_diagnostics,
            "source_fidelity_legacy": source_fidelity_result.to_dict(),
        },
    }
