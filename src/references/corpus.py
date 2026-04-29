"""
CorpusLoader — loads the metabench paper corpus and provides BM25 search.

The BM25 index is built in memory at first call to load().  With 200 papers
and typical chunk counts (~20 chunks/paper), the full index fits comfortably
in RAM and searches complete in < 50 ms.

BM25Okapi formula (k1=1.5, b=0.75):
    score(q,d) = Σ_t  IDF(t) * tf(t,d)*(k1+1) / (tf(t,d) + k1*(1-b+b*|d|/avgdl))
    IDF(t)     = ln((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
"""
from __future__ import annotations

import logging
import math
import re
import string
from collections import Counter
from typing import Dict, List, Optional, Tuple

from .loaders.markdown_loader import load_corpus_dir

logger = logging.getLogger(__name__)


# ── Tokenizer ─────────────────────────────────────────────────────────────────

_PUNCT_TABLE = str.maketrans(string.punctuation, " " * len(string.punctuation))
_STOPWORDS = frozenset(
    "a an the and or but in on at to for of with from by is are was were be been "
    "being have has had do does did will would could should may might shall can "
    "this that these those it its we our they their he she his her i my you your "
    "not no nor so yet also just each many any all some such only both either "
    "than then when where which who whom what how as if".split()
)


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, remove stopwords, return token list."""
    lowered = text.lower().translate(_PUNCT_TABLE)
    return [t for t in lowered.split() if len(t) >= 2 and t not in _STOPWORDS]


# ── BM25 index ────────────────────────────────────────────────────────────────

class _BM25Index:
    """In-memory BM25Okapi index over a list of documents."""

    K1 = 1.5
    B = 0.75

    def __init__(self, documents: List[List[str]]):
        self._n = len(documents)
        self._dl: List[int] = [len(d) for d in documents]
        self._avgdl: float = sum(self._dl) / max(self._n, 1)
        self._tf: List[Counter] = [Counter(d) for d in documents]
        self._df: Counter = Counter()
        for doc in documents:
            for term in set(doc):
                self._df[term] += 1

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log((self._n - df + 0.5) / (df + 0.5) + 1.0)

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self._n
        for term in query_tokens:
            idf = self._idf(term)
            for idx in range(self._n):
                tf = self._tf[idx].get(term, 0)
                if tf == 0:
                    continue
                dl = self._dl[idx]
                denom = tf + self.K1 * (1 - self.B + self.B * dl / self._avgdl)
                scores[idx] += idf * tf * (self.K1 + 1) / denom
        return scores

    def top_k(self, query_tokens: List[str], k: int) -> List[Tuple[int, float]]:
        """Return (doc_index, score) pairs for the top-k results."""
        scores = self.get_scores(query_tokens)
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(idx, s) for idx, s in indexed[:k] if s > 0.0]


# ── CorpusLoader ──────────────────────────────────────────────────────────────

class CorpusLoader:
    """
    Loads the paper corpus and answers retrieval queries via BM25.

    Usage
    -----
    corpus = CorpusLoader("./metabench/output")
    corpus.load()           # parses all .md files, builds index
    results = corpus.search("treatment of mitochondrial disease", top_k=5)

    The search index is over abstract+title (high weight) and chunk text.
    Each entry in the index corresponds to one chunk; paper-level matching
    is promoted by including abstract tokens in every chunk's index entry.
    """

    def __init__(self, corpus_dir: str):
        self._corpus_dir = corpus_dir
        self._papers: Dict[str, dict] = {}          # paper_id → paper dict
        self._chunks: Dict[str, dict] = {}           # chunk_id → chunk dict
        self._index: Optional[_BM25Index] = None
        self._index_entries: List[Tuple[str, str]] = []  # [(paper_id, chunk_id)]
        self._loaded = False

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Parse all corpus files and build the BM25 index. Idempotent."""
        if self._loaded:
            return

        papers = load_corpus_dir(self._corpus_dir)
        if not papers:
            logger.warning("CorpusLoader: no papers found in %s", self._corpus_dir)
            self._loaded = True
            return

        documents: List[List[str]] = []

        for paper in papers:
            pid = paper["paper_id"]
            self._papers[pid] = paper

            abstract_tokens = _tokenize(paper.get("index_text", ""))

            for chunk in paper.get("chunks", []):
                cid = chunk["chunk_id"]
                self._chunks[cid] = {**chunk, "paper_id": pid, "title": paper["title"]}

                # index entry: abstract tokens (x2 weight) + chunk text tokens
                chunk_tokens = _tokenize(chunk.get("text", ""))
                entry_tokens = abstract_tokens * 2 + chunk_tokens
                documents.append(entry_tokens)
                self._index_entries.append((pid, cid))

        self._index = _BM25Index(documents)
        self._loaded = True
        logger.info(
            "CorpusLoader: indexed %d papers, %d chunks",
            len(self._papers),
            len(self._chunks),
        )

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 20) -> List[dict]:
        """
        Return up to top_k chunk-level results for a free-text query.

        Each result dict:
            paper_id, title, chunk_id, section_title, text, score
        """
        if not self._loaded:
            self.load()
        if self._index is None or not self._index_entries:
            return []

        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        hits = self._index.top_k(q_tokens, top_k)
        results: List[dict] = []
        for doc_idx, score in hits:
            pid, cid = self._index_entries[doc_idx]
            chunk = self._chunks.get(cid, {})
            paper = self._papers.get(pid, {})
            results.append({
                "paper_id": pid,
                "title": paper.get("title", ""),
                "chunk_id": cid,
                "section_title": chunk.get("section_title", ""),
                "text": chunk.get("text", ""),
                "score": score,
            })
        return results

    # ── Lookup helpers ────────────────────────────────────────────────────────

    def list_papers(self) -> List[dict]:
        """Return all loaded papers (without chunks, for listing)."""
        if not self._loaded:
            self.load()
        return [
            {
                "paper_id": p["paper_id"],
                "title": p["title"],
                "abstract": p["abstract"][:300],
                "subject_areas": p["subject_areas"],
            }
            for p in self._papers.values()
        ]

    def get_paper(self, paper_id: str) -> Optional[dict]:
        if not self._loaded:
            self.load()
        return self._papers.get(paper_id)

    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        if not self._loaded:
            self.load()
        return self._chunks.get(chunk_id)

    def paper_count(self) -> int:
        if not self._loaded:
            self.load()
        return len(self._papers)

    def chunk_count(self) -> int:
        if not self._loaded:
            self.load()
        return len(self._chunks)
