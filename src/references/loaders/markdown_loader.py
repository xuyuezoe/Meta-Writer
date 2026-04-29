"""
Markdown corpus loader for metabench/output/*.md files.

Each file has a YAML frontmatter block (between --- delimiters) followed by
the paper body in Markdown.  We extract structured metadata from the frontmatter
and split the body into chunks at every Markdown heading (# / ##).

Returned paper dict schema
--------------------------
{
    "paper_id":     str,
    "title":        str,
    "abstract":     str,
    "authors":      List[str],
    "subject_areas":List[str],
    "year":         int | None,
    "doi":          str | None,
    "word_count":   int,
    "chunks": [
        {
            "chunk_id":      str,          # "{paper_id}_s{N}"
            "section_title": str,
            "text":          str,
        },
        ...
    ]
}
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── YAML frontmatter parser (no external deps) ────────────────────────────────

def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    """
    Split raw file content into (frontmatter_dict, body).

    Handles the simple subset of YAML used by the metabench files:
    scalar strings, lists, nested dicts (affiliations), nulls.
    Returns ({}, raw) if no frontmatter delimiters are found.
    """
    if not raw.startswith("---"):
        return {}, raw

    end = raw.find("\n---", 3)
    if end == -1:
        return {}, raw

    yaml_block = raw[3:end].strip()
    body = raw[end + 4:].lstrip("\n")

    try:
        import yaml  # type: ignore
        meta = yaml.safe_load(yaml_block) or {}
    except Exception:
        meta = _simple_yaml_parse(yaml_block)

    return meta, body


def _simple_yaml_parse(text: str) -> dict:
    """
    Minimal YAML parser covering the metabench frontmatter structure.
    Falls back gracefully on anything it cannot handle.
    """
    result: dict = {}
    lines = text.splitlines()
    i = 0
    current_key: Optional[str] = None
    current_list: Optional[list] = None

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # skip blank / comment
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        # list item continuation
        if stripped.startswith("- ") and current_key is not None and current_list is not None:
            current_list.append(stripped[2:].strip().strip("'\""))
            i += 1
            continue

        # key: value pair
        m = re.match(r"^(\w[\w\s]*?):\s*(.*)", line)
        if m:
            current_list = None
            key = m.group(1).strip()
            val = m.group(2).strip()

            if val == "" or val == "null":
                result[key] = None
                current_key = key
            elif val.startswith("'") or val.startswith('"'):
                result[key] = val.strip("'\"")
                current_key = key
            elif val == "[]":
                result[key] = []
                current_key = key
                current_list = result[key]
            else:
                try:
                    result[key] = int(val)
                except ValueError:
                    try:
                        result[key] = float(val)
                    except ValueError:
                        result[key] = val.strip("'\"")
                current_key = key
        i += 1

    return result


# ── Markdown chunker ──────────────────────────────────────────────────────────

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)", re.MULTILINE)


def _chunk_markdown(paper_id: str, body: str) -> List[dict]:
    """
    Split a Markdown paper body into chunks at heading boundaries.

    Each chunk is:
        chunk_id      = "{paper_id}_s{N}"  (0-based)
        section_title = heading text (or "Introduction" for pre-heading content)
        text          = heading line + all text up to the next heading
    """
    chunks: List[dict] = []

    # Find all heading positions
    headings = [(m.start(), m.group(2).strip()) for m in _HEADING_RE.finditer(body)]

    if not headings:
        # No headings: treat entire body as a single chunk
        text = body.strip()
        if text:
            chunks.append({
                "chunk_id": f"{paper_id}_s0",
                "section_title": "Body",
                "text": text[:4000],  # cap at 4 000 chars to stay within prompt budget
            })
        return chunks

    # Content before first heading (preamble / title block)
    pre = body[: headings[0][0]].strip()
    if pre:
        chunks.append({
            "chunk_id": f"{paper_id}_s0",
            "section_title": "Preamble",
            "text": pre[:4000],
        })
        offset = 1
    else:
        offset = 0

    for idx, (start, title) in enumerate(headings):
        end = headings[idx + 1][0] if idx + 1 < len(headings) else len(body)
        text = body[start:end].strip()
        if not text:
            continue
        chunks.append({
            "chunk_id": f"{paper_id}_s{idx + offset}",
            "section_title": title,
            "text": text[:4000],
        })

    return chunks


# ── Public API ────────────────────────────────────────────────────────────────

def load_paper_file(path: Path) -> Optional[dict]:
    """
    Parse one metabench .md file and return a structured paper dict.
    Returns None if the file cannot be parsed meaningfully.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Cannot read %s: %s", path, exc)
        return None

    meta, body = _parse_frontmatter(raw)

    paper_id = str(meta.get("paper_id") or path.stem)
    title = str(meta.get("title") or "")
    abstract = str(meta.get("abstract") or "")

    # authors: list of dicts or strings
    raw_authors = meta.get("authors") or []
    authors: List[str] = []
    for a in raw_authors:
        if isinstance(a, dict):
            name = a.get("name") or ""
            if name:
                authors.append(str(name))
        elif isinstance(a, str):
            authors.append(a)

    subject_areas_raw = meta.get("subject_areas") or []
    subject_areas = [str(s) for s in subject_areas_raw if s]

    doi: Optional[str] = meta.get("doi")
    if doi:
        doi = str(doi)

    stats = meta.get("stats") or {}
    word_count = int(stats.get("word_count") or 0) if isinstance(stats, dict) else 0

    # Build index text: abstract + title (high-weight retrieval fields)
    index_text = f"{title}\n{abstract}"

    chunks = _chunk_markdown(paper_id, body)

    return {
        "paper_id": paper_id,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "subject_areas": subject_areas,
        "doi": doi,
        "word_count": word_count,
        "index_text": index_text,
        "chunks": chunks,
    }


def load_corpus_dir(corpus_dir: str) -> List[dict]:
    """
    Load all .md files in corpus_dir and return a list of paper dicts.
    Silently skips files that fail to parse.
    """
    papers: List[dict] = []
    p = Path(corpus_dir)
    if not p.is_dir():
        logger.warning("Corpus directory not found: %s", corpus_dir)
        return papers

    for md_file in sorted(p.glob("*.md")):
        paper = load_paper_file(md_file)
        if paper and paper.get("paper_id"):
            papers.append(paper)

    logger.info("Loaded %d papers from %s", len(papers), corpus_dir)
    return papers
