"""Build experiment assets from a real review article markdown file."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.tasks import TASK_REGISTRY
from meta_bench.experiments.review_task_builder import build_task_snapshot_from_article
from src.references.loaders.markdown_loader import _parse_frontmatter


REFERENCE_HEADING_RE = re.compile(r"(?mi)^##\s+(references|reference|bibliography)\s*$")
REFERENCE_START_RE = re.compile(r"^\s*(\d+)(?:[.)])?\s+(.*\S)\s*$")
DOI_RE = re.compile(r"\b(?:doi:\s*)?(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b", re.IGNORECASE)
PMID_RE = re.compile(r"\bPMID:\s*([0-9]+)\b", re.IGNORECASE)
PMCID_RE = re.compile(r"\bPMCID:\s*(PMC[0-9]+)\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\((19|20)\d{2}[a-z]?\)")


def _normalize_whitespace(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", normalized).strip()


def _portable_source_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return f"external:{resolved.parent.name}/{resolved.name}"


def _normalize_doi(doi: str) -> str:
    normalized = doi.strip().lower()
    normalized = normalized.removeprefix("https://doi.org/")
    normalized = normalized.removeprefix("http://doi.org/")
    normalized = normalized.removeprefix("doi:")
    return normalized.strip()


def _build_title_key(title: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", title.casefold())
    return "_".join(tokens)


def _extract_references_text(body: str) -> str:
    match = REFERENCE_HEADING_RE.search(body)
    if match is None:
        raise ValueError("No '## References' section found in the article body")
    return body[match.end() :].strip()


def _split_reference_blocks(references_text: str) -> list[tuple[int, str]]:
    blocks: list[tuple[int, str]] = []
    current_number: int | None = None
    current_lines: list[str] = []

    for raw_line in references_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = REFERENCE_START_RE.match(line)
        if match is not None:
            if current_number is not None and current_lines:
                blocks.append((current_number, _normalize_whitespace(" ".join(current_lines))))
            current_number = int(match.group(1))
            current_lines = [match.group(2).strip()]
            continue

        if current_number is not None:
            current_lines.append(line)

    if current_number is not None and current_lines:
        blocks.append((current_number, _normalize_whitespace(" ".join(current_lines))))

    return blocks


def _extract_title_journal_year(reference_text: str) -> tuple[str, str, int | None, str]:
    working = reference_text
    for pattern in (DOI_RE, PMID_RE, PMCID_RE):
        working = pattern.sub("", working)
    working = re.sub(r"\s*,\s*$", "", working).strip()

    year_value: int | None = None
    year_match = YEAR_RE.search(working)
    before_year = working
    after_year = ""
    if year_match is not None:
        year_text = year_match.group(0).strip("()")
        year_digits = re.match(r"(19|20)\d{2}", year_text)
        if year_digits is not None:
            year_value = int(year_digits.group(0))
        before_year = working[: year_match.start()].strip()
        after_year = working[year_match.end() :].strip()

    parts = [part.strip(" .;,") for part in re.split(r"\.\s+", before_year) if part.strip(" .;,")]
    title = parts[1] if len(parts) >= 2 else ""
    journal = parts[2] if len(parts) >= 3 else ""

    if not journal and after_year:
        journal = after_year.split(".")[0].strip(" .;,")

    return title, journal, year_value, parts[0] if parts else ""


def _parse_reference_entry(reference_number: int, reference_text: str) -> dict[str, Any]:
    doi_match = DOI_RE.search(reference_text)
    pmid_match = PMID_RE.search(reference_text)
    pmcid_match = PMCID_RE.search(reference_text)
    title, journal, year_value, authors_text = _extract_title_journal_year(reference_text)

    doi = _normalize_doi(doi_match.group(1)) if doi_match is not None else ""
    title_key = _build_title_key(title)
    reference_key = f"doi:{doi}" if doi else f"title:{title_key}"

    return {
        "reference_number": reference_number,
        "raw_text": reference_text,
        "authors_text": authors_text,
        "title": title,
        "journal": journal,
        "year": year_value,
        "doi": doi,
        "pmid": pmid_match.group(1).strip() if pmid_match is not None else "",
        "pmcid": pmcid_match.group(1).strip() if pmcid_match is not None else "",
        "title_key": title_key,
        "reference_key": reference_key,
    }


def _read_article_summary(summary_csv_path: Path, paper_id: str) -> dict[str, Any]:
    with summary_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("文章编号", "")).strip() != paper_id:
                continue
            return {
                "paper_id": paper_id,
                "file_name": str(row.get("文件名", "")).strip(),
                "word_count_without_references": int(str(row.get("总字数(不含参考文献)", "0")).strip() or 0),
                "word_count_with_references": int(str(row.get("总字数(含参考文献)", "0")).strip() or 0),
                "body_citation_event_count": int(str(row.get("总引用数(不含参考文献)", "0")).strip() or 0),
                "citation_per_word_without_references": float(
                    str(row.get("引用数/字数(不含参考文献)", "0")).strip() or 0.0
                ),
                "max_single_source_mentions": int(str(row.get("单篇被引最多次数", "0")).strip() or 0),
                "max_single_source_mentions_per_word": float(
                    str(row.get("最多次数/总字数(不含参考文献)", "0")).strip() or 0.0
                ),
                "has_reference_section": bool(int(str(row.get("是否有参考文献区", "0")).strip() or 0)),
            }
    raise ValueError(f"Paper id {paper_id} was not found in {summary_csv_path}")


def _build_task_snapshot(task_name: str) -> dict[str, Any]:
    task_factory = TASK_REGISTRY.get(task_name)
    if task_factory is None:
        raise ValueError(f"Unknown task name: {task_name}")
    config = task_factory()
    return {
        "task_name": task_name,
        "task": config.get("task"),
        "constraints": config.get("constraints"),
        "outline": config.get("outline"),
        "session_name": config.get("session_name"),
        "reference": config.get("reference"),
        "corpus_dir": config.get("corpus_dir"),
    }


def _build_task_snapshot_from_article(
    article_path: Path,
    *,
    summary_csv_path: Path | None = None,
) -> dict[str, Any]:
    snapshot = build_task_snapshot_from_article(
        article_path=article_path,
        summary_csv_path=summary_csv_path,
    )
    return dict(snapshot)


def build_experiment_assets(
    *,
    article_path: Path,
    output_dir: Path,
    summary_csv_path: Path | None = None,
    task_name: str | None = None,
    derive_task_snapshot: bool = False,
) -> dict[str, Any]:
    raw_text = article_path.read_text(encoding="utf-8", errors="replace")
    frontmatter, body = _parse_frontmatter(raw_text)

    paper_id = str(frontmatter.get("paper_id") or article_path.stem).strip()
    article_title = str(frontmatter.get("title") or "").strip()
    article_doi = _normalize_doi(str(frontmatter.get("doi") or "").strip())

    references_text = _extract_references_text(body)
    reference_blocks = _split_reference_blocks(references_text)
    references = [
        _parse_reference_entry(reference_number, reference_text)
        for reference_number, reference_text in reference_blocks
    ]

    reference_numbers = [item["reference_number"] for item in references]
    missing_reference_numbers: list[int] = []
    if reference_numbers:
        expected = set(range(min(reference_numbers), max(reference_numbers) + 1))
        missing_reference_numbers = sorted(expected.difference(reference_numbers))

    article_profile: dict[str, Any] = {
        "paper_id": paper_id,
        "title": article_title,
        "doi": article_doi,
        "source_article_path": _portable_source_path(article_path),
        "reference_section_found": True,
        "parsed_reference_count": len(references),
        "first_reference_number": min(reference_numbers) if reference_numbers else None,
        "last_reference_number": max(reference_numbers) if reference_numbers else None,
        "missing_reference_numbers": missing_reference_numbers,
    }

    if summary_csv_path is not None:
        article_profile["article_summary"] = _read_article_summary(summary_csv_path, paper_id)

    output_dir.mkdir(parents=True, exist_ok=True)

    article_profile_path = output_dir / "article_profile.json"
    article_profile_path.write_text(
        json.dumps(article_profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    gold_references_payload = {
        "paper_id": paper_id,
        "title": article_title,
        "doi": article_doi,
        "reference_count": len(references),
        "references": references,
    }
    gold_references_path = output_dir / "gold_references.json"
    gold_references_path.write_text(
        json.dumps(gold_references_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    written_files: dict[str, str] = {
        "article_profile": str(article_profile_path),
        "gold_references": str(gold_references_path),
    }

    if task_name is not None or derive_task_snapshot:
        task_snapshot_path = output_dir / "task_config_snapshot.json"
        if task_name is not None:
            task_snapshot = _build_task_snapshot(task_name)
        else:
            task_snapshot = _build_task_snapshot_from_article(
                article_path,
                summary_csv_path=summary_csv_path,
            )
        task_snapshot_path.write_text(
            json.dumps(task_snapshot, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        written_files["task_config_snapshot"] = str(task_snapshot_path)

    return {
        "paper_id": paper_id,
        "output_dir": str(output_dir.resolve()),
        "written_files": written_files,
        "parsed_reference_count": len(references),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build review-article experiment assets")
    parser.add_argument("--article-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--task-name", type=str)
    parser.add_argument("--derive-task-snapshot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = build_experiment_assets(
        article_path=args.article_path,
        output_dir=args.output_dir,
        summary_csv_path=args.summary_csv,
        task_name=args.task_name,
        derive_task_snapshot=args.derive_task_snapshot,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
