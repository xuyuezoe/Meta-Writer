"""Build a deduplicated merged markdown corpus from multiple source directories."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.references.loaders.markdown_loader import _parse_frontmatter


def _normalize_space(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _normalize_doi(value: str) -> str:
    normalized = _normalize_space(value).casefold()
    normalized = normalized.removeprefix("https://doi.org/")
    normalized = normalized.removeprefix("http://doi.org/")
    normalized = normalized.removeprefix("doi:")
    return normalized.strip()


def _title_key(title: str) -> str:
    return "_".join(re.findall(r"[a-z0-9]+", title.casefold()))


def _iter_markdown_files(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("*.md") if path.is_file())


def _paper_identity(path: Path) -> dict[str, str]:
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    meta, _body = _parse_frontmatter(raw_text)
    meta = meta if isinstance(meta, dict) else {}
    doi = _normalize_doi(str(meta.get("doi") or ""))
    title = _normalize_space(meta.get("title"))
    paper_id = _normalize_space(meta.get("paper_id") or path.stem)
    return {
        "doi": doi,
        "title": title,
        "title_key": _title_key(title) if title else "",
        "paper_id": paper_id,
        "stem": path.stem,
    }


def _dedupe_key(identity: dict[str, str]) -> str:
    if identity["doi"]:
        return f'doi:{identity["doi"]}'
    if identity["title_key"]:
        return f'title:{identity["title_key"]}'
    if identity["paper_id"]:
        return f'paper_id:{identity["paper_id"]}'
    return f'stem:{identity["stem"]}'


def build_deduped_corpus(*, roots: list[Path], output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.glob("*.md"):
        existing.unlink()

    kept: dict[str, dict[str, object]] = {}
    duplicates: list[dict[str, object]] = []
    filename_counts: dict[str, int] = {}

    for root in roots:
        for source_path in _iter_markdown_files(root):
            identity = _paper_identity(source_path)
            dedupe_key = _dedupe_key(identity)
            if dedupe_key in kept:
                duplicates.append(
                    {
                        "dedupe_key": dedupe_key,
                        "kept_from": kept[dedupe_key]["source_path"],
                        "skipped_source_path": str(source_path),
                    }
                )
                continue

            target_name = source_path.name
            if target_name in filename_counts:
                filename_counts[target_name] += 1
                target_name = f"{source_path.stem}__dup{filename_counts[target_name]}{source_path.suffix}"
            else:
                filename_counts[target_name] = 0
            target_path = output_dir / target_name
            shutil.copy2(source_path, target_path)
            kept[dedupe_key] = {
                "source_path": str(source_path),
                "target_path": str(target_path),
                "identity": identity,
            }

    manifest = {
        "source_roots": [str(root) for root in roots],
        "output_dir": str(output_dir),
        "kept_file_count": len(kept),
        "duplicate_count": len(duplicates),
        "kept_items": list(kept.values()),
        "duplicates": duplicates,
    }
    manifest_path = output_dir / "merge_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deduplicated merged markdown corpus")
    parser.add_argument("--root", type=Path, action="append", required=True, help="Source corpus directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Merged output directory")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_deduped_corpus(roots=args.root, output_dir=args.output_dir)
    rendered = json.dumps(manifest, ensure_ascii=False, indent=2)
    try:
        print(rendered)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(rendered.encode("utf-8"))
        sys.stdout.buffer.write(b"\n")


if __name__ == "__main__":
    main()
