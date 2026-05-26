"""Build six-slot section priors from the 300-review reference corpus."""

from __future__ import annotations

import json
from pathlib import Path

from meta_bench.six_slot_priors import (
    SIX_SLOT_CITATION_PRIORS_PATH,
    SIX_SLOT_PRIORS_PATH,
    derive_six_slot_citation_priors_from_corpus,
    derive_six_slot_priors_from_corpus,
)


DEFAULT_CORPUS_DIR: Path | None = None


def build_priors(corpus_dir: Path) -> None:
    section_payload = derive_six_slot_priors_from_corpus(corpus_dir)
    SIX_SLOT_PRIORS_PATH.write_text(
        json.dumps(section_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    citation_payload = derive_six_slot_citation_priors_from_corpus(corpus_dir)
    SIX_SLOT_CITATION_PRIORS_PATH.write_text(
        json.dumps(citation_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build six-slot section and citation priors")
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR, required=DEFAULT_CORPUS_DIR is None)
    args = parser.parse_args()
    build_priors(args.corpus_dir)


if __name__ == "__main__":
    main()
