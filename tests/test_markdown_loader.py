import tempfile
import textwrap
import unittest
from pathlib import Path

from src.references.loaders.markdown_loader import load_corpus_dir


class MarkdownLoaderTests(unittest.TestCase):
    def test_load_corpus_dir_skips_placeholder_only_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            good_file = root / "good.md"
            good_file.write_text(
                textwrap.dedent(
                    """
                    ---
                    paper_id: GOOD001
                    title: A usable review
                    abstract: This abstract is present.
                    stats:
                      word_count: 120
                    ---

                    # A usable review

                    ## Introduction
                    This paper contains actual body content for retrieval.
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            placeholder_file = root / "placeholder.md"
            placeholder_file.write_text(
                textwrap.dedent(
                    """
                    ---
                    paper_id: BAD001
                    title: Placeholder paper
                    abstract: ""
                    stats:
                      word_count: 10
                    ---

                    # Placeholder paper

                    ## Abstract
                    Abstract unavailable.

                    ## Evidence Summary
                    No abstract or full text could be retrieved.
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            papers = load_corpus_dir(str(root))

        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0]["paper_id"], "GOOD001")

    def test_load_corpus_dir_reads_nested_markdown_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nested = root / "bucket" / "aa"
            nested.mkdir(parents=True)
            (nested / "nested.md").write_text(
                textwrap.dedent(
                    """
                    ---
                    paper_id: NESTED001
                    title: A nested review
                    abstract: This abstract is present.
                    stats:
                      word_count: 120
                    ---

                    # A nested review

                    ## Introduction
                    This nested paper contains actual body content for retrieval.
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            papers = load_corpus_dir(str(root))

        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0]["paper_id"], "NESTED001")


if __name__ == "__main__":
    unittest.main()
