"""
Tests for src/references 模块：reciprocal_rank_fusion、DenseRetriever。

验证范围：
    - RRF 分数计算与排名逻辑（fusion.py 仍保留，供其他场景复用）
    - 模块公开接口导出（__init__.py）
    - DenseRetriever（HyDERetriever 别名）在空语料库上不崩溃
    - Closing 辅助路去重与补充逻辑

注意：HyDEGenerator 已在 Phase 2 删除，其对应测试已移除。
"""
from __future__ import annotations

import unittest
from typing import List, Optional
from unittest.mock import MagicMock


class ReciprocaRankFusionTests(unittest.TestCase):
    """验证 RRF 实现的核心数学性质。"""

    def _rrf(self, *args, **kwargs):
        from src.references.fusion import reciprocal_rank_fusion
        return reciprocal_rank_fusion(*args, **kwargs)

    def test_single_list_preserves_order(self) -> None:
        result = self._rrf([["a", "b", "c"]])
        ids = [r[0] for r in result]
        self.assertEqual(ids, ["a", "b", "c"])

    def test_empty_input_returns_empty(self) -> None:
        result = self._rrf([])
        self.assertEqual(result, [])

    def test_empty_ranked_list_is_skipped(self) -> None:
        result = self._rrf([[], ["x", "y"]])
        ids = [r[0] for r in result]
        self.assertEqual(ids, ["x", "y"])

    def test_symmetric_lists_produce_tied_top_two(self) -> None:
        # ['a','b'] と ['b','a'] は対称：a と b のスコアが等しくなる
        result = self._rrf([["a", "b"], ["b", "a"]])
        scores = {r[0]: r[1] for r in result}
        self.assertAlmostEqual(scores["a"], scores["b"], places=10)

    def test_double_appearance_lifts_item(self) -> None:
        # 'b' が両方のリストの先頭 → 最高スコア
        result = self._rrf([["b", "a"], ["b", "c"]])
        ids = [r[0] for r in result]
        self.assertEqual(ids[0], "b")

    def test_custom_k_changes_scores_not_order(self) -> None:
        from src.references.fusion import reciprocal_rank_fusion
        r60 = reciprocal_rank_fusion([["a", "b"]], k=60)
        r10 = reciprocal_rank_fusion([["a", "b"]], k=10)
        # order stays the same
        self.assertEqual([x[0] for x in r60], [x[0] for x in r10])
        # scores differ
        self.assertNotAlmostEqual(r60[0][1], r10[0][1], places=5)

    def test_scores_are_positive(self) -> None:
        result = self._rrf([["a", "b", "c"], ["c", "b"]])
        for _, score in result:
            self.assertGreater(score, 0.0)

    def test_results_are_sorted_descending(self) -> None:
        result = self._rrf([["a", "b", "c"], ["c", "a", "d"]])
        scores = [s for _, s in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_all_input_ids_appear_in_output(self) -> None:
        result = self._rrf([["a", "b"], ["b", "c", "d"]])
        ids = {r[0] for r in result}
        self.assertEqual(ids, {"a", "b", "c", "d"})


class ReferencesInitExportTests(unittest.TestCase):
    """验证 src/references/__init__.py 的公开导出。

    HyDEGenerator 和 reciprocal_rank_fusion 已从 __init__.py 移除（Phase 2）。
    DenseRetriever 为新主类，HyDERetriever/ReferenceRetriever 为向后兼容别名。
    """

    def test_dense_retriever_exported(self) -> None:
        from src.references import DenseRetriever
        from src.references.retriever import DenseRetriever as Direct
        self.assertIs(DenseRetriever, Direct)

    def test_hyde_retriever_exported(self) -> None:
        from src.references import HyDERetriever
        from src.references.retriever import HyDERetriever as Direct
        self.assertIs(HyDERetriever, Direct)

    def test_reference_retriever_alias_exported(self) -> None:
        from src.references import ReferenceRetriever
        from src.references import HyDERetriever
        self.assertIs(ReferenceRetriever, HyDERetriever)


class HyDERetrieverSmokeTests(unittest.TestCase):
    """验证 HyDERetriever 在空语料库上不崩溃。"""

    def _make_retriever(self):
        from src.references.retriever import HyDERetriever
        mock_corpus = MagicMock()
        mock_corpus.papers = []
        return HyDERetriever(mock_corpus, llm_client=None)

    def test_retrieve_global_returns_empty_index_on_empty_corpus(self) -> None:
        retriever = self._make_retriever()
        index = retriever.retrieve_global("test task")
        from src.references.retriever import GlobalPaperIndex
        self.assertIsInstance(index, GlobalPaperIndex)
        self.assertEqual(index.entries, [])

    def test_rank_for_section_returns_empty_on_empty_index(self) -> None:
        retriever = self._make_retriever()
        from src.references.retriever import GlobalPaperIndex
        empty_index = GlobalPaperIndex(entries=[])
        # section_intent=None は許可されているため、型問題なしにテスト可能
        result = retriever.rank_for_section(
            global_index=empty_index,
            section_title="Introduction",
            section_intent=None,
        )
        self.assertEqual(result, [])


class ClosingPathTests(unittest.TestCase):
    """验证第三路（Closing BM25 辅助路）的去重与补充逻辑。"""

    def _make_chunk(self, paper_id: str, score: float, title: str = "") -> dict:
        return {
            "paper_id": paper_id,
            "chunk_id": f"{paper_id}_0",
            "text": f"chunk text for {paper_id}",
            "title": title or paper_id,
            "score": score,
        }

    def _make_retriever_with_corpus(
        self,
        main_ids: List[str],
        closing_ids: List[str],
        top_closing_extra: int = 8,
    ):
        """构造一个 mock 语料库，控制两路检索结果。"""
        from src.references.retriever import HyDERetriever
        mock_corpus = MagicMock()
        mock_corpus.paper_count.return_value = len(set(main_ids + closing_ids))

        # paper_meta 查询
        def get_paper(pid):
            return {"title": pid, "authors": [], "abstract": f"abstract of {pid}", "doi": ""}
        mock_corpus.get_paper.side_effect = get_paper

        # search 的行为：第一次调用（kw 路）返回 main_ids，
        # 第二次不调用（llm_client=None 跳过 HyDE），
        # 第三次调用（closing 路）返回 closing_ids。
        call_count = [0]
        def search(query, top_k):
            call_count[0] += 1
            if call_count[0] == 1:  # kw 路
                return [self._make_chunk(pid, 1.0 - i * 0.05) for i, pid in enumerate(main_ids)]
            else:  # closing 路（第 2 次，因为 HyDE 被跳过）
                return [self._make_chunk(pid, 0.8 - i * 0.05) for i, pid in enumerate(closing_ids)]
        mock_corpus.search.side_effect = search

        return HyDERetriever(
            mock_corpus,
            llm_client=None,
            top_global=len(main_ids),
            top_closing_extra=top_closing_extra,
        )

    def test_closing_path_appends_new_papers_only(self) -> None:
        """辅助路应只补充主路未覆盖的 paper_id。"""
        main_ids = ["p1", "p2", "p3"]
        closing_ids = ["p2", "p4", "p5"]  # p2 重复，p4/p5 是新的
        retriever = self._make_retriever_with_corpus(main_ids, closing_ids)

        index = retriever.retrieve_global("test task")

        paper_ids = [e.paper_id for e in index.entries]
        # 主路 3 篇 + 辅助路新增 2 篇（p4, p5）
        self.assertEqual(len(paper_ids), 5)
        self.assertIn("p4", paper_ids)
        self.assertIn("p5", paper_ids)
        # p2 不应重复
        self.assertEqual(paper_ids.count("p2"), 1)

    def test_closing_path_r_index_continues_from_main(self) -> None:
        """辅助路 entry 的 r_index 应接续主路（从 len(main)+1 开始）。"""
        main_ids = ["p1", "p2"]
        closing_ids = ["p3", "p4"]
        retriever = self._make_retriever_with_corpus(main_ids, closing_ids)

        index = retriever.retrieve_global("test task")

        # 按 r_index 排序验证
        r_indices = [e.r_index for e in index.entries]
        self.assertEqual(sorted(r_indices), list(range(1, len(r_indices) + 1)))

    def test_closing_path_respects_top_closing_extra(self) -> None:
        """top_closing_extra=1 时辅助路最多补充 1 篇。"""
        main_ids = ["p1"]
        closing_ids = ["p2", "p3", "p4"]  # 3 个新论文
        retriever = self._make_retriever_with_corpus(
            main_ids, closing_ids, top_closing_extra=1
        )

        index = retriever.retrieve_global("test task")
        self.assertEqual(len(index.entries), 2)  # 主路1 + 辅助路1

    def test_closing_path_zero_disables_supplementary(self) -> None:
        """top_closing_extra=0 时完全跳过辅助路，结果数量不变。"""
        main_ids = ["p1", "p2"]
        closing_ids = ["p3", "p4"]
        retriever = self._make_retriever_with_corpus(
            main_ids, closing_ids, top_closing_extra=0
        )

        index = retriever.retrieve_global("test task")
        self.assertEqual(len(index.entries), 2)  # 仅主路

    def test_closing_path_all_duplicates_adds_nothing(self) -> None:
        """辅助路结果全部已在主路中时，不添加任何论文。"""
        main_ids = ["p1", "p2"]
        closing_ids = ["p1", "p2"]  # 全部重复
        retriever = self._make_retriever_with_corpus(main_ids, closing_ids)

        index = retriever.retrieve_global("test task")
        self.assertEqual(len(index.entries), 2)


if __name__ == "__main__":
    unittest.main()
