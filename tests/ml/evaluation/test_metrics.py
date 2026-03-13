"""
tests/ml/evaluation/test_metrics.py
------------------------------------
Testes para ml/evaluation/metrics.py.
"""

from __future__ import annotations

import pytest

from ml.evaluation.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k


class TestPrecisionAtK:
    def test_all_relevant(self) -> None:
        assert precision_at_k(["a", "b", "c"], ["a", "b", "c"], k=3) == pytest.approx(1.0)

    def test_none_relevant(self) -> None:
        assert precision_at_k(["a", "b", "c"], ["x", "y"], k=3) == pytest.approx(0.0)

    def test_partial_hits(self) -> None:
        assert precision_at_k(["a", "b", "c", "d"], ["a", "c"], k=4) == pytest.approx(0.5)

    def test_k_smaller_than_list(self) -> None:
        # only top-2 considered: ["a", "b"] — "a" is relevant → 1/2
        assert precision_at_k(["a", "b", "c"], ["a"], k=2) == pytest.approx(0.5)

    def test_k_zero_returns_zero(self) -> None:
        assert precision_at_k(["a", "b"], ["a"], k=0) == pytest.approx(0.0)

    def test_empty_recommended_returns_zero(self) -> None:
        assert precision_at_k([], ["a"], k=5) == pytest.approx(0.0)

    def test_empty_relevant_gives_zero_hits(self) -> None:
        assert precision_at_k(["a", "b"], [], k=2) == pytest.approx(0.0)

    def test_k_larger_than_list(self) -> None:
        # k=10 but only 2 items: hits=1, score = 1/10
        assert precision_at_k(["a", "x"], ["a"], k=10) == pytest.approx(0.1)


class TestRecallAtK:
    def test_all_relevant_retrieved(self) -> None:
        assert recall_at_k(["a", "b", "c"], ["a", "b", "c"], k=3) == pytest.approx(1.0)

    def test_none_retrieved(self) -> None:
        assert recall_at_k(["a", "b"], ["x", "y"], k=2) == pytest.approx(0.0)

    def test_partial_retrieval(self) -> None:
        # 1 of 2 relevant items in top-3
        assert recall_at_k(["a", "x", "y"], ["a", "b"], k=3) == pytest.approx(0.5)

    def test_k_limits_window(self) -> None:
        # "b" is at position 2 (0-indexed), k=1 → not retrieved
        assert recall_at_k(["a", "b"], ["b"], k=1) == pytest.approx(0.0)

    def test_k_zero_returns_zero(self) -> None:
        assert recall_at_k(["a"], ["a"], k=0) == pytest.approx(0.0)

    def test_empty_relevant_returns_zero(self) -> None:
        assert recall_at_k(["a", "b"], [], k=2) == pytest.approx(0.0)

    def test_empty_recommended_returns_zero(self) -> None:
        assert recall_at_k([], ["a"], k=5) == pytest.approx(0.0)


class TestNdcgAtK:
    def test_perfect_ranking(self) -> None:
        # top item is relevant → dcg = 1/log2(2) = 1.0; ideal = same → ndcg = 1.0
        assert ndcg_at_k(["a", "b", "c"], ["a"], k=3) == pytest.approx(1.0)

    def test_no_relevant_items(self) -> None:
        assert ndcg_at_k(["a", "b"], ["x"], k=2) == pytest.approx(0.0)

    def test_relevant_at_last_position(self) -> None:
        # relevant only at rank 3 (0-indexed 2) → dcg = 1/log2(4); ideal = 1/log2(2)
        result = ndcg_at_k(["x", "y", "a"], ["a"], k=3)
        import math

        expected = (1.0 / math.log2(4)) / (1.0 / math.log2(2))
        assert result == pytest.approx(expected)

    def test_all_relevant_perfect_order(self) -> None:
        assert ndcg_at_k(["a", "b", "c"], ["a", "b", "c"], k=3) == pytest.approx(1.0)

    def test_k_zero_returns_zero(self) -> None:
        assert ndcg_at_k(["a"], ["a"], k=0) == pytest.approx(0.0)

    def test_empty_relevant_returns_zero(self) -> None:
        assert ndcg_at_k(["a", "b"], [], k=2) == pytest.approx(0.0)

    def test_value_between_zero_and_one(self) -> None:
        result = ndcg_at_k(["a", "x", "b", "y"], ["a", "b", "c"], k=4)
        assert 0.0 <= result <= 1.0

    def test_k_caps_relevant_in_ideal(self) -> None:
        # k=1 → ideal considers only 1 relevant item
        assert ndcg_at_k(["a"], ["a", "b", "c"], k=1) == pytest.approx(1.0)


class TestMrr:
    def test_first_item_relevant(self) -> None:
        assert mrr(["a", "b", "c"], ["a"]) == pytest.approx(1.0)

    def test_second_item_relevant(self) -> None:
        assert mrr(["x", "a", "b"], ["a"]) == pytest.approx(0.5)

    def test_third_item_relevant(self) -> None:
        assert mrr(["x", "y", "a"], ["a"]) == pytest.approx(1.0 / 3)

    def test_no_relevant_in_list(self) -> None:
        assert mrr(["x", "y", "z"], ["a"]) == pytest.approx(0.0)

    def test_empty_recommended_returns_zero(self) -> None:
        assert mrr([], ["a"]) == pytest.approx(0.0)

    def test_empty_relevant_returns_zero(self) -> None:
        assert mrr(["a", "b"], []) == pytest.approx(0.0)

    def test_multiple_relevant_uses_first_hit(self) -> None:
        # "b" is at rank 2, "a" is at rank 3 — first hit is "b" → 1/2
        assert mrr(["x", "b", "a"], ["a", "b"]) == pytest.approx(0.5)
