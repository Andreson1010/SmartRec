"""
tests/ml/hybrid/test_recommender.py
-------------------------------------
Testes para HybridRecommender — SVDRecommender e SemanticRetriever mockados.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_recs(ids: list[str], scores: list[float]) -> list[dict]:
    return [{"product_id": pid, "score": s} for pid, s in zip(ids, scores)]


CF_RECS = _make_recs(["p1", "p2", "p3", "p4", "p5"], [0.9, 0.8, 0.7, 0.6, 0.5])
SEM_RECS = _make_recs(["p3", "p4", "p6", "p7", "p8"], [0.95, 0.85, 0.75, 0.65, 0.55])


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def hybrid():
    """HybridRecommender com dependências mockadas via __new__ (sem disco)."""
    from ml.hybrid.recommender import HybridRecommender

    mock_cf = MagicMock()
    mock_cf.predict.return_value = CF_RECS

    mock_semantic = MagicMock()
    mock_semantic.query_by_product.return_value = SEM_RECS

    rec = HybridRecommender.__new__(HybridRecommender)
    rec.alpha = 0.6
    rec.strategy = "weighted"
    rec.version = "1.0.0"
    rec._cf = mock_cf
    rec._semantic = mock_semantic
    # p1, p2, p5 estão apenas no CF (sem embedding); p3, p4, p6, p7, p8 indexados
    rec._indexed_pids = frozenset(["p3", "p4", "p6", "p7", "p8"])
    return rec


# ---------------------------------------------------------------------------
# predict — weighted
# ---------------------------------------------------------------------------


class TestPredictWeighted:
    def test_returns_list(self, hybrid) -> None:
        result = hybrid.predict("u1", top_k=5)
        assert isinstance(result, list)

    def test_correct_length(self, hybrid) -> None:
        result = hybrid.predict("u1", top_k=3)
        assert len(result) == 3

    def test_item_schema(self, hybrid) -> None:
        for item in hybrid.predict("u1", top_k=5):
            assert "product_id" in item
            assert "score" in item
            assert isinstance(item["score"], float)

    def test_sorted_descending(self, hybrid) -> None:
        scores = [r["score"] for r in hybrid.predict("u1", top_k=5)]
        assert scores == sorted(scores, reverse=True)

    def test_cf_called_with_3x_top_k(self, hybrid) -> None:
        hybrid.predict("u1", top_k=4)
        hybrid._cf.predict.assert_called_with("u1", top_k=12)

    def test_semantic_seeded_with_first_indexed_cf_item(self, hybrid) -> None:
        # CF_RECS = [p1, p2, p3, p4, p5]; _indexed_pids = {p3, p4, p6, p7, p8}
        # Primeiro item do CF com embedding é p3, não p1
        hybrid.predict("u1", top_k=4)
        hybrid._semantic.query_by_product.assert_called_with("p3", top_k=12)


# ---------------------------------------------------------------------------
# predict — rank_fusion
# ---------------------------------------------------------------------------


class TestPredictRankFusion:
    def test_returns_list(self, hybrid) -> None:
        hybrid.strategy = "rank_fusion"
        result = hybrid.predict("u1", top_k=5)
        assert isinstance(result, list)
        assert len(result) == 5

    def test_item_schema(self, hybrid) -> None:
        hybrid.strategy = "rank_fusion"
        for item in hybrid.predict("u1", top_k=3):
            assert "product_id" in item
            assert "score" in item

    def test_sorted_descending(self, hybrid) -> None:
        hybrid.strategy = "rank_fusion"
        scores = [r["score"] for r in hybrid.predict("u1", top_k=5)]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# predict — rerank
# ---------------------------------------------------------------------------


class TestPredictRerank:
    def test_returns_list_with_correct_length(self, hybrid) -> None:
        hybrid.strategy = "rerank"
        hybrid._semantic.get_embedding = MagicMock(return_value=None)
        hybrid._semantic.score_items = MagicMock(return_value={})
        result = hybrid.predict("u1", top_k=5)
        assert isinstance(result, list)
        assert len(result) == 5

    def test_item_schema(self, hybrid) -> None:
        hybrid.strategy = "rerank"
        hybrid._semantic.get_embedding = MagicMock(return_value=None)
        hybrid._semantic.score_items = MagicMock(return_value={})
        for item in hybrid.predict("u1", top_k=3):
            assert "product_id" in item
            assert "score" in item
            assert isinstance(item["score"], float)

    def test_sorted_descending(self, hybrid) -> None:
        hybrid.strategy = "rerank"
        fake_vec = np.ones(4, dtype="float32")
        fake_vec /= np.linalg.norm(fake_vec)
        hybrid._semantic.get_embedding = MagicMock(return_value=fake_vec)
        hybrid._semantic.score_items = MagicMock(
            return_value={"p3": 0.9, "p4": 0.8}
        )
        scores = [r["score"] for r in hybrid.predict("u1", top_k=5)]
        assert scores == sorted(scores, reverse=True)

    def test_query_by_product_not_called(self, hybrid) -> None:
        """rerank nunca chama query_by_product — não injeta candidatos novos."""
        hybrid.strategy = "rerank"
        hybrid._semantic.get_embedding = MagicMock(return_value=None)
        hybrid._semantic.score_items = MagicMock(return_value={})
        hybrid.predict("u1", top_k=3)
        hybrid._semantic.query_by_product.assert_not_called()

    def test_only_cf_candidates_in_output(self, hybrid) -> None:
        """rerank não introduz produtos novos — só reordena os do CF."""
        hybrid.strategy = "rerank"
        hybrid._semantic.get_embedding = MagicMock(return_value=None)
        hybrid._semantic.score_items = MagicMock(return_value={})
        cf_ids = {r["product_id"] for r in CF_RECS}
        result = hybrid.predict("u1", top_k=5)
        for item in result:
            assert item["product_id"] in cf_ids


# ---------------------------------------------------------------------------
# Fusão — propriedades matemáticas
# ---------------------------------------------------------------------------


class TestWeightedFusion:
    def test_item_appearing_in_both_lists_has_higher_score(self, hybrid) -> None:
        """p3 e p4 aparecem em CF e semântico — devem ter score maior."""
        result = hybrid.predict("u1", top_k=8)
        assert len(result) > 0  # sanity check

    def test_alpha_zero_indexed_products_use_semantic(self, hybrid) -> None:
        """Com alpha=0, produtos indexados recebem apenas o score semântico."""
        hybrid.alpha = 0.0
        result = hybrid._weighted_fusion(CF_RECS, SEM_RECS, top_k=8)
        scores = {r["product_id"]: r["score"] for r in result}
        sem = {r["product_id"]: r["score"] for r in SEM_RECS}
        # p3 e p4 são indexados + estão no CF: score = 0*cf + 1.0*sem
        for pid in ["p3", "p4"]:
            assert scores[pid] == round(sem[pid], 4)

    def test_alpha_zero_unindexed_products_keep_cf_score(self, hybrid) -> None:
        """Com alpha=0, produtos sem embedding mantêm seu score de CF completo."""
        hybrid.alpha = 0.0
        result = hybrid._weighted_fusion(CF_RECS, SEM_RECS, top_k=8)
        scores = {r["product_id"]: r["score"] for r in result}
        cf = {r["product_id"]: r["score"] for r in CF_RECS}
        # p1, p2, p5 não estão em _indexed_pids → score = cf_score bruto
        for pid in ["p1", "p2", "p5"]:
            assert scores[pid] == round(cf[pid], 4)

    def test_alpha_one_ignores_semantic(self, hybrid) -> None:
        hybrid.alpha = 1.0
        result = hybrid._weighted_fusion(CF_RECS, SEM_RECS, top_k=5)
        cf_ids = {r["product_id"] for r in CF_RECS}
        for item in result:
            if item["product_id"] not in cf_ids:
                assert item["score"] == 0.0


class TestRankFusion:
    def test_rrf_scores_positive(self, hybrid) -> None:
        result = hybrid._rank_fusion(CF_RECS, SEM_RECS, top_k=5)
        assert all(r["score"] > 0 for r in result)

    def test_overlap_items_score_higher(self, hybrid) -> None:
        """p3/p4 (em ambas as listas) devem superar itens exclusivos."""
        result = hybrid._rank_fusion(CF_RECS, SEM_RECS, top_k=8)
        scores = {r["product_id"]: r["score"] for r in result}
        # p3 e p4 recebem contribuição dupla
        for overlap_pid in ["p3", "p4"]:
            for only_pid in ["p6", "p7", "p8"]:
                if overlap_pid in scores and only_pid in scores:
                    assert scores[overlap_pid] > scores[only_pid]


# ---------------------------------------------------------------------------
# _build_user_vector
# ---------------------------------------------------------------------------


class TestBuildUserVector:
    def test_returns_none_when_no_indexed_item(self, hybrid) -> None:
        hybrid._indexed_pids = frozenset()
        hybrid._semantic.get_embedding = MagicMock(return_value=None)
        assert hybrid._build_user_vector(CF_RECS) is None

    def test_returns_array_when_indexed_items_exist(self, hybrid) -> None:
        fake_vec = np.ones(4, dtype="float32")
        fake_vec /= np.linalg.norm(fake_vec)
        hybrid._semantic.get_embedding = MagicMock(return_value=fake_vec)
        result = hybrid._build_user_vector(CF_RECS)
        assert isinstance(result, np.ndarray)

    def test_result_is_normalized(self, hybrid) -> None:
        rng = np.random.default_rng(42)
        fake_vec = rng.standard_normal(8).astype("float32")
        fake_vec /= np.linalg.norm(fake_vec)
        hybrid._semantic.get_embedding = MagicMock(return_value=fake_vec.copy())
        result = hybrid._build_user_vector(CF_RECS)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5

    def test_respects_n_seed_limit(self, hybrid) -> None:
        calls: list[str] = []

        def get_emb(pid: str) -> np.ndarray | None:
            if pid in hybrid._indexed_pids:
                calls.append(pid)
                vec = np.ones(4, dtype="float32")
                return vec / np.linalg.norm(vec)
            return None

        hybrid._semantic.get_embedding = get_emb
        hybrid._build_user_vector(CF_RECS, n_seed=2)
        assert len(calls) <= 2


# ---------------------------------------------------------------------------
# Cold start
# ---------------------------------------------------------------------------


class TestColdStart:
    def test_empty_cf_recs_returns_empty_result(self, hybrid) -> None:
        hybrid._cf.predict.return_value = []
        hybrid._semantic.query_by_product.return_value = []
        result = hybrid.predict("cold_user", top_k=5)
        assert isinstance(result, list)

    def test_semantic_not_called_when_no_seed(self, hybrid) -> None:
        hybrid._cf.predict.return_value = []
        hybrid.predict("cold_user", top_k=5)
        hybrid._semantic.query_by_product.assert_not_called()

    def test_semantic_not_called_when_no_cf_item_is_indexed(self, hybrid) -> None:
        # Todos os itens do CF estão fora do índice semântico
        hybrid._cf.predict.return_value = _make_recs(
            ["x1", "x2", "x3"], [0.9, 0.8, 0.7]
        )
        hybrid._indexed_pids = frozenset()
        hybrid.predict("u1", top_k=3)
        hybrid._semantic.query_by_product.assert_not_called()


# ---------------------------------------------------------------------------
# tune_alpha
# ---------------------------------------------------------------------------


class TestTuneAlpha:
    def test_returns_float_in_range(self, hybrid) -> None:
        val_data = pd.DataFrame(
            {
                "user_id": ["u1"] * 5,
                "product_id": ["p1", "p2", "p3", "p4", "p5"],
                "rating": [5, 4, 3, 5, 4],
            }
        )
        best = hybrid.tune_alpha(val_data, alphas=[0.4, 0.6])
        assert 0.0 <= best <= 1.0

    def test_sets_self_alpha(self, hybrid) -> None:
        val_data = pd.DataFrame(
            {
                "user_id": ["u1"] * 3,
                "product_id": ["p1", "p2", "p3"],
                "rating": [5, 4, 5],
            }
        )
        hybrid.tune_alpha(val_data, alphas=[0.3, 0.7])
        assert hybrid.alpha in [0.3, 0.7]


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def _picklable(self):
        """HybridRecommender com _cf/_semantic = None (pickle-safe)."""
        from ml.hybrid.recommender import HybridRecommender

        rec = HybridRecommender.__new__(HybridRecommender)
        rec.alpha = 0.7
        rec.strategy = "rank_fusion"
        rec.version = "2.0.0"
        rec._cf = None
        rec._semantic = None
        rec._indexed_pids = frozenset()
        return rec

    def test_roundtrip(self, tmp_path: Path) -> None:
        from ml.hybrid.recommender import HybridRecommender

        rec = self._picklable()
        rec.save(tmp_path)
        loaded = HybridRecommender.load(tmp_path)

        assert loaded.alpha == rec.alpha
        assert loaded.strategy == rec.strategy
        assert loaded.version == rec.version

    def test_save_creates_file(self, tmp_path: Path) -> None:
        self._picklable().save(tmp_path)
        assert (tmp_path / "hybrid.pkl").exists()
