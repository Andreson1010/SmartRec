"""
tests/ml/collaborative/test_knn.py
------------------------------------
Testes para ml/collaborative/knn.py (KNNRecommender).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from ml.collaborative.knn import KNNRecommender

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(n_users: int = 10, n_items: int = 15, n_rows: int = 80) -> pd.DataFrame:
    """DataFrame sintetico determinístico."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in rng.integers(0, n_users, n_rows)],
            "product_id": [f"p{j}" for j in rng.integers(0, n_items, n_rows)],
            "rating": rng.integers(1, 6, n_rows).astype("float32"),
        }
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def interactions_df() -> pd.DataFrame:
    """DataFrame sintetico com escopo de modulo."""
    rng = np.random.default_rng(42)
    n_users, n_products, n_rows = 20, 30, 200
    return pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in rng.integers(0, n_users, n_rows)],
            "product_id": [f"p{i}" for i in rng.integers(0, n_products, n_rows)],
            "rating": rng.integers(1, 6, n_rows).astype("float32"),
            "timestamp": rng.integers(1_600_000_000, 1_700_000_000, n_rows),
        }
    )


@pytest.fixture(scope="module")
def trained_model(interactions_df: pd.DataFrame) -> KNNRecommender:
    with patch("ml.collaborative.knn.mlflow"):
        model = KNNRecommender(k=5)
        model.fit(interactions_df)
    return model


# ---------------------------------------------------------------------------
# Construcao e fit
# ---------------------------------------------------------------------------


class TestKNNRecommenderFit:
    def test_fit_returns_self(self, interactions_df: pd.DataFrame) -> None:
        with patch("ml.collaborative.knn.mlflow"):
            model = KNNRecommender(k=5)
            result = model.fit(interactions_df)
        assert result is model

    def test_fit_sets_is_fitted(self, interactions_df: pd.DataFrame) -> None:
        with patch("ml.collaborative.knn.mlflow"):
            model = KNNRecommender(k=5)
            assert model._is_fitted is False
            model.fit(interactions_df)
        assert model._is_fitted is True

    def test_fit_populates_indices(self, interactions_df: pd.DataFrame) -> None:
        with patch("ml.collaborative.knn.mlflow"):
            model = KNNRecommender(k=5).fit(interactions_df)
        assert len(model._user_index) > 0
        assert len(model._item_index) > 0

    def test_fit_populates_popular_items(self, interactions_df: pd.DataFrame) -> None:
        with patch("ml.collaborative.knn.mlflow"):
            model = KNNRecommender(k=5).fit(interactions_df)
        assert len(model._popular_items) > 0

    def test_fit_builds_sparse_matrix(self, interactions_df: pd.DataFrame) -> None:
        with patch("ml.collaborative.knn.mlflow"):
            model = KNNRecommender(k=5).fit(interactions_df)
        assert model._matrix is not None
        n_users = len(interactions_df["user_id"].unique())
        n_items = len(interactions_df["product_id"].unique())
        assert model._matrix.shape == (n_users, n_items)

    def test_fit_logs_mlflow_params(self, interactions_df: pd.DataFrame) -> None:
        mock_mlflow = MagicMock()
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        with patch("ml.collaborative.knn.mlflow", mock_mlflow):
            KNNRecommender(k=5).fit(interactions_df)

        mock_mlflow.log_params.assert_called_once()
        call_kwargs = mock_mlflow.log_params.call_args[0][0]
        assert "k" in call_kwargs

    def test_fit_small_dataset(self) -> None:
        """k maior que numero de usuarios nao deve falhar."""
        small_df = _make_df(n_users=3, n_items=5, n_rows=15)
        with patch("ml.collaborative.knn.mlflow"):
            model = KNNRecommender(k=100).fit(small_df)
        assert model._is_fitted is True


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------


class TestKNNRecommenderPredict:
    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            KNNRecommender().predict("u0")

    def test_predict_returns_list(self, trained_model: KNNRecommender) -> None:
        result = trained_model.predict("u0", top_k=5)
        assert isinstance(result, list)

    def test_predict_top_k_length(self, trained_model: KNNRecommender) -> None:
        result = trained_model.predict("u0", top_k=5)
        assert len(result) == 5

    def test_predict_item_schema(self, trained_model: KNNRecommender) -> None:
        for item in trained_model.predict("u0", top_k=3):
            assert "product_id" in item
            assert "score" in item
            assert 0.0 <= item["score"] <= 1.0

    def test_predict_scores_descending(self, trained_model: KNNRecommender) -> None:
        recs = trained_model.predict("u0", top_k=10)
        scores = [r["score"] for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_predict_cold_start_no_exception(
        self, trained_model: KNNRecommender
    ) -> None:
        result = trained_model.predict("unknown_user_xyz", top_k=5)
        assert isinstance(result, list)
        assert len(result) <= 5

    def test_predict_cold_start_score_zero(self, trained_model: KNNRecommender) -> None:
        result = trained_model.predict("unknown_user_xyz", top_k=3)
        for item in result:
            assert item["score"] == 0.0

    def test_predict_no_duplicate_items(self, trained_model: KNNRecommender) -> None:
        recs = trained_model.predict("u0", top_k=10)
        ids = [r["product_id"] for r in recs]
        assert len(ids) == len(set(ids))

    def test_predict_excludes_rated_items(self, interactions_df: pd.DataFrame) -> None:
        """Itens ja avaliados pelo usuario nao devem aparecer nas recomendacoes."""
        with patch("ml.collaborative.knn.mlflow"):
            model = KNNRecommender(k=5).fit(interactions_df)

        user_id = "u0"
        rated = set(
            interactions_df[interactions_df["user_id"] == user_id]["product_id"]
        )
        recs = model.predict(user_id, top_k=10)
        rec_ids = {r["product_id"] for r in recs}

        assert rec_ids.isdisjoint(rated)


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------


class TestKNNRecommenderEvaluate:
    def test_evaluate_returns_required_keys(
        self, trained_model: KNNRecommender, interactions_df: pd.DataFrame
    ) -> None:
        metrics = trained_model.evaluate(interactions_df)
        required = {"precision_at_10", "recall_at_10", "ndcg_at_10", "mrr"}
        assert required.issubset(metrics.keys())

    def test_evaluate_values_in_range(
        self, trained_model: KNNRecommender, interactions_df: pd.DataFrame
    ) -> None:
        metrics = trained_model.evaluate(interactions_df)
        for v in metrics.values():
            assert 0.0 <= v <= 1.0

    def test_evaluate_before_fit_raises(self, interactions_df: pd.DataFrame) -> None:
        with pytest.raises(RuntimeError):
            KNNRecommender().evaluate(interactions_df)

    def test_evaluate_empty_relevant_returns_zeros(self) -> None:
        """Sem ratings >= 4 retorna metricas zero sem erros."""
        df = pd.DataFrame(
            {
                "user_id": ["u0", "u1"],
                "product_id": ["p0", "p1"],
                "rating": [1.0, 2.0],
            }
        )
        with patch("ml.collaborative.knn.mlflow"):
            model = KNNRecommender(k=1).fit(df)
        metrics = model.evaluate(df)
        for v in metrics.values():
            assert v == 0.0


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


class TestKNNRecommenderPersistence:
    def test_save_creates_file(
        self, trained_model: KNNRecommender, tmp_path: Path
    ) -> None:
        trained_model.save(tmp_path / "model")
        assert (tmp_path / "model" / "knn.pkl").exists()

    def test_load_returns_fitted_model(
        self, trained_model: KNNRecommender, tmp_path: Path
    ) -> None:
        trained_model.save(tmp_path / "model")
        loaded = KNNRecommender.load(tmp_path / "model")
        assert loaded._is_fitted is True

    def test_load_predict_consistent(
        self, trained_model: KNNRecommender, tmp_path: Path
    ) -> None:
        trained_model.save(tmp_path / "model")
        loaded = KNNRecommender.load(tmp_path / "model")
        original = trained_model.predict("u0", top_k=5)
        restored = loaded.predict("u0", top_k=5)
        assert [r["product_id"] for r in original] == [
            r["product_id"] for r in restored
        ]
