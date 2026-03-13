"""
tests/ml/collaborative/test_svd.py
------------------------------------
Testes para ml/collaborative/svd.py (SVDRecommender).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from ml.collaborative.svd import SVDRecommender

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
    """DataFrame sintetico com escopo de modulo (evita recriar por teste)."""
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
def trained_model(interactions_df: pd.DataFrame) -> SVDRecommender:
    with patch("ml.collaborative.svd.mlflow"):
        model = SVDRecommender(n_factors=5, random_state=42)
        model.fit(interactions_df)
    return model


# ---------------------------------------------------------------------------
# Construcao e fit
# ---------------------------------------------------------------------------


class TestSVDRecommenderFit:
    def test_fit_returns_self(self, interactions_df: pd.DataFrame) -> None:
        with patch("ml.collaborative.svd.mlflow"):
            model = SVDRecommender(n_factors=5)
            result = model.fit(interactions_df)
        assert result is model

    def test_fit_sets_is_fitted(self, interactions_df: pd.DataFrame) -> None:
        with patch("ml.collaborative.svd.mlflow"):
            model = SVDRecommender(n_factors=5)
            assert model._is_fitted is False
            model.fit(interactions_df)
        assert model._is_fitted is True

    def test_fit_populates_indices(self, interactions_df: pd.DataFrame) -> None:
        with patch("ml.collaborative.svd.mlflow"):
            model = SVDRecommender(n_factors=5).fit(interactions_df)
        assert len(model._user_index) > 0
        assert len(model._item_index) > 0

    def test_fit_logs_mlflow_params(self, interactions_df: pd.DataFrame) -> None:
        mock_mlflow = MagicMock()
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        with patch("ml.collaborative.svd.mlflow", mock_mlflow):
            SVDRecommender(n_factors=5).fit(interactions_df)

        mock_mlflow.log_params.assert_called_once()
        call_kwargs = mock_mlflow.log_params.call_args[0][0]
        assert "n_factors" in call_kwargs

    def test_n_factors_capped_to_matrix_rank(self) -> None:
        """n_factors maior que min(n_users, n_items)-1 nao deve falhar."""
        small_df = _make_df(n_users=4, n_items=4, n_rows=20)
        with patch("ml.collaborative.svd.mlflow"):
            model = SVDRecommender(n_factors=100).fit(small_df)
        assert model._is_fitted is True


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------


class TestSVDRecommenderPredict:
    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            SVDRecommender().predict("u0")

    def test_predict_returns_list(self, trained_model: SVDRecommender) -> None:
        result = trained_model.predict("u0", top_k=5)
        assert isinstance(result, list)

    def test_predict_top_k_length(self, trained_model: SVDRecommender) -> None:
        result = trained_model.predict("u0", top_k=5)
        assert len(result) == 5

    def test_predict_item_schema(self, trained_model: SVDRecommender) -> None:
        for item in trained_model.predict("u0", top_k=3):
            assert "product_id" in item
            assert "score" in item
            assert 0.0 <= item["score"] <= 1.0

    def test_predict_scores_descending(self, trained_model: SVDRecommender) -> None:
        recs = trained_model.predict("u0", top_k=10)
        scores = [r["score"] for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_predict_cold_start_no_exception(
        self, trained_model: SVDRecommender
    ) -> None:
        result = trained_model.predict("unknown_user_xyz", top_k=5)
        assert isinstance(result, list)
        assert len(result) <= 5

    def test_predict_cold_start_score_zero(self, trained_model: SVDRecommender) -> None:
        result = trained_model.predict("unknown_user_xyz", top_k=3)
        for item in result:
            assert item["score"] == 0.0

    def test_predict_no_duplicate_items(self, trained_model: SVDRecommender) -> None:
        recs = trained_model.predict("u0", top_k=10)
        ids = [r["product_id"] for r in recs]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------


class TestSVDRecommenderEvaluate:
    def test_evaluate_returns_required_keys(
        self, trained_model: SVDRecommender, interactions_df: pd.DataFrame
    ) -> None:
        metrics = trained_model.evaluate(interactions_df)
        required = {"precision_at_10", "recall_at_10", "ndcg_at_10", "mrr"}
        assert required.issubset(metrics.keys())

    def test_evaluate_values_in_range(
        self, trained_model: SVDRecommender, interactions_df: pd.DataFrame
    ) -> None:
        metrics = trained_model.evaluate(interactions_df)
        for v in metrics.values():
            assert 0.0 <= v <= 1.0

    def test_evaluate_before_fit_raises(self, interactions_df: pd.DataFrame) -> None:
        with pytest.raises(RuntimeError):
            SVDRecommender().evaluate(interactions_df)


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


class TestSVDRecommenderPersistence:
    def test_save_creates_file(
        self, trained_model: SVDRecommender, tmp_path: Path
    ) -> None:
        trained_model.save(tmp_path / "model")
        assert (tmp_path / "model" / "svd.pkl").exists()

    def test_load_returns_fitted_model(
        self, trained_model: SVDRecommender, tmp_path: Path
    ) -> None:
        trained_model.save(tmp_path / "model")
        loaded = SVDRecommender.load(tmp_path / "model")
        assert loaded._is_fitted is True

    def test_load_predict_consistent(
        self, trained_model: SVDRecommender, tmp_path: Path
    ) -> None:
        trained_model.save(tmp_path / "model")
        loaded = SVDRecommender.load(tmp_path / "model")
        original = trained_model.predict("u0", top_k=5)
        restored = loaded.predict("u0", top_k=5)
        assert [r["product_id"] for r in original] == [
            r["product_id"] for r in restored
        ]
