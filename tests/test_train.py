"""
tests/test_train.py
-------------------
Testes do pipeline de treinamento (scripts/train.py).

Todas as dependências externas (MLflow, modelos ML, I/O de disco) são
mockadas — nenhum dado real ou artefato é lido/escrito.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from scripts.train import (
    _maybe_register,
    evaluate_hybrid,
    temporal_split,
    train,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def interactions_df() -> pd.DataFrame:
    """200 interações sintéticas com timestamp crescente."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in rng.integers(0, 20, n)],
            "product_id": [f"p{i}" for i in rng.integers(0, 30, n)],
            "rating": rng.integers(1, 6, n).astype("float32"),
            "timestamp": np.sort(rng.integers(1_600_000_000, 1_700_000_000, n)),
        }
    )


@pytest.fixture()
def products_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_id": [f"p{i}" for i in range(30)],
            "title": [f"Product {i}" for i in range(30)],
            "description": [f"Desc {i}" for i in range(30)],
        }
    )


def _make_mock_mlflow() -> MagicMock:
    """Cria um mock completo do módulo mlflow com contexto de run."""
    mock_mlflow = MagicMock()
    mock_run = MagicMock()
    mock_run.__enter__ = MagicMock(return_value=mock_run)
    mock_run.__exit__ = MagicMock(return_value=False)
    mock_run.info.run_id = "test-run-id"
    mock_mlflow.start_run.return_value = mock_run
    return mock_mlflow


# ---------------------------------------------------------------------------
# temporal_split
# ---------------------------------------------------------------------------


class TestTemporalSplit:
    def test_sizes_sum_to_total(self, interactions_df: pd.DataFrame) -> None:
        train, val, test = temporal_split(interactions_df)
        assert len(train) + len(val) + len(test) == len(interactions_df)

    def test_default_ratios(self, interactions_df: pd.DataFrame) -> None:
        train, val, test = temporal_split(interactions_df)
        n = len(interactions_df)
        assert len(train) == pytest.approx(n * 0.70, abs=1)
        assert len(val) == pytest.approx(n * 0.15, abs=1)

    def test_no_timestamp_overlap(self, interactions_df: pd.DataFrame) -> None:
        train, val, test = temporal_split(interactions_df)
        assert train["timestamp"].max() <= val["timestamp"].min()
        assert val["timestamp"].max() <= test["timestamp"].min()

    def test_chronological_order(self, interactions_df: pd.DataFrame) -> None:
        train, val, test = temporal_split(interactions_df)
        for df in (train, val, test):
            assert df["timestamp"].is_monotonic_increasing

    def test_custom_ratios(self, interactions_df: pd.DataFrame) -> None:
        train, val, test = temporal_split(
            interactions_df, train_ratio=0.6, val_ratio=0.2
        )
        n = len(interactions_df)
        assert len(train) == pytest.approx(n * 0.6, abs=1)
        assert len(val) == pytest.approx(n * 0.2, abs=1)

    def test_empty_dataframe(self) -> None:
        empty = pd.DataFrame(
            columns=["user_id", "product_id", "rating", "timestamp"]
        )
        train, val, test = temporal_split(empty)
        assert len(train) == 0
        assert len(val) == 0
        assert len(test) == 0


# ---------------------------------------------------------------------------
# evaluate_hybrid
# ---------------------------------------------------------------------------


class TestEvaluateHybrid:
    def _make_model(self, recs: list[dict]) -> MagicMock:
        model = MagicMock()
        model.predict.return_value = recs
        return model

    def test_returns_four_metrics(self, interactions_df: pd.DataFrame) -> None:
        model = self._make_model(
            [{"product_id": "p0", "score": 0.9}, {"product_id": "p1", "score": 0.8}]
        )
        metrics = evaluate_hybrid(model, interactions_df, k=10)
        assert set(metrics) == {
            "precision_at_10", "recall_at_10", "ndcg_at_10", "mrr"
        }

    def test_metrics_in_range(self, interactions_df: pd.DataFrame) -> None:
        model = self._make_model(
            [{"product_id": f"p{i}", "score": 1.0 - i * 0.1} for i in range(10)]
        )
        metrics = evaluate_hybrid(model, interactions_df, k=10)
        for v in metrics.values():
            assert 0.0 <= v <= 1.0

    def test_no_relevant_items_returns_zeros(self) -> None:
        # Todos os ratings abaixo de 4 — sem itens relevantes
        df = pd.DataFrame(
            {
                "user_id": ["u0", "u1"],
                "product_id": ["p0", "p1"],
                "rating": [2.0, 3.0],
                "timestamp": [1, 2],
            }
        )
        model = self._make_model([{"product_id": "p0", "score": 0.9}])
        metrics = evaluate_hybrid(model, df, k=10)
        assert all(v == 0.0 for v in metrics.values())

    def test_perfect_recommendations(self) -> None:
        df = pd.DataFrame(
            {
                "user_id": ["u0"],
                "product_id": ["p0"],
                "rating": [5.0],
                "timestamp": [1],
            }
        )
        model = self._make_model([{"product_id": "p0", "score": 1.0}])
        metrics = evaluate_hybrid(model, df, k=10)
        assert metrics["precision_at_10"] > 0.0
        assert metrics["recall_at_10"] == 1.0
        assert metrics["ndcg_at_10"] > 0.0
        assert metrics["mrr"] == 1.0


# ---------------------------------------------------------------------------
# _maybe_register
# ---------------------------------------------------------------------------


class TestMaybeRegister:
    def test_registers_when_above_threshold(self) -> None:
        mock_mlflow = _make_mock_mlflow()
        with patch("scripts.train.mlflow", mock_mlflow):
            _maybe_register(
                "run-123",
                {"ndcg_at_10": 0.05},
                {"ndcg_at_10": 0.01},
            )
        mock_mlflow.register_model.assert_called_once()

    def test_does_not_register_when_below_threshold(self) -> None:
        mock_mlflow = _make_mock_mlflow()
        with patch("scripts.train.mlflow", mock_mlflow):
            _maybe_register(
                "run-123",
                {"ndcg_at_10": 0.005},
                {"ndcg_at_10": 0.01},
            )
        mock_mlflow.register_model.assert_not_called()

    def test_registers_with_correct_model_uri(self) -> None:
        mock_mlflow = _make_mock_mlflow()
        with patch("scripts.train.mlflow", mock_mlflow):
            _maybe_register(
                "abc-run",
                {"ndcg_at_10": 1.0},
                {"ndcg_at_10": 0.0},
            )
        call_kwargs = str(mock_mlflow.register_model.call_args)
        assert "abc-run" in call_kwargs
        assert "smartrec-hybrid" in call_kwargs


# ---------------------------------------------------------------------------
# train() — integração com mocks
# ---------------------------------------------------------------------------


class TestTrain:
    """Testa o pipeline train() com todos os componentes externos mockados."""

    _BASE_METRICS = {
        "precision_at_10": 0.1,
        "recall_at_10": 0.08,
        "ndcg_at_10": 0.05,
        "mrr": 0.12,
    }

    def _build_mocks(self, interactions_df: pd.DataFrame, products_df: pd.DataFrame):
        mock_mlflow = _make_mock_mlflow()

        mock_svd = MagicMock()
        mock_svd.fit.return_value = mock_svd
        mock_svd.evaluate.return_value = self._BASE_METRICS

        mock_hybrid = MagicMock()
        mock_hybrid.tune_alpha.return_value = 0.6
        mock_hybrid.predict.return_value = []

        mock_embedder = MagicMock()
        mock_embedder.fit_transform.return_value = np.zeros((30, 384), dtype="float32")

        return mock_mlflow, mock_svd, mock_hybrid, mock_embedder

    def test_returns_four_metrics(
        self, interactions_df: pd.DataFrame, products_df: pd.DataFrame
    ) -> None:
        mock_mlflow, mock_svd, mock_hybrid, mock_embedder = self._build_mocks(
            interactions_df, products_df
        )
        with patch("scripts.train.pd.read_parquet", side_effect=[interactions_df, products_df]), \
             patch("scripts.train.mlflow", mock_mlflow), \
             patch("scripts.train.SVDRecommender", return_value=mock_svd), \
             patch("scripts.train.HybridRecommender", return_value=mock_hybrid), \
             patch("scripts.train.ProductEmbedder", return_value=mock_embedder), \
             patch("scripts.train.HYBRID_ARTIFACTS", MagicMock()):
            metrics = train(n_factors=10, register=False)

        assert set(metrics) == {"precision_at_10", "recall_at_10", "ndcg_at_10", "mrr"}

    def test_svd_fit_called_on_train_split(
        self, interactions_df: pd.DataFrame, products_df: pd.DataFrame
    ) -> None:
        mock_mlflow, mock_svd, mock_hybrid, mock_embedder = self._build_mocks(
            interactions_df, products_df
        )
        with patch("scripts.train.pd.read_parquet", side_effect=[interactions_df, products_df]), \
             patch("scripts.train.mlflow", mock_mlflow), \
             patch("scripts.train.SVDRecommender", return_value=mock_svd), \
             patch("scripts.train.HybridRecommender", return_value=mock_hybrid), \
             patch("scripts.train.ProductEmbedder", return_value=mock_embedder), \
             patch("scripts.train.HYBRID_ARTIFACTS", MagicMock()):
            train(n_factors=10, register=False)

        mock_svd.fit.assert_called_once()
        fit_df = mock_svd.fit.call_args[0][0]
        # Train split é 70% — deve ser menor que o total
        assert len(fit_df) < len(interactions_df)

    def test_skip_embeddings_skips_embedder(
        self, interactions_df: pd.DataFrame, products_df: pd.DataFrame
    ) -> None:
        mock_mlflow, mock_svd, mock_hybrid, _ = self._build_mocks(
            interactions_df, products_df
        )
        mock_embedder_cls = MagicMock()
        with patch("scripts.train.pd.read_parquet", side_effect=[interactions_df, products_df]), \
             patch("scripts.train.mlflow", mock_mlflow), \
             patch("scripts.train.SVDRecommender", return_value=mock_svd), \
             patch("scripts.train.HybridRecommender", return_value=mock_hybrid), \
             patch("scripts.train.ProductEmbedder", mock_embedder_cls), \
             patch("scripts.train.HYBRID_ARTIFACTS", MagicMock()):
            train(skip_embeddings=True, register=False)

        mock_embedder_cls.assert_not_called()

    def test_mlflow_experiment_set(
        self, interactions_df: pd.DataFrame, products_df: pd.DataFrame
    ) -> None:
        mock_mlflow, mock_svd, mock_hybrid, mock_embedder = self._build_mocks(
            interactions_df, products_df
        )
        with patch("scripts.train.pd.read_parquet", side_effect=[interactions_df, products_df]), \
             patch("scripts.train.mlflow", mock_mlflow), \
             patch("scripts.train.SVDRecommender", return_value=mock_svd), \
             patch("scripts.train.HybridRecommender", return_value=mock_hybrid), \
             patch("scripts.train.ProductEmbedder", return_value=mock_embedder), \
             patch("scripts.train.HYBRID_ARTIFACTS", MagicMock()):
            train(experiment_name="smartrec/test", register=False)

        mock_mlflow.set_experiment.assert_called_once_with("smartrec/test")

    def test_register_called_when_enabled(
        self, interactions_df: pd.DataFrame, products_df: pd.DataFrame
    ) -> None:
        mock_mlflow, mock_svd, mock_hybrid, mock_embedder = self._build_mocks(
            interactions_df, products_df
        )
        # Forçar métricas acima do threshold (ndcg_at_10=0.05 > 0.01)
        mock_eval = MagicMock(return_value=self._BASE_METRICS)
        with patch("scripts.train.pd.read_parquet", side_effect=[interactions_df, products_df]), \
             patch("scripts.train.mlflow", mock_mlflow), \
             patch("scripts.train.SVDRecommender", return_value=mock_svd), \
             patch("scripts.train.HybridRecommender", return_value=mock_hybrid), \
             patch("scripts.train.ProductEmbedder", return_value=mock_embedder), \
             patch("scripts.train.evaluate_hybrid", mock_eval), \
             patch("scripts.train.HYBRID_ARTIFACTS", MagicMock()):
            train(register=True)

        mock_mlflow.register_model.assert_called_once()
