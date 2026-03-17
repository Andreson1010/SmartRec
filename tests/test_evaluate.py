"""
tests/test_evaluate.py
-----------------------
Testes do pipeline de avaliação comparativa (scripts/evaluate.py).

Todas as dependências externas (MLflow, modelos ML, I/O de disco) são
mockadas — nenhum dado real ou artefato é lido/escrito.
"""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate import (
    _print_table,
    compute_metrics,
    evaluate,
    temporal_split,
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


def _build_mocks(interactions_df: pd.DataFrame, products_df: pd.DataFrame):
    mock_mlflow = _make_mock_mlflow()
    mock_svd = MagicMock()
    mock_svd.fit.return_value = mock_svd
    mock_svd.predict.return_value = []
    mock_knn = MagicMock()
    mock_knn.fit.return_value = mock_knn
    mock_knn.predict.return_value = []
    mock_hybrid = MagicMock()
    mock_hybrid.tune_alpha.return_value = 0.6
    mock_hybrid.predict.return_value = []
    mock_embedder = MagicMock()
    mock_embedder.fit_transform.return_value = np.zeros((30, 384), dtype="float32")
    return mock_mlflow, mock_svd, mock_knn, mock_hybrid, mock_embedder


def _apply_patches(
    interactions_df: pd.DataFrame,
    products_df: pd.DataFrame,
    mock_mlflow: MagicMock,
    mock_svd: MagicMock,
    mock_knn: MagicMock,
    mock_hybrid: MagicMock,
    mock_embedder: MagicMock,
    stack: ExitStack,
) -> None:
    """Registra todos os patches no ExitStack fornecido."""
    stack.enter_context(
        patch(
            "scripts.evaluate.pd.read_parquet",
            side_effect=[interactions_df, products_df],
        )
    )
    stack.enter_context(patch("scripts.evaluate.mlflow", mock_mlflow))
    stack.enter_context(patch("scripts.evaluate.SVDRecommender", return_value=mock_svd))
    stack.enter_context(patch("scripts.evaluate.KNNRecommender", return_value=mock_knn))
    stack.enter_context(
        patch("scripts.evaluate.HybridRecommender", return_value=mock_hybrid)
    )
    stack.enter_context(
        patch("scripts.evaluate.ProductEmbedder", return_value=mock_embedder)
    )
    stack.enter_context(patch("scripts.evaluate.HYBRID_ARTIFACTS", MagicMock()))


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

    def test_empty_dataframe(self) -> None:
        empty = pd.DataFrame(columns=["user_id", "product_id", "rating", "timestamp"])
        train, val, test = temporal_split(empty)
        assert len(train) == 0 and len(val) == 0 and len(test) == 0


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def _make_model(self, recs: list[dict]) -> MagicMock:
        model = MagicMock()
        model.predict.return_value = recs
        return model

    def test_returns_four_keys(self, interactions_df: pd.DataFrame) -> None:
        model = self._make_model(
            [{"product_id": "p0", "score": 0.9}, {"product_id": "p1", "score": 0.8}]
        )
        metrics = compute_metrics(model, interactions_df)
        assert set(metrics) == {
            "precision_at_10",
            "recall_at_10",
            "ndcg_at_10",
            "mrr",
        }

    def test_metrics_in_range(self, interactions_df: pd.DataFrame) -> None:
        model = self._make_model(
            [{"product_id": f"p{i}", "score": 1.0 - i * 0.1} for i in range(10)]
        )
        metrics = compute_metrics(model, interactions_df)
        for v in metrics.values():
            assert 0.0 <= v <= 1.0

    def test_no_relevant_items_returns_zeros(self) -> None:
        df = pd.DataFrame(
            {
                "user_id": ["u0", "u1"],
                "product_id": ["p0", "p1"],
                "rating": [2.0, 3.0],
                "timestamp": [1, 2],
            }
        )
        model = self._make_model([{"product_id": "p0", "score": 0.9}])
        metrics = compute_metrics(model, df)
        assert all(v == 0.0 for v in metrics.values())

    def test_perfect_recommendation(self) -> None:
        df = pd.DataFrame(
            {
                "user_id": ["u0"],
                "product_id": ["p0"],
                "rating": [5.0],
                "timestamp": [1],
            }
        )
        model = self._make_model([{"product_id": "p0", "score": 1.0}])
        metrics = compute_metrics(model, df)
        assert metrics["recall_at_10"] == 1.0
        assert metrics["mrr"] == 1.0


# ---------------------------------------------------------------------------
# _print_table
# ---------------------------------------------------------------------------


class TestPrintTable:
    def test_runs_without_error(self, capsys: pytest.CaptureFixture) -> None:
        results = {
            "svd": {
                "precision_at_10": 0.1,
                "recall_at_10": 0.08,
                "ndcg_at_10": 0.05,
                "mrr": 0.12,
            },
            "knn": {
                "precision_at_10": 0.09,
                "recall_at_10": 0.07,
                "ndcg_at_10": 0.04,
                "mrr": 0.11,
            },
        }
        _print_table(results)
        captured = capsys.readouterr()
        assert "svd" in captured.out
        assert "knn" in captured.out

    def test_contains_metric_names(self, capsys: pytest.CaptureFixture) -> None:
        _print_table(
            {
                "model": {
                    "precision_at_10": 0.1,
                    "recall_at_10": 0.08,
                    "ndcg_at_10": 0.05,
                    "mrr": 0.12,
                }
            }
        )
        captured = capsys.readouterr()
        for metric in ["precision_at_10", "recall_at_10", "ndcg_at_10", "mrr"]:
            assert metric in captured.out


# ---------------------------------------------------------------------------
# evaluate() — integração com mocks
# ---------------------------------------------------------------------------


class TestEvaluate:
    """Testa o pipeline evaluate() com todos os componentes externos mockados."""

    def test_returns_three_model_keys(
        self, interactions_df: pd.DataFrame, products_df: pd.DataFrame
    ) -> None:
        mocks = _build_mocks(interactions_df, products_df)
        with ExitStack() as stack:
            _apply_patches(interactions_df, products_df, *mocks, stack=stack)
            results = evaluate(n_factors=5, knn_k=3)
        assert set(results) == {"svd", "knn", "hybrid"}

    def test_each_result_has_four_metrics(
        self, interactions_df: pd.DataFrame, products_df: pd.DataFrame
    ) -> None:
        mocks = _build_mocks(interactions_df, products_df)
        with ExitStack() as stack:
            _apply_patches(interactions_df, products_df, *mocks, stack=stack)
            results = evaluate(n_factors=5, knn_k=3)
        expected = {"precision_at_10", "recall_at_10", "ndcg_at_10", "mrr"}
        for model_metrics in results.values():
            assert set(model_metrics) == expected

    def test_svd_fit_called_once(
        self, interactions_df: pd.DataFrame, products_df: pd.DataFrame
    ) -> None:
        mocks = _build_mocks(interactions_df, products_df)
        with ExitStack() as stack:
            _apply_patches(interactions_df, products_df, *mocks, stack=stack)
            evaluate(n_factors=5, knn_k=3)
        mocks[1].fit.assert_called_once()

    def test_knn_fit_called_once(
        self, interactions_df: pd.DataFrame, products_df: pd.DataFrame
    ) -> None:
        mocks = _build_mocks(interactions_df, products_df)
        with ExitStack() as stack:
            _apply_patches(interactions_df, products_df, *mocks, stack=stack)
            evaluate(n_factors=5, knn_k=3)
        mocks[2].fit.assert_called_once()

    def test_skip_embeddings_skips_embedder(
        self, interactions_df: pd.DataFrame, products_df: pd.DataFrame
    ) -> None:
        mocks = _build_mocks(interactions_df, products_df)
        mock_embedder_cls = MagicMock()
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "scripts.evaluate.pd.read_parquet",
                    side_effect=[interactions_df, products_df],
                )
            )
            stack.enter_context(patch("scripts.evaluate.mlflow", mocks[0]))
            stack.enter_context(
                patch("scripts.evaluate.SVDRecommender", return_value=mocks[1])
            )
            stack.enter_context(
                patch("scripts.evaluate.KNNRecommender", return_value=mocks[2])
            )
            stack.enter_context(
                patch("scripts.evaluate.HybridRecommender", return_value=mocks[3])
            )
            stack.enter_context(
                patch("scripts.evaluate.ProductEmbedder", mock_embedder_cls)
            )
            stack.enter_context(patch("scripts.evaluate.HYBRID_ARTIFACTS", MagicMock()))
            evaluate(n_factors=5, knn_k=3, skip_embeddings=True)
        mock_embedder_cls.assert_not_called()

    def test_mlflow_experiment_set(
        self, interactions_df: pd.DataFrame, products_df: pd.DataFrame
    ) -> None:
        mocks = _build_mocks(interactions_df, products_df)
        with ExitStack() as stack:
            _apply_patches(interactions_df, products_df, *mocks, stack=stack)
            evaluate(n_factors=5, knn_k=3, experiment_name="smartrec/test")
        mocks[0].set_experiment.assert_called_once_with("smartrec/test")

    def test_mlflow_metrics_logged_for_all_models(
        self, interactions_df: pd.DataFrame, products_df: pd.DataFrame
    ) -> None:
        mocks = _build_mocks(interactions_df, products_df)
        with ExitStack() as stack:
            _apply_patches(interactions_df, products_df, *mocks, stack=stack)
            evaluate(n_factors=5, knn_k=3)
        calls = [str(c) for c in mocks[0].log_metrics.call_args_list]
        logged = " ".join(calls)
        assert "svd_" in logged
        assert "knn_" in logged
        assert "hybrid_" in logged
