"""
tests/ml/test_base.py
---------------------
Testes da interface BaseRecommender.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from ml.base import BaseRecommender


class ConcreteRecommender(BaseRecommender):
    """Implementação mínima para testar a interface."""

    def fit(self, data: pd.DataFrame) -> "ConcreteRecommender":
        self._is_fitted = True
        return self

    def predict(self, user_id: str, top_k: int = 10) -> list[dict[str, Any]]:
        self._check_fitted()
        return [{"product_id": "p0", "score": 1.0}]

    def evaluate(self, test_data: pd.DataFrame) -> dict[str, float]:
        return {
            "precision_at_10": 0.0,
            "recall_at_10": 0.0,
            "ndcg_at_10": 0.0,
            "mrr": 0.0,
        }

    def save(self, path: Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(cls, path: Path) -> "ConcreteRecommender":
        return cls()


class TestBaseRecommender:
    def test_fit_returns_self(self, interactions_df: pd.DataFrame) -> None:
        model = ConcreteRecommender()
        result = model.fit(interactions_df)
        assert result is model

    def test_fit_sets_is_fitted(self, interactions_df: pd.DataFrame) -> None:
        model = ConcreteRecommender()
        assert model._is_fitted is False
        model.fit(interactions_df)
        assert model._is_fitted is True

    def test_predict_before_fit_raises(self) -> None:
        model = ConcreteRecommender()
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            model.predict("u0")

    def test_predict_returns_list(self, interactions_df: pd.DataFrame) -> None:
        model = ConcreteRecommender().fit(interactions_df)
        result = model.predict("u0", top_k=5)
        assert isinstance(result, list)

    def test_predict_item_schema(self, interactions_df: pd.DataFrame) -> None:
        model = ConcreteRecommender().fit(interactions_df)
        for item in model.predict("u0"):
            assert "product_id" in item
            assert "score" in item
            assert 0.0 <= item["score"] <= 1.0

    def test_evaluate_returns_required_keys(
        self, interactions_df: pd.DataFrame
    ) -> None:
        model = ConcreteRecommender().fit(interactions_df)
        metrics = model.evaluate(interactions_df)
        required = {"precision_at_10", "recall_at_10", "ndcg_at_10", "mrr"}
        assert required.issubset(metrics.keys())
