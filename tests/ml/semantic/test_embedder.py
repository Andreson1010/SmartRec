"""
tests/ml/semantic/test_embedder.py
-----------------------------------
Testes para ProductEmbedder — sem modelo real de Sentence Transformers.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def products_df() -> pd.DataFrame:
    """DataFrame mínimo de produtos para testes."""
    return pd.DataFrame(
        {
            "product_id": ["p0", "p1", "p2", "p3", "p4"],
            "title": ["Produto A", "Produto B", None, "Produto D", "Produto E"],
            "description": ["Desc A", None, "Desc C", "Desc D", None],
        }
    )


@pytest.fixture()
def fake_embeddings(products_df: pd.DataFrame) -> np.ndarray:
    """Embeddings aleatórios normalizados (norma L2 = 1)."""
    rng = np.random.default_rng(42)
    n = len(products_df)
    raw = rng.standard_normal((n, 384)).astype("float32")
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / norms


@pytest.fixture()
def embedder_with_mock():
    """ProductEmbedder com SentenceTransformer mockado."""
    from ml.semantic.embedder import ProductEmbedder

    embedder = ProductEmbedder(model_name="mock-model", batch_size=4)
    return embedder


# ---------------------------------------------------------------------------
# _build_text
# ---------------------------------------------------------------------------


class TestBuildText:
    def test_title_and_description(self, embedder_with_mock) -> None:
        row = pd.Series({"product_id": "p0", "title": "Tênis", "description": "Corrida"})
        assert embedder_with_mock._build_text(row) == "Tênis | Corrida"

    def test_only_title(self, embedder_with_mock) -> None:
        row = pd.Series({"product_id": "p1", "title": "Tênis", "description": None})
        assert embedder_with_mock._build_text(row) == "Tênis"

    def test_only_description(self, embedder_with_mock) -> None:
        row = pd.Series({"product_id": "p2", "title": None, "description": "Corrida"})
        assert embedder_with_mock._build_text(row) == "Corrida"

    def test_fallback_to_product_id(self, embedder_with_mock) -> None:
        row = pd.Series({"product_id": "p3", "title": None, "description": None})
        assert embedder_with_mock._build_text(row) == "p3"


# ---------------------------------------------------------------------------
# fit_transform
# ---------------------------------------------------------------------------


class TestFitTransform:
    def test_returns_float32_array(
        self, embedder_with_mock, products_df, fake_embeddings
    ) -> None:
        with patch.object(
            embedder_with_mock, "_load_model"
        ) as mock_load:
            mock_model = MagicMock()
            mock_model.encode.return_value = fake_embeddings
            mock_load.return_value = mock_model

            result = embedder_with_mock.fit_transform(products_df)

        assert result.dtype == np.float32

    def test_shape_matches_products(
        self, embedder_with_mock, products_df, fake_embeddings
    ) -> None:
        with patch.object(embedder_with_mock, "_load_model") as mock_load:
            mock_model = MagicMock()
            mock_model.encode.return_value = fake_embeddings
            mock_load.return_value = mock_model

            result = embedder_with_mock.fit_transform(products_df)

        assert result.shape[0] == len(products_df)

    def test_encode_called_with_normalize(
        self, embedder_with_mock, products_df, fake_embeddings
    ) -> None:
        with patch.object(embedder_with_mock, "_load_model") as mock_load:
            mock_model = MagicMock()
            mock_model.encode.return_value = fake_embeddings
            mock_load.return_value = mock_model

            embedder_with_mock.fit_transform(products_df)

        _, kwargs = mock_model.encode.call_args
        assert kwargs.get("normalize_embeddings") is True


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_roundtrip(
        self, embedder_with_mock, products_df, fake_embeddings, tmp_path: Path
    ) -> None:
        embedder_with_mock.save(fake_embeddings, products_df["product_id"], path=tmp_path)
        loaded_embs, loaded_ids = embedder_with_mock.load(path=tmp_path)

        assert loaded_embs.shape == fake_embeddings.shape
        assert list(loaded_ids) == list(products_df["product_id"])

    def test_save_creates_files(
        self, embedder_with_mock, products_df, fake_embeddings, tmp_path: Path
    ) -> None:
        embedder_with_mock.save(fake_embeddings, products_df["product_id"], path=tmp_path)

        assert (tmp_path / "embeddings.npy").exists()
        assert (tmp_path / "product_ids.npy").exists()

    def test_load_preserves_dtype(
        self, embedder_with_mock, products_df, fake_embeddings, tmp_path: Path
    ) -> None:
        embedder_with_mock.save(fake_embeddings, products_df["product_id"], path=tmp_path)
        loaded_embs, _ = embedder_with_mock.load(path=tmp_path)

        assert loaded_embs.dtype == np.float32
