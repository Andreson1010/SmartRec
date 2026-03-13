"""
tests/ml/semantic/test_retriever.py
-------------------------------------
Testes para SemanticRetriever — usa embeddings aleatórios normalizados,
sem carregar modelo real de Sentence Transformers.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_PRODUCTS = 20
EMBED_DIM = 16


@pytest.fixture()
def fake_embeddings_dir(tmp_path: Path) -> Path:
    """Diretório temporário com embeddings e product_ids sintéticos."""
    rng = np.random.default_rng(42)
    raw = rng.standard_normal((N_PRODUCTS, EMBED_DIM)).astype("float32")
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    embeddings = raw / norms

    product_ids = np.array([f"p{i}" for i in range(N_PRODUCTS)])

    np.save(tmp_path / "embeddings.npy", embeddings)
    np.save(tmp_path / "product_ids.npy", product_ids)
    return tmp_path


@pytest.fixture()
def retriever(fake_embeddings_dir: Path):
    """SemanticRetriever carregado com dados sintéticos."""
    from ml.semantic.retriever import SemanticRetriever

    return SemanticRetriever(embeddings_dir=fake_embeddings_dir)


# ---------------------------------------------------------------------------
# query_by_product
# ---------------------------------------------------------------------------


class TestQueryByProduct:
    def test_returns_list(self, retriever) -> None:
        result = retriever.query_by_product("p0", top_k=5)
        assert isinstance(result, list)

    def test_correct_length(self, retriever) -> None:
        result = retriever.query_by_product("p0", top_k=5)
        assert len(result) == 5

    def test_item_schema(self, retriever) -> None:
        results = retriever.query_by_product("p0", top_k=3)
        for item in results:
            assert "product_id" in item
            assert "score" in item
            assert isinstance(item["score"], float)

    def test_excludes_query_product(self, retriever) -> None:
        results = retriever.query_by_product("p0", top_k=5)
        ids = [r["product_id"] for r in results]
        assert "p0" not in ids

    def test_scores_in_range(self, retriever) -> None:
        results = retriever.query_by_product("p0", top_k=5)
        for item in results:
            assert -1.0 <= item["score"] <= 1.0

    def test_scores_sorted_descending(self, retriever) -> None:
        results = retriever.query_by_product("p0", top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_unknown_product_returns_empty(self, retriever) -> None:
        result = retriever.query_by_product("produto_inexistente", top_k=5)
        assert result == []

    def test_top_k_larger_than_corpus(self, retriever) -> None:
        """top_k maior que o corpus não deve lançar exceção."""
        result = retriever.query_by_product("p0", top_k=N_PRODUCTS + 100)
        assert len(result) <= N_PRODUCTS - 1  # -1 excluindo o próprio item


# ---------------------------------------------------------------------------
# query_by_vector
# ---------------------------------------------------------------------------


class TestQueryByVector:
    def test_returns_list(self, retriever) -> None:
        rng = np.random.default_rng(0)
        vec = rng.standard_normal(EMBED_DIM).astype("float32")
        vec /= np.linalg.norm(vec)
        result = retriever.query_by_vector(vec, top_k=5)
        assert isinstance(result, list)
        assert len(result) == 5

    def test_item_schema(self, retriever) -> None:
        rng = np.random.default_rng(0)
        vec = rng.standard_normal(EMBED_DIM).astype("float32")
        vec /= np.linalg.norm(vec)
        results = retriever.query_by_vector(vec, top_k=3)
        for item in results:
            assert "product_id" in item
            assert "score" in item

    def test_scores_sorted_descending(self, retriever) -> None:
        rng = np.random.default_rng(0)
        vec = rng.standard_normal(EMBED_DIM).astype("float32")
        vec /= np.linalg.norm(vec)
        results = retriever.query_by_vector(vec, top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)
