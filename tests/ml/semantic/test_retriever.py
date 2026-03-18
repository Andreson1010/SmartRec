"""
tests/ml/semantic/test_retriever.py
-------------------------------------
Testes para SemanticRetriever — usa embeddings aleatórios normalizados,
sem carregar modelo real de Sentence Transformers.
"""

from __future__ import annotations

from pathlib import Path

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


# ---------------------------------------------------------------------------
# get_embedding
# ---------------------------------------------------------------------------


class TestGetEmbedding:
    def test_returns_array_for_indexed_product(self, retriever) -> None:
        result = retriever.get_embedding("p0")
        assert isinstance(result, np.ndarray)
        assert result.shape == (EMBED_DIM,)

    def test_returns_none_for_unknown_product(self, retriever) -> None:
        assert retriever.get_embedding("inexistente") is None

    def test_returns_copy_not_view(self, retriever) -> None:
        """Modificar o vetor retornado não deve alterar o índice interno."""
        vec = retriever.get_embedding("p0")
        original = retriever.get_embedding("p0").copy()
        vec[:] = 0.0
        assert np.allclose(retriever.get_embedding("p0"), original)


# ---------------------------------------------------------------------------
# score_items
# ---------------------------------------------------------------------------


class TestScoreItems:
    def test_returns_dict(self, retriever) -> None:
        rng = np.random.default_rng(0)
        vec = rng.standard_normal(EMBED_DIM).astype("float32")
        vec /= np.linalg.norm(vec)
        result = retriever.score_items(["p0", "p1", "p2"], vec)
        assert isinstance(result, dict)

    def test_scores_only_indexed_products(self, retriever) -> None:
        rng = np.random.default_rng(0)
        vec = rng.standard_normal(EMBED_DIM).astype("float32")
        vec /= np.linalg.norm(vec)
        result = retriever.score_items(["p0", "p1", "inexistente"], vec)
        assert "p0" in result
        assert "p1" in result
        assert "inexistente" not in result

    def test_scores_in_range(self, retriever) -> None:
        rng = np.random.default_rng(0)
        vec = rng.standard_normal(EMBED_DIM).astype("float32")
        vec /= np.linalg.norm(vec)
        result = retriever.score_items([f"p{i}" for i in range(5)], vec)
        for score in result.values():
            assert -1.0 <= score <= 1.0

    def test_empty_list_returns_empty_dict(self, retriever) -> None:
        rng = np.random.default_rng(0)
        vec = rng.standard_normal(EMBED_DIM).astype("float32")
        vec /= np.linalg.norm(vec)
        assert retriever.score_items([], vec) == {}

    def test_consistent_with_query_by_vector(self, retriever) -> None:
        """score_items deve concordar com query_by_vector para os mesmos produtos."""
        rng = np.random.default_rng(7)
        vec = rng.standard_normal(EMBED_DIM).astype("float32")
        vec /= np.linalg.norm(vec)
        all_pids = [f"p{i}" for i in range(N_PRODUCTS)]
        scores_dict = retriever.score_items(all_pids, vec)
        results_list = retriever.query_by_vector(vec, top_k=N_PRODUCTS)
        for item in results_list:
            assert abs(scores_dict[item["product_id"]] - item["score"]) < 1e-5
