"""
ml/semantic/retriever.py
------------------------
Busca por similaridade de cosseno sobre embeddings de produtos.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ml.semantic.embedder import EMBEDDINGS_DIR, ProductEmbedder

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """Busca os K produtos mais similares dado um embedding de consulta.

    Usa similaridade de cosseno via dot product — funciona porque os embeddings
    são normalizados (``normalize_embeddings=True`` no :class:`ProductEmbedder`).

    Parameters
    ----------
    embeddings_dir:
        Diretório com ``embeddings.npy`` e ``product_ids.npy`` gerados por
        :class:`ProductEmbedder`.
    """

    def __init__(self, embeddings_dir: Path = EMBEDDINGS_DIR) -> None:
        self._embeddings, self._product_ids = ProductEmbedder.load(embeddings_dir)
        logger.info(
            "SemanticRetriever carregado: %d produtos, dim=%d",
            len(self._product_ids),
            self._embeddings.shape[1],
        )

    def query_by_product(
        self, product_id: str, top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Retorna os K produtos mais similares a um produto dado.

        Cold start: produto não indexado retorna lista vazia sem exceção.

        Parameters
        ----------
        product_id:
            Produto de referência (deve estar no índice de embeddings).
        top_k:
            Número de resultados a retornar.

        Returns
        -------
        list[dict]
            Lista de ``{"product_id": str, "score": float}`` ordenada por
            score decrescente. ``score`` está em ``[0, 1]``.
        """
        idx = self._find_index(product_id)
        if idx is None:
            logger.warning("product_id=%s não encontrado no índice.", product_id)
            return []

        query_vec = self._embeddings[idx]  # shape (dim,)
        scores = self._embeddings @ query_vec  # shape (n_products,)
        scores[idx] = -np.inf  # marca para exclusão

        results = self._top_k_results(scores, top_k)
        # garante exclusão mesmo quando top_k >= N_PRODUCTS
        return [r for r in results if r["product_id"] != product_id]

    def query_by_vector(
        self, vector: np.ndarray, top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Busca por similaridade dado um vetor de consulta externo.

        Parameters
        ----------
        vector:
            Vetor de consulta já normalizado (norma L2 = 1), shape ``(dim,)``.
        top_k:
            Número de resultados a retornar.

        Returns
        -------
        list[dict]
            Lista de ``{"product_id": str, "score": float}``.
        """
        scores = self._embeddings @ vector
        return self._top_k_results(scores, top_k)

    def _top_k_results(
        self, scores: np.ndarray, top_k: int
    ) -> list[dict[str, Any]]:
        """Extrai os top-K índices e monta a lista de resultado."""
        top_k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [
            {"product_id": str(self._product_ids[i]), "score": float(scores[i])}
            for i in top_indices
        ]

    def _find_index(self, product_id: str) -> int | None:
        matches = np.where(self._product_ids == product_id)[0]
        return int(matches[0]) if len(matches) > 0 else None
