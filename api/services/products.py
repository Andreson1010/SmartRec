"""Lógica de negócio para o endpoint de produtos similares."""

from __future__ import annotations

import logging
from pathlib import Path

from api.models.products import SimilarItem, SimilarProductsResponse

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent


class ProductService:
    """Orquestra a chamada ao SemanticRetriever e monta o response."""

    def __init__(self) -> None:
        from ml.semantic.retriever import SemanticRetriever

        self._retriever = SemanticRetriever()

    def find_similar(self, product_id: str, top_k: int = 10) -> SimilarProductsResponse:
        """Retorna produtos similares ao produto dado.

        Parameters
        ----------
        product_id:
            Identificador do produto de referência.
        top_k:
            Número de produtos similares a retornar.

        Raises
        ------
        ValueError:
            Se o ``product_id`` não estiver no índice de embeddings.
        """
        results = self._retriever.query_by_product(product_id, top_k=top_k)
        if not results:
            raise ValueError(f"product_id '{product_id}' não encontrado.")
        return SimilarProductsResponse(
            product_id=product_id,
            similar=[SimilarItem(**r) for r in results],
            model_version="1.0.0",
        )
