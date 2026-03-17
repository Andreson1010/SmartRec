"""Endpoints de produtos."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.models.products import SimilarProductsResponse
from api.services.products import ProductService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/products", tags=["products"])


@router.get(
    "/{product_id}/similar",
    response_model=SimilarProductsResponse,
    summary="Retorna produtos similares via busca semântica",
)
async def get_similar_products(
    product_id: str,
    top_k: int = Query(10, ge=1, le=100, description="Número de produtos similares"),
    service: ProductService = Depends(),
) -> SimilarProductsResponse:
    """Retorna os K produtos mais similares ao produto informado.

    Usa embeddings semânticos gerados por Sentence Transformers
    (all-MiniLM-L6-v2) e similaridade de cosseno.

    Raises
    ------
    HTTPException 404:
        ``product_id`` não encontrado no índice de embeddings.
    HTTPException 500:
        Erro interno inesperado.
    """
    t0 = time.perf_counter()

    try:
        result = service.find_similar(product_id, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.exception("Erro inesperado em /products/%s/similar", product_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno — tente novamente.",
        ) from exc

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "/products/%s/similar top_k=%d latency=%.1fms",
        product_id,
        top_k,
        latency_ms,
    )
    return result
