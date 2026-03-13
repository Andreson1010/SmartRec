"""Endpoints de recomendações."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, status

from api.models.recommendations import RecommendationRequest, RecommendationResponse
from api.services.recommendations import RecommendationService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.post(
    "/",
    response_model=RecommendationResponse,
    summary="Gera recomendações híbridas para um usuário",
)
async def get_recommendations(
    payload: RecommendationRequest,
    service: RecommendationService = Depends(),
) -> RecommendationResponse:
    """Retorna os top-K produtos recomendados para o usuário.

    Combina filtragem colaborativa (SVD) com busca semântica (Sentence
    Transformers). Usuários sem histórico recebem recomendações populares
    como fallback (cold start).

    Raises
    ------
    HTTPException 404:
        ``user_id`` inválido ou não encontrado.
    HTTPException 422:
        Payload inválido (tratado automaticamente pelo Pydantic).
    HTTPException 500:
        Erro interno inesperado.
    """
    t0 = time.perf_counter()

    try:
        result = service.run(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.exception("Erro inesperado em /recommendations user_id=%s", payload.user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno — tente novamente.",
        ) from exc

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "/recommendations user_id=%s top_k=%d latency=%.1fms",
        payload.user_id,
        payload.top_k,
        latency_ms,
    )
    return result
