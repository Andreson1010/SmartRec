"""Lógica de negócio para o endpoint de recomendações."""

from __future__ import annotations

import logging
from pathlib import Path

from api.models.recommendations import (
    RecommendationRequest,
    RecommendationResponse,
    RecommendedItem,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent


class RecommendationService:
    """Orquestra a chamada ao HybridRecommender e monta o response."""

    def __init__(self) -> None:
        from ml.hybrid.recommender import HybridRecommender

        self._model = HybridRecommender.load(
            ROOT / "ml" / "hybrid" / "artifacts"
        )

    def run(self, payload: RecommendationRequest) -> RecommendationResponse:
        """Executa a recomendação e retorna o response tipado.

        Parameters
        ----------
        payload:
            Request validado pelo Pydantic.

        Raises
        ------
        ValueError:
            Se o ``user_id`` for inválido ou não encontrado.
        """
        recs = self._model.predict(payload.user_id, top_k=payload.top_k)
        return RecommendationResponse(
            user_id=payload.user_id,
            recommendations=[RecommendedItem(**r) for r in recs],
            model_version=self._model.version,
        )
