"""Schemas Pydantic para o endpoint de recomendações."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    """Payload de entrada para recomendações."""

    user_id: str = Field(..., min_length=1, description="Identificador do usuário")
    top_k: int = Field(10, ge=1, le=100, description="Número de recomendações")


class RecommendedItem(BaseModel):
    """Item individual retornado na lista de recomendações."""

    product_id: str
    score: float = Field(..., ge=0.0, le=1.0)


class RecommendationResponse(BaseModel):
    """Payload de resposta para recomendações."""

    user_id: str
    recommendations: list[RecommendedItem]
    model_version: str
