"""Schemas Pydantic para o endpoint de produtos similares."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SimilarItem(BaseModel):
    """Item individual retornado na lista de produtos similares."""

    product_id: str
    score: float = Field(..., ge=0.0, le=1.0)


class SimilarProductsResponse(BaseModel):
    """Payload de resposta para produtos similares."""

    product_id: str
    similar: list[SimilarItem]
    model_version: str
