"""
api/models/health.py
--------------------
Schema de resposta do endpoint /health.
"""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Resposta do healthcheck com status da API e do modelo carregado.

    Attributes
    ----------
    status:
        ``"ok"`` se a API está operacional.
    model_version:
        Versão do HybridRecommender em memória.
    model_strategy:
        Estratégia de fusão ativa (``"rerank"``, ``"weighted"`` ou ``"rank_fusion"``).
    loaded_at:
        ISO 8601 — instante em que o modelo foi carregado no startup.
    uptime_s:
        Segundos desde o startup da aplicação.
    """

    status: str
    model_version: str
    model_strategy: str
    loaded_at: str
    uptime_s: float
