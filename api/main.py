"""
api/main.py
-----------
Aplicação FastAPI do SmartRec.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

from api.models.health import HealthResponse
from api.routers.products import router as products_router
from api.routers.recommendations import router as recommendations_router
from api.services.recommendations import RecommendationService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class LatencyMiddleware(BaseHTTPMiddleware):
    """Loga método, path, status e latência de todas as requisições.

    Injeta o header ``X-Response-Time-Ms`` na resposta para que clientes
    e gateways possam observar a latência sem acessar os logs do servidor.
    """

    async def dispatch(self, request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "%s %s %d %.1fms",
            request.method,
            request.url.path,
            response.status_code,
            ms,
        )
        response.headers["X-Response-Time-Ms"] = f"{ms:.1f}"
        return response


# ---------------------------------------------------------------------------
# Lifespan — carrega recursos uma vez no startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa o RecommendationService uma única vez no startup.

    Armazena a instância em ``app.state`` para que todos os endpoints
    compartilhem o mesmo modelo sem recarregá-lo a cada request.
    """
    app.state.rec_service = RecommendationService()
    app.state.started_at = datetime.now(UTC)
    logger.info(
        "SmartRec iniciado — model_version=%s strategy=%s",
        app.state.rec_service._model.version,
        app.state.rec_service._model.strategy,
    )
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


app = FastAPI(
    title="SmartRec API",
    description="API de recomendação híbrida: CF + busca semântica.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(LatencyMiddleware)
app.include_router(recommendations_router)
app.include_router(products_router)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["infra"])
async def health(request: Request) -> HealthResponse:
    """Retorna status da API e informações do modelo em memória."""
    svc = request.app.state.rec_service
    uptime = (datetime.now(UTC) - request.app.state.started_at).total_seconds()
    return HealthResponse(
        status="ok",
        model_version=svc._model.version,
        model_strategy=svc._model.strategy,
        loaded_at=svc.loaded_at.isoformat(),
        uptime_s=round(uptime, 1),
    )
