"""
api/main.py
-----------
Aplicação FastAPI do SmartRec.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from api.routers.recommendations import router as recommendations_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SmartRec API",
    description="API de recomendação híbrida: filtragem colaborativa + busca semântica.",
    version="1.0.0",
)

app.include_router(recommendations_router)


@app.get("/health", tags=["infra"])
async def health() -> dict[str, str]:
    """Verifica se a API está no ar."""
    return {"status": "ok"}
