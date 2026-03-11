# Skill: api-endpoint

Padrão para criar endpoints FastAPI no SmartRec.

## Estrutura de arquivos

```
api/
├── models/      ← schemas Pydantic (request + response)
├── routers/     ← definição dos endpoints
└── services/    ← lógica de negócio (chama ml/)
```

Cada feature nova ganha um arquivo em cada camada.

## Template

### 1. Schema (`api/models/<feature>.py`)

```python
"""Schemas Pydantic para <feature>."""
from __future__ import annotations

from pydantic import BaseModel, Field


class <Feature>Request(BaseModel):
    """Payload de entrada para <feature>."""

    user_id: str = Field(..., description="Identificador do usuário")
    top_k: int  = Field(10, ge=1, le=100, description="Número de recomendações")


class RecommendedItem(BaseModel):
    """Item retornado na lista de recomendações."""

    product_id: str
    score:      float = Field(..., ge=0.0, le=1.0)
    title:      str | None = None


class <Feature>Response(BaseModel):
    """Payload de resposta para <feature>."""

    user_id:         str
    recommendations: list[RecommendedItem]
    model_version:   str
```

### 2. Router (`api/routers/<feature>.py`)

```python
"""Endpoints de <feature>."""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, status

from api.models.<feature> import <Feature>Request, <Feature>Response
from api.services.<feature> import <Feature>Service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/<feature>", tags=["<feature>"])


@router.post(
    "/",
    response_model=<Feature>Response,
    summary="Descrição curta do endpoint",
)
async def create_<feature>(
    payload: <Feature>Request,
    service: <Feature>Service = Depends(),
) -> <Feature>Response:
    """Descrição longa do endpoint.

    Raises
    ------
    HTTPException 404:
        Usuário não encontrado.
    HTTPException 422:
        Payload inválido (tratado automaticamente pelo Pydantic).
    HTTPException 500:
        Erro interno inesperado.
    """
    t0 = time.perf_counter()

    try:
        result = service.run(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Erro inesperado em /<feature>")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno — tente novamente.",
        ) from exc

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info("/<feature> user_id=%s latency=%.1fms", payload.user_id, latency_ms)

    return result
```

### 3. Service (`api/services/<feature>.py`)

```python
"""Lógica de negócio para <feature>."""
from __future__ import annotations

from api.models.<feature> import <Feature>Request, <Feature>Response
from ml.hybrid.recommender import HybridRecommender


class <Feature>Service:
    """Orquestra a chamada ao modelo e monta o response."""

    def __init__(self) -> None:
        self._model = HybridRecommender.load("ml/hybrid/artifacts/")

    def run(self, payload: <Feature>Request) -> <Feature>Response:
        """Executa a recomendação e retorna o response tipado.

        Parameters
        ----------
        payload:
            Request validado pelo Pydantic.

        Raises
        ------
        ValueError:
            Se o ``user_id`` não existir na base.
        """
        recommendations = self._model.predict(payload.user_id, top_k=payload.top_k)
        return <Feature>Response(
            user_id=payload.user_id,
            recommendations=recommendations,
            model_version=self._model.version,
        )
```

### 4. Registro no app (`api/main.py`)

```python
from api.routers.<feature> import router as <feature>_router
app.include_router(<feature>_router)
```

## Checklist

- [ ] Schema de request e response em `api/models/`
- [ ] Campos com `Field(...)` — descrição e validadores (`ge`, `le`, `min_length`)
- [ ] Router com `prefix` e `tags` definidos
- [ ] `try/except` separando `ValueError` (404) de `Exception` (500)
- [ ] Log de latência em ms após cada request bem-sucedido
- [ ] Service isolado do router — router não importa `ml/` diretamente
- [ ] Router registrado em `api/main.py`
- [ ] Teste correspondente em `tests/api/test_<feature>.py`
