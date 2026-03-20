"""
api/security.py
---------------
Dependência de autenticação via API key (header X-API-Key).
"""

from __future__ import annotations

import logging
import os

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=True)


def verify_api_key(api_key: str = Security(_API_KEY_HEADER)) -> str:
    """Valida o API key enviado no header ``X-API-Key``.

    A chave esperada é lida da variável de ambiente ``API_KEY``.
    Configure-a no ``.env`` antes de iniciar a API.

    Parameters
    ----------
    api_key:
        Valor do header ``X-API-Key`` extraído automaticamente pelo FastAPI.

    Returns
    -------
    str
        O próprio ``api_key`` validado.

    Raises
    ------
    HTTPException 401:
        Chave ausente, inválida ou ``API_KEY`` não configurada no ambiente.
    """
    expected = os.getenv("API_KEY", "")
    if not expected:
        logger.error("API_KEY não configurada — recusando request.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Serviço sem API key configurada. Contate o administrador.",
        )
    if api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key inválida.",
        )
    return api_key
