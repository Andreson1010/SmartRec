"""
ml/base.py
----------
Interface base para todos os modelos do SmartRec.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class BaseRecommender(ABC):
    """Interface obrigatória para todos os modelos do SmartRec.

    Subclasses devem implementar os cinco métodos abstratos.
    O atributo ``_is_fitted`` deve ser definido como ``True``
    ao final de :meth:`fit`.
    """

    _is_fitted: bool = False

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> "BaseRecommender":
        """Treina o modelo com os dados de interação.

        Parameters
        ----------
        data:
            DataFrame com colunas ``user_id``, ``product_id``, ``rating``.

        Returns
        -------
        BaseRecommender
            Retorna ``self`` para permitir encadeamento.
        """

    @abstractmethod
    def predict(self, user_id: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Retorna os top-K itens recomendados para o usuário.

        Cold start (usuário desconhecido) deve retornar fallback
        sem lançar exceção.

        Parameters
        ----------
        user_id:
            Identificador do usuário.
        top_k:
            Número de recomendações a retornar.

        Returns
        -------
        list[dict]
            Lista de ``{"product_id": str, "score": float}``
            com ``score`` normalizado em ``[0.0, 1.0]``,
            ordenada por score decrescente.
        """

    @abstractmethod
    def evaluate(self, test_data: pd.DataFrame) -> dict[str, float]:
        """Calcula métricas de avaliação no conjunto de teste.

        Parameters
        ----------
        test_data:
            DataFrame com colunas ``user_id``, ``product_id``, ``rating``.

        Returns
        -------
        dict
            Chaves: ``precision_at_10``, ``recall_at_10``,
            ``ndcg_at_10``, ``mrr``.
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Serializa o modelo em disco.

        Parameters
        ----------
        path:
            Diretório de destino (criado se não existir).
        """

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseRecommender":
        """Carrega o modelo do disco.

        Parameters
        ----------
        path:
            Diretório onde o modelo foi salvo por :meth:`save`.
        """

    def _check_fitted(self) -> None:
        """Lança RuntimeError se o modelo ainda não foi treinado.

        Raises
        ------
        RuntimeError
            Se ``_is_fitted`` for ``False``.
        """
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} não treinado. Chame fit() primeiro."
            )
