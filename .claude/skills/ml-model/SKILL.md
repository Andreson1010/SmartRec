---
name: ml-model
description: Padrão para criar ou implementar qualquer modelo de ML no SmartRec. Use esta skill sempre que for criar um novo modelo (SVD, KNN, embedder, híbrido, ou qualquer outro), adicionar métodos fit/predict/evaluate/save/load, implementar cold start, ou integrar MLflow num modelo. Também use ao criar a classe base BaseRecommender ou ao herdar dela.
---

# Skill: ml-model

Padrão para criar modelos de ML no SmartRec.

## Interface obrigatória

Todo modelo deve herdar de uma classe base com os cinco métodos abaixo. Use este template como ponto de partida:

```python
"""
ml/<submodule>/<model_name>.py
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseModel:
    """Interface base para todos os modelos SmartRec."""

    def fit(self, data: pd.DataFrame) -> "BaseModel":
        raise NotImplementedError

    def predict(self, user_id: str, top_k: int = 10) -> list[dict[str, Any]]:
        raise NotImplementedError

    def evaluate(self, test_data: pd.DataFrame) -> dict[str, float]:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path) -> "BaseModel":
        raise NotImplementedError


class MyModel(BaseModel):
    """Descrição do modelo.

    Parameters
    ----------
    param_a:
        Descrição do parâmetro.
    """

    def __init__(self, param_a: int = 10) -> None:
        self.param_a = param_a
        self._is_fitted = False

    # ------------------------------------------------------------------
    def fit(self, data: pd.DataFrame) -> "MyModel":
        """Treina o modelo e loga métricas no MLflow.

        Parameters
        ----------
        data:
            DataFrame com colunas ``user_id``, ``product_id``, ``rating``.
        """
        with mlflow.start_run(nested=True):
            mlflow.log_param("param_a", self.param_a)

            # ... treinamento ...

            metrics = self.evaluate(data)          # avalia no próprio split
            mlflow.log_metrics(metrics)

        self._is_fitted = True
        logger.info("Modelo treinado. Métricas: %s", metrics)
        return self

    # ------------------------------------------------------------------
    def predict(self, user_id: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Retorna os top-K itens recomendados para o usuário.

        Trata cold start: se o usuário não foi visto no treino,
        devolve recomendações populares como fallback.

        Parameters
        ----------
        user_id:
            Identificador do usuário.
        top_k:
            Número de recomendações a retornar.

        Returns
        -------
        list[dict]
            Lista de ``{"product_id": str, "score": float}``.
        """
        if not self._is_fitted:
            raise RuntimeError("Modelo não treinado. Chame fit() primeiro.")

        if user_id not in self._known_users:
            logger.warning("Cold start para user_id=%s — usando fallback popular.", user_id)
            return self._popular_fallback(top_k)

        # ... lógica de predição ...
        return []

    def _popular_fallback(self, top_k: int) -> list[dict[str, Any]]:
        """Retorna os itens mais populares como fallback de cold start."""
        # ... implementar com base em self._popularity ...
        return []

    # ------------------------------------------------------------------
    def evaluate(self, test_data: pd.DataFrame) -> dict[str, float]:
        """Calcula métricas de avaliação.

        Returns
        -------
        dict
            Chaves: ``precision_at_10``, ``recall_at_10``,
            ``ndcg_at_10``, ``mrr``.
        """
        from ml.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k, mrr

        # ... calcular métricas ...
        return {
            "precision_at_10": 0.0,
            "recall_at_10":    0.0,
            "ndcg_at_10":      0.0,
            "mrr":             0.0,
        }

    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        """Serializa o modelo em disco.

        Parameters
        ----------
        path:
            Diretório de destino (criado se não existir).
        """
        import pickle

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", "wb") as f:
            pickle.dump(self, f)
        logger.info("Modelo salvo em %s", path)

    @classmethod
    def load(cls, path: Path) -> "MyModel":
        """Carrega o modelo do disco.

        Parameters
        ----------
        path:
            Diretório onde ``model.pkl`` foi salvo por :meth:`save`.
        """
        import pickle

        with open(Path(path) / "model.pkl", "rb") as f:
            model = pickle.load(f)
        logger.info("Modelo carregado de %s", path)
        return model
```

## Checklist

- [ ] Herda de `BaseModel` ou implementa os 5 métodos (`fit`, `predict`, `evaluate`, `save`, `load`)
- [ ] `fit` abre um `mlflow.start_run` e loga params + métricas
- [ ] `predict` trata cold start (usuário ou item não visto) com fallback
- [ ] `evaluate` usa as métricas de `ml/evaluation/metrics.py`
- [ ] Type hints em todos os métodos públicos
- [ ] Docstring na classe e em cada método público
- [ ] Teste correspondente em `tests/ml/<submodule>/test_<model_name>.py`
