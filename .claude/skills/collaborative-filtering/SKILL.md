---
name: collaborative-filtering
description: Padrão para implementar filtragem colaborativa (SVD e KNN) no SmartRec. Use esta skill sempre que for criar ou modificar ml/collaborative/svd.py ou ml/collaborative/knn.py, implementar SVDRecommender via scipy.sparse.linalg.svds (NÃO scikit-surprise — não funciona no Windows), ou criar KNNRecommender baseado em similaridade de cosseno com scipy.
---

# Skill: collaborative-filtering

Padrão para implementar filtragem colaborativa (SVD e KNN) no SmartRec.

> **Atenção:** `scikit-surprise` foi removido do projeto — não instala no Windows sem Visual Studio.
> CF é implementado com `scipy.sparse.linalg.svds` (já disponível via scipy).

## Localização

```
ml/collaborative/
├── svd.py        ← SVDRecommender via scipy.sparse.linalg.svds
├── knn.py        ← KNNRecommender user-based / item-based
└── artifacts/    ← modelos serializados
```

## SVDRecommender (`svd.py`)

```python
"""
ml/collaborative/svd.py
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from ml.base import BaseRecommender
from ml.evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k, mrr

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent


class SVDRecommender(BaseRecommender):
    """Filtragem colaborativa via SVD truncado (scipy.sparse.linalg.svds).

    Parameters
    ----------
    n_factors:
        Número de fatores latentes.
    """

    def __init__(self, n_factors: int = 50) -> None:
        self.n_factors = n_factors
        self._user_index: dict[str, int] = {}
        self._item_index: dict[str, int] = {}
        self._items: list[str] = []
        self._user_factors: np.ndarray | None = None
        self._item_factors: np.ndarray | None = None
        self._popularity: list[str] = []
        self._is_fitted = False

    def fit(self, data: pd.DataFrame) -> "SVDRecommender":
        """Treina o SVD e registra experimento no MLflow.

        Parameters
        ----------
        data:
            DataFrame com colunas ``user_id``, ``product_id``, ``rating``.
        """
        users = data["user_id"].unique()
        items = data["product_id"].unique()
        self._user_index = {u: i for i, u in enumerate(users)}
        self._item_index = {it: i for i, it in enumerate(items)}
        self._items = list(items)

        # Popularidade para cold start
        self._popularity = (
            data.groupby("product_id")["rating"]
            .count()
            .sort_values(ascending=False)
            .index.tolist()
        )

        rows = data["user_id"].map(self._user_index).values
        cols = data["product_id"].map(self._item_index).values

        # Remover bias de usuário
        user_means = data.groupby("user_id")["rating"].mean()
        ratings = data["rating"].values - data["user_id"].map(user_means).values

        matrix = csr_matrix(
            (ratings.astype("float32"), (rows, cols)),
            shape=(len(users), len(items)),
        )

        k = min(self.n_factors, min(matrix.shape) - 1)
        U, sigma, Vt = svds(matrix, k=k)

        self._user_factors = U * sigma          # (n_users, k)
        self._item_factors = Vt.T               # (n_items, k)

        with mlflow.start_run(nested=True):
            mlflow.log_params({"n_factors": self.n_factors})
            metrics = self.evaluate(data)
            mlflow.log_metrics(metrics)

        self._is_fitted = True
        logger.info("SVD treinado. Métricas: %s", metrics)
        return self

    def predict(self, user_id: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Retorna top-K recomendações. Cold start → itens populares com score 0.0."""
        self._check_fitted()

        if user_id not in self._user_index:
            logger.warning("Cold start para user_id=%s", user_id)
            return [{"product_id": pid, "score": 0.0} for pid in self._popularity[:top_k]]

        u_idx = self._user_index[user_id]
        scores = self._user_factors[u_idx] @ self._item_factors.T  # (n_items,)

        # Normalização min-max → [0, 1]
        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            scores = (scores - s_min) / (s_max - s_min)
        else:
            scores = np.zeros_like(scores)

        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        return [
            {"product_id": self._items[i], "score": float(scores[i])}
            for i in top_idx
        ]

    def evaluate(self, test_data: pd.DataFrame, k: int = 10) -> dict[str, float]:
        """Calcula métricas padrão do projeto."""
        relevant = (
            test_data[test_data["rating"] >= 4]
            .groupby("user_id")["product_id"]
            .apply(list)
            .to_dict()
        )
        users = list(relevant.keys())[:200]

        recs = {u: [r["product_id"] for r in self.predict(u, top_k=k)] for u in users}

        return {
            "precision_at_10": float(np.mean([precision_at_k(recs[u], relevant[u], k) for u in users])),
            "recall_at_10":    float(np.mean([recall_at_k(recs[u], relevant[u], k) for u in users])),
            "ndcg_at_10":      float(np.mean([ndcg_at_k(recs[u], relevant[u], k) for u in users])),
            "mrr":             float(np.mean([mrr(recs[u], relevant[u]) for u in users])),
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "svd.pkl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "SVDRecommender":
        with open(Path(path) / "svd.pkl", "rb") as f:
            return pickle.load(f)
```

## KNNRecommender (`knn.py`)

KNN baseado em similaridade de cosseno entre vetores de usuário construídos a partir da matriz de interações.

```python
"""
ml/collaborative/knn.py
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from ml.base import BaseRecommender

logger = logging.getLogger(__name__)


class KNNRecommender(BaseRecommender):
    """KNN colaborativo (user-based) via similaridade de cosseno.

    Parameters
    ----------
    k:
        Número de vizinhos.
    """

    def __init__(self, k: int = 20) -> None:
        self.k = k
        self._user_index: dict[str, int] = {}
        self._item_index: dict[str, int] = {}
        self._items: list[str] = []
        self._matrix: csr_matrix | None = None
        self._popularity: list[str] = []
        self._is_fitted = False

    def fit(self, data: pd.DataFrame) -> "KNNRecommender":
        users = data["user_id"].unique()
        items = data["product_id"].unique()
        self._user_index = {u: i for i, u in enumerate(users)}
        self._item_index = {it: i for i, it in enumerate(items)}
        self._items = list(items)
        self._popularity = (
            data.groupby("product_id")["rating"]
            .count()
            .sort_values(ascending=False)
            .index.tolist()
        )

        rows = data["user_id"].map(self._user_index).values
        cols = data["product_id"].map(self._item_index).values
        self._matrix = csr_matrix(
            (data["rating"].values.astype("float32"), (rows, cols)),
            shape=(len(users), len(items)),
        )

        with mlflow.start_run(nested=True):
            mlflow.log_params({"k": self.k})

        self._is_fitted = True
        return self

    def predict(self, user_id: str, top_k: int = 10) -> list[dict[str, Any]]:
        self._check_fitted()

        if user_id not in self._user_index:
            return [{"product_id": pid, "score": 0.0} for pid in self._popularity[:top_k]]

        u_idx = self._user_index[user_id]
        user_vec = self._matrix[u_idx]

        sims = cosine_similarity(user_vec, self._matrix).flatten()  # (n_users,)
        sims[u_idx] = -1

        neighbor_idx = np.argpartition(sims, -self.k)[-self.k:]
        neighbor_vecs = self._matrix[neighbor_idx]
        weights = sims[neighbor_idx][:, None]

        scores = (neighbor_vecs.toarray() * weights).sum(axis=0)  # (n_items,)

        # Zerar itens que o usuário já avaliou
        rated_cols = self._matrix[u_idx].nonzero()[1]
        scores[rated_cols] = -1

        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        s_max = scores[top_idx].max()
        norm = scores[top_idx] / s_max if s_max > 0 else scores[top_idx]

        return [
            {"product_id": self._items[i], "score": float(norm[j])}
            for j, i in enumerate(top_idx)
        ]

    def evaluate(self, test_data: pd.DataFrame, k: int = 10) -> dict[str, float]:
        from ml.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k, mrr

        relevant = (
            test_data[test_data["rating"] >= 4]
            .groupby("user_id")["product_id"]
            .apply(list)
            .to_dict()
        )
        users = list(relevant.keys())[:100]
        recs = {u: [r["product_id"] for r in self.predict(u, top_k=k)] for u in users}

        return {
            "precision_at_10": float(np.mean([precision_at_k(recs[u], relevant[u], k) for u in users])),
            "recall_at_10":    float(np.mean([recall_at_k(recs[u], relevant[u], k) for u in users])),
            "ndcg_at_10":      float(np.mean([ndcg_at_k(recs[u], relevant[u], k) for u in users])),
            "mrr":             float(np.mean([mrr(recs[u], relevant[u]) for u in users])),
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "knn.pkl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "KNNRecommender":
        with open(Path(path) / "knn.pkl", "rb") as f:
            return pickle.load(f)
```

## Checklist

- [ ] Usar `scipy.sparse.linalg.svds` — NÃO `scikit-surprise`
- [ ] Matriz esparsa `csr_matrix` — eficiente para dados de interação
- [ ] Bias removal por usuário antes do SVD
- [ ] `predict` normaliza scores para `[0, 1]`
- [ ] Cold start: usuário não visto retorna `_popularity[:top_k]` com `score=0.0`
- [ ] `fit` loga params e métricas no MLflow (seguir skill `mlflow-experiment-tracking`)
- [ ] Métricas calculadas via `ml/evaluation/metrics.py` (precision@10, recall@10, ndcg@10, mrr)
- [ ] Artefatos salvos em `ml/collaborative/artifacts/`
- [ ] Testes em `tests/ml/collaborative/` com dados sintéticos e MLflow mockado
