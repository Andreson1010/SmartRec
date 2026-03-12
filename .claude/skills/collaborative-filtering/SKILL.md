# Skill: collaborative-filtering

Padrão para implementar filtragem colaborativa (SVD e KNN) no SmartRec.

## Localização

```
ml/collaborative/
├── svd.py        ← SVD via Surprise (matrix factorization)
├── knn.py        ← KNN user-based / item-based
└── artifacts/    ← modelos serializados
```

## SVD com Surprise (`svd.py`)

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
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

from ml.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k, mrr

logger = logging.getLogger(__name__)


class SVDRecommender:
    """Filtragem colaborativa via SVD (Singular Value Decomposition).

    Usa a biblioteca Surprise. Treinado em interações explícitas (ratings 1–5).

    Parameters
    ----------
    n_factors:
        Dimensão dos vetores latentes.
    n_epochs:
        Épocas de SGD.
    lr_all:
        Taxa de aprendizado global.
    reg_all:
        Regularização L2 global.
    """

    def __init__(
        self,
        n_factors: int = 100,
        n_epochs: int = 20,
        lr_all: float = 0.005,
        reg_all: float = 0.02,
    ) -> None:
        self.n_factors = n_factors
        self.n_epochs  = n_epochs
        self.lr_all    = lr_all
        self.reg_all   = reg_all

        self._algo: SVD | None = None
        self._all_items: list[str] = []
        self._is_fitted = False

    # ------------------------------------------------------------------
    def fit(self, data: pd.DataFrame) -> "SVDRecommender":
        """Treina o SVD e registra experimento no MLflow.

        Parameters
        ----------
        data:
            DataFrame com colunas ``user_id``, ``product_id``, ``rating``.
        """
        reader = Reader(rating_scale=(1, 5))
        dataset = Dataset.load_from_df(data[["user_id", "product_id", "rating"]], reader)
        trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)

        self._algo = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
        )

        with mlflow.start_run(nested=True):
            mlflow.log_params({
                "n_factors": self.n_factors,
                "n_epochs":  self.n_epochs,
                "lr_all":    self.lr_all,
                "reg_all":   self.reg_all,
            })

            self._algo.fit(trainset)
            self._all_items = data["product_id"].unique().tolist()

            metrics = self._evaluate_surprise(testset, data)
            mlflow.log_metrics(metrics)

        self._is_fitted = True
        logger.info("SVD treinado. Métricas: %s", metrics)
        return self

    # ------------------------------------------------------------------
    def predict(self, user_id: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Retorna top-K recomendações para o usuário.

        Cold start: usuário desconhecido recebe fallback de itens populares.
        """
        if not self._is_fitted:
            raise RuntimeError("Modelo não treinado. Chame fit() primeiro.")

        try:
            predictions = [
                self._algo.predict(user_id, iid)
                for iid in self._all_items
            ]
        except Exception:
            logger.warning("Cold start para user_id=%s", user_id)
            return self._popular_fallback(top_k)

        predictions.sort(key=lambda x: x.est, reverse=True)
        return [
            {"product_id": p.iid, "score": float(p.est) / 5.0}
            for p in predictions[:top_k]
        ]

    def _popular_fallback(self, top_k: int) -> list[dict[str, Any]]:
        """Retorna os primeiros itens do índice como proxy de popularidade."""
        return [{"product_id": iid, "score": 0.0} for iid in self._all_items[:top_k]]

    # ------------------------------------------------------------------
    def _evaluate_surprise(self, testset: list, data: pd.DataFrame) -> dict[str, float]:
        """Avalia usando as métricas padrão do projeto."""
        k = 10
        user_recs: dict[str, list[str]] = {}
        user_relevant: dict[str, list[str]] = {}

        for uid, iid, true_r in testset:
            if uid not in user_recs:
                preds = self.predict(uid, top_k=k)
                user_recs[uid] = [p["product_id"] for p in preds]
            user_relevant.setdefault(uid, [])
            if true_r >= 4:
                user_relevant[uid].append(iid)

        p  = np.mean([precision_at_k(user_recs[u], user_relevant.get(u, []), k) for u in user_recs])
        r  = np.mean([recall_at_k(user_recs[u], user_relevant.get(u, []), k) for u in user_recs])
        nd = np.mean([ndcg_at_k(user_recs[u], user_relevant.get(u, []), k) for u in user_recs])
        m  = np.mean([mrr(user_recs[u], user_relevant.get(u, [])) for u in user_recs])

        return {
            "precision_at_10": float(p),
            "recall_at_10":    float(r),
            "ndcg_at_10":      float(nd),
            "mrr":             float(m),
        }

    # ------------------------------------------------------------------
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

## KNN (`knn.py`)

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
import pandas as pd
from surprise import KNNWithMeans, Dataset, Reader
from surprise.model_selection import train_test_split

logger = logging.getLogger(__name__)


class KNNRecommender:
    """KNN colaborativo (user-based ou item-based) via Surprise.

    Parameters
    ----------
    k:
        Número de vizinhos.
    sim_options:
        Dicionário de opções de similaridade do Surprise.
        Ex: ``{"name": "cosine", "user_based": True}``
    """

    def __init__(
        self,
        k: int = 40,
        sim_options: dict | None = None,
    ) -> None:
        self.k = k
        self.sim_options = sim_options or {"name": "cosine", "user_based": False}
        self._algo: KNNWithMeans | None = None
        self._all_items: list[str] = []
        self._is_fitted = False

    def fit(self, data: pd.DataFrame) -> "KNNRecommender":
        reader  = Reader(rating_scale=(1, 5))
        dataset = Dataset.load_from_df(data[["user_id", "product_id", "rating"]], reader)
        trainset, _ = train_test_split(dataset, test_size=0.2, random_state=42)

        self._algo = KNNWithMeans(k=self.k, sim_options=self.sim_options)

        with mlflow.start_run(nested=True):
            mlflow.log_params({"k": self.k, **self.sim_options})
            self._algo.fit(trainset)
            self._all_items = data["product_id"].unique().tolist()

        self._is_fitted = True
        return self

    def predict(self, user_id: str, top_k: int = 10) -> list[dict[str, Any]]:
        if not self._is_fitted:
            raise RuntimeError("Modelo não treinado. Chame fit() primeiro.")

        predictions = [self._algo.predict(user_id, iid) for iid in self._all_items]
        predictions.sort(key=lambda x: x.est, reverse=True)
        return [
            {"product_id": p.iid, "score": float(p.est) / 5.0}
            for p in predictions[:top_k]
        ]

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

- [ ] `Reader(rating_scale=(1, 5))` — escala explícita obrigatória
- [ ] `train_test_split(random_state=42)` — splits reprodutíveis
- [ ] `predict` normaliza score para `[0, 1]` dividindo por 5.0
- [ ] Cold start: usuário não visto retorna fallback sem exceção
- [ ] `fit` loga params e métricas no MLflow (seguir skill `mlflow-experiment-tracking`)
- [ ] Métricas calculadas via `ml/evaluation/metrics.py` (precision@10, recall@10, ndcg@10, mrr)
- [ ] Artefatos salvos em `ml/collaborative/artifacts/`
- [ ] Testes em `tests/ml/collaborative/` com dados sintéticos e MLflow mockado
