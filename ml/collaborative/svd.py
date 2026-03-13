"""
ml/collaborative/svd.py
------------------------
Filtragem colaborativa via SVD usando scipy.sparse.linalg.svds.

Nao depende de scikit-surprise: usa apenas scipy + numpy, ja presentes
nos requisitos do projeto.
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
from ml.evaluation.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k

logger = logging.getLogger(__name__)


class SVDRecommender(BaseRecommender):
    """Filtragem colaborativa via Singular Value Decomposition.

    Constroi uma matriz usuario-item esparsa, aplica SVD truncado e
    reconstroi os scores previstos para cada par (usuario, item).

    Parameters
    ----------
    n_factors:
        Numero de fatores latentes (k na decomposicao U Sigma Vt).
    random_state:
        Semente para reproducibilidade do svds (via v0 aleatorio).
    """

    def __init__(
        self,
        n_factors: int = 50,
        random_state: int = 42,
    ) -> None:
        self.n_factors = n_factors
        self.random_state = random_state

        self._user_index: dict[str, int] = {}
        self._item_index: dict[str, int] = {}
        self._items: list[str] = []
        self._predicted: np.ndarray = np.array([])
        self._popular_items: list[str] = []

    # ------------------------------------------------------------------
    # Interface BaseRecommender
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> "SVDRecommender":
        """Treina o modelo SVD e registra experimento no MLflow.

        Parameters
        ----------
        data:
            DataFrame com colunas ``user_id``, ``product_id``, ``rating``.

        Returns
        -------
        SVDRecommender
            Retorna ``self`` para permitir encadeamento.
        """
        users = data["user_id"].unique().tolist()
        items = data["product_id"].unique().tolist()

        self._user_index = {u: i for i, u in enumerate(users)}
        self._item_index = {it: j for j, it in enumerate(items)}
        self._items = items

        # Popularidade para cold start (ordem de frequencia decrescente)
        self._popular_items = data["product_id"].value_counts().index.tolist()

        # Matriz esparsa usuario x item com ratings medios por par
        rows = data["user_id"].map(self._user_index).values
        cols = data["product_id"].map(self._item_index).values
        ratings = data["rating"].values.astype(np.float32)

        n_users = len(users)
        n_items = len(items)
        matrix = csr_matrix((ratings, (rows, cols)), shape=(n_users, n_items))

        # Normalizar pela media do usuario (bias removal)
        user_means = np.array(matrix.mean(axis=1)).flatten()
        # Subtrai media de cada usuario nas entradas nao-zero
        matrix_norm = matrix.copy().astype(np.float64)
        for uid_idx, mean in enumerate(user_means):
            start, end = matrix_norm.indptr[uid_idx], matrix_norm.indptr[uid_idx + 1]
            matrix_norm.data[start:end] -= mean

        k = min(self.n_factors, n_users - 1, n_items - 1)
        rng = np.random.default_rng(self.random_state)
        v0 = rng.standard_normal(min(n_users, n_items))

        U, sigma, Vt = svds(matrix_norm, k=k, v0=v0)

        # Reconstruir matriz densa de scores (adiciona bias de volta)
        predicted_norm = U @ np.diag(sigma) @ Vt
        self._predicted = predicted_norm + user_means[:, np.newaxis]

        with mlflow.start_run(run_name="svd-fit", nested=True):
            mlflow.log_params(
                {"n_factors": self.n_factors, "random_state": self.random_state}
            )

        self._is_fitted = True
        logger.info("SVD treinado — %d usuarios, %d itens", n_users, n_items)
        return self

    def predict(self, user_id: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Retorna os top-K itens recomendados para o usuario.

        Cold start: usuario desconhecido recebe os itens mais populares
        sem lancar excecao.

        Parameters
        ----------
        user_id:
            Identificador do usuario.
        top_k:
            Numero de recomendacoes a retornar.

        Returns
        -------
        list[dict]
            Lista de ``{"product_id": str, "score": float}`` com ``score``
            normalizado em ``[0.0, 1.0]``, ordenada por score decrescente.
        """
        self._check_fitted()

        if user_id not in self._user_index:
            logger.debug("Cold start para user_id=%s", user_id)
            return self._popular_fallback(top_k)

        uid_idx = self._user_index[user_id]
        scores = self._predicted[uid_idx]

        # Normaliza para [0, 1] usando min-max da linha
        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            norm_scores = (scores - s_min) / (s_max - s_min)
        else:
            norm_scores = np.zeros_like(scores)

        top_indices = np.argpartition(norm_scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(norm_scores[top_indices])[::-1]]

        return [
            {"product_id": self._items[j], "score": float(norm_scores[j])}
            for j in top_indices
        ]

    def evaluate(self, test_data: pd.DataFrame) -> dict[str, float]:
        """Calcula metricas de avaliacao no conjunto de teste.

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
        self._check_fitted()
        return self._compute_metrics(test_data)

    def save(self, path: Path) -> None:
        """Serializa o modelo em disco.

        Parameters
        ----------
        path:
            Diretorio de destino (criado se nao existir).
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "svd.pkl", "wb") as f:
            pickle.dump(self, f)
        logger.info("SVDRecommender salvo em %s", path)

    @classmethod
    def load(cls, path: Path) -> "SVDRecommender":
        """Carrega o modelo do disco.

        Parameters
        ----------
        path:
            Diretorio onde o modelo foi salvo por :meth:`save`.
        """
        with open(Path(path) / "svd.pkl", "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _popular_fallback(self, top_k: int) -> list[dict[str, Any]]:
        """Retorna os itens mais populares com score 0.0 (cold start)."""
        candidates = self._popular_items[:top_k]
        return [{"product_id": iid, "score": 0.0} for iid in candidates]

    def _compute_metrics(self, data: pd.DataFrame) -> dict[str, float]:
        """Calcula precision@10, recall@10, ndcg@10 e mrr para todos os usuarios."""
        k = 10
        relevant_by_user: dict[str, list[str]] = {}
        for row in data.itertuples(index=False):
            if row.rating >= 4.0:
                relevant_by_user.setdefault(row.user_id, []).append(row.product_id)

        p_vals, r_vals, nd_vals, mrr_vals = [], [], [], []
        for uid, relevant in relevant_by_user.items():
            recs = [d["product_id"] for d in self.predict(uid, top_k=k)]
            p_vals.append(precision_at_k(recs, relevant, k))
            r_vals.append(recall_at_k(recs, relevant, k))
            nd_vals.append(ndcg_at_k(recs, relevant, k))
            mrr_vals.append(mrr(recs, relevant))

        if not p_vals:
            return {
                "precision_at_10": 0.0,
                "recall_at_10": 0.0,
                "ndcg_at_10": 0.0,
                "mrr": 0.0,
            }

        return {
            "precision_at_10": float(np.mean(p_vals)),
            "recall_at_10": float(np.mean(r_vals)),
            "ndcg_at_10": float(np.mean(nd_vals)),
            "mrr": float(np.mean(mrr_vals)),
        }
