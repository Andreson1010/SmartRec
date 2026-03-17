"""
ml/collaborative/knn.py
------------------------
Filtragem colaborativa user-based via similaridade de cosseno.

Nao depende de scikit-surprise: usa scipy para a matriz esparsa e
sklearn.metrics.pairwise para o calculo eficiente de similaridade.
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
from ml.evaluation.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k

logger = logging.getLogger(__name__)


class KNNRecommender(BaseRecommender):
    """Filtragem colaborativa user-based via K vizinhos mais proximos.

    Constroi uma matriz usuario-item esparsa e calcula a similaridade de
    cosseno entre usuarios para agregar os ratings dos vizinhos como score
    ponderado.

    Parameters
    ----------
    k :
        Numero de vizinhos a considerar na agregacao.
    """

    def __init__(self, k: int = 20) -> None:
        self.k = k

        self._user_index: dict[str, int] = {}
        self._item_index: dict[str, int] = {}
        self._items: list[str] = []
        self._matrix: csr_matrix | None = None
        self._popular_items: list[str] = []

    # ------------------------------------------------------------------
    # Interface BaseRecommender
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> "KNNRecommender":
        """Treina o modelo KNN e registra experimento no MLflow.

        Parameters
        ----------
        data :
            DataFrame com colunas ``user_id``, ``product_id``, ``rating``.

        Returns
        -------
        KNNRecommender
            Retorna ``self`` para permitir encadeamento.
        """
        users = data["user_id"].unique().tolist()
        items = data["product_id"].unique().tolist()

        self._user_index = {u: i for i, u in enumerate(users)}
        self._item_index = {it: j for j, it in enumerate(items)}
        self._items = items

        # Popularidade para cold start (ordem de frequencia decrescente)
        self._popular_items = data["product_id"].value_counts().index.tolist()

        rows = data["user_id"].map(self._user_index).values
        cols = data["product_id"].map(self._item_index).values
        ratings = data["rating"].values.astype(np.float32)

        self._matrix = csr_matrix(
            (ratings, (rows, cols)),
            shape=(len(users), len(items)),
        )

        with mlflow.start_run(run_name="knn-fit", nested=True):
            mlflow.log_params({"k": self.k})

        self._is_fitted = True
        logger.info("KNN treinado — %d usuarios, %d itens", len(users), len(items))
        return self

    def predict(self, user_id: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Retorna os top-K itens recomendados para o usuario.

        Cold start: usuario desconhecido recebe os itens mais populares
        sem lancar excecao.

        Parameters
        ----------
        user_id :
            Identificador do usuario.
        top_k :
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

        u_idx = self._user_index[user_id]
        user_vec = self._matrix[u_idx]  # (1, n_items)

        # Similaridade de cosseno entre o usuario alvo e todos os outros
        sims = cosine_similarity(user_vec, self._matrix).flatten()  # (n_users,)
        sims[u_idx] = -1.0  # exclui o proprio usuario

        # Seleciona os k vizinhos mais similares
        n_neighbors = min(self.k, len(sims) - 1)
        neighbor_idx = np.argpartition(sims, -n_neighbors)[-n_neighbors:]

        neighbor_matrix = self._matrix[neighbor_idx].toarray()  # (k, n_items)
        weights = sims[neighbor_idx]  # (k,)

        # Score ponderado pela similaridade
        scores = (neighbor_matrix * weights[:, np.newaxis]).sum(axis=0)  # (n_items,)

        # Zera itens que o usuario alvo ja avaliou
        rated_cols = self._matrix[u_idx].nonzero()[1]
        scores[rated_cols] = -np.inf

        # Remove itens com score invalido antes do top-k
        valid_mask = np.isfinite(scores) & (scores > -np.inf)
        n_valid = int(valid_mask.sum())
        effective_k = min(top_k, n_valid)

        if effective_k == 0:
            return self._popular_fallback(top_k)

        top_indices = np.argpartition(scores, -effective_k)[-effective_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        # Normaliza min-max para [0, 1]
        top_scores = scores[top_indices]
        s_min, s_max = top_scores.min(), top_scores.max()
        if s_max > s_min:
            top_scores = (top_scores - s_min) / (s_max - s_min)
        else:
            top_scores = np.ones(len(top_indices))

        return [
            {"product_id": self._items[i], "score": float(top_scores[j])}
            for j, i in enumerate(top_indices)
        ]

    def evaluate(self, test_data: pd.DataFrame) -> dict[str, float]:
        """Calcula metricas de avaliacao no conjunto de teste.

        Parameters
        ----------
        test_data :
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
        path :
            Diretorio de destino (criado se nao existir).
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "knn.pkl", "wb") as f:
            pickle.dump(self, f)
        logger.info("KNNRecommender salvo em %s", path)

    @classmethod
    def load(cls, path: Path) -> "KNNRecommender":
        """Carrega o modelo do disco.

        Parameters
        ----------
        path :
            Diretorio onde o modelo foi salvo por :meth:`save`.
        """
        with open(Path(path) / "knn.pkl", "rb") as f:
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
