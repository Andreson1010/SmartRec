"""
ml/hybrid/recommender.py
------------------------
HybridRecommender: combina filtragem colaborativa (SVD) com busca semântica.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from ml.collaborative.svd import SVDRecommender
from ml.semantic.retriever import SemanticRetriever

_EMPTY_USER_ITEMS: dict[str, list[str]] = {}

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
FusionStrategy = Literal["weighted", "rank_fusion"]


class HybridRecommender:
    """Combina filtragem colaborativa com busca semântica.

    Parameters
    ----------
    alpha:
        Peso do modelo colaborativo em ``[0, 1]``.
        ``1 - alpha`` é o peso do modelo semântico.
        Relevante apenas para ``strategy="weighted"``.
    strategy:
        Estratégia de fusão: ``"weighted"`` ou ``"rank_fusion"``.
    cf_model_path:
        Diretório do SVDRecommender serializado.
    embeddings_dir:
        Diretório com os embeddings de produtos.
    version:
        Versão do modelo (retornada pelo endpoint da API).
    """

    def __init__(
        self,
        alpha: float = 0.6,
        strategy: FusionStrategy = "rank_fusion",
        cf_model_path: Path = ROOT / "ml" / "collaborative" / "artifacts",
        embeddings_dir: Path = ROOT / "data" / "embeddings",
        version: str = "1.0.0",
        train_interactions: pd.DataFrame | None = None,
    ) -> None:
        self.alpha = alpha
        self.strategy = strategy
        self.version = version
        self._cf: SVDRecommender = SVDRecommender.load(cf_model_path)
        self._semantic: SemanticRetriever = SemanticRetriever(embeddings_dir)
        # Mapeia user_id → lista de product_ids do histórico de treino
        if train_interactions is not None:
            self._user_items: dict[str, list[str]] = (
                train_interactions.groupby("user_id")["product_id"]
                .apply(list)
                .to_dict()
            )
        else:
            self._user_items = _EMPTY_USER_ITEMS

    # ------------------------------------------------------------------
    def predict(self, user_id: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Gera recomendações híbridas para o usuário.

        Obtém candidatos do CF e do modelo semântico (usando o top item do CF
        como semente), depois funde os scores pela estratégia configurada.

        Cold start completo (usuário sem histórico e sem item semente) delega
        inteiramente ao modelo semântico via vetor zero como fallback.

        Parameters
        ----------
        user_id:
            Identificador do usuário.
        top_k:
            Número de recomendações a retornar.

        Returns
        -------
        list[dict]
            Lista de ``{"product_id": str, "score": float}`` ordenada por
            score decrescente.
        """
        cf_recs = self._cf.predict(user_id, top_k=top_k * 3)

        # Semente semântica: média dos embeddings do histórico do usuário.
        # Captura preferência real do usuário em vez de similaridade de
        # conteúdo com um único item semente.
        sem_recs = self._query_user_history(user_id, top_k=top_k * 3, cf_recs=cf_recs)

        if self.strategy == "weighted":
            return self._weighted_fusion(cf_recs, sem_recs, top_k)
        return self._rank_fusion(cf_recs, sem_recs, top_k)

    # ------------------------------------------------------------------
    def _query_user_history(
        self,
        user_id: str,
        top_k: int,
        cf_recs: list[dict],
    ) -> list[dict]:
        """Consulta semântica via média dos embeddings do histórico do usuário.

        Tenta construir um vetor de consulta como média (normalizada) dos
        embeddings de todos os produtos avaliados pelo usuário no treino.
        Se nenhum produto do histórico estiver indexado, faz fallback para
        o top item do CF como semente (comportamento anterior).

        Parameters
        ----------
        user_id:
            Identificador do usuário.
        top_k:
            Número de resultados semânticos a buscar.
        cf_recs:
            Recomendações do modelo colaborativo (usado no fallback).

        Returns
        -------
        list[dict]
            Lista de ``{"product_id": str, "score": float}``.
        """
        history = self._user_items.get(user_id, [])
        if history:
            vecs = [
                self._semantic.get_embedding(pid)
                for pid in history
            ]
            vecs = [v for v in vecs if v is not None]
            if vecs:
                mean_vec = np.mean(vecs, axis=0).astype("float32")
                norm = np.linalg.norm(mean_vec)
                if norm > 0:
                    mean_vec /= norm
                    return self._semantic.query_by_vector(mean_vec, top_k=top_k)

        # Fallback: top item do CF como semente
        seed_item = cf_recs[0]["product_id"] if cf_recs else None
        if seed_item:
            return self._semantic.query_by_product(seed_item, top_k=top_k)
        return []

    # ------------------------------------------------------------------
    def _weighted_fusion(
        self,
        cf_recs: list[dict],
        sem_recs: list[dict],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Média ponderada de scores: alpha * CF + (1-alpha) * semântico."""
        scores: dict[str, float] = {}

        for item in cf_recs:
            scores[item["product_id"]] = self.alpha * item["score"]

        for item in sem_recs:
            pid = item["product_id"]
            scores[pid] = scores.get(pid, 0.0) + (1 - self.alpha) * item["score"]

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {"product_id": pid, "score": round(score, 4)}
            for pid, score in ranked[:top_k]
        ]

    def _rank_fusion(
        self,
        cf_recs: list[dict],
        sem_recs: list[dict],
        top_k: int,
        k: int = 60,
    ) -> list[dict[str, Any]]:
        """Reciprocal Rank Fusion (RRF): score = sum(1 / (k + rank_i)).

        Robusto a diferenças de escala entre CF e semântico.
        """
        rrf_scores: dict[str, float] = {}

        for rank, item in enumerate(cf_recs, start=1):
            pid = item["product_id"]
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (k + rank)

        for rank, item in enumerate(sem_recs, start=1):
            pid = item["product_id"]
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (k + rank)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {"product_id": pid, "score": round(score, 6)}
            for pid, score in ranked[:top_k]
        ]

    # ------------------------------------------------------------------
    def tune_alpha(
        self,
        val_data: pd.DataFrame,
        alphas: list[float] | None = None,
        k: int = 10,
    ) -> float:
        """Busca em grade para encontrar o melhor alpha no conjunto de validação.

        Parameters
        ----------
        val_data:
            DataFrame com colunas ``user_id``, ``product_id``, ``rating``.
        alphas:
            Valores de alpha a testar. Default: ``[0.3, 0.4, 0.5, 0.6, 0.7, 0.8]``.
        k:
            Tamanho do corte para NDCG.

        Returns
        -------
        float
            Melhor alpha encontrado (também atualiza ``self.alpha``).
        """
        from ml.evaluation.metrics import ndcg_at_k

        alphas = alphas or [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        relevant = (
            val_data[val_data["rating"] >= 4]
            .groupby("user_id")["product_id"]
            .apply(list)
            .to_dict()
        )
        users = list(relevant.keys())[:200]  # amostra para velocidade

        best_alpha, best_ndcg = self.alpha, -1.0
        for alpha in alphas:
            self.alpha = alpha
            ndcgs = [
                ndcg_at_k(
                    [r["product_id"] for r in self.predict(uid, top_k=k)],
                    relevant.get(uid, []),
                    k,
                )
                for uid in users
            ]
            mean_ndcg = float(np.mean(ndcgs))
            logger.info("alpha=%.1f  ndcg@%d=%.4f", alpha, k, mean_ndcg)
            if mean_ndcg > best_ndcg:
                best_ndcg = mean_ndcg
                best_alpha = alpha

        self.alpha = best_alpha
        logger.info("Melhor alpha: %.1f  (ndcg@%d=%.4f)", best_alpha, k, best_ndcg)
        return best_alpha

    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        """Serializa o modelo em disco.

        Parameters
        ----------
        path:
            Diretório de destino (criado se não existir).
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "hybrid.pkl", "wb") as f:
            pickle.dump(self, f)
        logger.info("HybridRecommender salvo em %s", path)

    @classmethod
    def load(cls, path: Path) -> "HybridRecommender":
        """Carrega o modelo do disco.

        Parameters
        ----------
        path:
            Diretório onde ``hybrid.pkl`` foi salvo por :meth:`save`.
        """
        with open(Path(path) / "hybrid.pkl", "rb") as f:
            model = pickle.load(f)
        logger.info("HybridRecommender carregado de %s", path)
        return model
