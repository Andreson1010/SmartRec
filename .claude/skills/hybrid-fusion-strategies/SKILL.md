---
name: hybrid-fusion-strategies
description: Padrão para implementar o HybridRecommender no SmartRec, combinando filtragem colaborativa (CF) com busca semântica. Use esta skill sempre que for criar ou modificar ml/hybrid/recommender.py, implementar estratégias de fusão (weighted average ou Reciprocal Rank Fusion), tunear o parâmetro alpha, ou registrar o modelo híbrido no MLflow Model Registry.
---

# Skill: hybrid-fusion-strategies

Padrão para combinar scores do modelo colaborativo (CF) e semântico no SmartRec.

## Localização

```
ml/hybrid/
├── recommender.py   ← HybridRecommender (modelo de produção)
└── artifacts/       ← modelo serializado
```

## Estratégias de fusão disponíveis

| Estratégia       | Quando usar                                                       |
|------------------|-------------------------------------------------------------------|
| `weighted`       | Padrão. Alpha fixo tunado offline. Simples e interpretável.       |
| `rank_fusion`    | Quando as escalas dos scores forem muito diferentes.              |
| `learned_alpha`  | Quando houver dados suficientes para aprender o peso dinamicamente.|

## Implementação (`recommender.py`)

```python
"""
ml/hybrid/recommender.py
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Literal

import mlflow
import numpy as np
import pandas as pd

from ml.collaborative.svd import SVDRecommender
from ml.semantic.retriever import SemanticRetriever

logger = logging.getLogger(__name__)

FusionStrategy = Literal["weighted", "rank_fusion"]


class HybridRecommender:
    """Combina filtragem colaborativa com busca semântica.

    Parameters
    ----------
    alpha:
        Peso do modelo colaborativo em ``[0, 1]``.
        ``1 - alpha`` é o peso do modelo semântico.
        Relevante apenas para strategy="weighted".
    strategy:
        Estratégia de fusão: ``"weighted"`` ou ``"rank_fusion"``.
    cf_model_path:
        Diretório do SVDRecommender serializado.
    embeddings_dir:
        Diretório com os embeddings de produtos.
    version:
        Versão do modelo (usada pelo endpoint da API).
    """

    def __init__(
        self,
        alpha: float = 0.6,
        strategy: FusionStrategy = "weighted",
        cf_model_path: Path = Path("ml/collaborative/artifacts"),
        embeddings_dir: Path = Path("data/embeddings"),
        version: str = "1.0.0",
    ) -> None:
        self.alpha        = alpha
        self.strategy     = strategy
        self.version      = version
        self._cf          = SVDRecommender.load(cf_model_path)
        self._semantic    = SemanticRetriever(embeddings_dir)

    # ------------------------------------------------------------------
    def predict(self, user_id: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Gera recomendações híbridas para o usuário.

        Parameters
        ----------
        user_id:
            Identificador do usuário.
        top_k:
            Número de recomendações a retornar.

        Returns
        -------
        list[dict]
            Lista de ``{"product_id": str, "score": float}`` ordenada por score desc.
        """
        cf_recs  = self._cf.predict(user_id, top_k=top_k * 3)    # candidatos extras
        # Semantic: busca itens similares ao top item do CF como semente
        seed_item = cf_recs[0]["product_id"] if cf_recs else None
        sem_recs  = (
            self._semantic.query_by_product(seed_item, top_k=top_k * 3)
            if seed_item else []
        )

        if self.strategy == "weighted":
            return self._weighted_fusion(cf_recs, sem_recs, top_k)
        return self._rank_fusion(cf_recs, sem_recs, top_k)

    # ------------------------------------------------------------------
    def _weighted_fusion(
        self,
        cf_recs: list[dict],
        sem_recs: list[dict],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Média ponderada de scores normalizados.

        score_final = alpha * score_cf + (1 - alpha) * score_semantic
        """
        scores: dict[str, float] = {}

        for item in cf_recs:
            scores[item["product_id"]] = self.alpha * item["score"]

        for item in sem_recs:
            pid = item["product_id"]
            scores[pid] = scores.get(pid, 0.0) + (1 - self.alpha) * item["score"]

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{"product_id": pid, "score": round(score, 4)} for pid, score in ranked[:top_k]]

    def _rank_fusion(
        self,
        cf_recs: list[dict],
        sem_recs: list[dict],
        top_k: int,
        k: int = 60,
    ) -> list[dict[str, Any]]:
        """Reciprocal Rank Fusion (RRF).

        score_rrf = sum(1 / (k + rank_i))  para cada lista de origem.
        Não depende da escala dos scores — robusto quando CF e semântico têm escalas diferentes.
        """
        rrf_scores: dict[str, float] = {}

        for rank, item in enumerate(cf_recs, start=1):
            pid = item["product_id"]
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (k + rank)

        for rank, item in enumerate(sem_recs, start=1):
            pid = item["product_id"]
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (k + rank)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [{"product_id": pid, "score": round(score, 6)} for pid, score in ranked[:top_k]]

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
            Melhor alpha encontrado.
        """
        from ml.evaluation.metrics import ndcg_at_k

        alphas = alphas or [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        relevant = (
            val_data[val_data["rating"] >= 4]
            .groupby("user_id")["product_id"]
            .apply(list)
            .to_dict()
        )
        users = list(relevant.keys())[:200]   # amostra para velocidade

        best_alpha, best_ndcg = self.alpha, -1.0
        for alpha in alphas:
            self.alpha = alpha
            ndcgs = []
            for uid in users:
                recs = [r["product_id"] for r in self.predict(uid, top_k=k)]
                ndcgs.append(ndcg_at_k(recs, relevant.get(uid, []), k))
            mean_ndcg = float(np.mean(ndcgs))
            logger.info("alpha=%.1f  ndcg@%d=%.4f", alpha, k, mean_ndcg)
            if mean_ndcg > best_ndcg:
                best_ndcg  = mean_ndcg
                best_alpha = alpha

        self.alpha = best_alpha
        logger.info("Melhor alpha: %.1f  (ndcg@%d=%.4f)", best_alpha, k, best_ndcg)
        return best_alpha

    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "hybrid.pkl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "HybridRecommender":
        with open(Path(path) / "hybrid.pkl", "rb") as f:
            return pickle.load(f)
```

## Guia de escolha de alpha

| Cenário                            | Alpha recomendado |
|------------------------------------|-------------------|
| Usuários com muitas interações     | 0.7 – 0.8         |
| Usuários com poucas interações     | 0.3 – 0.5         |
| Cold start (sem histórico)         | 0.0 (só semântico)|
| Baseline inicial                   | 0.6               |

## Checklist

- [ ] `predict` chama CF e semântico com `top_k * 3` candidatos antes da fusão
- [ ] `_weighted_fusion`: scores normalizados para `[0, 1]` antes de combinar
- [ ] `_rank_fusion`: usar quando as escalas dos dois modelos forem incompatíveis
- [ ] `tune_alpha` executado no conjunto de validação, resultado logado no MLflow
- [ ] `version` exposta e retornada pelo endpoint da API
- [ ] Cold start completo: usuário sem histórico e sem item semente usa semântico puro
- [ ] Testes mockam `SVDRecommender.load` e `SemanticRetriever` para isolar a fusão
- [ ] Modelo final registrado no MLflow Model Registry como `smartrec-hybrid`
