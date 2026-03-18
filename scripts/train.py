"""
scripts/train.py
----------------
Pipeline de treinamento do SmartRec.

Etapas
------
1. Carrega interactions.parquet e products.parquet
2. Split temporal 70 / 15 / 15 (train / val / test)
3. Treina SVDRecommender no conjunto de treino
4. Gera embeddings com ProductEmbedder (pulável com --skip-embeddings)
5. Tune alpha do HybridRecommender no conjunto de validação
6. Avalia HybridRecommender no conjunto de teste
7. Loga tudo no MLflow e registra no Model Registry

Uso
---
python -m scripts.train
python -m scripts.train --n-factors 100 --strategy rank_fusion --skip-embeddings
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from ml.collaborative.svd import SVDRecommender
from ml.evaluation.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k
from ml.hybrid.recommender import HybridRecommender
from ml.semantic.embedder import ProductEmbedder

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
CF_ARTIFACTS = ROOT / "ml" / "collaborative" / "artifacts"
HYBRID_ARTIFACTS = ROOT / "ml" / "hybrid" / "artifacts"
EMBEDDINGS_DIR = ROOT / "data" / "embeddings"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split temporal de interações em treino / validação / teste.

    Parameters
    ----------
    df:
        DataFrame com coluna ``timestamp`` (int ou datetime).
    train_ratio:
        Fração destinada ao treino.
    val_ratio:
        Fração destinada à validação. O restante vira teste.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train, val, test)`` sem sobreposição de datas.
    """
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return (
        df_sorted.iloc[:train_end].copy(),
        df_sorted.iloc[train_end:val_end].copy(),
        df_sorted.iloc[val_end:].copy(),
    )


def evaluate_hybrid(
    model: HybridRecommender,
    test_data: pd.DataFrame,
    k: int = 10,
) -> dict[str, float]:
    """Calcula métricas do HybridRecommender no conjunto de teste.

    Parameters
    ----------
    model:
        Modelo híbrido já treinado.
    test_data:
        DataFrame com colunas ``user_id``, ``product_id``, ``rating``.
    k:
        Corte de avaliação (top-k).

    Returns
    -------
    dict
        Chaves: ``precision_at_10``, ``recall_at_10``, ``ndcg_at_10``, ``mrr``.
    """
    relevant_by_user: dict[str, list[str]] = {}
    for row in test_data.itertuples(index=False):
        if row.rating >= 4.0:
            relevant_by_user.setdefault(row.user_id, []).append(row.product_id)

    p_vals, r_vals, nd_vals, mrr_vals = [], [], [], []
    for uid, relevant in relevant_by_user.items():
        recs = [d["product_id"] for d in model.predict(uid, top_k=k)]
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


def _maybe_register(
    run_id: str,
    metrics: dict[str, float],
    threshold: dict[str, float],
) -> None:
    """Registra no Model Registry se as métricas passam no threshold.

    Parameters
    ----------
    run_id:
        ID do run MLflow ativo.
    metrics:
        Métricas avaliadas no conjunto de teste.
    threshold:
        Valores mínimos aceitáveis por métrica.
    """
    if all(metrics.get(m, 0.0) >= v for m, v in threshold.items()):
        mlflow.register_model(
            model_uri=f"runs:/{run_id}/hybrid_model",
            name="smartrec-hybrid",
        )
        logger.info("Modelo registrado no MLflow Model Registry como smartrec-hybrid")
    else:
        logger.info("Métricas abaixo do threshold — modelo não registrado")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------


def train(
    n_factors: int = 50,
    strategy: str = "rank_fusion",
    skip_embeddings: bool = False,
    experiment_name: str = "smartrec/hybrid",
    register: bool = True,
) -> dict[str, float]:
    """Pipeline completo de treinamento do SmartRec.

    Parameters
    ----------
    n_factors:
        Número de fatores latentes do SVD.
    strategy:
        Estratégia de fusão do Hybrid (``"weighted"`` ou ``"rank_fusion"``).
    skip_embeddings:
        Se ``True``, pula a geração de embeddings (reutiliza os já salvos).
    experiment_name:
        Nome do experimento MLflow.
    register:
        Se ``True``, tenta registrar no Model Registry ao final.

    Returns
    -------
    dict
        Métricas finais do conjunto de teste.
    """
    # --- Carregar dados ---
    logger.info("Carregando dados processados de %s", PROCESSED_DIR)
    interactions = pd.read_parquet(PROCESSED_DIR / "interactions.parquet")
    products = pd.read_parquet(PROCESSED_DIR / "products.parquet")

    # --- Split temporal ---
    logger.info("Split temporal 70/15/15...")
    train_df, val_df, test_df = temporal_split(interactions)
    logger.info(
        "Tamanhos — train=%d  val=%d  test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    # --- MLflow ---
    mlflow.set_experiment(experiment_name)
    run_name = f"hybrid_{pd.Timestamp.now():%Y%m%d_%H%M%S}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "model_type": "hybrid",
                "dataset_version": "v1",
                "git_commit": _git_sha(),
                "strategy": strategy,
            }
        )
        mlflow.log_params(
            {
                "n_factors": n_factors,
                "strategy": strategy,
                "train_size": len(train_df),
                "val_size": len(val_df),
                "test_size": len(test_df),
            }
        )

        # --- Treinar SVD ---
        logger.info("Treinando SVDRecommender (n_factors=%d)...", n_factors)
        svd = SVDRecommender(n_factors=n_factors)
        svd.fit(train_df)
        svd.save(CF_ARTIFACTS)

        svd_val_metrics = svd.evaluate(val_df)
        mlflow.log_metrics({f"svd_val_{k}": v for k, v in svd_val_metrics.items()})
        logger.info("SVD val: %s", svd_val_metrics)

        # --- Embeddings ---
        if skip_embeddings:
            logger.info("Pulando geração de embeddings (--skip-embeddings ativo)")
        else:
            logger.info("Gerando embeddings semânticos...")
            embedder = ProductEmbedder()
            embeddings = embedder.fit_transform(products)
            embedder.save(embeddings, products["product_id"], EMBEDDINGS_DIR)
            mlflow.log_param("embeddings_shape", str(embeddings.shape))
            logger.info(
                "Embeddings gerados: shape=%s salvos em %s",
                embeddings.shape,
                EMBEDDINGS_DIR,
            )

        # --- Hybrid: tune alpha ---
        logger.info("Tunando alpha do HybridRecommender no val set...")
        hybrid = HybridRecommender(
            strategy=strategy,
            cf_model_path=CF_ARTIFACTS,
            embeddings_dir=EMBEDDINGS_DIR,
            train_interactions=train_df,
        )
        best_alpha = hybrid.tune_alpha(val_df)
        mlflow.log_param("best_alpha", best_alpha)
        logger.info("Melhor alpha: %.2f", best_alpha)

        # --- Avaliar no test ---
        logger.info("Avaliando HybridRecommender no conjunto de teste...")
        test_metrics = evaluate_hybrid(hybrid, test_df)
        mlflow.log_metrics(test_metrics)
        logger.info("Test metrics: %s", test_metrics)

        # --- Salvar artefatos ---
        HYBRID_ARTIFACTS.mkdir(parents=True, exist_ok=True)
        hybrid.save(HYBRID_ARTIFACTS)
        mlflow.log_artifact(
            str(HYBRID_ARTIFACTS / "hybrid.pkl"), artifact_path="hybrid_model"
        )
        mlflow.log_artifact(str(CF_ARTIFACTS / "svd.pkl"), artifact_path="svd_model")

        # --- Registrar no Model Registry ---
        if register:
            _maybe_register(
                run.info.run_id,
                test_metrics,
                threshold={"ndcg_at_10": 0.01},
            )

    return test_metrics


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Entrypoint CLI do pipeline de treino."""
    parser = argparse.ArgumentParser(
        description="SmartRec — pipeline completo de treinamento"
    )
    parser.add_argument(
        "--n-factors",
        type=int,
        default=50,
        help="Fatores latentes do SVD (default: 50)",
    )
    parser.add_argument(
        "--strategy",
        choices=["weighted", "rank_fusion"],
        default="rank_fusion",
        help="Estratégia de fusão do Hybrid (default: rank_fusion)",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Reutiliza embeddings já gerados em data/embeddings/",
    )
    parser.add_argument(
        "--experiment",
        default="smartrec/hybrid",
        help="Nome do experimento MLflow (default: smartrec/hybrid)",
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Não registra o modelo no MLflow Model Registry",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    metrics = train(
        n_factors=args.n_factors,
        strategy=args.strategy,
        skip_embeddings=args.skip_embeddings,
        experiment_name=args.experiment,
        register=not args.no_register,
    )

    print("\n=== Métricas finais (test set) ===")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
