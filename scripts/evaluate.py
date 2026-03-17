"""
scripts/evaluate.py
--------------------
Avaliação comparativa de todos os modelos do SmartRec no dataset real.

Etapas
------
1. Carrega interactions.parquet
2. Split temporal 70 / 15 / 15 (train / val / test)
3. Treina e avalia SVDRecommender
4. Treina e avalia KNNRecommender
5. Treina e avalia HybridRecommender (reutiliza SVD;
   pula embeddings se --skip-embeddings)
6. Loga todas as métricas no MLflow (experimento smartrec/evaluation)
7. Imprime tabela comparativa

Uso
---
python -m scripts.evaluate
python -m scripts.evaluate --n-factors 100 --knn-k 30 --skip-embeddings
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from ml.collaborative.knn import KNNRecommender
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
    df :
        DataFrame com coluna ``timestamp``.
    train_ratio :
        Fração destinada ao treino.
    val_ratio :
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


def compute_metrics(
    model: SVDRecommender | KNNRecommender | HybridRecommender,
    test_data: pd.DataFrame,
    k: int = 10,
) -> dict[str, float]:
    """Calcula precision@k, recall@k, ndcg@k e mrr para um modelo.

    Parameters
    ----------
    model :
        Modelo já treinado com interface ``predict(user_id, top_k)``.
    test_data :
        DataFrame com colunas ``user_id``, ``product_id``, ``rating``.
    k :
        Corte de avaliação.

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


def _print_table(results: dict[str, dict[str, float]]) -> None:
    """Imprime tabela comparativa de métricas."""
    metrics = ["precision_at_10", "recall_at_10", "ndcg_at_10", "mrr"]
    col_w = 16
    header = f"{'Model':<12}" + "".join(f"{m:>{col_w}}" for m in metrics)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for model_name, m in results.items():
        row = f"{model_name:<12}" + "".join(
            f"{m.get(metric, 0.0):>{col_w}.4f}" for metric in metrics
        )
        print(row)
    print("=" * len(header) + "\n")


# ---------------------------------------------------------------------------
# Pipeline de avaliação
# ---------------------------------------------------------------------------


def evaluate(
    n_factors: int = 50,
    knn_k: int = 20,
    strategy: str = "weighted",
    skip_embeddings: bool = False,
    experiment_name: str = "smartrec/evaluation",
) -> dict[str, dict[str, float]]:
    """Avalia SVD, KNN e Hybrid no dataset real e loga no MLflow.

    Parameters
    ----------
    n_factors :
        Número de fatores latentes do SVD.
    knn_k :
        Número de vizinhos do KNN.
    strategy :
        Estratégia de fusão do Hybrid (``"weighted"`` ou ``"rank_fusion"``).
    skip_embeddings :
        Se ``True``, reutiliza embeddings já salvos em ``data/embeddings/``.
    experiment_name :
        Nome do experimento MLflow.

    Returns
    -------
    dict
        ``{"svd": {...}, "knn": {...}, "hybrid": {...}}`` com métricas de cada modelo.
    """
    logger.info("Carregando dados processados de %s", PROCESSED_DIR)
    interactions = pd.read_parquet(PROCESSED_DIR / "interactions.parquet")
    products = pd.read_parquet(PROCESSED_DIR / "products.parquet")

    logger.info("Split temporal 70/15/15 (%d interações)...", len(interactions))
    train_df, val_df, test_df = temporal_split(interactions)
    logger.info(
        "Tamanhos — train=%d  val=%d  test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    mlflow.set_experiment(experiment_name)
    run_name = f"eval_{pd.Timestamp.now():%Y%m%d_%H%M%S}"

    results: dict[str, dict[str, float]] = {}

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "model_type": "comparison",
                "dataset_version": "v1",
                "git_commit": _git_sha(),
                "strategy": strategy,
            }
        )
        mlflow.log_params(
            {
                "n_factors": n_factors,
                "knn_k": knn_k,
                "strategy": strategy,
                "train_size": len(train_df),
                "val_size": len(val_df),
                "test_size": len(test_df),
            }
        )

        # --- SVD ---
        logger.info("Treinando SVDRecommender (n_factors=%d)...", n_factors)
        svd = SVDRecommender(n_factors=n_factors)
        svd.fit(train_df)
        svd.save(CF_ARTIFACTS)
        svd_metrics = compute_metrics(svd, test_df)
        results["svd"] = svd_metrics
        mlflow.log_metrics({f"svd_{k}": v for k, v in svd_metrics.items()})
        logger.info("SVD test: %s", svd_metrics)

        # --- KNN ---
        logger.info("Treinando KNNRecommender (k=%d)...", knn_k)
        knn = KNNRecommender(k=knn_k)
        knn.fit(train_df)
        knn.save(CF_ARTIFACTS)
        knn_metrics = compute_metrics(knn, test_df)
        results["knn"] = knn_metrics
        mlflow.log_metrics({f"knn_{k}": v for k, v in knn_metrics.items()})
        logger.info("KNN test: %s", knn_metrics)

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
                "Embeddings shape=%s salvos em %s", embeddings.shape, EMBEDDINGS_DIR
            )

        # --- Hybrid ---
        logger.info("Configurando HybridRecommender (strategy=%s)...", strategy)
        hybrid = HybridRecommender(
            strategy=strategy,
            cf_model_path=CF_ARTIFACTS,
            embeddings_dir=EMBEDDINGS_DIR,
        )
        best_alpha = hybrid.tune_alpha(val_df)
        mlflow.log_param("best_alpha", best_alpha)
        logger.info("Melhor alpha: %.2f", best_alpha)

        hybrid_metrics = compute_metrics(hybrid, test_df)
        results["hybrid"] = hybrid_metrics
        mlflow.log_metrics({f"hybrid_{k}": v for k, v in hybrid_metrics.items()})
        logger.info("Hybrid test: %s", hybrid_metrics)

        # --- Salvar artefatos ---
        HYBRID_ARTIFACTS.mkdir(parents=True, exist_ok=True)
        hybrid.save(HYBRID_ARTIFACTS)
        mlflow.log_artifact(
            str(HYBRID_ARTIFACTS / "hybrid.pkl"), artifact_path="hybrid_model"
        )
        mlflow.log_artifact(str(CF_ARTIFACTS / "svd.pkl"), artifact_path="svd_model")
        mlflow.log_artifact(str(CF_ARTIFACTS / "knn.pkl"), artifact_path="knn_model")

        logger.info("Run ID: %s", run.info.run_id)

    return results


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Entrypoint CLI do pipeline de avaliação."""
    parser = argparse.ArgumentParser(
        description="SmartRec — avaliação comparativa de modelos"
    )
    parser.add_argument(
        "--n-factors",
        type=int,
        default=50,
        help="Fatores latentes do SVD (default: 50)",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=20,
        help="Número de vizinhos do KNN (default: 20)",
    )
    parser.add_argument(
        "--strategy",
        choices=["weighted", "rank_fusion"],
        default="weighted",
        help="Estratégia de fusão do Hybrid (default: weighted)",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Reutiliza embeddings já gerados em data/embeddings/",
    )
    parser.add_argument(
        "--experiment",
        default="smartrec/evaluation",
        help="Nome do experimento MLflow (default: smartrec/evaluation)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    results = evaluate(
        n_factors=args.n_factors,
        knn_k=args.knn_k,
        strategy=args.strategy,
        skip_embeddings=args.skip_embeddings,
        experiment_name=args.experiment,
    )

    _print_table(results)


if __name__ == "__main__":
    main()
