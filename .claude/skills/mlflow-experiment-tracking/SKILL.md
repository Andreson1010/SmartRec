---
name: mlflow-experiment-tracking
description: Padrão para rastreamento de experimentos com MLflow no SmartRec. Use esta skill sempre que for configurar mlflow.set_experiment(), registrar parâmetros, métricas e artefatos, definir tags (model_type, dataset_version, git_commit), promover modelos no Model Registry, ou comparar runs via MlflowClient. Também use ao criar scripts de treino (train.py) que integram MLflow.
---

# Skill: mlflow-experiment-tracking

Padrão para rastreamento de experimentos com MLflow no SmartRec.

## Convenções de nomenclatura

```
Experiment : smartrec/<componente>          ex: smartrec/collaborative, smartrec/hybrid
Run name   : <modelo>_<timestamp>           ex: svd_20260312_143000
Tags        : model_type, dataset_version, git_commit
```

## Estrutura de um experimento completo

```python
"""
ml/<submodule>/train.py
"""
from __future__ import annotations

import mlflow
import mlflow.sklearn          # ou mlflow.pytorch, mlflow.pyfunc
import pandas as pd
from pathlib import Path


EXPERIMENT_NAME = "smartrec/<componente>"


def train(interactions: pd.DataFrame, params: dict) -> None:
    """Treina o modelo e registra tudo no MLflow."""

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"<modelo>_{pd.Timestamp.now():%Y%m%d_%H%M%S}"):

        # 1. Tags de contexto
        mlflow.set_tags({
            "model_type":       "<tipo>",
            "dataset_version":  "v1",
            "git_commit":       _git_sha(),
        })

        # 2. Parâmetros (tudo que afeta o modelo)
        mlflow.log_params(params)

        # 3. Treino
        model = _fit(interactions, params)

        # 4. Métricas (no mínimo as 4 padrão do projeto)
        metrics = model.evaluate(interactions)   # precision@10, recall@10, ndcg@10, mrr
        mlflow.log_metrics(metrics)

        # 5. Artefatos
        mlflow.log_artifact("data/processed/interactions.parquet", artifact_path="dataset")
        mlflow.sklearn.log_model(model, artifact_path="model")   # ajustar conforme lib

        # 6. Registrar no Model Registry se for candidato a produção
        run_id = mlflow.active_run().info.run_id
        _maybe_register(run_id, metrics, threshold={"ndcg_at_10": 0.05})


def _maybe_register(run_id: str, metrics: dict, threshold: dict) -> None:
    """Registra no Model Registry se métricas passam no threshold."""
    if all(metrics.get(k, 0) >= v for k, v in threshold.items()):
        mlflow.register_model(
            model_uri=f"runs:/{run_id}/model",
            name="smartrec-<componente>",
        )


def _git_sha() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"
```

## Carregando um modelo do Registry

```python
import mlflow.pyfunc

# Última versão em produção
model = mlflow.pyfunc.load_model("models:/smartrec-hybrid/Production")

# Versão específica
model = mlflow.pyfunc.load_model("models:/smartrec-hybrid/3")
```

## Comparando runs via Python

```python
import mlflow

client = mlflow.MlflowClient()
experiment = client.get_experiment_by_name("smartrec/collaborative")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.ndcg_at_10 DESC"],
    max_results=5,
)

for run in runs:
    print(run.info.run_id, run.data.metrics)
```

## Métricas obrigatórias por componente

| Componente          | Métricas mínimas                              |
|---------------------|-----------------------------------------------|
| collaborative       | precision_at_10, recall_at_10, ndcg_at_10, mrr |
| semantic            | precision_at_10, recall_at_10, ndcg_at_10, mrr |
| hybrid              | todas acima + latency_ms_p99                  |

## Checklist

- [ ] `mlflow.set_experiment("smartrec/<componente>")` antes de qualquer run
- [ ] Tags: `model_type`, `dataset_version`, `git_commit`
- [ ] `log_params` com todos os hiperparâmetros
- [ ] `log_metrics` com as 4 métricas padrão
- [ ] Artefato do dataset logado para reprodutibilidade
- [ ] Modelo logado com `mlflow.<lib>.log_model`
- [ ] `_maybe_register` promove para Model Registry se bater threshold
- [ ] Testes mockam `mlflow.start_run`, `log_param`, `log_metrics`
