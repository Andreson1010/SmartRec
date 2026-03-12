# SmartRec - Sistema de Recomendação Híbrido

## Overview

SmartRec é um sistema de recomendação híbrido que combina filtragem colaborativa com busca semântica por embeddings para gerar recomendações mais precisas e contextualizadas. O sistema é servido via API REST (FastAPI) e tem experimentos rastreados com MLflow.

## Architecture

```
smartrec/
├── data/
│   ├── processing.py   # pipeline: raw → interactions/products/users.parquet
│   ├── eda.py          # análise exploratória → reports/figures/
│   ├── raw/            # dados brutos (reviews.jsonl — não versionado)
│   ├── processed/      # interactions.parquet, products.parquet, users.parquet
│   └── embeddings/     # embeddings.npy + product_ids.npy (não versionados)
├── ml/
│   ├── collaborative/  # SVDRecommender, KNNRecommender (via scikit-surprise)
│   ├── semantic/       # ProductEmbedder + SemanticRetriever (Sentence Transformers)
│   ├── hybrid/         # HybridRecommender — modelo de produção (weighted fusion / RRF)
│   └── evaluation/     # precision_at_k, recall_at_k, ndcg_at_k, mrr
├── api/
│   ├── main.py         # app FastAPI
│   ├── routers/        # endpoints (router não importa ml/ diretamente)
│   ├── models/         # schemas Pydantic request/response
│   └── services/       # orquestração → chama ml/hybrid/
├── notebooks/
│   └── 01_eda.ipynb    # exploração interativa
├── reports/figures/    # PNGs gerados por eda.py (não versionados)
├── mlflow/             # configuração do servidor MLflow
└── docker/             # Dockerfiles e docker-compose
```

**Fluxo de dados:**

```
data/raw/
    └─► data/processing.py ──► data/processed/*.parquet
                                        │
                    ┌───────────────────┴────────────────────┐
                    ▼                                        ▼
          ml/collaborative/                         ml/semantic/
          SVD + KNN (Surprise)               ProductEmbedder (all-MiniLM-L6-v2)
          artifacts/svd.pkl                 data/embeddings/*.npy
                    │                                        │
                    └───────────────┬────────────────────────┘
                                    ▼
                             ml/hybrid/
                         HybridRecommender
                       (alpha * CF + (1-alpha) * semântico)
                         artifacts/hybrid.pkl
                                    │
                                    ▼
                          api/ → POST /recommendations
```

Experimentos e métricas (`precision@10`, `recall@10`, `ndcg@10`, `mrr`) registrados no MLflow em `smartrec/collaborative`, `smartrec/semantic` e `smartrec/hybrid`.

## Setup

```bash
# Clonar o repositório
git clone <repo-url>
cd smartrec

# Criar e ativar ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt

# Configurar variáveis de ambiente
cp .env.example .env
# Edite .env com suas configurações
```

## Usage

```bash
# Subir o servidor MLflow
mlflow server --host 0.0.0.0 --port 5000

# Iniciar a API
uvicorn api.main:app --reload --port 8000

# Rodar os testes
pytest
```

## Results

| Modelo       | Precision@10 | Recall@10 | NDCG@10 |
|--------------|:------------:|:---------:|:-------:|
| Colaborativo | -            | -         | -       |
| Semântico    | -            | -         | -       |
| Híbrido      | -            | -         | -       |

_Tabela será preenchida após os primeiros experimentos._
