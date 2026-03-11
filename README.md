# SmartRec - Sistema de Recomendação Híbrido

## Overview

SmartRec é um sistema de recomendação híbrido que combina filtragem colaborativa com busca semântica por embeddings para gerar recomendações mais precisas e contextualizadas. O sistema é servido via API REST (FastAPI) e tem experimentos rastreados com MLflow.

## Architecture

```
smartrec/
├── data/
│   ├── raw/            # Dados brutos (interações, metadados de itens)
│   ├── processed/      # Dados após pré-processamento
│   └── embeddings/     # Vetores gerados pelos modelos semânticos
├── ml/
│   ├── collaborative/  # Filtragem colaborativa (matrix factorization, KNN)
│   ├── semantic/       # Recomendação por similaridade semântica (embeddings)
│   ├── hybrid/         # Combinação dos modelos colaborativo + semântico
│   └── evaluation/     # Métricas: Precision@K, Recall@K, NDCG, MRR
├── api/
│   ├── routers/        # Endpoints FastAPI
│   ├── models/         # Schemas Pydantic
│   └── services/       # Lógica de negócio / orquestração
├── frontend/           # Interface de demonstração
├── mlflow/             # Configuração do servidor MLflow
├── docker/             # Dockerfiles e docker-compose
└── .github/workflows/  # CI/CD pipelines
```

**Fluxo de dados:**
1. Dados brutos ingeridos em `data/raw/`
2. Pré-processamento gera artefatos em `data/processed/` e `data/embeddings/`
3. Modelos treinados em `ml/collaborative/` e `ml/semantic/`, combinados em `ml/hybrid/`
4. API em `api/` serve predições usando os modelos treinados
5. Experimentos e métricas registrados no MLflow

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
