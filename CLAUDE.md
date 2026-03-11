# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SmartRec - Sistema de Recomendação Híbrido
FastAPI + React + ML (SVD + Sentence Transformers)

## Commands

```bash
# Instalar dependências
pip install -r requirements.txt

# Rodar testes com cobertura
pytest tests/ --cov=.

# Rodar API local
uvicorn api.main:app --reload

# Rodar frontend
cd frontend && npm run dev

# Subir tudo via Docker
docker-compose up

# Run MLflow tracking server
mlflow server --host 0.0.0.0 --port 5000
```

## Architecture

- **`ml/`** — modelos de ML
  - `collaborative/` — filtragem colaborativa (SVD / KNN)
  - `semantic/` — similaridade semântica via Sentence Transformers; embeddings em `data/embeddings/`
  - `hybrid/` — combina scores dos dois modelos; modelo de produção
  - `evaluation/` — métricas compartilhadas (Precision@K, Recall@K, NDCG, MRR)
- **`api/`** — FastAPI REST API
  - `routers/` — definição das rotas
  - `models/` — schemas Pydantic de request/response
  - `services/` — lógica de negócio que chama `ml/hybrid/`
- **`frontend/`** — React + Tailwind SPA
- **`data/`** — datasets e embeddings
  - `raw/` → `data/processed/` + `data/embeddings/` → treinamento → inferência via API

## Conventions

- Type hints obrigatórios em todo código Python
- Docstrings em todas as classes e funções públicas
- Testes para toda nova classe em `tests/`
- Black para formatação, flake8 para lint
