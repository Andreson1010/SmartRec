# SmartRec — Hybrid Recommendation System

![CI](https://github.com/Andreson1010/SmartRec/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-200%20passing-brightgreen)

Sistema de recomendação híbrido que combina **filtragem colaborativa** (SVD + KNN) com **busca semântica** (Sentence Transformers) para gerar recomendações precisas e contextualizadas, servidas via API REST.

---

## Como funciona

```text
data/raw/ (Amazon Reviews 2023 — 5.4M interações)
    │
    └─► data/processing.py ──► interactions.parquet
                                products.parquet
                                        │
                    ┌───────────────────┴────────────────────┐
                    ▼                                        ▼
          ml/collaborative/                         ml/semantic/
          SVDRecommender                       ProductEmbedder
          KNNRecommender                       (all-MiniLM-L6-v2)
          (scipy.sparse.svds)                  SemanticRetriever
                    │                                        │
                    └───────────────┬────────────────────────┘
                                    ▼
                             ml/hybrid/
                         HybridRecommender
                    estratégia rerank (CF → semântico re-ranker)
                                    │
                                    ▼
                   api/ → POST /recommendations/
                          GET  /products/{id}/similar
                          GET  /health
```

Experimentos rastreados com **MLflow** (`smartrec/collaborative`, `smartrec/semantic`, `smartrec/hybrid`).

---

## Stack

| Camada | Tecnologia |
| ------ | ---------- |
| API | FastAPI + uvicorn |
| ML — Colaborativo | scipy (`svds`, `cosine_similarity`) |
| ML — Semântico | sentence-transformers (`all-MiniLM-L6-v2`) |
| ML — Tracking | MLflow |
| Dados | pandas + pyarrow + numpy |
| Schemas | pydantic |
| Testes | pytest (200 testes) |
| Infra | Docker + Docker Compose + GitHub Actions CI |

---

## Quickstart

### Docker (recomendado)

```bash
git clone https://github.com/Andreson1010/SmartRec.git
cd SmartRec
cp .env.example .env
docker compose up
```

API disponível em `http://localhost:8000` · MLflow em `http://localhost:5000`

### Manual

```bash
git clone https://github.com/Andreson1010/SmartRec.git
cd SmartRec

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env

# Processar dados (requer reviews.jsonl em data/raw/)
python -m data.processing

# Treinar modelos
python -m scripts.train

# Iniciar API
uvicorn api.main:app --reload --port 8000
```

---

## Endpoints

| Método | Endpoint | Descrição |
| ------ | -------- | --------- |
| `POST` | `/recommendations/` | Recomendações híbridas para um usuário |
| `GET` | `/products/{id}/similar` | Produtos similares por embedding |
| `GET` | `/health` | Status da API e modelos carregados |

**Exemplo:**

```bash
curl -X POST http://localhost:8000/recommendations/ \
  -H "Content-Type: application/json" \
  -d '{"user_id": "ABC123", "n_recommendations": 10}'
```

```json
{
  "user_id": "ABC123",
  "recommendations": [
    {"product_id": "B08XYZ", "score": 0.92},
    {"product_id": "B07ABC", "score": 0.87}
  ]
}
```

---

## Resultados

> Execute `python -m scripts.train` para treinar os modelos e registrar as métricas no MLflow.

| Modelo | Precision@10 | Recall@10 | NDCG@10 | MRR |
| ------ | :----------: | :-------: | :-----: | :-: |
| SVD (val) | 0.0101 | 0.0309 | 0.0208 | 0.0257 |
| **Híbrido rerank (test)** | 0.0078 | 0.0236 | 0.0160 | 0.0202 |

> Dataset: Amazon Reviews 2023 — Electronics. 539k interações de treino, 9.321 produtos com
> embedding semântico de 14.795 únicos. Métricas baixas são esperadas em CF esparso com alta
> dimensionalidade de usuários (86k) e cold-start frequente.

---

## Estrutura do projeto

```text
smartrec/
├── data/
│   ├── processing.py       # raw → parquet
│   ├── eda.py              # análise exploratória
│   └── processed/          # .gitignored
├── ml/
│   ├── base.py             # BaseRecommender (ABC)
│   ├── collaborative/      # SVDRecommender, KNNRecommender
│   ├── semantic/           # ProductEmbedder, SemanticRetriever
│   ├── hybrid/             # HybridRecommender (produção)
│   └── evaluation/         # precision_at_k, recall_at_k, ndcg_at_k, mrr
├── api/
│   ├── main.py
│   ├── routers/            # recommendations.py, products.py
│   ├── models/             # schemas Pydantic (recommendations, products, health)
│   └── services/           # recommendations.py, products.py
├── scripts/
│   ├── train.py            # pipeline completo de treinamento
│   └── evaluate.py         # avaliação comparativa SVD / KNN / Hybrid
├── tests/                  # 200 testes (espelham estrutura do projeto)
├── notebooks/
│   └── 01_eda.ipynb
├── Dockerfile
└── docker-compose.yml
```

---

## Dados

O projeto usa o dataset [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) (categoria Electronics):

- **5.4M** interações de usuários
- **1.6M** produtos
- Split temporal: 70% treino / 15% validação / 15% teste

Os dados não são versionados. Consulte `data/processing.py` para o pipeline de ingestão.

---

## Testes

```bash
pytest tests/ -v --cov=ml --cov=api --cov-report=term-missing
```

---

## License

MIT
