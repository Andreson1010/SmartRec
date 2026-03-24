# SmartRec — Hybrid Recommendation System

![CI](https://github.com/Andreson1010/SmartRec/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-200%20passing-brightgreen)

Sistema de recomendação híbrido que combina **filtragem colaborativa** (SVD + KNN via scipy) com **busca semântica** (Sentence Transformers) em um único pipeline treinável, com API REST, rastreamento de experimentos via MLflow e deploy via Docker.

---

## O problema

Sistemas de recomendação têm dois desafios centrais:

- **Cold start:** novos usuários sem histórico não recebem recomendações úteis com CF puro
- **Semântica ignorada:** CF baseado em rating não sabe que "cabo USB-C" e "carregador rápido" são similares

SmartRec resolve os dois: o modelo semântico cobre cold start e captura similaridade de conteúdo; o CF captura padrões de comportamento coletivo.

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

### Estratégia de fusão híbrida

O `HybridRecommender` suporta três estratégias configuráveis:

| Estratégia | Descrição |
|------------|-----------|
| `weighted` | Média ponderada dos scores normalizados (CF × α + semântico × (1−α)) |
| `rank_fusion` | Reciprocal Rank Fusion — combina rankings sem depender de escala de scores |
| `rerank` (padrão) | CF gera candidatos → semântico re-ranqueia usando embedding do item seed |

O parâmetro `alpha` é tunable via `tune_alpha()` com busca em grade por NDCG@10 no conjunto de validação.

### Cold start

Usuários sem histórico recebem os itens mais populares do catálogo com score `0.0` — sinal explícito de fallback, não silêncio.

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
| `GET` | `/products/{id}/similar` | Produtos similares por embedding semântico |
| `GET` | `/health` | Status da API, versão do modelo e uptime |

**Autenticação:** todos os endpoints exigem `X-API-Key` no header (configurada via `.env`).

**Exemplo:**

```bash
curl -X POST http://localhost:8000/recommendations/ \
  -H "Content-Type: application/json" \
  -H "X-API-Key: seu-token" \
  -d '{"user_id": "ABC123", "top_k": 10}'
```

```json
{
  "user_id": "ABC123",
  "recommendations": [
    {"product_id": "B08XYZ", "score": 0.9231},
    {"product_id": "B07ABC", "score": 0.8714}
  ]
}
```

---

## Resultados

| Modelo | Precision@10 | Recall@10 | NDCG@10 | MRR |
| ------ | :----------: | :-------: | :-----: | :-: |
| SVD (validação) | 0.0101 | 0.0309 | 0.0208 | 0.0257 |
| **Híbrido rerank (teste)** | 0.0078 | 0.0236 | 0.0160 | 0.0202 |

**Por que os números são baixos?**

O dataset tem esparsidade de **99,95%** — cada usuário avaliou em média 7,8 de 14.876 produtos. Com 86k usuários e cold start frequente, métricas absolutas como precision@10 são naturalmente baixas nesse regime. Valores entre 0.01–0.03 são típicos em CF esparso na literatura (MovieLens-1M com SVD atinge ~0.02–0.04 em configurações similares).

O que importa para o portfólio é a **arquitetura e o pipeline**, não o número absoluto.

> Dataset: Amazon Reviews 2023 — Electronics. 539k interações de treino, split temporal 70/15/15.
> Execute `python -m scripts.evaluate` para comparação completa SVD × KNN × Híbrido.

---

## Estrutura do projeto

```text
smartrec/
├── data/
│   ├── processing.py       # raw → parquet
│   ├── eda.py              # análise exploratória → reports/
│   └── processed/          # .gitignored
├── ml/
│   ├── base.py             # BaseRecommender (ABC): fit/predict/evaluate/save/load
│   ├── collaborative/      # SVDRecommender, KNNRecommender
│   ├── semantic/           # ProductEmbedder, SemanticRetriever
│   ├── hybrid/             # HybridRecommender (produção)
│   └── evaluation/         # precision_at_k, recall_at_k, ndcg_at_k, mrr
├── api/
│   ├── main.py             # app, lifespan, LatencyMiddleware
│   ├── routers/            # recommendations.py, products.py
│   ├── models/             # schemas Pydantic
│   └── services/           # lógica de negócio → chama ml/hybrid/
├── scripts/
│   ├── train.py            # pipeline completo: split → treino → tune → MLflow
│   └── evaluate.py         # comparação SVD / KNN / Hybrid com tabela de métricas
├── tests/                  # 200 testes (espelham estrutura do projeto)
├── docs/
│   └── architecture.md     # decisões de design e arquitetura técnica
├── Dockerfile
└── docker-compose.yml
```

---

## Dados

O projeto usa o dataset [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) (categoria Electronics):

- **5,4M** interações brutas → **770k** após filtragem (usuários com ≥ 5 interações)
- **14.876** produtos únicos nas interações; **9.321** com embedding semântico
- Sparsidade: 99,95% (matriz usuário × item)
- Split temporal: 70% treino / 15% validação / 15% teste

Os dados não são versionados. Consulte `data/processing.py` para o pipeline de ingestão.

---

## Decisões de design

Para o raciocínio por trás das escolhas arquiteturais, veja [`docs/architecture.md`](docs/architecture.md).

---

## Testes

```bash
pytest tests/ -v --cov=ml --cov=api --cov-report=term-missing
```

200 testes cobrindo unitários (métricas, modelos, embeddings), integração (API endpoints) e contratos de interface (BaseRecommender).

---

## License

MIT
