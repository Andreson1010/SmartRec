# agents.md

Agentes especializados para o projeto SmartRec.
Cada agente tem escopo restrito para evitar interferências entre camadas.

> **Regra sobre `requirements.txt`:** apenas o `infra-agent` commita mudanças nesse arquivo.
> Todos os outros agentes que precisarem adicionar dependências devem **propor a adição** no output
> (ex: "adicionar `scikit-surprise` ao requirements.txt") e aguardar o `infra-agent` efetivar.

---

## Agente 0 — `data-agent`

**Responsabilidade:** Pipeline de ingestão, pré-processamento e análise exploratória dos dados.

**Arquivos que pode criar/modificar:**
- `data/processing.py` — pipeline raw → parquet
- `data/eda.py` — análise exploratória e geração de figuras
- `data/__init__.py`
- `notebooks/` — notebooks de exploração
- `reports/` — sumários JSON e figuras PNG (não versionados)
- `tests/data/test_processing.py`
- `tests/data/test_eda.py`

**Arquivos que nunca deve tocar:**
- `ml/` — não modifica modelos
- `api/` — não modifica API
- `data/raw/` — apenas lê; nunca sobrescreve dados brutos
- `data/embeddings/` — escopo do `embedding-agent`

**Ferramentas disponíveis:** `pandas`, `pyarrow`, `numpy`, `matplotlib`, `seaborn`, `scipy`

**Skills de referência:** `test-pattern`

**Saída esperada:**
- `data/processed/interactions.parquet` — colunas: `user_id`, `product_id`, `rating`, `timestamp`
- `data/processed/products.parquet` — colunas: `product_id`, `title`, `description`, `category`, `price`
- `data/processed/users.parquet` — colunas: `user_id`, `total_reviews`, `avg_rating`
- `reports/figures/*.png` e `reports/eda_summary.json`

---

## Agente 1 — `embedding-agent`

**Responsabilidade:** Geração, persistência e busca de embeddings semânticos de produtos.

**Arquivos que pode criar/modificar:**
- `ml/semantic/embedder.py` — geração de embeddings com Sentence Transformers
- `ml/semantic/retriever.py` — busca por similaridade coseno
- `ml/semantic/__init__.py`
- `data/embeddings/` — artefatos `.npy` (embeddings e product_ids)
- `tests/ml/semantic/test_embedder.py`
- `tests/ml/semantic/test_retriever.py`

**Arquivos que nunca deve tocar:**
- `data/processing.py` — pipeline de dados é responsabilidade do agente de dados
- `ml/collaborative/` — escopo de outro agente
- `ml/hybrid/` — escopo do agente de fusão
- `api/` — escopo do agente de API
- `data/processed/` — ler é permitido; nunca sobrescrever

**Ferramentas disponíveis:** `sentence-transformers`, `numpy`, `pandas`

**Skills de referência:** `embedding-pipeline`, `ml-model`, `test-pattern`, `mlflow-experiment-tracking`

**Contexto de entrada esperado:**
- `data/processed/products.parquet` (colunas: `product_id`, `title`, `description`)

**Saída esperada:**
- `data/embeddings/embeddings.npy` — float32, shape `(n_products, 384)`
- `data/embeddings/product_ids.npy` — array de strings alinhado por índice

---

## Agente 2 — `collaborative-filtering-agent`

**Responsabilidade:** Implementação e treinamento dos modelos de filtragem colaborativa (SVD e KNN).

**Arquivos que pode criar/modificar:**
- `ml/collaborative/svd.py` — SVDRecommender via Surprise
- `ml/collaborative/knn.py` — KNNRecommender via Surprise
- `ml/collaborative/__init__.py`
- `ml/collaborative/artifacts/` — modelos serializados `.pkl`
- `ml/evaluation/metrics.py` — precision_at_k, recall_at_k, ndcg_at_k, mrr
- `ml/evaluation/__init__.py`
- `tests/ml/collaborative/test_svd.py`
- `tests/ml/collaborative/test_knn.py`
- `tests/ml/evaluation/test_metrics.py`

**Arquivos que nunca deve tocar:**
- `data/processing.py` — pipeline de dados é separado
- `ml/semantic/` — escopo do agente de embeddings
- `ml/hybrid/` — escopo do agente de fusão
- `api/` — escopo do agente de API
- `data/processed/` — ler é permitido; nunca sobrescrever

**Ferramentas disponíveis:** `scikit-surprise`, `numpy`, `pandas`, `mlflow`
> ⚠️ `scikit-surprise` ainda não está em `requirements.txt` — propor adição ao `infra-agent` antes de implementar.

**Skills de referência:** `collaborative-filtering`, `ml-model`, `test-pattern`, `mlflow-experiment-tracking`

**Contexto de entrada esperado:**
- `data/processed/interactions.parquet` (colunas: `user_id`, `product_id`, `rating`)

**Saída esperada:**
- `ml/collaborative/artifacts/svd.pkl`
- `ml/collaborative/artifacts/knn.pkl`
- Métricas logadas no experimento MLflow `smartrec/collaborative`

---

## Agente 3 — `hybrid-fusion-agent`

**Responsabilidade:** Combinação dos scores colaborativo e semântico no modelo de produção.

**Arquivos que pode criar/modificar:**
- `ml/hybrid/recommender.py` — HybridRecommender (weighted fusion + RRF)
- `ml/hybrid/__init__.py`
- `ml/hybrid/artifacts/` — modelo serializado `hybrid.pkl`
- `tests/ml/hybrid/test_recommender.py`

**Arquivos que nunca deve tocar:**
- `ml/collaborative/` — apenas consome os artefatos; não modifica
- `ml/semantic/` — apenas consome os artefatos; não modifica
- `ml/evaluation/metrics.py` — apenas lê; não modifica
- `data/` — apenas lê parquets; não modifica
- `api/` — escopo do agente de API

**Dependências obrigatórias antes de executar:**
- `data/embeddings/embeddings.npy` deve existir (produzido pelo `embedding-agent`)
- Pelo menos um dos seguintes deve existir (produzido pelo `collaborative-filtering-agent`):
  - `ml/collaborative/artifacts/svd.pkl` ← preferido por padrão
  - `ml/collaborative/artifacts/knn.pkl` ← alternativa; ajustar `cf_model_path` no `HybridRecommender`

**Ferramentas disponíveis:** `numpy`, `pandas`, `mlflow`

**Skills de referência:** `hybrid-fusion-strategies`, `ml-model`, `test-pattern`, `mlflow-experiment-tracking`

**Saída esperada:**
- `ml/hybrid/artifacts/hybrid.pkl`
- Experimento MLflow `smartrec/hybrid` com métricas e alpha tunado
- Modelo registrado no MLflow Model Registry como `smartrec-hybrid`

---

## Agente 4 — `api-agent`

**Responsabilidade:** Implementação da API REST FastAPI: endpoints, schemas Pydantic e lógica de negócio.

**Arquivos que pode criar/modificar:**
- `api/main.py` — app FastAPI + registro de routers + lifespan
- `api/routers/recommendations.py` — endpoint POST /recommendations
- `api/models/recommendations.py` — schemas Request/Response
- `api/services/recommendations.py` — orquestra chamada ao HybridRecommender
- `api/routers/__init__.py`, `api/models/__init__.py`, `api/services/__init__.py`, `api/__init__.py`
- `tests/api/test_recommendations.py`

**Arquivos que nunca deve tocar:**
- `ml/` — apenas consome via `HybridRecommender.load()`; não modifica modelos
- `data/` — nunca acessa diretamente; dados chegam via modelos carregados
- `.env.example` — pode ler; nunca sobrescrever sem instrução explícita

**Padrão obrigatório:**
- Router não importa `ml/` diretamente — tudo passa pelo Service
- Separação: `api/routers/` → `api/services/` → `ml/hybrid/`
- Log de latência em ms após cada request bem-sucedido
- `ValueError` → HTTP 404 | `Exception` → HTTP 500

**Ferramentas disponíveis:** `fastapi`, `uvicorn`, `pydantic`, `python-dotenv`

**Skills de referência:** `api-endpoint`, `test-pattern`

**Variáveis de ambiente consumidas:**
- `MODEL_PATH` — caminho para `ml/hybrid/artifacts/` (onde `hybrid.pkl` está salvo)
- `API_PORT` — porta da API (default: 8000)

> `MLFLOW_TRACKING_URI` **não é consumido pela API** — a API apenas carrega o modelo do disco via `MODEL_PATH`.
> O tracking URI é usado exclusivamente durante o treino pelos agentes de ML.

---

## Agente 5 — `mlflow-agent`

**Responsabilidade:** Configurar a infraestrutura do servidor MLflow e instrumentar scripts de utilidade para gerenciar experimentos e o Model Registry. Não é responsável por *operar* o servidor — isso é responsabilidade do `infra-agent` via Docker.

**Arquivos que pode criar/modificar:**
- `mlflow/docker-compose.yml` — definição do serviço MLflow
- `mlflow/Dockerfile` — imagem customizada se necessário
- `mlflow/.env` — variáveis de ambiente do servidor
- `mlflow/promote_model.py` — script para promover modelo Staging → Production
- `mlflow/compare_runs.py` — script para comparar runs por `ndcg_at_10`

**Arquivos que nunca deve tocar:**
- `ml/` — apenas lê experimentos registrados; não modifica código de modelos
- `data/` — não acessa
- `api/` — não acessa

**Responsabilidades:**
- Configurar o servidor MLflow para rodar em `http://localhost:5000`
- Criar os experimentos com naming `smartrec/<componente>` antes do primeiro treino
- Escrever scripts para promover modelos no Model Registry (Staging → Production)
- Escrever scripts para comparar runs e identificar o melhor modelo por `ndcg_at_10`

**Skills de referência:** `mlflow-experiment-tracking`

**Convenções de experimento:**
```
smartrec/collaborative   ← runs do SVD e KNN
smartrec/semantic        ← runs do embedder (quando aplicável)
smartrec/hybrid          ← runs do HybridRecommender com diferentes alphas
```

---

## Agente 6 — `infra-agent`

**Responsabilidade:** Docker, docker-compose, CI/CD e infraestrutura de build.

**Arquivos que pode criar/modificar:**
- `docker/Dockerfile.api` — imagem da API FastAPI
- `docker/Dockerfile.frontend` — imagem do frontend (quando existir)
- `docker-compose.yml` — orquestração local (api + mlflow + frontend)
- `.github/workflows/ci.yml` — pipeline de CI (lint + testes)
- `.github/workflows/cd.yml` — pipeline de CD (build + deploy)
- `requirements.txt` — **único agente autorizado a commitar mudanças aqui**; aplica adições propostas pelos outros agentes

**Arquivos que nunca deve tocar:**
- `ml/` — não modifica código de modelos
- `api/` — não modifica código da API
- `data/` — não modifica pipelines de dados
- `.env` — nunca commitar; usar `.env.example` como referência

**Padrões obrigatórios:**
- Imagens Docker baseadas em `python:3.11-slim`
- Variáveis sensíveis via variáveis de ambiente (nunca hardcoded)
- CI deve executar: `black --check`, `flake8`, `pytest tests/`
- Dados e modelos nunca incluídos na imagem Docker (volume mount ou object storage)

**Skills de referência:** nenhuma skill específica criada; seguir convenções do `CLAUDE.md`

---

## Relação entre agentes (ordem de execução)

```
infra-agent          →  requirements.txt, docker-compose.yml
mlflow-agent         →  mlflow/ configurado, servidor em :5000
        │
        │  (MLflow deve estar rodando antes de qualquer treino)
        │
data-agent           →  data/processed/*.parquet
        │
        ├─► embedding-agent             →  data/embeddings/
        │
        └─► collaborative-filtering-agent  →  ml/collaborative/artifacts/
                        │
                        └─── (ambos prontos) ──►  hybrid-fusion-agent
                                                         │
                                                         └──►  api-agent  →  API em :8000
```

`infra-agent` e `mlflow-agent` são pré-condições de infraestrutura — devem ser configurados
antes de qualquer agente de ML executar treino.
