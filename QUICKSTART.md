# Quickstart — SmartRec

Guia para rodar o SmartRec em qualquer máquina do zero.

---

## Pré-requisitos

| Via Docker | Via Python direto |
|---|---|
| Docker + Docker Compose | Python 3.11 |
| Git | Git |

---

## 1. Clonar o repositório

```bash
git clone https://github.com/Andreson1010/SmartRec
cd SmartRec
```

---

## 2. Configurar variáveis de ambiente

```bash
cp .env.example .env
```

Edite o `.env` e ajuste:

```env
MLFLOW_TRACKING_URI=http://localhost:5000
MODEL_PATH=./ml/hybrid/model
API_PORT=8000
API_KEY=troque-por-um-token-seguro   # gere com o comando abaixo
```

Para gerar uma chave segura:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## 3. Instalar dependências

### Opção A — Docker (recomendado)

```bash
docker-compose up --build
```

Serviços disponíveis:
- API: http://localhost:8000
- MLflow UI: http://localhost:5000
- Docs interativos: http://localhost:8000/docs

### Opção B — Python direto

```bash
python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

---

## 4. Preparar os dados

> Os dados não estão no repositório. É necessário baixá-los e processá-los.

### 4.1 Baixar o dataset

Dataset: **Amazon Reviews 2023** (McAuley-Lab) — categoria *All_Beauty*

```bash
# Cria os diretórios necessários
mkdir -p data/raw data/processed data/embeddings
```

Baixe o arquivo `reviews.jsonl` e coloque em `data/raw/reviews.jsonl`.

### 4.2 Processar os dados

```bash
python -m data.processing
```

Gera em `data/processed/`:
- `interactions.parquet`
- `products.parquet`
- `users.parquet`

---

## 5. Treinar os modelos

```bash
# Inicia o MLflow (necessário para logging)
mlflow server --host 0.0.0.0 --port 5000 &

# Treina SVD, KNN, embeddings e modelo híbrido
python scripts/train.py
```

Os modelos são salvos em `ml/hybrid/model/`.

---

## 6. Subir a API

### Via Docker (se ainda não rodou o passo 3A)

```bash
docker-compose up
```

### Via Python direto

```bash
uvicorn api.main:app --reload --port 8000
```

---

## 7. Testar a API

### Health check

```bash
curl http://localhost:8000/health
```

### Recomendações

```bash
curl -X POST http://localhost:8000/recommendations/ \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <sua-api-key>" \
  -d '{"user_id": "A1B2C3", "top_k": 10}'
```

Acesse a documentação interativa em: http://localhost:8000/docs

---

## 8. Rodar os testes

```bash
pytest tests/ --cov=. --cov-report=term-missing
```

---

## Resumo do fluxo completo

```
git clone → .env → pip install → data/processing → scripts/train.py → uvicorn
```

---

## Problemas comuns

| Erro | Causa | Solução |
|---|---|---|
| `MODEL_PATH not found` | Modelos não treinados | Rodar `scripts/train.py` |
| `401 Unauthorized` | API_KEY ausente/errada | Verificar header `X-API-Key` |
| `mlflow.exceptions.MlflowException` | MLflow não está rodando | Iniciar `mlflow server` antes do treino |
| Erro de memória no treino | Dataset muito grande | Reduzir amostra em `scripts/train.py` |
