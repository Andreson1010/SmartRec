# CLAUDE.md

**SmartRec** — Recomendação híbrida: filtragem colaborativa (SVD/KNN) + busca semântica (Sentence Transformers) + API REST.

## Stack

| Camada         | Tecnologia |
|----------------|------------|
| API            | FastAPI + uvicorn[standard] |
| ML — CF        | scipy — SVD via `svds`, KNN via cosseno                 |
| ML — Semântico | sentence-transformers |
| ML — Tracking  | mlflow |
| Dados          | pandas + pyarrow + numpy + scipy |
| Schemas        | pydantic |
| Formatação     | black (line-length 88) |
| Lint           | flake8 (max-line-length=88, ignore E203,W503) |
| Testes         | pytest |

> Python 3.11. `scikit-learn` está em requirements.txt mas não é usado para CF — serve só para avaliação. CF usa `scipy` (`svds` para SVD, `cosine_similarity` para KNN).

## Estrutura

```
smartrec/
├── data/
│   ├── processing.py     # raw → parquet
│   ├── eda.py            # análise → reports/figures/
│   ├── raw/              # .gitignored; reviews.jsonl disponível
│   ├── processed/        # interactions.parquet, products.parquet, users.parquet
│   └── embeddings/       # .gitignored
├── ml/
│   ├── collaborative/    # SVDRecommender, KNNRecommender
│   ├── semantic/         # ProductEmbedder, SemanticRetriever
│   ├── hybrid/           # HybridRecommender (produção)
│   └── evaluation/       # precision_at_k, recall_at_k, ndcg_at_k, mrr
├── api/
│   ├── main.py           # app FastAPI + routers
│   ├── routers/          # um arquivo por feature
│   ├── models/           # schemas Pydantic
│   └── services/         # lógica de negócio → chama ml/hybrid/
├── .claude/hooks/        # pre_write_py.py, post_test_file.py
├── .env.example          # MODEL_PATH, API_PORT, MLFLOW_TRACKING_URI
└── requirements.txt
```

## Comandos

```bash
python -m data.processing                       # raw → parquet
python data/eda.py                              # gera reports/
mlflow server --host 0.0.0.0 --port 5000
uvicorn api.main:app --reload --port 8000
pytest tests/ --cov=. --cov-report=term-missing
```

## Convenções

**Todo arquivo .py:** `from __future__ import annotations` como primeira linha.

**Paths:** sempre `pathlib.Path`. Constantes no topo: `ROOT = Path(__file__).resolve().parent`.

**Logging:** `logger = logging.getLogger(__name__)` — nunca `print()` em produção.

**Docstrings:** estilo NumPy (`Parameters\n----------`) em todas as classes e funções públicas.

**Hooks (automáticos):**
- `pre_write_py` bloqueia escrita de `.py` se `black --check` ou `flake8` falharem — corrigir antes de salvar
- `post_test_file` roda `pytest <arquivo> -v` automaticamente ao salvar `test_*.py`

**Testes:** espelham estrutura em `tests/`. Fixtures com `numpy.random.default_rng(42)` — nunca dados reais. MLflow sempre mockado com `unittest.mock.patch`.

**Modelos ML:** interface `fit / predict / evaluate / save / load` definida em `ml/base.py` (`BaseRecommender`). `predict` retorna `list[dict]` com `product_id` e `score ∈ [0, 1]`. Cold start sem exceção.

**API:** router → service → ml/ (router não importa `ml/` diretamente). `ValueError` → 404, `Exception` → 500.

**MLflow:** experimentos `smartrec/<componente>`. Tags: `model_type`, `dataset_version`, `git_commit`. Métricas: `precision_at_10`, `recall_at_10`, `ndcg_at_10`, `mrr`.

**Branches:** `feat/`, `fix/`, `refactor/`, `data/`. Commits em português no imperativo: `feat(ml): adicionar SVDRecommender com cold start`.

## Nunca fazer

- Commitar `data/raw/`, `data/processed/`, `data/embeddings/`, `*.pkl`, `.env`
- Usar `str` para caminhos — usar `pathlib.Path`
- Importar `ml/` nos routers — passar pelo service
- Hardcodar caminhos absolutos
- Usar dados reais em testes
- Rodar MLflow sem mock em testes

## Task Management

**Antes de qualquer tarefa não-trivial (3+ passos):**
1. Escrever plano em `tasks/todo.md` com itens marcáveis `- [ ]`
2. Aguardar confirmação antes de implementar
3. Marcar itens completos `- [x]` conforme avança
4. Resumo alto nível a cada etapa concluída
5. Ao final: adicionar seção `## Revisão` no todo.md
6. Após qualquer correção: atualizar `tasks/lessons.md` com o padrão aprendido

## Diário de Aprendizado

Arquivo: `docs/aprendizado.md` — **não versionado** (já em .gitignore)

**Atualizar automaticamente ao longo da sessão**, organizando por componente:
`### ml/collaborative`, `### ml/semantic`, `### ml/hybrid`, `### api`, `### data`

Capturar dentro de cada seção:
- **P/R:** pergunta do usuário + resposta resumida
- **Decisão:** escolha arquitetural e justificativa
- **Erro:** o que quebrou + como foi resolvido
- **Código:** trecho relevante com contexto de uso

Formato de entrada:
```
#### [YYYY-MM-DD] Título curto
**Tipo:** P/R | Decisão | Erro | Código
**Contexto:** ...
**Conteúdo:** ...
```
```

