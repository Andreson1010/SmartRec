# Arquitetura Técnica — SmartRec

Este documento registra as decisões de design do SmartRec: o que foi escolhido, por quê, e quais alternativas foram descartadas.

---

## Visão geral

SmartRec é organizado em três camadas independentes que se comunicam por interfaces bem definidas:

```
data/          → ingestão e pré-processamento
ml/            → modelos (CF, semântico, híbrido) e métricas
api/           → serviço REST (FastAPI)
```

A separação é intencional: qualquer modelo pode ser substituído sem tocar na API, desde que implemente `BaseRecommender`.

---

## ml/base.py — Contrato de interface

Todos os modelos implementam `BaseRecommender` (ABC):

```python
fit(data: pd.DataFrame) → self
predict(user_id: str, top_k: int) → list[dict]   # [{"product_id": str, "score": float ∈ [0,1]}]
evaluate(test_data: pd.DataFrame) → dict[str, float]
save(path: Path) → None
load(path: Path) → BaseRecommender
```

**Por quê ABC?** Força o contrato em tempo de importação, não em runtime. `_check_fitted()` lança `RuntimeError` ao chamar `predict` antes de `fit` — falha rápida e mensagem clara.

**Scores normalizados em [0, 1]:** o HybridRecommender precisa combinar scores de fontes diferentes. Normalizar na saída de cada modelo evita que a escala de um domine a fusão.

---

## ml/collaborative/ — Filtragem Colaborativa

### Por que scipy e não scikit-surprise?

`scikit-surprise` não compila no Windows sem Visual Studio (MSVC). Como o ambiente de desenvolvimento é Windows 11 e o objetivo é portabilidade sem dependências nativas complexas, a escolha foi reimplementar SVD com `scipy.sparse.linalg.svds`, que já vem como dependência transitiva de outros pacotes.

O custo foi implementar manualmente bias removal por usuário e normalização min-max. O benefício foi eliminar uma dependência problemática.

### SVDRecommender

- Constrói matriz esparsa usuário × item (CSR)
- Remove bias de usuário antes da decomposição
- SVD truncado com `k` fatores latentes (padrão: 50)
- `predict`: score = dot(user_vec, item_vecs.T), normalizado min-max
- Cold start: retorna itens mais populares com score `0.0`

### KNNRecommender

- Similaridade de cosseno entre vetores de usuário (baseado em histórico)
- `k` vizinhos mais próximos → agrega itens por score ponderado
- Cold start: mesma estratégia de popularidade do SVD

---

## ml/semantic/ — Busca Semântica

### Modelo escolhido: all-MiniLM-L6-v2

Modelo leve (22M parâmetros, embeddings de 384 dimensões) com boa relação qualidade/velocidade para busca semântica em inglês. O dataset é Amazon Reviews em inglês — match direto.

Alternativas consideradas: `all-mpnet-base-v2` (melhor qualidade, 4× mais lento), `paraphrase-MiniLM-L3-v2` (mais rápido, qualidade inferior). O L6 foi o ponto de equilíbrio para um projeto de portfólio sem GPU.

### ProductEmbedder

- Concatena `title + description` de cada produto como texto de entrada
- `normalize_embeddings=True`: força norma unitária → similaridade de cosseno via dot product
- Salva embeddings como numpy array (`.npy`) + mapeamento de índice

### SemanticRetriever

- `query_by_product(product_id, top_k)`: busca os k mais similares ao embedding de um produto
- `query_by_vector(vector, top_k)`: busca por vetor arbitrário (usado pelo HybridRecommender)
- Exclui o próprio item da busca com score `-inf` antes do argsort

---

## ml/hybrid/ — Fusão Híbrida

O `HybridRecommender` é o modelo de produção. Recebe um `user_id` e retorna recomendações combinando CF e semântico.

### Estratégia rerank (padrão)

```
1. CF gera top_k × 3 candidatos
2. Seleciona o primeiro candidato que possui embedding (item "seed")
3. SemanticRetriever busca os top_k mais similares ao seed
4. Score final = alpha × score_cf + (1 - alpha) × score_semântico
```

**Por que top_k × 3 candidatos do CF?** O seed precisa ter embedding. Com dataset esparso (63% dos produtos não têm embedding), pegar só top_k causava seed inválido em 74% dos casos. Expandir para 3× resolve o problema sem custo significativo.

### Estratégia weighted

Normaliza os scores de CF e semântico separadamente, depois combina com peso `alpha`. Simples e interpretável, mas sensível a outliers na normalização.

### Estratégia rank_fusion (RRF)

Reciprocal Rank Fusion: `score = Σ 1/(k + rank_i)` para cada lista. Não depende de escala de scores — robusto quando as distribuições de CF e semântico são muito diferentes.

### Tuning de alpha

`tune_alpha()` faz busca em grade sobre `alphas` no conjunto de validação, maximizando NDCG@10. O alpha ótimo é salvo junto com o modelo.

---

## api/ — Camada de Serviço

### Padrão router → service → ml

Os routers **nunca** importam `ml/` diretamente. O fluxo é:

```
router.py → RecommendationService → HybridRecommender.load()
```

**Por quê?** Separa protocolo HTTP (status codes, serialização) de lógica de negócio. Facilita trocar o modelo sem tocar no router, e simplifica testes (mock no service, não no modelo).

### Singleton do service via lifespan

O `HybridRecommender` carrega embeddings (~centenas de MB) na inicialização. Carregar por request seria inaceitável. A solução usa o `lifespan` do FastAPI:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = RecommendationService()  # carrega uma vez
    yield
```

O router acessa via `request.app.state.service`.

### LatencyMiddleware

Injeta o header `X-Process-Time-Ms` em todas as respostas e loga método, path, status e latência. Implementado como ASGI middleware puro — sem dependência de biblioteca de observabilidade.

### Tratamento de erros

| Exceção | Status HTTP |
|---------|-------------|
| `ValueError` (ex: usuário inválido) | 404 |
| Qualquer outra `Exception` | 500 |

Erros 500 logam o traceback completo mas retornam mensagem genérica ao cliente — sem vazar detalhes internos.

---

## data/ — Pipeline de dados

### Split temporal (não aleatório)

O split é feito por timestamp, não por amostragem aleatória:

- **Treino:** primeiros 70% do período
- **Validação:** próximos 15%
- **Teste:** últimos 15%

**Por quê temporal?** Split aleatório vaza o futuro para o treino — o modelo "vê" interações futuras durante o treinamento. Split temporal simula o cenário real: o modelo é treinado com histórico passado e avaliado em comportamento futuro.

### Filtragem de usuários esparsos

Apenas usuários com ≥ 5 interações entram no dataset processado. Isso reduz de 5,4M para 770k interações mas elimina usuários cujo histórico é insuficiente para qualquer modelo de CF.

---

## Decisões que não tomei (e por quê)

| Alternativa | Por que descartada |
|-------------|-------------------|
| scikit-surprise para SVD | Não instala no Windows sem MSVC |
| Redis para cache | Overhead de infraestrutura desnecessário para portfólio |
| JWT para autenticação | API key simples é suficiente para o caso de uso |
| Faiss para busca semântica | numpy + argsort é suficiente para 14k embeddings; Faiss vale a partir de 100k+ |
| PostgreSQL para persistência | Parquet + numpy satisfaz os requisitos sem banco relacional |
| Celery para treinamento assíncrono | Scripts síncronos são mais fáceis de debugar e auditar |

---

## Métricas de avaliação

Todas as métricas assumem **feedback binário implícito**: uma interação no conjunto de teste é considerada "relevante" (`1`), qualquer item não interagido é "não relevante" (`0`).

| Métrica | O que mede |
|---------|-----------|
| Precision@k | Fração dos k recomendados que são relevantes |
| Recall@k | Fração dos itens relevantes que aparecem nos k recomendados |
| NDCG@k | Qualidade do ranking (penaliza relevantes nas posições inferiores) |
| MRR | Posição média do primeiro item relevante |

Com sparsidade de 99,95%, valores absolutos baixos (0.01–0.03) são esperados e consistentes com a literatura em CF esparso.
