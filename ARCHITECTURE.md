# ARCHITECTURE.md — SmartRec

> Documento de arquitetura técnica do SmartRec.
> Destinado a qualquer pessoa com conhecimento básico de Python e ML.
> Cobre teoria, decisões de design, fluxo de dados e explicação do código módulo a módulo.

---

## Sumário

1. [Visão Geral do Projeto](#1-visão-geral-do-projeto)
2. [Glossário](#2-glossário)
3. [Arquitetura Geral](#3-arquitetura-geral)
4. [Fluxo de Dados — Passo a Passo](#4-fluxo-de-dados--passo-a-passo)
5. [Módulo: `data/processing.py`](#5-módulo-dataprocessingpy)
6. [Módulo: `ml/base.py`](#6-módulo-mlbasepy)
7. [Módulo: `ml/collaborative/svd.py`](#7-módulo-mlcollaborativesvdpy)
8. [Módulo: `ml/semantic/embedder.py`](#8-módulo-mlsemanticembedderpy)
9. [Módulo: `ml/semantic/retriever.py`](#9-módulo-mlsemanticretrieverpy)
10. [Módulo: `ml/hybrid/recommender.py`](#10-módulo-mlhybridrecommenderpy)
11. [Módulo: `ml/evaluation/metrics.py`](#11-módulo-mlevaluationmetricspy)
12. [Módulo: `api/`](#12-módulo-api)
13. [Decisões de Arquitetura (ADRs)](#13-decisões-de-arquitetura-adrs)
14. [Métricas e Resultados](#14-métricas-e-resultados)
15. [Infra e Deploy](#15-infra-e-deploy)
16. [Referências e Leituras](#16-referências-e-leituras)

---

## 1. Visão Geral do Projeto

### O que é o SmartRec?

O SmartRec é um **sistema de recomendação híbrido**: um software que, dado um usuário, retorna uma lista dos produtos mais relevantes para aquela pessoa. O resultado é exposto via uma **API REST** — ou seja, qualquer front-end ou aplicativo pode fazer uma chamada HTTP e receber as recomendações em formato JSON.

### Qual problema ele resolve?

Em plataformas de e-commerce, o catálogo pode ter milhões de produtos. Mostrar produtos aleatórios ou apenas os mais vendidos é ineficiente. O ideal é personalizar: mostrar para cada usuário os produtos mais prováveis de interessar a ele, com base no que ele e pessoas parecidas já compraram ou avaliaram.

### Por que "híbrido"?

O SmartRec combina duas abordagens complementares:

- **Filtragem Colaborativa (CF):** "Usuários parecidos com você gostaram desses produtos."
- **Busca Semântica:** "Esses produtos têm descrições/títulos parecidos com o que você já avaliou bem."

Cada abordagem tem pontos fracos que a outra compensa, como detalhado na seção de ADRs.

### Dataset

O projeto usa o **Amazon Reviews 2023 — Electronics**, disponível publicamente no HuggingFace. São 5,4 milhões de interações de usuários reais com produtos reais.

---

## 2. Glossário

Todos os termos técnicos usados no projeto, explicados no contexto do SmartRec.

---

### ABC (Abstract Base Class)

Uma classe Python que não pode ser instanciada diretamente — ela existe apenas para definir uma interface que outras classes devem seguir. No SmartRec, `BaseRecommender` é um ABC: ele define que todo modelo precisa ter os métodos `fit`, `predict`, `evaluate`, `save` e `load`, mas não implementa nenhum deles. É como um contrato.

```python
from abc import ABC, abstractmethod

class BaseRecommender(ABC):
    @abstractmethod
    def fit(self, data): ...  # Obrigatório implementar nas subclasses
```

---

### Alpha (α)

Parâmetro numérico entre 0 e 1 que controla o peso relativo dos dois modelos no `HybridRecommender`. Um alpha de 0.6 significa: 60% do score final vem do modelo colaborativo, 40% do modelo semântico. Quanto maior o alpha, mais o sistema confia no histórico de comportamento coletivo (CF) e menos no conteúdo textual dos produtos.

---

### Batch Size

Quantidade de textos processados de uma vez pelo modelo de embeddings. No `ProductEmbedder`, o padrão é 128 produtos por batch. Processar em batches é mais eficiente do que processar um por um, pois aproveita melhor a CPU/GPU. Um batch muito grande pode causar erro de memória; um muito pequeno é lento.

---

### BaseRecommender

Interface definida em `ml/base.py`. Todo modelo do SmartRec herda desta classe e deve implementar: `fit`, `predict`, `evaluate`, `save` e `load`. Isso garante que o `HybridRecommender` possa usar o `SVDRecommender` sem se preocupar com detalhes internos — apenas chama `predict`.

---

### Bias (viés de usuário)

No SVD, cada usuário tende a dar notas sistematicamente mais altas ou mais baixas que a média. Um usuário que nunca dá menos que 4 estrelas "contamina" os scores — todos os seus itens parecem bons. O SmartRec remove esse viés subtraindo a média de notas do usuário antes de aplicar o SVD, e soma de volta depois. Isso é chamado de *bias removal*.

---

### Busca Semântica

Técnica de busca que entende o **significado** do texto, não apenas palavras-chave. No SmartRec, "busca semântica" significa encontrar produtos cujos títulos/descrições são semanticamente similares a outros produtos que o usuário avaliou bem — mesmo que não compartilhem nenhuma palavra em comum.

Exemplo: "fone de ouvido bluetooth" e "headphone sem fio" são semanticamente similares, mesmo sendo frases diferentes.

---

### Cache da Matriz Normalizada

A `_predicted` do `SVDRecommender` é uma matriz densa (numpy array) de shape `(n_usuários, n_produtos)` que armazena os scores pré-computados para todos os pares usuário-produto. Ela é calculada uma vez durante o `fit()` e mantida em memória. Quando `predict()` é chamado, apenas busca a linha do usuário nessa matriz — operação O(1). Sem esse cache, seria necessário recalcular o produto matricial a cada requisição.

---

### Cold Start

Situação em que não há dados históricos suficientes para personalizar a recomendação. Existem dois tipos no SmartRec:

- **Cold start de usuário:** usuário nunca interagiu com nenhum produto. O `SVDRecommender` retorna os produtos mais populares como fallback.
- **Cold start de produto:** produto sem metadados (título/descrição). O `ProductEmbedder` usa apenas o `product_id` como texto, gerando um embedding pouco informativo.

---

### Cosine Similarity (Similaridade de Cosseno)

Medida de similaridade entre dois vetores, calculada pelo ângulo entre eles (não pela magnitude). O resultado varia de -1 (completamente opostos) a 1 (idênticos). No SmartRec, é usada para comparar embeddings de produtos.

Quando os vetores são **normalizados** (norma L2 = 1), a similaridade de cosseno é equivalente ao produto interno (dot product): `sim = a · b`. Isso é importante porque torna a busca muito eficiente via multiplicação matricial.

```
Fórmula: cos(θ) = (A · B) / (||A|| × ||B||)
Com normalização: cos(θ) = A · B  (pois ||A|| = ||B|| = 1)
```

---

### CSR Matrix (Compressed Sparse Row)

Formato eficiente para armazenar matrizes esparsas (com muitos zeros). No SmartRec, a matriz usuário-produto tem 86.000 usuários × 14.795 produtos, mas a maioria dos pares não tem interação. Armazenar isso como uma matriz densa desperdiçaria memória. A CSR armazena apenas os valores não-zero e seus índices.

```python
from scipy.sparse import csr_matrix
matrix = csr_matrix((ratings, (rows, cols)), shape=(n_users, n_items))
```

---

### DCG / NDCG

**Discounted Cumulative Gain (DCG):** métrica que valoriza mais os acertos que aparecem nas primeiras posições da lista de recomendações. Um acerto na posição 1 vale mais do que na posição 10.

**Normalized DCG (NDCG):** versão normalizada (0 a 1) que divide o DCG obtido pelo DCG ideal (se todos os acertos estivessem no topo). No SmartRec, usamos NDCG@10 — apenas as 10 primeiras recomendações são avaliadas.

```
DCG@k = Σ (relevância_i / log2(posição_i + 2))
NDCG@k = DCG@k / IDCG@k
```

---

### Dependency Injection (Injeção de Dependência)

Padrão onde uma classe recebe seus colaboradores como parâmetros em vez de criá-los internamente. No FastAPI, o `Depends()` implementa isso: o router de recomendações recebe o `RecommendationService` pronto, sem precisar instanciá-lo. Isso facilita testes (basta passar um mock) e evita acoplamento.

```python
def get_rec_service(request: Request) -> RecommendationService:
    return request.app.state.rec_service

@router.post("/")
async def get_recommendations(
    service: RecommendationService = Depends(get_rec_service),
):
    ...
```

---

### Embedding

Representação numérica de um texto como um vetor de números reais. O modelo `all-MiniLM-L6-v2` transforma qualquer texto em um vetor de **384 dimensões**. Textos semanticamente similares ficam próximos no espaço vetorial. No SmartRec, cada produto é representado pelo embedding do seu título + descrição.

---

### FastAPI

Framework Python para criar APIs REST de alta performance. Usa anotações de tipo Python para validar automaticamente os dados de entrada e saída (via Pydantic) e gera documentação interativa automaticamente em `/docs`.

---

### Fatores Latentes

No SVD, os "fatores latentes" são dimensões abstratas que o modelo aprende dos dados. Por exemplo, um fator pode representar "preferência por eletrônicos de áudio" sem que isso seja explicitamente programado. O parâmetro `n_factors=50` define quantas dessas dimensões o modelo usa. Mais fatores = mais expressividade, mas mais risco de overfitting e mais memória.

---

### Filtragem Colaborativa (CF)

Abordagem de recomendação baseada no comportamento coletivo: "usuários parecidos com você gostaram disso." Não usa nenhuma informação sobre o conteúdo dos produtos — só os padrões de interação. A variante do SmartRec usa SVD para encontrar esses padrões em uma matriz esparsa de ratings.

---

### Frozenset

Versão imutável de um `set` Python. No `HybridRecommender`, `_indexed_pids` é um `frozenset` dos `product_ids` que têm embedding. Usar `frozenset` em vez de `list` permite verificar se um produto está indexado em O(1) em vez de O(n), e a imutabilidade evita bugs por modificação acidental.

```python
self._indexed_pids: frozenset[str] = frozenset(
    str(pid) for pid in self._semantic._product_ids
)
```

---

### HybridRecommender

Classe principal do SmartRec, em `ml/hybrid/recommender.py`. Orquestra os dois modelos: usa o `SVDRecommender` para gerar candidatos e o `SemanticRetriever` para re-rankear (estratégia padrão: `rerank`). É este modelo que a API chama.

---

### Joblib

Biblioteca Python para serialização eficiente de objetos Python, especialmente aqueles com arrays numpy grandes. No SmartRec, os modelos são salvos com `joblib.dump()` e carregados com `joblib.load()`. É mais rápido que `pickle` para objetos com grandes arrays.

---

### Lifespan (FastAPI)

Mecanismo do FastAPI para executar código no **startup** e **shutdown** da aplicação. No SmartRec, o `lifespan` carrega o `HybridRecommender` uma única vez quando a API inicia, e armazena em `app.state`. Sem isso, o modelo seria recarregado do disco a cada requisição — inaceitável para produção.

---

### MRR (Mean Reciprocal Rank)

Métrica que mede quão cedo o primeiro item relevante aparece na lista de recomendações. Se o primeiro item relevante está na posição 1, MRR = 1.0. Se está na posição 5, MRR = 0.2. É calculado como a média dos reciprocal ranks de todos os usuários.

```
MRR = média( 1 / posição_do_primeiro_acerto )
```

---

### Middleware

Componente que intercepta todas as requisições HTTP antes de chegarem ao endpoint (e todas as respostas antes de saírem). No SmartRec, o `LatencyMiddleware` mede o tempo de cada requisição e injeta o header `X-Response-Time-Ms` na resposta.

---

### MLflow

Plataforma de rastreamento de experimentos de ML. No SmartRec, cada treino registra automaticamente os hiperparâmetros usados e as métricas obtidas. Isso permite comparar "SVD com 50 fatores" vs "SVD com 100 fatores" de forma organizada, sem perder resultados anteriores.

---

### Normalização L2

Processo de escalar um vetor para que sua norma (comprimento) seja igual a 1. No SmartRec, os embeddings são normalizados no momento da geração (`normalize_embeddings=True`), o que permite usar produto interno como proxy de similaridade de cosseno — tornando a busca muito mais eficiente.

```python
# Com normalização:
norm = np.linalg.norm(user_vec)
return user_vec / norm if norm > 1e-8 else None
```

---

### Parquet

Formato de arquivo colunar para dados tabulares, muito mais eficiente que CSV para grandes volumes. No SmartRec, os dados processados são salvos como `interactions.parquet`, `products.parquet` e `users.parquet`. Operações como "selecionar apenas a coluna rating" leem apenas aquela coluna do disco, sem carregar o resto.

---

### Pathlib.Path

Classe Python para manipulação de caminhos de arquivos de forma orientada a objetos e portável entre Windows/Linux/macOS. No SmartRec, é a única forma permitida de trabalhar com caminhos — `str` é proibido para isso.

```python
ROOT = Path(__file__).resolve().parent.parent.parent
model_path = ROOT / "ml" / "hybrid" / "artifacts"  # Funciona em qualquer SO
```

---

### Precision@K

Fração dos K primeiros itens recomendados que são relevantes para o usuário. Se o modelo recomenda 10 produtos e 2 são relevantes, Precision@10 = 0.2 (20%). Mede a "taxa de acerto" dentro das recomendações.

```
Precision@K = (acertos nos top-K) / K
```

---

### Pydantic

Biblioteca Python de validação de dados. No SmartRec, os schemas de entrada e saída da API são definidos com Pydantic. Ele valida automaticamente os tipos, garante que `score` esteja entre 0 e 1, e retorna erros 422 com mensagem clara se o payload for inválido.

---

### Rank Fusion / RRF (Reciprocal Rank Fusion)

Técnica para combinar duas listas de rankings em uma só, sem depender da escala dos scores. Para cada item, soma `1 / (k + rank)` de cada lista. Itens que aparecem bem rankeados em ambas as listas ganham score alto. O parâmetro `k=60` suaviza o impacto de posições muito altas (evita que o item #1 domine completamente).

```
RRF_score(item) = 1/(60 + rank_CF) + 1/(60 + rank_semântico)
```

---

### Recall@K

Fração dos itens relevantes para o usuário que aparecem nos K primeiros recomendados. Se o usuário tem 5 produtos relevantes e o modelo acerta 2 nos top-10, Recall@10 = 0.4 (40%). Mede o quanto de cobertura o modelo tem.

```
Recall@K = (acertos nos top-K) / (total de relevantes)
```

---

### Rerank (Re-ranqueamento)

Estratégia do `HybridRecommender` onde o CF gera os candidatos e o modelo semântico apenas **reordena** esses candidatos. Nenhum produto novo é introduzido. A vantagem é preservar o alcance do CF (100% dos produtos) enquanto melhora a ordem usando contexto semântico.

---

### Router (FastAPI)

Objeto que agrupa endpoints relacionados. No SmartRec, há dois routers: `recommendations_router` (endpoints de recomendação) e `products_router` (endpoints de produtos similares). Eles são registrados no `app` principal com `include_router()`. Isso mantém o código organizado sem colocar tudo em um único arquivo.

---

### Schemas Pydantic

Classes que definem a estrutura dos dados de entrada e saída da API. No SmartRec:

- `RecommendationRequest`: define que a entrada deve ter `user_id` (string não-vazia) e `top_k` (int entre 1 e 100).
- `RecommendationResponse`: define que a saída terá `user_id`, uma lista de `RecommendedItem` e `model_version`.
- `RecommendedItem`: cada item tem `product_id` e `score` (float entre 0 e 1).

---

### Sentence Transformers

Biblioteca Python que fornece modelos pré-treinados para gerar embeddings de textos. O SmartRec usa o modelo `all-MiniLM-L6-v2`: rápido, leve (80MB) e preciso para tarefas de similaridade semântica. O "MiniLM" é uma versão comprimida de modelos maiores, com 384 dimensões de saída.

---

### Singleton

Padrão onde uma única instância de um objeto é criada e reutilizada. No SmartRec, o `HybridRecommender` é carregado uma vez no startup via `lifespan` e armazenado em `app.state.rec_service`. Todos os requests compartilham esse mesmo objeto — evita recarregar o modelo (~segundos de latência) a cada requisição.

---

### SVD (Singular Value Decomposition)

Técnica matemática de decomposição de matrizes. No SmartRec, a matriz usuário-produto é decomposta em três matrizes: U (perfis de usuários), Σ (importância de cada fator) e Vt (perfis de produtos). O produto `U × Σ × Vt` reconstrói os scores para todos os pares, incluindo pares que o usuário ainda não interagiu. É assim que o CF faz previsões.

```
Matriz original (esparsa) ≈ U × Σ × Vt (reconstruída densa)
```

O parâmetro `k` (n_factors) define quantas colunas de U e Vt usar — é o SVD **truncado**.

---

### `scipy.sparse.linalg.svds`

Função que calcula o SVD truncado de matrizes esparsas de forma eficiente. Diferente do SVD completo (que calcularia todos os fatores), o `svds` calcula apenas os `k` maiores fatores — muito mais rápido para matrizes grandes como a do SmartRec (86k × 14k).

---

### Vetor de Perfil de Usuário

No método `_build_user_vector` do `HybridRecommender`, os embeddings dos top-5 produtos recomendados pelo CF (que tenham embedding semântico) são combinados pela média, criando um vetor que representa o "gosto médio" do usuário. Esse vetor é então usado para re-rankear semanticamente os candidatos.

---

### `v0` (vetor inicial do svds)

O `svds` é um algoritmo iterativo que precisa de um ponto de partida. O parâmetro `v0` define esse vetor inicial. No SmartRec, ele é gerado com `np.random.default_rng(42)` para garantir **reprodutibilidade**: toda vez que o modelo é treinado com os mesmos dados, o resultado é idêntico.

---

## 3. Arquitetura Geral

### Diagrama de Componentes

```
┌─────────────────────────────────────────────────────────────┐
│                        Cliente HTTP                          │
└───────────────────────────┬─────────────────────────────────┘
                            │ POST /recommendations/
                            │ GET  /products/{id}/similar
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      api/main.py                            │
│  LatencyMiddleware → Router → Service → ml/hybrid/          │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              ml/hybrid/HybridRecommender                    │
│                                                             │
│   ┌─────────────────────┐   ┌──────────────────────────┐   │
│   │  SVDRecommender     │   │   SemanticRetriever       │   │
│   │  (CF via SVD)       │   │   (embeddings cosseno)    │   │
│   │                     │   │                           │   │
│   │  _predicted         │   │   embeddings.npy          │   │
│   │  (matriz densa)     │   │   product_ids.npy         │   │
│   └─────────────────────┘   └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │ (treinamento offline)
                            │
┌─────────────────────────────────────────────────────────────┐
│                  data/processing.py                         │
│                                                             │
│   data/raw/reviews.jsonl                                    │
│        → interactions.parquet                               │
│        → products.parquet                                   │
│        → users.parquet                                      │
└─────────────────────────────────────────────────────────────┘
```

### Separação entre treinamento e inferência

O SmartRec tem dois momentos distintos:

**Treinamento (offline):** `python -m scripts.train` — processa os dados, treina os modelos, salva em disco (`ml/collaborative/artifacts/`, `data/embeddings/`). Acontece uma vez (ou quando há novos dados).

**Inferência (online):** `uvicorn api.main:app` — carrega os modelos do disco no startup e responde requisições em tempo real. Não acessa o disco durante as requisições.

---

## 4. Fluxo de Dados — Passo a Passo

### Fluxo 1: Treinamento

```
1. reviews.jsonl (5,4M linhas JSON)
       ↓ data/processing.py
2. interactions.parquet  (user_id, product_id, rating, timestamp)
   products.parquet      (product_id, title, description, ...)
       ↓ SVDRecommender.fit()
3. Matriz CSR esparsa (86k × 14k)
       ↓ Normalização por média de usuário
4. Matriz normalizada
       ↓ scipy.sparse.linalg.svds (k=50)
5. U (86k×50), Σ (50,), Vt (50×14k)
       ↓ U @ diag(Σ) @ Vt + bias
6. _predicted: matriz densa (86k × 14k) — cache em memória
       ↓ joblib.dump()
7. svd.joblib em ml/collaborative/artifacts/

       ↓ (paralelo) ProductEmbedder.fit_transform()
8. Para cada produto: _build_text() → "título | descrição"
       ↓ all-MiniLM-L6-v2 (batches de 128)
9. embeddings: float32 (14795 × 384) — normalizados
       ↓ np.save()
10. embeddings.npy + product_ids.npy em data/embeddings/
```

### Fluxo 2: Requisição de Recomendação (online)

```
POST /recommendations/
{"user_id": "ABC123", "top_k": 10}
        ↓ Pydantic valida o payload
        ↓ LatencyMiddleware (t0 = agora)
        ↓ Router injeta RecommendationService (singleton)
        ↓ service.run(payload)
        ↓ HybridRecommender.predict("ABC123", top_k=10)

1. SVDRecommender.predict("ABC123", top_k=30)
   → Busca linha do usuário em _predicted (O(1))
   → Normaliza scores min-max para [0,1]
   → Retorna top 30 produtos por score

2. _build_user_vector(cf_recs, n_seed=5)
   → Para cada um dos top-5 produtos do CF que tem embedding:
     busca o vetor em embeddings.npy (O(1) por índice)
   → Calcula média dos vetores
   → Normaliza o vetor resultante

3. _semantic_rerank(cf_recs, top_k=10)
   → score_items(pids, user_vector)
     = embeddings[pids] @ user_vector (1 multiplicação matricial)
   → Para cada produto:
     se tem score semântico: 0.6*cf_score + 0.4*sem_score
     senão: cf_score puro
   → Ordena por score combinado
   → Retorna top 10

        ↓ RecommendationResponse(...)
        ↓ Pydantic serializa para JSON
        ↓ LatencyMiddleware injeta X-Response-Time-Ms
        ↓ HTTP 200
{"user_id": "ABC123", "recommendations": [...], "model_version": "1.0.0"}
```

### Fluxo 3: Cold Start

```
POST /recommendations/
{"user_id": "USUARIO_NOVO", "top_k": 10}

1. SVDRecommender.predict("USUARIO_NOVO")
   → "USUARIO_NOVO" não está em _user_index
   → Retorna _popular_fallback(10): os 10 produtos mais populares
     com score=0.0

2. _build_user_vector(cf_recs)
   → Tenta buscar embeddings dos top produtos populares
   → Se nenhum tiver embedding: retorna None

3. _semantic_rerank(cf_recs)
   → user_vector é None
   → Retorna CF puro (produtos populares) sem reranking

Resultado: usuário novo recebe os mais populares.
```

---

## 5. Módulo: `data/processing.py`

### Responsabilidade

Transformar o arquivo bruto `data/raw/reviews.jsonl` em três arquivos Parquet estruturados:

- `interactions.parquet`: cada linha é uma avaliação (user_id, product_id, rating, timestamp)
- `products.parquet`: metadados dos produtos (product_id, title, description, ...)
- `users.parquet`: lista de usuários únicos

### Por que Parquet e não CSV?

CSV é texto puro — lento para ler e ocupa muito espaço. Parquet é binário e colunar: ler apenas a coluna `rating` não carrega `title` e `description` do disco. Para 5,4M de registros, isso faz diferença significativa.

### Por que o split é temporal?

O split de dados usa a coluna `timestamp` para separar treino (70%), validação (15%) e teste (15%). Um split aleatório seria um erro: na realidade, o modelo é treinado com dados passados e avaliado em dados futuros. Se misturarmos temporalmente, o modelo pode "aprender do futuro", inflando artificialmente as métricas.

---

## 6. Módulo: `ml/base.py`

### Responsabilidade

Define a interface `BaseRecommender` que todo modelo deve seguir.

### Por que uma interface abstrata?

Sem ela, o `HybridRecommender` precisaria saber se está usando `SVDRecommender` ou `KNNRecommender` e chamar métodos diferentes. Com a interface, ele apenas chama `self._cf.predict(user_id)` e não precisa saber qual implementação está por baixo — **polimorfismo**.

### `_check_fitted()`

Método de guarda: lança `RuntimeError` se alguém chamar `predict()` antes de `fit()`. Evita bugs silenciosos onde o modelo retorna resultados errados sem avisar.

```python
def _check_fitted(self) -> None:
    if not self._is_fitted:
        raise RuntimeError(
            f"{self.__class__.__name__} não treinado. Chame fit() primeiro."
        )
```

### Por que `fit()` retorna `self`?

Para permitir **encadeamento de métodos**:

```python
model = SVDRecommender().fit(data).evaluate(test_data)
# Em vez de:
model = SVDRecommender()
model.fit(data)
model.evaluate(test_data)
```

---

## 7. Módulo: `ml/collaborative/svd.py`

### Responsabilidade

Implementa `SVDRecommender`: filtragem colaborativa via Singular Value Decomposition usando `scipy`.

### Por que `scipy` e não `scikit-surprise`?

`scikit-surprise` é uma biblioteca dedicada a CF, mas adiciona uma dependência extra e é mais lenta para grandes datasets. O `scipy.sparse.linalg.svds` já está nas dependências do projeto e oferece SVD truncado eficiente. Para o SmartRec, a implementação custom com scipy é suficiente e mais transparente.

### O que acontece em `fit()`?

```python
# 1. Mapeia user_id/product_id para índices inteiros
self._user_index = {u: i for i, u in enumerate(users)}  # {"ABC": 0, "XYZ": 1, ...}
self._item_index = {it: j for j, it in enumerate(items)}

# 2. Cria matriz esparsa CSR
matrix = csr_matrix((ratings, (rows, cols)), shape=(n_users, n_items))

# 3. Remove bias de usuário (normalização)
user_means = np.array(matrix.mean(axis=1)).flatten()  # média de cada usuário
matrix_norm.data[start:end] -= mean  # subtrai apenas nas entradas não-zero

# 4. SVD truncado
U, sigma, Vt = svds(matrix_norm, k=50, v0=v0)

# 5. Reconstrói matriz completa (densa) + adiciona bias de volta
predicted_norm = U @ np.diag(sigma) @ Vt
self._predicted = predicted_norm + user_means[:, np.newaxis]
```

O `[:, np.newaxis]` transforma `user_means` de shape `(n_users,)` para `(n_users, 1)`, permitindo que numpy faça o **broadcasting**: soma o bias de cada usuário em todas as colunas da sua linha.

### O que acontece em `predict()`?

```python
uid_idx = self._user_index[user_id]
scores = self._predicted[uid_idx]  # Linha inteira do usuário: shape (n_items,)

# Normalização min-max: transforma scores absolutos em [0, 1]
norm_scores = (scores - s_min) / (s_max - s_min)

# np.argpartition é mais rápido que sort completo para pegar top-K
top_indices = np.argpartition(norm_scores, -top_k)[-top_k:]
```

`np.argpartition` é O(n) em vez de O(n log n) do sort completo. Para 14k produtos, isso importa em produção.

---

## 8. Módulo: `ml/semantic/embedder.py`

### Responsabilidade

Gerar e persistir embeddings de texto para todos os produtos usando o modelo `all-MiniLM-L6-v2`.

### `_build_text()`

Constrói o texto de entrada para o modelo a partir dos campos do produto:

```python
def _build_text(self, row: pd.Series) -> str:
    parts = []
    # Adiciona título se existir e não for NaN
    if title is not None and not pd.isna(title):
        parts.append(str(title))
    # Adiciona descrição (pode ser lista ou string)
    if desc and text.strip():
        parts.append(text)
    return " | ".join(parts) if parts else str(row.get("product_id", ""))
```

O separador `" | "` ajuda o modelo a distinguir título de descrição. Para produtos sem metadados, usa o `product_id` como fallback — gerando um embedding pouco informativo, mas sem quebrar o pipeline.

### Por que `normalize_embeddings=True`?

Com vetores normalizados, similaridade de cosseno = produto interno. Isso permite fazer a busca de todos os embeddings contra um vetor de consulta com uma única operação: `embeddings @ query_vector` — uma multiplicação matriz-vetor, extremamente otimizada pelo numpy.

### Formato de persistência

```
data/embeddings/
    embeddings.npy   → matriz float32 (14795, 384)
    product_ids.npy  → array de strings (14795,)
```

Os dois arrays são alinhados por índice: `product_ids[i]` corresponde a `embeddings[i]`. Isso é crucial — se a ordem mudar entre save e load, todas as buscas retornariam produtos errados.

---

## 9. Módulo: `ml/semantic/retriever.py`

### Responsabilidade

Buscar os K produtos mais similares dado um embedding de consulta ou um `product_id`.

### Como funciona a busca?

```python
# query_by_product: busca produtos similares a um produto dado
query_vec = self._embeddings[idx]      # shape: (384,)
scores = self._embeddings @ query_vec  # shape: (14795,) — todos os scores de uma vez
```

Essa multiplicação matricial `(14795, 384) @ (384,)` calcula a similaridade de cosseno do produto de consulta contra todos os 14.795 produtos em uma única operação vetorizada. É muito mais eficiente do que um loop.

### `score_items()` vs `query_by_vector()`

- `score_items(pids, vector)`: pontuea apenas produtos específicos (lista de `product_ids`). Usado pelo `HybridRecommender` no rerank — só precisa dos scores dos candidatos do CF, não de todos os 14.795 produtos.
- `query_by_vector(vector)`: retorna os K mais similares do catálogo inteiro. Útil para busca livre.

### `_find_index()`

Busca o índice de um `product_id` no array:

```python
matches = np.where(self._product_ids == product_id)[0]
return int(matches[0]) if len(matches) > 0 else None
```

`np.where` percorre o array inteiro — O(n). Para 14.795 produtos, é aceitável. Para catálogos maiores, seria necessário um índice invertido (dicionário).

---

## 10. Módulo: `ml/hybrid/recommender.py`

### Responsabilidade

Combinar os modelos colaborativo e semântico em uma única classe de produção.

### Estratégias de Fusão

O `HybridRecommender` suporta três estratégias, configuráveis pelo parâmetro `strategy`:

#### `rerank` (padrão de produção)

```
CF gera top-30 candidatos
      ↓
_build_user_vector: média dos embeddings dos top-5 candidatos com embedding
      ↓
score_items: similaridade de cada candidato com o vetor do usuário
      ↓
score_final = 0.6 * cf_score + 0.4 * sem_score
      ↓
Retorna top-10 reordenados
```

**Vantagem:** preserva o alcance total do CF. O semântico apenas melhora a ordem.

#### `weighted`

Média ponderada direta: `alpha * CF_score + (1-alpha) * sem_score`. Requer que ambos os modelos pontuem o mesmo produto — mais restritivo.

#### `rank_fusion` (RRF)

Combina posições dos rankings, não os scores. Robusto quando CF e semântico têm escalas muito diferentes.

### `_build_user_vector()`

```python
def _build_user_vector(self, cf_recs, n_seed=5):
    vecs = []
    for r in cf_recs:
        if r["product_id"] in self._indexed_pids:
            vec = self._semantic.get_embedding(r["product_id"])
            if vec is not None:
                vecs.append(vec)
        if len(vecs) >= n_seed:
            break

    user_vec = np.stack(vecs).mean(axis=0)  # Média dos embeddings
    norm = np.linalg.norm(user_vec)
    return user_vec / norm if norm > 1e-8 else None   # Normaliza
```

A verificação `norm > 1e-8` evita divisão por zero em casos extremos onde a média resulta em vetor nulo.

### `tune_alpha()`

Busca em grade (grid search) para encontrar o melhor `alpha`:

```python
for alpha in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    self.alpha = alpha
    # Avalia NDCG@10 em 200 usuários do conjunto de validação
    mean_ndcg = mean([ndcg_at_k(predict(uid), relevant[uid]) for uid in users])
    if mean_ndcg > best_ndcg:
        best_alpha = alpha
```

Usa apenas 200 usuários por velocidade. O resultado atual é `alpha=0.6`.

---

## 11. Módulo: `ml/evaluation/metrics.py`

### Responsabilidade

Implementa as quatro métricas padrão de sistemas de recomendação.

### Convenção de entrada

Todas as funções recebem:
- `recommended`: lista de `product_ids` **ordenada por score decrescente** (o que o modelo retornou)
- `relevant`: lista de `product_ids` que o usuário realmente gostou (rating >= 4 no conjunto de teste)

### `precision_at_k`

```python
hits = sum(1 for item in recommended[:k] if item in relevant_set)
return hits / k
```

Usa `set(relevant)` para que a busca por pertencimento seja O(1) em vez de O(n).

### `ndcg_at_k`

```python
def dcg(items, cutoff):
    return sum(
        1.0 / math.log2(rank + 2)   # posição 0 → log2(2)=1.0, posição 1 → log2(3)≈0.63
        for rank, item in enumerate(items[:cutoff])
        if item in relevant_set
    )

actual = dcg(recommended, k)
ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(relevant), k)))
return actual / ideal
```

O denominador `log2(rank + 2)` em vez de `log2(rank + 1)` evita log2(1)=0 na posição 0.

### `mrr`

```python
for rank, item in enumerate(recommended, start=1):
    if item in relevant_set:
        return 1.0 / rank  # Para no primeiro acerto
return 0.0
```

---

## 12. Módulo: `api/`

### Estrutura e responsabilidades

```
api/
├── main.py              # App FastAPI, middleware, lifespan, /health
├── routers/
│   ├── recommendations.py  # POST /recommendations/
│   └── products.py         # GET /products/{id}/similar
├── models/
│   ├── recommendations.py  # RecommendationRequest, RecommendationResponse
│   ├── products.py         # SimilarProductsResponse
│   └── health.py           # HealthResponse
└── services/
    ├── recommendations.py  # RecommendationService (orquestra HybridRecommender)
    └── products.py         # ProductService (orquestra SemanticRetriever)
```

### Regra de dependência

```
Router → Service → ml/
```

Os routers **nunca** importam diretamente de `ml/`. Isso facilita testes (pode-se mockar o service sem tocar em ML) e mantém a separação de responsabilidades clara.

### `api/main.py`

**LatencyMiddleware:** intercepta todas as requisições, mede a duração com `time.perf_counter()` (mais preciso que `time.time()`) e injeta o header `X-Response-Time-Ms`.

**Lifespan:** carrega o `HybridRecommender` uma única vez. O modelo fica em `app.state.rec_service` — acessível de qualquer endpoint via `request.app.state`.

**`/health`:** endpoint de diagnóstico que retorna versão do modelo, estratégia ativa, hora de carregamento e uptime. Essencial para monitoramento em produção.

### `api/routers/recommendations.py`

```python
def get_rec_service(request: Request) -> RecommendationService:
    return request.app.state.rec_service  # Singleton carregado no startup
```

Esta função é injetada via `Depends(get_rec_service)`. O FastAPI chama `get_rec_service` antes de chamar o endpoint e passa o resultado como argumento.

### Tratamento de erros

```python
try:
    result = service.run(payload)
except ValueError as exc:
    raise HTTPException(status_code=404, detail=str(exc))
except Exception as exc:
    raise HTTPException(status_code=500, detail="Erro interno")
```

`ValueError` → 404 (recurso não encontrado, ex: `product_id` inválido)
`Exception` → 500 (erro inesperado)
Erros de validação Pydantic → 422 (tratado automaticamente pelo FastAPI)

### `api/models/` — Schemas Pydantic

```python
class RecommendationRequest(BaseModel):
    user_id: str = Field(..., min_length=1)       # Obrigatório, não pode ser vazio
    top_k: int = Field(10, ge=1, le=100)          # Default 10, entre 1 e 100

class RecommendedItem(BaseModel):
    product_id: str
    score: float = Field(..., ge=0.0, le=1.0)     # Validado: sempre entre 0 e 1

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: list[RecommendedItem]
    model_version: str
```

O `Field(..., min_length=1)` significa que o campo é obrigatório (`...`) e deve ter pelo menos 1 caractere. Se vier vazio, Pydantic retorna 422 automaticamente.

---

## 13. Decisões de Arquitetura (ADRs)

### ADR-001: SVD via scipy em vez de scikit-surprise

**Contexto:** Precisávamos de um modelo de CF robusto para 86k usuários e 14k produtos.

**Opções consideradas:**
- `scikit-surprise`: biblioteca dedicada a CF, simples de usar
- `scipy.svds`: mais baixo nível, já presente nas dependências

**Decisão:** `scipy.svds`

**Justificativa:** Evita uma dependência extra. O `scipy` já estava no projeto para operações com matrizes esparsas. A implementação manual com scipy é mais transparente e permite controle total sobre normalização e bias removal.

**Consequência:** Mais código para manter, mas total transparência sobre o que acontece internamente.

---

### ADR-002: Estratégia `rerank` como padrão de produção

**Contexto:** Três estratégias de fusão foram implementadas (weighted, rank_fusion, rerank).

**Problema do weighted:** Requer que o produto tenha embedding semântico. Com 37.3% dos produtos sem metadados, penalizaria muito os candidatos sem embedding.

**Problema do rank_fusion:** Combina os rankings dos dois modelos, introduzindo produtos que o CF não selecionou — pode trazer itens sem relação com o histórico do usuário.

**Decisão:** `rerank`

**Justificativa:** O CF mantém controle total sobre quais produtos são candidatos (cobertura de 100%). O semântico apenas melhora a ordem. Produtos sem embedding mantêm seu score de CF — nenhum produto é prejudicado por falta de metadados.

---

### ADR-003: Cache da matriz predita em memória

**Contexto:** O SVD produz uma matriz densa de (86k × 14k) scores.

**Opções:**
- Calcular os scores de um usuário on-demand durante `predict()`
- Pré-calcular toda a matriz no `fit()` e manter em RAM

**Decisão:** Pré-calcular e manter em RAM.

**Justificativa:** O cálculo on-demand exigiria multiplicação matricial por requisição — latência inaceitável em produção. A matriz ocupa ~4GB de RAM (86k × 14k × 4 bytes float32), mas permite `predict()` em microssegundos (apenas lookup de linha).

**Trade-off:** Uso de memória elevado. Para escala maior, seria necessário aproximação com hashing ou busca por vizinhos aproximados (ANN).

---

### ADR-004: Singleton do modelo via `app.state`

**Contexto:** O `HybridRecommender` demora alguns segundos para carregar (SVD + embeddings do disco).

**Decisão:** Carregar uma única vez no `lifespan` e armazenar em `app.state`.

**Justificativa:** Carregar por requisição seria inviável (~2-5 segundos de latência). O `app.state` é thread-safe para leitura (o modelo não é modificado durante inferência).

---

### ADR-005: `all-MiniLM-L6-v2` como modelo de embeddings

**Contexto:** Precisávamos de um modelo de embeddings para produtos de e-commerce.

**Opções:**
- `all-mpnet-base-v2`: mais preciso, 768 dims, mais lento
- `all-MiniLM-L6-v2`: 384 dims, ~5x mais rápido, boa precisão
- `text-embedding-ada-002` (OpenAI): excelente, mas pago e com dependência de API externa

**Decisão:** `all-MiniLM-L6-v2`

**Justificativa:** Para um portfólio, o custo zero e a independência de APIs externas são importantes. O modelo é local, reprodutível e suficientemente preciso para a tarefa. Os embeddings são gerados offline — velocidade de geração importa menos que velocidade de inferência.

---

## 14. Métricas e Resultados

### Resultados do README

| Modelo | Precision@10 | Recall@10 | NDCG@10 | MRR |
|--------|:---:|:---:|:---:|:---:|
| SVD (validação) | 0.0101 | 0.0309 | 0.0208 | 0.0257 |
| **Híbrido rerank (teste)** | 0.0078 | 0.0236 | 0.0160 | 0.0202 |

### Por que as métricas são tão baixas?

São baixas em valor absoluto, mas **esperadas** para este tipo de problema. Três fatores explicam:

1. **Esparsidade:** 86k usuários × 14k produtos, mas a maioria dos usuários avaliou poucos produtos. O CF tem pouco sinal para trabalhar.

2. **Cold start:** Usuários com poucos histórico recebem fallback com produtos populares — baixa personalização, baixa métrica.

3. **37,3% de produtos sem metadados:** Com títulos mas sem descrições, o reranking semântico tem menos informação para melhorar a ordem.

### Por que o Híbrido é ligeiramente pior que o SVD puro nas métricas?

O SVD é avaliado em validação; o Híbrido em teste — conjuntos diferentes. Além disso, as métricas de ranking como NDCG são muito sensíveis a pequenas mudanças de posição. O reranking pode piorar a posição de alguns acertos (que o CF tinha rankeado alto) ao promover itens semanticamente mais coerentes mas menos previstos pelos dados de rating.

---

## 15. Infra e Deploy

### Docker

```dockerfile
# Dockerfile simplificado
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

Sobe dois serviços:
- `smartrec-api` na porta 8000
- `mlflow-server` na porta 5000

Os modelos treinados são montados como volume — não ficam dentro da imagem.

### CI com GitHub Actions

O badge `![CI]` no README indica que a cada push, o GitHub executa automaticamente:

```
1. pip install -r requirements.txt
2. black --check .           (formatação)
3. flake8 .                  (lint)
4. pytest tests/ --cov=.     (200 testes)
```

Se qualquer passo falhar, o CI marca o commit como vermelho.

### Variáveis de Ambiente

```
MODEL_PATH=ml/hybrid/artifacts
API_PORT=8000
MLFLOW_TRACKING_URI=http://localhost:5000
```

Definidas em `.env` (não versionado) e lidas via `python-dotenv`.

---

## 16. Referências e Leituras

### Filtragem Colaborativa e SVD

- **Paper original:** Koren, Y. (2009). *Matrix Factorization Techniques for Recommender Systems.* IEEE Computer. Fundamento teórico do SVD para CF.
- **Documentação scipy:** [scipy.sparse.linalg.svds](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.svds.html)

### Embeddings e Sentence Transformers

- **Repositório:** [sentence-transformers.net](https://www.sbert.net/) — documentação completa do framework
- **Modelo:** [all-MiniLM-L6-v2 no HuggingFace](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

### Dataset

- **Amazon Reviews 2023:** [HuggingFace — McAuley-Lab](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)

### Métricas de Avaliação

- **Tutorial:** *Evaluating Recommendation Systems* — Shani & Gunawardana (2011). Explica Precision@K, Recall@K, NDCG e MRR em profundidade.

### Fusão de Rankings

- **Reciprocal Rank Fusion:** Cormack, G. V. et al. (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods.*

### FastAPI

- **Documentação oficial:** [fastapi.tiangolo.com](https://fastapi.tiangolo.com) — inclui Pydantic, Depends, lifespan

### MLflow

- **Documentação oficial:** [mlflow.org](https://mlflow.org/docs/latest/index.html)

### Boas Práticas Python

- **pathlib:** [PEP 428](https://peps.python.org/pep-0428/) — motivação para usar Path em vez de str
- **from __future__ import annotations:** [PEP 563](https://peps.python.org/pep-0563/) — avaliação lazy de anotações de tipo
