---
name: embedding-pipeline
description: Padrão para geração e busca de embeddings semânticos no SmartRec. Use esta skill sempre que for implementar ml/semantic/embedder.py ou ml/semantic/retriever.py, gerar ou cachear embeddings de produtos com Sentence Transformers (all-MiniLM-L6-v2), implementar busca por similaridade de cosseno, ou trabalhar com ProductEmbedder e SemanticRetriever.
---

# Skill: embedding-pipeline

Padrão para geração e busca de embeddings semânticos no SmartRec.

## Arquivos envolvidos

```
ml/semantic/
├── embedder.py        ← gera e persiste embeddings
├── retriever.py       ← busca por similaridade coseno
└── artifacts/         ← embeddings salvos em disco
data/embeddings/       ← embeddings de produtos (gerados a partir de products.parquet)
```

## Geração de embeddings (`embedder.py`)

```python
"""
ml/semantic/embedder.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"   # 384 dims, rápido e preciso
EMBEDDINGS_DIR = Path("data/embeddings")


class ProductEmbedder:
    """Gera embeddings de texto para produtos usando Sentence Transformers.

    Parameters
    ----------
    model_name:
        Identificador do modelo no HuggingFace Hub.
    batch_size:
        Tamanho do batch para inferência em GPU/CPU.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        batch_size: int = 128,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: SentenceTransformer | None = None

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Carregando modelo %s ...", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _build_text(self, row: pd.Series) -> str:
        """Concatena title + description em um único texto para embedding."""
        parts = []
        if pd.notna(row.get("title")):
            parts.append(str(row["title"]))
        if pd.notna(row.get("description")):
            parts.append(str(row["description"]))
        return " | ".join(parts) if parts else row.get("product_id", "")

    def fit_transform(self, products: pd.DataFrame) -> np.ndarray:
        """Gera embeddings para todos os produtos.

        Parameters
        ----------
        products:
            DataFrame com colunas ``product_id``, ``title``, ``description``.

        Returns
        -------
        np.ndarray
            Shape (n_products, embedding_dim).
        """
        model = self._load_model()
        texts = products.apply(self._build_text, axis=1).tolist()
        logger.info("Gerando embeddings para %d produtos ...", len(texts))
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,   # cosine similarity = dot product
        )
        return embeddings.astype("float32")

    def save(self, embeddings: np.ndarray, product_ids: pd.Series, path: Path = EMBEDDINGS_DIR) -> None:
        """Persiste embeddings e mapeamento de product_id em disco.

        Arquivos gerados:
            <path>/embeddings.npy    — matriz float32
            <path>/product_ids.npy  — array de strings (alinhado por índice)
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "embeddings.npy", embeddings)
        np.save(path / "product_ids.npy", product_ids.values.astype(str))
        logger.info("Embeddings salvos em %s (%s)", path, embeddings.shape)

    @staticmethod
    def load(path: Path = EMBEDDINGS_DIR) -> tuple[np.ndarray, np.ndarray]:
        """Carrega embeddings e product_ids do disco.

        Returns
        -------
        (embeddings, product_ids)
        """
        path = Path(path)
        embeddings = np.load(path / "embeddings.npy")
        product_ids = np.load(path / "product_ids.npy", allow_pickle=True)
        return embeddings, product_ids
```

## Busca por similaridade (`retriever.py`)

```python
"""
ml/semantic/retriever.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ml.semantic.embedder import ProductEmbedder, EMBEDDINGS_DIR


class SemanticRetriever:
    """Busca os K produtos mais similares dado um embedding de consulta.

    Usa similaridade coseno via dot product (embeddings normalizados).
    """

    def __init__(self, embeddings_dir: Path = EMBEDDINGS_DIR) -> None:
        self._embeddings, self._product_ids = ProductEmbedder.load(embeddings_dir)
        # _embeddings já normalizados → dot product = cosine similarity

    def query_by_product(self, product_id: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Retorna os K produtos mais similares a um produto dado.

        Parameters
        ----------
        product_id:
            Produto de referência (deve estar no índice de embeddings).
        top_k:
            Número de resultados a retornar.

        Returns
        -------
        list[dict]
            Lista de ``{"product_id": str, "score": float}``.
        """
        idx = self._find_index(product_id)
        if idx is None:
            return []

        query_vec = self._embeddings[idx]                    # shape (dim,)
        scores = self._embeddings @ query_vec                # shape (n_products,)
        scores[idx] = -1                                     # excluir o próprio item

        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [
            {"product_id": str(self._product_ids[i]), "score": float(scores[i])}
            for i in top_indices
        ]

    def query_by_vector(self, vector: np.ndarray, top_k: int = 10) -> list[dict[str, Any]]:
        """Busca por similaridade dado um vetor de consulta externo (já normalizado)."""
        scores = self._embeddings @ vector
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [
            {"product_id": str(self._product_ids[i]), "score": float(scores[i])}
            for i in top_indices
        ]

    def _find_index(self, product_id: str) -> int | None:
        matches = np.where(self._product_ids == product_id)[0]
        return int(matches[0]) if len(matches) > 0 else None
```

## Script de geração (executar uma vez antes do treino)

```python
# python -m ml.semantic.embedder   (ou incluir no Makefile/pipeline de treino)
if __name__ == "__main__":
    import pandas as pd
    from ml.semantic.embedder import ProductEmbedder

    products = pd.read_parquet("data/processed/products.parquet")
    embedder = ProductEmbedder()
    embeddings = embedder.fit_transform(products)
    embedder.save(embeddings, products["product_id"])
```

## Checklist

- [ ] `normalize_embeddings=True` no `encode` — obrigatório para usar dot product como coseno
- [ ] `embeddings.npy` e `product_ids.npy` sempre alinhados por índice
- [ ] `query_by_product` exclui o próprio item (`scores[idx] = -1`)
- [ ] Cold start: produto sem embedding retorna lista vazia (sem exceção)
- [ ] Testes usam embeddings aleatórios normalizados em vez do modelo real
- [ ] Script de geração logado como artefato MLflow no experimento `smartrec/semantic`
