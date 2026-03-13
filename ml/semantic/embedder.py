"""
ml/semantic/embedder.py
-----------------------
Geração e persistência de embeddings de produtos com Sentence Transformers.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_NAME = "all-MiniLM-L6-v2"  # 384 dims, rápido e preciso
EMBEDDINGS_DIR = ROOT / "data" / "embeddings"


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
        return " | ".join(parts) if parts else str(row.get("product_id", ""))

    def fit_transform(self, products: pd.DataFrame) -> np.ndarray:
        """Gera embeddings para todos os produtos.

        Parameters
        ----------
        products:
            DataFrame com colunas ``product_id``, ``title``, ``description``.

        Returns
        -------
        np.ndarray
            Shape ``(n_products, embedding_dim)``.
        """
        model = self._load_model()
        texts = products.apply(self._build_text, axis=1).tolist()
        logger.info("Gerando embeddings para %d produtos ...", len(texts))
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # cosine similarity = dot product
        )
        return embeddings.astype("float32")

    def save(
        self,
        embeddings: np.ndarray,
        product_ids: pd.Series,
        path: Path = EMBEDDINGS_DIR,
    ) -> None:
        """Persiste embeddings e mapeamento de product_id em disco.

        Arquivos gerados:
            ``<path>/embeddings.npy``   — matriz float32
            ``<path>/product_ids.npy`` — array de strings (alinhado por índice)

        Parameters
        ----------
        embeddings:
            Matriz de embeddings gerada por :meth:`fit_transform`.
        product_ids:
            Series com os product_ids alinhados com as linhas de ``embeddings``.
        path:
            Diretório de destino (criado se não existir).
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "embeddings.npy", embeddings)
        np.save(path / "product_ids.npy", product_ids.values.astype(str))
        logger.info("Embeddings salvos em %s %s", path, embeddings.shape)

    @staticmethod
    def load(path: Path = EMBEDDINGS_DIR) -> tuple[np.ndarray, np.ndarray]:
        """Carrega embeddings e product_ids do disco.

        Parameters
        ----------
        path:
            Diretório onde os arquivos foram salvos por :meth:`save`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(embeddings, product_ids)`` — alinhados por índice.
        """
        path = Path(path)
        embeddings = np.load(path / "embeddings.npy")
        product_ids = np.load(path / "product_ids.npy", allow_pickle=True)
        return embeddings, product_ids


if __name__ == "__main__":
    products = pd.read_parquet(ROOT / "data" / "processed" / "products.parquet")
    embedder = ProductEmbedder()
    embs = embedder.fit_transform(products)
    embedder.save(embs, products["product_id"])
