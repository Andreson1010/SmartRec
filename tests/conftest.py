"""
tests/conftest.py
-----------------
Fixtures compartilhadas por todos os testes do SmartRec.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def interactions_df() -> pd.DataFrame:
    """DataFrame mínimo de interações para treino/teste.

    Gerado deterministicamente com seed=42.
    20 usuários × 30 produtos × 200 interações.
    """
    rng = np.random.default_rng(42)
    n_users = 20
    n_products = 30
    n_rows = 200

    return pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in rng.integers(0, n_users, n_rows)],
            "product_id": [f"p{i}" for i in rng.integers(0, n_products, n_rows)],
            "rating": rng.integers(1, 6, n_rows).astype("float32"),
            "timestamp": rng.integers(1_600_000_000, 1_700_000_000, n_rows),
        }
    )


@pytest.fixture()
def products_df() -> pd.DataFrame:
    """DataFrame mínimo de produtos com metadados de texto."""
    return pd.DataFrame(
        {
            "product_id": [f"p{i}" for i in range(30)],
            "title": [f"Product {i}" for i in range(30)],
            "description": [f"Description for product {i}" for i in range(30)],
            "category": ["Electronics"] * 30,
            "price": [float(i * 10 + 9) for i in range(30)],
        }
    )
