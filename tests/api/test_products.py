"""
tests/api/test_products.py
--------------------------
Testes para o endpoint GET /products/{product_id}/similar.

ProductService é mockado — nenhum embedding real é carregado.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from api.models.products import SimilarItem, SimilarProductsResponse

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_RESPONSE = SimilarProductsResponse(
    product_id="p1",
    similar=[
        SimilarItem(product_id="p2", score=0.95),
        SimilarItem(product_id="p3", score=0.87),
        SimilarItem(product_id="p4", score=0.72),
    ],
    model_version="1.0.0",
)


@pytest.fixture()
def client():
    """TestClient com ProductService mockado."""
    from api.main import app
    from api.routers.products import ProductService

    mock_service = MagicMock(spec=ProductService)
    mock_service.find_similar.return_value = MOCK_RESPONSE

    app.dependency_overrides[ProductService] = lambda: mock_service
    yield TestClient(app)
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestGetSimilarProducts:
    def test_returns_200(self, client) -> None:
        resp = client.get("/products/p1/similar")
        assert resp.status_code == 200

    def test_response_schema(self, client) -> None:
        resp = client.get("/products/p1/similar")
        body = resp.json()

        assert body["product_id"] == "p1"
        assert "similar" in body
        assert "model_version" in body

    def test_similar_items(self, client) -> None:
        resp = client.get("/products/p1/similar")
        items = resp.json()["similar"]

        assert len(items) == 3
        for item in items:
            assert "product_id" in item
            assert "score" in item
            assert 0.0 <= item["score"] <= 1.0

    def test_top_k_passed_to_service(self) -> None:
        """Verifica que top_k da query string é repassado ao service."""
        from api.main import app
        from api.routers.products import ProductService

        mock_service = MagicMock(spec=ProductService)
        mock_service.find_similar.return_value = MOCK_RESPONSE
        app.dependency_overrides[ProductService] = lambda: mock_service

        TestClient(app).get("/products/p1/similar?top_k=5")
        mock_service.find_similar.assert_called_once_with("p1", top_k=5)

        app.dependency_overrides.clear()

    def test_default_top_k_is_10(self) -> None:
        """Sem top_k, o padrão é 10."""
        from api.main import app
        from api.routers.products import ProductService

        mock_service = MagicMock(spec=ProductService)
        mock_service.find_similar.return_value = MOCK_RESPONSE
        app.dependency_overrides[ProductService] = lambda: mock_service

        TestClient(app).get("/products/p1/similar")
        mock_service.find_similar.assert_called_once_with("p1", top_k=10)

        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Erros de validação (422)
# ---------------------------------------------------------------------------


class TestValidation:
    def test_top_k_zero_returns_422(self, client) -> None:
        resp = client.get("/products/p1/similar?top_k=0")
        assert resp.status_code == 422

    def test_top_k_above_100_returns_422(self, client) -> None:
        resp = client.get("/products/p1/similar?top_k=101")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Erros do service
# ---------------------------------------------------------------------------


class TestServiceErrors:
    def test_value_error_returns_404(self, client) -> None:
        from api.main import app
        from api.routers.products import ProductService

        mock_service = MagicMock(spec=ProductService)
        mock_service.find_similar.side_effect = ValueError("produto não encontrado")
        app.dependency_overrides[ProductService] = lambda: mock_service

        resp = client.get("/products/p_bad/similar")
        assert resp.status_code == 404
        assert "produto não encontrado" in resp.json()["detail"]

        app.dependency_overrides.clear()

    def test_unexpected_error_returns_500(self, client) -> None:
        from api.main import app
        from api.routers.products import ProductService

        mock_service = MagicMock(spec=ProductService)
        mock_service.find_similar.side_effect = RuntimeError("falha interna")
        app.dependency_overrides[ProductService] = lambda: mock_service

        resp = client.get("/products/p1/similar")
        assert resp.status_code == 500

        app.dependency_overrides.clear()
