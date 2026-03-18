"""
tests/api/test_recommendations.py
-----------------------------------
Testes para os endpoints POST /recommendations e GET /health — service mockado.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from api.models.recommendations import (
    RecommendationResponse,
    RecommendedItem,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_service():
    """RecommendationService mockado com _model e loaded_at preenchidos."""
    from api.services.recommendations import RecommendationService

    mock_service = MagicMock(spec=RecommendationService)
    mock_service.run.return_value = RecommendationResponse(
        user_id="u1",
        recommendations=[
            RecommendedItem(product_id="p1", score=0.9),
            RecommendedItem(product_id="p2", score=0.7),
            RecommendedItem(product_id="p3", score=0.5),
        ],
        model_version="1.0.0",
    )
    mock_service._model = MagicMock()
    mock_service._model.version = "1.0.0"
    mock_service._model.strategy = "rerank"
    mock_service.loaded_at = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    return mock_service


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    """TestClient com app.state.rec_service mockado."""
    from api.main import app
    from api.routers.recommendations import get_rec_service

    mock_service = _make_mock_service()
    app.dependency_overrides[get_rec_service] = lambda: mock_service
    app.state.rec_service = mock_service
    app.state.started_at = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    yield TestClient(app)
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestGetRecommendations:
    def test_returns_200(self, client) -> None:
        resp = client.post("/recommendations/", json={"user_id": "u1"})
        assert resp.status_code == 200

    def test_response_schema(self, client) -> None:
        resp = client.post("/recommendations/", json={"user_id": "u1", "top_k": 3})
        body = resp.json()

        assert body["user_id"] == "u1"
        assert "recommendations" in body
        assert "model_version" in body

    def test_recommendations_items(self, client) -> None:
        resp = client.post("/recommendations/", json={"user_id": "u1", "top_k": 3})
        items = resp.json()["recommendations"]

        assert len(items) == 3
        for item in items:
            assert "product_id" in item
            assert "score" in item
            assert 0.0 <= item["score"] <= 1.0

    def test_default_top_k_is_10(self, client) -> None:
        """Sem top_k no payload, padrão é 10."""
        resp = client.post("/recommendations/", json={"user_id": "u1"})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Erros de validação (422)
# ---------------------------------------------------------------------------


class TestValidation:
    def test_missing_user_id_returns_422(self, client) -> None:
        resp = client.post("/recommendations/", json={"top_k": 5})
        assert resp.status_code == 422

    def test_empty_user_id_returns_422(self, client) -> None:
        resp = client.post("/recommendations/", json={"user_id": "", "top_k": 5})
        assert resp.status_code == 422

    def test_top_k_zero_returns_422(self, client) -> None:
        resp = client.post("/recommendations/", json={"user_id": "u1", "top_k": 0})
        assert resp.status_code == 422

    def test_top_k_above_100_returns_422(self, client) -> None:
        resp = client.post("/recommendations/", json={"user_id": "u1", "top_k": 101})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Erros do service
# ---------------------------------------------------------------------------


class TestServiceErrors:
    def test_value_error_returns_404(self, client) -> None:
        from api.main import app
        from api.routers.recommendations import get_rec_service

        mock_service = _make_mock_service()
        mock_service.run.side_effect = ValueError("usuário não encontrado")
        app.dependency_overrides[get_rec_service] = lambda: mock_service

        resp = client.post("/recommendations/", json={"user_id": "u_bad"})
        assert resp.status_code == 404
        assert "usuário não encontrado" in resp.json()["detail"]

        app.dependency_overrides.clear()

    def test_unexpected_error_returns_500(self, client) -> None:
        from api.main import app
        from api.routers.recommendations import get_rec_service

        mock_service = _make_mock_service()
        mock_service.run.side_effect = RuntimeError("falha interna")
        app.dependency_overrides[get_rec_service] = lambda: mock_service

        resp = client.post("/recommendations/", json={"user_id": "u1"})
        assert resp.status_code == 500

        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_200(self, client) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_schema(self, client) -> None:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert "model_version" in body
        assert "model_strategy" in body
        assert "loaded_at" in body
        assert "uptime_s" in body

    def test_health_model_info(self, client) -> None:
        body = client.get("/health").json()
        assert body["model_version"] == "1.0.0"
        assert body["model_strategy"] == "rerank"

    def test_health_loaded_at_is_iso(self, client) -> None:
        body = client.get("/health").json()
        datetime.fromisoformat(body["loaded_at"])

    def test_health_uptime_is_non_negative(self, client) -> None:
        body = client.get("/health").json()
        assert body["uptime_s"] >= 0.0


# ---------------------------------------------------------------------------
# Middleware — X-Response-Time-Ms
# ---------------------------------------------------------------------------


class TestLatencyMiddleware:
    def test_header_present_on_recommendations(self, client) -> None:
        resp = client.post("/recommendations/", json={"user_id": "u1"})
        assert "x-response-time-ms" in resp.headers

    def test_header_present_on_health(self, client) -> None:
        resp = client.get("/health")
        assert "x-response-time-ms" in resp.headers

    def test_header_is_numeric(self, client) -> None:
        resp = client.post("/recommendations/", json={"user_id": "u1"})
        value = float(resp.headers["x-response-time-ms"])
        assert value >= 0.0
