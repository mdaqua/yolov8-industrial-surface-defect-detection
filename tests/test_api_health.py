import pytest

pytest.importorskip("fastapi")
pytest.importorskip("PIL")
pytest.importorskip("yaml")

from fastapi.testclient import TestClient

from src.api.main import app


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "model_available" in payload
