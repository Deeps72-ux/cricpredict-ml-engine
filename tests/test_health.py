"""Tests for health check endpoint."""

from fastapi.testclient import TestClient


def test_health_check_endpoint(client: TestClient):
    """Verify health endpoint returns healthy state and system metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "cricpredict-ml-engine"
    assert data["version"] == "1.0.0"
    assert data["model_loaded"] is True
    assert data["deliveries_loaded"] > 0
