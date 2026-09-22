"""Pytest configuration and shared fixtures for CricPredict ML Engine."""

import pytest
from fastapi.testclient import TestClient

from app.core.database import MatchAnalyticsDatabase
from app.main import app
from app.ml.model_loader import get_model_manager


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Session-scoped FastAPI test client."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def model_manager():
    """Ensure ML model is pre-warmed for testing."""
    return get_model_manager()


@pytest.fixture(scope="session")
def analytics_db():
    """Ensure database is loaded for testing."""
    return MatchAnalyticsDatabase.get_instance()
