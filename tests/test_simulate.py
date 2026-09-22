"""Tests for Monte Carlo simulation API endpoints."""

from fastapi.testclient import TestClient


def test_simulate_chase_standard(client: TestClient):
    """Test 10,000 iteration simulation endpoint output structure and performance."""
    payload = {
        "target_runs": 180,
        "current_score": 90,
        "overs_completed": 10.0,
        "wickets_fallen": 2,
        "iterations": 10000,
        "seed": 42,
    }
    response = client.post("/api/v1/simulate/chase", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert 0.0 <= data["win_probability"] <= 1.0
    assert data["iterations"] == 10000
    assert data["projected_score_mean"] > 100
    assert data["simulation_time_ms"] < 300.0  # Must be fast (< 300ms)

    # Check confidence intervals
    ci = data["confidence_intervals"]
    assert "50%" in ci and "80%" in ci and "95%" in ci
    assert ci["95%"][0] <= ci["50%"][0] <= ci["50%"][1] <= ci["95%"][1]

    # Check histogram
    hist = data["histogram"]
    assert "counts" in hist and "bin_edges" in hist
    assert sum(hist["counts"]) == 10000


def test_simulate_chase_boundary_reached(client: TestClient):
    """Test deterministic output when target is already reached."""
    payload = {
        "target_runs": 150,
        "current_score": 155,
        "overs_completed": 18.0,
        "wickets_fallen": 3,
        "iterations": 5000,
    }
    response = client.post("/api/v1/simulate/chase", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["win_probability"] == 1.0
    assert data["projected_score_mean"] == 155.0


def test_simulate_chase_boundary_all_out(client: TestClient):
    """Test deterministic output when all 10 wickets are down."""
    payload = {
        "target_runs": 180,
        "current_score": 110,
        "overs_completed": 14.2,
        "wickets_fallen": 10,
        "iterations": 5000,
    }
    response = client.post("/api/v1/simulate/chase", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["win_probability"] == 0.0
