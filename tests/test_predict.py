"""Tests for win probability prediction API endpoints."""

from fastapi.testclient import TestClient


def test_predict_win_probability_standard(client: TestClient):
    """Test standard live in-play prediction."""
    payload = {
        "team_batting": "Royal Challengers Bengaluru",
        "team_bowling": "Chennai Super Kings",
        "venue": "M Chinnaswamy Stadium, Bengaluru",
        "target_runs": 185,
        "current_score": 90,
        "overs_completed": 10.0,
        "wickets_fallen": 2,
        "dew_factor": 0.1,
    }
    response = client.post("/api/v1/predict/win-probability", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["batting_team"] == "Royal Challengers Bengaluru"
    assert data["bowling_team"] == "Chennai Super Kings"
    assert 0.0 <= data["win_probability_batting"] <= 1.0
    assert 0.0 <= data["win_probability_bowling"] <= 1.0
    assert round(data["win_probability_batting"] + data["win_probability_bowling"], 2) == 1.0
    assert data["match_phase"] == "middle"
    assert data["runs_needed"] == 95
    assert data["balls_remaining"] == 60
    assert data["required_run_rate"] == 9.5
    assert data["current_run_rate"] == 9.0
    assert data["is_boundary_case"] is False


def test_predict_boundary_target_reached(client: TestClient):
    """Test boundary condition where chasing team already achieved the target."""
    payload = {
        "team_batting": "Mumbai Indians",
        "team_bowling": "Kolkata Knight Riders",
        "venue": "Wankhede Stadium, Mumbai",
        "target_runs": 150,
        "current_score": 152,
        "overs_completed": 17.2,
        "wickets_fallen": 3,
    }
    response = client.post("/api/v1/predict/win-probability", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["win_probability_batting"] == 1.0
    assert data["win_probability_bowling"] == 0.0
    assert data["is_boundary_case"] is True
    assert "already reached" in data["boundary_reason"]


def test_predict_boundary_all_out(client: TestClient):
    """Test boundary condition where chasing team lost all 10 wickets."""
    payload = {
        "team_batting": "Delhi Capitals",
        "team_bowling": "Rajasthan Royals",
        "venue": "Arun Jaitley Stadium, Delhi",
        "target_runs": 190,
        "current_score": 120,
        "overs_completed": 16.0,
        "wickets_fallen": 10,
    }
    response = client.post("/api/v1/predict/win-probability", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["win_probability_batting"] == 0.0
    assert data["win_probability_bowling"] == 1.0
    assert data["is_boundary_case"] is True
    assert "wickets fallen" in data["boundary_reason"]


def test_predict_boundary_impossible_target(client: TestClient):
    """Test impossible chase (e.g. 50 runs needed off 3 balls)."""
    payload = {
        "team_batting": "Sunrisers Hyderabad",
        "team_bowling": "Gujarat Titans",
        "venue": "Rajiv Gandhi International Stadium, Hyderabad",
        "target_runs": 200,
        "current_score": 150,
        "overs_completed": 19.3,  # 3 balls remaining (max possible = 18)
        "wickets_fallen": 6,
    }
    response = client.post("/api/v1/predict/win-probability", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["win_probability_batting"] <= 0.01
    assert data["is_boundary_case"] is True


def test_predict_validation_error(client: TestClient):
    """Verify pydantic rejection of invalid inputs."""
    payload = {
        "team_batting": "RCB",
        "team_bowling": "CSK",
        "target_runs": -10,  # Invalid
        "current_score": 50,
        "overs_completed": 25.0,  # Invalid
        "wickets_fallen": 12,  # Invalid
    }
    response = client.post("/api/v1/predict/win-probability", json=payload)
    assert response.status_code == 422
