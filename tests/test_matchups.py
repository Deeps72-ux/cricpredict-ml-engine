"""Tests for batter vs bowler historical matchup endpoints."""

from fastapi.testclient import TestClient


def test_matchup_known_players(client: TestClient):
    """Test lookup between known marquee players."""
    response = client.get("/api/v1/matchup/V%20Kohli/JJ%20Bumrah")
    assert response.status_code == 200
    data = response.json()

    assert data["batter"] == "V Kohli"
    assert data["bowler"] == "JJ Bumrah"
    assert data["balls_faced"] > 50
    assert data["runs_scored"] > 50
    assert data["strike_rate"] > 100.0
    assert 0.0 <= data["dot_ball_percentage"] <= 100.0
    assert "phase_breakdown" in data
    assert "powerplay" in data["phase_breakdown"]
    assert "middle" in data["phase_breakdown"]
    assert "death" in data["phase_breakdown"]


def test_matchup_unknown_players(client: TestClient):
    """Test lookup for unknown fictional players returns 404."""
    response = client.get("/api/v1/matchup/UnknownBatterX/UnknownBowlerY")
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower() or "neither" in data["detail"].lower()


def test_matchup_player_listings(client: TestClient):
    """Test player registry listing endpoints."""
    r_bat = client.get("/api/v1/matchup/players/batters")
    assert r_bat.status_code == 200
    assert r_bat.json()["count"] > 100
    assert "V Kohli" in r_bat.json()["items"]

    r_bowl = client.get("/api/v1/matchup/players/bowlers")
    assert r_bowl.status_code == 200
    assert r_bowl.json()["count"] > 100
    assert "JJ Bumrah" in r_bowl.json()["items"]
