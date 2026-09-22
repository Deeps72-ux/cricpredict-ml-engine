"""Unit tests for feature engineering logic."""

from app.etl.feature_engineering import (
    PHASE_POWERPLAY,
    PHASE_MIDDLE,
    PHASE_DEATH,
    get_match_phase,
    calculate_crr,
    calculate_rrr,
    MatchFeatureEngineer,
)


def test_match_phase_detection():
    """Verify correct phase classification across overs."""
    assert get_match_phase(0.0) == PHASE_POWERPLAY
    assert get_match_phase(5.5) == PHASE_POWERPLAY
    assert get_match_phase(6.0) == PHASE_MIDDLE
    assert get_match_phase(14.5) == PHASE_MIDDLE
    assert get_match_phase(15.0) == PHASE_DEATH
    assert get_match_phase(19.5) == PHASE_DEATH


def test_run_rate_calculations():
    """Verify CRR and RRR math."""
    # 60 runs in 6 overs = 10.0 CRR
    assert calculate_crr(60, 6.0) == 10.0
    # 0 runs in 0 overs = 0.0
    assert calculate_crr(0, 0.0) == 0.0
    # 60 needed in 6 overs = 10.0 RRR
    assert calculate_rrr(60, 6.0) == 10.0
    # 0 needed in 5 overs = 0.0 RRR
    assert calculate_rrr(0, 5.0) == 0.0


def test_build_feature_dict():
    """Verify dictionary building for single match states."""
    fd = MatchFeatureEngineer.build_feature_dict(
        target_runs=180,
        current_score=90,
        overs_completed=10.0,
        wickets_fallen=3,
        venue="Wankhede Stadium, Mumbai",
    )
    assert fd["runs_needed"] == 90
    assert fd["balls_remaining"] == 60
    assert fd["wickets_fallen"] == 3
    assert fd["wickets_remaining"] == 7
    assert fd["current_run_rate"] == 9.0
    assert fd["required_run_rate"] == 9.0
    assert fd["match_phase"] == PHASE_MIDDLE
