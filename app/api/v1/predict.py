"""Win probability prediction endpoint."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.core.config import settings
from app.ml.model_loader import get_model_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["Prediction"])


class WinProbabilityRequest(BaseModel):
    """Payload for live match state win probability calculation."""

    team_batting: str = Field(..., description="Chasing batting team name", example="Royal Challengers Bengaluru")
    team_bowling: str = Field(..., description="Defending bowling team name", example="Chennai Super Kings")
    venue: str = Field(default="Wankhede Stadium, Mumbai", description="Match venue / stadium name")
    target_runs: int = Field(..., ge=1, le=400, description="Target runs set by 1st innings to win", example=185)
    current_score: int = Field(..., ge=0, le=400, description="Current runs scored by chasing team", example=95)
    overs_completed: float = Field(
        ...,
        ge=0.0,
        le=20.0,
        description="Overs completed (e.g. 10.3 means 10 overs and 3 balls)",
        example=11.2,
    )
    wickets_fallen: int = Field(..., ge=0, le=10, description="Wickets lost by chasing team (0 to 10)", example=3)
    dew_factor: Optional[float] = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Dew impact index (0.0 = dry, 1.0 = heavy dew)",
        example=0.2,
    )


class WinProbabilityResponse(BaseModel):
    """Response containing live win probability metrics."""

    batting_team: str
    bowling_team: str
    win_probability_batting: float
    win_probability_bowling: float
    required_run_rate: float
    current_run_rate: float
    runs_needed: int
    balls_remaining: int
    wickets_fallen: int
    wickets_remaining: int
    match_phase: str
    venue: str
    model_version: str
    is_boundary_case: bool
    boundary_reason: Optional[str] = None


@router.post(
    "/win-probability",
    response_model=WinProbabilityResponse,
    status_code=status.HTTP_200_OK,
    summary="Calculate live in-play win probability",
    description="Accepts current match state and outputs calibrated win probabilities for both teams via XGBoost.",
)
async def predict_win_probability(payload: WinProbabilityRequest) -> WinProbabilityResponse:
    """Evaluate live match state through calibrated XGBoost pipeline."""
    try:
        manager = get_model_manager()
        result = manager.predict_win_probability(
            target_runs=payload.target_runs,
            current_score=payload.current_score,
            overs_completed=payload.overs_completed,
            wickets_fallen=payload.wickets_fallen,
            venue=payload.venue,
            dew_factor=payload.dew_factor or 0.0,
        )

        fd = result["feature_dict"]
        return WinProbabilityResponse(
            batting_team=payload.team_batting,
            bowling_team=payload.team_bowling,
            win_probability_batting=result["win_probability_batting"],
            win_probability_bowling=result["win_probability_bowling"],
            required_run_rate=fd["required_run_rate"],
            current_run_rate=fd["current_run_rate"],
            runs_needed=fd["runs_needed"],
            balls_remaining=fd["balls_remaining"],
            wickets_fallen=fd["wickets_fallen"],
            wickets_remaining=fd["wickets_remaining"],
            match_phase=fd["match_phase"],
            venue=fd["venue"],
            model_version=settings.VERSION,
            is_boundary_case=result.get("is_boundary_case", False),
            boundary_reason=result.get("boundary_reason"),
        )
    except Exception as e:
        logger.error("Error evaluating win probability: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Win probability calculation failed: {str(e)}",
        )
