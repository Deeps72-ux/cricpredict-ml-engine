"""Player head-to-head matchup endpoints."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.core.database import MatchAnalyticsDatabase

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/matchup", tags=["Matchups"])


class MatchupStatsResponse(BaseModel):
    """Historical head-to-head metrics between batter and bowler."""

    batter: str
    bowler: str
    balls_faced: int
    runs_scored: int
    dismissals: int
    strike_rate: float
    dot_ball_percentage: float
    boundary_percentage: float
    average: Optional[float] = None
    fours: int
    sixes: int
    phase_breakdown: Dict[str, Any]


class PlayerListResponse(BaseModel):
    """List of players or venues."""

    count: int
    items: List[str]


@router.get(
    "/players/batters",
    response_model=PlayerListResponse,
    summary="List available batters",
    description="Returns all unique batters recorded in the ball-by-ball database.",
)
async def list_batters() -> PlayerListResponse:
    db = MatchAnalyticsDatabase.get_instance()
    batters = db.get_unique_batters()
    return PlayerListResponse(count=len(batters), items=batters)


@router.get(
    "/players/bowlers",
    response_model=PlayerListResponse,
    summary="List available bowlers",
    description="Returns all unique bowlers recorded in the ball-by-ball database.",
)
async def list_bowlers() -> PlayerListResponse:
    db = MatchAnalyticsDatabase.get_instance()
    bowlers = db.get_unique_bowlers()
    return PlayerListResponse(count=len(bowlers), items=bowlers)


@router.get(
    "/players/venues",
    response_model=PlayerListResponse,
    summary="List available venues",
    description="Returns all unique normalized venues in the database.",
)
async def list_venues() -> PlayerListResponse:
    db = MatchAnalyticsDatabase.get_instance()
    venues = db.get_venues()
    return PlayerListResponse(count=len(venues), items=venues)


@router.get(
    "/players/teams",
    response_model=PlayerListResponse,
    summary="List available teams",
    description="Returns all unique teams in the database.",
)
async def list_teams() -> PlayerListResponse:
    db = MatchAnalyticsDatabase.get_instance()
    teams = db.get_teams()
    return PlayerListResponse(count=len(teams), items=teams)


@router.get(
    "/{batter}/{bowler}",
    response_model=MatchupStatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve batter vs bowler historical head-to-head metrics",
    description="Calculates balls faced, strike rate, dot ball percentage, dismissals, and phase breakdowns.",
)
async def get_head_to_head_matchup(batter: str, bowler: str) -> MatchupStatsResponse:
    """Fetch head-to-head historical matchup stats."""
    db = MatchAnalyticsDatabase.get_instance()
    decoded_batter = unquote(batter).strip()
    decoded_bowler = unquote(bowler).strip()

    stats = db.get_matchup_stats(decoded_batter, decoded_bowler)
    if not stats:
        # Check if players exist in registry
        resolved_bat = db.resolve_batter(decoded_batter)
        resolved_bowl = db.resolve_bowler(decoded_bowler)

        if not resolved_bat and not resolved_bowl:
            msg = f"Neither batter '{decoded_batter}' nor bowler '{decoded_bowler}' were found in dataset."
        elif not resolved_bat:
            msg = f"Batter '{decoded_batter}' not found in dataset."
        elif not resolved_bowl:
            msg = f"Bowler '{decoded_bowler}' not found in dataset."
        else:
            msg = f"No historical head-to-head balls recorded between {resolved_bat} and {resolved_bowl}."

        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=msg)

    return MatchupStatsResponse(**stats)
