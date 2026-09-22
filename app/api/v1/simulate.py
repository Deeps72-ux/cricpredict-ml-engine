"""Monte Carlo chase simulation endpoint."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.ml.monte_carlo import MonteCarloSimulator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/simulate", tags=["Simulation"])


class ChaseSimulationRequest(BaseModel):
    """Payload for Monte Carlo stochastic chase simulation."""

    target_runs: int = Field(..., ge=1, le=400, description="Target runs needed to win", example=185)
    current_score: int = Field(default=0, ge=0, le=400, description="Current runs scored in chase", example=80)
    overs_completed: float = Field(
        default=0.0,
        ge=0.0,
        le=20.0,
        description="Overs completed so far (e.g. 10.0)",
        example=10.0,
    )
    wickets_fallen: int = Field(default=0, ge=0, le=10, description="Wickets lost so far", example=2)
    iterations: int = Field(
        default=10000,
        ge=100,
        le=100000,
        description="Number of parallel stochastic trajectories",
        example=10000,
    )
    venue: Optional[str] = Field(default=None, description="Venue stadium name for pitch tuning")
    seed: Optional[int] = Field(default=None, description="Optional random seed for deterministic reproduction")


class ChaseSimulationResponse(BaseModel):
    """Output distribution and confidence interval metrics from simulation."""

    win_probability: float
    iterations: int
    projected_score_mean: float
    projected_score_median: float
    projected_score_std: float
    confidence_intervals: Dict[str, List[float]]
    histogram: Dict[str, Any]
    median_balls_to_win: Optional[float] = None
    simulation_time_ms: float
    target_runs: int
    current_score: int
    wickets_fallen: int


@router.post(
    "/chase",
    response_model=ChaseSimulationResponse,
    status_code=status.HTTP_200_OK,
    summary="Simulate match chase trajectories via Vectorized Monte Carlo",
    description="Simulates 10,000 match outcomes in parallel using vectorized NumPy to return target reach distributions and confidence intervals.",
)
async def simulate_chase(payload: ChaseSimulationRequest) -> ChaseSimulationResponse:
    """Run 10,000-iteration stochastic simulation."""
    try:
        simulator = MonteCarloSimulator(random_seed=payload.seed)
        result = simulator.simulate_chase(
            target_runs=payload.target_runs,
            current_score=payload.current_score,
            overs_completed=payload.overs_completed,
            wickets_fallen=payload.wickets_fallen,
            iterations=payload.iterations,
            venue=payload.venue,
        )
        return ChaseSimulationResponse(**result)
    except Exception as e:
        logger.error("Simulation failure: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Monte Carlo simulation failed: {str(e)}",
        )
