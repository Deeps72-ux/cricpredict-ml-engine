"""Main FastAPI application entrypoint for CricPredict ML Engine."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict

import uvicorn
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from app.api.router import api_router
from app.core.config import settings
from app.core.database import MatchAnalyticsDatabase
from app.ml.model_loader import get_model_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cricpredict")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan event handler to pre-warm caches and models."""
    logger.info("Starting CricPredict ML Engine v%s...", settings.VERSION)
    try:
        # Pre-load database and model
        db = MatchAnalyticsDatabase.get_instance()
        logger.info("Loaded database with %d deliveries.", len(db.df))
        model_mgr = get_model_manager()
        logger.info("Pre-warmed XGBoost model (Loaded: %s)", model_mgr.artifact is not None)
    except Exception as e:
        logger.error("Error during application warmup: %s", e, exc_info=True)
    yield
    logger.info("Shutting down CricPredict ML Engine...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Production ML match analytics & 10,000-iteration vectorized Monte Carlo simulation engine "
        "for T20 cricket (IPL). Features live calibrated win probability curves, batter-vs-bowler "
        "micro-matchup embeddings, and venue-adjusted chase confidence intervals."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API v1 Router
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redirect root path to interactive Swagger documentation."""
    return RedirectResponse(url="/docs")


@app.get(
    "/health",
    status_code=status.HTTP_200_OK,
    tags=["System"],
    summary="Health check & readiness probe",
)
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for container orchestrators and monitoring."""
    db_loaded = False
    deliveries_count = 0
    try:
        db = MatchAnalyticsDatabase.get_instance()
        deliveries_count = len(db.df)
        db_loaded = deliveries_count > 0
    except Exception:
        pass

    model_loaded = False
    try:
        mgr = get_model_manager()
        model_loaded = mgr.pipeline is not None
    except Exception:
        pass

    healthy = db_loaded and model_loaded

    return {
        "status": "healthy" if healthy else "degraded",
        "service": "cricpredict-ml-engine",
        "version": settings.VERSION,
        "model_loaded": model_loaded,
        "deliveries_loaded": deliveries_count,
        "environment": settings.APP_ENV,
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
