"""Top-level API v1 router bundling all sub-routers."""

from fastapi import APIRouter

from app.api.v1.matchups import router as matchups_router
from app.api.v1.predict import router as predict_router
from app.api.v1.simulate import router as simulate_router

api_router = APIRouter()

api_router.include_router(predict_router)
api_router.include_router(simulate_router)
api_router.include_router(matchups_router)
