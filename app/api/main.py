from fastapi import APIRouter

from app.api.routes import stats

api_router = APIRouter()
api_router.include_router(stats.router, prefix="/metrics", tags=["metrics"])
