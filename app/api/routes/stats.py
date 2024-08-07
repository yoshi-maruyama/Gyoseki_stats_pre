from typing import Any
from app.services.stats_service import StatsService
from fastapi import APIRouter, Depends
from pydantic import BaseModel

router = APIRouter()

class Stats(BaseModel):
    id: str

@router.get("/", response_model=Stats)
def read_stats(stats_service: StatsService = Depends()) -> Any:
    """
    Retrieve string
    """
    hoge = stats_service.create_matrix()
    return Stats(id=hoge)
