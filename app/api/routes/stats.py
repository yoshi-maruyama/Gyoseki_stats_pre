from typing import Any
from app.services.stats_service import StatsService
from fastapi import APIRouter, Depends
from app.schemas.summary import SummaryResponse

router = APIRouter()

@router.get("/", response_model=Any)
def read_stats(stats_service: StatsService = Depends()) -> Any:
    """
    Retrieve string
    """
    mtrx = stats_service.create_matrix()
    return SummaryResponse(data=mtrx)
