from app.services.stats_service import StatsService
from fastapi import APIRouter, Depends
from app.schemas.summary import SummaryRequest, SummaryResponse

router = APIRouter()

@router.post("/", response_model=SummaryResponse)
def summarise_stats(stats_request: SummaryRequest, stats_service: StatsService = Depends()) -> SummaryResponse:
    """
    Retrieve "returns", "benchmark", and "start_date"
    and returns summarised statistics based on these values
    """
    mtrx = stats_service.create_matrix(stats_request)
    return SummaryResponse(data=mtrx)
