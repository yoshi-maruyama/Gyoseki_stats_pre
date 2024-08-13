from fastapi import APIRouter, Depends
from app.services.create_metrics_service import CreateMetricsService
from app.schemas.metrics import SummaryRequest, MetricsSummary

router = APIRouter()

@router.post("/", response_model=MetricsSummary)
def summarise_stats(stats_request: SummaryRequest, stats_service: CreateMetricsService = Depends()) -> MetricsSummary:
    """
    Retrieve "returns", "benchmark", and "start_date"
    and returns summarised statistics based on these values
    """
    mtrx_summary = stats_service.create_matrics(stats_request)
    return mtrx_summary
