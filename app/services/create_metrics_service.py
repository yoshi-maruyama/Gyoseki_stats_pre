from app.models.metrics.metrics_builder import MetricsBuilder
from app.models.metrics.metrics_entry import MetricsEntry
from app.schemas.metrics import MetricsSummary

class CreateMetricsService:
    def create_matrics(self, data):
        metrics = MetricsEntry().run(data)
        returns = MetricsBuilder("returns").from_dataframe(metrics)
        benchmark = MetricsBuilder("benchmark").from_dataframe(metrics)
        return MetricsSummary(returns=returns, benchmark=benchmark)
