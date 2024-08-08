from app.models.summary.summary import get_metrix
class StatsService:
    def create_matrix(self, data):
        metrics = get_metrix(data)
        return metrics.to_json()
