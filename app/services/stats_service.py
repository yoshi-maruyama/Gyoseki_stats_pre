from app.models.summary.summary import get_metrix
class StatsService:
    def create_matrix(self):
        data = {
            "returns": [17137.240234, 17344.710938, 17642.730469, 17613.039062, 17572.730469, 17755.070312],
            "benchmark": [17190.240234, 17369.710938, 17665.730469, 17675.039062, 17666.730469, 17808.070312],
            "start_date": "2024-08-03"
        }

        metrics = get_metrix(data)

        return metrics.to_json()
