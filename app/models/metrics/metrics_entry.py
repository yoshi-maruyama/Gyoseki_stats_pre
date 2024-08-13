from app.libs.summary import metrics_summary
from datetime import datetime as dt
import pandas as pd

class MetricsEntry():
    def run(self, data):
        start_date = data.start_date
        end_date = dt.today().strftime('%Y-%m-%d')
        dates = pd.date_range(start=start_date, end=end_date)

        returns = pd.Series(data.returns, index=dates).pct_change().dropna()
        benchmark = pd.Series(data.benchmark, index=dates).pct_change().dropna()

        rf = 0.0
        compounded = True
        periods_per_year = 252
        strategy_title = "returns"
        benchmark_title = "benchmark"
        mtrx = metrics_summary(
            returns=returns,
            benchmark=benchmark,
            rf=rf,
            display=False,
            mode="full",
            sep=True,
            internal="True",
            compounded=compounded,
            periods_per_year=periods_per_year,
            prepare_returns=False,
            benchmark_title=benchmark_title,
            strategy_title=strategy_title,
            match_dates=False,
        )[2:]
        return mtrx
