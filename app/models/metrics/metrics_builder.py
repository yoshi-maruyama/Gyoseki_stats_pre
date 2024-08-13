import pandas as pd
from app.models.metrics.metrics import Metrics

class MetricsBuilder():
    def __init__(self, metrics_name) -> None:
        self.metrics_name = metrics_name

    def from_dataframe(self, df: pd.DataFrame) -> "MetricsBuilder":
        mtrx = df[self.metrics_name]
        return Metrics(
            risk_free_rate=mtrx.get("Risk-Free Rate", None),
            time_in_market=mtrx.get("Time in Market", None),
            cumulative_return=mtrx.get("Cumulative Return", None),
            total_return=mtrx.get("Total Return", None),
            cagr=mtrx.get("CAGR﹪", None),
            prob_sharpe_ratio=mtrx.get("Sharpe", None),
            smart_sharp=mtrx.get("Smart Sharpe", None),
            sortino=mtrx.get("Sortino", None),
            smart_sortino=mtrx.get("Smart Sortino", None),
            sortino_root_2=mtrx.get("Sortino/√2", None),
            smart_sortino_root_2=mtrx.get("Smart Sortino/√2", None),
            omega=mtrx.get("Omega", None),
            max_drawdown=mtrx.get("Max Drawdown", None),
            volatility=mtrx.get("Volatility (ann.)", None),
            r_2=mtrx.get("R^2", None),
            imformation_ratio=mtrx.get("Information Ratio", None),
            calmer=mtrx.get("Calmar", None),
            skew=mtrx.get("Skew", None),
            kurtosis=mtrx.get("Kurtosis", None),
            expected_daily=mtrx.get("Expected Daily", None),
            expected_monthly=mtrx.get("Expected Monthly", None),
            expected_yearly=mtrx.get("Expected Yearly", None),
            kelly_criterion=mtrx.get("Kelly Criterion", None),
            risk_of_ruin=mtrx.get("Risk of Ruin", None),
            daily_value_at_risk=mtrx.get("Daily Value-at-Risk", None),
            expected_shortfall_cvar=mtrx.get("Expected Shortfall (cVaR)", None),
            max_consecutive_wins=mtrx.get("Max Consecutive Wins", None),
            max_consecutive_losses=mtrx.get("Max Consecutive Losses", None),
            gain_pain_ratio=mtrx.get("Gain/Pain Ratio", None),
            gain_pain_1m=mtrx.get("Gain/Pain (1M)", None),
            payoff_ratio=mtrx.get("Payoff Ratio", None),
            profit_factor=mtrx.get("Profit Factor", None),
            common_sense_ratio=mtrx.get("Common Sense Ratio", None),
            cpc_index=mtrx.get("CPC Index", None),
            tail_ratio=mtrx.get("Tail Ratio", None),
            outer_win_ratio=mtrx.get("Outlier Win Ratio", None),
            outer_loss_ratio=mtrx.get("Outlier Loss Ratio", None),
            mtd=mtrx.get("MTD", None),
            three_m=mtrx.get("3M", None),
            six_m=mtrx.get("6M", None),
            y_t_d=mtrx.get("YTD", None),
            one_y=mtrx.get("1Y", None),
            three_y=mtrx.get("3Y (ann.)", None),
            five_y=mtrx.get("5Y (ann.)", None),
            ten_y=mtrx.get("10Y (ann.)", None),
            all_time=mtrx.get("All-time (ann.)", None),
            best_day=mtrx.get("Best Day", None),
            worst_day=mtrx.get("Worst Day", None),
            best_month=mtrx.get("Best Month", None),
            best_year=mtrx.get("Best Year", None),
            worst_month=mtrx.get("Worst Month", None),
            worst_year=mtrx.get("Worst Year", None),
            recovery_factor=mtrx.get("Recovery Factor", None),
            ulcer_index=mtrx.get("Ulcer Index", None),
            serenity_index=mtrx.get("Serenity Index", None),
            avg_up_month=mtrx.get("Avg. Up Month", None),
            avg_down_month=mtrx.get("Avg. Down Month", None),
            win_days=mtrx.get("Win Days", None),
            win_month=mtrx.get("Win Month", None),
            win_quater=mtrx.get("Win Quarter", None),
            win_year=mtrx.get("Win Year", None),
            beta=mtrx.get("Beta", None),
            alpha=mtrx.get("Alpha", None),
            correlation=mtrx.get("Correlation", None),
            treynor_ratio=mtrx.get("Treynor Ratio", None),
            longest_dd_days=mtrx.get("Longest DD Days", None),
            avg_drawdown_days=mtrx.get("Avg. Drawdown Days", None),
        )
