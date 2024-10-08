from dataclasses import dataclass
from typing import Optional

@dataclass
class Metrics():
    risk_free_rate: Optional[str] = None
    time_in_market: Optional[str] = None
    cumulative_return: Optional[str] = None
    total_return: Optional[str] = None
    cagr: Optional[str] = None
    prob_sharpe_ratio: Optional[str] = None
    smart_sharp: Optional[str] = None
    sortino: Optional[str] = None
    smart_sortino: Optional[str] = None
    sortino_root_2: Optional[str] = None
    smart_sortino_root_2: Optional[str] = None
    omega: Optional[str] = None
    max_drawdown: Optional[str] = None
    longest_dd_days: Optional[str] = None
    volatility: Optional[str] = None
    r_2: Optional[str] = None
    imformation_ratio: Optional[str] = None
    calmer: Optional[str] = None
    skew: Optional[str] = None
    kurtosis: Optional[str] = None
    expected_daily: Optional[str] = None
    expected_monthly: Optional[str] = None
    expected_yearly: Optional[str] = None
    kelly_criterion: Optional[str] = None
    risk_of_ruin: Optional[str] = None
    daily_value_at_risk: Optional[str] = None
    expected_shortfall_cvar: Optional[str] = None
    max_consecutive_wins: Optional[str] = None
    max_consecutive_losses: Optional[str] = None
    gain_pain_ratio: Optional[str] = None
    gain_pain_1m: Optional[str] = None
    payoff_ratio: Optional[str] = None
    profit_factor: Optional[str] = None
    common_sense_ratio: Optional[str] = None
    cpc_index: Optional[str] = None
    tail_ratio: Optional[str] = None
    outer_win_ratio: Optional[str] = None
    outer_loss_ratio: Optional[str] = None
    mtd: Optional[str] = None
    three_m: Optional[str] = None
    six_m: Optional[str] = None
    y_t_d: Optional[str] = None
    one_y: Optional[str] = None
    three_y: Optional[str] = None
    five_y: Optional[str] = None
    ten_y: Optional[str] = None
    all_time: Optional[str] = None
    best_day: Optional[str] = None
    worst_day: Optional[str] = None
    best_month: Optional[str] = None
    best_year: Optional[str] = None
    worst_month: Optional[str] = None
    worst_year: Optional[str] = None
    recovery_factor: Optional[str] = None
    ulcer_index: Optional[str] = None
    serenity_index: Optional[str] = None
    avg_up_month: Optional[str] = None
    avg_down_month: Optional[str] = None
    win_days: Optional[str] = None
    win_month: Optional[str] = None
    win_quater: Optional[str] = None
    win_year: Optional[str] = None
    beta: Optional[str] = None
    alpha: Optional[str] = None
    correlation: Optional[str] = None
    treynor_ratio: Optional[str] = None
    longest_dd_days: Optional[str] = None
    avg_drawdown_days: Optional[str] = None
