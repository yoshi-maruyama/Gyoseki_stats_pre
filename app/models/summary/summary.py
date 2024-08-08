import pandas as pd
import numpy as np
import yfinance as yf
from dateutil.relativedelta import relativedelta
from datetime import datetime as dt
from math import sqrt as sqrt, ceil as ceil
from tabulate import tabulate as _tabulate

import app.models.summary.stats as _stats
import app.models.summary.utils as _utils


def _get_trading_periods(periods_per_year=252):
    half_year = ceil(periods_per_year / 2)
    return periods_per_year, half_year


def _match_dates(returns, benchmark):
    if isinstance(returns, pd.DataFrame):
        loc = max(returns[returns.columns[0]].ne(0).idxmax(), benchmark.ne(0).idxmax())
    else:
        loc = max(returns.ne(0).idxmax(), benchmark.ne(0).idxmax())
    returns = returns.loc[loc:]
    benchmark = benchmark.loc[loc:]

    return returns, benchmark

def _calc_dd(df, display=True, as_pct=False):
    dd = _stats.to_drawdown_series(df)
    dd_info = _stats.drawdown_details(dd)

    if dd_info.empty:
        return pd.DataFrame()

    if "returns" in dd_info:
        ret_dd = dd_info["returns"]
    # to match multiple columns like returns_1, returns_2, ...
    elif (
        any(dd_info.columns.get_level_values(0).str.contains("returns"))
        and dd_info.columns.get_level_values(0).nunique() > 1
    ):
        ret_dd = dd_info.loc[
            :, dd_info.columns.get_level_values(0).str.contains("returns")
        ]
    else:
        ret_dd = dd_info

    if (
        any(ret_dd.columns.get_level_values(0).str.contains("returns"))
        and ret_dd.columns.get_level_values(0).nunique() > 1
    ):
        dd_stats = {
            col: {
                "Max Drawdown %": ret_dd[col]
                .sort_values(by="max drawdown", ascending=True)["max drawdown"]
                .values[0]
                / 100,
                "Longest DD Days": str(
                    np.round(
                        ret_dd[col]
                        .sort_values(by="days", ascending=False)["days"]
                        .values[0]
                    )
                ),
                "Avg. Drawdown %": ret_dd[col]["max drawdown"].mean() / 100,
                "Avg. Drawdown Days": str(np.round(ret_dd[col]["days"].mean())),
            }
            for col in ret_dd.columns.get_level_values(0)
        }
    else:
        dd_stats = {
            "returns": {
                "Max Drawdown %": ret_dd.sort_values(by="max drawdown", ascending=True)[
                    "max drawdown"
                ].values[0]
                / 100,
                "Longest DD Days": str(
                    np.round(
                        ret_dd.sort_values(by="days", ascending=False)["days"].values[0]
                    )
                ),
                "Avg. Drawdown %": ret_dd["max drawdown"].mean() / 100,
                "Avg. Drawdown Days": str(np.round(ret_dd["days"].mean())),
            }
        }
    if "benchmark" in df and (dd_info.columns, pd.MultiIndex):
        bench_dd = dd_info["benchmark"].sort_values(by="max drawdown")
        dd_stats["benchmark"] = {
            "Max Drawdown %": bench_dd.sort_values(by="max drawdown", ascending=True)[
                "max drawdown"
            ].values[0]
            / 100,
            "Longest DD Days": str(
                np.round(
                    bench_dd.sort_values(by="days", ascending=False)["days"].values[0]
                )
            ),
            "Avg. Drawdown %": bench_dd["max drawdown"].mean() / 100,
            "Avg. Drawdown Days": str(np.round(bench_dd["days"].mean())),
        }

    # pct multiplier
    pct = 100 if display or as_pct else 1

    dd_stats = pd.DataFrame(dd_stats).T
    dd_stats["Max Drawdown %"] = dd_stats["Max Drawdown %"].astype(float) * pct
    dd_stats["Avg. Drawdown %"] = dd_stats["Avg. Drawdown %"].astype(float) * pct

    return dd_stats.T

def metrics_summary(
    returns,
    benchmark=None,
    rf=0.0,
    display=True,
    mode="basic",
    sep=False,
    compounded=True,
    periods_per_year=252,
    prepare_returns=True,
    match_dates=True,
    **kwargs,
):

    if match_dates:
        returns = returns.dropna()

    returns.index = returns.index.tz_localize(None)
    win_year, _ = _get_trading_periods(periods_per_year)

    benchmark_colname = kwargs.get("benchmark_title", "Benchmark")
    strategy_colname = kwargs.get("strategy_title", "Strategy")

    if benchmark is not None:
        if isinstance(benchmark, str):
            benchmark_colname = f"Benchmark ({benchmark.upper()})"
        elif isinstance(benchmark, pd.DataFrame) and len(benchmark.columns) > 1:
            raise ValueError(
                "`benchmark` must be a pandas Series, "
                "but a multi-column DataFrame was passed"
            )

    if isinstance(returns, pd.DataFrame):
        if len(returns.columns) > 1:
            blank = [""] * len(returns.columns)
            if isinstance(strategy_colname, str):
                strategy_colname = list(returns.columns)
    else:
        blank = [""]

    if prepare_returns:
        df = _utils._prepare_returns(returns)

    if isinstance(returns, pd.Series):
        df = pd.DataFrame({"returns": returns})
    elif isinstance(returns, pd.DataFrame):
        df = pd.DataFrame(
            {
                "returns_" + str(i + 1): returns[strategy_col]
                for i, strategy_col in enumerate(returns.columns)
            }
        )

    if benchmark is not None:
        # benchmark = _utils._prepare_benchmark(benchmark, returns.index, rf)
        benchmark = _utils._prepare_benchmark(benchmark, returns)
        benchmark.index = benchmark.index.tz_localize(None)
        if match_dates is True:
            returns, benchmark = _match_dates(returns, benchmark)
        df["benchmark"] = benchmark
        if isinstance(returns, pd.Series):
            blank = ["", ""]
            df["returns"] = returns
        elif isinstance(returns, pd.DataFrame):
            blank = [""] * len(returns.columns) + [""]
            for i, strategy_col in enumerate(returns.columns):
                df["returns_" + str(i + 1)] = returns[strategy_col]

    if isinstance(returns, pd.Series):
        s_start = {"returns": df["returns"].index.strftime("%Y-%m-%d")[0]}
        s_end = {"returns": df["returns"].index.strftime("%Y-%m-%d")[-1]}
        s_rf = {"returns": rf}
    elif isinstance(returns, pd.DataFrame):
        df_strategy_columns = [col for col in df.columns if col != "benchmark"]
        s_start = {
            strategy_col: df[strategy_col].dropna().index.strftime("%Y-%m-%d")[0]
            for strategy_col in df_strategy_columns
        }
        s_end = {
            strategy_col: df[strategy_col].dropna().index.strftime("%Y-%m-%d")[-1]
            for strategy_col in df_strategy_columns
        }
        s_rf = {strategy_col: rf for strategy_col in df_strategy_columns}

    if "benchmark" in df:
        s_start["benchmark"] = df["benchmark"].index.strftime("%Y-%m-%d")[0]
        s_end["benchmark"] = df["benchmark"].index.strftime("%Y-%m-%d")[-1]
        s_rf["benchmark"] = rf

    df = df.fillna(0)

    # pct multiplier
    pct = 100 if display or "internal" in kwargs else 1
    if kwargs.get("as_pct", False):
        pct = 100

    # return df
    dd = _calc_dd(
        df,
        display=(display or "internal" in kwargs),
        as_pct=kwargs.get("as_pct", False),
    )

    metrics = pd.DataFrame()
    metrics["Start Period"] = pd.Series(s_start)
    metrics["End Period"] = pd.Series(s_end)
    if mode.lower() == "full":
        metrics["Risk-Free Rate %"] = pd.Series(s_rf) * 100
        metrics["Time in Market %"] = _stats.exposure(df, prepare_returns=False) * pct

    if compounded:
        metrics["Cumulative Return %"] = (_stats.comp(df) * pct).map("{:,.2f}".format)
    else:
        metrics["Total Return %"] = (df.sum() * pct).map("{:,.2f}".format)

    metrics["CAGR﹪%"] = _stats.cagr(df, rf, compounded) * pct

    metrics["Sharpe"] = _stats.sharpe(df, rf, win_year, True)
    metrics["Prob. Sharpe Ratio %"] = (
        _stats.probabilistic_sharpe_ratio(df, rf, win_year, False) * pct
    )
    if mode.lower() == "full":
        metrics["Smart Sharpe"] = _stats.smart_sharpe(df, rf, win_year, True)
        # metrics['Prob. Smart Sharpe Ratio %'] = _stats.probabilistic_sharpe_ratio(df, rf, win_year, False, True) * pct

    metrics["Sortino"] = _stats.sortino(df, rf, win_year, True)
    if mode.lower() == "full":
        # metrics['Prob. Sortino Ratio %'] = _stats.probabilistic_sortino_ratio(df, rf, win_year, False) * pct
        metrics["Smart Sortino"] = _stats.smart_sortino(df, rf, win_year, True)
        # metrics['Prob. Smart Sortino Ratio %'] = _stats.probabilistic_sortino_ratio(df, rf, win_year, False, True) * pct

    if mode.lower() == "full":
        metrics["Sortino/√2"] = metrics["Sortino"] / sqrt(2)
        # metrics['Prob. Sortino/√2 Ratio %'] = _stats.probabilistic_adjusted_sortino_ratio(df, rf, win_year, False) * pct
        metrics["Smart Sortino/√2"] = metrics["Smart Sortino"] / sqrt(2)
        # metrics['Prob. Smart Sortino/√2 Ratio %'] = _stats.probabilistic_adjusted_sortino_ratio(df, rf, win_year, False, True) * pct
        metrics["Omega"] = _stats.omega(df, rf, 0.0, win_year)

    metrics["Max Drawdown %"] = blank
    metrics["Longest DD Days"] = blank

    # if mode.lower() == "full":
    if isinstance(returns, pd.Series):
        ret_vol = (
            _stats.volatility(df["returns"], win_year, True, prepare_returns=False)
            * pct
        )
    elif isinstance(returns, pd.DataFrame):
        ret_vol = [
            _stats.volatility(
                df[strategy_col], win_year, True, prepare_returns=False
            )
            * pct
            for strategy_col in df_strategy_columns
        ]
    if "benchmark" in df:
        bench_vol = (
            _stats.volatility(
                df["benchmark"], win_year, True, prepare_returns=False
            )
            * pct
        )

        vol_ = [ret_vol, bench_vol]
        if isinstance(ret_vol, list):
            metrics["Volatility (ann.) %"] = list(pd.core.common.flatten(vol_))
        else:
            metrics["Volatility (ann.) %"] = vol_

        if isinstance(returns, pd.Series):
            metrics["R^2"] = _stats.r_squared(
                df["returns"], df["benchmark"], prepare_returns=False
            )
            metrics["Information Ratio"] = _stats.information_ratio(
                df["returns"], df["benchmark"], prepare_returns=False
            )
        elif isinstance(returns, pd.DataFrame):
            metrics["R^2"] = (
                [
                    _stats.r_squared(
                        df[strategy_col], df["benchmark"], prepare_returns=False
                    ).round(2)
                    for strategy_col in df_strategy_columns
                ]
            ) + ["-"]
            metrics["Information Ratio"] = (
                [
                    _stats.information_ratio(
                        df[strategy_col], df["benchmark"], prepare_returns=False
                    ).round(2)
                    for strategy_col in df_strategy_columns
                ]
            ) + ["-"]
    else:
        if isinstance(returns, pd.Series):
            metrics["Volatility (ann.) %"] = [ret_vol]
        elif isinstance(returns, pd.DataFrame):
            metrics["Volatility (ann.) %"] = ret_vol

    metrics["Calmar"] = _stats.calmar(df, prepare_returns=False)
    metrics["Skew"] = _stats.skew(df, prepare_returns=False)
    metrics["Kurtosis"] = _stats.kurtosis(df, prepare_returns=False)

    # metrics["~~~~~~~~~~"] = blank

    if mode.lower() == "full":
        metrics["Expected Daily %%"] = (
            _stats.expected_return(df, compounded=compounded, prepare_returns=False) * pct
        )
        metrics["Expected Monthly %%"] = (
            _stats.expected_return(df, compounded=compounded, aggregate="ME", prepare_returns=False) * pct
        )
        metrics["Expected Yearly %%"] = (
            _stats.expected_return(df, compounded=compounded, aggregate="YE", prepare_returns=False) * pct
        )
        metrics["Kelly Criterion %"] = (
            _stats.kelly_criterion(df, prepare_returns=False) * pct
        )
        metrics["Risk of Ruin %"] = _stats.risk_of_ruin(df, prepare_returns=False)

    metrics["Daily Value-at-Risk %"] = -abs(
        _stats.var(df, prepare_returns=False) * pct
    )
    metrics["Expected Shortfall (cVaR) %"] = -abs(
        _stats.cvar(df, prepare_returns=False) * pct
    )

    if mode.lower() == "full":
        metrics["Max Consecutive Wins *int"] = _stats.consecutive_wins(df)
        metrics["Max Consecutive Losses *int"] = _stats.consecutive_losses(df)

    metrics["Gain/Pain Ratio"] = _stats.gain_to_pain_ratio(df, rf)
    metrics["Gain/Pain (1M)"] = _stats.gain_to_pain_ratio(df, rf, "ME")

    metrics["Payoff Ratio"] = _stats.payoff_ratio(df, prepare_returns=False)
    metrics["Profit Factor"] = _stats.profit_factor(df, prepare_returns=False)
    if mode.lower() == "full":
        metrics["Common Sense Ratio"] = _stats.common_sense_ratio(df, prepare_returns=False)
        metrics["CPC Index"] = _stats.cpc_index(df, prepare_returns=False)
        metrics["Tail Ratio"] = _stats.tail_ratio(df, prepare_returns=False)
        metrics["Outlier Win Ratio"] = _stats.outlier_win_ratio(df, prepare_returns=False)
        metrics["Outlier Loss Ratio"] = _stats.outlier_loss_ratio(df, prepare_returns=False)

    comp_func = _stats.comp if compounded else np.sum

    today = df.index[-1]  # dt.today()
    metrics["MTD %"] = comp_func(df[df.index >= dt(today.year, today.month, 1)]) * pct

    d = today - relativedelta(months=3)
    metrics["3M %"] = comp_func(df[df.index >= d]) * pct

    d = today - relativedelta(months=6)
    metrics["6M %"] = comp_func(df[df.index >= d]) * pct

    metrics["YTD %"] = comp_func(df[df.index >= dt(today.year, 1, 1)]) * pct

    d = today - relativedelta(years=1)
    metrics["1Y %"] = comp_func(df[df.index >= d]) * pct

    d = today - relativedelta(months=35)
    metrics["3Y (ann.) %"] = _stats.cagr(df[df.index >= d], 0.0, compounded) * pct

    d = today - relativedelta(months=59)
    metrics["5Y (ann.) %"] = _stats.cagr(df[df.index >= d], 0.0, compounded) * pct

    d = today - relativedelta(years=10)
    metrics["10Y (ann.) %"] = _stats.cagr(df[df.index >= d], 0.0, compounded) * pct

    metrics["All-time (ann.) %"] = _stats.cagr(df, 0.0, compounded) * pct

    # best/worst
    if mode.lower() == "full":
        # metrics["~~~"] = blank
        metrics["Best Day %"] = _stats.best(df, compounded=compounded, prepare_returns=False) * pct
        metrics["Worst Day %"] = _stats.worst(df, prepare_returns=False) * pct
        metrics["Best Month %"] = (
            _stats.best(df, compounded=compounded, aggregate="M", prepare_returns=False) * pct
        )
        metrics["Worst Month %"] = (
            _stats.worst(df, aggregate="ME", prepare_returns=False) * pct
        )
        metrics["Best Year %"] = (
            _stats.best(df, compounded=compounded, aggregate="YE", prepare_returns=False) * pct
        )
        metrics["Worst Year %"] = (
            _stats.worst(df, compounded=compounded, aggregate="YE", prepare_returns=False) * pct
        )

    # dd

    for ix, row in dd.iterrows():
        metrics[ix] = row
    metrics["Recovery Factor"] = _stats.recovery_factor(df)
    metrics["Ulcer Index"] = _stats.ulcer_index(df)
    metrics["Serenity Index"] = _stats.serenity_index(df, rf)

    # win rate
    if mode.lower() == "full":

        metrics["Avg. Up Month %"] = (
            _stats.avg_win(df, compounded=compounded, aggregate="ME", prepare_returns=False) * pct
        )
        metrics["Avg. Down Month %"] = (
            _stats.avg_loss(df, compounded=compounded, aggregate="ME", prepare_returns=False) * pct
        )
        metrics["Win Days %%"] = _stats.win_rate(df, prepare_returns=False) * pct
        metrics["Win Month %%"] = (
            _stats.win_rate(df, compounded=compounded, aggregate="ME", prepare_returns=False) * pct
        )
        metrics["Win Quarter %%"] = (
            _stats.win_rate(df, compounded=compounded, aggregate="QE", prepare_returns=False) * pct
        )
        metrics["Win Year %%"] = (
            _stats.win_rate(df, compounded=compounded, aggregate="YE", prepare_returns=False) * pct
        )

        if "benchmark" in df:
            if isinstance(returns, pd.Series):
                greeks = _stats.greeks(
                    df["returns"], df["benchmark"], win_year, prepare_returns=False
                )
                metrics["Beta"] = [str(round(greeks["beta"], 2)), "-"]
                metrics["Alpha"] = [str(round(greeks["alpha"], 2)), "-"]
                metrics["Correlation"] = [
                    str(round(df["benchmark"].corr(df["returns"]) * pct, 2)) + "%",
                    "-",
                ]
                metrics["Treynor Ratio"] = [
                    str(
                        round(
                            _stats.treynor_ratio(
                                df["returns"], df["benchmark"], win_year, rf
                            )
                            * pct,
                            2,
                        )
                    )
                    + "%",
                    "-",
                ]
            elif isinstance(returns, pd.DataFrame):
                greeks = [
                    _stats.greeks(
                        df[strategy_col],
                        df["benchmark"],
                        win_year,
                        prepare_returns=False,
                    )
                    for strategy_col in df_strategy_columns
                ]
                metrics["Beta"] = [str(round(g["beta"], 2)) for g in greeks] + ["-"]
                metrics["Alpha"] = [str(round(g["alpha"], 2)) for g in greeks] + ["-"]
                metrics["Correlation"] = (
                    [
                        str(round(df["benchmark"].corr(df[strategy_col]) * pct, 2))
                        + "%"
                        for strategy_col in df_strategy_columns
                    ]
                ) + ["-"]
                metrics["Treynor Ratio"] = (
                    [
                        str(
                            round(
                                _stats.treynor_ratio(
                                    df[strategy_col], df["benchmark"], win_year, rf
                                )
                                * pct,
                                2,
                            )
                        )
                        + "%"
                        for strategy_col in df_strategy_columns
                    ]
                ) + ["-"]

    # prepare for display
    for col in metrics.columns:
        try:
            metrics[col] = metrics[col].astype(float).round(2)
            if display or "internal" in kwargs:
                metrics[col] = metrics[col].astype(str)
        except Exception:
            pass
        if (display or "internal" in kwargs) and "*int" in col:
            metrics[col] = metrics[col].str.replace(".0", "", regex=False)
            metrics.rename({col: col.replace("*int", "")}, axis=1, inplace=True)
        if (display or "internal" in kwargs) and "%" in col:
            metrics[col] = metrics[col] + "%"

    try:
        metrics["Longest DD Days"] = pd.to_numeric(metrics["Longest DD Days"]).astype(
            "int"
        )
        metrics["Avg. Drawdown Days"] = pd.to_numeric(
            metrics["Avg. Drawdown Days"]
        ).astype("int")

        if display or "internal" in kwargs:
            metrics["Longest DD Days"] = metrics["Longest DD Days"].astype(str)
            metrics["Avg. Drawdown Days"] = metrics["Avg. Drawdown Days"].astype(str)
    except Exception:
        metrics["Longest DD Days"] = "-"
        metrics["Avg. Drawdown Days"] = "-"
        if display or "internal" in kwargs:
            metrics["Longest DD Days"] = "-"
            metrics["Avg. Drawdown Days"] = "-"

    metrics.columns = [col if "~" not in col else "" for col in metrics.columns]
    metrics.columns = [col[:-1] if "%" in col else col for col in metrics.columns]
    metrics = metrics.T

    if "benchmark" in df:
        column_names = [strategy_colname, benchmark_colname]
        if isinstance(strategy_colname, list):
            metrics.columns = list(pd.core.common.flatten(column_names))
        else:
            metrics.columns = column_names
    else:
        if isinstance(strategy_colname, list):
            metrics.columns = strategy_colname
        else:
            metrics.columns = [strategy_colname]

    # cleanups
    metrics.replace([-0, "-0"], 0, inplace=True)
    metrics.replace(
        [
            np.nan,
            -np.nan,
            np.inf,
            -np.inf,
            "-nan%",
            "nan%",
            "-nan",
            "nan",
            "-inf%",
            "inf%",
            "-inf",
            "inf",
        ],
        "-",
        inplace=True,
    )

    # move benchmark to be the first column always if present
    #if "benchmark" in df:
    #    metrics = metrics[
    #        [benchmark_colname]
    #        + [col for col in metrics.columns if col != benchmark_colname]
    #    ]

    if display:
        print(_tabulate(metrics, headers="keys", tablefmt="simple"))
        return None

    if not sep:
        metrics = metrics[metrics.index != ""]

    # remove spaces from column names
    metrics = metrics.T
    metrics.columns = [
        c.replace(" %", "").replace(" *int", "").strip() for c in metrics.columns
    ]
    metrics = metrics.T

    return metrics

def get_nasdaq_sp500_data(start_date):
    """
    Reads the historical daily prices for Nasdaq and S&P500 into a pandas dataframe.

    Args:
    start_date (str): The start date for the historical data in 'YYYY-MM-DD' format.

    Returns:
    pandas.DataFrame: A dataframe containing the historical daily prices for Nasdaq and S&P500.
    """
    # Define the symbols for Nasdaq and S&P500
    nasdaq_symbol = "^NDX"  # Nasdaq Composite index
    sp500_symbol = "^GSPC"  # S&P 500 index

    # Download the historical data
    nasdaq_data = yf.download(nasdaq_symbol, start=start_date)
    sp500_data = yf.download(sp500_symbol, start=start_date)

    # Select only the 'Close' prices and rename the columns
    nasdaq_close = nasdaq_data['Close'].rename('Nasdaq 100')
    sp500_close = sp500_data['Close'].rename('S&P500')

    # Combine the data into a single dataframe
    combined_data = pd.concat([nasdaq_close, sp500_close], axis=1)

    return combined_data

def get_metrix(data):
    # Example usage:
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

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print("-" * 50)
    print(mtrx)
    print("-" * 50)
    return mtrx
