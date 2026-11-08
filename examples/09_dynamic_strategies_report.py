"""
09 - Dynamic strategies on the S&P 500: backtest + automatic client report.

Compares buy & hold, a 10 % volatility-target strategy and a 10-month trend
filter, rebalanced monthly with 5 bp of transaction costs. Writes a
factsheet (PNG) and a Markdown report in reports/dynamic_strategies/.

Data: monthly S&P 500 price index (dividends not included), 1990 onwards.
Cash rate: flat 2 % per year (simplifying assumption).
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.dynamic_strategies import (weights_buy_and_hold, weights_vol_target, weights_trend,
                                    run_strategy, generate_report)

CASH = 0.02
COST_BPS = 5.0

root = os.path.join(os.path.dirname(__file__), "..")
prices = pd.read_csv(os.path.join(root, "data", "sp500_monthly.csv"),
                     parse_dates=["Date"], index_col="Date")["SP500"]

# Signals are computed on the full history; all strategies start once every
# signal is available (12-month warm-up), so the comparison is like-for-like.
signals = {
    "Buy & hold": weights_buy_and_hold(prices),
    "Vol target 10%": weights_vol_target(prices, 0.10, 12, 1.5),
    "Trend 10m SMA": weights_trend(prices, 10),
}
start = prices.index[12]
results = [run_strategy(prices.loc[start:], w.loc[start:], name, CASH, COST_BPS)
           for name, w in signals.items()]

out = generate_report(results, os.path.join(root, "reports", "dynamic_strategies"), CASH)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")
print(out["stats"])
print("\nReport written to", out["report"])
