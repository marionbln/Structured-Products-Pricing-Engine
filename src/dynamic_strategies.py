"""
dynamic_strategies.py
=====================
Rule-based ("dynamic") strategies on an index, as wrapped in Delta One
certificates / AMCs: an exposure rule decides each month how much of the
portfolio is invested in the index, the rest earns the cash rate.

Implemented rules
-----------------
- buy & hold                (benchmark, 100 % invested)
- volatility target         w_t = min(cap, target_vol / realised_vol_t)
- trend following (SMA)     w_t = 1 if price_t > SMA_n(price)_t else 0

No look-ahead: the weight decided at the end of month t is applied to the
return of month t+1. Monthly rebalancing with a cost in basis points on the
traded notional (turnover after market drift).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

MONTHS = 12


# ---------------------------------------------------------------------------
# Exposure rules (return a weight series aligned on the price index)
# ---------------------------------------------------------------------------

def weights_buy_and_hold(prices: pd.Series) -> pd.Series:
    return pd.Series(1.0, index=prices.index, name="weight")


def weights_vol_target(prices: pd.Series, target_vol: float = 0.10, window: int = 12,
                       max_leverage: float = 1.5) -> pd.Series:
    """Exposure scaled so that the realised volatility hits `target_vol`."""
    rets = prices.pct_change()
    realised = rets.rolling(window).std() * np.sqrt(MONTHS)
    w = (target_vol / realised).clip(upper=max_leverage)
    return w.fillna(0.0).rename("weight")


def weights_trend(prices: pd.Series, window: int = 10) -> pd.Series:
    """Invested when the index is above its `window`-month moving average."""
    sma = prices.rolling(window).mean()
    w = (prices > sma).astype(float)
    w[sma.isna()] = 0.0
    return w.rename("weight")


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

@dataclass
class StrategyResult:
    name: str
    nav: pd.Series        # net asset value, starts at 100
    weights: pd.Series    # exposure decided at each date
    returns: pd.Series    # monthly net returns
    turnover: pd.Series   # traded notional / NAV at each rebalancing


def run_strategy(prices: pd.Series, weights: pd.Series, name: str,
                 cash_rate: float = 0.0, cost_bps: float = 5.0) -> StrategyResult:
    """Monthly rebalanced backtest of an exposure rule.

    cash_rate : annual rate earned on the non-invested part (and paid on
                leverage above 100 %).
    cost_bps  : transaction cost on traded notional, in basis points.
    """
    prices = prices.dropna().astype(float)
    weights = weights.reindex(prices.index).fillna(0.0)
    asset_ret = prices.pct_change().fillna(0.0)
    cash_m = (1 + cash_rate) ** (1 / MONTHS) - 1

    nav = [100.0]
    rets = [0.0]
    turnover = [abs(weights.iloc[0])]
    for t in range(1, len(prices)):
        w_prev = weights.iloc[t - 1]                      # decided at t-1, no look-ahead
        r_gross = w_prev * asset_ret.iloc[t] + (1 - w_prev) * cash_m
        # weight after market drift, before rebalancing to the new target
        drifted = w_prev * (1 + asset_ret.iloc[t]) / (1 + r_gross) if (1 + r_gross) != 0 else 0.0
        trade = abs(weights.iloc[t] - drifted)
        r_net = r_gross - trade * cost_bps / 1e4
        nav.append(nav[-1] * (1 + r_net))
        rets.append(r_net)
        turnover.append(trade)

    idx = prices.index
    return StrategyResult(name=name,
                          nav=pd.Series(nav, index=idx, name=name),
                          weights=weights.rename(name),
                          returns=pd.Series(rets, index=idx, name=name),
                          turnover=pd.Series(turnover, index=idx, name=name))


# ---------------------------------------------------------------------------
# Performance statistics
# ---------------------------------------------------------------------------

def max_drawdown(nav: pd.Series) -> float:
    return float((nav / nav.cummax() - 1).min())


def performance_stats(res: StrategyResult, cash_rate: float = 0.0) -> dict:
    r = res.returns.iloc[1:]
    years = len(r) / MONTHS
    cagr = (res.nav.iloc[-1] / res.nav.iloc[0]) ** (1 / years) - 1
    vol = r.std() * np.sqrt(MONTHS)
    excess = r - ((1 + cash_rate) ** (1 / MONTHS) - 1)
    sharpe = excess.mean() / r.std() * np.sqrt(MONTHS) if r.std() > 0 else np.nan
    mdd = max_drawdown(res.nav)
    return {
        "Strategy": res.name,
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Max drawdown": mdd,
        "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "Worst month": r.min(),
        "Avg exposure": res.weights.mean(),
        "Turnover / year": res.turnover.iloc[1:].sum() / years,
    }


def stats_table(results: list[StrategyResult], cash_rate: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame([performance_stats(r, cash_rate) for r in results]).set_index("Strategy")


def calendar_returns(res: StrategyResult) -> pd.Series:
    """Yearly returns, as shown in a client factsheet."""
    yearly = res.nav.groupby(res.nav.index.year).last()
    first = res.nav.iloc[0]
    return yearly.pct_change().fillna(yearly.iloc[0] / first - 1).rename(res.name)


# ---------------------------------------------------------------------------
# Client report (PNG + Markdown), generated automatically
# ---------------------------------------------------------------------------

def _md_table(df: pd.DataFrame) -> str:
    """Markdown table without extra dependency (no tabulate needed)."""
    cols = [str(df.index.name or "")] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row]) + " |")
    return "\n".join(lines)


def generate_report(results: list[StrategyResult], out_dir: str, cash_rate: float = 0.0,
                    title: str = "Dynamic strategies on the S&P 500") -> dict:
    """Write a factsheet-style report: chart (NAV, drawdown, exposure) + tables."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    stats = stats_table(results, cash_rate)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
    for r in results:
        axes[0].plot(r.nav.index, r.nav, label=r.name, lw=1.4)
        axes[1].plot(r.nav.index, (r.nav / r.nav.cummax() - 1) * 100, lw=1)
        axes[2].plot(r.weights.index, r.weights * 100, lw=1)
    axes[0].set_yscale("log"); axes[0].set_ylabel("NAV (log, start = 100)")
    axes[0].legend(frameon=False); axes[0].set_title(title)
    axes[1].set_ylabel("Drawdown (%)"); axes[2].set_ylabel("Exposure (%)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    chart = os.path.join(out_dir, "factsheet.png")
    fig.savefig(chart, dpi=130); plt.close(fig)

    fmt = stats.copy()
    for c in ["CAGR", "Volatility", "Max drawdown", "Worst month", "Avg exposure", "Turnover / year"]:
        fmt[c] = fmt[c].map(lambda x: f"{x:.1%}")
    for c in ["Sharpe", "Calmar"]:
        fmt[c] = fmt[c].map(lambda x: f"{x:.2f}")
    cal = pd.concat([calendar_returns(r) for r in results], axis=1).tail(10)

    md = [f"# {title}", "",
          f"Period: {results[0].nav.index[0]:%b %Y} - {results[0].nav.index[-1]:%b %Y}, "
          f"monthly rebalancing, cash rate {cash_rate:.1%}.", "",
          "![factsheet](factsheet.png)", "", "## Key figures", "",
          _md_table(fmt), "", "## Calendar returns (last 10 years)", "",
          _md_table(cal.map(lambda x: f"{x:.1%}").rename_axis("Year"))]
    report = os.path.join(out_dir, "report.md")
    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    return {"chart": chart, "report": report, "stats": stats}
