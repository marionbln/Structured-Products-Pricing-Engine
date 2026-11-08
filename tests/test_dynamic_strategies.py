"""
Tests for rule-based dynamic strategies. Run with: pytest tests/test_dynamic_strategies.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import pytest

from src.dynamic_strategies import (weights_buy_and_hold, weights_vol_target, weights_trend,
                                    run_strategy, max_drawdown, performance_stats, stats_table)


def _prices(values):
    idx = pd.date_range("2000-01-01", periods=len(values), freq="MS")
    return pd.Series(values, index=idx, dtype=float)


def test_buy_and_hold_tracks_index_without_costs():
    p = _prices([100, 110, 99, 120, 130])
    res = run_strategy(p, weights_buy_and_hold(p), "B&H", cash_rate=0.0, cost_bps=0.0)
    assert res.nav.iloc[-1] == pytest.approx(130.0)


def test_all_cash_earns_cash_rate():
    p = _prices(np.linspace(100, 50, 13))
    res = run_strategy(p, pd.Series(0.0, index=p.index), "cash", cash_rate=0.02, cost_bps=0.0)
    assert res.nav.iloc[-1] == pytest.approx(102.0)


def test_no_look_ahead():
    # +50 % in month 2; a weight set only at month 2 must not capture it
    p = _prices([100, 100, 150, 150])
    w = pd.Series([0.0, 0.0, 1.0, 0.0], index=p.index)
    res = run_strategy(p, w, "x", cost_bps=0.0)
    assert res.nav.iloc[2] == pytest.approx(100.0)


def test_costs_reduce_nav():
    rng = np.random.default_rng(0)
    p = _prices(100 * np.cumprod(1 + rng.normal(0.005, 0.04, 120)))
    w = weights_trend(p, 10)
    free = run_strategy(p, w, "free", cost_bps=0.0).nav.iloc[-1]
    paid = run_strategy(p, w, "paid", cost_bps=20.0).nav.iloc[-1]
    assert paid < free


def test_vol_target_scales_exposure():
    rng = np.random.default_rng(1)
    p = _prices(100 * np.cumprod(1 + rng.normal(0.0, 0.20 / np.sqrt(12), 600)))
    w = weights_vol_target(p, target_vol=0.10, window=12, max_leverage=3.0)
    assert w.iloc[24:].median() == pytest.approx(0.5, abs=0.1)
    capped = weights_vol_target(p, target_vol=1.0, max_leverage=1.5)
    assert capped.max() <= 1.5


def test_trend_rule_and_warm_up():
    p = _prices(list(range(100, 120)) + list(range(119, 99, -1)))
    w = weights_trend(p, 5)
    assert (w.iloc[:4] == 0).all()           # warm-up: no signal yet
    assert w.iloc[10] == 1.0                  # uptrend: invested
    assert w.iloc[-1] == 0.0                  # downtrend: out


def test_max_drawdown_and_stats():
    nav = pd.Series([100, 120, 90, 130.0])
    assert max_drawdown(nav) == pytest.approx(-0.25)
    p = _prices(100 * np.cumprod(np.full(25, 1.01)))
    res = run_strategy(p, weights_buy_and_hold(p), "B&H", cost_bps=0.0)
    s = performance_stats(res)
    assert s["CAGR"] == pytest.approx(1.01 ** 12 - 1, rel=1e-9)
    assert s["Max drawdown"] == 0.0
    assert list(stats_table([res]).index) == ["B&H"]
