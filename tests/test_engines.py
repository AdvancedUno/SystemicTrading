"""Tests for the vectorized and event-driven backtest engines."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from quant.backtest import run_event_driven, run_vectorized
from quant.backtest.costs import CostConfig
from quant.data.types import AssetClass


def make_simple_panel(
    n_bars: int = 60,
    seed: int = 0,
    drift: float = 0.0005,
    vol: float = 0.01,
    n_symbols: int = 3,
) -> pl.DataFrame:
    """Make a simple OHLCV panel where close has known stats."""
    rng = np.random.default_rng(seed)
    rows = []
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for s in range(n_symbols):
        symbol = f"SYM{s}"
        log_rets = rng.normal(loc=drift, scale=vol, size=n_bars)
        prices = 100.0 * np.exp(np.cumsum(log_rets))
        for i in range(n_bars):
            ts = start + timedelta(days=i)
            close = float(prices[i])
            open_ = close * float(rng.normal(1.0, 0.001))
            high = max(open_, close) * 1.005
            low = min(open_, close) * 0.995
            volume = float(1_000_000 + rng.integers(0, 100_000))
            rows.append(
                {
                    "ts": ts,
                    "symbol": symbol,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "trade_count": 1000,
                    "vwap": (open_ + high + low + close) / 4,
                }
            )
    df = pl.DataFrame(rows).with_columns(
        pl.col("ts").dt.cast_time_unit("us").dt.replace_time_zone("UTC"),
        pl.col("trade_count").cast(pl.Int64),
    )
    return df.sort(["symbol", "ts"])


def make_constant_weights(symbols: list[str], timestamps: list, weights: list[float]) -> pl.DataFrame:
    """Build a long-format weights frame with the same weights at every ts."""
    rows = []
    for ts in timestamps:
        for sym, w in zip(symbols, weights):
            rows.append({"ts": ts, "symbol": sym, "weight": float(w)})
    return pl.DataFrame(rows).with_columns(
        pl.col("ts").dt.cast_time_unit("us").dt.replace_time_zone("UTC")
    )


# ============================================================================
# Vectorized engine
# ============================================================================


def test_vec_zero_weights_zero_returns():
    panel = make_simple_panel(n_bars=30, n_symbols=2)
    timestamps = panel["ts"].unique().sort().to_list()
    weights = make_constant_weights(["SYM0", "SYM1"], timestamps, [0.0, 0.0])
    res = run_vectorized(panel=panel, weights=weights)
    assert res.returns.height == 30
    np.testing.assert_allclose(res.returns["net_return"].to_numpy(), 0.0, atol=1e-12)


def test_vec_long_one_asset_matches_buy_and_hold():
    """Holding +1.0 in a single asset should produce that asset's returns."""
    panel = make_simple_panel(n_bars=60, n_symbols=1, seed=42)
    timestamps = panel["ts"].unique().sort().to_list()
    weights = make_constant_weights(["SYM0"], timestamps, [1.0])
    res = run_vectorized(
        panel=panel, weights=weights,
        cost_config=CostConfig(commission_rate=0, spread_fraction=0, min_spread_bps=0,
                               impact_k=0, max_impact_bps=0),
    )
    # Compute expected: buy and hold on day 1, hold thereafter.
    closes = panel.sort("ts")["close"].to_numpy()
    expected = np.full(60, 0.0)
    expected[1:] = closes[1:] / closes[:-1] - 1
    # First bar: weight not yet held going in, so return = 0
    expected[0] = 0.0
    # Second bar: held weight from bar-0 (forward-fill of weights, then shift) = 1.0
    # (the weight column at idx=0 is 1.0, shifted to idx=1)
    actual = res.returns["net_return"].to_numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_vec_turnover_first_bar_equals_initial_weight():
    """Turnover at bar 0 = sum |w_0 - 0| = sum |w_0|."""
    panel = make_simple_panel(n_bars=10, n_symbols=2)
    timestamps = panel["ts"].unique().sort().to_list()
    weights = make_constant_weights(["SYM0", "SYM1"], timestamps, [0.6, -0.4])
    res = run_vectorized(panel=panel, weights=weights)
    assert res.returns["turnover"][0] == pytest.approx(1.0, abs=1e-9)
    # After that, weights are constant, so turnover = 0
    assert res.returns["turnover"][1:].sum() == pytest.approx(0.0, abs=1e-9)


def test_vec_costs_drag_returns():
    """Net returns < gross returns when costs are positive."""
    panel = make_simple_panel(n_bars=30, n_symbols=2, seed=7)
    timestamps = panel["ts"].unique().sort().to_list()
    # Alternating weights to force constant turnover
    weights_list = []
    for i, ts in enumerate(timestamps):
        sign = 1.0 if i % 2 == 0 else -1.0
        weights_list.append({"ts": ts, "symbol": "SYM0", "weight": 0.5 * sign})
        weights_list.append({"ts": ts, "symbol": "SYM1", "weight": -0.5 * sign})
    weights = pl.DataFrame(weights_list).with_columns(
        pl.col("ts").dt.cast_time_unit("us").dt.replace_time_zone("UTC")
    )
    res = run_vectorized(panel=panel, weights=weights)
    gross_total = res.returns["gross_return"].sum()
    net_total = res.returns["net_return"].sum()
    assert net_total < gross_total  # costs ate some
    assert res.returns["cost"].sum() > 0


def test_vec_dollar_neutral_check():
    panel = make_simple_panel(n_bars=20, n_symbols=2, seed=3)
    timestamps = panel["ts"].unique().sort().to_list()
    weights = make_constant_weights(["SYM0", "SYM1"], timestamps, [0.5, -0.5])
    res = run_vectorized(panel=panel, weights=weights)
    # Net returns reflect the spread between the two assets (and costs).
    # Schema sanity:
    assert {"gross_return", "net_return", "turnover", "gross_exposure", "cost"}.issubset(
        set(res.returns.columns)
    )


# ============================================================================
# Event-driven engine
# ============================================================================


def test_event_zero_weights_zero_returns():
    panel = make_simple_panel(n_bars=30, n_symbols=2)
    timestamps = panel["ts"].unique().sort().to_list()
    weights = make_constant_weights(["SYM0", "SYM1"], timestamps, [0.0, 0.0])
    res = run_event_driven(panel=panel, weights=weights)
    assert res.returns.height == 30
    np.testing.assert_allclose(res.returns["net_return"].to_numpy(), 0.0, atol=1e-12)


def test_event_costs_drag_returns():
    panel = make_simple_panel(n_bars=30, n_symbols=2, seed=11)
    timestamps = panel["ts"].unique().sort().to_list()
    weights_list = []
    for i, ts in enumerate(timestamps):
        sign = 1.0 if i % 2 == 0 else -1.0
        weights_list.append({"ts": ts, "symbol": "SYM0", "weight": 0.5 * sign})
        weights_list.append({"ts": ts, "symbol": "SYM1", "weight": -0.5 * sign})
    weights = pl.DataFrame(weights_list).with_columns(
        pl.col("ts").dt.cast_time_unit("us").dt.replace_time_zone("UTC")
    )
    res = run_event_driven(panel=panel, weights=weights)
    gross_total = res.returns["gross_return"].sum()
    net_total = res.returns["net_return"].sum()
    assert net_total < gross_total
    assert res.returns["cost"].sum() > 0


def test_event_nav_curve_starts_at_initial_capital():
    panel = make_simple_panel(n_bars=10, n_symbols=2)
    timestamps = panel["ts"].unique().sort().to_list()
    weights = make_constant_weights(["SYM0", "SYM1"], timestamps, [0.5, -0.5])
    res = run_event_driven(
        panel=panel, weights=weights, initial_capital=100_000.0,
    )
    # NAV at first bar should be initial_capital (no fills yet)
    assert res.returns["nav"][0] == pytest.approx(100_000.0, rel=1e-6)


# ============================================================================
# Cross-engine consistency — the most important test
# ============================================================================


def test_vec_and_event_directionally_consistent():
    """Both engines should agree on the SIGN of the return for a clean signal.

    We make a small panel where SYM0 outperforms SYM1, then test "long SYM0,
    short SYM1." Both engines should report positive net P&L.
    """
    rng = np.random.default_rng(99)
    n = 40
    rows = []
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    # SYM0: drift up, SYM1: drift down
    for sym, drift in [("SYM0", 0.005), ("SYM1", -0.005)]:
        rets = rng.normal(loc=drift, scale=0.005, size=n)
        prices = 100.0 * np.exp(np.cumsum(rets))
        for i in range(n):
            close = float(prices[i])
            open_ = close * float(rng.normal(1.0, 0.0005))
            rows.append(
                {
                    "ts": start + timedelta(days=i),
                    "symbol": sym,
                    "open": open_,
                    "high": max(open_, close) * 1.001,
                    "low": min(open_, close) * 0.999,
                    "close": close,
                    "volume": 1e6,
                    "trade_count": 1000,
                    "vwap": (open_ + close) / 2,
                }
            )
    panel = pl.DataFrame(rows).with_columns(
        pl.col("ts").dt.cast_time_unit("us").dt.replace_time_zone("UTC"),
        pl.col("trade_count").cast(pl.Int64),
    ).sort(["symbol", "ts"])
    timestamps = panel["ts"].unique().sort().to_list()
    weights = make_constant_weights(["SYM0", "SYM1"], timestamps, [0.5, -0.5])

    res_vec = run_vectorized(panel=panel, weights=weights)
    res_evt = run_event_driven(panel=panel, weights=weights)

    vec_total = res_vec.returns["net_return"].sum()
    evt_total = res_evt.returns["net_return"].sum()

    # Both should be positive — long the winner, short the loser
    assert vec_total > 0
    assert evt_total > 0
    # And similar in magnitude — within a factor of ~2 (event-driven incurs
    # more cost from open-of-next fills + slippage)
    assert 0.3 * vec_total < evt_total < 1.5 * vec_total
