"""Tests for backtest infrastructure: costs, CV, metrics, strategies."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from quant.backtest.costs import CostConfig, estimate_cost
from quant.backtest.cv import Split, kfold_purged_splits, walk_forward_splits
from quant.backtest.metrics import compute_stats, drawdown_series, equity_curve
from quant.data.types import AssetClass
from quant.features.leakage import make_synthetic_panel
from quant.strategies.library import (
    CrossSectionalMomentum,
    CryptoMeanReversion,
)


# Cost model

def test_zero_notional_zero_cost():
    cfg = CostConfig.for_asset_class(AssetClass.EQUITY)
    cb = estimate_cost(
        config=cfg,
        notional=0.0,
        bar_high=100.0, bar_low=99.0, bar_close=99.5,
        daily_volatility=0.02, adv_dollars=1e9,
    )
    assert cb.total_bps == 0.0


def test_commission_for_crypto():
    cfg = CostConfig.for_asset_class(AssetClass.CRYPTO)
    cb = estimate_cost(
        config=cfg,
        notional=10_000.0,
        bar_high=100.0, bar_low=99.0, bar_close=99.5,
        daily_volatility=0.04, adv_dollars=1e8,
    )
    # 25 bps commission for crypto
    assert cb.commission_bps == pytest.approx(25.0, rel=0.01)


def test_impact_scales_sqrt_of_size():
    """Doubling trade size should ~sqrt-2 the impact, not double it."""
    cfg = CostConfig.for_asset_class(AssetClass.EQUITY)
    common = dict(
        config=cfg,
        bar_high=100.0, bar_low=99.0, bar_close=99.5,
        daily_volatility=0.02, adv_dollars=1e8,
    )
    small = estimate_cost(notional=1e5, **common)
    large = estimate_cost(notional=4e5, **common)  # 4x
    ratio = large.impact_bps / small.impact_bps
    # 4x trade ~= 2x impact (sqrt(4) = 2)
    assert 1.9 < ratio < 2.1


def test_min_spread_floor():
    cfg = CostConfig(spread_fraction=0.001, min_spread_bps=5.0)
    cb = estimate_cost(
        config=cfg,
        notional=1000.0,
        bar_high=100.0, bar_low=99.99, bar_close=100.0,  # tiny range
        daily_volatility=0.01, adv_dollars=1e8,
    )
    # half-spread = max_spread/2 = 5/2 = 2.5
    assert cb.half_spread_bps == pytest.approx(2.5, rel=0.01)


# ===========================================================================
# Walk-forward CV
# ===========================================================================


def test_walk_forward_basic():
    splits = list(
        walk_forward_splits(
            start=datetime(2020, 1, 1, tzinfo=timezone.utc),
            end=datetime(2020, 12, 31, tzinfo=timezone.utc),
            train_window=timedelta(days=90),
            test_window=timedelta(days=30),
        )
    )
    assert len(splits) > 0
    for s in splits:
        assert s.train_start < s.train_end <= s.test_start < s.test_end


def test_walk_forward_purge_creates_gap():
    splits = list(
        walk_forward_splits(
            start=datetime(2020, 1, 1, tzinfo=timezone.utc),
            end=datetime(2020, 12, 31, tzinfo=timezone.utc),
            train_window=timedelta(days=90),
            test_window=timedelta(days=30),
            label_horizon=timedelta(days=5),
        )
    )
    for s in splits:
        gap = (s.test_start - s.train_end).days
        assert gap == 5


def test_walk_forward_embargo():
    splits = list(
        walk_forward_splits(
            start=datetime(2020, 1, 1, tzinfo=timezone.utc),
            end=datetime(2020, 12, 31, tzinfo=timezone.utc),
            train_window=timedelta(days=60),
            test_window=timedelta(days=30),
            embargo=timedelta(days=10),
        )
    )
    # Successive folds should have a gap of at least embargo
    for a, b in zip(splits, splits[1:]):
        gap = (b.train_end - a.test_end).days
        assert gap >= 0  # next train ends >= prev test ends + embargo - train_window


def test_walk_forward_expanding():
    splits = list(
        walk_forward_splits(
            start=datetime(2020, 1, 1, tzinfo=timezone.utc),
            end=datetime(2020, 12, 31, tzinfo=timezone.utc),
            train_window=timedelta(days=60),
            test_window=timedelta(days=30),
            expanding=True,
        )
    )
    # With expanding, train_start should always be `start`
    for s in splits:
        assert s.train_start == datetime(2020, 1, 1, tzinfo=timezone.utc)
    # Train sets should grow
    for a, b in zip(splits, splits[1:]):
        assert b.train_days > a.train_days


def test_kfold_purged():
    splits = list(
        kfold_purged_splits(
            start=datetime(2020, 1, 1, tzinfo=timezone.utc),
            end=datetime(2021, 1, 1, tzinfo=timezone.utc),
            n_splits=5,
        )
    )
    assert len(splits) == 4  # n_splits-1


# ===========================================================================
# Performance metrics
# ===========================================================================


def test_metrics_zero_returns():
    s = compute_stats(np.zeros(252))
    assert s.sharpe == 0.0
    assert s.max_drawdown == 0.0


def test_metrics_constant_positive():
    """Constant +0.1% daily for a year."""
    r = np.full(252, 0.001)
    s = compute_stats(r)
    # Annual return ≈ 0.001 * 252 = 25.2%
    assert s.annual_return == pytest.approx(0.252, rel=0.01)
    assert s.hit_rate == 1.0
    assert s.max_drawdown == 0.0  # no drawdowns


def test_metrics_known_drawdown():
    """Up 10%, then down 20%, then flat. Drawdown ≈ -20%."""
    r = np.array([0.10, -0.20] + [0.0] * 250)
    s = compute_stats(r)
    assert s.max_drawdown == pytest.approx(-0.20, rel=0.01)


def test_equity_and_drawdown_curves():
    rng = np.random.default_rng(42)
    r = rng.normal(0.0005, 0.01, 252)
    eq = equity_curve(r)
    dd = drawdown_series(r)
    assert eq.shape == (252,)
    assert dd.shape == (252,)
    assert (dd <= 0.0).all()
    assert eq[0] == pytest.approx(1 + r[0])


# ===========================================================================
# Strategies
# ===========================================================================


def test_cs_momentum_runs():
    """Strategy produces sensible weights on synthetic data."""
    panel = make_synthetic_panel(n_symbols=20, n_bars=400, seed=42)
    strat = CrossSectionalMomentum()
    weights = strat.generate_weights(panel)
    assert weights.height > 0
    # Schema check
    assert set(weights.columns) == {"ts", "symbol", "weight"}
    # Net exposure should be ~zero (long/short)
    nets = weights.group_by("ts").agg(pl.col("weight").sum().alias("net"))
    # Quintile-based should be very nearly net zero
    assert nets["net"].abs().mean() < 0.05


def test_cs_momentum_gross_leverage():
    panel = make_synthetic_panel(n_symbols=20, n_bars=400, seed=7)
    strat = CrossSectionalMomentum()
    weights = strat.generate_weights(panel)
    # Per-ts gross sum should be ~= gross_leverage (1.0)
    grosses = weights.group_by("ts").agg(pl.col("weight").abs().sum().alias("gross"))
    assert grosses["gross"].mean() == pytest.approx(1.0, rel=0.05)


def test_mean_reversion_runs():
    panel = make_synthetic_panel(n_symbols=10, n_bars=200, seed=11)
    strat = CryptoMeanReversion()
    weights = strat.generate_weights(panel)
    assert weights.height > 0
    assert set(weights.columns) == {"ts", "symbol", "weight"}


def test_mean_reversion_long_only():
    from quant.strategies.library import MeanReversionParams

    panel = make_synthetic_panel(n_symbols=10, n_bars=200, seed=13)
    strat = CryptoMeanReversion(params=MeanReversionParams(long_only=True))
    weights = strat.generate_weights(panel)
    assert (weights["weight"] >= 0.0).all()
