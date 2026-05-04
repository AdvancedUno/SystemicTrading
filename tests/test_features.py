"""Tests for feature pipeline and leakage detection."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from quant.features import REGISTRY, FeaturePipeline
from quant.features.leakage import detect_leakage, make_synthetic_panel


def test_registry_has_features():
    """All built-in features registered."""
    assert len(REGISTRY) >= 25, f"Expected 25+ features, got {len(REGISTRY)}"


def test_categories():
    """Expected categories present."""
    cats = set(REGISTRY.categories())
    assert "returns" in cats
    assert "volatility" in cats
    assert "volume" in cats
    assert "technical" in cats
    assert "cross_sectional" in cats


def test_pipeline_runs_on_synthetic_data():
    """Full pipeline runs end-to-end on synthetic panel."""
    panel = make_synthetic_panel(n_symbols=5, n_bars=300)
    pipe = FeaturePipeline()  # all features
    out = pipe.apply(panel)

    # Sanity: shape preserved, feature columns added
    assert out.height == panel.height
    for fname in pipe.feature_names():
        assert fname in out.columns

    # After the warmup period, we should have non-null values
    warmup = pipe.max_lookback() + 5
    tail = out.group_by("symbol").tail(50)  # well past warmup
    for fname in pipe.feature_names():
        non_null = tail[fname].drop_nulls().drop_nans()
        assert non_null.len() > 0, f"Feature {fname} has no non-null values"


def test_no_leakage_on_any_feature():
    """Critical: every registered feature must pass leakage detection."""
    panel = make_synthetic_panel(n_symbols=5, n_bars=300, seed=7)
    failures = []
    for feat in REGISTRY.all():
        # Cross-sectional features can't be tested in isolation — they
        # depend on time-series features. Test them as part of the full pipeline.
        if feat.category == "cross_sectional":
            continue
        leaks, msg = detect_leakage(feat, panel=panel)
        if leaks:
            failures.append(f"{feat.name}: {msg}")
    assert not failures, "Leakage detected:\n" + "\n".join(failures)


def test_no_leakage_full_pipeline():
    """Run leakage check on the full pipeline output, including cs features."""
    panel = make_synthetic_panel(n_symbols=5, n_bars=300, seed=11)
    pipe = FeaturePipeline()

    cutoff = 150
    orig = pipe.apply(panel).sort(["symbol", "ts"])

    # Mutate future bars
    rng = np.random.default_rng(99)
    parts = []
    for sym_df in panel.partition_by("symbol"):
        sym = sym_df["symbol"][0]
        n_replace = sym_df.height - cutoff
        kept = sym_df.head(cutoff)
        if n_replace <= 0:
            parts.append(sym_df)
            continue
        replacement = pl.DataFrame(
            {
                "ts": sym_df["ts"][cutoff:].to_list(),
                "symbol": [sym] * n_replace,
                "open": rng.uniform(1, 1e4, n_replace).tolist(),
                "high": rng.uniform(1, 1e4, n_replace).tolist(),
                "low": rng.uniform(1, 1e4, n_replace).tolist(),
                "close": rng.uniform(1, 1e4, n_replace).tolist(),
                "volume": rng.uniform(1, 1e9, n_replace).tolist(),
                "trade_count": rng.integers(1, 1e5, n_replace).tolist(),
                "vwap": rng.uniform(1, 1e4, n_replace).tolist(),
            }
        ).with_columns(
            pl.col("ts").dt.cast_time_unit("us").dt.replace_time_zone("UTC"),
            pl.col("trade_count").cast(pl.Int64),
        )
        parts.append(pl.concat([kept, replacement]))
    mutated_panel = pl.concat(parts).sort(["symbol", "ts"])
    mut = pipe.apply(mutated_panel).sort(["symbol", "ts"])

    # All feature columns at index < cutoff should match
    failures = []
    for fname in pipe.feature_names():
        a = (
            orig.group_by("symbol", maintain_order=True)
            .head(cutoff)[fname]
            .to_numpy()
        )
        b = (
            mut.group_by("symbol", maintain_order=True)
            .head(cutoff)[fname]
            .to_numpy()
        )
        a_nan, b_nan = np.isnan(a), np.isnan(b)
        if not np.array_equal(a_nan, b_nan):
            failures.append(f"{fname}: NaN pattern differs")
            continue
        m = ~a_nan
        if m.any():
            d = float(np.abs(a[m] - b[m]).max())
            if d > 1e-9:
                failures.append(f"{fname}: max diff = {d:.6g}")
    assert not failures, "Leakage in full pipeline:\n" + "\n".join(failures)


def test_pipeline_subset():
    """Pipeline can be initialized with a subset of features."""
    pipe = FeaturePipeline(["return_1d", "return_20d", "rsi_14"])
    assert len(pipe) == 3
    panel = make_synthetic_panel(n_symbols=3, n_bars=100)
    out = pipe.apply(panel)
    assert "return_1d" in out.columns
    assert "rsi_14" in out.columns
    assert "return_5d" not in out.columns


def test_feature_lag_applied():
    """Verify the central .shift(1) lag is in effect.

    Constructs a single-symbol panel where return_1d is known, and checks
    that the feature column is shifted by one bar relative to the raw return.
    """
    panel = make_synthetic_panel(n_symbols=1, n_bars=20, seed=1)
    pipe = FeaturePipeline(["return_1d"])
    out = pipe.apply(panel).sort("ts")

    closes = out["close"].to_numpy()
    raw_ret = closes[1:] / closes[:-1] - 1
    lagged_ret = out["return_1d"].to_numpy()

    # The feature at index t should equal raw_ret at t-1.
    # So lagged_ret[2:] should equal raw_ret[:-1] (within float tolerance).
    assert np.allclose(lagged_ret[2:], raw_ret[:-1], equal_nan=True)
    assert np.isnan(lagged_ret[0])  # first bar — no return yet
    assert np.isnan(lagged_ret[1])  # second bar — value at t=0 lagged out


def test_pipeline_idempotent():
    """Running the pipeline twice on the same data gives the same result."""
    panel = make_synthetic_panel(n_symbols=3, n_bars=200)
    pipe = FeaturePipeline()
    out1 = pipe.apply(panel)
    out2 = pipe.apply(panel)
    assert out1.equals(out2)
