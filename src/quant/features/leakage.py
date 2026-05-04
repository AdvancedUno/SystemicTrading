"""Leakage detection.

A feature has *look-ahead leakage* if its value at time t depends on
information from time >= t. This is the #1 silent killer of backtests.

The test:
    1. Compute feature values on the original panel.
    2. Mutate the panel by replacing all bars at time >= t0 with garbage
       (random values, NaN, whatever).
    3. Recompute features.
    4. Compare the two: feature values at time < t0 must be IDENTICAL.

If they differ, the feature peeks at the future. By construction, a feature
that only uses past bars cannot change when we mutate future bars.

We run this test on every registered feature in the test suite. Any new
feature that fails leaks — bug, not a false alarm.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import polars as pl

from quant.features.base import Feature
from quant.features.pipeline import FeaturePipeline


def make_synthetic_panel(
    n_symbols: int = 5,
    n_bars: int = 300,
    seed: int = 42,
    start: datetime | None = None,
) -> pl.DataFrame:
    """Make a synthetic long-format OHLCV panel for testing.

    Generates plausible-looking GBM-like prices so that volatility and
    return features have non-degenerate values to test against.
    """
    rng = np.random.default_rng(seed)
    start = start or datetime(2024, 1, 1, tzinfo=timezone.utc)
    timestamps = [start.replace(day=1) for _ in range(n_bars)]
    # Use simple sequential daily timestamps
    import datetime as _dt

    ts_list = [
        start + _dt.timedelta(days=i) for i in range(n_bars)
    ]

    rows = []
    for s in range(n_symbols):
        symbol = f"SYM{s}"
        # GBM-like log returns
        log_rets = rng.normal(loc=0.0003, scale=0.015, size=n_bars)
        prices = 100.0 * np.exp(np.cumsum(log_rets))
        # Make plausible OHLCV
        for i, ts in enumerate(ts_list):
            close = float(prices[i])
            open_ = close * float(rng.normal(1.0, 0.005))
            high = max(open_, close) * float(rng.uniform(1.0, 1.01))
            low = min(open_, close) * float(rng.uniform(0.99, 1.0))
            volume = float(rng.lognormal(15.0, 0.3))
            rows.append(
                {
                    "ts": ts,
                    "symbol": symbol,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "trade_count": int(rng.integers(1000, 10000)),
                    "vwap": (open_ + high + low + close) / 4.0,
                }
            )
    df = pl.DataFrame(rows).with_columns(
        pl.col("ts").dt.cast_time_unit("us").dt.replace_time_zone("UTC")
    )
    return df.sort(["symbol", "ts"])


def detect_leakage(
    feature: Feature,
    panel: pl.DataFrame | None = None,
    cutoff_idx: int | None = None,
    seed: int = 42,
) -> tuple[bool, str]:
    """Test a single feature for look-ahead leakage.

    Returns (leaks, message). leaks=False is good.

    Algorithm:
        1. Compute feature on the panel.
        2. Make a "future-mutated" copy: replace all rows at index >= cutoff_idx
           with random nonsense (per symbol).
        3. Recompute the feature on the mutated panel.
        4. The feature values at index < cutoff_idx must be unchanged.

    If they change, the feature uses future information.
    """
    if panel is None:
        panel = make_synthetic_panel(seed=seed)
    panel = panel.sort(["symbol", "ts"])

    # Pick a cutoff in the middle, well past lookback
    if cutoff_idx is None:
        per_symbol = panel.height // panel["symbol"].n_unique()
        cutoff_idx = per_symbol // 2 + feature.lookback

    pipe = FeaturePipeline([feature])

    # Original
    orig = pipe.apply(panel).sort(["symbol", "ts"])

    # Mutated: replace future rows with random garbage
    rng = np.random.default_rng(seed + 1)
    mutated = panel.clone()
    mutated_rows = []
    for sym_df in mutated.partition_by("symbol", as_dict=False):
        sym = sym_df["symbol"][0]
        h = sym_df.height
        # Replace rows at index >= cutoff_idx
        n_replace = max(h - cutoff_idx, 0)
        if n_replace > 0:
            # Random OHLCV values, far from the original distribution
            replacement = pl.DataFrame(
                {
                    "ts": sym_df["ts"][cutoff_idx:].to_list(),
                    "symbol": [sym] * n_replace,
                    "open": rng.uniform(1.0, 10000.0, n_replace).tolist(),
                    "high": rng.uniform(1.0, 10000.0, n_replace).tolist(),
                    "low": rng.uniform(1.0, 10000.0, n_replace).tolist(),
                    "close": rng.uniform(1.0, 10000.0, n_replace).tolist(),
                    "volume": rng.uniform(1.0, 1e9, n_replace).tolist(),
                    "trade_count": rng.integers(1, 100000, n_replace).tolist(),
                    "vwap": rng.uniform(1.0, 10000.0, n_replace).tolist(),
                }
            ).with_columns(
                pl.col("ts").dt.cast_time_unit("us").dt.replace_time_zone("UTC"),
                pl.col("trade_count").cast(pl.Int64),
            )
            kept = sym_df.head(cutoff_idx)
            mutated_rows.append(pl.concat([kept, replacement]))
        else:
            mutated_rows.append(sym_df)
    mutated_df = pl.concat(mutated_rows).sort(["symbol", "ts"])
    mut = pipe.apply(mutated_df).sort(["symbol", "ts"])

    # Compare feature column at indices < cutoff_idx, per symbol
    col = feature.name
    orig_pre = (
        orig.group_by("symbol", maintain_order=True)
        .head(cutoff_idx)
        .select(["symbol", "ts", col])
    )
    mut_pre = (
        mut.group_by("symbol", maintain_order=True)
        .head(cutoff_idx)
        .select(["symbol", "ts", col])
    )

    # Compare with NaN-aware logic
    a = orig_pre[col].to_numpy()
    b = mut_pre[col].to_numpy()

    # Both nan in same positions?
    a_nan = np.isnan(a)
    b_nan = np.isnan(b)
    if not np.array_equal(a_nan, b_nan):
        return True, (
            f"NaN pattern differs at indices < cutoff: "
            f"{(a_nan != b_nan).sum()} mismatched positions"
        )

    finite_mask = ~a_nan
    if finite_mask.any():
        diffs = np.abs(a[finite_mask] - b[finite_mask])
        max_diff = float(diffs.max())
        if max_diff > 1e-9:
            return True, (
                f"Feature values differ at indices < cutoff: "
                f"max abs diff = {max_diff:.6g}"
            )

    return False, "No leakage detected"
