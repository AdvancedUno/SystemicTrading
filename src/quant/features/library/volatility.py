"""Volatility features.

We compute several volatility estimators because each has different bias
and efficiency properties.

────────────────────────────────────────────────────────────────────────────
1. Realized volatility (close-to-close)
────────────────────────────────────────────────────────────────────────────
   σ_realized = sqrt(sum of squared log-returns over window)

Or equivalently the rolling stdev of log returns. This is the simplest
estimator. It uses only ONE data point per bar (the close) and discards
the high/low, so it's noisy.

────────────────────────────────────────────────────────────────────────────
2. Parkinson volatility
────────────────────────────────────────────────────────────────────────────
Parkinson (1980) showed that the high-low range carries information about
the volatility of the underlying Brownian motion. Specifically:

   σ²_parkinson(t) = (1 / (4 ln 2)) * (ln(high(t) / low(t)))²

Then over an N-bar window we average:

   σ²_park_window = (1/N) * sum over k of σ²_parkinson(t-k)

The constant 1/(4 ln 2) ≈ 0.36 is what makes this an unbiased estimator
of σ² when prices follow geometric Brownian motion. Parkinson is roughly
5x more efficient than close-to-close: it produces the same accuracy with
1/5 the data, because high/low contain more information than close alone.

Caveat: Parkinson assumes continuous trading and no drift. For overnight
gaps in equities (close at 4pm, open at 9:30am with a gap), it misses
overnight vol entirely. We document this explicitly.

────────────────────────────────────────────────────────────────────────────
3. Garman-Klass volatility
────────────────────────────────────────────────────────────────────────────
Garman & Klass (1980) extended Parkinson by also using open and close:

   σ²_GK(t) = 0.5 * (ln(high/low))² - (2 ln 2 - 1) * (ln(close/open))²

GK is roughly 7-8x more efficient than close-to-close. It assumes no
overnight gaps either. Most practitioners use GK or its drift-robust
variant Garman-Klass-Yang-Zhang.

────────────────────────────────────────────────────────────────────────────
4. Vol-of-vol
────────────────────────────────────────────────────────────────────────────
The standard deviation of rolling realized vol. Captures whether volatility
itself is stable or chaotic — useful for regime detection and risk control.
Often a strong feature: when vol-of-vol spikes, you're entering a regime
shift.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import math

import polars as pl

from quant.features.registry import register

# Constants
_PARK_CONST = 1.0 / (4.0 * math.log(2.0))  # ≈ 0.3607
_GK_CONST_2 = 2.0 * math.log(2.0) - 1.0  # ≈ 0.3863


# ---------------------------------------------------------------------------
# Close-to-close realized volatility
# ---------------------------------------------------------------------------
# Implementation note: rolling_std on log returns. We don't annualize at
# the feature level; downstream code can multiply by sqrt(252) if it wants.


@register(category="volatility", lookback=20)
def realized_vol_20d() -> pl.Expr:
    """Stdev of 1-bar log returns over a 20-bar window."""
    log_ret = (pl.col("close") / pl.col("close").shift(1)).log()
    return log_ret.rolling_std(window_size=20)


@register(category="volatility", lookback=60)
def realized_vol_60d() -> pl.Expr:
    """Stdev of 1-bar log returns over a 60-bar window."""
    log_ret = (pl.col("close") / pl.col("close").shift(1)).log()
    return log_ret.rolling_std(window_size=60)


# ---------------------------------------------------------------------------
# Parkinson volatility
# ---------------------------------------------------------------------------


@register(category="volatility", lookback=20)
def parkinson_vol_20d() -> pl.Expr:
    """Parkinson high-low volatility estimator over 20 bars.

    σ² = (1 / (4 ln 2)) * mean over window of (ln(high/low))²
    Returned as σ (sqrt of variance).
    """
    hl_log_sq = ((pl.col("high") / pl.col("low")).log()) ** 2
    var = _PARK_CONST * hl_log_sq.rolling_mean(window_size=20)
    return var.sqrt()


# ---------------------------------------------------------------------------
# Garman-Klass volatility
# ---------------------------------------------------------------------------


@register(category="volatility", lookback=20)
def garman_klass_vol_20d() -> pl.Expr:
    """Garman-Klass volatility estimator over 20 bars.

    σ² = mean over window of [
        0.5 * (ln(high/low))² - (2 ln 2 - 1) * (ln(close/open))²
    ]
    """
    hl_term = 0.5 * ((pl.col("high") / pl.col("low")).log()) ** 2
    co_term = _GK_CONST_2 * ((pl.col("close") / pl.col("open")).log()) ** 2
    per_bar = hl_term - co_term
    var = per_bar.rolling_mean(window_size=20)
    # GK can produce negative values in low-volatility / weird-bar cases;
    # clip to zero before sqrt
    return var.clip(lower_bound=0.0).sqrt()


# ---------------------------------------------------------------------------
# Vol of vol
# ---------------------------------------------------------------------------


@register(category="volatility", lookback=80, depends_on=("realized_vol_20d",))
def vol_of_vol_60d() -> pl.Expr:
    """Stdev of 20-day realized vol over a 60-bar window.

    Captures whether the vol regime is stable or shifting.
    """
    log_ret = (pl.col("close") / pl.col("close").shift(1)).log()
    rv20 = log_ret.rolling_std(window_size=20)
    return rv20.rolling_std(window_size=60)


# ---------------------------------------------------------------------------
# Realized range — simple and surprisingly useful
# ---------------------------------------------------------------------------
# Mean (high - low) / close over a window, sometimes called "average true
# range normalized." Robust, easy to interpret, and adds info beyond
# return-based vol.


@register(category="volatility", lookback=20)
def avg_range_20d() -> pl.Expr:
    """Average normalized intraday range over 20 bars: mean((high-low)/close)."""
    bar_range = (pl.col("high") - pl.col("low")) / pl.col("close")
    return bar_range.rolling_mean(window_size=20)
