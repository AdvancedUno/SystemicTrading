"""Technical / oscillator features.

────────────────────────────────────────────────────────────────────────────
1. RSI — Relative Strength Index (Wilder, 1978)
────────────────────────────────────────────────────────────────────────────

The classic momentum oscillator. Despite living in the world of "technical
analysis," RSI is just a particular smoothing of the up/down ratio of
returns and works fine as an ML feature.

Definition (period N, classic = 14):

    For each bar t, define:
        gain(t) = max(close(t) - close(t-1), 0)
        loss(t) = max(close(t-1) - close(t), 0)

    Wilder's smoothing — an EMA with alpha = 1/N (NOT the usual 2/(N+1)):
        avg_gain(t) = (avg_gain(t-1) * (N-1) + gain(t)) / N
        avg_loss(t) = (avg_loss(t-1) * (N-1) + loss(t)) / N

    RS(t)  = avg_gain(t) / avg_loss(t)
    RSI(t) = 100 - 100 / (1 + RS(t))

RSI ranges in [0, 100]. Traditionally:
    > 70 = "overbought" (mean-reversion candidates)
    < 30 = "oversold" (bounce candidates)

For our ML pipeline we don't care about the thresholds; the raw value
goes in as a feature and the model learns the relationship.

We approximate Wilder's smoothing with Polars' `ewm_mean` using
`alpha = 1/N`, which gives the same EMA recursion.

────────────────────────────────────────────────────────────────────────────
2. Z-score of price relative to moving average
────────────────────────────────────────────────────────────────────────────

    z(t) = (close(t) - mean_N(close)) / std_N(close)

A purely statistical mean-reversion signal. Captures "how far is the
price from its recent average, in stdev units?"

────────────────────────────────────────────────────────────────────────────
3. MACD-like momentum (fast EMA - slow EMA)
────────────────────────────────────────────────────────────────────────────

    macd(t) = EMA_12(close) - EMA_26(close)

We normalize by the close so it's comparable across price levels:

    macd_norm(t) = (EMA_12(close) - EMA_26(close)) / close(t)
"""

from __future__ import annotations

import polars as pl

from quant.features.registry import register


def _rsi_expr(period: int) -> pl.Expr:
    """Build an RSI expression with Wilder smoothing."""
    delta = pl.col("close") - pl.col("close").shift(1)
    gain = pl.when(delta > 0).then(delta).otherwise(0.0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0.0)
    # Wilder smoothing: alpha = 1/period
    avg_gain = gain.ewm_mean(alpha=1.0 / period, adjust=False, min_samples=period)
    avg_loss = loss.ewm_mean(alpha=1.0 / period, adjust=False, min_samples=period)
    rs = avg_gain / avg_loss.clip(lower_bound=1e-12)  # avoid div-by-zero
    return 100.0 - (100.0 / (1.0 + rs))


@register(category="technical", lookback=14)
def rsi_14() -> pl.Expr:
    """Wilder's 14-period RSI."""
    return _rsi_expr(14)


@register(category="technical", lookback=30)
def rsi_30() -> pl.Expr:
    """30-period RSI — slower, less noisy."""
    return _rsi_expr(30)


@register(category="technical", lookback=20)
def price_zscore_20d() -> pl.Expr:
    """Z-score of close vs its 20-bar mean and std.

    Interpretation: a value of +2 means the price is 2 standard deviations
    above its recent mean — a classic mean-reversion candidate.
    """
    c = pl.col("close")
    mean = c.rolling_mean(window_size=20)
    std = c.rolling_std(window_size=20)
    return (c - mean) / std.clip(lower_bound=1e-12)


@register(category="technical", lookback=60)
def price_zscore_60d() -> pl.Expr:
    """60-bar version of price z-score."""
    c = pl.col("close")
    mean = c.rolling_mean(window_size=60)
    std = c.rolling_std(window_size=60)
    return (c - mean) / std.clip(lower_bound=1e-12)


@register(category="technical", lookback=26)
def macd_norm() -> pl.Expr:
    """Normalized MACD: (EMA_12 - EMA_26) / close.

    Normalizing by the price level makes it comparable across stocks
    (a $400 stock and a $20 stock should have features on the same scale).
    """
    ema_fast = pl.col("close").ewm_mean(span=12, adjust=False, min_samples=12)
    ema_slow = pl.col("close").ewm_mean(span=26, adjust=False, min_samples=26)
    return (ema_fast - ema_slow) / pl.col("close")
