"""Return features.

Math notes:

Simple return over k bars:
    r_k(t) = close(t) / close(t-k) - 1

Log return over k bars:
    lr_k(t) = log(close(t)) - log(close(t-k)) = log(close(t) / close(t-k))

Why both? Simple returns aggregate cleanly across assets at a single time
(portfolio return = weighted sum of asset returns), while log returns
aggregate cleanly across time (k-day log return = sum of 1-day log returns).
ML models often prefer log returns because they're roughly symmetric
around zero (a +50% then -50% sequence gives a log return of 0 in expectation,
not -25%). For features, we'll keep both since downstream models may want either.

Forward shift discipline:
- These features are computed at the close of bar t.
- The lag of 1 (applied centrally in Feature.expr) means the value
  available "as of time t" is what we'd compute using bars up to t-1.
- That means in our final feature panel, row t will hold the return
  computed from close(t-1) / close(t-1-k) - 1. This is what we intend.
"""

from __future__ import annotations

import polars as pl

from quant.features.registry import register


# ---------------------------------------------------------------------------
# Simple returns at multiple horizons
# ---------------------------------------------------------------------------
# We register horizons that span "fast mean reversion" (1d), "weekly trends"
# (5d), "monthly momentum" (20d), and "quarterly trends" (60d).
# These four horizons together let cross-sectional momentum + reversal
# models pick up different regimes.


@register(category="returns", lookback=1)
def return_1d() -> pl.Expr:
    """1-bar simple return."""
    return pl.col("close") / pl.col("close").shift(1) - 1


@register(category="returns", lookback=5)
def return_5d() -> pl.Expr:
    """5-bar simple return."""
    return pl.col("close") / pl.col("close").shift(5) - 1


@register(category="returns", lookback=20)
def return_20d() -> pl.Expr:
    """20-bar simple return — classic ~1-month momentum window."""
    return pl.col("close") / pl.col("close").shift(20) - 1


@register(category="returns", lookback=60)
def return_60d() -> pl.Expr:
    """60-bar simple return — quarterly momentum."""
    return pl.col("close") / pl.col("close").shift(60) - 1


# ---------------------------------------------------------------------------
# Log returns
# ---------------------------------------------------------------------------


@register(category="returns", lookback=1)
def log_return_1d() -> pl.Expr:
    """1-bar log return."""
    return (pl.col("close") / pl.col("close").shift(1)).log()


@register(category="returns", lookback=5)
def log_return_5d() -> pl.Expr:
    """5-bar log return."""
    return (pl.col("close") / pl.col("close").shift(5)).log()


@register(category="returns", lookback=20)
def log_return_20d() -> pl.Expr:
    """20-bar log return."""
    return (pl.col("close") / pl.col("close").shift(20)).log()


# ---------------------------------------------------------------------------
# Skip-1 momentum (Jegadeesh-Titman style: r[t-12, t-1])
# ---------------------------------------------------------------------------
# Classic finding (Jegadeesh & Titman, 1993): cross-sectional momentum
# works best when you SKIP the most recent month, because that recent
# month tends to mean-revert (microstructure noise / liquidity effects).
# In our daily-bar setting we approximate "skip 1 month" as skip 20 bars.
#
#   skip1_mom_12_1(t) = close(t-20) / close(t-240) - 1
#
# = "the return over the year ending 1 month ago"


@register(category="returns", lookback=240, name="momentum_12m_1m")
def momentum_12m_1m() -> pl.Expr:
    """Jegadeesh-Titman 12-month momentum, skipping last month.

    Defined as the return from t-240 to t-20. Captures medium-term
    momentum while excluding short-term mean-reversion noise.
    """
    return pl.col("close").shift(20) / pl.col("close").shift(240) - 1
