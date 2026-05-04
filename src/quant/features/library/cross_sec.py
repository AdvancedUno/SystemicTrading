"""Cross-sectional features.

────────────────────────────────────────────────────────────────────────────
What "cross-sectional" means
────────────────────────────────────────────────────────────────────────────

Most features above are *time-series* features: they describe one symbol's
behavior over time. Cross-sectional features describe a symbol *relative
to the universe at the same instant*.

At every timestamp t, we have N stocks each with a feature value.
A cross-sectional rank or z-score asks: "where does AAPL stand among
all stocks today?" rather than "how does AAPL compare to its own past?"

Cross-sectional features are the bedrock of equity long/short strategies
because they're naturally *dollar-neutral*: rank-based signals are by
construction zero-mean across the universe at every t. Long the top
decile, short the bottom decile, and your portfolio has no net market
exposure. This is why cross-sectional momentum (Jegadeesh-Titman)
out-of-sample doesn't depend on whether the market is up or down.

────────────────────────────────────────────────────────────────────────────
Cross-sectional rank
────────────────────────────────────────────────────────────────────────────

For a feature x, at each t:
    rank_t(x_i) = (number of stocks with x_j < x_i at time t) + 1

We normalize to [0, 1]:
    cs_rank(x_i, t) = (rank_t(x_i) - 1) / (N_t - 1)

Or sometimes to [-0.5, 0.5] for symmetry. We use [-0.5, 0.5] because
many models train better on zero-centered features.

────────────────────────────────────────────────────────────────────────────
Cross-sectional z-score
────────────────────────────────────────────────────────────────────────────

    cs_z(x_i, t) = (x_i,t - mean_t(x)) / std_t(x)

Same idea as rank but parametric. More efficient if x is roughly Gaussian,
less robust to outliers. We provide both and let downstream models choose.

────────────────────────────────────────────────────────────────────────────
Implementation note
────────────────────────────────────────────────────────────────────────────

Cross-sectional features can't be computed with `.over("symbol")` like
time-series features. They need `.over("ts")` — group by timestamp, not
symbol. So they need a different code path in the pipeline.

We mark these features by having `Feature.lookback = 0` and a special
category "cross_sectional", and the pipeline applies them differently.
"""

from __future__ import annotations

import polars as pl

from quant.features.registry import register


# Cross-sectional features look different from time-series ones — they
# don't shift over time, they normalize across symbols at a given t.
# We give them lag=0 and let the pipeline apply them after time-series
# features are computed (and already lagged). The lag from the underlying
# feature carries through.


@register(
    category="cross_sectional",
    lookback=0,
    lag=0,
    depends_on=("return_20d",),
    name="cs_rank_return_20d",
)
def cs_rank_return_20d() -> pl.Expr:
    """Cross-sectional rank of 20d return, in [-0.5, 0.5]."""
    return _cs_rank("return_20d")


@register(
    category="cross_sectional",
    lookback=0,
    lag=0,
    depends_on=("momentum_12m_1m",),
    name="cs_rank_momentum_12_1",
)
def cs_rank_momentum_12_1() -> pl.Expr:
    return _cs_rank("momentum_12m_1m")


@register(
    category="cross_sectional",
    lookback=0,
    lag=0,
    depends_on=("realized_vol_20d",),
    name="cs_rank_realized_vol_20d",
)
def cs_rank_realized_vol_20d() -> pl.Expr:
    """Cross-sectional rank of 20-day realized vol.

    High rank = high vol stock today vs peers. Used in low-vol-anomaly
    strategies (sort low) and in vol-targeting (sort high to reduce).
    """
    return _cs_rank("realized_vol_20d")


@register(
    category="cross_sectional",
    lookback=0,
    lag=0,
    depends_on=("dollar_volume_zscore_20d",),
    name="cs_rank_dollar_volume_z",
)
def cs_rank_dollar_volume_z() -> pl.Expr:
    return _cs_rank("dollar_volume_zscore_20d")


@register(
    category="cross_sectional",
    lookback=0,
    lag=0,
    depends_on=("amihud_illiq_20d",),
    name="cs_rank_amihud",
)
def cs_rank_amihud() -> pl.Expr:
    return _cs_rank("amihud_illiq_20d")


@register(
    category="cross_sectional",
    lookback=0,
    lag=0,
    depends_on=("rsi_14",),
    name="cs_zscore_rsi_14",
)
def cs_zscore_rsi_14() -> pl.Expr:
    return _cs_zscore("rsi_14")


# ---------------------------------------------------------------------------
# Helpers — these get .over("ts") applied at the pipeline level
# ---------------------------------------------------------------------------


def _cs_rank(col: str) -> pl.Expr:
    """Rank values cross-sectionally (within ts), scaled to [-0.5, 0.5]."""
    rank = pl.col(col).rank(method="average")
    n = pl.col(col).count()
    # Result of rank is 1..N; (rank-1)/(N-1) gives [0,1]; subtract 0.5 to center.
    return (rank - 1.0) / (n - 1.0) - 0.5


def _cs_zscore(col: str) -> pl.Expr:
    """Z-score values cross-sectionally (within ts)."""
    x = pl.col(col)
    mean = x.mean()
    std = x.std()
    return (x - mean) / std.clip(lower_bound=1e-12)
