"""Cross-sectional momentum (equity).

The classic AQR / Two Sigma-style equity long/short strategy:

  At each rebalance:
    score_i = w_long * cs_rank(momentum_12_1)_i
              + w_short * cs_rank(return_20d)_i  # negative weight = mean-reversion

    Long the top quintile (highest scores)
    Short the bottom quintile (lowest scores)
    Equal weight within each side
    Net dollar-neutral, gross leverage configurable

Why these features?
  - momentum_12_1: 12-month momentum skipping the last month (Jegadeesh-Titman).
    The classic medium-term momentum signal that's persisted out-of-sample
    for 30+ years.
  - return_20d (with NEGATIVE weight): captures short-term reversal —
    stocks that ran up over the last month tend to revert (microstructure
    noise, profit-taking). This is a separate, almost-orthogonal signal
    to the 12-1 momentum.

Vol filter (optional):
  Within the long and short books, downweight high-vol names so we don't
  load up on lottery tickets. Implemented as: w_i ∝ (1 / vol_i)^vol_filter_power.

This is meant as a learning example, not an alpha strategy. Real production
versions add: sector neutrality, beta hedging, factor exposure constraints,
turnover penalties, capacity adjustments, regime filters, etc.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from quant.features import FeaturePipeline
from quant.strategies.base import Strategy, StrategyConfig, WEIGHT_COLUMNS


@dataclass(frozen=True)
class CSMomentumParams:
    """Strategy-specific parameters."""

    momentum_weight: float = 1.0  # weight on cs_rank_momentum_12_1
    reversal_weight: float = -0.3  # NEGATIVE weight on cs_rank_return_20d
    quantile: float = 0.2  # top/bottom 20% (quintile)
    vol_filter_power: float = 0.5  # 0 = no filter, 1 = inverse vol weighting
    min_universe_size: int = 10  # below this, hold cash


class CrossSectionalMomentum(Strategy):
    """Cross-sectional momentum + reversal long/short equity strategy."""

    def __init__(
        self,
        params: CSMomentumParams | None = None,
        config: StrategyConfig | None = None,
    ) -> None:
        super().__init__(
            config
            or StrategyConfig(
                name="cs_momentum",
                rebalance_every=5,  # weekly on daily bars
                gross_leverage=1.0,
            )
        )
        self.params = params or CSMomentumParams()
        self._pipe = FeaturePipeline(
            [
                "momentum_12m_1m",
                "return_20d",
                "realized_vol_20d",
                "cs_rank_momentum_12_1",
                "cs_rank_return_20d",
                "cs_rank_realized_vol_20d",
            ]
        )

    def generate_weights(self, panel: pl.DataFrame) -> pl.DataFrame:
        # Compute features
        features = self._pipe.apply(panel)

        # Combined score
        p = self.params
        scored = features.with_columns(
            (
                p.momentum_weight * pl.col("cs_rank_momentum_12_1")
                + p.reversal_weight * pl.col("cs_rank_return_20d")
            ).alias("_score")
        ).filter(pl.col("_score").is_not_null())

        if scored.is_empty():
            return _empty_weights()

        # Per-ts: rank scores, pick top/bottom quantile, equal weight within
        # (with optional inverse-vol scaling)
        weights = (
            scored.with_columns(
                pl.col("_score").rank(method="average").over("ts").alias("_rank"),
                pl.col("_score").count().over("ts").alias("_n"),
            )
            .with_columns(
                # Normalized rank in [0, 1]
                ((pl.col("_rank") - 1) / (pl.col("_n") - 1)).alias("_pct")
            )
            # Drop ts with too few names
            .filter(pl.col("_n") >= p.min_universe_size)
            .with_columns(
                pl.when(pl.col("_pct") >= 1 - p.quantile)
                .then(1.0)  # long
                .when(pl.col("_pct") <= p.quantile)
                .then(-1.0)  # short
                .otherwise(0.0)
                .alias("_side")
            )
            .filter(pl.col("_side") != 0.0)
        )

        if weights.is_empty():
            return _empty_weights()

        # Inverse-vol scaling (within each side, per ts)
        weights = weights.with_columns(
            pl.when(pl.col("realized_vol_20d") > 1e-8)
            .then(pl.col("realized_vol_20d") ** (-p.vol_filter_power))
            .otherwise(0.0)
            .alias("_invvol")
        )

        # Normalize so |weight| sums to gross_leverage / 2 on each side
        # (so total gross sum = gross_leverage)
        side_sum = pl.col("_invvol").sum().over(["ts", "_side"])
        weights = weights.with_columns(
            (
                pl.col("_side")
                * pl.col("_invvol")
                / pl.when(side_sum > 1e-12).then(side_sum).otherwise(1.0)
                * (self.config.gross_leverage / 2.0)
            ).alias("weight")
        ).select(list(WEIGHT_COLUMNS))

        return weights.sort(["ts", "symbol"])


def _empty_weights() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            "symbol": pl.Series([], dtype=pl.Utf8),
            "weight": pl.Series([], dtype=pl.Float64),
        }
    )
