"""Z-score mean reversion (crypto).

Crypto markets show strong short-term mean-reversion at the daily level —
big moves over 1-3 days tend to retrace. This is the simplest strategy
that takes advantage of it:

  At each rebalance:
    z_i = price_zscore_20d(close)_i  # how many stdevs the price is from its 20-day mean
    weight_i = -clip(z_i, -2, 2) / 2  # negative = mean reversion

  When z is high (price stretched up), short.
  When z is low (price stretched down), long.
  Clip to ±2 so single-day-vol-spikes don't blow up the position.

Vol-targeting:
  We scale to a target portfolio vol — small positions in volatile assets,
  larger in stable ones. Avoids the strategy being dominated by whichever
  coin is most volatile that week.

Why crypto-specific?
  - Mean-reversion works on daily crypto more reliably than on equity
    indices (which trend more).
  - Crypto is 24/7 so "1d" bars actually cover 24h of trading; equity 1d
    bars miss overnight gaps which complicates this signal there.
  - Crypto markets are dominated by retail flow, which creates the kind
    of overshoot/correction patterns this strategy harvests.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from quant.features import FeaturePipeline
from quant.strategies.base import Strategy, StrategyConfig, WEIGHT_COLUMNS


@dataclass(frozen=True)
class MeanReversionParams:
    z_clip: float = 2.0  # |z| above this is clipped before sizing
    target_vol: float = 0.10  # 10% annualized portfolio vol target
    long_only: bool = False  # if True, only take longs (no shorts)


class CryptoMeanReversion(Strategy):
    """Z-score-based mean-reversion for daily-bar crypto."""

    def __init__(
        self,
        params: MeanReversionParams | None = None,
        config: StrategyConfig | None = None,
    ) -> None:
        super().__init__(
            config
            or StrategyConfig(
                name="crypto_mean_reversion",
                rebalance_every=1,  # daily
                gross_leverage=1.0,
            )
        )
        self.params = params or MeanReversionParams()
        self._pipe = FeaturePipeline(
            [
                "price_zscore_20d",
                "realized_vol_20d",
            ]
        )

    def generate_weights(self, panel: pl.DataFrame) -> pl.DataFrame:
        features = self._pipe.apply(panel)

        # Filter rows with valid features
        features = features.filter(
            pl.col("price_zscore_20d").is_not_null()
            & pl.col("realized_vol_20d").is_not_null()
            & (pl.col("realized_vol_20d") > 1e-8)
        )
        if features.is_empty():
            return _empty_weights()

        p = self.params
        # Raw signal: -z (negative because mean reversion)
        signal = -features["price_zscore_20d"].clip(-p.z_clip, p.z_clip) / p.z_clip
        if p.long_only:
            signal = signal.clip(lower_bound=0.0)
        features = features.with_columns(signal.alias("_signal"))

        # Per-asset vol targeting: w ∝ signal / vol
        # Then normalize per ts so sum |w| = gross_leverage
        weights = (
            features.with_columns(
                (pl.col("_signal") / pl.col("realized_vol_20d")).alias("_raw_w")
            )
            .with_columns(
                pl.col("_raw_w").abs().sum().over("ts").alias("_norm")
            )
            .with_columns(
                pl.when(pl.col("_norm") > 1e-12)
                .then(pl.col("_raw_w") * self.config.gross_leverage / pl.col("_norm"))
                .otherwise(0.0)
                .alias("weight")
            )
            .filter(pl.col("weight").abs() > 1e-9)
            .select(list(WEIGHT_COLUMNS))
        )

        return weights.sort(["ts", "symbol"])


def _empty_weights() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            "symbol": pl.Series([], dtype=pl.Utf8),
            "weight": pl.Series([], dtype=pl.Float64),
        }
    )
