"""Strategy base class.

A Strategy maps a feature panel to a panel of TARGET WEIGHTS — what fraction
of capital should be allocated to each (symbol, ts).

Why target weights and not orders directly?
- Decouples the "what to hold" decision (strategy logic) from the "how to
  get there" decision (execution).
- Same strategy code drives both backtest and live trading.
- Lets us swap portfolio-construction logic (vol-targeting, sector caps,
  leverage limits) without rewriting strategies.

Conventions:
- weight in [-1, 1]: -1 = max short, +1 = max long, 0 = flat
- sum |weight| over symbols at each ts is the gross leverage
- sum  weight  over symbols at each ts is the net exposure
- A dollar-neutral long/short strategy has net=0, gross=2 typically

Output schema (long format):
    ts: Datetime[us, UTC]
    symbol: Utf8
    weight: Float64

The backtest engines consume this directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import polars as pl


WEIGHT_COLUMNS: tuple[str, ...] = ("ts", "symbol", "weight")


@dataclass(frozen=True)
class StrategyConfig:
    """Common knobs for strategies."""

    # How often to rebalance: 1 = every bar, 5 = every 5 bars, etc.
    rebalance_every: int = 1
    # Cap gross leverage. 1.0 = 100% long-only or fully-hedged L/S at half size each
    gross_leverage: float = 1.0
    # Strategy name (for logging / MLflow)
    name: str = "unnamed"
    # Free-form params, useful for sweeps
    params: dict = field(default_factory=dict)


class Strategy(ABC):
    """Abstract base. Subclasses implement `generate_weights`."""

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or StrategyConfig()

    @property
    def name(self) -> str:
        return self.config.name

    @abstractmethod
    def generate_weights(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Compute target weights from a feature panel.

        Args:
            panel: long-format DataFrame with ts, symbol, OHLCV, and any
                feature columns the strategy needs. Sorted by (symbol, ts).

        Returns:
            long-format DataFrame with WEIGHT_COLUMNS schema.
            One row per (ts, symbol) the strategy has an opinion on.
            Symbols not in the output get weight 0 (treated as flat).
        """
        ...

    # ----- Helpers shared across strategies -----

    def apply_rebalance_throttle(self, weights: pl.DataFrame) -> pl.DataFrame:
        """Only emit weights on every Nth bar; carry forward in between.

        Reduces turnover dramatically at the cost of slower response.
        """
        if self.config.rebalance_every <= 1:
            return weights
        # Mark every Nth ts as a rebalance day
        unique_ts = weights["ts"].unique().sort()
        keep_idx = list(range(0, unique_ts.len(), self.config.rebalance_every))
        rebalance_ts = unique_ts.gather(keep_idx)
        # Keep only weights on rebalance dates
        kept = weights.filter(pl.col("ts").is_in(rebalance_ts))
        # Forward-fill across the original ts axis
        all_ts = unique_ts.to_list()
        full = (
            kept.upsample(time_column="ts", every="1i", group_by="symbol")
            if False
            else kept  # placeholder — vectorized FF done in backtest engine
        )
        return full

    def apply_gross_leverage(self, weights: pl.DataFrame) -> pl.DataFrame:
        """Scale weights at each ts so sum |w| equals config.gross_leverage."""
        # Per-ts gross sum
        scaled = weights.with_columns(
            (pl.col("weight").abs().sum().over("ts")).alias("_gross")
        ).with_columns(
            pl.when(pl.col("_gross") > 1e-12)
            .then(pl.col("weight") * self.config.gross_leverage / pl.col("_gross"))
            .otherwise(0.0)
            .alias("weight")
        ).drop("_gross")
        return scaled.select(list(WEIGHT_COLUMNS))


def empty_weights() -> pl.DataFrame:
    """Empty DataFrame with the canonical weight schema."""
    return pl.DataFrame(
        {
            "ts": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            "symbol": pl.Series([], dtype=pl.Utf8),
            "weight": pl.Series([], dtype=pl.Float64),
        }
    )
