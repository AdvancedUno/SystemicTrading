"""Shared output type for backtest engines.

Both vectorized and event-driven engines produce a BacktestResult.
Downstream code (reporting, plotting, comparison) operates on this type
without caring which engine produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import polars as pl

from quant.backtest.metrics import PerfStats, compute_stats


@dataclass
class BacktestResult:
    """Output of a single backtest run.

    Schema:
        returns: long-format DataFrame [ts, gross_return, net_return, turnover, gross_exposure]
            One row per timestamp. gross_return is before costs, net is after.
            turnover is sum |w_t - w_{t-1}|.
        positions: long-format DataFrame [ts, symbol, weight, pnl_contribution]
            Per-bar position state and contribution to that bar's return.
            Useful for attribution.
        meta: free-form dict — strategy params, cost config, run timestamp, etc.
    """

    returns: pl.DataFrame
    positions: pl.DataFrame
    meta: dict = field(default_factory=dict)

    @property
    def n_periods(self) -> int:
        return self.returns.height

    @property
    def start(self) -> datetime | None:
        if self.returns.is_empty():
            return None
        return self.returns["ts"].min()

    @property
    def end(self) -> datetime | None:
        if self.returns.is_empty():
            return None
        return self.returns["ts"].max()

    def stats(self, periods_per_year: int = 252) -> PerfStats:
        """Compute performance stats on net returns."""
        if self.returns.is_empty():
            return compute_stats(np.array([]))
        return compute_stats(
            self.returns["net_return"].to_numpy(),
            periods_per_year=periods_per_year,
            turnover=self.returns["turnover"].to_numpy()
            if "turnover" in self.returns.columns
            else None,
        )

    def stats_gross(self, periods_per_year: int = 252) -> PerfStats:
        """Performance stats on gross (pre-cost) returns — useful for diagnosing cost drag."""
        if self.returns.is_empty():
            return compute_stats(np.array([]))
        return compute_stats(
            self.returns["gross_return"].to_numpy(),
            periods_per_year=periods_per_year,
        )

    def attribution(self) -> pl.DataFrame:
        """Per-symbol total contribution to P&L across the whole run."""
        if self.positions.is_empty():
            return pl.DataFrame({"symbol": [], "total_pnl": [], "n_active_periods": []})
        return (
            self.positions
            .group_by("symbol")
            .agg(
                pl.col("pnl_contribution").sum().alias("total_pnl"),
                (pl.col("weight").abs() > 1e-9).sum().alias("n_active_periods"),
            )
            .sort("total_pnl", descending=True)
        )

    def __repr__(self) -> str:
        if self.returns.is_empty():
            return "BacktestResult(empty)"
        s = self.stats()
        return (
            f"BacktestResult(n={self.n_periods}, "
            f"{self.start.date()} -> {self.end.date()}, "
            f"sharpe={s.sharpe:.2f}, ret_ann={s.annual_return:.2%}, "
            f"max_dd={s.max_drawdown:.2%})"
        )
