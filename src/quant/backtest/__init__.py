"""Backtesting infrastructure."""

from quant.backtest.costs import CostBreakdown, CostConfig, estimate_cost
from quant.backtest.cv import Split, kfold_purged_splits, walk_forward_splits
from quant.backtest.engine_event import run_event_driven
from quant.backtest.engine_vec import run_vectorized
from quant.backtest.metrics import (
    PerfStats,
    compute_stats,
    drawdown_series,
    equity_curve,
)
from quant.backtest.result import BacktestResult

__all__ = [
    "CostBreakdown",
    "CostConfig",
    "estimate_cost",
    "Split",
    "walk_forward_splits",
    "kfold_purged_splits",
    "PerfStats",
    "compute_stats",
    "equity_curve",
    "drawdown_series",
    "BacktestResult",
    "run_vectorized",
    "run_event_driven",
]
