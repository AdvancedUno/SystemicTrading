"""Performance metrics.

All inputs assume DAILY returns unless noted. For other frequencies,
adjust `periods_per_year` (e.g., 24*365 for hourly crypto).

────────────────────────────────────────────────────────────────────────────
Definitions
────────────────────────────────────────────────────────────────────────────

Annualization:
  Returns:    R_ann = mean(r_daily) * periods_per_year
  Volatility: o_ann = stdev(r_daily) * sqrt(periods_per_year)

Sharpe (with rf=0):
  SR = R_ann / o_ann

Sortino (downside-vol denominator):
  o_down = stdev of negative returns (zeros for positive)
  Sortino = R_ann / (o_down * sqrt(periods_per_year))

Calmar:
  Calmar = R_ann / |max_drawdown|

Max drawdown:
  Equity curve E(t) = cumprod(1 + r). Running max M(t).
  Drawdown D(t) = E(t) / M(t) - 1, always ≤ 0.
  Max DD = min over t of D(t)

Hit rate:
  fraction of days where return > 0

Turnover (per period):
  mean over t of sum_i |w_i(t) - w_i(t-1)|
  Captures how much portfolio churn there is.

────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class PerfStats:
    n_periods: int
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    hit_rate: float
    best_day: float
    worst_day: float
    avg_turnover: float

    def to_dict(self) -> dict:
        return {
            "n_periods": self.n_periods,
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "annual_volatility": self.annual_volatility,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "calmar": self.calmar,
            "hit_rate": self.hit_rate,
            "best_day": self.best_day,
            "worst_day": self.worst_day,
            "avg_turnover": self.avg_turnover,
        }

    def __repr__(self) -> str:
        return (
            f"PerfStats(n={self.n_periods}, "
            f"ret_ann={self.annual_return:.2%}, "
            f"vol_ann={self.annual_volatility:.2%}, "
            f"sharpe={self.sharpe:.2f}, "
            f"sortino={self.sortino:.2f}, "
            f"max_dd={self.max_drawdown:.2%}, "
            f"calmar={self.calmar:.2f}, "
            f"hit_rate={self.hit_rate:.2%}, "
            f"turnover={self.avg_turnover:.2%})"
        )


def compute_stats(
    returns: np.ndarray | list[float] | pl.Series,
    *,
    periods_per_year: int = 252,
    turnover: np.ndarray | list[float] | pl.Series | None = None,
) -> PerfStats:
    """Compute performance stats from a sequence of period returns.

    Args:
        returns: per-period returns (e.g. daily). Length N.
        periods_per_year: 252 for equities, 365 for crypto daily, ...
        turnover: optional per-period turnover; if None, set to NaN.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size == 0:
        return _empty_stats()

    eq = np.cumprod(1.0 + r)
    total_ret = float(eq[-1] - 1.0)

    mean_p = float(np.mean(r))
    std_p = float(np.std(r, ddof=1)) if r.size > 1 else 0.0
    ann_ret = mean_p * periods_per_year
    ann_vol = std_p * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0

    neg = r[r < 0]
    down_std = float(np.std(neg, ddof=1)) if neg.size > 1 else 0.0
    sortino = ann_ret / (down_std * np.sqrt(periods_per_year)) if down_std > 1e-12 else 0.0

    # Max drawdown
    running_max = np.maximum.accumulate(eq)
    dd = eq / running_max - 1.0
    max_dd = float(dd.min())

    calmar = ann_ret / abs(max_dd) if max_dd < -1e-12 else 0.0
    hit = float((r > 0).mean())
    best = float(r.max())
    worst = float(r.min())

    if turnover is not None:
        t = np.asarray(turnover, dtype=float)
        t = t[~np.isnan(t)]
        avg_to = float(np.mean(t)) if t.size > 0 else float("nan")
    else:
        avg_to = float("nan")

    return PerfStats(
        n_periods=r.size,
        total_return=total_ret,
        annual_return=ann_ret,
        annual_volatility=ann_vol,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        calmar=calmar,
        hit_rate=hit,
        best_day=best,
        worst_day=worst,
        avg_turnover=avg_to,
    )


def equity_curve(returns: np.ndarray | list[float] | pl.Series) -> np.ndarray:
    """Cumulative compounded equity from per-period returns. Starts at 1.0."""
    r = np.asarray(returns, dtype=float)
    return np.cumprod(1.0 + r)


def drawdown_series(returns: np.ndarray | list[float] | pl.Series) -> np.ndarray:
    """Drawdown series — values in [-1, 0]."""
    eq = equity_curve(returns)
    running_max = np.maximum.accumulate(eq)
    return eq / running_max - 1.0


def _empty_stats() -> PerfStats:
    return PerfStats(
        n_periods=0,
        total_return=0.0,
        annual_return=0.0,
        annual_volatility=0.0,
        sharpe=0.0,
        sortino=0.0,
        max_drawdown=0.0,
        calmar=0.0,
        hit_rate=0.0,
        best_day=0.0,
        worst_day=0.0,
        avg_turnover=float("nan"),
    )
