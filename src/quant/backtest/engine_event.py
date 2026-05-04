"""Event-driven backtest engine.

Realistic engine. Walks bars in chronological order, simulating order
flow with explicit fill timing and slippage.

────────────────────────────────────────────────────────────────────────────
Bar timing model
────────────────────────────────────────────────────────────────────────────
End-of-bar t (close prints):
    1. Strategy generates target weights using info up to t.
Start-of-bar t+1 (open prints):
    2. Orders fill at open_price + sign(trade) * open_price * total_cost_frac
During bar t+1:
    3. Mark-to-market at every close.

Why open-of-next-bar fills?
    A vectorized engine that fills at the close of the decision bar is
    optimistic — you can't both look at the close and trade at it in real
    life. Real market orders submitted at the close fill at the next open
    (after an overnight gap for equities, or just the next bar for crypto).

────────────────────────────────────────────────────────────────────────────
NAV tracking
────────────────────────────────────────────────────────────────────────────
NAV(t) = cash(t) + sum_i shares_i * close_i(t)
Net return = NAV(t) / NAV(t-1) - 1.

We run the loop twice on internal computation: once with the user's costs
to get net returns, and once with zero costs to get the gross returns.
The difference is the per-bar cost contribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl

from quant.backtest.costs import CostConfig, estimate_cost
from quant.backtest.result import BacktestResult
from quant.data.types import AssetClass


_ZERO_COST = CostConfig(
    commission_rate=0.0,
    spread_fraction=0.0,
    min_spread_bps=0.0,
    impact_k=0.0,
    max_impact_bps=0.0,
)


@dataclass
class _SimOutput:
    returns: pl.DataFrame
    positions: pl.DataFrame


def run_event_driven(
    *,
    panel: pl.DataFrame,
    weights: pl.DataFrame,
    asset_class: AssetClass = AssetClass.EQUITY,
    cost_config: CostConfig | None = None,
    initial_capital: float = 1.0,
    start: datetime | None = None,
    end: datetime | None = None,
) -> BacktestResult:
    """Run an event-driven backtest. See module docstring for timing model."""
    cost_config = cost_config or CostConfig.for_asset_class(asset_class)

    if panel.is_empty() or weights.is_empty():
        return BacktestResult(
            returns=pl.DataFrame(),
            positions=pl.DataFrame(),
            meta={"engine": "event_driven", "asset_class": asset_class.value},
        )

    if start is not None:
        panel = panel.filter(pl.col("ts") >= start)
        weights = weights.filter(pl.col("ts") >= start)
    if end is not None:
        panel = panel.filter(pl.col("ts") < end)
        weights = weights.filter(pl.col("ts") < end)
    if panel.is_empty():
        return BacktestResult(
            returns=pl.DataFrame(),
            positions=pl.DataFrame(),
            meta={"engine": "event_driven"},
        )

    # Two paths: with cost (the real result) and without (for gross attribution).
    net_run = _simulate(
        panel=panel,
        weights=weights,
        cost_config=cost_config,
        initial_capital=initial_capital,
    )
    gross_run = _simulate(
        panel=panel,
        weights=weights,
        cost_config=_ZERO_COST,
        initial_capital=initial_capital,
    )

    # Splice gross_return / cost into the net returns frame
    if not net_run.returns.is_empty():
        gross_returns = gross_run.returns["net_return"].to_list()
        returns_df = (
            net_run.returns
            .with_columns(pl.Series("gross_return", gross_returns))
            .with_columns((pl.col("gross_return") - pl.col("net_return")).alias("cost"))
        )
    else:
        returns_df = net_run.returns

    return BacktestResult(
        returns=returns_df,
        positions=net_run.positions,
        meta={
            "engine": "event_driven",
            "asset_class": asset_class.value,
            "initial_capital": initial_capital,
            "cost_config": {
                "commission_rate": cost_config.commission_rate,
                "spread_fraction": cost_config.spread_fraction,
                "impact_k": cost_config.impact_k,
            },
        },
    )


def _simulate(
    *,
    panel: pl.DataFrame,
    weights: pl.DataFrame,
    cost_config: CostConfig,
    initial_capital: float,
) -> _SimOutput:
    """Inner simulation loop. Returns net P&L only — caller handles gross."""
    panel = panel.sort(["ts", "symbol"])
    timestamps = panel["ts"].unique().sort().to_list()
    symbols = panel["symbol"].unique().sort().to_list()
    n_t, n_s = len(timestamps), len(symbols)

    def _to_arr(col: str) -> np.ndarray:
        w = panel.pivot(values=col, index="ts", on="symbol").sort("ts")
        for c in symbols:
            if c not in w.columns:
                w = w.with_columns(pl.lit(None).cast(pl.Float64).alias(c))
        return w.select(symbols).to_numpy()

    open_a = _to_arr("open")
    high_a = _to_arr("high")
    low_a = _to_arr("low")
    close_a = _to_arr("close")
    vol_a = _to_arr("volume")

    weights_wide = (
        weights.pivot(values="weight", index="ts", on="symbol").sort("ts")
    )
    weights_wide = pl.DataFrame({"ts": timestamps}).join(
        weights_wide, on="ts", how="left"
    )
    for s in symbols:
        if s not in weights_wide.columns:
            weights_wide = weights_wide.with_columns(
                pl.lit(None).cast(pl.Float64).alias(s)
            )
    weights_wide = weights_wide.select(
        ["ts"] + [pl.col(s).forward_fill().fill_null(0.0).alias(s) for s in symbols]
    )
    target_w = weights_wide.select(symbols).to_numpy()

    # Lagged 20-bar realized vol and ADV for cost model
    ret_a = np.full_like(close_a, np.nan)
    ret_a[1:] = close_a[1:] / close_a[:-1] - 1.0
    rv_a = _rolling_std(ret_a, window=20)
    rv_a_lag = np.vstack([np.full((1, n_s), np.nan), rv_a[:-1]])
    dv_a = close_a * vol_a
    adv_a = _rolling_mean(dv_a, window=20)
    adv_a_lag = np.vstack([np.full((1, n_s), np.nan), adv_a[:-1]])

    # State
    shares = np.zeros(n_s)
    cash = initial_capital
    prev_nav = initial_capital
    last_target = np.zeros(n_s)

    rows_returns = []
    rows_positions = []

    for t_idx, ts in enumerate(timestamps):
        # Fill orders submitted at end of t-1 at the open of t
        if t_idx > 0:
            opens = open_a[t_idx]
            highs = high_a[t_idx]
            lows = low_a[t_idx]
            valid_opens = np.where(np.isnan(opens), 0.0, opens)
            nav_at_open = cash + float(np.sum(shares * valid_opens))

            with np.errstate(divide="ignore", invalid="ignore"):
                target_shares = np.where(
                    (~np.isnan(opens)) & (opens > 0),
                    last_target * nav_at_open / opens,
                    shares,
                )
            delta_shares = target_shares - shares

            for s_idx in range(n_s):
                ds = delta_shares[s_idx]
                if abs(ds) < 1e-12:
                    continue
                price = opens[s_idx]
                if np.isnan(price) or price <= 0:
                    continue
                hi = highs[s_idx] if not np.isnan(highs[s_idx]) else price
                lo = lows[s_idx] if not np.isnan(lows[s_idx]) else price
                rv_v = (
                    float(rv_a_lag[t_idx, s_idx])
                    if not np.isnan(rv_a_lag[t_idx, s_idx])
                    else 0.02
                )
                adv_v = (
                    float(adv_a_lag[t_idx, s_idx])
                    if not np.isnan(adv_a_lag[t_idx, s_idx]) and adv_a_lag[t_idx, s_idx] > 0
                    else 1e9
                )
                notional = abs(ds) * price
                cb = estimate_cost(
                    config=cost_config,
                    notional=notional,
                    bar_high=float(hi),
                    bar_low=float(lo),
                    bar_close=float(price),
                    daily_volatility=rv_v,
                    adv_dollars=adv_v,
                )
                fill_price = price * (1.0 + np.sign(ds) * cb.total_fraction)
                cash -= ds * fill_price
                shares[s_idx] += ds

        closes_t = close_a[t_idx]
        valid_closes = np.where(np.isnan(closes_t), 0.0, closes_t)
        position_value = float(np.sum(shares * valid_closes))
        nav = cash + position_value
        net_return = nav / prev_nav - 1.0 if prev_nav > 0 else 0.0
        gross_exp = (
            float(np.sum(np.abs(shares * valid_closes))) / max(nav, 1e-12)
        )
        turnover = (
            float(np.sum(np.abs(target_w[t_idx] - target_w[t_idx - 1])))
            if t_idx > 0
            else 0.0
        )

        rows_returns.append(
            {
                "ts": ts,
                "gross_return": float(net_return),  # placeholder
                "net_return": float(net_return),
                "turnover": float(turnover),
                "gross_exposure": float(gross_exp),
                "cost": 0.0,  # filled by caller
                "nav": float(nav),
            }
        )

        for s_idx in range(n_s):
            close_v = closes_t[s_idx] if not np.isnan(closes_t[s_idx]) else 0.0
            w_now = (shares[s_idx] * close_v) / max(nav, 1e-12)
            if abs(w_now) > 1e-9:
                rows_positions.append(
                    {
                        "ts": ts,
                        "symbol": symbols[s_idx],
                        "weight": float(w_now),
                        "pnl_contribution": 0.0,
                    }
                )

        last_target = target_w[t_idx].copy()
        prev_nav = nav

    returns_df = pl.DataFrame(rows_returns)
    positions_df = (
        pl.DataFrame(rows_positions)
        if rows_positions
        else pl.DataFrame({"ts": [], "symbol": [], "weight": [], "pnl_contribution": []})
    )
    return _SimOutput(returns=returns_df, positions=positions_df)


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    n_t, n_s = arr.shape
    out = np.full_like(arr, np.nan, dtype=np.float64)
    for s in range(n_s):
        col = arr[:, s]
        for t in range(window - 1, n_t):
            seg = col[t - window + 1 : t + 1]
            seg = seg[~np.isnan(seg)]
            if seg.size > 1:
                out[t, s] = float(np.std(seg, ddof=1))
    return out


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    n_t, n_s = arr.shape
    out = np.full_like(arr, np.nan, dtype=np.float64)
    for s in range(n_s):
        col = arr[:, s]
        for t in range(window - 1, n_t):
            seg = col[t - window + 1 : t + 1]
            seg = seg[~np.isnan(seg)]
            if seg.size > 0:
                out[t, s] = float(np.mean(seg))
    return out
