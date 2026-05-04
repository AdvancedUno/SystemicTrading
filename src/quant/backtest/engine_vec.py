"""Vectorized backtest engine.

Fast research engine. Computes P&L on a bar-aligned grid using vectorized
Polars operations.

────────────────────────────────────────────────────────────────────────────
The math (per bar t)
────────────────────────────────────────────────────────────────────────────

Let:
    w_i(t)        = target weight for symbol i decided at bar t (post-lag)
    w_held_i(t)   = weight actually held going into bar t = w_i(t-1)
    r_i(t)        = simple return of symbol i during bar t = close(t)/close(t-1) - 1
    Δw_i(t)       = w_i(t) - w_held_i(t) = required trade

Per-bar gross portfolio return (before costs):
    R_gross(t) = sum_i  w_held_i(t) * r_i(t)
              = sum_i  w_i(t-1) * r_i(t)

Per-bar turnover:
    TO(t) = sum_i  |Δw_i(t)| = sum_i  |w_i(t) - w_i(t-1)|

Per-bar cost as fraction of NAV:
    cost(t) = sum_i  |Δw_i(t)| * c_i(t)
where c_i(t) is the per-asset cost (commission + half-spread + impact)
expressed as a fraction.

Per-bar net return:
    R_net(t) = R_gross(t) - cost(t)

────────────────────────────────────────────────────────────────────────────
Critical timing
────────────────────────────────────────────────────────────────────────────
Weights are LAGGED by one bar before computing P&L. That is, the strategy
generates weights at bar t (using info up to t), but the P&L from those
weights is realized in bar t+1's return. This matches what would happen
in live trading: you see today's close, decide, and the new position is
in place starting tomorrow.

The Strategy class already applies a lag of 1 to its features, so we
think of "the weight at bar t" as already lagged-friendly. Here we apply
ONE more shift to align position with next-period return:

    pnl_contrib_i(t) = w_i(t-1) * r_i(t)

This double-lag is the most common source of subtle bugs in vectorized
backtests. We test it explicitly.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl

from quant.backtest.costs import CostConfig, estimate_cost
from quant.backtest.result import BacktestResult
from quant.data.types import AssetClass
from quant.utils.logging import logger


def _to_wide(panel: pl.DataFrame, value_col: str) -> pl.DataFrame:
    """Pivot a long-format (ts, symbol, value_col) frame to wide (ts × symbol)."""
    return panel.pivot(values=value_col, index="ts", on="symbol").sort("ts")


def run_vectorized(
    *,
    panel: pl.DataFrame,
    weights: pl.DataFrame,
    asset_class: AssetClass = AssetClass.EQUITY,
    cost_config: CostConfig | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> BacktestResult:
    """Run a vectorized backtest.

    Args:
        panel: long-format bars, must include columns:
            ts, symbol, close, high, low, volume.
        weights: long-format target weights from a Strategy.
        asset_class: used to pick default CostConfig.
        cost_config: override the default cost config.
        start, end: optional date filter.

    Returns:
        BacktestResult.
    """
    if panel.is_empty() or weights.is_empty():
        return BacktestResult(
            returns=pl.DataFrame(),
            positions=pl.DataFrame(),
            meta={"engine": "vectorized", "asset_class": asset_class.value},
        )

    cost_config = cost_config or CostConfig.for_asset_class(asset_class)

    # Filter date range
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
            meta={"engine": "vectorized", "asset_class": asset_class.value},
        )

    panel = panel.sort(["symbol", "ts"])
    weights = weights.sort(["ts", "symbol"])

    # ----- Step 1: Build wide tables aligned on (ts × symbol) -----
    # We use the panel's union of (ts, symbol) as the master grid.
    close_w = _to_wide(panel.select("ts", "symbol", "close"), "close")
    high_w = _to_wide(panel.select("ts", "symbol", "high"), "high")
    low_w = _to_wide(panel.select("ts", "symbol", "low"), "low")
    volume_w = _to_wide(panel.select("ts", "symbol", "volume"), "volume")

    symbols = [c for c in close_w.columns if c != "ts"]
    if not symbols:
        return BacktestResult(
            returns=pl.DataFrame(),
            positions=pl.DataFrame(),
            meta={"engine": "vectorized"},
        )

    # Weights wide. Symbols missing from weights mean weight=0.
    weight_w = (
        weights.pivot(values="weight", index="ts", on="symbol")
        .sort("ts")
    )
    # Align weight_w to close_w on ts: left-join, then forward-fill weights
    # (so we hold the position between rebalance dates).
    weight_w = (
        close_w.select("ts").join(weight_w, on="ts", how="left")
    )
    # Add any missing symbol columns with nulls
    for sym in symbols:
        if sym not in weight_w.columns:
            weight_w = weight_w.with_columns(pl.lit(None).cast(pl.Float64).alias(sym))
    # Forward-fill within each column, then fill nulls with 0
    weight_w = weight_w.select(
        ["ts"] + [pl.col(s).forward_fill().fill_null(0.0).alias(s) for s in symbols]
    )

    # ----- Step 2: Per-bar returns, on the grid -----
    # r_i(t) = close_i(t) / close_i(t-1) - 1
    ret_w = close_w.select(
        ["ts"]
        + [
            (pl.col(s) / pl.col(s).shift(1) - 1.0).fill_null(0.0).alias(s)
            for s in symbols
        ]
    )

    # ----- Step 3: Compute P&L -----
    # Held weight at bar t = weight at bar t-1 (the second lag)
    held_w = weight_w.select(
        ["ts"] + [pl.col(s).shift(1).fill_null(0.0).alias(s) for s in symbols]
    )

    # Per-symbol per-bar P&L contribution
    weight_arr = held_w.select(symbols).to_numpy()
    ret_arr = ret_w.select(symbols).to_numpy()
    pnl_per_sym = weight_arr * ret_arr  # shape (T, S)
    gross_ret = pnl_per_sym.sum(axis=1)  # shape (T,)

    # ----- Step 4: Turnover and per-bar costs -----
    # Δw_i(t) = w_i(t) - w_i(t-1). Use the actual (unshifted) weights here.
    weight_now = weight_w.select(symbols).to_numpy()
    weight_prev = np.vstack([np.zeros((1, len(symbols))), weight_now[:-1]])
    delta_w = weight_now - weight_prev
    abs_delta_w = np.abs(delta_w)
    turnover = abs_delta_w.sum(axis=1)
    gross_exposure = np.abs(weight_now).sum(axis=1)

    # Per-asset cost in bps requires per-asset numbers (vol, ADV).
    # We compute realized vol of returns and ADV of dollar volume from the
    # wide tables, both lagged by 1 bar.
    rv_w = ret_w.select(
        ["ts"] + [pl.col(s).rolling_std(window_size=20).shift(1).alias(s) for s in symbols]
    )
    dv_w = pl.DataFrame({"ts": close_w["ts"]}).hstack(
        close_w.select(symbols) * volume_w.select(symbols)
    )
    adv_w = dv_w.select(
        ["ts"]
        + [pl.col(s).rolling_mean(window_size=20).shift(1).alias(s) for s in symbols]
    )

    rv_arr = rv_w.select(symbols).to_numpy()
    adv_arr = adv_w.select(symbols).to_numpy()
    high_arr = high_w.select(symbols).to_numpy()
    low_arr = low_w.select(symbols).to_numpy()
    close_arr = close_w.select(symbols).to_numpy()

    # NAV proxy: gross dollar notional traded = |Δw_i| * NAV. For relative
    # P&L we don't need an absolute NAV — the per-asset cost is in bps of
    # *its* trade notional, and the contribution to portfolio return is
    # cost_bps * |Δw_i| / 10000.
    cost_bps_per_sym = np.zeros_like(weight_now)
    n_t = weight_now.shape[0]
    n_s = weight_now.shape[1]

    # Vectorize the cost calc per (t, s) — simple loop is fine since
    # n_t * n_s is typically <100k cells.
    for t in range(n_t):
        for s in range(n_s):
            if abs_delta_w[t, s] < 1e-12:
                continue
            high_v = high_arr[t, s]
            low_v = low_arr[t, s]
            close_v = close_arr[t, s]
            rv_v = rv_arr[t, s]
            adv_v = adv_arr[t, s]
            if (
                close_v is None or np.isnan(close_v) or close_v <= 0
                or rv_v is None or np.isnan(rv_v)
                or adv_v is None or np.isnan(adv_v)
            ):
                continue
            # Notional traded ~ |Δw_i| * NAV. We use NAV=1, so notional=|Δw_i|.
            # That makes participation = |Δw_i| / adv_dollars — valid only if
            # NAV is interpreted as 1 unit. For a real NAV scale, multiply.
            cb = estimate_cost(
                config=cost_config,
                notional=float(abs_delta_w[t, s]),
                bar_high=float(high_v) if not np.isnan(high_v) else float(close_v),
                bar_low=float(low_v) if not np.isnan(low_v) else float(close_v),
                bar_close=float(close_v),
                daily_volatility=float(rv_v),
                adv_dollars=float(adv_v) if adv_v > 0 else 1e9,
            )
            cost_bps_per_sym[t, s] = cb.total_bps

    # cost contribution to portfolio return = cost_bps * |Δw_i| / 10000
    cost_contrib = (cost_bps_per_sym / 10_000.0) * abs_delta_w
    cost_per_bar = cost_contrib.sum(axis=1)
    net_ret = gross_ret - cost_per_bar

    logger.debug(
        f"Vectorized BT: {n_t} bars, gross={gross_ret.sum():.4f}, "
        f"cost={cost_per_bar.sum():.4f}, net={net_ret.sum():.4f}"
    )

    # ----- Step 5: Pack output -----
    ts_col = close_w["ts"]
    returns_df = pl.DataFrame(
        {
            "ts": ts_col,
            "gross_return": gross_ret,
            "net_return": net_ret,
            "turnover": turnover,
            "gross_exposure": gross_exposure,
            "cost": cost_per_bar,
        }
    )

    # Long-format positions
    positions_rows = []
    ts_list = ts_col.to_list()
    for t_idx, t in enumerate(ts_list):
        for s_idx, sym in enumerate(symbols):
            w = weight_now[t_idx, s_idx]
            pnl = pnl_per_sym[t_idx, s_idx]
            if abs(w) > 1e-9 or abs(pnl) > 1e-12:
                positions_rows.append(
                    {
                        "ts": t,
                        "symbol": sym,
                        "weight": float(w),
                        "pnl_contribution": float(pnl),
                    }
                )
    positions_df = (
        pl.DataFrame(positions_rows)
        if positions_rows
        else pl.DataFrame({"ts": [], "symbol": [], "weight": [], "pnl_contribution": []})
    )

    return BacktestResult(
        returns=returns_df,
        positions=positions_df,
        meta={
            "engine": "vectorized",
            "asset_class": asset_class.value,
            "cost_config": {
                "commission_rate": cost_config.commission_rate,
                "spread_fraction": cost_config.spread_fraction,
                "impact_k": cost_config.impact_k,
            },
            "n_symbols": len(symbols),
        },
    )
