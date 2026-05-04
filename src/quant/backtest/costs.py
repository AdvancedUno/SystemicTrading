"""Transaction cost models.

A cost model takes a proposed trade (symbol, size, price, market state) and
returns an estimated execution price plus explicit fees. This is what the
backtester uses to compute realistic P&L.

We model three components separately, because they behave differently
and respond differently to trade size:

────────────────────────────────────────────────────────────────────────────
1. Commission (fixed per-trade or per-share)
────────────────────────────────────────────────────────────────────────────
- Alpaca equities: $0 commission (regulatory fees ~0.001% are tiny — we ignore)
- Alpaca crypto: 0% maker, 0.15-0.25% taker depending on volume tier.
  We use 0.25% as a conservative default.
- Binance crypto (if you ever use it): 0.10% taker.

────────────────────────────────────────────────────────────────────────────
2. Half-spread (fixed cost from crossing the bid-ask)
────────────────────────────────────────────────────────────────────────────
For market orders, you pay half the spread to cross. We don't have the
actual bid-ask in our daily bars, so we estimate it from intra-bar range:

  estimated_spread_bps ≈ 2 * (high - low) / close * spread_fraction

Where spread_fraction is the fraction of the daily range we attribute to
typical spread (usually 1-3%). The half-spread cost is then:

  half_spread_bps = spread_bps / 2

For very liquid US equities, this works out to ~0.5-2 bps. For less liquid
names or crypto, it can be 5-50 bps. We err conservative.

────────────────────────────────────────────────────────────────────────────
3. Market impact (variable, depends on trade size)
────────────────────────────────────────────────────────────────────────────
Market impact is what your trade itself does to the price. The empirical
literature (Almgren et al. 2005, BARRA, AQR, etc.) finds that impact scales
with the SQUARE ROOT of participation rate (not linearly):

  impact_bps = k * sigma_bps * sqrt(trade_size / ADV)

Where:
  k = a constant, typically 0.5 to 1.0 for equities. We default to 0.6.
  sigma_bps = the asset's daily volatility in basis points
  trade_size = shares (or dollar) being traded
  ADV = average daily volume (shares or dollar)

The square-root form is one of the most-replicated empirical findings in
quant finance. It says: doubling your trade size doesn't double your
cost — it raises it by a factor of sqrt(2) ≈ 1.41. This is why scaling
up a profitable strategy is hard: at some point impact eats your edge.

We compute ADV from a rolling 20-day mean of dollar volume.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quant.data.types import AssetClass


@dataclass(frozen=True, slots=True)
class CostConfig:
    """Tuning knobs for the cost model."""

    # Commission as fraction of notional (e.g. 0.0025 = 25 bps)
    commission_rate: float = 0.0
    # Fraction of daily range attributed to typical spread
    spread_fraction: float = 0.02
    # Floor on the spread estimate, in bps (so we don't go to zero on
    # super-tight ranges)
    min_spread_bps: float = 1.0
    # Square-root-impact coefficient
    impact_k: float = 0.6
    # Cap on impact in bps (sanity)
    max_impact_bps: float = 200.0

    @classmethod
    def for_asset_class(cls, ac: AssetClass) -> "CostConfig":
        """Reasonable defaults per asset class."""
        if ac == AssetClass.EQUITY:
            return cls(
                commission_rate=0.0,  # Alpaca free
                spread_fraction=0.015,
                min_spread_bps=0.5,
                impact_k=0.6,
            )
        elif ac == AssetClass.CRYPTO:
            return cls(
                commission_rate=0.0025,  # 25 bps Alpaca crypto taker
                spread_fraction=0.03,
                min_spread_bps=2.0,
                impact_k=0.8,  # crypto books are thinner
            )
        raise ValueError(f"Unknown asset class: {ac}")


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Itemized cost output."""

    commission_bps: float
    half_spread_bps: float
    impact_bps: float

    @property
    def total_bps(self) -> float:
        return self.commission_bps + self.half_spread_bps + self.impact_bps

    @property
    def total_fraction(self) -> float:
        return self.total_bps / 10_000.0


def estimate_cost(
    *,
    config: CostConfig,
    notional: float,
    bar_high: float,
    bar_low: float,
    bar_close: float,
    daily_volatility: float,
    adv_dollars: float,
) -> CostBreakdown:
    """Estimate cost of executing one trade, in basis points.

    Args:
        config: CostConfig for this asset class
        notional: |dollar notional| of the trade. Always positive.
        bar_high, bar_low, bar_close: from the bar where the trade executes
        daily_volatility: stdev of 1-bar log returns (e.g. realized_vol_20d)
        adv_dollars: rolling-mean of close * volume
    """
    if notional <= 0:
        return CostBreakdown(0.0, 0.0, 0.0)

    # 1. Commission
    commission_bps = config.commission_rate * 10_000.0

    # 2. Half-spread, estimated from intra-bar range
    spread_bps = (
        (bar_high - bar_low) / max(bar_close, 1e-12)
        * config.spread_fraction
        * 10_000.0
    )
    spread_bps = max(spread_bps, config.min_spread_bps)
    half_spread_bps = spread_bps / 2.0

    # 3. Market impact (square-root model)
    if adv_dollars <= 0 or daily_volatility <= 0:
        impact_bps = 0.0
    else:
        participation = notional / adv_dollars
        sigma_bps = daily_volatility * 10_000.0
        impact_bps = config.impact_k * sigma_bps * float(np.sqrt(participation))
        impact_bps = min(impact_bps, config.max_impact_bps)

    return CostBreakdown(
        commission_bps=commission_bps,
        half_spread_bps=half_spread_bps,
        impact_bps=impact_bps,
    )
