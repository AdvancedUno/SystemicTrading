"""Data source protocol — every market data provider implements this."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl

from quant.data.types import Asset, BarInterval, BarRequest


@runtime_checkable
class BarSource(Protocol):
    """A source of historical OHLCV bars.

    Implementations: AlpacaBarSource, YFinanceBarSource, BinanceBarSource, ...

    Contract:
    - `fetch_bars` returns a Polars DataFrame with the canonical schema
      defined in `quant.data.types.BAR_COLUMNS`.
    - Timestamps are bar OPEN times, in UTC.
    - Returned data is sorted by (symbol, ts) ascending.
    - Missing bars are simply absent rows; no forward-fill at this layer.
    - If a symbol has no data, return an empty DataFrame (not an error).
    """

    name: str  # e.g. "alpaca", "yfinance", "binance"

    def fetch_bars(self, request: BarRequest) -> pl.DataFrame:
        """Fetch bars for a single asset over a time range."""
        ...

    def supports(self, asset: Asset, interval: BarInterval) -> bool:
        """Whether this source can serve the given asset+interval."""
        ...
