"""Binance historical klines for crypto.

Public klines endpoint requires no API key, no rate limit issues for
typical research use. We use the REST endpoint directly via httpx
rather than python-binance, because it's simpler and we control the
request fully.

Symbol convention:
- Our internal: "BTC-USD"
- Binance: "BTCUSDT" (USDT is the standard USD stablecoin pair on Binance)

We translate at the boundary.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import httpx
import polars as pl
from tenacity import retry, stop_after_attempt, wait_exponential

from quant.data.sources.base import BarSource
from quant.data.types import BAR_COLUMNS, Asset, AssetClass, BarInterval, BarRequest
from quant.utils.logging import logger

_BASE_URL = "https://api.binance.com/api/v3/klines"
_LIMIT = 1000  # max bars per request

_INTERVAL_MAP = {
    BarInterval.MIN_1: "1m",
    BarInterval.MIN_5: "5m",
    BarInterval.MIN_15: "15m",
    BarInterval.HOUR_1: "1h",
    BarInterval.DAY_1: "1d",
}


def _to_binance_symbol(internal: str) -> str:
    """'BTC-USD' -> 'BTCUSDT'. Binance uses USDT for USD pairs."""
    base, quote = internal.split("-")
    if quote.upper() == "USD":
        quote = "USDT"
    return f"{base.upper()}{quote.upper()}"


class BinanceBarSource:
    """Historical bars from Binance public klines endpoint."""

    name = "binance"

    def __init__(self, timeout: float = 30.0) -> None:
        self._client = httpx.Client(timeout=timeout)

    def supports(self, asset: Asset, interval: BarInterval) -> bool:
        return asset.asset_class == AssetClass.CRYPTO and interval in _INTERVAL_MAP

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _fetch_chunk(
        self, symbol: str, interval: str, start_ms: int, end_ms: int
    ) -> list[list]:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": _LIMIT,
        }
        r = self._client.get(_BASE_URL, params=params)
        r.raise_for_status()
        return r.json()

    def fetch_bars(self, request: BarRequest) -> pl.DataFrame:
        if not self.supports(request.asset, request.interval):
            raise ValueError(
                f"Binance does not support {request.asset.asset_class} "
                f"@ {request.interval}"
            )

        binance_symbol = _to_binance_symbol(request.asset.symbol)
        binance_interval = _INTERVAL_MAP[request.interval]
        bar_seconds = request.interval.seconds

        start_ms = int(request.start.timestamp() * 1000)
        end_ms = int(request.end.timestamp() * 1000)

        all_rows: list[list] = []
        cursor = start_ms
        while cursor < end_ms:
            chunk = self._fetch_chunk(binance_symbol, binance_interval, cursor, end_ms)
            if not chunk:
                break
            all_rows.extend(chunk)
            # Advance cursor past last bar's open time + one bar
            last_open_ms = chunk[-1][0]
            next_cursor = last_open_ms + bar_seconds * 1000
            if next_cursor <= cursor:  # safety
                break
            cursor = next_cursor
            # be polite — Binance is generous but no need to hammer
            time.sleep(0.05)

        logger.debug(
            f"Binance fetched {len(all_rows)} bars for {binance_symbol} "
            f"{binance_interval}"
        )

        if not all_rows:
            return _empty_bar_df()

        # Binance kline schema:
        # [open_time_ms, open, high, low, close, volume,
        #  close_time_ms, quote_volume, trade_count,
        #  taker_base_vol, taker_quote_vol, ignore]
        df = pl.DataFrame(
            {
                "ts": [row[0] for row in all_rows],
                "symbol": [request.asset.symbol] * len(all_rows),
                "open": [float(row[1]) for row in all_rows],
                "high": [float(row[2]) for row in all_rows],
                "low": [float(row[3]) for row in all_rows],
                "close": [float(row[4]) for row in all_rows],
                "volume": [float(row[5]) for row in all_rows],
                "trade_count": [int(row[8]) for row in all_rows],
                "quote_volume": [float(row[7]) for row in all_rows],
            }
        ).with_columns(
            pl.from_epoch(pl.col("ts"), time_unit="ms")
            .dt.replace_time_zone("UTC")
            .dt.cast_time_unit("us"),
            # vwap not provided by Binance — approximate as quote_volume / volume
            (pl.col("quote_volume") / pl.col("volume")).alias("vwap"),
        ).select(list(BAR_COLUMNS)).sort(["symbol", "ts"])

        return df


def _empty_bar_df() -> pl.DataFrame:
    """Empty DataFrame with the canonical bar schema."""
    return pl.DataFrame(
        {
            "ts": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            "symbol": pl.Series([], dtype=pl.Utf8),
            "open": pl.Series([], dtype=pl.Float64),
            "high": pl.Series([], dtype=pl.Float64),
            "low": pl.Series([], dtype=pl.Float64),
            "close": pl.Series([], dtype=pl.Float64),
            "volume": pl.Series([], dtype=pl.Float64),
            "trade_count": pl.Series([], dtype=pl.Int64),
            "vwap": pl.Series([], dtype=pl.Float64),
        }
    )


if __name__ == "__main__":
    src = BinanceBarSource()
    asset = Asset(symbol="BTC-USD", asset_class=AssetClass.CRYPTO)
    req = BarRequest(
        asset=asset,
        interval=BarInterval.HOUR_1,
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 8, tzinfo=timezone.utc),
    )
    df = src.fetch_bars(req)
    print(df)
    print(f"Rows: {df.height}")
