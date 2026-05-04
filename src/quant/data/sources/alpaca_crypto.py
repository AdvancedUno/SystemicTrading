"""Alpaca historical bars for crypto.

Why a separate source from the equity AlpacaBarSource?
- Different SDK client class (CryptoHistoricalDataClient vs StockHistoricalDataClient)
- Different request type (CryptoBarsRequest vs StockBarsRequest)
- Different symbol convention (Alpaca uses "BTC/USD", we use "BTC-USD")
- No API keys required for crypto historical data
- No SIP/IEX feed distinction
- US-allowed (unlike Binance public API which 451's US IPs)

Symbol mapping:
  Internal: "BTC-USD"
  Alpaca:   "BTC/USD"
We translate at the boundary.
"""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from tenacity import retry, stop_after_attempt, wait_exponential

from quant.data.sources.base import BarSource
from quant.data.types import BAR_COLUMNS, Asset, AssetClass, BarInterval, BarRequest
from quant.utils.logging import logger


def _interval_to_alpaca(interval: BarInterval) -> TimeFrame:
    mapping = {
        BarInterval.MIN_1: TimeFrame(1, TimeFrameUnit.Minute),
        BarInterval.MIN_5: TimeFrame(5, TimeFrameUnit.Minute),
        BarInterval.MIN_15: TimeFrame(15, TimeFrameUnit.Minute),
        BarInterval.HOUR_1: TimeFrame(1, TimeFrameUnit.Hour),
        BarInterval.DAY_1: TimeFrame(1, TimeFrameUnit.Day),
    }
    return mapping[interval]


def _to_alpaca_symbol(internal: str) -> str:
    """'BTC-USD' -> 'BTC/USD'."""
    base, quote = internal.split("-")
    return f"{base.upper()}/{quote.upper()}"


class AlpacaCryptoBarSource:
    """Historical crypto bars from Alpaca."""

    name = "alpaca_crypto"

    def __init__(self) -> None:
        # No API keys needed for crypto historical data
        self._client = CryptoHistoricalDataClient()

    def supports(self, asset: Asset, interval: BarInterval) -> bool:
        return asset.asset_class == AssetClass.CRYPTO

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def fetch_bars(self, request: BarRequest) -> pl.DataFrame:
        if not self.supports(request.asset, request.interval):
            raise ValueError(
                f"AlpacaCryptoBarSource does not support {request.asset.asset_class}"
            )

        alpaca_symbol = _to_alpaca_symbol(request.asset.symbol)
        logger.debug(
            f"Alpaca crypto fetch: {alpaca_symbol} {request.interval.value} "
            f"{request.start.isoformat()} -> {request.end.isoformat()}"
        )

        req = CryptoBarsRequest(
            symbol_or_symbols=alpaca_symbol,
            timeframe=_interval_to_alpaca(request.interval),
            start=request.start,
            end=request.end,
        )
        bars = self._client.get_crypto_bars(req)

        if not bars.data:
            return _empty_bar_df()
        pdf = bars.df  # MultiIndex (symbol, timestamp)
        if pdf.empty:
            return _empty_bar_df()

        df = pl.from_pandas(pdf.reset_index()).rename({"timestamp": "ts"})

        # Crypto trade_count and vwap may be missing/null in some bars.
        # Add columns if they don't exist.
        if "trade_count" not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Int64).alias("trade_count"))
        if "vwap" not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("vwap"))

        df = (
            df.with_columns(
                pl.col("ts").dt.convert_time_zone("UTC").dt.cast_time_unit("us"),
                # Translate symbol back to internal format
                pl.lit(request.asset.symbol).alias("symbol"),
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Float64),
                pl.col("trade_count").cast(pl.Int64),
                pl.col("vwap").cast(pl.Float64),
            )
            .select(list(BAR_COLUMNS))
            .sort(["symbol", "ts"])
        )
        return df


def _empty_bar_df() -> pl.DataFrame:
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
    src = AlpacaCryptoBarSource()
    asset = Asset(symbol="BTC-USD", asset_class=AssetClass.CRYPTO)
    req = BarRequest(
        asset=asset,
        interval=BarInterval.DAY_1,
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 2, 1, tzinfo=timezone.utc),
    )
    df = src.fetch_bars(req)
    print(df)
    print(f"Rows: {df.height}")
