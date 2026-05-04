"""Alpaca historical bars for US equities.

Free tier specifics (as of 2026):
- IEX feed (lower volume than SIP but free)
- Rate limits: 200 requests/min on free
- Historical data: ~5+ years of minute bars, ~25+ years of daily

We hit Alpaca via the official `alpaca-py` SDK and normalize to our schema.
"""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from tenacity import retry, stop_after_attempt, wait_exponential

from quant.data.sources.base import BarSource
from quant.data.types import BAR_COLUMNS, Asset, AssetClass, BarInterval, BarRequest
from quant.utils.config import settings
from quant.utils.logging import logger


def _interval_to_alpaca(interval: BarInterval) -> TimeFrame:
    """Map our BarInterval to Alpaca's TimeFrame object."""
    mapping = {
        BarInterval.MIN_1: TimeFrame(1, TimeFrameUnit.Minute),
        BarInterval.MIN_5: TimeFrame(5, TimeFrameUnit.Minute),
        BarInterval.MIN_15: TimeFrame(15, TimeFrameUnit.Minute),
        BarInterval.HOUR_1: TimeFrame(1, TimeFrameUnit.Hour),
        BarInterval.DAY_1: TimeFrame(1, TimeFrameUnit.Day),
    }
    return mapping[interval]


class AlpacaBarSource:
    """Historical bars from Alpaca for US equities."""

    name = "alpaca"

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
    ) -> None:
        key = api_key or settings.alpaca_api_key
        secret = api_secret or settings.alpaca_api_secret
        if not key or not secret:
            raise RuntimeError(
                "Alpaca credentials missing. Set ALPACA_API_KEY and "
                "ALPACA_API_SECRET in your .env file."
            )
        self._client = StockHistoricalDataClient(api_key=key, secret_key=secret)

    def supports(self, asset: Asset, interval: BarInterval) -> bool:
        return asset.asset_class == AssetClass.EQUITY

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def fetch_bars(self, request: BarRequest) -> pl.DataFrame:
        if not self.supports(request.asset, request.interval):
            raise ValueError(
                f"Alpaca does not support {request.asset.asset_class} "
                f"@ {request.interval}"
            )

        logger.debug(
            f"Alpaca fetch: {request.asset.symbol} {request.interval.value} "
            f"{request.start.isoformat()} -> {request.end.isoformat()}"
        )

        req = StockBarsRequest(
            symbol_or_symbols=request.asset.symbol,
            timeframe=_interval_to_alpaca(request.interval),
            start=request.start,
            end=request.end,
            adjustment="all",
            feed="iex",
        )
        bars = self._client.get_stock_bars(req)

        # alpaca-py returns a BarSet; convert via .df (a pandas frame) then
        # to polars for a uniform downstream type.
        if not bars.data:
            return _empty_bar_df()
        pdf = bars.df  # MultiIndex (symbol, timestamp)
        if pdf.empty:
            return _empty_bar_df()

        df = (
            pl.from_pandas(pdf.reset_index())
            .rename({"timestamp": "ts"})
            .with_columns(
                pl.col("ts").dt.convert_time_zone("UTC").dt.cast_time_unit("us"),
                pl.col("symbol").cast(pl.Utf8),
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
    # Quick sanity check when run directly:
    # `python -m quant.data.sources.alpaca`
    src = AlpacaBarSource()
    asset = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
    req = BarRequest(
        asset=asset,
        interval=BarInterval.DAY_1,
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 2, 1, tzinfo=timezone.utc),
    )
    df = src.fetch_bars(req)
    print(df)
    print(f"Rows: {df.height}")
