"""Ingestion service — fetch from sources, persist to store.

Routes asset+interval to the appropriate BarSource, handles incremental
updates (only fetch what's missing), and tracks coverage.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from quant.data.sources.alpaca import AlpacaBarSource
from quant.data.sources.alpaca_crypto import AlpacaCryptoBarSource
from quant.data.sources.base import BarSource
from quant.data.sources.binance import BinanceBarSource  # noqa: F401  # kept for opt-in use
from quant.data.store import ParquetBarStore
from quant.data.types import Asset, AssetClass, BarInterval, BarRequest
from quant.utils.logging import logger


class Ingestor:
    """Coordinates BarSources and a ParquetBarStore."""

    def __init__(
        self,
        store: ParquetBarStore | None = None,
        sources: dict[AssetClass, BarSource] | None = None,
    ) -> None:
        self.store = store or ParquetBarStore()
        if sources is None:
            sources = {}
            try:
                sources[AssetClass.EQUITY] = AlpacaBarSource()
            except RuntimeError as e:
                logger.warning(f"Alpaca source unavailable: {e}")
            sources[AssetClass.CRYPTO] = AlpacaCryptoBarSource()
        self.sources = sources

    def _source_for(self, asset: Asset) -> BarSource:
        src = self.sources.get(asset.asset_class)
        if src is None:
            raise RuntimeError(
                f"No source configured for asset class {asset.asset_class}"
            )
        return src

    def ingest(
        self,
        asset: Asset,
        interval: BarInterval,
        start: datetime,
        end: datetime,
        *,
        incremental: bool = True,
    ) -> int:
        """Fetch+store bars. If incremental, skip ranges already covered.

        Returns number of new rows added.
        """
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("start and end must be timezone-aware (UTC)")

        # Incremental: only fetch from after our latest stored ts
        if incremental:
            cov = self.store.coverage(asset, interval)
            if cov is not None:
                _, latest = cov
                # Re-fetch the last bar in case it was incomplete (live bar)
                fetch_from = latest - timedelta(seconds=interval.seconds)
                if fetch_from > start:
                    start = fetch_from

        if start >= end:
            logger.info(f"{asset.key} {interval.value}: nothing to fetch")
            return 0

        src = self._source_for(asset)
        req = BarRequest(asset=asset, interval=interval, start=start, end=end)
        logger.info(
            f"Ingesting {asset.key} {interval.value} via {src.name}: "
            f"{start.isoformat()} -> {end.isoformat()}"
        )
        df = src.fetch_bars(req)
        if df.is_empty():
            logger.warning(f"{asset.key} {interval.value}: empty result")
            return 0
        return self.store.write_bars(asset, interval, df)

    def ingest_universe(
        self,
        assets: list[Asset],
        interval: BarInterval,
        start: datetime,
        end: datetime | None = None,
        *,
        incremental: bool = True,
    ) -> dict[str, int]:
        """Ingest a list of assets at one interval. Returns {asset_key: rows_added}."""
        end = end or datetime.now(timezone.utc)
        results: dict[str, int] = {}
        for asset in assets:
            try:
                n = self.ingest(asset, interval, start, end, incremental=incremental)
                results[asset.key] = n
            except Exception as e:  # noqa: BLE001 — we log and continue
                logger.exception(f"Failed to ingest {asset.key}: {e}")
                results[asset.key] = -1
        return results
