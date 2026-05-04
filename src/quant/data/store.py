"""Local Parquet bar storage.

Layout on disk:
    data/processed/bars/
        equity/
            AAPL/
                interval=1d/year=2024/data.parquet
                interval=1m/year=2024/data.parquet
        crypto/
            BTC-USD/
                interval=1h/year=2024/data.parquet

Why Parquet + this layout:
- Parquet is columnar, compressed, and Polars/DuckDB read it natively
- Partitioning by year keeps individual files small and readable in chunks
- The (interval, year) partitioning lets us trivially query a date range
  for a symbol without reading everything

Idempotency:
- `write_bars` deduplicates within (symbol, ts) so re-fetching overlapping
  ranges doesn't corrupt the store.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from quant.data.types import Asset, BarInterval
from quant.utils.config import settings
from quant.utils.logging import logger


class ParquetBarStore:
    """Append-and-merge bar store backed by partitioned Parquet."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or settings.processed_dir) / "bars"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, asset: Asset, interval: BarInterval, year: int) -> Path:
        return (
            self.root
            / asset.asset_class.value
            / asset.symbol
            / f"interval={interval.value}"
            / f"year={year}"
            / "data.parquet"
        )

    def write_bars(
        self, asset: Asset, interval: BarInterval, df: pl.DataFrame
    ) -> int:
        """Write bars, merging with existing data and dedup'ing on (symbol, ts).

        Returns number of new rows added.
        """
        if df.is_empty():
            return 0
        # Split by year and merge into each partition
        df = df.with_columns(pl.col("ts").dt.year().alias("_year"))
        new_total = 0
        for year_df in df.partition_by("_year"):
            year = int(year_df["_year"][0])
            year_df = year_df.drop("_year")
            path = self._path(asset, interval, year)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                existing = pl.read_parquet(path)
                combined = (
                    pl.concat([existing, year_df])
                    .unique(subset=["symbol", "ts"], keep="last")
                    .sort(["symbol", "ts"])
                )
                new_count = combined.height - existing.height
            else:
                combined = year_df.unique(subset=["symbol", "ts"]).sort(
                    ["symbol", "ts"]
                )
                new_count = combined.height
            combined.write_parquet(path, compression="zstd")
            new_total += max(new_count, 0)
        logger.debug(
            f"Wrote {new_total} new bars for {asset.key} {interval.value}"
        )
        return new_total

    def read_bars(
        self,
        asset: Asset,
        interval: BarInterval,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        """Read bars for an asset, optionally filtered to a date range."""
        sym_dir = self.root / asset.asset_class.value / asset.symbol / f"interval={interval.value}"
        if not sym_dir.exists():
            return pl.DataFrame()
        # Glob all year partitions, lazily
        paths = sorted(sym_dir.glob("year=*/data.parquet"))
        if not paths:
            return pl.DataFrame()

        lf = pl.scan_parquet([str(p) for p in paths])
        if start is not None:
            lf = lf.filter(pl.col("ts") >= start)
        if end is not None:
            lf = lf.filter(pl.col("ts") < end)
        return lf.sort("ts").collect()

    def coverage(
        self, asset: Asset, interval: BarInterval
    ) -> tuple[datetime, datetime] | None:
        """Return (min_ts, max_ts) of stored bars, or None if empty."""
        df = self.read_bars(asset, interval)
        if df.is_empty():
            return None
        return df["ts"].min(), df["ts"].max()
