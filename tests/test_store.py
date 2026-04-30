"""Tests for ParquetBarStore."""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from quant.data.store import ParquetBarStore
from quant.data.types import BAR_COLUMNS, Asset, AssetClass, BarInterval


@pytest.fixture
def store(tmp_path):
    return ParquetBarStore(root=tmp_path)


@pytest.fixture
def asset():
    return Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)


def _make_bars(symbol: str, ts_list: list[datetime]) -> pl.DataFrame:
    n = len(ts_list)
    return pl.DataFrame(
        {
            "ts": ts_list,
            "symbol": [symbol] * n,
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.5] * n,
            "volume": [1_000_000.0] * n,
            "trade_count": [5000] * n,
            "vwap": [100.25] * n,
        }
    ).with_columns(pl.col("ts").dt.cast_time_unit("us"))


def test_write_then_read(store, asset):
    bars = _make_bars(
        "AAPL",
        [
            datetime(2024, 1, 2, tzinfo=timezone.utc),
            datetime(2024, 1, 3, tzinfo=timezone.utc),
            datetime(2024, 1, 4, tzinfo=timezone.utc),
        ],
    )
    n = store.write_bars(asset, BarInterval.DAY_1, bars)
    assert n == 3
    out = store.read_bars(asset, BarInterval.DAY_1)
    assert out.height == 3
    assert list(out.columns) == list(BAR_COLUMNS)


def test_dedup_on_overlap(store, asset):
    bars1 = _make_bars(
        "AAPL",
        [
            datetime(2024, 1, 2, tzinfo=timezone.utc),
            datetime(2024, 1, 3, tzinfo=timezone.utc),
        ],
    )
    bars2 = _make_bars(
        "AAPL",
        [
            datetime(2024, 1, 3, tzinfo=timezone.utc),  # overlap
            datetime(2024, 1, 4, tzinfo=timezone.utc),
        ],
    )
    store.write_bars(asset, BarInterval.DAY_1, bars1)
    store.write_bars(asset, BarInterval.DAY_1, bars2)
    out = store.read_bars(asset, BarInterval.DAY_1)
    assert out.height == 3  # not 4 — dedup'd


def test_date_range_filter(store, asset):
    bars = _make_bars(
        "AAPL",
        [
            datetime(2024, 1, 2, tzinfo=timezone.utc),
            datetime(2024, 1, 3, tzinfo=timezone.utc),
            datetime(2024, 1, 4, tzinfo=timezone.utc),
        ],
    )
    store.write_bars(asset, BarInterval.DAY_1, bars)
    out = store.read_bars(
        asset,
        BarInterval.DAY_1,
        start=datetime(2024, 1, 3, tzinfo=timezone.utc),
        end=datetime(2024, 1, 4, tzinfo=timezone.utc),
    )
    assert out.height == 1


def test_year_partitioning(store, asset):
    """Bars across years go into separate partitions."""
    bars = _make_bars(
        "AAPL",
        [
            datetime(2023, 12, 29, tzinfo=timezone.utc),
            datetime(2024, 1, 2, tzinfo=timezone.utc),
        ],
    )
    store.write_bars(asset, BarInterval.DAY_1, bars)
    base = store.root / "equity" / "AAPL" / "interval=1d"
    assert (base / "year=2023" / "data.parquet").exists()
    assert (base / "year=2024" / "data.parquet").exists()


def test_coverage(store, asset):
    bars = _make_bars(
        "AAPL",
        [
            datetime(2024, 1, 2, tzinfo=timezone.utc),
            datetime(2024, 1, 5, tzinfo=timezone.utc),
        ],
    )
    store.write_bars(asset, BarInterval.DAY_1, bars)
    cov = store.coverage(asset, BarInterval.DAY_1)
    assert cov is not None
    assert cov[0] == datetime(2024, 1, 2, tzinfo=timezone.utc)
    assert cov[1] == datetime(2024, 1, 5, tzinfo=timezone.utc)


def test_empty_returns_none_coverage(store, asset):
    assert store.coverage(asset, BarInterval.DAY_1) is None
