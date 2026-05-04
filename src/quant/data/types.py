"""Core data types used across the system.

Design notes:
- All timestamps are UTC. Always. Convert at display time, not in data.
- Bars carry an explicit interval and asset_class so cross-asset code stays clean.
- We use Polars for everything; pandas is only for libs that demand it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class AssetClass(StrEnum):
    EQUITY = "equity"
    CRYPTO = "crypto"


class BarInterval(StrEnum):
    """Standard bar intervals. Strings chosen to match common API params."""

    MIN_1 = "1m"
    MIN_5 = "5m"
    MIN_15 = "15m"
    HOUR_1 = "1h"
    DAY_1 = "1d"

    @property
    def seconds(self) -> int:
        return {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "1d": 86400,
        }[self.value]


@dataclass(frozen=True, slots=True)
class Asset:
    """A tradable instrument."""

    symbol: str  # e.g. "AAPL", "BTC-USD"
    asset_class: AssetClass
    exchange: str | None = None  # e.g. "NASDAQ", "BINANCE"
    name: str | None = None

    @property
    def key(self) -> str:
        """Storage-safe identifier, e.g. 'equity/AAPL' or 'crypto/BTC-USD'."""
        return f"{self.asset_class.value}/{self.symbol}"


@dataclass(frozen=True, slots=True)
class BarRequest:
    """Specification of historical bars to fetch."""

    asset: Asset
    interval: BarInterval
    start: datetime  # UTC
    end: datetime  # UTC

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("start and end must be timezone-aware (UTC)")
        if self.start >= self.end:
            raise ValueError("start must be before end")


# Canonical bar schema. Every BarSource MUST return Polars DataFrames with
# exactly these columns and dtypes. This contract is what lets us swap sources.
BAR_COLUMNS: tuple[str, ...] = (
    "ts",  # Datetime[us, UTC] — bar OPEN time
    "symbol",  # Utf8
    "open",  # Float64
    "high",  # Float64
    "low",  # Float64
    "close",  # Float64
    "volume",  # Float64 (float not int — crypto has fractional volume)
    "trade_count",  # Int64 (nullable; some sources don't provide)
    "vwap",  # Float64 (nullable; computed if absent)
)
