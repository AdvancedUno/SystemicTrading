"""Feature abstraction.

A `Feature` is a named, categorized Polars expression with metadata about
its lookback window and lag. Every feature in the system is a `Feature`
instance, registered in a global registry, with point-in-time correctness
enforced by convention (see `lag` below).

Why a class and not just a function?
- Metadata: lookback, category, lag, dependencies are queryable
- Composition: features can be combined into pipelines / models
- Caching: we can hash (feature, params) to invalidate caches correctly
- Logging: MLflow / monitoring needs structured info per feature
- Testing: leakage checks need to know each feature's lag
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import polars as pl


# A feature builder takes the panel as a LazyFrame and returns a single Expr.
# The expr operates over the "symbol" group (we apply .over("symbol") at the
# pipeline level, but features can also do their own partitioning if needed).
FeatureBuilder = Callable[[], pl.Expr]


@dataclass(frozen=True)
class Feature:
    """A single named feature.

    Attributes:
        name: Unique snake_case identifier, e.g. "return_20d".
        category: Grouping for organization, e.g. "returns", "volatility".
        builder: Zero-arg function returning the Polars expression.
        lookback: Number of bars of history the feature needs.
            Used for warmup logic — first `lookback` bars per symbol will be
            null/NaN and should be dropped before training.
        lag: Number of bars to shift the result forward to prevent look-ahead.
            Default 1. A lag of 1 means: "the feature value at time t uses
            only data strictly before t" (i.e., closes through t-1).
            See module docstring of leakage.py for the full discipline.
        description: Human-readable description for docs / MLflow.
        depends_on: Other feature names this one builds on. Empty for raw features.
    """

    name: str
    category: str
    builder: FeatureBuilder
    lookback: int
    lag: int = 1
    description: str = ""
    depends_on: tuple[str, ...] = field(default_factory=tuple)

    def expr(self) -> pl.Expr:
        """Materialize the Polars expression with lag applied.

        The lag is applied here, in one place, so individual feature
        definitions don't need to remember it. This is the heart of our
        anti-leakage discipline.
        """
        raw = self.builder()
        if self.lag > 0:
            return raw.shift(self.lag).over("symbol").alias(self.name)
        return raw.over("symbol").alias(self.name)

    def __repr__(self) -> str:
        return (
            f"Feature(name={self.name!r}, category={self.category!r}, "
            f"lookback={self.lookback}, lag={self.lag})"
        )
