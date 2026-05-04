"""Feature registry — central catalog of all features.

Usage:
    from quant.features.registry import register, REGISTRY

    @register(category="returns", lookback=20)
    def return_20d() -> pl.Expr:
        return pl.col("close") / pl.col("close").shift(20) - 1

    # Elsewhere:
    feat = REGISTRY.get("return_20d")
    all_returns = REGISTRY.by_category("returns")
"""

from __future__ import annotations

from typing import Callable

import polars as pl

from quant.features.base import Feature


class FeatureRegistry:
    """Singleton-style registry for features."""

    def __init__(self) -> None:
        self._features: dict[str, Feature] = {}

    def add(self, feature: Feature) -> None:
        if feature.name in self._features:
            raise ValueError(f"Feature {feature.name!r} already registered")
        self._features[feature.name] = feature

    def get(self, name: str) -> Feature:
        if name not in self._features:
            raise KeyError(f"Feature {name!r} not in registry")
        return self._features[name]

    def all(self) -> list[Feature]:
        return list(self._features.values())

    def names(self) -> list[str]:
        return sorted(self._features.keys())

    def by_category(self, category: str) -> list[Feature]:
        return [f for f in self._features.values() if f.category == category]

    def categories(self) -> list[str]:
        return sorted({f.category for f in self._features.values()})

    def __len__(self) -> int:
        return len(self._features)

    def __contains__(self, name: str) -> bool:
        return name in self._features


# Global registry. Importing any feature module populates it.
REGISTRY = FeatureRegistry()


def register(
    *,
    category: str,
    lookback: int,
    lag: int = 1,
    description: str = "",
    depends_on: tuple[str, ...] = (),
    name: str | None = None,
) -> Callable[[Callable[[], pl.Expr]], Feature]:
    """Decorator: turn a builder function into a registered Feature.

    The decorated function should return a Polars expression — the *raw*
    formula, with no lag applied. The Feature wrapper applies the lag.
    """

    def decorator(builder: Callable[[], pl.Expr]) -> Feature:
        feat_name = name or builder.__name__
        feat = Feature(
            name=feat_name,
            category=category,
            builder=builder,
            lookback=lookback,
            lag=lag,
            description=description or (builder.__doc__ or "").strip(),
            depends_on=depends_on,
        )
        REGISTRY.add(feat)
        return feat

    return decorator
