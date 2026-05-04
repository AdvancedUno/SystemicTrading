"""Feature pipeline — apply features to a bar panel.

Input: long-format bar panel (one row per (symbol, ts)) with OHLCV.
Output: same long-format panel with feature columns added.

The pipeline handles two kinds of features differently:

1. Time-series features (lookback > 0): grouped by symbol.
   The .over("symbol") in Feature.expr() handles this.

2. Cross-sectional features (lookback == 0, category="cross_sectional"):
   computed AFTER time-series features, grouped by ts.
   These reference columns produced by time-series features
   (declared via depends_on).
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from quant.features.base import Feature
from quant.features.registry import REGISTRY
from quant.utils.logging import logger


class FeaturePipeline:
    """Compute a set of features over a bar panel."""

    def __init__(self, features: Sequence[Feature] | Sequence[str] | None = None):
        """
        Args:
            features: List of Feature objects or feature names. If None,
                all registered features are used.
        """
        if features is None:
            self.features = REGISTRY.all()
        else:
            self.features = [
                REGISTRY.get(f) if isinstance(f, str) else f for f in features
            ]
        self._validate()

    def _validate(self) -> None:
        """Ensure dependencies are present and ordering is sane."""
        names = {f.name for f in self.features}
        for f in self.features:
            for dep in f.depends_on:
                if dep not in names and dep not in REGISTRY.names():
                    raise ValueError(
                        f"Feature {f.name!r} depends on {dep!r} which isn't "
                        f"in the pipeline or registry"
                    )

    @property
    def time_series_features(self) -> list[Feature]:
        return [f for f in self.features if f.category != "cross_sectional"]

    @property
    def cross_sectional_features(self) -> list[Feature]:
        return [f for f in self.features if f.category == "cross_sectional"]

    def apply(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Apply all features to a bar panel.

        Args:
            panel: Long-format DataFrame with at minimum:
                ts, symbol, open, high, low, close, volume

        Returns:
            DataFrame with original columns + one column per feature.
        """
        # Sort once — features rely on ordering within each symbol
        panel = panel.sort(["symbol", "ts"])

        # ---- Time-series features ----
        ts_exprs = [f.expr() for f in self.time_series_features]
        if ts_exprs:
            logger.debug(f"Applying {len(ts_exprs)} time-series features")
            panel = panel.with_columns(ts_exprs)

        # ---- Cross-sectional features ----
        # These reference columns added by time-series features above.
        # They group by ts (not symbol) so we apply .over("ts") here at the
        # pipeline level, NOT in the feature itself.
        cs_features = self.cross_sectional_features
        if cs_features:
            logger.debug(f"Applying {len(cs_features)} cross-sectional features")
            cs_exprs = [
                f.builder().over("ts").alias(f.name) for f in cs_features
            ]
            panel = panel.with_columns(cs_exprs)

        return panel

    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]

    def max_lookback(self) -> int:
        """Bars of history needed before features become valid."""
        return max((f.lookback for f in self.features), default=0)

    def __len__(self) -> int:
        return len(self.features)

    def __repr__(self) -> str:
        return (
            f"FeaturePipeline(n={len(self)}, "
            f"max_lookback={self.max_lookback()})"
        )
