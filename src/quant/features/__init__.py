"""Feature engineering.

Importing this package triggers registration of all built-in features.
"""

from quant.features.base import Feature
from quant.features.pipeline import FeaturePipeline
from quant.features.registry import REGISTRY, register

# Import library — this populates REGISTRY
from quant.features import library  # noqa: F401

__all__ = ["Feature", "FeaturePipeline", "REGISTRY", "register"]
