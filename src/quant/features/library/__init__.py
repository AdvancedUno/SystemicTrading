"""Feature library — importing this module registers all features.

Order matters: cross-sectional features depend on time-series features
being registered first.
"""

# Time-series features (ordered by category)
from quant.features.library import returns  # noqa: F401
from quant.features.library import volatility  # noqa: F401
from quant.features.library import volume  # noqa: F401
from quant.features.library import technical  # noqa: F401

# Cross-sectional features depend on the above being registered
from quant.features.library import cross_sec  # noqa: F401
