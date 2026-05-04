"""Volume and liquidity features.

────────────────────────────────────────────────────────────────────────────
1. Dollar volume
────────────────────────────────────────────────────────────────────────────
DV(t) = close(t) * volume(t)

The most basic liquidity proxy. We normalize across time within a stock
using a rolling z-score so it's comparable across symbols.

────────────────────────────────────────────────────────────────────────────
2. Turnover (volume / shares outstanding) — proxy
────────────────────────────────────────────────────────────────────────────
True turnover requires shares-outstanding data, which the free Alpaca tier
doesn't include. As a stand-in we use:

    turnover_proxy(t) = volume(t) / mean(volume) over window

This captures "how busy is this name today vs. its own average." It's
a within-stock z-score-ish feature and works well across the cross-section.

────────────────────────────────────────────────────────────────────────────
3. Amihud illiquidity (2002)
────────────────────────────────────────────────────────────────────────────
Amihud's "illiq" measure captures how much price moves per dollar traded:

    illiq(t) = |return(t)| / dollar_volume(t)

The intuition: in a deep market, $1M of trading barely moves the price;
in an illiquid name, the same $1M moves it a lot. Average over a window:

    ILLIQ_window = mean over k of |return(t-k)| / dollar_volume(t-k)

Amihud showed this measure prices a liquidity risk premium in stock
returns: illiquid stocks earn higher returns on average. It's still
widely used as a feature in cross-sectional ML models.

We multiply by a scale (10^6) so the numbers aren't tiny.

────────────────────────────────────────────────────────────────────────────
4. Volume z-score
────────────────────────────────────────────────────────────────────────────
A within-stock standardized volume:

    vol_z(t) = (volume(t) - mean_window) / std_window

Captures unusual volume — a value of +3 means today's volume is 3 sigma
above the recent mean. Often shows up before earnings, news, or large
institutional rebalancing.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import polars as pl

from quant.features.registry import register


@register(category="volume", lookback=1)
def dollar_volume() -> pl.Expr:
    """close * volume — basic liquidity measure in dollar terms."""
    return pl.col("close") * pl.col("volume")


@register(category="volume", lookback=20)
def dollar_volume_zscore_20d() -> pl.Expr:
    """20-bar z-score of dollar volume within a symbol."""
    dv = pl.col("close") * pl.col("volume")
    mean = dv.rolling_mean(window_size=20)
    std = dv.rolling_std(window_size=20)
    return (dv - mean) / std


@register(category="volume", lookback=20)
def turnover_proxy_20d() -> pl.Expr:
    """volume / 20-bar mean volume.

    Stand-in for true turnover when shares-outstanding isn't available.
    A value of 2.0 means today's volume is 2x the 20-day average.
    """
    return pl.col("volume") / pl.col("volume").rolling_mean(window_size=20)


@register(category="volume", lookback=20)
def amihud_illiq_20d() -> pl.Expr:
    """Amihud (2002) illiquidity measure averaged over 20 bars.

    illiq(t) = |return(t)| / dollar_volume(t),  scaled by 1e6
    Returns the rolling mean of this ratio over the window.
    """
    log_ret = (pl.col("close") / pl.col("close").shift(1)).log()
    dv = pl.col("close") * pl.col("volume")
    # Avoid div-by-zero on zero-volume bars (rare for liquid names but possible)
    illiq_per_bar = (
        log_ret.abs() / dv.clip(lower_bound=1.0)
    ) * 1e6
    return illiq_per_bar.rolling_mean(window_size=20)


@register(category="volume", lookback=60)
def volume_zscore_60d() -> pl.Expr:
    """60-bar z-score of raw volume."""
    vol = pl.col("volume")
    mean = vol.rolling_mean(window_size=60)
    std = vol.rolling_std(window_size=60)
    return (vol - mean) / std
