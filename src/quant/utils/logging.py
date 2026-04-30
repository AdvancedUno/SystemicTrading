"""Logging configuration. Use `from quant.utils.logging import logger`."""

from __future__ import annotations

import sys

from loguru import logger

from quant.utils.config import settings

# Remove default handler and configure our own
logger.remove()
logger.add(
    sys.stderr,
    level=settings.log_level,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
)

__all__ = ["logger"]
