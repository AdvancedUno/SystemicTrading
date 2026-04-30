"""Centralized settings loaded from environment / .env.

Anything secret or environment-specific lives here. Code elsewhere should
import `settings` from this module rather than reading env vars directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Alpaca
    alpaca_api_key: str = Field(default="")
    alpaca_api_secret: str = Field(default="")
    alpaca_paper: bool = True
    alpaca_base_url: str = "https://paper-api.alpaca.markets"

    # Binance
    binance_api_key: str = Field(default="")
    binance_api_secret: str = Field(default="")
    binance_testnet: bool = True

    # Storage
    data_dir: Path = Path("./data")
    database_url: str = "postgresql://quant:quant@localhost:5432/quant"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"

    # Logging
    log_level: str = "INFO"
    environment: Literal["development", "staging", "production"] = "development"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"


settings = Settings()


def ensure_data_dirs() -> None:
    """Create data directories if they don't exist."""
    for d in (settings.raw_dir, settings.processed_dir, settings.features_dir):
        d.mkdir(parents=True, exist_ok=True)
