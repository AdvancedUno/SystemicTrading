"""Ingest the configured universe.

Usage:
    python scripts/ingest.py --interval 1d --start 2020-01-01
    python scripts/ingest.py --interval 1m --start 2024-01-01 --asset-class crypto
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from quant.data.ingest import Ingestor
from quant.data.types import Asset, AssetClass, BarInterval
from quant.utils.config import ensure_data_dirs
from quant.utils.logging import logger

app = typer.Typer(add_completion=False)
console = Console()


def _load_universe(path: Path) -> dict[AssetClass, list[Asset]]:
    raw = yaml.safe_load(path.read_text())
    return {
        AssetClass.EQUITY: [
            Asset(symbol=s, asset_class=AssetClass.EQUITY)
            for s in raw.get("equities", [])
        ],
        AssetClass.CRYPTO: [
            Asset(symbol=s, asset_class=AssetClass.CRYPTO)
            for s in raw.get("crypto", [])
        ],
    }


@app.command()
def main(
    interval: str = typer.Option("1d", help="Bar interval: 1m, 5m, 15m, 1h, 1d"),
    start: str = typer.Option("2020-01-01", help="Start date YYYY-MM-DD (UTC)"),
    end: str | None = typer.Option(None, help="End date YYYY-MM-DD; default=now"),
    asset_class: str | None = typer.Option(
        None, help="Restrict to 'equity' or 'crypto'; default=both"
    ),
    universe_path: Path = typer.Option(
        Path("config/universe.yaml"), help="Universe YAML"
    ),
    incremental: bool = typer.Option(True, help="Only fetch missing data"),
) -> None:
    ensure_data_dirs()

    bar_interval = BarInterval(interval)
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = (
        datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
        if end
        else datetime.now(timezone.utc)
    )

    universe = _load_universe(universe_path)
    if asset_class:
        ac = AssetClass(asset_class)
        universe = {ac: universe.get(ac, [])}

    ingestor = Ingestor()
    all_results: dict[str, int] = {}
    for ac, assets in universe.items():
        if not assets:
            continue
        logger.info(f"Ingesting {len(assets)} {ac.value} assets at {interval}")
        results = ingestor.ingest_universe(
            assets, bar_interval, start_dt, end_dt, incremental=incremental
        )
        all_results.update(results)

    # Summary
    table = Table(title=f"Ingestion summary ({interval})")
    table.add_column("Asset")
    table.add_column("Rows added", justify="right")
    table.add_column("Status")
    for key, n in sorted(all_results.items()):
        status = "OK" if n >= 0 else "FAILED"
        table.add_row(key, str(n) if n >= 0 else "-", status)
    console.print(table)


if __name__ == "__main__":
    app()
