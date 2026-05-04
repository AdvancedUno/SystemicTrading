"""End-to-end backtest demo.

Loads ingested data from your Parquet store, runs both strategies through
both engines, and prints summary stats side by side.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --start 2022-01-01 --end 2025-01-01
    python scripts/run_backtest.py --asset-class crypto
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import typer
import yaml
from rich.console import Console
from rich.table import Table

from quant.backtest import run_event_driven, run_vectorized
from quant.data.store import ParquetBarStore
from quant.data.types import Asset, AssetClass, BarInterval
from quant.strategies.library import (
    CrossSectionalMomentum,
    CryptoMeanReversion,
)
from quant.utils.logging import logger

app = typer.Typer(add_completion=False)
console = Console()


def _load_panel(
    store: ParquetBarStore,
    universe: list[str],
    asset_class: AssetClass,
    interval: BarInterval,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pl.DataFrame:
    """Read all symbols' bars and concatenate to a long-format panel."""
    parts = []
    for sym in universe:
        asset = Asset(symbol=sym, asset_class=asset_class)
        df = store.read_bars(asset, interval, start=start, end=end)
        if df.is_empty():
            logger.warning(f"No data for {asset_class.value}/{sym}")
            continue
        parts.append(df)
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts).sort(["symbol", "ts"])


def _load_universe(path: Path) -> dict[AssetClass, list[str]]:
    raw = yaml.safe_load(path.read_text())
    return {
        AssetClass.EQUITY: raw.get("equities", []),
        AssetClass.CRYPTO: raw.get("crypto", []),
    }


def _print_stats_table(title: str, results: dict, periods_per_year: int):
    table = Table(title=title)
    table.add_column("Engine")
    table.add_column("Sharpe", justify="right")
    table.add_column("Sortino", justify="right")
    table.add_column("Ann. Return", justify="right")
    table.add_column("Ann. Vol", justify="right")
    table.add_column("Max DD", justify="right")
    table.add_column("Hit Rate", justify="right")
    table.add_column("Avg Turnover", justify="right")
    for engine, res in results.items():
        s = res.stats(periods_per_year=periods_per_year)
        table.add_row(
            engine,
            f"{s.sharpe:.2f}",
            f"{s.sortino:.2f}",
            f"{s.annual_return:.2%}",
            f"{s.annual_volatility:.2%}",
            f"{s.max_drawdown:.2%}",
            f"{s.hit_rate:.2%}",
            f"{s.avg_turnover:.2%}" if s.avg_turnover == s.avg_turnover else "n/a",
        )
    console.print(table)


@app.command()
def main(
    asset_class: str = typer.Option(
        "both", help="'equity', 'crypto', or 'both'"
    ),
    interval: str = typer.Option("1d", help="Bar interval"),
    start: str | None = typer.Option("2021-01-01", help="Start date YYYY-MM-DD"),
    end: str | None = typer.Option(None, help="End date YYYY-MM-DD; default=last"),
    universe_path: Path = typer.Option(
        Path("config/universe.yaml"), help="Universe YAML"
    ),
):
    bar_interval = BarInterval(interval)
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc) if start else None
    end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc) if end else None

    store = ParquetBarStore()
    universe = _load_universe(universe_path)

    if asset_class in ("equity", "both"):
        console.rule("[bold cyan]Equity — Cross-Sectional Momentum[/bold cyan]")
        panel = _load_panel(
            store, universe[AssetClass.EQUITY], AssetClass.EQUITY,
            bar_interval, start_dt, end_dt,
        )
        if panel.is_empty():
            console.print("[red]No equity data found. Run scripts/ingest.py first.[/red]")
        else:
            console.print(
                f"Loaded equity panel: {panel.height} rows, "
                f"{panel['symbol'].n_unique()} symbols, "
                f"{panel['ts'].min().date()} -> {panel['ts'].max().date()}"
            )
            strat = CrossSectionalMomentum()
            weights = strat.generate_weights(panel)
            console.print(f"Generated {weights.height} weight rows")
            res_vec = run_vectorized(
                panel=panel, weights=weights, asset_class=AssetClass.EQUITY,
            )
            res_evt = run_event_driven(
                panel=panel, weights=weights, asset_class=AssetClass.EQUITY,
            )
            _print_stats_table(
                "Equity strategy: Cross-Sectional Momentum",
                {"Vectorized": res_vec, "Event-Driven": res_evt},
                periods_per_year=252,
            )

    if asset_class in ("crypto", "both"):
        console.rule("[bold cyan]Crypto — Mean Reversion[/bold cyan]")
        panel = _load_panel(
            store, universe[AssetClass.CRYPTO], AssetClass.CRYPTO,
            bar_interval, start_dt, end_dt,
        )
        if panel.is_empty():
            console.print("[red]No crypto data found. Run scripts/ingest.py first.[/red]")
        else:
            console.print(
                f"Loaded crypto panel: {panel.height} rows, "
                f"{panel['symbol'].n_unique()} symbols, "
                f"{panel['ts'].min().date()} -> {panel['ts'].max().date()}"
            )
            strat = CryptoMeanReversion()
            weights = strat.generate_weights(panel)
            console.print(f"Generated {weights.height} weight rows")
            res_vec = run_vectorized(
                panel=panel, weights=weights, asset_class=AssetClass.CRYPTO,
            )
            res_evt = run_event_driven(
                panel=panel, weights=weights, asset_class=AssetClass.CRYPTO,
            )
            _print_stats_table(
                "Crypto strategy: Mean Reversion",
                {"Vectorized": res_vec, "Event-Driven": res_evt},
                periods_per_year=365,  # crypto trades 7 days/week
            )


if __name__ == "__main__":
    app()
