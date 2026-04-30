# Quant — Systematic Trading Platform

Local-first systematic trading research and paper-trading platform for US equities and crypto. Built to mirror how modern quant firms structure their research stack, scaled down for one engineer.

## Status

**Phase 1, Milestone 1: Data layer — DONE **

- [x] Repo skeleton, dependencies, Docker setup
- [x] Canonical bar schema + asset types
- [x] Pluggable `BarSource` protocol
- [x] Alpaca source (US equities, free tier)
- [x] Binance source (crypto, public API)
- [x] Parquet bar store with year-partitioning + dedup
- [x] Incremental ingestion service
- [x] CLI (`scripts/ingest.py`)
- [x] Tests (6/6 passing)

**Next up — Phase 1 remaining milestones:**

- [ ] Feature library (Polars-expression-based, ~30 features)
- [ ] First strategy (cross-sectional momentum + low-vol filter)
- [ ] Backtest harness (Nautilus Trader integration)
- [ ] Walk-forward purged CV (López de Prado)
- [ ] Reporting (Sharpe, Sortino, max DD, turnover, attribution)
- [ ] Paper trading wiring (Alpaca paper account)

**Phase 2 — Modern ML:** LightGBM with purged CV, sequence models (TCN/transformer) on intraday features, meta-labeling, HRP for portfolio construction.

**Phase 3 — Execution & microstructure:** order book features, optimal execution algos, transaction cost analysis.

## Setup

### 1. Create a virtualenv and install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Or with plain pip:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configure credentials

Copy `.env.example` to `.env` and fill in:

- **Alpaca** — sign up at https://alpaca.markets, generate paper trading keys (free, no credit card)
- **Binance Testnet** — https://testnet.binance.vision (only needed when we get to crypto paper trading)

### 3. (Optional) Bring up Postgres + MLflow

```bash
cd docker
docker compose up -d postgres mlflow
```

You can skip this for now — Phase 1 milestone 1 only needs the file system.

### 4. Ingest some data

```bash
# Daily bars for the whole universe, last 5 years
python scripts/ingest.py --interval 1d --start 2020-01-01

# 1-min bars for crypto, last year
python scripts/ingest.py --interval 1m --start 2025-01-01 --asset-class crypto

# Run again — incremental: only fetches what's missing
python scripts/ingest.py --interval 1d --start 2020-01-01
```

Data lands in `data/processed/bars/{asset_class}/{symbol}/interval={interval}/year={year}/data.parquet`.

### 5. Run tests

```bash
pytest -v
```

## Architecture

```
src/quant/
├── data/         Ingestion, storage, point-in-time access
├── features/     Feature engineering (Polars expressions)
├── models/       ML training pipelines
├── strategies/   Signal → target positions
├── backtest/     Event-driven backtester + cost models
├── portfolio/    Position sizing, optimization
├── execution/    Broker adapters (Alpaca, Binance), OMS
├── monitoring/   Live P&L, drift, alerts
└── utils/        Logging, config, time
```

The repo is local-first: everything runs on your laptop. Once strategies are stable, the same code deploys to a VM or container with no changes — config is env-driven.

## Design principles

1. **Local-first, cloud-ready.** Develop offline; deploy when ready.
2. **Source/store separation.** Swapping data providers is a one-file change.
3. **One bar schema everywhere.** Equity, crypto, future asset classes all use the same canonical schema.
4. **Polars over pandas.** Faster, lazier, less footgunny. Pandas only at library boundaries.
5. **Same code for backtest and live.** Nautilus Trader gives us this; strategies are written once.
6. **Tests for the pipeline, not the alpha.** We test correctness; the market judges the alpha.
7. **Realistic costs from day one.** Half-spread, commission, slippage, market impact — all modeled.

## What this is NOT

- Not a high-frequency platform. Retail broker latency is ~50-500ms; we operate at second-and-up timescales.
- Not financial advice. This is research code; live trading with real money is your decision and your risk.
- Not a black box. Every component is meant to be understood, modified, and improved by you.
