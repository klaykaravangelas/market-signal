# MarketSignal

MarketSignal is a local-first market data and forecasting project. Its first implemented phase ingests historical daily prices, retains the provider response, and stores normalized Parquet files for DuckDB queries. See [Spec 001](specs/001-market-data-ingestion.md) and the [architecture](docs/architecture.md).

## Set up

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run:

```sh
uv sync --dev --no-editable
```

DuckDB is installed as a Python dependency in `.venv`; no separate database server or DuckDB CLI is required. The Python project targets Python 3.13 or newer.

Because this setup uses a non-editable local install, rebuild the package after changing files under `src/`: `uv sync --dev --no-editable --reinstall-package marketsignal`.

## Ingest daily prices

Create a personal [Tiingo API token](https://www.tiingo.com/documentation/general/connecting) and set it in your shell. Keep the token out of committed files. Tiingo's basic data is for personal or internal use; this repository does not distribute downloaded prices.

```sh
export TIINGO_API_TOKEN='your-token'
uv run --no-editable marketsignal ingest SPY QQQ --start 2024-01-02 --end 2024-01-31
```

The date range is inclusive. Use `--data-dir /path/to/data` to change the default `data/` location. Each successful fetch writes an immutable raw JSON response and manifest under `data/raw/prices/tiingo/<capture-id>/`, and updates `data/processed/prices/<ticker>.parquet`. Reruns preserve dates outside the new range and replace dates returned in the new response. The command prints the received date span and row count. Empty or malformed responses fail without replacing existing processed data.

To rebuild the processed view from a saved response without network access:

```sh
uv run --no-editable marketsignal replay data/raw/prices/tiingo/<capture-id>
```

## Query with DuckDB

DuckDB reads Parquet directly. This example uses its Python API:

```sh
uv run --no-editable python -c "import duckdb; print(duckdb.sql(\"SELECT ticker, session_date, close, adjusted_close FROM read_parquet('data/processed/prices/SPY.parquet') ORDER BY session_date LIMIT 10\").fetchall())"
```

Raw OHLC values are as traded according to Tiingo; `adjusted_close` is Tiingo's split and dividend adjusted field. Historical adjustments can change later. The raw captures allow comparison and replay, but this phase does not establish point-in-time data correctness for model validation.

## Build features and labels

After ingesting prices, build the local datasets without another API request or token:

```sh
uv run --no-editable marketsignal features SPY QQQ
```

Use `--data-dir /path/to/data` for another local dataset. SPY and QQQ use the XNYS session calendar. For another ticker, supply its exchange calendar explicitly with `--calendar XNYS` or another supported calendar name.

The command writes `data/features/<ticker>.parquet` with seven past-and-current-session features, `data/labels/<ticker>.parquet` with known five-session outcomes, and `data/features/<ticker>.manifest.json` with input and output hashes. Recent sessions remain in the feature file even though their future labels are not yet known. Missing expected exchange sessions invalidate windows that cross them; the command reports their count. [Spec 002](specs/002-feature-engineering.md) defines the formulas and timing rules.

Query the two files with DuckDB, joining on ticker and session date:

```sh
uv run --no-editable python - <<'PY'
import duckdb

rows = duckdb.sql("""
    SELECT f.ticker, f.session_date, f.return_20, l.target_positive_5
    FROM read_parquet('data/features/SPY.parquet') AS f
    JOIN read_parquet('data/labels/SPY.parquet') AS l
      USING (ticker, session_date)
    WHERE f.return_20 IS NOT NULL
    ORDER BY f.session_date
    LIMIT 10
""").fetchall()
print(rows)
PY
```

Features are available only after that session's final daily bar is published. Historical provider corrections can change the backfilled dataset; the manifest identifies the exact source file used. This is not a point-in-time backtest dataset.

## Evaluate direction models

Ingest several years of prices and rebuild features before requesting full-year validation folds. A short January–March 2024 capture is too short: each fold needs at least 500 complete training rows, 100 complete validation rows, and both target classes in each set.

```sh
uv run --no-editable marketsignal evaluate SPY QQQ --first-validation-year 2022 --folds 3
```

The command checks feature, label, and source price hashes, then evaluates each ticker independently. Annual training windows expand forward while five-session labels crossing a fold boundary are excluded. It compares always-positive and training-prevalence baselines with logistic regression and a random forest. Each completed run is saved under `data/evaluations/<run-id>/` with `predictions.parquet`, `fold_metrics.parquet`, `leaderboard.parquet`, and `manifest.json`. The manifest includes input/output hashes, fold counts and ranges, dropped-row counts, parameters, and software versions. Use `--data-dir /path/to/data` for another local dataset.

Query a completed run directly with DuckDB:

```sh
uv run --no-editable python -c "import duckdb; print(duckdb.sql(\"SELECT ticker, model, roc_auc_mean, brier_score_mean FROM read_parquet('data/evaluations/<run-id>/leaderboard.parquet')\").fetchall())"
```

These are predictive metrics on historical data, not trading returns or evidence of future profitability. [Spec 003](specs/003-model-training.md) defines the evaluation contract and its limits.

## Backtest a saved evaluation

Use an evaluation run ID printed by `marketsignal evaluate` or listed under `data/evaluations/`:

```sh
uv run --no-editable marketsignal backtest <evaluation-run-id>
```

The command verifies the evaluation files and requires the current processed prices to match the price hash saved with that evaluation. If prices were reingested, rebuild features and rerun evaluation first. It saves `intervals.parquet`, `fold_metrics.parquet`, `summary.parquet`, and `manifest.json` under `data/backtests/<run-id>/`. No API token is needed. Use `--data-dir /path/to/data` for another dataset.

Each model's probability drives a daily long-or-cash position at the fixed `0.5` threshold. A signal is available after one session's close, executes at the **next session's close**, and first earns the following close-to-close return. The simulation ends after the last usable signal's return and liquidates the position. It compares models with buy and hold and all cash over the same intervals. The five-session target informs a daily position decision; the simulation does not hold each signal for five sessions.

Default costs are `--fee-bps 1` and `--slippage-bps 5` per one-way position change, including entry and final exit. These are configurable research assumptions. The fold results include gross and net returns, cost drag, annualized return and volatility, Sharpe ratio, drawdown, trade counts, turnover, and exposure. Query them directly:

```sh
uv run --no-editable python -c "import duckdb; print(duckdb.sql(\"SELECT ticker, fold_year, strategy, net_cumulative_return, excess_vs_buy_and_hold FROM read_parquet('data/backtests/<run-id>/fold_metrics.parquet') ORDER BY ticker, fold_year, strategy\").fetchall())"
```

Adjusted closes are a historical return proxy, not guaranteed execution prices. The data may contain later provider corrections, and this backtest does not include market impact, taxes, or cash interest. Results from these same validation years should not be used as proof of future performance. [Spec 004](specs/004-backtesting.md) defines the timing, cost arithmetic, and limitations.

## Checks

```sh
uv run --no-editable pytest
uv run --no-editable ruff check .
uv run --no-editable ruff format --check src tests
```

Automated tests use a fake provider and require neither a token nor network access. A live smoke check requires `TIINGO_API_TOKEN`; use a short date range and note the returned dates and row count. The downloaded data is ignored by Git.
