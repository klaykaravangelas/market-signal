# Spec 001 — Local Market Data Ingestion

## Status

Implemented and verified with live Tiingo responses for SPY and QQQ.

## Goal

Build the first usable local data pipeline: fetch historical daily prices for a small, configurable set of securities, preserve the provider response, normalize it into a stable schema, and write Parquet files that DuckDB can query. The pipeline must run without AWS credentials.

This implements Phase 1 of [Spec 000](000-project-overview.md). Feature engineering, model training, MLflow, cloud storage, and scheduled ingestion belong to later specs.

## Inputs and command

- A local command accepts one or more ticker symbols, a start date, an end date, and a data directory. SPY must work; QQQ is a second supported example, not a hard-coded limit.
- Dates are ISO `YYYY-MM-DD` calendar dates and define an inclusive range of market sessions. Reject an invalid date, an empty ticker list, or a start date after the end date before calling the provider.
- Normalize ticker input to uppercase and remove duplicate symbols. The command must not require AWS credentials or an always-running service.
- The selected provider and any credentials are supplied through configuration or environment variables. Credentials must never appear in logs or persisted data.
- Provide a local way to replay a saved raw capture through normalization without another provider request.
- Document one reproducible local command in the README once implementation exists.

## Provider boundary

Define a typed `MarketDataProvider` interface for fetching daily price records for one ticker and date range. The ingestion workflow depends on this interface, not on a provider SDK or response shape. Implement one real historical daily-price adapter in this phase; a second adapter is out of scope.

The first provider must supply daily open, high, low, close, adjusted close, and volume for SPY and QQQ over a user-selected historical range. Choose a provider whose access method and terms permit this project's intended open-source demonstration. Record the choice and its tradeoffs in a short ADR before implementing the adapter. Provider rate limits and network errors must produce clear failures rather than partial success reported as a complete dataset.

Use a fake provider in automated tests. Tests must not depend on live API availability or credentials. A real-provider smoke run is a manual verification step and should be documented separately from the automated test suite.

## Data contract

The normalized dataset has one row per `(ticker, session_date)` with these fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `ticker` | string | Uppercase input symbol. |
| `session_date` | date | Trading session date as reported by the provider; no intraday timestamp. |
| `open`, `high`, `low`, `close` | finite positive number | Provider's daily prices, in the provider's quoted currency. |
| `adjusted_close` | finite positive number | Provider's historical adjusted close. Do not calculate it from unadjusted OHLC values. |
| `volume` | nonnegative integer | Provider's daily reported volume. |

The adapter must document whether the OHLC prices are raw or adjusted and what the provider's `adjusted_close` represents. Do not imply that historical adjusted prices were necessarily available in their present form on the original session date. Later feature and validation specs must address point-in-time use of revised data.

Normalization must reject duplicate `(ticker, session_date)` rows, missing required values, non-finite or invalid prices, negative volume, and dates outside the requested range. An empty response is a clear failure and must not replace existing data. Sort output by ticker and date. Do not invent rows for weekends, holidays, or missing provider sessions; report the returned date span and row count so gaps can be investigated.

## Storage and reproducibility

Use the same local data directory structure described in the architecture:

```text
data/
├── raw/prices/<provider>/<capture-id>/...
└── processed/prices/<ticker>.parquet
```

- Save each successful provider response under a unique capture ID before normalization. Preserve its original fields and values in a lossless format suitable for replay. Never store credentials. Do not overwrite an earlier capture.
- Keep a small manifest with provider name, requested ticker and date range, capture time in UTC, raw file location, and normalized schema version. A capture must be traceable to the processed output produced from it.
- Write normalized prices as Parquet. The processed file for a ticker is the current local view. Reprocessing the same captured response must yield the same normalized rows, regardless of when it is replayed.
- Merge a successful capture with existing rows for that ticker: preserve dates outside the requested range and replace dates present in the new capture. Repeating ingestion for the same ticker and range must not create duplicate normalized rows. If the provider revises a historical row, the latest successful capture may update the current processed view; the earlier raw capture remains available for comparison.
- Replace processed files atomically so a failed validation or interrupted write does not leave a partial Parquet file. A failed ticker must not be reported as successful.
- Keep local data and credentials out of Git. The implementation should add appropriate ignore rules.

This phase does not require a distributed data lake, a database server, or a general dataset-versioning system.

## DuckDB access

Document a DuckDB query that reads the processed Parquet files directly and returns SPY rows ordered by `session_date`. DuckDB is a local query tool here; ingestion does not need to create or maintain a DuckDB database file.

## Testing and verification

Automated tests should cover:

1. Valid and invalid command inputs, including date bounds and repeated ticker symbols.
2. Provider-response normalization, sorting, schema, and rejection of malformed or duplicate records.
3. Raw capture and manifest creation, with credentials absent from both.
4. Parquet output read back through DuckDB with expected rows and types.
5. Repeated ingestion, overlapping date ranges, and offline replay without duplicate rows or loss of dates outside the requested range.
6. Provider failure and invalid data leaving any prior processed file intact.

Run `pytest` and `ruff` checks for the Python changes. The real-provider smoke run should use a small historical range and record the provider, requested dates, returned date span, row count, and any access limitation without committing downloaded data.

## Acceptance criteria

- [x] A documented local command uses the Tiingo adapter for a specified date range; SPY and QQQ use the same code path in offline tests.
- [x] A live Tiingo smoke run fetches daily historical prices for SPY and QQQ with a personal API token.
- [x] The provider is isolated behind a typed interface and its selection is documented in an ADR.
- [x] Every successful fetch produces a replayable raw capture and provenance manifest.
- [x] Validated normalized rows are stored in Parquet with the schema above and are directly queryable with DuckDB.
- [x] Rerunning the same request or replaying a capture produces no duplicate normalized rows.
- [x] Invalid input, malformed data, and provider failures produce clear errors without corrupting previously written processed data.
- [x] Automated tests pass without network access or credentials.
- [x] The live smoke run records the provider, requested dates, returned date span, row count, and any access limitation without committing downloaded data.
- [x] The pipeline works locally without AWS services, and generated data and secrets are not committed.

Live smoke evidence (2026-09-23): Tiingo returned 42 daily rows each for SPY and QQQ for the requested range 2024-01-02 through 2024-03-01. Both captures have `status: processed`, the normalized Parquet files contain the same date span, and no access limitation was observed. The responses and manifests remain under the Git-ignored `data/` directory.

## Known limits

Provider data may contain missing sessions, corrected historical prices, or retrospective corporate-action adjustments. This phase preserves raw captures and reports what was received; it does not certify point-in-time market-data correctness. Incremental scheduling, data freshness monitoring, and cloud ingestion will be specified later.
