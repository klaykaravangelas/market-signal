# Spec 002 — Local Feature Engineering and Five-Session Labels

## Status

Implemented and verified offline. A live price dataset is optional for this phase.

## Goal and scope

Turn the normalized daily price Parquet files from [Spec 001](001-market-data-ingestion.md) into deterministic, per-ticker feature and label datasets. This is Phase 2 of [Spec 000](000-project-overview.md) and prepares data for the first model-training spec.

This phase runs locally from existing Parquet files. It makes no provider requests and needs no API token, AWS service, MLflow server, or model library. Model fitting, train/validation splits, probability calibration, and backtesting remain out of scope.

## Inputs and command

- A local command accepts one or more ticker symbols and a data directory, reusing the ticker normalization rules from Spec 001. The default input for each ticker is `data/processed/prices/<ticker>.parquet`.
- SPY and QQQ use the XNYS calendar in this phase. Reject another symbol unless its exchange calendar is explicitly configured; do not infer an exchange from its ticker text.
- The command reads the full available history for each requested ticker. A requested output date range may be added later, but must never remove the lookback rows needed to calculate features correctly.
- Reject a missing input file, an unexpected price schema, duplicate or unsorted session dates, another ticker in the file, or invalid price and volume values. A failed ticker must not replace its previous feature or label files.
- Document the command and a DuckDB query for both outputs in the README when implemented.

## Time and price conventions

For a ticker, index its validated daily bars by ascending exchange session: `t = 0, 1, ...`. A feature row stamped `session_date = d_t` represents information available **after the final daily bar for that session is published**. It must never be treated as a pre-open or intraday signal for `d_t`.

Use `adjusted_close` for price-derived features and the target. The initial price features are ratios of adjusted closes, which are invariant to a uniform rescaling of all observations at or before `t` from a later split or dividend adjustment. Do not use absolute adjusted price levels as predictors. This does not eliminate the effect of provider corrections made after a historical session; the current merged price file is not a point-in-time archive. Record that limitation in generated metadata and do not describe results from this dataset as point-in-time backtests.

Use `volume` as reported in the normalized dataset. Relative volume compares current volume with earlier sessions for the same ticker. Do not mix observations across tickers when calculating windows or labels.

## Session completeness

For SPY and QQQ, use an explicit US equity exchange-session calendar (XNYS) to check the input dates. Do not fill missing sessions with synthetic prices. A window is usable only when its required bars correspond to consecutive expected exchange sessions; an unexplained missing session invalidates any feature or label window that crosses it. Non-session dates in the input are errors. Report the count and date span of missing expected sessions within the input range.

The calendar is used only for completeness checks. All calculations still use observed bars. Record the calendar identifier and calendar-library version in output metadata so the data can be reproduced. Automatic exchange-to-calendar mapping is deferred.

## Initial feature set

The following definitions are fixed for this version. Windows include session `t` unless stated otherwise. A feature is `NULL` until its entire required history exists; do not shorten a window or forward-fill a value.

| Column | Definition at session `t` | Required history |
| --- | --- | --- |
| `return_1` | `adjusted_close[t] / adjusted_close[t-1] - 1` | 1 prior session |
| `return_5` | `adjusted_close[t] / adjusted_close[t-5] - 1` | 5 prior sessions |
| `return_20` | `adjusted_close[t] / adjusted_close[t-20] - 1` | 20 prior sessions |
| `volatility_5` | Sample standard deviation (`ddof=1`) of `return_1` for sessions `t-4` through `t` | 5 daily returns |
| `volatility_20` | Sample standard deviation (`ddof=1`) of `return_1` for sessions `t-19` through `t` | 20 daily returns |
| `ma_distance_20` | `adjusted_close[t] / mean(the closes from t-19 through t, inclusive) - 1` | 20 closes |
| `relative_volume_20` | `volume[t] / mean(the volumes from t-20 through t-1, inclusive)`, using the **20 prior sessions only** | 20 prior volumes |

If the prior-volume mean is zero, `relative_volume_20` is `NULL`; do not emit infinity. Feature values must otherwise be finite. The first complete feature row normally occurs at the 21st observed session (`t = 20`). A missing exchange session may delay or invalidate rows that cross it.

RSI, 60-session volatility, macro data, fundamentals, and cross-asset features are deferred. Each future feature requires its own definition and leakage test.

## Five-session target

The initial target answers whether the asset's adjusted close is higher after **five subsequent exchange sessions**:

```text
future_return_5[t] = adjusted_close[t+5] / adjusted_close[t] - 1
target_positive_5[t] = 1 if future_return_5[t] > 0 else 0
label_end_date[t] = session_date[t+5]
```

A zero return maps to class `0`. A label is absent until all five subsequent expected sessions and their prices are present. In particular, the last five sessions in a complete input history remain unlabeled; do not assign them class `0`. If a price gap occurs within the five-session horizon, leave that label absent.

Store labels separately from predictors. `future_return_5`, `target_positive_5`, and `label_end_date` must never appear in the feature file. Later training code must join on `(ticker, session_date)` and must explicitly exclude labels and label metadata from model inputs. Because adjacent five-session labels overlap, the training spec must prevent labels whose `label_end_date` crosses a validation boundary from entering the training fold.

## Outputs and provenance

Write two Parquet files per ticker:

```text
data/features/<ticker>.parquet
data/labels/<ticker>.parquet
```

- The feature file has unique, ascending `(ticker, session_date)` rows. It contains the seven feature columns above and no future-derived fields. Keep all valid input sessions, including warm-up rows with `NULL` features and recent rows without labels; downstream training can select complete rows.
- The label file has unique, ascending `(ticker, session_date)` rows only where the five-session outcome is known and valid. It contains `label_end_date`, `future_return_5`, and `target_positive_5`.
- Write both outputs from one validated input snapshot. Complete validation and calculation before replacing either file, and replace each file atomically. A validation or calculation failure must leave prior outputs intact. Publish the manifest last; readers must verify its output hashes to detect an interrupted publication. Repeating the command with identical input bytes and feature definitions must produce identical table rows.
- Write a small manifest for the pair containing ticker, source Parquet path and SHA-256 hash, source date span and row count, feature-definition version (`v1`), target horizon, calendar identifier and library version, output paths, hashes and row counts, and generation time in UTC. The manifest must make it clear which source snapshot produced the current outputs.
- No generated data files should be committed to Git. Feature and label files live under the ignored `data/` directory.

## Testing and verification

Use synthetic daily price fixtures with known session dates. Automated tests must run without a network connection or API token and cover:

1. Each formula, its exact window endpoints, sample volatility convention, warm-up `NULL`s, and zero prior-volume handling.
2. A feature value at `t` staying unchanged when only prices or volumes after `t` change; a label at `t` changing when its future outcome changes.
3. The fifth subsequent session as the label endpoint, zero-return class `0`, and unlabeled trailing sessions.
4. Missing expected sessions, duplicate dates, non-session dates, malformed input schema, and cross-ticker isolation.
5. A DuckDB read and join of generated Parquet files with the expected schema and row counts.
6. Repeated runs producing the same rows; validation or calculation failure preserving previous outputs; interrupted publication detected by manifest hash checks.

Run `pytest`, `ruff check`, and `ruff format --check` for the Python changes. A manual local smoke run may use existing SPY or QQQ Parquet data if present, but no live data fetch is required for this spec.

## Acceptance criteria

- [x] The local command builds feature and label Parquet files from a Spec 001 price file without a provider request or secret.
- [x] The seven defined features use only the current and earlier complete sessions, and their formulas match the table above.
- [x] The five-session label uses the fifth subsequent expected exchange session; unknown outcomes are absent rather than treated as negative.
- [x] Output features contain no label or future-derived fields, and ticker histories are isolated.
- [x] Missing-session windows and invalid inputs are handled explicitly; validation and calculation failures preserve prior outputs, and interrupted publication is detectable.
- [x] A manifest identifies the source snapshot, feature version, calendar, and resulting files.
- [x] Synthetic unit and Parquet/DuckDB integration tests pass offline, and the README documents the command and output contract.

## Known limits

This dataset is suitable for building and testing the first local modeling pipeline, subject to the documented historical-data revision limit. It does not provide a point-in-time vendor feed, a trading execution assumption, or a claim of predictive performance.
