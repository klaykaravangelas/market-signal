# Spec 006 — Local Model Diagnostics

## Status

Implemented and verified locally on 2026-09-24. All acceptance criteria below are satisfied. AWS remains deferred and optional.

## Goal and scope

Explain the behavior of the saved out-of-sample predictions from [Spec 003](003-model-training.md) in terms a user can inspect: when a model predicts positive, how its probabilities are distributed, which outcomes it gets wrong, and whether it improves on simple baselines. Optionally place the trading results from a linked [Spec 004](004-backtesting.md) run beside the predictive results. Produce a local, portable report and make it available in MLflow when the optional tracking dependency is installed.

This is a **diagnostic** step. It reads saved predictions and results only. It does not fetch prices, rebuild features, load or fit a model, calibrate probabilities, tune a threshold, select a winner, change the target, or run a new backtest. An older evaluation without model artifacts is fully supported because its row-level predictions were saved. No AWS resource or account is required; cloud deployment is deferred.

## Commands and verified inputs

- Add `marketsignal diagnose <evaluation-run-id> [--backtest-run-id <run-id>] [--data-dir data]`. The optional backtest must refer to that exact evaluation run and manifest hash. The command returns the new diagnostic run ID and path to `report.html`.
- Read the evaluation's `manifest.json`, `predictions.parquet`, `fold_metrics.parquet`, and `leaderboard.parquet`. Verify its run ID, supported feature/target version, every declared output path and SHA-256 hash, and required Parquet schemas. If a backtest ID is supplied, likewise verify its manifest and every declared output hash and schema, plus its source evaluation ID and manifest hash. Current feature, label, and price files need not still exist or match: this command inspects the saved result snapshots.
- Require exactly the ticker/year/candidate coverage specified by the evaluation manifest; unique `(ticker, fold_year, model, session_date)` prediction keys; common validation session dates, true labels, and label end dates for every candidate within a ticker/year; finite probabilities in `[0, 1]`; and `predicted == int(probability_positive >= 0.5)`. Reject malformed or partial inputs. Recalculate the saved fold metrics from the row-level predictions and reject material differences (floating-point tolerance `1e-10`). Do not silently correct or drop source rows.
- Use only out-of-sample validation predictions. Read the saved training positive count and row count as context; never treat training predictions as validation results. Preserve ticker and year boundaries in calculations and displays.

## Diagnostic calculations

For each `(ticker, fold_year, model)`, calculate and save:

- validation row count, actual positive count/rate, and predicted positive count/rate;
- confusion counts `TP`, `FP`, `TN`, and `FN` for the existing `0.5` decision threshold, with the positive class meaning **adjusted close higher five exchange sessions later**;
- probability minimum, median, mean, maximum, and standard deviation, plus the fraction of probabilities below, equal to, and above `0.5`;
- accuracy, ROC-AUC, precision, recall, F1, and Brier score from the saved predictions, shown alongside the existing saved values;
- differences versus **both** saved baselines on the same ticker/year/session rows. Define `accuracy_delta = model_accuracy - baseline_accuracy`, `auc_delta = model_auc - baseline_auc`, and `brier_improvement = baseline_brier - model_brier`, so a positive difference always means an improvement for that metric.

Build ten fixed, equal-width probability bins: `[0.0, 0.1)`, ..., `[0.9, 1.0]`. For every ticker/year/model/bin, save the count, mean predicted probability, and observed positive fraction. Empty bins have count zero and null means/fractions. Mark nonempty bins with fewer than 20 observations as sparse; show their values but do not use them to claim calibration. Include a probability histogram and a reliability diagram with a `y=x` reference line. A constant-probability baseline should naturally occupy one bin, and its ROC-AUC of `0.5` should be labeled as a constant-score baseline, not evidence of discrimination. Explain that Brier score measures overall probability error and **does not by itself establish calibration**.

When a linked backtest is provided, show a separate trading context for each matching ticker/year/strategy: net cumulative return, difference from buy and hold, fraction invested, turnover, and fee/slippage assumptions. Do not infer an execution policy from classification metrics or present trading returns as predictive accuracy. Highlight where a model's predicted-positive rate and the strategy's fraction invested are similar, while explaining that next-session execution and end-of-fold boundaries prevent them from being identical by definition. Keep buy and hold and all cash visible as trading baselines.

## Report and outputs

Publish an immutable diagnostic run only after all validation, calculations, and rendering succeed:

```text
data/diagnostics/<run-id>/
├── fold_diagnostics.parquet
├── calibration_bins.parquet
├── trading_context.parquet       # only when a backtest is supplied
├── report.html
└── manifest.json
```

`report.html` is self-contained, opens locally without a server, and includes tables plus accessible, labeled inline charts. Begin with a short plain-language explanation of the five-session target, the `0.5` decision rule, and the two baselines. For each ticker/year, display the confusion counts, probability distribution, reliability diagram, and baseline differences. Include an explicit example of what an all-positive model can and cannot conclude. Show the fold years separately; do not pool their rows into a single apparent independent sample or rank a model as a deployment recommendation. If trading context exists, put it in a visibly separate section. State the report's data limitations and the source run IDs.

The manifest records source evaluation ID and manifest/output hashes; optional backtest ID and hashes; the `0.5` threshold and ten-bin convention; calculation/report version; per-fold coverage; software versions; UTC creation time; and output paths and SHA-256 hashes. Use a temporary directory and publish by rename. A failure leaves no partial published report and never changes the source evaluation or backtest. Identical source bytes and rules yield identical Parquet diagnostic rows; run IDs and timestamps may differ. Keep all files under Git-ignored `data/`.

## Optional MLflow access

Add `marketsignal track-diagnostics <diagnostic-run-id> [--data-dir data]`, using the optional `tracking` extra and the Spec 005 local store configuration. Verify the diagnostic manifest and all declared outputs before import. Create a `marketsignal-diagnostics` experiment with one run per diagnostic identity (run ID and manifest hash), tagged with the source evaluation ID/hash and optional backtest ID/hash. Log the report and manifest as artifacts and the per-fold diagnostic table as an artifact. Record the corresponding evaluation MLflow run ID when it is already present, without requiring it or modifying its completed run. Repeated imports return the finished MLflow run; a failed attempt stays visible and can be retried. A tracking failure does not affect the saved local report. Do not log raw market-provider responses, secrets, or speculative conclusions as metrics.

## Testing and verification

Use deterministic synthetic evaluation Parquet fixtures and a temporary data directory; tests require no provider request, API token, AWS account, or MLflow server. Cover:

1. Confusion counts and positive rates for all-positive, all-negative, mixed, and constant-probability predictions, including values exactly at `0.5`.
2. Probability summaries, ten-bin edge behavior for `0`, `0.1`, `0.5`, and `1`, empty bins, sparse-bin marking, and calibration-chart values.
3. Correct baseline differences and Brier/ROC-AUC interpretation, including a model worse than a baseline.
4. Rejection of altered output bytes, mismatched backtest links, duplicate/missing prediction rows, unequal candidate dates or labels, nonfinite/out-of-range probabilities, and disagreement with saved fold metrics, without publishing a partial run.
5. Consistent ticker/year isolation, deterministic Parquet rows, manifest hashes, DuckDB reads, and a self-contained report whose labels and tables match saved diagnostics.
6. Optional MLflow import identity, source links, report artifact, retry behavior, and a useful missing-extra error.

During implementation, run `pytest`, `ruff check`, and `ruff format --check`. Manually generate the report from the existing SPY/QQQ evaluation `20260923T201510626018Z-189e5ba7` and linked backtest `20260923T205213434489Z-b0a7c66d`. Confirm that QQQ 2024 logistic regression is reported as predicting positive on all 247 validation rows, matching the always-positive baseline's accuracy, and producing the same backtest return as buy and hold. Document the commands, output locations, and interpretation guide in the README.

## Acceptance criteria

- [x] The command verifies an immutable saved evaluation and optional matching backtest without fetching data or refitting.
- [x] Fold diagnostics expose class balance, confusion counts, probability behavior, calibration-bin counts, and both baseline comparisons with clear metric directions.
- [x] A portable report explains predictive results and optional trading context separately, including sparse bins and the limits of the five-session target.
- [x] Published outputs are hashed, reproducible, queryable, and all-or-nothing; invalid sources yield clear errors without altering source runs.
- [x] Optional MLflow import makes the report discoverable, links its source IDs, and is idempotent for completed imports.
- [x] Offline tests and quality checks pass; the README shows how to generate and interpret the report.

## Verification record

- `pytest -q`: 67 passed. `ruff check .` and `ruff format --check src tests`: passed.
- Generated and hash-verified local report `20260924T204052741852Z-4a33f9a0` from evaluation `20260923T201510626018Z-189e5ba7` and backtest `20260923T205213434489Z-b0a7c66d`.
- QQQ 2024 logistic regression predicted positive on all 247 validation rows. Its accuracy was `0.6275303643724697`, equal to always positive. Its net backtest return was `0.33582819413137077`, equal to buy and hold.
- Imported the final report into local MLflow as run `7af23b7fb13544f3996e9a8b62bbc5e6`.

## Known limits

These diagnostics describe a small set of historical, overlapping five-session outcomes from SPY and QQQ. Adjacent rows are not independent observations. Historical adjusted prices may include later provider corrections and are not point-in-time certified. Reliability diagrams built from a few hundred rows can be noisy, especially in sparse probability bins. Inspecting the 2022–2024 results to decide on new features, thresholds, or models makes those years part of the research process; a future untouched validation period is needed to assess such changes. The report is evidence about past model behavior, not a trading recommendation or proof of future returns.

## Implementation references

- [scikit-learn probability calibration and reliability diagrams](https://scikit-learn.org/stable/modules/calibration.html)
- [MLflow tracking artifacts and runs](https://mlflow.org/docs/latest/ml/tracking)
