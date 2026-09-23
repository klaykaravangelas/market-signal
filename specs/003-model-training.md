# Spec 003 — Local Model Training and Chronological Evaluation

## Status

Implemented and verified offline and with local SPY/QQQ data covering 2018–2024. The completed three-fold 2022–2024 evaluation is `20260923T201510626018Z-189e5ba7`.

## Goal and scope

Compare simple classifiers with explicit baselines for the five-session positive-return target from [Spec 002](002-feature-engineering.md). Train and evaluate each ticker independently using expanding chronological folds, then save enough detail to reproduce and audit every result. This is Phase 3 of [Spec 000](000-project-overview.md).

This phase measures prediction quality. It does not simulate trades, publish forecasts, deploy a model, tune hyperparameters against a holdout set, or require MLflow or AWS. Model artifacts and experiment tracking can build on these records in a later phase.

## Inputs and command

- A local `marketsignal evaluate` command accepts one or more tickers, a data directory, a first validation calendar year, and a number of consecutive annual folds. Example: `marketsignal evaluate SPY QQQ --first-validation-year 2022 --folds 3`.
- The command reads `data/features/<ticker>.parquet`, `data/labels/<ticker>.parquet`, and `data/features/<ticker>.manifest.json`. It uses the manifest verifier from Spec 002, checks that the current source price file still matches the manifest's source hash, and refuses stale or inconsistent inputs with an instruction to rebuild features.
- Use only feature-definition version `v1` and the five-session target in this phase. Reject an unexpected schema, duplicate `(ticker, session_date)` keys, inconsistent ticker, invalid target, or label whose end date is not after its feature date.
- Join features and labels on `(ticker, session_date)` using an explicit column list. Predictor columns are exactly `return_1`, `return_5`, `return_20`, `volatility_5`, `volatility_20`, `ma_distance_20`, and `relative_volume_20`. `session_date`, `label_end_date`, `future_return_5`, `target_positive_5`, ticker, and manifest metadata are never predictors.
- Drop rows with an unknown label or any `NULL` or non-finite predictor. Count and report dropped rows by reason. Do not impute missing features or silently substitute a target value.
- A short January–March 2024 sample has only 17 complete labeled rows per ticker and is insufficient for model evaluation. Ingest and build features from several years of history before using this command for a real comparison.

## Expanding chronological folds

Each validation fold covers one full calendar year. Require the source price history to include the first and last expected exchange sessions of each requested validation year. For validation year `Y`:

```text
training:   session_date < first session of Y
            AND label_end_date < first session of Y
validation: session_date within Y
            AND label_end_date <= last session of Y
```

Training expands to include older labeled rows as `Y` advances. The `label_end_date` rule excludes training examples whose five-session outcome crosses into the validation year. Validation excludes examples whose outcome extends beyond that fold's year. Neither features nor labels from another ticker enter a ticker's fold. Never shuffle or use a random train/test split.

All models and baselines for a ticker and fold must use identical training and validation rows. Require at least 500 complete training rows and 100 complete validation rows, with both target classes present in each set. If a requested fold does not meet these conditions, fail with its ticker, year, row counts, and class counts; do not produce a partial leaderboard. These are evaluation guardrails, not claims that the resulting sample is statistically sufficient for a profitable strategy.

## Models and baselines

Use a fixed feature order and fixed, recorded parameters for the first comparison:

| Candidate | Definition |
| --- | --- |
| Always positive | Predict class `1` and probability `1.0` for every validation row. |
| Training prevalence | Predict the training positive-class fraction for every validation row; classify as `1` when it is at least `0.5`. |
| Logistic regression | `StandardScaler` fitted on the training fold only, followed by scikit-learn `LogisticRegression(C=1.0, max_iter=1000)`. |
| Random forest | scikit-learn `RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=1)`. |

The model implementations may use scikit-learn, added through `uv`. Set and record every applicable random seed. Fit a fresh scaler and model for every ticker and fold. Validation data may be transformed using training-fitted parameters but must never influence fitting. If a model fails to converge or score, surface the failure with ticker, fold, and model name rather than dropping it from the comparison.

XGBoost or LightGBM is deferred until the simpler models and baselines are working and a separate experiment justifies the dependency. No hyperparameter search is part of this spec.

## Evaluation

For every candidate and fold, save each validation row's ticker, fold year, session date, label end date, true target, predicted class, and estimated probability of class `1`. Probabilities must be finite and within `[0, 1]`; use the same decision threshold (`>= 0.5`) for all candidates.

Calculate and report these metrics for every candidate on the same rows:

- accuracy
- ROC-AUC
- precision, recall, and F1 for positive class `1` (`zero_division=0`)
- Brier score
- validation positive-class rate and row count

Because both classes are required in validation, ROC-AUC must be defined for every completed fold. Keep fold-level results visible. A summary leaderboard may show the unweighted mean and standard deviation of each metric across folds, along with the fold count. Do not select a model solely from one metric or present these results as trading returns. Probability estimates are not claimed to be calibrated; the Brier score is diagnostic.

## Outputs and reproducibility

Write one immutable local run directory under `data/evaluations/<run-id>/` only after all requested ticker/fold/model work succeeds:

```text
data/evaluations/<run-id>/
├── predictions.parquet
├── fold_metrics.parquet
├── leaderboard.parquet
└── manifest.json
```

The manifest records the command configuration, input paths and SHA-256 hashes, feature version and ordered feature list, target definition, each ticker's train/validation date ranges and class counts by fold, baseline and model parameters, random seed, Python/scikit-learn/DuckDB versions, output hashes, and run time in UTC. The run files must be queryable directly with DuckDB. Identical input bytes and configuration should produce the same predictions and metrics; run IDs and timestamps may differ. Keep negative performance results; a failed model fit instead reports an error and publishes no run.

Use a temporary run directory and publish it only after all files and their hashes are complete. Generated evaluations remain under the Git-ignored `data/` directory.

## Testing and verification

Automated tests use deterministic synthetic feature and label Parquet fixtures, without a data-provider request or AWS credentials. Cover:

1. Exact fold boundaries and exclusion of training labels that end in validation, plus validation labels that end after the fold year.
2. The seven-column predictor allowlist, `NULL` filtering, and rejection of malformed or stale inputs.
3. Identical rows across candidates within a fold and ticker isolation.
4. Baseline probabilities derived only from each training fold and scaler statistics fitted only on that fold.
5. Metric calculations against small known prediction sets, including a candidate worse than a baseline.
6. Insufficient sample or one-class folds failing clearly, without publishing partial output.
7. Repeated runs on the same input producing identical prediction and metric rows, and a DuckDB read of all output files.

Run `pytest`, `ruff check`, and `ruff format --check` for implementation. A manual local run requires several years of SPY or QQQ data and should record the requested folds and row counts. The completed run above used 2022–2024 folds with 246, 245, and 247 validation rows per ticker.

## Acceptance criteria

- [x] The command reads verified Spec 002 feature and label files and uses only the seven approved predictors.
- [x] Each ticker is evaluated independently in annual expanding folds with the five-session label boundary enforced.
- [x] Two naive baselines, logistic regression, and random forest use identical fold rows and fixed, recorded parameters.
- [x] All specified fold metrics and row-level predictions are saved, including unfavorable results.
- [x] A manifest and immutable run directory make inputs, configuration, software versions, and outputs traceable.
- [x] Invalid data, stale inputs, inadequate folds, and model failures produce clear errors without publishing partial results.
- [x] Offline tests and quality checks pass; the README documents the command, outputs, and data requirements.

## Known limits

Historical adjusted prices may contain corrections made after the original sessions. These evaluations are research comparisons on the captured dataset, not point-in-time certified forecasts or trading backtests. Choosing a model based on these folds also requires a separate untouched future evaluation before making broader performance claims.
