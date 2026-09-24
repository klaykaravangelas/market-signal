# Spec 005 — Local Experiment Tracking and Model Artifacts

## Status

Implemented and verified offline, including a local MLflow import of the existing SPY/QQQ evaluation `20260923T201510626018Z-189e5ba7` and linked backtest `20260923T205213434489Z-b0a7c66d`. Those historical runs are indexed as results-only; new evaluations save fitted models and training diagnostics.

## Goal and scope

Complete the local MLOps phase of [Spec 000](000-project-overview.md). Make the training and backtesting results from [Spec 003](003-model-training.md) and [Spec 004](004-backtesting.md) searchable in MLflow, while retaining their immutable Parquet files and manifests as the authoritative records. Save fitted estimators from new evaluation runs so a trained fold can be inspected and reproduced without refitting.

This spec covers local experiment tracking, model artifacts, and fold-level training diagnostics. It does not introduce automatic model selection, promotion, a model registry, hyperparameter search, online inference, scheduled training, AWS, or an MLflow service exposed to other machines. Tracking a historical evaluation created before this spec must remain possible, with its missing model artifacts and training diagnostics stated explicitly.

## Local setup and commands

- Add MLflow as an optional `tracking` dependency extra. The ingestion, feature, evaluation, and backtest commands continue to work without it. During implementation, resolve and exercise the extra against the project's supported Python version; do not assume that an untested Python/MLflow combination works.
- Use `data/mlflow/tracking.db` as the SQLite tracking backend and `data/mlflow/artifacts/` as the local artifact root by default. Resolve both against the selected `--data-dir` as absolute paths before constructing MLflow URIs. Create the database, artifact directory, and experiments on first tracking use; keep them under the existing Git-ignored `data/` tree. Support overriding the tracking URI and artifact root through explicit configuration for local testing and later deployment, without embedding credentials or account-specific paths.
- Add `marketsignal track <evaluation-run-id> [--backtest-run-id <run-id>] [--data-dir data]`. The optional backtest must identify the supplied evaluation run. The command reports the MLflow experiment/run IDs and local UI command. It does not ingest data, create features, train, or rerun a backtest.
- Document `uv sync --extra tracking` and a localhost-only MLflow UI command using the same tracking database. Running a server is optional for logging and for the CLI command. The documentation must say that the UI is a local research tool and should not be exposed publicly without authentication and access controls.

## Canonical evaluation artifacts

Extend **new** Spec 003 evaluation runs without changing their fold selection, predictors, model definitions, probabilities, or metrics. After each successful fit, save the fitted logistic-regression pipeline and random-forest estimator under the evaluation's temporary run directory. Use one documented serialization format supported by the project, with an explicit direct dependency if needed. Preserve the ordered seven-feature input schema with each artifact. Baselines have no fitted estimator; their fixed rule or training prevalence is already defined by the manifest and fold details.

Add a small, machine-readable training record per ticker, validation year, and candidate. It contains model name, applicable seed and parameters, training/validation row counts and date boundaries, fit duration measured with a monotonic clock, and available algorithm diagnostics (for example, logistic-regression iteration count and random-forest tree count). Mark diagnostics unavailable for baselines; do not invent loss curves or epochs for models that train in one fit call. Report a clear convergence or fit failure message in the CLI; a failed evaluation still publishes no run, as in Spec 003.

The evaluation manifest lists every new artifact and training-record output with relative path, SHA-256 hash, and schema/version where applicable. Publish them atomically with the existing evaluation files. Preserve compatibility with the existing Spec 004 backtest reader: additional manifest outputs must not change how it validates the original prediction, metric, and leaderboard files. Old evaluation runs have no model artifacts or training records and must never be silently retrained or described as containing fitted models. Only load serialized models from trusted local evaluation runs; loading a Python model artifact from an untrusted source can execute code.

## Tracking records and relationships

The `track` command verifies the source evaluation manifest, its run ID, all declared output hashes, and required schemas before writing to MLflow. If a backtest ID is supplied, verify its manifest, its declared output hashes, and its link to the evaluation run. Read only these saved results and manifests; current feature, label, or price files need not match their old hashes to index an immutable historical result. Log their recorded source hashes as lineage, without claiming the source datasets are still available. Reject altered, incomplete, or mismatched runs with a clear error.

Create two local experiments, `marketsignal-evaluation` and `marketsignal-backtest`. An evaluation import creates one parent run for its evaluation run ID, with nested child runs for each `(ticker, validation year, candidate)` in the saved fold metrics. A backtest import creates one parent run for its backtest run ID, with nested child runs for each `(ticker, validation year, strategy)` in the saved fold metrics. Link the backtest parent to its evaluation through exact run IDs and manifest hashes, not a name-based guess. When both IDs are supplied, finish or reuse the evaluation import before importing the backtest.

For evaluation tracking, record:

- on the parent: source run ID and manifest hash, created timestamp, ticker and fold coverage, target and feature-definition version, ordered feature list, all recorded input/output hashes and paths, fold configuration, software versions, and the source manifest as an artifact;
- on each child: ticker, fold year, candidate, training/validation boundaries and class counts, row counts, model parameters and seed, all saved predictive fold metrics, and the saved training diagnostics when available;
- for fitted candidates in new runs: a loadable MLflow scikit-learn model artifact derived from the canonical saved estimator, with its feature order and source artifact hash recorded. For baselines and historical runs, tag `model_artifact_available=false` and explain why. A historical run must be tagged `results_only=true` at the parent level.

For backtest tracking, record the source backtest run ID/hash and linked evaluation ID/hash on the parent, plus strategy timing, threshold, fees, slippage, software versions, and the source manifest. On children, log the saved trading fold metrics, including returns, cost drag, volatility, Sharpe where defined, drawdown, turnover, trade counts, and exposure. Keep predictive and trading metrics in their respective experiments and use names that preserve their meaning. Skip undefined metrics such as a null Sharpe instead of logging a made-up numeric value. MLflow values are an index of saved results; the Parquet outputs remain authoritative.

Use explicit logging rather than MLflow autologging so each run's contents are fixed by this spec and do not depend on autolog compatibility with a scikit-learn release. Tags/parameters must preserve exact source identifiers and hashes. Large structured metadata (ordered features, file hash maps, and fold detail) may be logged as JSON artifacts when MLflow's parameter/tag length limits make scalar fields unsuitable. Do not log API tokens, environment variables, or raw provider responses.

## Idempotency and failures

Define an import identity as the source run ID **and** manifest SHA-256 within its experiment. Before logging, search for an already finished parent with that identity and return its MLflow run ID instead of creating another successful copy. If the same run ID is already tracked with a different manifest hash, refuse the import as a source-integrity conflict. A failed partial MLflow attempt remains visibly failed; retrying may create a new attempt, but only one finished import may exist for an identity. Recheck for a finished import before declaring success. Document that this local workflow assumes one writer at a time; cross-process exactly-once coordination is outside scope.

A tracking or MLflow storage error exits nonzero and identifies the source run and failed stage. It must not modify or delete the immutable evaluation or backtest directory. Model-artifact publication during `evaluate` remains subject to Spec 003's all-or-nothing rule. Backtests continue to run without MLflow installed. Do not change metric values or suppress unfavorable results while importing.

## Training progress and verification

Report each candidate's ticker/fold start, completion, and elapsed fit time in the evaluation CLI, with fold context in any failure. The MLflow UI shows completed fold metrics and recorded fit diagnostics; these scikit-learn fits do not expose a meaningful live epoch-by-epoch learning curve. Do not present the sequence of yearly folds as progressive improvement in one model: each fold is a separate fit on an expanding historical window.

Automated tests use small synthetic evaluation/backtest fixtures and a temporary local SQLite MLflow store, with no API token, network service, or AWS credentials. Cover:

1. New evaluation model artifacts load and reproduce their saved validation probabilities on the exact recorded feature order; output hashes and the all-or-nothing publication rule hold.
2. The existing backtest reader accepts new evaluation manifests and rejects a changed source prediction file as before.
3. Evaluation and backtest imports log the expected parent/child coverage, lineage, parameters, predictive and trading metrics, and loadable fitted models.
4. Historical evaluations import as results-only where model artifacts or diagnostics were never saved; their linked historical backtests import without refitting or claiming models exist.
5. Repeated imports return existing finished run IDs; altered, mismatched, malformed, and partially missing sources fail; interrupted tracking leaves source runs untouched and supports a clear retry.
6. Null or unavailable metrics, baseline artifact absence, and secrets are handled as specified.

Run `pytest`, `ruff check`, and `ruff format --check` during implementation, plus a manual import of the existing SPY/QQQ evaluation `20260923T201510626018Z-189e5ba7` and backtest `20260923T205213434489Z-b0a7c66d`. Verify their relationship and historical `results_only` status in the local MLflow UI. Document the setup, import and UI commands, storage locations, model trust boundary, and interpretation of fold-level results in the README and architecture notes.

## Acceptance criteria

- [x] MLflow is optional; a local SQLite-backed tracking store and artifact directory work without a running server or cloud credentials.
- [x] New evaluations save hashed, loadable fitted estimators and fold-level training diagnostics without changing Spec 003 predictions or breaking Spec 004 backtests.
- [x] The `track` command verifies immutable inputs and logs evaluation and optional linked backtest runs with complete fold coverage and distinct predictive versus trading metrics.
- [x] Old runs import with honest results-only metadata and no fabricated model artifacts or diagnostics.
- [x] Import is idempotent for completed source identities; failures are visible and never damage source runs.
- [x] Offline tests and quality checks pass, and documentation explains setup, run lineage, fit progress, limitations, and safe local model loading.

## Known limits

MLflow organizes and displays experiments; it does not make historical data point-in-time accurate, establish statistical significance, or prove that a strategy will work in the future. A saved model is tied to its exact feature order, input snapshot, and library versions. Historical runs created before this spec can be inspected for saved metrics but cannot yield their original fitted estimators. Local SQLite and filesystem artifacts are suitable for one developer; shared or remote tracking, access control, backups, registry/promotion, and AWS deployment require later design.

## Implementation references

- [MLflow local database tracking](https://www.mlflow.org/docs/latest/ml/tracking/tutorials/local-database/)
- [MLflow tracking API and nested runs](https://mlflow.org/docs/latest/ml/tracking/tracking-api)
- [MLflow scikit-learn model logging](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html)
