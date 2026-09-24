# ADR 002 — Local MLflow tracking over immutable research runs

## Status

Accepted for Spec 005.

## Context

Specs 003 and 004 publish immutable local Parquet results with hashes. Experiment tracking should make those results easier to compare and inspect without turning MLflow into a prerequisite for training or backtesting. A single developer needs a low-cost setup and must be able to import the evaluation and backtest runs created before MLflow was added.

## Decision

Use an optional MLflow dependency, a local SQLite tracking database, and local filesystem artifacts. A separate `marketsignal track` command verifies and imports completed result runs. Evaluation and backtesting remain functional without MLflow. The saved manifests and Parquet files are the source of truth; MLflow is a searchable index and model-viewing tool. New evaluations retain hashed fitted estimators in their immutable output, then tracking logs them as loadable MLflow models. Historical evaluations import as results-only because their estimators were discarded.

## Consequences

Tracking failures cannot invalidate a successful evaluation or backtest. Retrying an import can reuse a finished MLflow run, while failed attempts remain visible. The local SQLite store assumes one writer and has no remote access controls or operational backup plan. Shared tracking, model promotion, and a hosted service need a separate design. Serialized estimators must only be loaded from trusted local runs.
