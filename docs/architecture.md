# MarketSignal Architecture

## Overview

MarketSignal is an open-source market forecasting and intelligence platform designed to demonstrate production-style platform engineering and ML engineering practices.

The system will ingest financial market data, create reproducible datasets and features, train and evaluate predictive models, track experiments, and eventually expose forecasts through an API or user interface.

Future versions may include an AI research agent that combines quantitative model output with financial documents, market news, and other contextual information.

The architecture should remain intentionally simple during early development and evolve only when additional complexity provides clear value.

---

# Architecture Goals

The platform should demonstrate:

- reproducible data ingestion
- modular provider integrations
- data lake concepts
- feature engineering
- time-series machine learning
- model evaluation
- backtesting
- experiment tracking
- AWS infrastructure
- Infrastructure as Code
- automated testing
- CI/CD
- observability
- agentic AI workflows in later versions

The project should prioritize correctness, reproducibility, and maintainability over architectural complexity.

---

# High-Level Architecture

```text
                    Market Data Provider
                           │
                           ▼
                  Data Ingestion Layer
                           │
                           ▼
                  Raw Market Dataset
                     S3 / Parquet
                           │
                           ▼
                 Normalization Pipeline
                           │
                           ▼
                Processed Market Dataset
                     S3 / Parquet
                           │
                           ▼
                  Feature Engineering
                           │
                           ▼
               Feature + Label Datasets
                     S3 / Parquet
                           │
                ┌──────────┴───────────┐
                │                      │
                ▼                      ▼
          Model Training           Backtesting
                │                      │
                ▼                      ▼
             MLflow                Metrics
                │
                ▼
           Model Artifact
                │
                ▼
          Prediction Service
                │
                ▼
          API / CLI / Dashboard
```

---

# Local Development Architecture

The complete development workflow should be usable locally without requiring AWS.

```text
Market Data API
      │
      ▼
Python ingestion
      │
      ▼
Local Parquet
      │
      ▼
DuckDB
      │
      ▼
Feature pipeline
      │
      ▼
Features + labels
      │
      ▼
Local annual expanding-fold evaluation
      │
      ▼
Predictions + fold metrics + leaderboard + hashed manifest
```

The current `marketsignal evaluate` command fits four independent candidates per ticker and year, using only complete rows and training labels that end before validation starts. It saves one immutable local run under `data/evaluations/` after every requested fold succeeds. MLflow tracking and model artifacts are later work.

`marketsignal backtest` consumes a saved evaluation and its exact source price snapshot. It applies next-session-close execution, explicit costs, and a daily long-or-cash rule, then stores interval returns and fold summaries under `data/backtests/`. This research simulation uses adjusted closes as a return proxy; it does not establish executable historical fills.

DuckDB should be able to query Parquet datasets directly.

This provides a lightweight local environment without requiring a traditional database.

---

# Cloud Architecture

The initial AWS architecture should use low-cost managed services.

```text
                 EventBridge
                      │
                      ▼
                  Lambda
             Data Ingestion Job
                      │
                      ▼
                     S3
              Raw Market Data
                      │
                      ▼
              Processing Pipeline
                      │
                      ▼
                     S3
             Processed / Features
                      │
                      ▼
                Model Training
                      │
                      ▼
                   MLflow
                      │
                      ▼
                Model Artifact
                      │
                      ▼
               Prediction API
```

The exact training infrastructure can evolve later.

Early model training may occur locally or through GitHub Actions rather than introducing dedicated AWS ML infrastructure immediately.

---

# Data Architecture

The project should use a simple lake-style storage model.

Potential layout:

```text
data/
├── raw/
│   └── prices/
├── processed/
│   └── prices/
├── features/
├── labels/
├── evaluations/
├── backtests/
└── predictions/
```

Equivalent S3 layout:

```text
s3://marketsignal-data/
├── raw/
│   └── prices/
├── processed/
│   └── prices/
├── features/
├── labels/
└── predictions/
```

Parquet should be the preferred storage format.

Partitioning should be introduced only where it improves query or ingestion behavior.

Potential partitioning:

```text
raw/prices/
    ticker=SPY/
        year=2025/
        year=2026/
```

---

# Market Data Provider Architecture

External market-data APIs must be isolated behind provider abstractions.

Conceptually:

```text
MarketDataProvider
        │
   ┌────┴────────────┐
   │                 │
   ▼                 ▼
AlphaVantage     OtherProvider
   │                 │
   └──────┬──────────┘
          ▼
   Normalized Dataset
```

Downstream pipelines should consume a normalized internal schema rather than provider-specific responses.

Example normalized fields:

```text
ticker
timestamp
open
high
low
close
adjusted_close
volume
```

Additional fields can be introduced as needed.

---

# Feature Engineering Architecture

Feature generation must be reproducible and deterministic.

The first local implementation writes predictor features and future-derived labels to separate Parquet files. Features for a session use only the daily bars available through that session's completed close. Labels use the fifth subsequent expected exchange session. An XNYS calendar flags missing sessions for SPY and QQQ; windows crossing a missing session are not used. Each output pair has a manifest with source and output hashes. Historical provider corrections can still revise a backfilled dataset, so these files are not a point-in-time archive.

Potential initial features:

## Returns

- 1-day return
- 5-day return
- 20-day return

## Volatility

- rolling 5-day volatility
- rolling 20-day volatility
- rolling 60-day volatility

## Momentum

- moving averages
- moving-average distance
- RSI
- rolling momentum

## Volume

- relative volume
- rolling average volume
- volume deviation

Future features may include:

- VIX
- interest rates
- macroeconomic indicators
- sector-relative returns
- market breadth
- earnings-event features
- fundamentals

---

# ML Architecture

The first prediction target should remain intentionally simple.

Initial example:

```text
Predict:

P(SPY return over the next 5 trading days > 0)
```

This is a binary classification problem.

Potential initial models:

```text
Naive Baseline
      │
      ├── Logistic Regression
      ├── Random Forest
      └── XGBoost
```

Each model should be evaluated consistently.

A model leaderboard should compare models and baselines.

Example:

```text
Model                 Accuracy    ROC-AUC
------------------------------------------------
Always Positive          53%        0.50
Logistic Regression      55%        0.55
Random Forest            55%        0.57
XGBoost                  57%        0.60
```

Numbers above are illustrative only.

---

# Validation Architecture

Financial time-series models must not use random train/test splits.

Preferred approaches:

```text
Train               Validation
2015 ───────── 2022 │ 2023

Train                     Validation
2015 ───────────── 2023   │ 2024

Train                           Validation
2015 ───────────────── 2024    │ 2025
```

This expanding-window approach approximates how the model would behave over time.

Walk-forward evaluation should eventually become a reusable component of the platform.

---

# Experiment Tracking

MLflow should be used for model experiment tracking.

Each experiment should record:

```text
model
hyperparameters
training window
validation window
features
metrics
artifact
dataset identifier
```

This allows model runs to be reproduced and compared.

---

# Backtesting

Backtesting must remain logically separate from pure predictive evaluation.

The system should distinguish:

```text
Prediction Quality
        │
        ▼
classification metrics

Trading Strategy
        │
        ▼
backtesting metrics
```

Potential strategy metrics:

- cumulative return
- annualized return
- volatility
- Sharpe ratio
- maximum drawdown
- number of trades
- turnover

Any future backtest should account for transaction assumptions.

---

# Prediction Interface

Future versions may expose model results through a CLI or API.

Example output:

```text
Ticker: SPY
Horizon: 5 trading days

Probability positive: 63%
Expected regime: Positive
Model: xgboost-v3
Prediction timestamp: 2026-09-23
```

The initial API may use FastAPI.

---

# Future AI Research Agent

A future phase may introduce a research agent.

Conceptually:

```text
                        User
                         │
                         ▼
                  Research Agent
                         │
             ┌───────────┼───────────┐
             │           │           │
             ▼           ▼           ▼
       Model Forecast   SEC Data    Market News
             │           │           │
             └───────────┼───────────┘
                         ▼
                    RAG / Context
                         │
                         ▼
                  Structured Report
```

The agent should not replace the quantitative model.

Instead, the quantitative model provides one structured tool available to the agent.

Possible agent tools:

```text
get_prediction(ticker)

get_recent_prices(ticker)

get_financials(ticker)

search_sec_filings(ticker)

search_market_news(ticker)
```

This creates a clear separation between:

- quantitative modeling
- information retrieval
- LLM reasoning

---

# Infrastructure as Code

All AWS infrastructure must be provisioned through Terraform.

Potential modules:

```text
infrastructure/terraform/
├── modules/
│   ├── data_bucket/
│   ├── ingestion_lambda/
│   ├── scheduler/
│   └── iam/
│
└── environments/
    └── dev/
```

Infrastructure should remain small during the MVP.

---

# CI/CD

GitHub Actions should eventually provide:

```text
Pull Request
    │
    ├── Python tests
    ├── Ruff
    ├── Type checks
    ├── Terraform fmt
    ├── Terraform validate
    └── Security checks
```

Model training should not automatically occur on every pull request.

Scheduled or explicitly triggered model training can be added later.

---

# Observability

Initial observability should include:

- structured application logs
- ingestion failure visibility
- model training logs
- prediction metadata

AWS deployments should use CloudWatch.

Future versions may include:

- pipeline metrics
- dataset freshness
- model performance monitoring
- model drift detection

---

# Cost Philosophy

This project should prefer free or low-cost services.

Avoid introducing:

- always-on compute
- large managed clusters
- unnecessary databases
- expensive managed ML services

unless there is a clear architectural reason.

The project should demonstrate thoughtful cloud engineering rather than maximum AWS service usage.

---

# Architecture Evolution

The expected progression is:

```text
Phase 1
Local ML pipeline

        ↓

Phase 2
AWS ingestion + S3 data lake

        ↓

Phase 3
Automated training + MLflow

        ↓

Phase 4
Prediction API

        ↓

Phase 5
AI research agent

        ↓

Phase 6
Observability + automated agent workflows
```

Each phase should leave the system usable and testable.
