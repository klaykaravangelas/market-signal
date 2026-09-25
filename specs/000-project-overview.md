# Spec 000 — MarketSignal Project Overview

## Status

Active roadmap. The local data, feature, evaluation, backtesting, experiment-tracking, and [model-diagnostics](006-model-diagnostics.md) specs are implemented and verified locally. AWS deployment is deferred and optional.

---

# Goal

Build an open-source, production-style financial market forecasting and intelligence platform.

MarketSignal should demonstrate practical experience across:

- platform engineering
- Python
- financial data pipelines
- machine learning
- MLOps
- APIs
- observability
- CI/CD
- AI and agentic workflows

Cloud infrastructure may be explored later if it adds enough value to justify its cost and complexity.

The project is intended primarily as a technical portfolio and learning project.

It is not intended to claim guaranteed investment performance.

---

# Core Product Idea

The initial system should answer questions such as:

```text
What is the probability that SPY will have a positive return
over the next five trading days?
```

Rather than attempting to predict an exact future stock price, the initial models should focus on probabilistic or classification-based outputs.

Example:

```text
Ticker: SPY

5-Day Forecast
--------------------------------
Probability positive: 63%
Model: XGBoost
Validation ROC-AUC: 0.59
Prediction timestamp: 2026-09-23
```

---

# Initial Scope

The MVP should support:

- historical daily market data
- a small configurable list of securities
- normalized market datasets
- Parquet storage
- feature engineering
- classification-based prediction
- multiple ML models
- simple baselines
- time-aware validation
- model comparison
- experiment tracking
- reproducible local development

Initial symbols may include:

```text
SPY
QQQ
```

Additional securities can be added after the architecture is validated.

---

# Initial Prediction Target

The first prediction target should be:

```text
Will the selected asset have a positive return
over the next five trading days?
```

Expressed as:

```text
target = 1 if future_5_day_return > 0 else 0
```

The model should output a probability.

Example:

```text
P(5-day return > 0) = 0.63
```

---

# Initial Features

The first feature set should remain simple.

Potential features include:

## Returns

- 1-day return
- 5-day return
- 20-day return

## Volatility

- 5-day rolling volatility
- 20-day rolling volatility
- 60-day rolling volatility

## Momentum

- moving-average relationships
- rolling momentum
- RSI

## Volume

- rolling average volume
- relative volume

More advanced features should be added only after the basic modeling pipeline works.

---

# Initial Models

At least the following models should be considered:

```text
Naive baseline
Logistic Regression
Random Forest
XGBoost or LightGBM
```

The system must support comparing these models consistently.

Complex models such as deep neural networks should not be introduced during the MVP unless there is a specific experiment being conducted.

---

# Validation Requirements

Random train/test splitting must not be used.

The system should use chronological validation.

Preferred approaches:

- expanding-window validation
- walk-forward validation

Example:

```text
Training       Validation
2015-2022      2023

2015-2023      2024

2015-2024      2025
```

Feature generation must prevent future-data leakage.

---

# Model Evaluation

Models should be evaluated using several metrics.

Initial metrics:

- accuracy
- ROC-AUC
- precision
- recall
- F1

Probability calibration metrics may be introduced later.

A model must be compared against a naive baseline.

The project should retain negative experimental results.

A more complex model performing worse than a simple baseline is considered a valid and useful result.

---

# Data Architecture

Data should initially be stored as Parquet.

Local development:

```text
local filesystem
    +
Parquet
    +
DuckDB
```

Optional future cloud development:

```text
Amazon S3
    +
Parquet
```

If cloud storage is introduced, keep the same logical dataset structure where practical.

---

# Market Data Provider

The application must not tightly couple itself to a single provider.

A provider abstraction should be introduced.

Conceptually:

```python
class MarketDataProvider(Protocol):
    def get_prices(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> DataFrame:
        ...
```

The first implementation can use one free market-data source.

Additional providers can be added later.

---

# Experiment Tracking

MLflow should be used to track experiments.

Each model run should eventually capture:

- model type
- parameters
- training period
- validation period
- feature configuration
- evaluation metrics
- resulting model artifact

---

# Optional Cloud Scope (Deferred)

The platform should remain usable entirely locally. AWS is not required for the current roadmap and should be reconsidered only when a concrete use case justifies its ongoing cost and operational work.

One possible future cloud architecture is:

```text
EventBridge
     │
     ▼
Lambda
     │
     ▼
S3 Data Lake
     │
     ▼
Feature / Training Pipeline
```

If AWS deployment is introduced, define its infrastructure using Terraform.

The architecture should favor serverless and low-cost services.

---

# API Scope

A prediction API is not required for the first implementation.

A later version may provide:

```http
GET /predictions/SPY?horizon=5d
```

Potential response:

```json
{
  "ticker": "SPY",
  "horizon_days": 5,
  "probability_positive": 0.63,
  "model": "xgboost-v3"
}
```

FastAPI is the preferred initial framework if an API is introduced.

---

# Future AI / Agent Scope

A future version may introduce a market research agent.

The agent may combine:

- quantitative forecasts
- market price data
- SEC filings
- company financial information
- market news
- macroeconomic data

Potential workflow:

```text
User
 │
 ▼
Research Agent
 │
 ├── Prediction Tool
 ├── Market Data Tool
 ├── SEC Retrieval Tool
 └── News Retrieval Tool
 │
 ▼
Structured Research Report
```

The agent should use the predictive model as one source of evidence rather than treating model output as fact.

---

# Out of Scope for MVP

The following are explicitly out of scope initially:

- intraday trading
- high-frequency trading
- options pricing
- futures trading infrastructure
- automated brokerage execution
- automated investment decisions
- reinforcement-learning trading agents
- deep-learning price prediction
- real-time streaming
- large distributed compute systems
- Kubernetes
- full production multi-account AWS architecture

These may be explored later if they provide clear educational value.

---

# Milestones

## Phase 1 — Local Data Pipeline

Build:

```text
Market API
   ↓
Ingestion
   ↓
Parquet
   ↓
DuckDB
```

Deliverables:

- provider abstraction
- historical price ingestion
- normalized dataset
- Parquet output
- unit tests
- integration tests

---

## Phase 2 — Feature Engineering

Build:

```text
Raw Data
   ↓
Feature Pipeline
   ↓
Training Dataset
```

Deliverables:

- return features
- volatility features
- momentum features
- volume features
- target generation
- leakage-safe transformations

---

## Phase 3 — Model Training

Build:

```text
Feature Dataset
     ↓
Training Pipeline
     ↓
Models
```

Deliverables:

- naive baseline
- logistic regression
- tree-based model
- chronological validation
- model comparison

---

## Phase 4 — Backtesting and MLOps

Introduce:

- local backtesting from saved evaluations
- MLflow
- experiment tracking
- model artifacts
- reproducible training configuration

---

## Phase 5 — Model Diagnostics

Introduce:

- inspection of saved out-of-sample predictions
- class-balance and confusion-matrix diagnostics
- probability-distribution and calibration views
- baseline comparisons
- a portable local report and optional MLflow artifact

---

## Phase 6 — Prediction Service

Introduce:

- FastAPI
- prediction endpoint
- model loading
- structured responses

---

## Phase 7 — AI Research Agent

Introduce:

- LLM tool calling
- RAG
- SEC retrieval
- news retrieval
- structured research reports
- local/open-source model support where practical

---

## Optional Future Phase — AWS Deployment

Reconsider a small AWS deployment only if a specific product or learning goal warrants its cost. Define its scope in a separate spec before implementation. If pursued, use Terraform, cost controls, and the fewest services needed.

---

# Success Criteria

The project should ultimately demonstrate that:

1. Market data can be ingested reproducibly.
2. Datasets can be reconstructed from source data.
3. Feature generation is tested and leakage-safe.
4. Models use time-aware validation.
5. Models are compared with meaningful baselines.
6. Experiments are reproducible.
7. Saved model predictions can be interpreted through local diagnostics and baseline comparisons.
8. The core workflow runs locally without cloud credentials or ongoing cloud costs.
9. CI/CD validates the code; any future cloud infrastructure is validated if introduced.
10. The system can evolve into an agentic market intelligence platform without requiring a major redesign.

---

# Guiding Principle

The project should optimize for:

```text
correctness
    >
reproducibility
    >
simplicity
    >
architectural sophistication
```

The objective is to demonstrate thoughtful engineering, not maximum complexity.
