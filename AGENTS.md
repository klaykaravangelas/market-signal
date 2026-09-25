# MarketSignal Engineering Guidelines

## Project Overview

MarketSignal is an open-source market forecasting and intelligence platform that combines:

- Platform engineering
- optional AWS and Infrastructure as Code work when justified
- Machine learning
- MLOps
- Financial market data
- AI/LLM-based research workflows

The project should be built incrementally, with each major feature defined by a specification before implementation. Local development is the priority; AWS deployment is deferred and optional unless a later specification identifies a clear need.

The goal is not to claim that the system can reliably beat the market. The goal is to build a production-style ML platform that demonstrates sound software engineering, platform engineering, and ML engineering practices.

---

## Development Workflow

Development should follow this general process:

1. Define or update a specification under `specs/`.
2. Review the specification and acceptance criteria.
3. Implement only the functionality in scope.
4. Add or update tests.
5. Run relevant tests and quality checks.
6. Verify each acceptance criterion.
7. Update documentation when architecture or behavior changes.

When implementing a spec, avoid adding unrelated features unless they are necessary to satisfy the requirements.

---

## Repository Structure

```text
marketsignal/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── .gitignore
│
├── docs/
│   ├── architecture.md
│   └── adr/
│
├── specs/
│   ├── 000-project-overview.md
│   ├── 001-market-data-ingestion.md
│   ├── 002-feature-engineering.md
│   ├── 003-model-training.md
│   ├── 004-backtesting.md
│   ├── 005-local-experiment-tracking.md
│   └── 006-model-diagnostics.md
│
├── src/
│   └── marketsignal/
│
├── tests/
│
└── infrastructure/             # optional future cloud work
    └── terraform/
```

---

## Primary Technology Choices

### Application

- Python 3.13+
- `uv` for dependency and environment management
- Type hints for public interfaces
- `pytest` for testing
- `ruff` for linting and formatting

### Data

- Apache Parquet for persisted datasets
- DuckDB for local querying and analytics
- Amazon S3 for cloud object storage

### Machine Learning

Initial models should favor simple and explainable approaches:

- Logistic Regression
- Random Forest
- XGBoost or LightGBM

More complex models should only be introduced when there is a clear experimental reason.

Potential supporting libraries:

- scikit-learn
- XGBoost
- LightGBM
- sktime

### MLOps

- MLflow for experiment tracking
- MLflow Model Registry if model promotion is added later
- Reproducible training runs
- Explicit recording of features, parameters, metrics, and model artifacts

### Optional Cloud Infrastructure

- Terraform
- AWS
- Amazon S3
- AWS Lambda where appropriate
- Amazon EventBridge for scheduled workloads
- CloudWatch for logs and operational monitoring

Prefer serverless or very low-cost AWS services where practical.

### CI/CD

- GitHub Actions
- Automated testing
- Linting
- Terraform validation if cloud infrastructure is introduced
- Security scanning where practical

---

## Software Engineering Guidelines

- New functionality should include appropriate tests.
- Prefer small, composable modules.
- External systems should be accessed through interfaces or abstractions.
- Avoid tightly coupling domain logic to AWS services.
- Local development should work without AWS credentials whenever practical.
- Infrastructure must be managed through Terraform rather than manual configuration.
- Configuration should be externalized rather than hard-coded.
- Secrets must never be committed to the repository.
- Prefer idempotent pipelines and operations.
- Favor simple implementations before adding distributed or complex infrastructure.

---

## Market Data Guidelines

Market data providers must be abstracted behind an interface.

Example conceptual interface:

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

The system should make it possible to switch providers without rewriting downstream processing.

Potential providers may include:

- Alpha Vantage
- Yahoo Finance-compatible sources
- Other free or low-cost APIs

Provider-specific code should remain isolated.

Raw provider data and normalized internal datasets should be treated separately.

---

## ML Engineering Guidelines

Financial time-series ML requires additional care.

### Data Leakage

Never allow future information to influence a prediction or training example.

Feature generation must ensure that features at time `t` only use information that would have been available at time `t`.

### Train/Test Splitting

Do not use random train/test splits for time-series forecasting.

Use:

- chronological splits
- walk-forward validation
- expanding-window validation
- rolling-window validation

as appropriate.

### Baselines

Every predictive model must be compared against at least one simple baseline.

Examples:

- always predict positive return
- historical majority class
- previous-period behavior
- logistic regression

Complexity alone does not justify a model.

### Evaluation

Do not evaluate models solely on raw classification accuracy.

Potential metrics include:

- ROC-AUC
- precision
- recall
- F1
- Brier score
- calibration
- directional accuracy

When strategy simulations are introduced, additional metrics may include:

- cumulative return
- volatility
- Sharpe ratio
- maximum drawdown

Backtesting metrics must clearly distinguish predictive performance from trading performance.

### Reproducibility

Training jobs should record:

- dataset/version
- feature set
- training period
- validation period
- model type
- hyperparameters
- random seed where applicable
- evaluation metrics
- resulting model artifact

---

## Backtesting Guidelines

Backtests must avoid unrealistic assumptions.

Specifically consider:

- look-ahead bias
- survivorship bias
- transaction costs
- slippage
- market timing assumptions
- data availability timing

A strong result should not be presented as evidence that future returns are guaranteed.

---

## AWS Guidelines

AWS deployment is deferred. If it later becomes part of the project, do not add AWS services simply for architectural complexity.

Prefer:

- S3 for durable data
- EventBridge for schedules
- Lambda for lightweight jobs
- CloudWatch for logging
- IAM roles with least privilege

Use higher-cost or operationally heavier services only when they provide meaningful value.

Add cost controls when cloud infrastructure is introduced.

At minimum, consider:

- AWS Budgets
- cost alerts
- minimal retention periods
- lifecycle policies
- resource tagging

---

## Terraform Guidelines

- Use reusable modules where appropriate.
- Keep environments configurable.
- Avoid embedding account-specific values.
- Run `terraform fmt`.
- Run `terraform validate`.
- Keep IAM permissions narrowly scoped.
- Tag provisioned resources.
- Do not commit Terraform state.
- Prefer remote state only when the project reaches a stage where multiple environments require it.

---

## Testing Guidelines

The project should use multiple levels of testing where appropriate.

### Unit Tests

Use for:

- feature calculations
- transformations
- validation logic
- model utilities
- provider adapters using mocked responses

### Integration Tests

Use for:

- market-data ingestion
- Parquet generation
- DuckDB queries
- interactions between pipeline stages

### Infrastructure Tests

Initially:

- `terraform fmt -check`
- `terraform validate`

Additional infrastructure testing may be added later.

---

## Documentation Guidelines

Important architectural decisions should be documented.

Use ADRs under:

```text
docs/adr/
```

when choosing between meaningful architectural alternatives.

Examples:

- Parquet versus relational storage
- Lambda versus ECS
- MLflow deployment architecture
- market-data provider selection

Documentation should explain why a decision was made, not merely what was chosen.

---

## Pull Request Expectations

Every significant feature should reference its specification.

A PR should explain:

- what was implemented
- which spec it implements
- tests performed
- acceptance criteria satisfied
- architectural decisions made
- known limitations
- follow-up work

---

## Codex Instructions

Before implementing a requested feature:

1. Read this `AGENTS.md`.
2. Read the relevant specification.
3. Inspect existing code and architecture.
4. Identify conflicts between the spec and current implementation.
5. Prefer the simplest implementation that satisfies the spec.
6. Do not implement future specs unless required by the current task.
7. Add tests.
8. Run appropriate checks.
9. Report which acceptance criteria were satisfied.

If a specification is ambiguous, favor the existing architecture and the smallest reasonable implementation rather than inventing large new subsystems.
