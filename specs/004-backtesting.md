# Spec 004 — Local Backtesting of Evaluation Predictions

## Status

Implemented and verified offline and against the local SPY/QQQ evaluation run `20260923T201510626018Z-189e5ba7`. The resulting backtest run is `20260923T205213434489Z-b0a7c66d`.

## Goal and scope

Use the out-of-sample predictions saved by [Spec 003](003-model-training.md) to measure the results of one fixed, auditable long-or-cash rule. Compare each model with simple trading baselines on identical dates and prices. This is a local research step after Phase 3 of [Spec 000](000-project-overview.md); the roadmap's Phase 4 remains MLOps.

The five-session target describes the probability that adjusted close will be higher five exchange sessions later. This first strategy **reconsiders exposure daily** using that probability; it does not hold each signal for exactly five sessions or create overlapping five-session trades. Record this distinction in the output and documentation. No shorting, leverage, position sizing, optimization, brokerage connection, live trading, MLflow, or AWS integration is in scope.

## Command and verified inputs

- Add `marketsignal backtest <evaluation-run-id> [--data-dir data] [--fee-bps 1] [--slippage-bps 5]`. The example run ID is the directory name under `data/evaluations/`. The two cost flags are nonnegative, finite basis points per one-way trade; one basis point is 0.01%. Their defaults are assumptions for this research simulation, not measured execution costs.
- Read the Spec 003 `manifest.json` and `predictions.parquet` from that run. Verify the manifest's run ID, supported feature version/target, output paths and SHA-256 hashes, prediction schema, and the requested ticker/fold/model combinations. Reject modified or malformed runs.
- Read `data/processed/prices/<ticker>.parquet` for each ticker in the evaluation. Require its SHA-256 hash to match the **source price hash recorded in the evaluation manifest**. If prices have since been reingested, instruct the user to rebuild features and rerun evaluation before backtesting. Do not silently combine predictions with a newer price snapshot. Current feature and label files need not remain unchanged because this command consumes the saved prediction run and matching source prices.
- Use the exchange calendar identifier recorded by new Spec 003 evaluation manifests. Existing SPY/QQQ runs, made before that field was added, use the XNYS calendar fixed by Spec 002. An older run for another ticker without a recorded calendar must be re-evaluated before backtesting.
- Validate the price schema, ticker, unique ascending exchange sessions, and finite positive adjusted closes. Validate unique prediction keys `(ticker, fold_year, model, session_date)`, finite probabilities in `[0, 1]`, chronological dates, and identical prediction session sets for all four Spec 003 candidates within each ticker/fold. Reject missing interior prediction sessions, missing execution or return price sessions, and dates outside the fold. The known missing predictions at the end of a fold, whose five-session labels could not finish inside that year, are handled by ending the simulation early as defined below.
- Use only `(ticker, fold_year, model, session_date, probability_positive)` to make trading decisions. `actual`, `predicted`, `label_end_date`, fold metrics, and leaderboard values are never strategy inputs. The backtest does not refit models or fetch prices.

## Signal, execution, and accounting rules

Run each ticker and validation year independently, starting in cash with normalized equity `1.0`. A forecast for session `t` uses information available after that session's close. Its order executes at the **next expected exchange session's close**, `t+1`; the resulting position earns the adjusted-close return from `t+1` to `t+2`. A forecast stamped `t` must never earn the return from `t` to `t+1`.

At each execution close, hold the asset at weight `1` if `probability_positive >= 0.5`; otherwise hold cash at weight `0`. Later daily signals may change that position at their respective next-session closes. No threshold is selected using the validation results. Cash earns zero, and the strategy does not borrow or short. Treat the adjusted-close series as a dividend- and split-adjusted **return proxy**, not an executable quoted price.

Require prediction dates to be consecutive expected exchange sessions from the fold's first session through the session five expected exchange sessions before year-end. Reject a shorter or gapped series; the normal five-session label boundary is the only accepted reason for absent year-end predictions. For each signal at `t`, record one return interval from close `t+1` to close `t+2`, with price return `adjusted_close[t+2] / adjusted_close[t+1] - 1`. The first executable close is the second session of the validation year. After the final interval, liquidate any remaining position at close `t_last+2`. Stop there, rather than inventing predictions for the final sessions omitted by Spec 003's label boundary. Require all these dates to fall inside the validation year, and report the first execution, final liquidation, number of return intervals, and excluded end-of-year sessions. Do not carry a position across fold boundaries.

Apply `cost_rate = (fee_bps + slippage_bps) / 10_000` to the absolute change in position at each execution close, including initial entry and final liquidation. A switch between cash and the asset has one unit of one-way turnover. With starting equity `E`, prior position `p_old`, new position `p`, and adjusted-close return `r` over the interval:

```text
entry_cost_fraction = cost_rate * abs(p - p_old)
exit_cost_fraction  = cost_rate * p, on the final interval only
E_next = E * (1 - entry_cost_fraction) * (1 + p * r)
             * (1 - exit_cost_fraction)
net_interval_return = E_next / E - 1
```

The final exit cost is applied after the final price move. Compute a parallel gross equity path with the same positions and no costs. Require total cost fraction for a single trade to be below `1`. Record entries, exits, turnover, and cost drag. No market impact, taxes, financing, or cash interest are modeled.

## Trading baselines and comparison

Backtest all four candidates saved by Spec 003, plus:

- **Buy and hold:** enter at the first executable close, hold weight `1` for every common interval, and liquidate at the same final close, with the same costs.
- **All cash:** hold weight `0` throughout the same intervals; equity stays at `1.0`.

Every strategy for a ticker/fold uses the same price intervals, initial equity, cost settings, and liquidation close. The always-positive candidate must match buy and hold exactly; treat a mismatch as a validation failure. Keep unfavorable model results. Report the difference between each strategy's net cumulative return and buy and hold's net cumulative return for the same fold, without treating that difference as statistical proof of an edge.

## Results and metrics

Save one row per ticker/fold/strategy/return interval with signal date (null for trading baselines), execution close date, return end date, probability (null for trading baselines), target position, opening turnover, final liquidation turnover, adjusted-close price return, gross and net interval returns, gross and net ending equity. Dates and positions must make the execution lag auditable.

For each ticker/fold/strategy calculate:

- gross and net cumulative return: final equity minus `1`;
- cost drag: gross cumulative return minus net cumulative return;
- annualized net return: `final_net_equity ** (252 / N) - 1`, where `N` is the number of return intervals;
- annualized volatility: sample standard deviation of net interval returns times `sqrt(252)`; store `NULL` when there are fewer than two intervals;
- annualized Sharpe ratio: mean net interval return divided by its sample standard deviation, times `sqrt(252)`, with zero risk-free rate; store `NULL` when the denominator is zero or there are fewer than two intervals;
- maximum drawdown: the largest percentage decline of net equity from its running high, including initial equity `1.0`;
- entry count, exit count, total one-way turnover (including liquidation), and fraction of intervals invested.

The 252-session annualization is a comparison convention. Fold durations are shortened at the end by unavailable labels, and the folds are not a continuous live portfolio. A summary may give the unweighted mean and standard deviation of fold metrics per ticker/strategy with a fold count. Do not compound across year boundaries or rank a strategy as a production recommendation. Keep prediction-quality metrics from Spec 003 separate from trading metrics in this spec.

## Outputs and reproducibility

Publish one immutable directory only after every ticker, fold, and strategy succeeds:

```text
data/backtests/<run-id>/
├── intervals.parquet
├── fold_metrics.parquet
├── summary.parquet
└── manifest.json
```

The manifest records the source evaluation run ID and manifest/prediction hashes; each source price path and hash; ticker/fold/model coverage; the signal and execution conventions; threshold, fee and slippage assumptions; first/last tradable close and row counts per fold; output paths and hashes; relevant software versions; and UTC creation time. Use a temporary directory and publish by rename after all files and hashes are complete. A failed validation or calculation must leave no partial published run. Identical input bytes and settings should produce identical Parquet result rows, apart from run IDs and timestamps. All files must be directly queryable with DuckDB and remain under Git-ignored `data/`.

## Testing and verification

Use deterministic, small synthetic prediction and adjusted-price Parquet fixtures; no provider request, AWS credentials, or MLflow server. Tests should cover:

1. A signal at `t` first affects the `t+1` to `t+2` return, with no same-session or earlier return leakage.
2. Exact entry, switch, final liquidation, and cost arithmetic, including zero-cost settings and always-positive matching buy and hold.
3. Consecutive exchange sessions, fold boundaries, the last usable signal, and equal return intervals across every strategy.
4. Correct cumulative return, annualization, volatility, Sharpe `NULL` case, maximum drawdown, turnover, trade counts, and exposure on known price paths.
5. Rejection of changed price files, altered evaluation outputs, duplicate/missing prediction rows, malformed probabilities, invalid costs, and missing price sessions without publishing a partial run.
6. Ticker and fold isolation, repeatability of result rows, manifest hashes, and DuckDB reads of all output files.

Run `pytest`, `ruff check`, and `ruff format --check` during implementation. Document the command, costs, timing, outputs, and limitations in the README. A manual run should use the completed Spec 003 SPY/QQQ evaluation and report the fold windows and comparison with buy and hold.

## Acceptance criteria

- [x] The command consumes a verified Spec 003 run and matching processed prices, rejecting stale or malformed inputs.
- [x] A forecast acts no earlier than the next exchange session's close, and all strategies use the same return intervals.
- [x] Long-or-cash candidates, buy and hold, and all cash follow the specified cost and liquidation rules.
- [x] Auditable interval rows, fold metrics, and a fold summary are saved with unfavorable results retained.
- [x] The manifest and immutable run directory identify inputs, timing, costs, versions, and output hashes.
- [x] Invalid data or computation failures produce clear errors without publishing partial results.
- [x] Offline tests and quality checks pass, and the README explains how to run and interpret the backtest.

## Known limits

Historical adjusted closes may include provider corrections made after their original sessions and do not represent guaranteed executable fills. This dataset is not point-in-time certified. The fixed close execution delay and configurable cost assumptions do not capture actual order placement, bid-ask dynamics, market impact, taxes, or cash yield. SPY and QQQ are selected surviving assets; this comparison does not address survivorship bias. The strategy is derived from a five-session target but rebalances daily. Reusing the same validation years to choose a winning model or rule introduces selection bias; a future untouched period is required before making broader performance claims. Backtest results are research measurements, not evidence of future returns.
