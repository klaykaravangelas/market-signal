"""Local, delayed-execution backtests of saved evaluation predictions."""

import json
import math
import platform
import re
import shutil
import statistics
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import duckdb
import exchange_calendars as xcals
from exchange_calendars.errors import CalendarError

from marketsignal.evaluation import MODELS, PREDICTION_SCHEMA
from marketsignal.features import FEATURE_VERSION, HORIZON, PRICE_SCHEMA, _sha256, _write_parquet
from marketsignal.ingestion import normalize_tickers

STRATEGIES = (*MODELS, "buy_and_hold", "all_cash")
OUTPUT_NAMES = ("intervals.parquet", "fold_metrics.parquet", "summary.parquet")
INTERVAL_SCHEMA = [
    ("ticker", "VARCHAR"),
    ("fold_year", "INTEGER"),
    ("strategy", "VARCHAR"),
    ("signal_date", "DATE"),
    ("execution_date", "DATE"),
    ("return_end_date", "DATE"),
    ("probability_positive", "DOUBLE"),
    ("position", "INTEGER"),
    ("opening_turnover", "DOUBLE"),
    ("liquidation_turnover", "DOUBLE"),
    ("price_return", "DOUBLE"),
    ("gross_return", "DOUBLE"),
    ("net_return", "DOUBLE"),
    ("gross_equity", "DOUBLE"),
    ("net_equity", "DOUBLE"),
]
FOLD_SCHEMA = [
    ("ticker", "VARCHAR"),
    ("fold_year", "INTEGER"),
    ("strategy", "VARCHAR"),
    ("first_execution", "DATE"),
    ("last_liquidation", "DATE"),
    ("intervals", "INTEGER"),
    ("gross_cumulative_return", "DOUBLE"),
    ("net_cumulative_return", "DOUBLE"),
    ("cost_drag", "DOUBLE"),
    ("excess_vs_buy_and_hold", "DOUBLE"),
    ("annualized_net_return", "DOUBLE"),
    ("annualized_volatility", "DOUBLE"),
    ("sharpe", "DOUBLE"),
    ("max_drawdown", "DOUBLE"),
    ("entry_count", "INTEGER"),
    ("exit_count", "INTEGER"),
    ("total_turnover", "DOUBLE"),
    ("exposure", "DOUBLE"),
]
SUMMARY_METRICS = (
    "gross_cumulative_return",
    "net_cumulative_return",
    "cost_drag",
    "excess_vs_buy_and_hold",
    "annualized_net_return",
    "annualized_volatility",
    "sharpe",
    "max_drawdown",
    "total_turnover",
    "exposure",
)
SUMMARY_SCHEMA = [
    ("ticker", "VARCHAR"),
    ("strategy", "VARCHAR"),
    ("fold_count", "INTEGER"),
    *[(f"{metric}_{suffix}", "DOUBLE") for metric in SUMMARY_METRICS for suffix in ("mean", "std")],
]


class BacktestError(ValueError):
    """The saved evaluation cannot form a faithful local backtest."""


@dataclass(frozen=True, slots=True)
class Signal:
    session_date: date
    probability: float


def _read_parquet(path: Path, schema: list[tuple[str, str]], columns: str) -> list[tuple]:
    try:
        with duckdb.connect() as connection:
            actual = connection.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]
            ).fetchall()
            if [(row[0], row[1]) for row in actual] != schema:
                raise BacktestError(f"Unexpected Parquet schema: {path}")
            return connection.execute(
                f"SELECT {columns} FROM read_parquet(?)", [str(path)]
            ).fetchall()
    except duckdb.Error as exc:
        raise BacktestError(f"Cannot read Parquet file: {path}") from exc


def _validated_evaluation(data_dir: Path, run_id: str) -> tuple[dict, dict]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id):
        raise BacktestError("Provide an evaluation run ID, not a path")
    run = data_dir / "evaluations" / run_id
    try:
        manifest_hash = _sha256(run / "manifest.json")
        manifest = json.loads((run / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BacktestError(f"Missing or invalid evaluation manifest: {run}") from exc
    if not isinstance(manifest, dict) or manifest.get("run_id") != run_id:
        raise BacktestError("Evaluation run ID does not match its manifest")
    if manifest.get("feature_version") != FEATURE_VERSION or manifest.get("target") != {
        "column": "target_positive_5",
        "horizon_sessions": HORIZON,
        "positive_if": "future_return_5 > 0",
    }:
        raise BacktestError("Unsupported evaluation feature or target version")
    try:
        configuration = manifest["configuration"]
        tickers = configuration["tickers"]
        first_year = configuration["first_validation_year"]
        folds = configuration["folds"]
        if (
            not isinstance(tickers, list)
            or not all(isinstance(ticker, str) for ticker in tickers)
            or normalize_tickers(tickers) != tickers
        ):
            raise ValueError("invalid tickers")
        if not isinstance(first_year, int) or not isinstance(folds, int) or folds < 1:
            raise ValueError("invalid folds")
        expected_folds = {
            (ticker, year) for ticker in tickers for year in range(first_year, first_year + folds)
        }
        fold_details = manifest["folds"]
        if {(item["ticker"], item["year"]) for item in fold_details} != expected_folds or len(
            fold_details
        ) != len(expected_folds):
            raise ValueError("fold details do not match configuration")
        for name in ("predictions.parquet", "fold_metrics.parquet", "leaderboard.parquet"):
            entry = manifest["outputs"][name]
            path = run / name
            if entry["path"] != name or not path.is_file() or _sha256(path) != entry["sha256"]:
                raise ValueError(f"{name} differs from evaluation manifest")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise BacktestError(f"Invalid or altered evaluation manifest: {exc}") from exc
    if _sha256(run / "manifest.json") != manifest_hash:
        raise BacktestError("Evaluation manifest changed while reading")
    return manifest, {
        "run": run,
        "tickers": tickers,
        "folds": expected_folds,
        "details": fold_details,
        "manifest_hash": manifest_hash,
    }


def _calendar_name(manifest: dict, ticker: str) -> str:
    calendars = manifest.get("calendars")
    if isinstance(calendars, dict) and isinstance(calendars.get(ticker), str):
        return calendars[ticker]
    if ticker in {"SPY", "QQQ"} and calendars is None:
        return "XNYS"
    raise BacktestError(
        f"{ticker}: evaluation does not record its exchange calendar; rerun evaluation"
    )


def _year_sessions(calendar_name: str, year: int, ticker: str) -> list[date]:
    try:
        calendar = xcals.get_calendar(calendar_name, start=f"{year}-01-01", end=f"{year}-12-31")
        sessions = [session.date() for session in calendar.sessions]
        if len(sessions) < HORIZON + 3:
            raise ValueError("calendar year has too few sessions")
        return sessions
    except (CalendarError, KeyError, ValueError) as exc:
        raise BacktestError(
            f"{ticker} {year}: cannot load exchange calendar {calendar_name}"
        ) from exc


def _source_prices(
    data_dir: Path, manifest: dict, ticker: str, calendar_name: str
) -> tuple[dict[date, float], dict]:
    relative = f"processed/prices/{ticker}.parquet"
    try:
        source = next(item for item in manifest["inputs"][ticker] if item["path"] == relative)
        path = data_dir / relative
        if not path.is_file() or _sha256(path) != source["sha256"]:
            raise BacktestError(
                f"{ticker}: source prices changed; rebuild features and rerun evaluation"
            )
    except (KeyError, TypeError, StopIteration) as exc:
        raise BacktestError(f"{ticker}: evaluation has no valid source price hash") from exc
    rows = _read_parquet(path, PRICE_SCHEMA, "ticker, session_date, adjusted_close")
    prices: dict[date, float] = {}
    dates: list[date] = []
    for symbol, session, adjusted_close in rows:
        if symbol != ticker or not isinstance(session, date) or session in prices:
            raise BacktestError(
                f"{ticker}: invalid ticker, date, or duplicate source price session"
            )
        if adjusted_close is None or not math.isfinite(adjusted_close) or adjusted_close <= 0:
            raise BacktestError(f"{ticker} {session}: invalid adjusted close")
        dates.append(session)
        prices[session] = adjusted_close
    if not dates or dates != sorted(dates):
        raise BacktestError(f"{ticker}: source price sessions must be nonempty and ascending")
    try:
        calendar = xcals.get_calendar(
            calendar_name,
            start=(dates[0] - timedelta(days=1)).isoformat(),
            end=(dates[-1] + timedelta(days=1)).isoformat(),
        )
        expected = {session.date() for session in calendar.sessions}
    except (CalendarError, KeyError, ValueError) as exc:
        raise BacktestError(f"{ticker}: cannot load exchange calendar {calendar_name}") from exc
    non_sessions = [session for session in dates if session not in expected]
    if non_sessions:
        raise BacktestError(f"{ticker}: price on a non-exchange session: {non_sessions[0]}")
    return prices, source


def _signals(
    run: Path, expected_folds: set[tuple[str, int]]
) -> dict[tuple[str, int, str], list[Signal]]:
    rows = _read_parquet(
        run / "predictions.parquet",
        PREDICTION_SCHEMA,
        "ticker, fold_year, model, session_date, probability_positive",
    )
    grouped: dict[tuple[str, int, str], list[Signal]] = {}
    seen: set[tuple[str, int, str, date]] = set()
    for ticker, year, model, session, probability in rows:
        if (
            (ticker, year) not in expected_folds
            or model not in MODELS
            or not isinstance(session, date)
        ):
            raise BacktestError("Prediction has an unexpected ticker, fold, model, or date")
        key = (ticker, year, model, session)
        if key in seen:
            raise BacktestError(f"Duplicate prediction key: {key}")
        seen.add(key)
        if probability is None or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise BacktestError(f"{ticker} {year} {model} {session}: invalid probability")
        grouped.setdefault((ticker, year, model), []).append(Signal(session, probability))
    if set(grouped) != {
        (ticker, year, model) for ticker, year in expected_folds for model in MODELS
    }:
        raise BacktestError("Evaluation predictions are missing a ticker, fold, or model")
    for signals in grouped.values():
        signals.sort(key=lambda item: item.session_date)
    return grouped


def _fold_window(
    ticker: str,
    year: int,
    calendar_name: str,
    prices: dict[date, float],
    grouped: dict[tuple[str, int, str], list[Signal]],
    detail: dict,
) -> tuple[list[date], dict]:
    sessions = _year_sessions(calendar_name, year, ticker)
    expected_signals = sessions[:-HORIZON]
    for model in MODELS:
        actual = [signal.session_date for signal in grouped[(ticker, year, model)]]
        if actual != expected_signals:
            raise BacktestError(
                f"{ticker} {year} {model}: prediction sessions must cover every exchange "
                "session through the five-session label boundary"
            )
    required_prices = sessions[1 : len(expected_signals) + 2]
    missing = [session for session in required_prices if session not in prices]
    if missing:
        raise BacktestError(f"{ticker} {year}: missing execution or return price: {missing[0]}")
    expected_set = set(sessions)
    non_sessions = [
        session for session in prices if session.year == year and session not in expected_set
    ]
    if non_sessions:
        raise BacktestError(f"{ticker} {year}: price on a non-exchange session: {non_sessions[0]}")
    try:
        if (
            detail["first_expected_session"] != sessions[0].isoformat()
            or detail["last_expected_session"] != sessions[-1].isoformat()
            or detail["validation"]["rows"] != len(expected_signals)
            or detail["validation"]["first_session"] != expected_signals[0].isoformat()
            or detail["validation"]["last_session"] != expected_signals[-1].isoformat()
        ):
            raise ValueError("fold dates or counts differ from evaluation manifest")
    except (KeyError, TypeError, ValueError) as exc:
        raise BacktestError(f"{ticker} {year}: invalid evaluation fold details: {exc}") from exc
    window = {
        "ticker": ticker,
        "year": year,
        "calendar": calendar_name,
        "first_execution": sessions[1].isoformat(),
        "last_liquidation": sessions[len(expected_signals) + 1].isoformat(),
        "intervals": len(expected_signals),
        "excluded_end_of_year_sessions": len(sessions) - (len(expected_signals) + 2),
    }
    return sessions, window


def _simulate(
    ticker: str,
    year: int,
    strategy: str,
    signals: list[Signal] | None,
    sessions: list[date],
    prices: dict[date, float],
    cost_rate: float,
) -> tuple[list[tuple], dict[str, object]]:
    interval_count = len(sessions) - HORIZON
    gross_equity = net_equity = 1.0
    previous = 0
    entry_count = exit_count = 0
    turnover_total = invested = 0
    maximum_equity = 1.0
    max_drawdown = 0.0
    net_returns: list[float] = []
    rows: list[tuple] = []
    for index in range(interval_count):
        signal = signals[index] if signals is not None else None
        position = (
            int(signal.probability >= 0.5)
            if signal is not None
            else int(strategy == "buy_and_hold")
        )
        execution = sessions[index + 1]
        end = sessions[index + 2]
        price_return = prices[end] / prices[execution] - 1
        opening_turnover = abs(position - previous)
        final_turnover = position if index == interval_count - 1 else 0
        entry_count += int(previous == 0 and position == 1)
        exit_count += int(previous == 1 and position == 0) + final_turnover
        turnover_total += opening_turnover + final_turnover
        invested += position
        gross_return = position * price_return
        net_multiplier = (
            (1 - cost_rate * opening_turnover)
            * (1 + gross_return)
            * (1 - cost_rate * final_turnover)
        )
        net_return = net_multiplier - 1
        gross_equity *= 1 + gross_return
        net_equity *= net_multiplier
        net_returns.append(net_return)
        maximum_equity = max(maximum_equity, net_equity)
        max_drawdown = max(max_drawdown, 1 - net_equity / maximum_equity)
        rows.append(
            (
                ticker,
                year,
                strategy,
                signal.session_date if signal else None,
                execution,
                end,
                signal.probability if signal else None,
                position,
                float(opening_turnover),
                float(final_turnover),
                price_return,
                gross_return,
                net_return,
                gross_equity,
                net_equity,
            )
        )
        previous = position
    volatility = statistics.stdev(net_returns) * math.sqrt(252) if len(net_returns) >= 2 else None
    sharpe = (
        statistics.fmean(net_returns) / statistics.stdev(net_returns) * math.sqrt(252)
        if volatility is not None and volatility > 0
        else None
    )
    metrics = {
        "ticker": ticker,
        "fold_year": year,
        "strategy": strategy,
        "first_execution": rows[0][4],
        "last_liquidation": rows[-1][5],
        "intervals": interval_count,
        "gross_cumulative_return": gross_equity - 1,
        "net_cumulative_return": net_equity - 1,
        "cost_drag": gross_equity - net_equity,
        "excess_vs_buy_and_hold": None,
        "annualized_net_return": net_equity ** (252 / interval_count) - 1,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "entry_count": entry_count,
        "exit_count": exit_count,
        "total_turnover": float(turnover_total),
        "exposure": invested / interval_count,
    }
    return rows, metrics


def _summary(metrics: list[dict[str, object]], tickers: list[str]) -> list[tuple]:
    result = []
    for ticker in tickers:
        for strategy in STRATEGIES:
            folds = [
                row for row in metrics if row["ticker"] == ticker and row["strategy"] == strategy
            ]
            values: list[float | None] = []
            for metric in SUMMARY_METRICS:
                series = [row[metric] for row in folds if row[metric] is not None]
                values.extend(
                    (statistics.fmean(series), statistics.pstdev(series))
                    if series
                    else (None, None)
                )
            result.append((ticker, strategy, len(folds), *values))
    return result


def backtest(
    data_dir: Path, evaluation_run_id: str, fee_bps: float = 1.0, slippage_bps: float = 5.0
) -> Path:
    """Apply one delayed long-or-cash rule and publish an auditable local run."""
    if any(not math.isfinite(value) or value < 0 for value in (fee_bps, slippage_bps)):
        raise BacktestError("Fee and slippage must be finite, nonnegative basis points")
    cost_rate = (fee_bps + slippage_bps) / 10_000
    if cost_rate >= 1:
        raise BacktestError("Combined cost must be below 10,000 basis points per trade")
    manifest, source = _validated_evaluation(data_dir, evaluation_run_id)
    grouped = _signals(source["run"], source["folds"])
    detail_by_fold = {(item["ticker"], item["year"]): item for item in source["details"]}
    intervals: list[tuple] = []
    metrics: list[dict[str, object]] = []
    windows: list[dict] = []
    price_sources: dict[str, dict] = {}
    for ticker in source["tickers"]:
        calendar_name = _calendar_name(manifest, ticker)
        prices, price_source = _source_prices(data_dir, manifest, ticker, calendar_name)
        price_sources[ticker] = price_source
        years = sorted(year for name, year in source["folds"] if name == ticker)
        for year in years:
            sessions, window = _fold_window(
                ticker, year, calendar_name, prices, grouped, detail_by_fold[(ticker, year)]
            )
            windows.append(window)
            fold_metrics = []
            for strategy in STRATEGIES:
                signals = grouped[(ticker, year, strategy)] if strategy in MODELS else None
                rows, result = _simulate(
                    ticker, year, strategy, signals, sessions, prices, cost_rate
                )
                intervals.extend(rows)
                fold_metrics.append(result)
            buy_and_hold = next(row for row in fold_metrics if row["strategy"] == "buy_and_hold")
            always_positive = next(
                row for row in fold_metrics if row["strategy"] == "always_positive"
            )
            if any(
                not math.isclose(always_positive[key], buy_and_hold[key], rel_tol=0, abs_tol=1e-12)
                for key in ("gross_cumulative_return", "net_cumulative_return", "total_turnover")
            ):
                raise BacktestError(f"{ticker} {year}: always-positive differs from buy and hold")
            for result in fold_metrics:
                result["excess_vs_buy_and_hold"] = (
                    result["net_cumulative_return"] - buy_and_hold["net_cumulative_return"]
                )
            metrics.extend(fold_metrics)
    summary = _summary(metrics, source["tickers"])
    root = data_dir / "backtests"
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC)
    run_id = timestamp.strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]
    output = root / run_id
    temporary = root / f".{run_id}.tmp"
    temporary.mkdir()
    try:
        _write_parquet(temporary / "intervals.parquet", INTERVAL_SCHEMA, intervals)
        fold_names = [name for name, _ in FOLD_SCHEMA]
        _write_parquet(
            temporary / "fold_metrics.parquet",
            FOLD_SCHEMA,
            [tuple(row[name] for name in fold_names) for row in metrics],
        )
        _write_parquet(temporary / "summary.parquet", SUMMARY_SCHEMA, summary)
        source_hashes = {
            "evaluation_manifest": source["manifest_hash"],
            "predictions": _sha256(source["run"] / "predictions.parquet"),
        }
        if _sha256(source["run"] / "manifest.json") != source["manifest_hash"]:
            raise BacktestError("Evaluation manifest changed during backtest")
        if source_hashes["predictions"] != manifest["outputs"]["predictions.parquet"]["sha256"]:
            raise BacktestError("Evaluation predictions changed during backtest")
        for ticker, item in price_sources.items():
            if _sha256(data_dir / item["path"]) != item["sha256"]:
                raise BacktestError(f"{ticker}: source prices changed during backtest")
        result_manifest = {
            "run_id": run_id,
            "generated_at_utc": timestamp.isoformat(),
            "evaluation_run_id": evaluation_run_id,
            "evaluation_inputs": source_hashes,
            "price_sources": price_sources,
            "strategies": list(STRATEGIES),
            "configuration": {
                "tickers": source["tickers"],
                "fee_bps": fee_bps,
                "slippage_bps": slippage_bps,
                "probability_threshold": 0.5,
            },
            "strategy": {
                "positions": "long_or_cash",
                "rebalances": "daily",
                "signal_available": "after signal session close",
                "execution": "next exchange session close",
                "return_window": "execution close to following exchange session close",
                "folds_start_in_cash": True,
                "liquidate_at_final_return_close": True,
                "adjusted_close_is_return_proxy": True,
            },
            "windows": windows,
            "versions": {
                "python": platform.python_version(),
                "duckdb": duckdb.__version__,
                "exchange_calendars": xcals.__version__,
            },
            "outputs": {
                name: {"path": name, "sha256": _sha256(temporary / name)} for name in OUTPUT_NAMES
            },
        }
        (temporary / "manifest.json").write_text(
            json.dumps(result_manifest, indent=2, sort_keys=True) + "\n"
        )
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output
