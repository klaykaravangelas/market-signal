import json
import math
import statistics
from datetime import date, timedelta
from pathlib import Path

import duckdb
import exchange_calendars as xcals
import pytest

from marketsignal.backtesting import BacktestError, Signal, _simulate, backtest
from marketsignal.cli import main
from marketsignal.evaluation import FOLD_SCHEMA as EVALUATION_FOLD_SCHEMA
from marketsignal.evaluation import LEADERBOARD_SCHEMA, MODELS, PREDICTION_SCHEMA
from marketsignal.features import PRICE_SCHEMA, _sha256, _write_parquet


def read_rows(path: Path) -> list[tuple]:
    with duckdb.connect() as connection:
        return connection.execute("SELECT * FROM read_parquet(?)", [str(path)]).fetchall()


def make_evaluation(data_dir: Path, tickers: tuple[str, ...] = ("SPY",)) -> tuple[Path, list[date]]:
    calendar = xcals.get_calendar("XNYS", start="2022-01-01", end="2022-12-31")
    sessions = [session.date() for session in calendar.sessions]
    run = data_dir / "evaluations" / "fixture"
    run.mkdir(parents=True)
    inputs = {}
    details = []
    predictions = []
    for ticker in tickers:
        source = data_dir / "processed" / "prices" / f"{ticker}.parquet"
        source.parent.mkdir(parents=True, exist_ok=True)
        price_rows = [
            (ticker, day, 100.0 + i, 101.0 + i, 99.0 + i, 100.0 + i, 100.0 + i, 1000)
            for i, day in enumerate(sessions)
        ]
        _write_parquet(source, PRICE_SCHEMA, price_rows)
        inputs[ticker] = [{"path": str(source.relative_to(data_dir)), "sha256": _sha256(source)}]
        details.append(
            {
                "ticker": ticker,
                "year": 2022,
                "first_expected_session": sessions[0].isoformat(),
                "last_expected_session": sessions[-1].isoformat(),
                "validation": {
                    "first_session": sessions[0].isoformat(),
                    "last_session": sessions[-6].isoformat(),
                    "rows": len(sessions) - 5,
                },
            }
        )
        for model in MODELS:
            for index, day in enumerate(sessions[:-5]):
                probability = {
                    "always_positive": 1.0,
                    "training_prevalence": 0.6,
                    "logistic_regression": 0.6 if index % 2 == 0 else 0.4,
                    "random_forest": 0.7 if index % 3 else 0.3,
                }[model]
                predictions.append(
                    (
                        ticker,
                        2022,
                        model,
                        day,
                        sessions[index + 5],
                        index % 2,
                        int(probability >= 0.5),
                        probability,
                    )
                )
    _write_parquet(run / "predictions.parquet", PREDICTION_SCHEMA, predictions)
    _write_parquet(run / "fold_metrics.parquet", EVALUATION_FOLD_SCHEMA, [])
    _write_parquet(run / "leaderboard.parquet", LEADERBOARD_SCHEMA, [])
    manifest = {
        "run_id": "fixture",
        "feature_version": "v1",
        "target": {
            "column": "target_positive_5",
            "horizon_sessions": 5,
            "positive_if": "future_return_5 > 0",
        },
        "configuration": {"tickers": list(tickers), "first_validation_year": 2022, "folds": 1},
        "calendars": {ticker: "XNYS" for ticker in tickers},
        "inputs": inputs,
        "folds": details,
        "outputs": {
            name: {"path": name, "sha256": _sha256(run / name)}
            for name in ("predictions.parquet", "fold_metrics.parquet", "leaderboard.parquet")
        },
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    return run, sessions


def rewrite_predictions(run: Path, rows: list[tuple]) -> None:
    path = run / "predictions.parquet"
    path.unlink()
    _write_parquet(path, PREDICTION_SCHEMA, rows)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"]["predictions.parquet"]["sha256"] = _sha256(path)
    manifest_path.write_text(json.dumps(manifest))


def test_signal_lag_costs_and_metrics() -> None:
    sessions = [date(2022, 1, 3) + timedelta(days=i) for i in range(8)]
    signals = [Signal(sessions[i], probability) for i, probability in enumerate((0.8, 0.2, 0.8))]
    prices = {sessions[1]: 100.0, sessions[2]: 110.0, sessions[3]: 99.0, sessions[4]: 108.9}
    rows, metrics = _simulate("SPY", 2022, "logistic_regression", signals, sessions, prices, 0.01)
    assert [row[7] for row in rows] == [1, 0, 1]
    assert [(row[3], row[4], row[5]) for row in rows] == [
        (sessions[0], sessions[1], sessions[2]),
        (sessions[1], sessions[2], sessions[3]),
        (sessions[2], sessions[3], sessions[4]),
    ]
    assert [row[8] for row in rows] == [1.0, 1.0, 1.0]
    assert [row[9] for row in rows] == [0.0, 0.0, 1.0]
    assert rows[0][12] == pytest.approx(0.99 * 1.1 - 1)
    assert rows[1][12] == pytest.approx(-0.01)
    assert rows[2][12] == pytest.approx(0.99 * 1.1 * 0.99 - 1)
    assert metrics["entry_count"] == 2
    assert metrics["exit_count"] == 2
    assert metrics["total_turnover"] == 4.0
    assert metrics["exposure"] == pytest.approx(2 / 3)
    assert metrics["net_cumulative_return"] == pytest.approx(
        (0.99 * 1.1) * 0.99 * (0.99 * 1.1 * 0.99) - 1
    )
    assert metrics["gross_cumulative_return"] == pytest.approx(1.1 * 1.1 - 1)
    assert metrics["max_drawdown"] == pytest.approx(0.01)
    net_returns = [row[12] for row in rows]
    expected_volatility = statistics.stdev(net_returns) * math.sqrt(252)
    assert metrics["annualized_net_return"] == pytest.approx(
        (1 + metrics["net_cumulative_return"]) ** (252 / 3) - 1
    )
    assert metrics["annualized_volatility"] == pytest.approx(expected_volatility)
    assert metrics["sharpe"] == pytest.approx(
        statistics.fmean(net_returns) / statistics.stdev(net_returns) * math.sqrt(252)
    )
    cash_rows, cash_metrics = _simulate("SPY", 2022, "all_cash", None, sessions, prices, 0.01)
    assert all(row[12] == 0 for row in cash_rows)
    assert cash_metrics["sharpe"] is None
    assert cash_metrics["net_cumulative_return"] == 0
    _, free_metrics = _simulate("SPY", 2022, "logistic_regression", signals, sessions, prices, 0)
    assert free_metrics["cost_drag"] == 0
    assert free_metrics["net_cumulative_return"] == free_metrics["gross_cumulative_return"]


def test_saved_run_is_queryable_and_reproducible(tmp_path: Path) -> None:
    _, sessions = make_evaluation(tmp_path, ("SPY", "QQQ"))
    first = backtest(tmp_path, "fixture")
    second = backtest(tmp_path, "fixture")
    for name in ("intervals.parquet", "fold_metrics.parquet", "summary.parquet"):
        assert read_rows(first / name) == read_rows(second / name)
    manifest = json.loads((first / "manifest.json").read_text())
    assert len(manifest["windows"]) == 2
    assert manifest["windows"][0]["first_execution"] == sessions[1].isoformat()
    assert manifest["windows"][0]["last_liquidation"] == sessions[-4].isoformat()
    assert manifest["windows"][0]["excluded_end_of_year_sessions"] == 3
    for name, entry in manifest["outputs"].items():
        assert _sha256(first / name) == entry["sha256"]
    intervals = read_rows(first / "intervals.parquet")
    metrics = read_rows(first / "fold_metrics.parquet")
    summary = read_rows(first / "summary.parquet")
    assert len(intervals) == 2 * 6 * (len(sessions) - 5)
    assert len(metrics) == len(summary) == 12
    by_key = {(row[0], row[2]): row for row in metrics}
    for ticker in ("SPY", "QQQ"):
        assert by_key[(ticker, "always_positive")][7] == by_key[(ticker, "buy_and_hold")][7]
        assert by_key[(ticker, "all_cash")][7] == 0
    assert all(row[2] == 1 for row in summary)


def test_stale_source_and_modified_evaluation_fail_without_output(tmp_path: Path) -> None:
    run, _ = make_evaluation(tmp_path)
    source = tmp_path / "processed" / "prices" / "SPY.parquet"
    source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(BacktestError, match="source prices changed"):
        backtest(tmp_path, "fixture")
    assert not (tmp_path / "backtests").exists()
    source.write_bytes(source.read_bytes()[:-7])
    prediction = run / "predictions.parquet"
    prediction.write_bytes(prediction.read_bytes() + b"changed")
    with pytest.raises(BacktestError, match="predictions.parquet differs"):
        backtest(tmp_path, "fixture")
    assert not (tmp_path / "backtests").exists()


def test_duplicate_missing_and_invalid_prediction_rejected(tmp_path: Path) -> None:
    run, _ = make_evaluation(tmp_path)
    original = read_rows(run / "predictions.parquet")
    for rows, message in (
        (original + [original[0]], "Duplicate prediction key"),
        (original[1:], "prediction sessions must cover"),
        ([(*original[0][:-1], float("nan")), *original[1:]], "invalid probability"),
    ):
        rewrite_predictions(run, rows)
        with pytest.raises(BacktestError, match=message):
            backtest(tmp_path, "fixture")
        assert not (tmp_path / "backtests").exists()


def test_missing_execution_price_rejected(tmp_path: Path) -> None:
    run, sessions = make_evaluation(tmp_path)
    source = tmp_path / "processed" / "prices" / "SPY.parquet"
    rows = [row for row in read_rows(source) if row[1] != sessions[2]]
    source.unlink()
    _write_parquet(source, PRICE_SCHEMA, rows)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["inputs"]["SPY"][0]["sha256"] = _sha256(source)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(BacktestError, match="missing execution or return price"):
        backtest(tmp_path, "fixture")
    assert not (tmp_path / "backtests").exists()


@pytest.mark.parametrize(
    "fee,slippage", [(-1, 5), (1, float("nan")), (float("inf"), 0), (10_000, 0)]
)
def test_invalid_costs_rejected(tmp_path: Path, fee: float, slippage: float) -> None:
    with pytest.raises(BacktestError, match="cost|Fee"):
        backtest(tmp_path, "fixture", fee, slippage)
    assert not (tmp_path / "backtests").exists()


def test_cli_backtest(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_evaluation(tmp_path)
    assert (
        main(
            [
                "backtest",
                "fixture",
                "--data-dir",
                str(tmp_path),
                "--fee-bps",
                "0",
                "--slippage-bps",
                "0",
            ]
        )
        == 0
    )
    assert "Backtest saved" in capsys.readouterr().out
