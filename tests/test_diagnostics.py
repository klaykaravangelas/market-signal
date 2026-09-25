"""Offline Spec 006 checks using small saved-result snapshots."""

import json
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pytest

from marketsignal.backtesting import (
    FOLD_SCHEMA as BACKTEST_FOLD_SCHEMA,
)
from marketsignal.backtesting import INTERVAL_SCHEMA, STRATEGIES, SUMMARY_SCHEMA
from marketsignal.diagnostics import DiagnosticsError, diagnose, track_diagnostics
from marketsignal.evaluation import (
    FEATURES,
    LEADERBOARD_SCHEMA,
    METRICS,
    MODELS,
    PREDICTION_SCHEMA,
    _scores,
)
from marketsignal.evaluation import FOLD_SCHEMA as EVALUATION_FOLD_SCHEMA
from marketsignal.features import _sha256, _write_parquet


def rows(path: Path) -> list[tuple]:
    with duckdb.connect() as connection:
        return connection.execute("SELECT * FROM read_parquet(?)", [str(path)]).fetchall()


def saved_evaluation(root: Path, tickers: tuple[str, ...] = ("SPY",), years: int = 1) -> Path:
    run = root / "evaluations" / "fixture"
    run.mkdir(parents=True)
    predictions = []
    folds = []
    details = []
    leaderboard = []
    actual = [0, 0, 1, 1]
    probabilities = {
        "always_positive": [1.0] * 4,
        "training_prevalence": [0.5] * 4,
        "logistic_regression": [0.0, 0.1, 0.5, 1.0],
        "random_forest": [0.9, 0.8, 0.2, 0.1],
    }
    for ticker in tickers:
        scores_by_model = {model: [] for model in MODELS}
        for year in range(2022, 2022 + years):
            dates = [date(year, 1, 3) + timedelta(days=i) for i in range(4)]
            details.append(
                {
                    "ticker": ticker,
                    "year": year,
                    "training": {
                        "first_session": "2020-01-02",
                        "last_session": "2021-12-30",
                        "rows": 10,
                        "positive": 5,
                        "negative": 5,
                    },
                    "validation": {
                        "first_session": dates[0].isoformat(),
                        "last_session": dates[-1].isoformat(),
                        "rows": 4,
                        "positive": 2,
                        "negative": 2,
                    },
                }
            )
            for model in MODELS:
                probs = probabilities[model]
                score = _scores(np.array(actual), np.array(probs))
                scores_by_model[model].append(score)
                for session, label, probability in zip(dates, actual, probs, strict=True):
                    predictions.append(
                        (
                            ticker,
                            year,
                            model,
                            session,
                            session + timedelta(days=7),
                            label,
                            int(probability >= 0.5),
                            probability,
                        )
                    )
                folds.append(
                    (ticker, year, model, 10, 4, 5, 2, 0.5, *(score[name] for name in METRICS))
                )
        for model in MODELS:
            values = []
            for metric in METRICS:
                series = [score[metric] for score in scores_by_model[model]]
                values.extend((float(np.mean(series)), float(np.std(series))))
            leaderboard.append((ticker, model, years, *values))
    for name, schema, content in (
        ("predictions.parquet", PREDICTION_SCHEMA, predictions),
        ("fold_metrics.parquet", EVALUATION_FOLD_SCHEMA, folds),
        ("leaderboard.parquet", LEADERBOARD_SCHEMA, leaderboard),
    ):
        _write_parquet(run / name, schema, content)
    manifest = {
        "run_id": "fixture",
        "configuration": {"tickers": list(tickers), "first_validation_year": 2022, "folds": years},
        "feature_version": "v1",
        "features": list(FEATURES),
        "target": {
            "column": "target_positive_5",
            "horizon_sessions": 5,
            "positive_if": "future_return_5 > 0",
        },
        "folds": details,
        "outputs": {
            name: {"path": name, "sha256": _sha256(run / name)}
            for name in ("predictions.parquet", "fold_metrics.parquet", "leaderboard.parquet")
        },
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    return run


def rewrite(run: Path, name: str, schema: list[tuple[str, str]], content: list[tuple]) -> None:
    (run / name).unlink()
    _write_parquet(run / name, schema, content)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"][name]["sha256"] = _sha256(run / name)
    manifest_path.write_text(json.dumps(manifest))


def saved_backtest(root: Path, evaluation: Path) -> Path:
    run = root / "backtests" / "simulation"
    run.mkdir(parents=True)
    fold_rows = []
    for strategy in STRATEGIES:
        values = {
            "ticker": "SPY",
            "fold_year": 2022,
            "strategy": strategy,
            "first_execution": date(2022, 1, 4),
            "last_liquidation": date(2022, 1, 7),
            "intervals": 4,
            "gross_cumulative_return": 0.02,
            "net_cumulative_return": 0.01 if strategy != "all_cash" else 0.0,
            "cost_drag": 0.01,
            "excess_vs_buy_and_hold": 0.0 if strategy != "all_cash" else -0.01,
            "annualized_net_return": 0.1,
            "annualized_volatility": 0.2,
            "sharpe": 0.5,
            "max_drawdown": 0.1,
            "entry_count": 1,
            "exit_count": 1,
            "total_turnover": 2.0 if strategy != "all_cash" else 0.0,
            "exposure": 1.0 if strategy != "all_cash" else 0.0,
        }
        fold_rows.append(tuple(values[name] for name, _ in BACKTEST_FOLD_SCHEMA))
    _write_parquet(run / "fold_metrics.parquet", BACKTEST_FOLD_SCHEMA, fold_rows)
    _write_parquet(run / "intervals.parquet", INTERVAL_SCHEMA, [])
    _write_parquet(run / "summary.parquet", SUMMARY_SCHEMA, [])
    evaluation_manifest = json.loads((evaluation / "manifest.json").read_text())
    manifest = {
        "run_id": "simulation",
        "evaluation_run_id": evaluation.name,
        "evaluation_inputs": {
            "evaluation_manifest": _sha256(evaluation / "manifest.json"),
            "predictions": evaluation_manifest["outputs"]["predictions.parquet"]["sha256"],
        },
        "configuration": {
            "tickers": ["SPY"],
            "fee_bps": 1.0,
            "slippage_bps": 5.0,
            "probability_threshold": 0.5,
        },
        "strategies": list(STRATEGIES),
        "windows": [{"ticker": "SPY", "year": 2022}],
        "outputs": {
            name: {"path": name, "sha256": _sha256(run / name)}
            for name in ("intervals.parquet", "fold_metrics.parquet", "summary.parquet")
        },
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    return run


def test_report_values_bins_and_reproducibility(tmp_path: Path) -> None:
    saved_evaluation(tmp_path, ("SPY", "QQQ"), years=2)
    first = diagnose(tmp_path, "fixture")
    second = diagnose(tmp_path, "fixture")
    for name in ("fold_diagnostics.parquet", "calibration_bins.parquet"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    manifest = json.loads((first / "manifest.json").read_text())
    for name, entry in manifest["outputs"].items():
        assert _sha256(first / name) == entry["sha256"]
    assert len(manifest["coverage"]) == 4
    diagnostics = rows(first / "fold_diagnostics.parquet")
    with duckdb.connect() as connection:
        cursor = connection.execute(
            "SELECT * FROM read_parquet(?) WHERE ticker='SPY' AND fold_year=2022 "
            "AND model='logistic_regression'",
            [str(first / "fold_diagnostics.parquet")],
        )
        output = dict(zip((item[0] for item in cursor.description), cursor.fetchone(), strict=True))
    assert len(diagnostics) == 16
    assert output["validation_rows"] == 4
    assert output["actual_positive_count"] == 2
    assert output["predicted_positive_count"] == 2
    assert (
        output["true_positive"],
        output["false_positive"],
        output["true_negative"],
        output["false_negative"],
    ) == (2, 0, 2, 0)
    assert output["probability_median"] == pytest.approx(0.3)
    assert output["probability_below_0_5_fraction"] == 0.5
    assert output["probability_equal_0_5_fraction"] == 0.25
    assert output["probability_above_0_5_fraction"] == 0.25
    assert output["accuracy_delta_vs_always_positive"] == 0.5
    assert output["auc_delta_vs_always_positive"] == 0.5
    assert output["brier_improvement_vs_always_positive"] > 0
    with duckdb.connect() as connection:
        bins = connection.execute(
            "SELECT bin_index,count,mean_predicted_probability,observed_positive_fraction,sparse "
            "FROM read_parquet(?) WHERE ticker='SPY' AND fold_year=2022 "
            "AND model='logistic_regression' ORDER BY bin_index",
            [str(first / "calibration_bins.parquet")],
        ).fetchall()
    assert [b[1] for b in bins] == [1, 1, 0, 0, 0, 1, 0, 0, 0, 1]
    assert bins[0][2:] == (0.0, 0.0, True)
    assert bins[2][2:] == (None, None, False)
    assert bins[5][2:] == (0.5, 1.0, True)
    assert bins[9][2:] == (1.0, 1.0, True)
    report = (first / "report.html").read_text()
    assert "SPY · 2022" in report and "QQQ · 2023" in report
    assert "reliability diagram" in report and "probability histogram" in report
    assert "Brier score measures" in report and "Sparse (&lt;20)" in report
    assert "<script" not in report and "http" not in report


def test_all_negative_and_worse_than_baseline(tmp_path: Path) -> None:
    run = saved_evaluation(tmp_path)
    original = rows(run / "predictions.parquet")
    changed = []
    for row in original:
        if row[2] == "random_forest":
            probability = [0.3, 0.2, 0.1, 0.0][len(changed) - 12]
            row = (*row[:6], 0, probability)
        changed.append(row)
    rewrite(run, "predictions.parquet", PREDICTION_SCHEMA, changed)
    metrics = rows(run / "fold_metrics.parquet")
    score = _scores(np.array([0, 0, 1, 1]), np.array([0.3, 0.2, 0.1, 0.0]))
    metrics[-1] = (*metrics[-1][:8], *(score[name] for name in METRICS))
    rewrite(run, "fold_metrics.parquet", EVALUATION_FOLD_SCHEMA, metrics)
    leaderboard = rows(run / "leaderboard.parquet")
    leaderboard[-1] = (
        *leaderboard[-1][:3],
        *(value for name in METRICS for value in (score[name], 0.0)),
    )
    rewrite(run, "leaderboard.parquet", LEADERBOARD_SCHEMA, leaderboard)
    output = diagnose(tmp_path, "fixture")
    with duckdb.connect() as connection:
        result = connection.execute(
            "SELECT predicted_positive_count, false_negative, accuracy_delta_vs_always_positive, "
            "auc_delta_vs_always_positive FROM read_parquet(?) WHERE model='random_forest'",
            [str(output / "fold_diagnostics.parquet")],
        ).fetchone()
    assert result == (0, 2, 0.0, -0.5)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("bytes", "altered or missing output"),
        ("duplicate", "Duplicate prediction key"),
        ("missing", "prediction counts"),
        ("unequal_label", "candidate dates or labels differ"),
        ("unequal_date", "candidate dates or labels differ"),
        ("probability", "invalid probability"),
        ("out_of_range", "invalid probability"),
        ("threshold", "invalid actual or threshold prediction"),
        ("metric", "saved accuracy differs"),
        ("unsafe_path", "unsafe output path"),
    ],
)
def test_invalid_sources_publish_nothing(tmp_path: Path, mutation: str, match: str) -> None:
    run = saved_evaluation(tmp_path)
    if mutation == "bytes":
        with (run / "predictions.parquet").open("ab") as file:
            file.write(b"altered")
    elif mutation == "unsafe_path":
        manifest_path = run / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["outputs"]["predictions.parquet"]["path"] = "../predictions.parquet"
        manifest_path.write_text(json.dumps(manifest))
    elif mutation == "metric":
        metrics = rows(run / "fold_metrics.parquet")
        metrics[0] = (*metrics[0][:8], 0.1, *metrics[0][9:])
        rewrite(run, "fold_metrics.parquet", EVALUATION_FOLD_SCHEMA, metrics)
    else:
        predictions = rows(run / "predictions.parquet")
        if mutation == "duplicate":
            predictions.append(predictions[0])
        elif mutation == "missing":
            predictions.pop(0)
        elif mutation == "unequal_label":
            predictions[-1] = (*predictions[-1][:5], 0, *predictions[-1][6:])
        elif mutation == "unequal_date":
            predictions[-1] = (
                *predictions[-1][:3],
                predictions[-1][3] + timedelta(days=1),
                *predictions[-1][4:],
            )
        elif mutation == "probability":
            predictions[-1] = (*predictions[-1][:7], float("nan"))
        elif mutation == "out_of_range":
            predictions[-1] = (*predictions[-1][:7], 1.1)
        elif mutation == "threshold":
            predictions[-1] = (*predictions[-1][:6], 1 - predictions[-1][6], predictions[-1][7])
        rewrite(run, "predictions.parquet", PREDICTION_SCHEMA, predictions)
    with pytest.raises(DiagnosticsError, match=match):
        diagnose(tmp_path, "fixture")
    assert not (tmp_path / "diagnostics").exists()


def test_write_failure_leaves_no_published_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marketsignal import diagnostics

    saved_evaluation(tmp_path)
    original = diagnostics._write_parquet

    def fail_on_bins(path, schema, content):
        if path.name == "calibration_bins.parquet":
            raise OSError("synthetic output failure")
        return original(path, schema, content)

    monkeypatch.setattr(diagnostics, "_write_parquet", fail_on_bins)
    with pytest.raises(OSError, match="synthetic output failure"):
        diagnose(tmp_path, "fixture")
    assert list((tmp_path / "diagnostics").iterdir()) == []


def test_linked_backtest_and_mismatch(tmp_path: Path) -> None:
    evaluation = saved_evaluation(tmp_path)
    simulation = saved_backtest(tmp_path, evaluation)
    output = diagnose(tmp_path, "fixture", "simulation")
    assert (output / "trading_context.parquet").exists()
    assert "Trading context (separate backtest)" in (output / "report.html").read_text()
    assert len(rows(output / "trading_context.parquet")) == 6
    manifest_path = simulation / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["evaluation_inputs"]["evaluation_manifest"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(DiagnosticsError, match="does not match"):
        diagnose(tmp_path, "fixture", "simulation")
    assert len(list((tmp_path / "diagnostics").iterdir())) == 1


def test_mlflow_import_and_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mlflow = pytest.importorskip("mlflow")
    from marketsignal.tracking import _store

    evaluation = saved_evaluation(tmp_path)
    output = diagnose(tmp_path, "fixture")
    uri, artifacts = _store(tmp_path, None, None)
    mlflow.set_tracking_uri(uri)
    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    evaluation_experiment = mlflow.create_experiment(
        "marketsignal-evaluation", artifact_location=f"{artifacts}/marketsignal-evaluation"
    )
    evaluation_mlflow_run = client.create_run(
        evaluation_experiment,
        tags={
            "source_run_id": evaluation.name,
            "source_manifest_sha256": _sha256(evaluation / "manifest.json"),
        },
    )
    client.set_terminated(evaluation_mlflow_run.info.run_id, status="FINISHED")
    original = mlflow.log_artifact

    def fail_once(*args, **kwargs):
        monkeypatch.setattr(mlflow, "log_artifact", original)
        raise RuntimeError("synthetic artifact failure")

    monkeypatch.setattr(mlflow, "log_artifact", fail_once)
    with pytest.raises(DiagnosticsError, match="synthetic artifact failure"):
        track_diagnostics(tmp_path, output.name)
    run_id, _, _ = track_diagnostics(tmp_path, output.name)
    assert track_diagnostics(tmp_path, output.name)[0] == run_id
    run = client.get_run(run_id)
    assert run.data.tags["evaluation_run_id"] == evaluation.name
    assert run.data.tags["evaluation_manifest_sha256"] == _sha256(evaluation / "manifest.json")
    assert run.data.tags["evaluation_mlflow_run_id"] == evaluation_mlflow_run.info.run_id
    names = {item.path for item in client.list_artifacts(run_id)}
    assert {"report.html", "manifest.json", "fold_diagnostics.parquet"} <= names
    attempts = [
        item
        for item in client.search_runs([run.info.experiment_id])
        if item.data.tags.get("source_run_id") == output.name
    ]
    assert sorted(item.info.status for item in attempts) == ["FAILED", "FINISHED"]


def test_missing_mlflow_extra(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from marketsignal import diagnostics

    saved_evaluation(tmp_path)
    output = diagnose(tmp_path, "fixture")

    def missing():
        raise TrackingError("Install the tracking extra: uv sync --extra tracking")

    from marketsignal.tracking import TrackingError

    monkeypatch.setattr(diagnostics, "_mlflow", missing)
    with pytest.raises(DiagnosticsError, match="Install the tracking extra"):
        track_diagnostics(tmp_path, output.name)
