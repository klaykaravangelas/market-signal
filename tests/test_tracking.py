"""Spec 005 integration checks against a temporary local MLflow store."""

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pytest
from test_evaluation import fixture_data, records

from marketsignal.backtesting import backtest
from marketsignal.evaluation import FEATURES, evaluate
from marketsignal.features import _sha256
from marketsignal.tracking import TrackingError, track


def test_new_models_and_linked_tracking(tmp_path: Path) -> None:
    mlflow = pytest.importorskip("mlflow")
    fixture_data(tmp_path)
    evaluation = evaluate(tmp_path, ["SPY"], 2022, 1)
    manifest = json.loads((evaluation / "manifest.json").read_text())
    saved = json.loads((evaluation / "training_records.json").read_text())
    assert len(saved["records"]) == 4
    assert saved["schema_version"] == 1
    assert all(record["fit_seconds"] is None for record in saved["records"][:2])
    assert all(record["fit_seconds"] >= 0 for record in saved["records"][2:])
    assert saved["records"][2]["diagnostics"]["iterations"] > 0
    assert saved["records"][3]["diagnostics"]["tree_count"] == 200
    for model in ("logistic_regression", "random_forest"):
        name = f"models/SPY/2022/{model}.joblib"
        assert _sha256(evaluation / name) == manifest["outputs"][name]["sha256"]
        payload = joblib.load(evaluation / name)
        assert payload["features"] == list(FEATURES)
        features = {row[1]: row[2:] for row in records(tmp_path / "features" / "SPY.parquet")}
        prediction_rows = [
            row
            for row in records(evaluation / "predictions.parquet")
            if row[2] == model and row[1] == 2022
        ]
        x = np.array([features[row[3]] for row in prediction_rows])
        actual = payload["estimator"].predict_proba(x)[:, 1]
        assert np.allclose(actual, [row[7] for row in prediction_rows])

    simulation = backtest(tmp_path, evaluation.name)
    evaluation_id, backtest_id, _, _ = track(tmp_path, evaluation.name, simulation.name)
    assert backtest_id is not None
    again = track(tmp_path, evaluation.name, simulation.name)
    assert again[:2] == (evaluation_id, backtest_id)
    client = mlflow.tracking.MlflowClient()
    evaluation_parent = client.get_run(evaluation_id)
    assert evaluation_parent.data.tags["results_only"] == "false"
    experiment_id = evaluation_parent.info.experiment_id
    children = client.search_runs(
        [experiment_id],
        filter_string=f"tags.`mlflow.parentRunId` = '{evaluation_id}'",
    )
    assert len(children) == 4
    fitted = next(run for run in children if run.data.tags["candidate"] == "logistic_regression")
    assert fitted.data.tags["model_artifact_available"] == "true"
    assert "roc_auc" in fitted.data.metrics
    loaded = mlflow.sklearn.load_model(fitted.data.tags["model_uri"])
    assert hasattr(loaded, "predict_proba")
    backtest_parent = client.get_run(backtest_id)
    assert backtest_parent.data.tags["evaluation_run_id"] == evaluation.name
    assert backtest_parent.data.tags["evaluation_manifest_sha256"] == _sha256(
        evaluation / "manifest.json"
    )
    trading_children = client.search_runs(
        [backtest_parent.info.experiment_id],
        filter_string=f"tags.`mlflow.parentRunId` = '{backtest_id}'",
    )
    assert len(trading_children) == 6
    assert "net_cumulative_return" in trading_children[0].data.metrics
    cash = next(run for run in trading_children if run.data.tags["strategy"] == "all_cash")
    assert "sharpe" not in cash.data.metrics


def test_historical_import_and_altered_source(tmp_path: Path) -> None:
    mlflow = pytest.importorskip("mlflow")
    fixture_data(tmp_path)
    current = evaluate(tmp_path, ["SPY"], 2022, 1)
    legacy = current.parent / "legacy"
    shutil.copytree(current, legacy)
    manifest_path = legacy / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["run_id"] = "legacy"
    for name in list(manifest["outputs"]):
        if name == "training_records.json" or name.startswith("models/"):
            (legacy / name).unlink()
            del manifest["outputs"][name]
    manifest_path.write_text(json.dumps(manifest))
    mlflow_id, _, _, _ = track(tmp_path, "legacy")
    client = mlflow.tracking.MlflowClient()
    assert client.get_run(mlflow_id).data.tags["results_only"] == "true"
    children = client.search_runs(
        [client.get_run(mlflow_id).info.experiment_id],
        filter_string=f"tags.`mlflow.parentRunId` = '{mlflow_id}'",
    )
    assert all(child.data.tags["model_artifact_available"] == "false" for child in children)
    (legacy / "fold_metrics.parquet").write_bytes(b"altered")
    with pytest.raises(TrackingError, match="altered or missing output"):
        track(tmp_path, "legacy")


def test_failed_import_is_visible_and_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mlflow = pytest.importorskip("mlflow")
    from marketsignal import tracking

    fixture_data(tmp_path)
    evaluation = evaluate(tmp_path, ["SPY"], 2022, 1)
    original_hash = _sha256(evaluation / "manifest.json")
    original = tracking._log_scalars

    def fail_once(*args, **kwargs) -> None:
        monkeypatch.setattr(tracking, "_log_scalars", original)
        raise RuntimeError("synthetic MLflow failure")

    monkeypatch.setattr(tracking, "_log_scalars", fail_once)
    with pytest.raises(TrackingError, match="synthetic MLflow failure"):
        track(tmp_path, evaluation.name)
    assert _sha256(evaluation / "manifest.json") == original_hash
    mlflow_id, _, _, _ = track(tmp_path, evaluation.name)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_run(mlflow_id).info.experiment_id
    parents = [
        run
        for run in client.search_runs([experiment])
        if run.data.tags.get("source_run_id") == evaluation.name
    ]
    assert sorted(run.info.status for run in parents) == ["FAILED", "FINISHED"]


def test_mismatched_backtest_is_rejected(tmp_path: Path) -> None:
    fixture_data(tmp_path)
    evaluation = evaluate(tmp_path, ["SPY"], 2022, 1)
    simulation = backtest(tmp_path, evaluation.name)
    manifest_path = simulation / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["evaluation_run_id"] = "other-run"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(TrackingError, match="does not match"):
        track(tmp_path, evaluation.name, simulation.name)
    assert not (tmp_path / "mlflow").exists()
