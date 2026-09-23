import json
from datetime import date
from pathlib import Path

import duckdb
import exchange_calendars as xcals
import numpy as np
import pytest

from marketsignal.evaluation import (
    FEATURES,
    EvaluationError,
    Example,
    _fold,
    _load_ticker,
    _probabilities,
    _require_source_boundary_sessions,
    _scores,
    evaluate,
)
from marketsignal.features import (
    FEATURE_SCHEMA,
    LABEL_SCHEMA,
    PRICE_SCHEMA,
    _sha256,
    _write_parquet,
)


def fixture_data(root: Path, ticker: str = "SPY") -> None:
    calendar = xcals.get_calendar("XNYS", start="2019-01-01", end="2023-12-31")
    days = [
        timestamp.date() for timestamp in calendar.sessions_in_range("2019-01-02", "2023-12-29")
    ]
    prices = []
    features = []
    labels = []
    for index, day in enumerate(days):
        target = int(index % 4 in (1, 2))
        # A varying signal keeps both classes in all folds and avoids degenerate fits.
        signal = ((index * 17) % 101) / 101
        prices.append((ticker, day, 100.0, 101.0, 99.0, 100.0, 100.0, 1000))
        features.append((ticker, day, *[signal + offset / 10 for offset in range(7)]))
        if index + 5 < len(days):
            labels.append((ticker, day, days[index + 5], 0.01 if target else -0.01, target))
    source = root / "processed" / "prices" / f"{ticker}.parquet"
    feature = root / "features" / f"{ticker}.parquet"
    label = root / "labels" / f"{ticker}.parquet"
    for path in (source, feature, label):
        path.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet(source, PRICE_SCHEMA, prices)
    _write_parquet(feature, FEATURE_SCHEMA, features)
    _write_parquet(label, LABEL_SCHEMA, labels)
    manifest = {
        "ticker": ticker,
        "source": {
            "path": str(source.relative_to(root)),
            "sha256": _sha256(source),
            "first_session_date": days[0].isoformat(),
            "last_session_date": days[-1].isoformat(),
            "row_count": len(days),
        },
        "feature_version": "v1",
        "target_horizon_sessions": 5,
        "calendar": {"identifier": "XNYS", "library_version": xcals.__version__},
        "feature_output": {
            "path": str(feature.relative_to(root)),
            "sha256": _sha256(feature),
            "row_count": len(features),
        },
        "label_output": {
            "path": str(label.relative_to(root)),
            "sha256": _sha256(label),
            "row_count": len(labels),
        },
    }
    (root / "features" / f"{ticker}.manifest.json").write_text(json.dumps(manifest))


def records(path: Path) -> list[tuple]:
    with duckdb.connect() as connection:
        return connection.execute("SELECT * FROM read_parquet(?)", [str(path)]).fetchall()


def test_fold_boundaries_and_class_guardrails() -> None:
    manifest = {
        "calendar": {"identifier": "XNYS"},
        "source": {"first_session_date": "2019-01-02", "last_session_date": "2023-12-29"},
    }
    examples = [
        Example(date(2021, 12, 30), date(2022, 1, 7), 1, (1.0,) * 7),
        Example(date(2022, 1, 3), date(2022, 1, 10), 0, (1.0,) * 7),
        Example(date(2022, 12, 30), date(2023, 1, 9), 1, (1.0,) * 7),
    ]
    with pytest.raises(EvaluationError, match="training 0 rows.*validation 1 rows"):
        _fold(examples, manifest, "SPY", 2022)


def test_one_class_fold_is_rejected() -> None:
    calendar = xcals.get_calendar("XNYS", start="2019-01-01", end="2022-12-31")
    days = [session.date() for session in calendar.sessions]
    examples = [
        Example(day, days[index + 5], int(day.year == 2022 and index % 2 == 0), (1.0,) * 7)
        for index, day in enumerate(days[:-5])
    ]
    manifest = {
        "calendar": {"identifier": "XNYS"},
        "source": {
            "first_session_date": days[0].isoformat(),
            "last_session_date": days[-1].isoformat(),
        },
    }
    with pytest.raises(EvaluationError, match="training.*positive=0"):
        _fold(examples, manifest, "SPY", 2022)


def test_metrics_known_values() -> None:
    actual = np.array([0, 0, 1, 1])
    good = _scores(actual, np.array([0.1, 0.2, 0.8, 0.9]))
    bad = _scores(actual, np.array([0.9, 0.8, 0.2, 0.1]))
    assert good["accuracy"] == 1.0
    assert good["roc_auc"] == 1.0
    assert good["brier_score"] == pytest.approx(0.025)
    assert bad["accuracy"] == 0.0
    assert bad["roc_auc"] == 0.0


def test_evaluation_outputs_reproducible_and_queryable(tmp_path: Path) -> None:
    fixture_data(tmp_path)
    first = evaluate(tmp_path, ["SPY"], 2022, 2)
    second = evaluate(tmp_path, ["SPY"], 2022, 2)
    for name in ("predictions.parquet", "fold_metrics.parquet", "leaderboard.parquet"):
        assert records(first / name) == records(second / name)
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["features"] == list(FEATURES)
    assert manifest["calendars"] == {"SPY": "XNYS"}
    assert manifest["configuration"]["folds"] == 2
    assert len(manifest["folds"]) == 2
    assert manifest["models"]["random_forest"]["random_state"] == 42
    for name, entry in manifest["outputs"].items():
        assert _sha256(first / name) == entry["sha256"]
    prediction_rows = records(first / "predictions.parquet")
    metrics = records(first / "fold_metrics.parquet")
    leaderboard = records(first / "leaderboard.parquet")
    assert len(metrics) == 8
    assert len(leaderboard) == 4
    for year in (2022, 2023):
        by_model = {}
        for row in prediction_rows:
            if row[1] == year:
                by_model.setdefault(row[2], []).append((row[3], row[4], row[5]))
        assert len(by_model) == 4
        assert all(rows == next(iter(by_model.values())) for rows in by_model.values())
    for row in metrics:
        assert row[3] >= 500 and row[4] >= 100
    assert all(row[2] == 2 for row in leaderboard)


def test_stale_source_and_inadequate_fold_publish_nothing(tmp_path: Path) -> None:
    fixture_data(tmp_path)
    with pytest.raises(EvaluationError, match="inadequate fold"):
        evaluate(tmp_path, ["SPY"], 2020, 1)
    assert not (tmp_path / "evaluations").exists()
    source = tmp_path / "processed" / "prices" / "SPY.parquet"
    source.write_bytes(source.read_bytes() + b"stale")
    with pytest.raises(EvaluationError, match="stale source"):
        evaluate(tmp_path, ["SPY"], 2022, 1)
    assert not (tmp_path / "evaluations").exists()


def test_invalid_schema_is_rejected(tmp_path: Path) -> None:
    fixture_data(tmp_path)
    feature = tmp_path / "features" / "SPY.parquet"
    with duckdb.connect() as connection:
        connection.execute("CREATE TABLE bad (ticker VARCHAR, session_date DATE)")
        connection.execute("INSERT INTO bad VALUES ('SPY', '2020-01-02')")
        feature.unlink()
        connection.execute(f"COPY bad TO '{feature}' (FORMAT PARQUET)")
    manifest_path = tmp_path / "features" / "SPY.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["feature_output"]["sha256"] = _sha256(feature)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(EvaluationError, match="schema"):
        evaluate(tmp_path, ["SPY"], 2022, 1)


def test_training_only_prevalence_and_scaling() -> None:
    train = [
        Example(date(2020, 1, 2), date(2020, 1, 9), target, (float(i),) * 7)
        for i, target in enumerate((0, 0, 0, 1))
    ]
    validation = [Example(date(2021, 1, 4), date(2021, 1, 11), 1, (1000.0,) * 7)]
    assert _probabilities("training_prevalence", train, validation).tolist() == [0.25]
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    expected = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    expected.fit(
        np.array([row.predictors for row in train]), np.array([row.target for row in train])
    )
    assert _probabilities("logistic_regression", train, validation)[0] == pytest.approx(
        expected.predict_proba(np.array([validation[0].predictors]))[0, 1]
    )


def test_unknown_labels_and_null_predictors_counted(tmp_path: Path) -> None:
    fixture_data(tmp_path)
    feature = tmp_path / "features" / "SPY.parquet"
    rows = records(feature)
    rows[0] = (*rows[0][:2], None, *rows[0][3:])
    feature.unlink()
    _write_parquet(feature, FEATURE_SCHEMA, rows)
    manifest_path = tmp_path / "features" / "SPY.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["feature_output"]["sha256"] = _sha256(feature)
    manifest_path.write_text(json.dumps(manifest))
    examples, _, drops = _load_ticker(tmp_path, "SPY")
    assert drops == {"unknown_label": 5, "null_or_nonfinite_predictor": 1}
    assert len(examples) == len(rows) - 6


def test_two_tickers_are_independent(tmp_path: Path) -> None:
    fixture_data(tmp_path, "SPY")
    fixture_data(tmp_path, "QQQ")
    run = evaluate(tmp_path, ["SPY", "QQQ"], 2022, 1)
    predictions = records(run / "predictions.parquet")
    assert {row[0] for row in predictions} == {"SPY", "QQQ"}
    counts = {ticker: sum(row[0] == ticker for row in predictions) for ticker in ("SPY", "QQQ")}
    assert counts["SPY"] == counts["QQQ"]
    manifest = json.loads((run / "manifest.json").read_text())
    assert len(manifest["folds"]) == 2


def test_source_must_have_year_boundary_sessions(tmp_path: Path) -> None:
    fixture_data(tmp_path)
    source = tmp_path / "processed" / "prices" / "SPY.parquet"
    rows = [row for row in records(source) if row[1] != date(2022, 1, 3)]
    source.unlink()
    _write_parquet(source, PRICE_SCHEMA, rows)
    with pytest.raises(EvaluationError, match="missing first or last"):
        _require_source_boundary_sessions(
            source,
            "SPY",
            {
                "year": 2022,
                "first_expected_session": "2022-01-03",
                "last_expected_session": "2022-12-30",
            },
        )


def test_model_failure_does_not_publish(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture_data(tmp_path)
    from marketsignal import evaluation

    original = evaluation._probabilities

    def fail_random_forest(
        model: str, train: list[Example], validation: list[Example]
    ) -> np.ndarray:
        if model == "random_forest":
            raise ValueError("synthetic fit failure")
        return original(model, train, validation)

    monkeypatch.setattr(evaluation, "_probabilities", fail_random_forest)
    with pytest.raises(EvaluationError, match="SPY 2022 random_forest.*synthetic fit failure"):
        evaluate(tmp_path, ["SPY"], 2022, 1)
    assert not (tmp_path / "evaluations").exists()
