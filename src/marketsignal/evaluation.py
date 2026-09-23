"""Reproducible local evaluation of five-session direction models."""

import json
import math
import platform
import shutil
import statistics
import warnings
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import uuid4

import duckdb
import exchange_calendars as xcals
import numpy as np
import sklearn
from exchange_calendars.errors import CalendarError
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from marketsignal.features import (
    FEATURE_SCHEMA,
    FEATURE_VERSION,
    HORIZON,
    LABEL_SCHEMA,
    _sha256,
    _write_parquet,
    verify_manifest,
)
from marketsignal.ingestion import normalize_tickers

FEATURES = (
    "return_1",
    "return_5",
    "return_20",
    "volatility_5",
    "volatility_20",
    "ma_distance_20",
    "relative_volume_20",
)
MODELS = ("always_positive", "training_prevalence", "logistic_regression", "random_forest")
METRICS = ("accuracy", "roc_auc", "precision", "recall", "f1", "brier_score")
PREDICTION_SCHEMA = [
    ("ticker", "VARCHAR"),
    ("fold_year", "INTEGER"),
    ("model", "VARCHAR"),
    ("session_date", "DATE"),
    ("label_end_date", "DATE"),
    ("actual", "INTEGER"),
    ("predicted", "INTEGER"),
    ("probability_positive", "DOUBLE"),
]
FOLD_SCHEMA = [
    ("ticker", "VARCHAR"),
    ("fold_year", "INTEGER"),
    ("model", "VARCHAR"),
    ("training_rows", "INTEGER"),
    ("validation_rows", "INTEGER"),
    ("training_positive_count", "INTEGER"),
    ("validation_positive_count", "INTEGER"),
    ("validation_positive_rate", "DOUBLE"),
    *[(metric, "DOUBLE") for metric in METRICS],
]
LEADERBOARD_SCHEMA = [
    ("ticker", "VARCHAR"),
    ("model", "VARCHAR"),
    ("fold_count", "INTEGER"),
    *[(f"{metric}_{suffix}", "DOUBLE") for metric in METRICS for suffix in ("mean", "std")],
]
PARAMETERS = {
    "always_positive": {"probability": 1.0},
    "training_prevalence": {"probability": "training positive fraction"},
    "logistic_regression": {"scaler": "StandardScaler", "C": 1.0, "max_iter": 1000},
    "random_forest": {"n_estimators": 200, "min_samples_leaf": 5, "random_state": 42, "n_jobs": 1},
}


class EvaluationError(ValueError):
    """The requested comparison cannot be evaluated faithfully."""


@dataclass(frozen=True, slots=True)
class Example:
    session_date: date
    label_end_date: date
    target: int
    predictors: tuple[float, ...]


def _read_rows(path: Path, schema: list[tuple[str, str]]) -> list[tuple[object, ...]]:
    with duckdb.connect() as connection:
        connection.execute("SET threads = 1")
        try:
            actual = connection.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]
            ).fetchall()
            if [(row[0], row[1]) for row in actual] != schema:
                raise EvaluationError(f"Unexpected Parquet schema: {path}; rebuild features")
            return connection.execute("SELECT * FROM read_parquet(?)", [str(path)]).fetchall()
        except duckdb.Error as exc:
            raise EvaluationError(f"Cannot read {path}; rebuild features") from exc


def _load_ticker(
    data_dir: Path, ticker: str
) -> tuple[list[Example], dict[str, object], dict[str, int]]:
    try:
        manifest = verify_manifest(data_dir, ticker)
    except (ValueError, OSError) as exc:
        raise EvaluationError(f"{ticker}: {exc}; rebuild features") from exc
    if (
        manifest.get("feature_version") != FEATURE_VERSION
        or manifest.get("target_horizon_sessions") != HORIZON
    ):
        raise EvaluationError(f"{ticker}: unsupported feature or target version; rebuild features")
    source = data_dir / "processed" / "prices" / f"{ticker}.parquet"
    source_entry = manifest.get("source")
    if not isinstance(source_entry, dict) or source_entry.get("path") != str(
        source.relative_to(data_dir)
    ):
        raise EvaluationError(f"{ticker}: invalid source path; rebuild features")
    if not source.is_file() or _sha256(source) != source_entry.get("sha256"):
        raise EvaluationError(f"{ticker}: stale source price file; rebuild features")
    feature_rows = _read_rows(data_dir / "features" / f"{ticker}.parquet", FEATURE_SCHEMA)
    label_rows = _read_rows(data_dir / "labels" / f"{ticker}.parquet", LABEL_SCHEMA)
    for key, path in (
        ("source", source),
        ("feature_output", data_dir / "features" / f"{ticker}.parquet"),
        ("label_output", data_dir / "labels" / f"{ticker}.parquet"),
    ):
        if _sha256(path) != manifest[key]["sha256"]:
            raise EvaluationError(f"{ticker}: {key} changed while reading; rebuild features")
    if len(feature_rows) != manifest["feature_output"].get("row_count") or len(
        label_rows
    ) != manifest["label_output"].get("row_count"):
        raise EvaluationError(f"{ticker}: row count differs from manifest; rebuild features")
    features_by_date: dict[date, tuple[object, ...]] = {}
    for row in feature_rows:
        if row[0] != ticker or not isinstance(row[1], date) or row[1] in features_by_date:
            raise EvaluationError(f"{ticker}: invalid ticker, date, or duplicate feature key")
        features_by_date[row[1]] = row[2:]
    labels_by_date: dict[date, tuple[object, ...]] = {}
    for row in label_rows:
        session, end, future_return, target = row[1:]
        if row[0] != ticker or not isinstance(session, date) or session in labels_by_date:
            raise EvaluationError(f"{ticker}: invalid ticker, date, or duplicate label key")
        if session not in features_by_date or not isinstance(end, date) or end <= session:
            raise EvaluationError(f"{ticker}: invalid label end date or missing feature key")
        if target not in (0, 1) or future_return is None or not math.isfinite(future_return):
            raise EvaluationError(f"{ticker}: invalid target or future return")
        if target != int(future_return > 0):
            raise EvaluationError(f"{ticker}: target disagrees with future return")
        labels_by_date[session] = row[2:]
    examples = []
    drops = {"unknown_label": 0, "null_or_nonfinite_predictor": 0}
    for session in sorted(features_by_date):
        label = labels_by_date.get(session)
        if label is None:
            drops["unknown_label"] += 1
            continue
        predictors = features_by_date[session]
        if any(value is None or not math.isfinite(value) for value in predictors):
            drops["null_or_nonfinite_predictor"] += 1
            continue
        examples.append(Example(session, label[0], label[2], tuple(predictors)))
    return examples, manifest, drops


def _fold(
    examples: list[Example], manifest: dict[str, object], ticker: str, year: int
) -> tuple[list[Example], list[Example], dict[str, object]]:
    calendar_entry = manifest.get("calendar")
    name = calendar_entry.get("identifier") if isinstance(calendar_entry, dict) else None
    if not isinstance(name, str):
        raise EvaluationError(f"{ticker}: missing exchange calendar; rebuild features")
    try:
        calendar = xcals.get_calendar(name, start=f"{year}-01-01", end=f"{year}-12-31")
        sessions = calendar.sessions
        first, last = sessions[0].date(), sessions[-1].date()
    except (CalendarError, ValueError, KeyError, IndexError) as exc:
        raise EvaluationError(f"{ticker} {year}: cannot load exchange calendar {name}") from exc
    source_entry = manifest["source"]
    source_first = date.fromisoformat(source_entry["first_session_date"])
    source_last = date.fromisoformat(source_entry["last_session_date"])
    if source_first > first or source_last < last:
        raise EvaluationError(f"{ticker} {year}: price history must cover {first} through {last}")
    train = [row for row in examples if row.session_date < first and row.label_end_date < first]
    validation = [
        row for row in examples if first <= row.session_date <= last and row.label_end_date <= last
    ]
    train_positive = sum(row.target for row in train)
    validation_positive = sum(row.target for row in validation)
    if (
        len(train) < 500
        or len(validation) < 100
        or train_positive in (0, len(train))
        or validation_positive in (0, len(validation))
    ):
        raise EvaluationError(
            f"{ticker} {year}: inadequate fold: training {len(train)} rows "
            f"(positive={train_positive}, negative={len(train) - train_positive}), "
            f"validation {len(validation)} rows (positive={validation_positive}, "
            f"negative={len(validation) - validation_positive}); need 500/100 rows and both classes"
        )
    detail = {
        "ticker": ticker,
        "year": year,
        "first_expected_session": first.isoformat(),
        "last_expected_session": last.isoformat(),
        "training": {
            "first_session": train[0].session_date.isoformat(),
            "last_session": train[-1].session_date.isoformat(),
            "rows": len(train),
            "positive": train_positive,
            "negative": len(train) - train_positive,
        },
        "validation": {
            "first_session": validation[0].session_date.isoformat(),
            "last_session": validation[-1].session_date.isoformat(),
            "rows": len(validation),
            "positive": validation_positive,
            "negative": len(validation) - validation_positive,
        },
    }
    return train, validation, detail


def _require_source_boundary_sessions(source: Path, ticker: str, detail: dict[str, object]) -> None:
    first = detail["first_expected_session"]
    last = detail["last_expected_session"]
    try:
        with duckdb.connect() as connection:
            dates = {
                row[0].isoformat()
                for row in connection.execute(
                    "SELECT session_date FROM read_parquet(?) WHERE session_date IN (?, ?)",
                    [str(source), first, last],
                ).fetchall()
            }
    except duckdb.Error as exc:
        raise EvaluationError(f"{ticker} {detail['year']}: cannot read source sessions") from exc
    if dates != {first, last}:
        raise EvaluationError(
            f"{ticker} {detail['year']}: source price file is missing first or last "
            "expected exchange session; rebuild features"
        )


def _scores(actual: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predicted = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "roc_auc": float(roc_auc_score(actual, probabilities)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "brier_score": float(brier_score_loss(actual, probabilities)),
    }


def _probabilities(model: str, train: list[Example], validation: list[Example]) -> np.ndarray:
    train_x = np.asarray([row.predictors for row in train], dtype=float)
    train_y = np.asarray([row.target for row in train], dtype=int)
    val_x = np.asarray([row.predictors for row in validation], dtype=float)
    if model == "always_positive":
        return np.ones(len(validation))
    if model == "training_prevalence":
        return np.full(len(validation), float(train_y.mean()))
    if model == "logistic_regression":
        estimator = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    else:
        estimator = RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=1
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator.fit(train_x, train_y)
    return estimator.predict_proba(val_x)[:, 1]


def evaluate(data_dir: Path, tickers: list[str], first_validation_year: int, folds: int) -> Path:
    """Evaluate all requested folds and atomically publish one immutable run."""
    if (
        folds < 1
        or not 1900 <= first_validation_year <= 9999
        or first_validation_year + folds - 1 > 9999
    ):
        raise EvaluationError("Use a valid first validation year and a positive fold count")
    tickers = normalize_tickers(tickers)
    predictions: list[tuple[object, ...]] = []
    fold_metrics: list[tuple[object, ...]] = []
    fold_details: list[dict[str, object]] = []
    inputs: dict[str, object] = {}
    calendars: dict[str, str] = {}
    drop_counts: dict[str, dict[str, int]] = {}
    for ticker in tickers:
        examples, manifest, drops = _load_ticker(data_dir, ticker)
        calendars[ticker] = manifest["calendar"]["identifier"]
        drop_counts[ticker] = drops
        paths = [
            data_dir / "features" / f"{ticker}.manifest.json",
            data_dir / "features" / f"{ticker}.parquet",
            data_dir / "labels" / f"{ticker}.parquet",
            data_dir / "processed" / "prices" / f"{ticker}.parquet",
        ]
        inputs[ticker] = [
            {"path": str(path.relative_to(data_dir)), "sha256": _sha256(path)} for path in paths
        ]
        for item, key in zip(
            inputs[ticker][1:], ("feature_output", "label_output", "source"), strict=True
        ):
            if item["sha256"] != manifest[key]["sha256"]:
                raise EvaluationError(f"{ticker}: {key} changed while reading; rebuild features")
        for year in range(first_validation_year, first_validation_year + folds):
            train, validation, detail = _fold(examples, manifest, ticker, year)
            _require_source_boundary_sessions(
                data_dir / "processed" / "prices" / f"{ticker}.parquet", ticker, detail
            )
            fold_details.append(detail)
            actual = np.asarray([row.target for row in validation], dtype=int)
            for model in MODELS:
                try:
                    probabilities = _probabilities(model, train, validation)
                    if not np.all(np.isfinite(probabilities)) or np.any(
                        (probabilities < 0) | (probabilities > 1)
                    ):
                        raise ValueError("invalid probabilities")
                    scores = _scores(actual, probabilities)
                except (ValueError, RuntimeError, ConvergenceWarning) as exc:
                    raise EvaluationError(
                        f"{ticker} {year} {model}: model fit or scoring failed: {exc}"
                    ) from exc
                for row, probability in zip(validation, probabilities, strict=True):
                    predictions.append(
                        (
                            ticker,
                            year,
                            model,
                            row.session_date,
                            row.label_end_date,
                            row.target,
                            int(probability >= 0.5),
                            float(probability),
                        )
                    )
                fold_metrics.append(
                    (
                        ticker,
                        year,
                        model,
                        len(train),
                        len(validation),
                        detail["training"]["positive"],
                        detail["validation"]["positive"],
                        float(actual.mean()),
                        *(scores[metric] for metric in METRICS),
                    )
                )
    leaderboard: list[tuple[object, ...]] = []
    for ticker in tickers:
        for model in MODELS:
            rows = [row for row in fold_metrics if row[0] == ticker and row[2] == model]
            values = []
            for metric in METRICS:
                index = [name for name, _ in FOLD_SCHEMA].index(metric)
                series = [row[index] for row in rows]
                values.extend((statistics.fmean(series), statistics.pstdev(series)))
            leaderboard.append((ticker, model, len(rows), *values))
    root = data_dir / "evaluations"
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC)
    run_id = timestamp.strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]
    final = root / run_id
    temporary = root / f".{run_id}.tmp"
    temporary.mkdir()
    try:
        for ticker_inputs in inputs.values():
            for item in ticker_inputs:
                if _sha256(data_dir / item["path"]) != item["sha256"]:
                    raise EvaluationError(
                        "An input changed during evaluation; rebuild features and retry"
                    )
        _write_parquet(temporary / "predictions.parquet", PREDICTION_SCHEMA, predictions)
        _write_parquet(temporary / "fold_metrics.parquet", FOLD_SCHEMA, fold_metrics)
        _write_parquet(temporary / "leaderboard.parquet", LEADERBOARD_SCHEMA, leaderboard)
        for ticker_inputs in inputs.values():
            for item in ticker_inputs:
                if _sha256(data_dir / item["path"]) != item["sha256"]:
                    raise EvaluationError("An input changed during evaluation; retry")
        manifest = {
            "run_id": run_id,
            "generated_at_utc": timestamp.isoformat(),
            "configuration": {
                "tickers": tickers,
                "first_validation_year": first_validation_year,
                "folds": folds,
            },
            "inputs": inputs,
            "calendars": calendars,
            "feature_version": FEATURE_VERSION,
            "features": list(FEATURES),
            "target": {
                "column": "target_positive_5",
                "horizon_sessions": HORIZON,
                "positive_if": "future_return_5 > 0",
            },
            "folds": fold_details,
            "dropped_rows": drop_counts,
            "models": PARAMETERS,
            "random_seed": 42,
            "versions": {
                "python": platform.python_version(),
                "scikit_learn": sklearn.__version__,
                "duckdb": duckdb.__version__,
            },
            "outputs": {
                name: {"path": name, "sha256": _sha256(temporary / name)}
                for name in ("predictions.parquet", "fold_metrics.parquet", "leaderboard.parquet")
            },
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        temporary.rename(final)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return final
