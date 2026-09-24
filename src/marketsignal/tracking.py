"""Import verified, immutable local results into an optional MLflow store."""

import json
import math
import re
from pathlib import Path, PurePosixPath

import duckdb
import joblib
import sklearn

from marketsignal.backtesting import FOLD_SCHEMA as BACKTEST_FOLD_SCHEMA
from marketsignal.backtesting import INTERVAL_SCHEMA, SUMMARY_SCHEMA
from marketsignal.evaluation import (
    FEATURES,
    LEADERBOARD_SCHEMA,
    MODELS,
    PREDICTION_SCHEMA,
)
from marketsignal.evaluation import (
    FOLD_SCHEMA as EVALUATION_FOLD_SCHEMA,
)
from marketsignal.features import _sha256

EVALUATION_SCHEMAS = {
    "predictions.parquet": PREDICTION_SCHEMA,
    "fold_metrics.parquet": EVALUATION_FOLD_SCHEMA,
    "leaderboard.parquet": LEADERBOARD_SCHEMA,
}
BACKTEST_SCHEMAS = {
    "intervals.parquet": INTERVAL_SCHEMA,
    "fold_metrics.parquet": BACKTEST_FOLD_SCHEMA,
    "summary.parquet": SUMMARY_SCHEMA,
}


class TrackingError(ValueError):
    """A saved run or its MLflow import is invalid."""


def _run(data_dir: Path, kind: str, run_id: str, schemas: dict) -> tuple[Path, dict, str]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id):
        raise TrackingError(f"Provide a {kind} run ID, not a path")
    root = data_dir / kind / run_id
    try:
        manifest_path = root / "manifest.json"
        manifest_hash = _sha256(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        if not isinstance(manifest, dict) or manifest.get("run_id") != run_id:
            raise ValueError("manifest run ID mismatch")
        outputs = manifest["outputs"]
        if not isinstance(outputs, dict) or not set(schemas) <= outputs.keys():
            raise ValueError("missing required outputs")
        for name, entry in outputs.items():
            relative = PurePosixPath(name)
            if (
                not isinstance(name, str)
                or relative.is_absolute()
                or ".." in relative.parts
                or entry["path"] != name
            ):
                raise ValueError(f"unsafe output path: {name}")
            path = root / name
            if not path.is_file() or path.is_symlink() or _sha256(path) != entry["sha256"]:
                raise ValueError(f"altered or missing output: {name}")
        for name, schema in schemas.items():
            with duckdb.connect() as connection:
                actual = connection.execute(
                    "DESCRIBE SELECT * FROM read_parquet(?)", [str(root / name)]
                ).fetchall()
                if [(item[0], item[1]) for item in actual] != schema:
                    raise ValueError(f"unexpected schema: {name}")
        if _sha256(manifest_path) != manifest_hash:
            raise ValueError("manifest changed while reading")
        return root, manifest, manifest_hash
    except (OSError, KeyError, TypeError, ValueError, duckdb.Error) as exc:
        raise TrackingError(f"{kind}/{run_id}: invalid or altered source: {exc}") from exc


def _rows(path: Path, schema: list[tuple[str, str]]) -> list[dict]:
    with duckdb.connect() as connection:
        rows = connection.execute("SELECT * FROM read_parquet(?)", [str(path)]).fetchall()
    return [dict(zip((name for name, _ in schema), row, strict=True)) for row in rows]


def _evaluation(data_dir: Path, run_id: str) -> tuple[Path, dict, str, list[dict], dict]:
    root, manifest, digest = _run(data_dir, "evaluations", run_id, EVALUATION_SCHEMAS)
    try:
        config = manifest["configuration"]
        tickers = config["tickers"]
        years = range(
            config["first_validation_year"], config["first_validation_year"] + config["folds"]
        )
        expected = {
            (ticker, year, model) for ticker in tickers for year in years for model in MODELS
        }
        metrics = _rows(root / "fold_metrics.parquet", EVALUATION_FOLD_SCHEMA)
        keys = [(row["ticker"], row["fold_year"], row["model"]) for row in metrics]
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError("fold metric coverage differs from configuration")
        details = {(item["ticker"], item["year"]): item for item in manifest["folds"]}
        if len(details) != len(tickers) * len(years):
            raise ValueError("fold details differ from configuration")
        if not set(manifest["features"]) == set(FEATURES) or manifest["features"] != list(FEATURES):
            raise ValueError("unexpected feature order")
        has_records = "training_records.json" in manifest["outputs"]
        records = {}
        if has_records:
            saved = json.loads((root / "training_records.json").read_text())
            if saved["schema_version"] != 1:
                raise ValueError("unsupported training record version")
            records = {(r["ticker"], r["fold_year"], r["model"]): r for r in saved["records"]}
            if len(records) != len(expected) or set(records) != expected:
                raise ValueError("training record coverage differs from metrics")
            for ticker, year, model in expected:
                name = f"models/{ticker}/{year}/{model}.joblib"
                if (model in ("logistic_regression", "random_forest")) != (
                    name in manifest["outputs"]
                ):
                    raise ValueError(f"model artifact coverage differs for {ticker} {year} {model}")
        elif any(name.startswith("models/") for name in manifest["outputs"]):
            raise ValueError("model artifacts lack training records")
        return root, manifest, digest, metrics, records
    except (OSError, KeyError, TypeError, ValueError, duckdb.Error) as exc:
        raise TrackingError(f"evaluations/{run_id}: invalid result content: {exc}") from exc


def _backtest(
    data_dir: Path, run_id: str, evaluation_id: str, evaluation_hash: str
) -> tuple[Path, dict, str, list[dict]]:
    root, manifest, digest = _run(data_dir, "backtests", run_id, BACKTEST_SCHEMAS)
    try:
        if manifest["evaluation_run_id"] != evaluation_id or (
            manifest["evaluation_inputs"]["evaluation_manifest"] != evaluation_hash
        ):
            raise ValueError("backtest does not match the evaluation run and manifest")
        config = manifest["configuration"]
        windows = {(item["ticker"], item["year"]) for item in manifest["windows"]}
        expected = {
            (ticker, year, strategy)
            for ticker, year in windows
            for strategy in manifest["strategies"]
        }
        rows = _rows(root / "fold_metrics.parquet", BACKTEST_FOLD_SCHEMA)
        keys = [(r["ticker"], r["fold_year"], r["strategy"]) for r in rows]
        if (
            len(keys) != len(expected)
            or set(keys) != expected
            or set(config["tickers"]) != {ticker for ticker, _ in windows}
        ):
            raise ValueError("backtest fold metric coverage differs from manifest")
        return root, manifest, digest, rows
    except (OSError, KeyError, TypeError, ValueError, duckdb.Error) as exc:
        raise TrackingError(f"backtests/{run_id}: invalid result content: {exc}") from exc


def _mlflow():
    try:
        import mlflow
        import mlflow.sklearn  # noqa: F401
        from mlflow.tracking import MlflowClient
    except ImportError as exc:
        raise TrackingError("Install the tracking extra: uv sync --extra tracking") from exc
    return mlflow, MlflowClient


def _store(data_dir: Path, tracking_uri: str | None, artifact_root: Path | None) -> tuple[str, str]:
    database = (data_dir / "mlflow" / "tracking.db").resolve()
    artifacts = (artifact_root or data_dir / "mlflow" / "artifacts").resolve()
    if tracking_uri is None:
        database.parent.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"sqlite:///{database}"
    artifacts.mkdir(parents=True, exist_ok=True)
    return tracking_uri, artifacts.as_uri()


def _experiment(mlflow, name: str, artifacts: str) -> str:
    existing = mlflow.get_experiment_by_name(name)
    if existing is not None:
        if existing.artifact_location != f"{artifacts}/{name}":
            raise TrackingError(
                f"{name}: experiment artifact location differs from configured root"
            )
        return existing.experiment_id
    return mlflow.create_experiment(name, artifact_location=f"{artifacts}/{name}")


def _finished(client, experiment_id: str, source_id: str, digest: str) -> str | None:
    matches = client.search_runs(
        [experiment_id], filter_string=f"tags.source_run_id = '{source_id}'", max_results=1000
    )
    completed = []
    for run in matches:
        if run.data.tags.get("mlflow.parentRunId"):
            continue
        if run.data.tags.get("source_manifest_sha256") != digest:
            raise TrackingError(f"{source_id}: previously tracked manifest has a different hash")
        if run.info.status == "FINISHED":
            completed.append(run.info.run_id)
    if len(completed) > 1:
        raise TrackingError(f"{source_id}: more than one completed import exists")
    return completed[0] if completed else None


def _log_scalars(mlflow, data: dict, metric: bool = False) -> None:
    for name, value in data.items():
        if value is None:
            continue
        if metric:
            if isinstance(value, (int, float)) and math.isfinite(value):
                mlflow.log_metric(name, float(value))
        elif isinstance(value, (str, int, float, bool)):
            mlflow.log_param(name, value)


def _import_evaluation(mlflow, client, experiment_id: str, source: tuple) -> str:
    root, manifest, digest, rows, records = source
    source_id = manifest["run_id"]
    existing = _finished(client, experiment_id, source_id, digest)
    if existing:
        return existing
    details = {(d["ticker"], d["year"]): d for d in manifest["folds"]}
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=source_id,
        tags={
            "source_run_id": source_id,
            "source_manifest_sha256": digest,
            "results_only": str(not bool(records)).lower(),
            "source_kind": "evaluation",
        },
    ) as parent:
        _log_scalars(mlflow, manifest["configuration"])
        _log_scalars(mlflow, {"feature_version": manifest["feature_version"]})
        mlflow.log_dict(
            {
                "features": manifest["features"],
                "target": manifest["target"],
                "inputs": manifest["inputs"],
                "outputs": manifest["outputs"],
                "versions": manifest["versions"],
                "folds": manifest["folds"],
                "generated_at_utc": manifest["generated_at_utc"],
            },
            "lineage.json",
        )
        mlflow.log_artifact(str(root / "manifest.json"))
        for row in rows:
            ticker, year, model = row["ticker"], row["fold_year"], row["model"]
            detail = details[(ticker, year)]
            record = records.get((ticker, year, model))
            name = f"models/{ticker}/{year}/{model}.joblib"
            available = name in manifest["outputs"]
            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name=f"{ticker}-{year}-{model}",
                nested=True,
                tags={
                    "ticker": ticker,
                    "fold_year": str(year),
                    "candidate": model,
                    "model_artifact_available": str(available).lower(),
                    "model_artifact_reason": (
                        "saved fitted estimator"
                        if available
                        else "baseline or historical evaluation"
                    ),
                },
            ):
                _log_scalars(mlflow, {"ticker": ticker, "fold_year": year, "candidate": model})
                _log_scalars(mlflow, manifest["models"][model])
                _log_scalars(
                    mlflow,
                    {
                        "training_rows": row["training_rows"],
                        "validation_rows": row["validation_rows"],
                    },
                )
                mlflow.log_dict(detail, "fold.json")
                if record:
                    _log_scalars(mlflow, {"random_seed": record["random_seed"]})
                    _log_scalars(mlflow, {"fit_seconds": record["fit_seconds"]}, metric=True)
                    _log_scalars(mlflow, record["diagnostics"] or {}, metric=True)
                _log_scalars(
                    mlflow,
                    {
                        key: value
                        for key, value in row.items()
                        if key not in ("ticker", "fold_year", "model")
                    },
                    metric=True,
                )
                if available:
                    payload = joblib.load(root / name)
                    if payload["features"] != list(FEATURES):
                        raise TrackingError(f"{ticker} {year} {model}: model feature order differs")
                    info = mlflow.sklearn.log_model(
                        payload["estimator"],
                        name="model",
                        serialization_format="cloudpickle",
                        pip_requirements=[f"scikit-learn=={sklearn.__version__}"],
                    )
                    mlflow.set_tag("model_uri", info.model_uri)
                    mlflow.set_tag("source_model_sha256", manifest["outputs"][name]["sha256"])
        return parent.info.run_id


def _import_backtest(mlflow, client, experiment_id: str, source: tuple) -> str:
    root, manifest, digest, rows = source
    source_id = manifest["run_id"]
    existing = _finished(client, experiment_id, source_id, digest)
    if existing:
        return existing
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=source_id,
        tags={
            "source_run_id": source_id,
            "source_manifest_sha256": digest,
            "source_kind": "backtest",
            "evaluation_run_id": manifest["evaluation_run_id"],
            "evaluation_manifest_sha256": manifest["evaluation_inputs"]["evaluation_manifest"],
        },
    ) as parent:
        _log_scalars(mlflow, {k: v for k, v in manifest["configuration"].items() if k != "tickers"})
        mlflow.log_dict(
            {
                "evaluation_inputs": manifest["evaluation_inputs"],
                "price_sources": manifest["price_sources"],
                "outputs": manifest["outputs"],
                "strategy": manifest["strategy"],
                "windows": manifest["windows"],
                "versions": manifest["versions"],
                "generated_at_utc": manifest["generated_at_utc"],
            },
            "lineage.json",
        )
        mlflow.log_artifact(str(root / "manifest.json"))
        for row in rows:
            ticker, year, strategy = row["ticker"], row["fold_year"], row["strategy"]
            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name=f"{ticker}-{year}-{strategy}",
                nested=True,
                tags={"ticker": ticker, "fold_year": str(year), "strategy": strategy},
            ):
                _log_scalars(mlflow, {"ticker": ticker, "fold_year": year, "strategy": strategy})
                _log_scalars(
                    mlflow,
                    {
                        key: value
                        for key, value in row.items()
                        if key
                        not in (
                            "ticker",
                            "fold_year",
                            "strategy",
                            "first_execution",
                            "last_liquidation",
                        )
                    },
                    metric=True,
                )
                mlflow.log_dict(
                    {
                        "first_execution": row["first_execution"].isoformat(),
                        "last_liquidation": row["last_liquidation"].isoformat(),
                    },
                    "timing.json",
                )
        return parent.info.run_id


def track(
    data_dir: Path,
    evaluation_run_id: str,
    backtest_run_id: str | None = None,
    tracking_uri: str | None = None,
    artifact_root: Path | None = None,
) -> tuple[str, str | None, str, str]:
    """Verify saved sources and import their recorded results without refitting."""
    evaluation = _evaluation(data_dir, evaluation_run_id)
    backtest = (
        _backtest(data_dir, backtest_run_id, evaluation_run_id, evaluation[2])
        if backtest_run_id
        else None
    )
    mlflow, client_class = _mlflow()
    uri, artifacts = _store(data_dir, tracking_uri, artifact_root)
    try:
        mlflow.set_tracking_uri(uri)
        client = client_class(tracking_uri=uri)
        evaluation_experiment = _experiment(mlflow, "marketsignal-evaluation", artifacts)
        evaluation_mlflow_id = _import_evaluation(mlflow, client, evaluation_experiment, evaluation)
        backtest_mlflow_id = None
        if backtest:
            backtest_experiment = _experiment(mlflow, "marketsignal-backtest", artifacts)
            backtest_mlflow_id = _import_backtest(mlflow, client, backtest_experiment, backtest)
        if (
            _finished(client, evaluation_experiment, evaluation_run_id, evaluation[2])
            != evaluation_mlflow_id
        ):
            raise TrackingError("evaluation import did not finish consistently")
        if (
            backtest
            and _finished(client, backtest_experiment, backtest_run_id, backtest[2])
            != backtest_mlflow_id
        ):
            raise TrackingError("backtest import did not finish consistently")
        return evaluation_mlflow_id, backtest_mlflow_id, uri, artifacts
    except TrackingError:
        raise
    except Exception as exc:
        raise TrackingError(f"MLflow import failed for {evaluation_run_id}: {exc}") from exc
