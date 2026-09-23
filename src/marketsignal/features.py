"""Build leakage-aware daily features and five-session labels from local prices."""

import hashlib
import json
import math
import os
import statistics
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import duckdb
import exchange_calendars as xcals
from exchange_calendars.errors import CalendarError

from marketsignal.ingestion import normalize_tickers
from marketsignal.prices import PriceRow, validate_prices

FEATURE_VERSION = "v1"
HORIZON = 5
PRICE_SCHEMA = [
    ("ticker", "VARCHAR"),
    ("session_date", "DATE"),
    ("open", "DOUBLE"),
    ("high", "DOUBLE"),
    ("low", "DOUBLE"),
    ("close", "DOUBLE"),
    ("adjusted_close", "DOUBLE"),
    ("volume", "BIGINT"),
]
FEATURE_SCHEMA = [
    ("ticker", "VARCHAR"),
    ("session_date", "DATE"),
    ("return_1", "DOUBLE"),
    ("return_5", "DOUBLE"),
    ("return_20", "DOUBLE"),
    ("volatility_5", "DOUBLE"),
    ("volatility_20", "DOUBLE"),
    ("ma_distance_20", "DOUBLE"),
    ("relative_volume_20", "DOUBLE"),
]
LABEL_SCHEMA = [
    ("ticker", "VARCHAR"),
    ("session_date", "DATE"),
    ("label_end_date", "DATE"),
    ("future_return_5", "DOUBLE"),
    ("target_positive_5", "INTEGER"),
]


class FeatureDataError(ValueError):
    """The source prices or generated dataset violate the feature contract."""


@dataclass(frozen=True, slots=True)
class BuildResult:
    ticker: str
    feature_path: Path
    label_path: Path
    manifest_path: Path
    feature_count: int
    label_count: int
    missing_sessions: tuple[date, ...]


def build_features(data_dir: Path, ticker: str, calendar_name: str | None = None) -> BuildResult:
    """Build current feature and label files from one validated price snapshot."""
    ticker = normalize_tickers([ticker])[0]
    if calendar_name is None:
        if ticker not in {"SPY", "QQQ"}:
            raise FeatureDataError(f"Specify --calendar for {ticker}; no exchange can be inferred")
        calendar_name = "XNYS"

    source = data_dir / "processed" / "prices" / f"{ticker}.parquet"
    if not source.is_file():
        raise FeatureDataError(f"Price file not found: {source}")
    source_hash = _sha256(source)
    prices = _read_prices(source, ticker)
    if _sha256(source) != source_hash:
        raise FeatureDataError("Source price file changed during the build")
    positions, missing = _session_positions(prices, calendar_name)
    feature_rows, label_rows = _calculate(prices, positions)

    feature_path = data_dir / "features" / f"{ticker}.parquet"
    label_path = data_dir / "labels" / f"{ticker}.parquet"
    manifest_path = data_dir / "features" / f"{ticker}.manifest.json"
    _publish(
        source,
        source_hash,
        prices,
        feature_path,
        feature_rows,
        label_path,
        label_rows,
        manifest_path,
        calendar_name,
        missing,
    )
    verify_manifest(data_dir, ticker)
    return BuildResult(
        ticker,
        feature_path,
        label_path,
        manifest_path,
        len(feature_rows),
        len(label_rows),
        tuple(missing),
    )


def verify_manifest(data_dir: Path, ticker: str) -> dict[str, object]:
    """Refuse an output pair that does not match its published manifest."""
    ticker = normalize_tickers([ticker])[0]
    path = data_dir / "features" / f"{ticker}.manifest.json"
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise FeatureDataError(f"Missing or invalid feature manifest: {path}") from exc
    if not isinstance(manifest, dict) or manifest.get("ticker") != ticker:
        raise FeatureDataError("Feature manifest has the wrong ticker")
    for key, expected in (
        ("feature_output", data_dir / "features" / f"{ticker}.parquet"),
        ("label_output", data_dir / "labels" / f"{ticker}.parquet"),
    ):
        entry = manifest.get(key)
        if not isinstance(entry, dict) or entry.get("path") != str(expected.relative_to(data_dir)):
            raise FeatureDataError(f"Feature manifest has an invalid {key} path")
        if not expected.is_file() or _sha256(expected) != entry.get("sha256"):
            raise FeatureDataError(f"{key} does not match its published manifest")
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_prices(source: Path, ticker: str) -> list[PriceRow]:
    with duckdb.connect() as connection:
        connection.execute("SET threads = 1")
        try:
            schema = connection.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [str(source)]
            ).fetchall()
            if [(row[0], row[1]) for row in schema] != PRICE_SCHEMA:
                raise FeatureDataError("Unexpected price Parquet schema")
            records = connection.execute("SELECT * FROM read_parquet(?)", [str(source)]).fetchall()
        except duckdb.Error as exc:
            raise FeatureDataError(f"Cannot read price Parquet file: {source}") from exc
    if not records:
        raise FeatureDataError("Price Parquet file is empty")
    rows = [PriceRow(*record) for record in records]
    dates = [row.session_date for row in rows]
    if dates != sorted(dates) or len(dates) != len(set(dates)):
        raise FeatureDataError("Price sessions must be unique and ascending")
    try:
        validate_prices(rows, ticker, dates[0], dates[-1])
    except ValueError as exc:
        raise FeatureDataError(str(exc)) from exc
    return rows


def _session_positions(prices: list[PriceRow], calendar_name: str) -> tuple[list[int], list[date]]:
    try:
        calendar = xcals.get_calendar(
            calendar_name,
            start=(prices[0].session_date - timedelta(days=1)).isoformat(),
            end=(prices[-1].session_date + timedelta(days=1)).isoformat(),
        )
        expected = [
            timestamp.date()
            for timestamp in calendar.sessions_in_range(
                prices[0].session_date.isoformat(), prices[-1].session_date.isoformat()
            )
        ]
    except (CalendarError, ValueError, KeyError) as exc:
        raise FeatureDataError(f"Cannot load exchange calendar {calendar_name}") from exc
    positions_by_date = {session: index for index, session in enumerate(expected)}
    observed = {row.session_date for row in prices}
    non_sessions = sorted(observed - positions_by_date.keys())
    if non_sessions:
        raise FeatureDataError(f"Price file contains a non-session date: {non_sessions[0]}")
    missing = sorted(positions_by_date.keys() - observed)
    return [positions_by_date[row.session_date] for row in prices], missing


def _calculate(
    prices: list[PriceRow], positions: list[int]
) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    count = len(prices)
    closes = [row.adjusted_close for row in prices]
    volumes = [row.volume for row in prices]

    def complete(start: int, end: int) -> bool:
        return start >= 0 and end < count and positions[end] - positions[start] == end - start

    def checked(value: float) -> float:
        if not math.isfinite(value):
            raise FeatureDataError("Feature or label calculation produced a non-finite value")
        return value

    features: list[tuple[object, ...]] = []
    labels: list[tuple[object, ...]] = []
    for index, price in enumerate(prices):
        returns = {
            window: checked(closes[index] / closes[index - window] - 1)
            if complete(index - window, index)
            else None
            for window in (1, 5, 20)
        }
        volatility = {}
        for window in (5, 20):
            if complete(index - window, index):
                daily = [
                    closes[j] / closes[j - 1] - 1 for j in range(index - window + 1, index + 1)
                ]
                volatility[window] = checked(statistics.stdev(daily))
            else:
                volatility[window] = None
        moving_average = (
            checked(closes[index] / statistics.fmean(closes[index - 19 : index + 1]) - 1)
            if complete(index - 19, index)
            else None
        )
        relative_volume = None
        if complete(index - 20, index):
            previous_mean = statistics.fmean(volumes[index - 20 : index])
            if previous_mean > 0:
                relative_volume = checked(volumes[index] / previous_mean)
        features.append(
            (
                price.ticker,
                price.session_date,
                returns[1],
                returns[5],
                returns[20],
                volatility[5],
                volatility[20],
                moving_average,
                relative_volume,
            )
        )

        if complete(index, index + HORIZON):
            future_return = checked(closes[index + HORIZON] / closes[index] - 1)
            labels.append(
                (
                    price.ticker,
                    price.session_date,
                    prices[index + HORIZON].session_date,
                    future_return,
                    int(future_return > 0),
                )
            )
    return features, labels


def _write_parquet(
    path: Path, schema: list[tuple[str, str]], rows: list[tuple[object, ...]]
) -> None:
    columns = ", ".join(f"{name} {data_type}" for name, data_type in schema)
    placeholders = ", ".join("?" for _ in schema)
    with duckdb.connect() as connection:
        connection.execute(f"CREATE TABLE output ({columns})")
        if rows:
            connection.executemany(f"INSERT INTO output VALUES ({placeholders})", rows)
        quoted = str(path).replace("'", "''")
        connection.execute(f"COPY output TO '{quoted}' (FORMAT PARQUET)")


def _publish(
    source: Path,
    source_hash: str,
    prices: list[PriceRow],
    feature_path: Path,
    feature_rows: list[tuple[object, ...]],
    label_path: Path,
    label_rows: list[tuple[object, ...]],
    manifest_path: Path,
    calendar_name: str,
    missing: list[date],
) -> None:
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    nonce = uuid4().hex
    temporary_feature = feature_path.with_name(f".{feature_path.stem}-{nonce}.parquet")
    temporary_label = label_path.with_name(f".{label_path.stem}-{nonce}.parquet")
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}-{nonce}.tmp")
    data_dir = source.parents[2]
    try:
        _write_parquet(temporary_feature, FEATURE_SCHEMA, feature_rows)
        _write_parquet(temporary_label, LABEL_SCHEMA, label_rows)
        manifest = {
            "ticker": prices[0].ticker,
            "source": {
                "path": str(source.relative_to(data_dir)),
                "sha256": source_hash,
                "first_session_date": prices[0].session_date.isoformat(),
                "last_session_date": prices[-1].session_date.isoformat(),
                "row_count": len(prices),
            },
            "feature_version": FEATURE_VERSION,
            "target_horizon_sessions": HORIZON,
            "calendar": {"identifier": calendar_name, "library_version": xcals.__version__},
            "feature_output": {
                "path": str(feature_path.relative_to(data_dir)),
                "sha256": _sha256(temporary_feature),
                "row_count": len(feature_rows),
            },
            "label_output": {
                "path": str(label_path.relative_to(data_dir)),
                "sha256": _sha256(temporary_label),
                "row_count": len(label_rows),
            },
            "missing_expected_sessions": {
                "count": len(missing),
                "first": missing[0].isoformat() if missing else None,
                "last": missing[-1].isoformat() if missing else None,
            },
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "point_in_time_certified": False,
        }
        temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary_feature, feature_path)
        os.replace(temporary_label, label_path)
        os.replace(temporary_manifest, manifest_path)
    finally:
        for temporary in (temporary_feature, temporary_label, temporary_manifest):
            temporary.unlink(missing_ok=True)
