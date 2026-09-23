"""Capture, replay, and atomic local Parquet storage for daily prices."""

import json
import os
import re
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import uuid4

import duckdb

from marketsignal.prices import MarketDataProvider, PriceDataError, PriceRow, validate_prices

SCHEMA_VERSION = 1
_TICKER = re.compile(r"[A-Z0-9][A-Z0-9-]*\Z")
_COLUMNS = "ticker, session_date, open, high, low, close, adjusted_close, volume"


def normalize_tickers(symbols: list[str]) -> list[str]:
    tickers = list(dict.fromkeys(symbol.strip().upper() for symbol in symbols))
    if not tickers or any(not _TICKER.fullmatch(ticker) for ticker in tickers):
        raise ValueError("Provide one or more ticker symbols using letters, digits, or hyphens")
    return tickers


def validate_range(start_date: date, end_date: date) -> None:
    if start_date > end_date:
        raise ValueError("Start date must not be after end date")


def ingest(
    provider: MarketDataProvider, ticker: str, start_date: date, end_date: date, data_dir: Path
) -> tuple[Path, list[PriceRow]]:
    ticker = normalize_tickers([ticker])[0]
    validate_range(start_date, end_date)
    payload = provider.fetch_prices(ticker, start_date, end_date)
    capture_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}-{uuid4().hex[:8]}"
    capture_dir = data_dir / "raw" / "prices" / provider.name / capture_id
    capture_dir.mkdir(parents=True, exist_ok=False)
    (capture_dir / "response.json").write_bytes(payload)
    manifest = {
        "provider": provider.name,
        "ticker": ticker,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "raw_file": "response.json",
        "schema_version": SCHEMA_VERSION,
        "status": "captured",
    }
    _write_manifest(capture_dir, manifest)
    return _process_capture(provider, capture_dir, manifest, payload, data_dir)


def replay(
    provider: MarketDataProvider, capture_dir: Path, data_dir: Path
) -> tuple[Path, list[PriceRow]]:
    manifest = json.loads((capture_dir / "manifest.json").read_text())
    if not isinstance(manifest, dict):
        raise PriceDataError("Capture manifest must be a JSON object")
    if (
        manifest.get("provider") != provider.name
        or manifest.get("schema_version") != SCHEMA_VERSION
    ):
        raise PriceDataError("Capture provider or schema version does not match")
    try:
        ticker = normalize_tickers([manifest["ticker"]])[0]
        start_date = date.fromisoformat(manifest["start_date"])
        end_date = date.fromisoformat(manifest["end_date"])
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise PriceDataError("Capture manifest has invalid ticker or dates") from exc
    validate_range(start_date, end_date)
    if manifest.get("raw_file") != "response.json":
        raise PriceDataError("Invalid raw file in manifest")
    payload = (capture_dir / "response.json").read_bytes()
    manifest["ticker"] = ticker
    return _process_capture(provider, capture_dir, manifest, payload, data_dir)


def _process_capture(
    provider: MarketDataProvider,
    capture_dir: Path,
    manifest: dict[str, object],
    payload: bytes,
    data_dir: Path,
) -> tuple[Path, list[PriceRow]]:
    ticker = str(manifest["ticker"])
    start_date = date.fromisoformat(str(manifest["start_date"]))
    end_date = date.fromisoformat(str(manifest["end_date"]))
    try:
        rows = validate_prices(
            provider.decode_prices(payload, ticker), ticker, start_date, end_date
        )
        output = data_dir / "processed" / "prices" / f"{ticker}.parquet"
        _merge_parquet(output, rows)
    except Exception:
        manifest["status"] = "failed"
        _write_manifest(capture_dir, manifest)
        raise
    manifest["status"] = "processed"
    manifest["processed_file"] = str(output.relative_to(data_dir))
    manifest["row_count"] = len(rows)
    manifest["first_session_date"] = rows[0].session_date.isoformat()
    manifest["last_session_date"] = rows[-1].session_date.isoformat()
    _write_manifest(capture_dir, manifest)
    return output, rows


def _write_manifest(capture_dir: Path, manifest: dict[str, object]) -> None:
    target = capture_dir / "manifest.json"
    temporary = capture_dir / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, target)


def _merge_parquet(output: Path, rows: list[PriceRow]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}-{uuid4().hex}.parquet")
    connection = duckdb.connect()
    try:
        connection.execute(
            "CREATE TABLE prices (ticker VARCHAR, session_date DATE, open DOUBLE, high DOUBLE, "
            "low DOUBLE, close DOUBLE, adjusted_close DOUBLE, volume BIGINT)"
        )
        if output.exists():
            existing = connection.execute(
                f"SELECT {_COLUMNS} FROM read_parquet(?)", [str(output)]
            ).fetchall()
            if any(record[0] != rows[0].ticker for record in existing):
                raise PriceDataError("Existing Parquet file contains another ticker")
            new_dates = {row.session_date for row in rows}
            retained = [record for record in existing if record[1] not in new_dates]
            if retained:
                connection.executemany(
                    "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?)", retained
                )
        connection.executemany(
            "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row.ticker,
                    row.session_date,
                    row.open,
                    row.high,
                    row.low,
                    row.close,
                    row.adjusted_close,
                    row.volume,
                )
                for row in rows
            ],
        )
        # The path is generated locally, never supplied as SQL input by a provider.
        quoted = str(temporary).replace("'", "''")
        connection.execute(
            f"COPY (SELECT {_COLUMNS} FROM prices ORDER BY ticker, session_date) "
            f"TO '{quoted}' (FORMAT PARQUET)"
        )
        os.replace(temporary, output)
    finally:
        connection.close()
        temporary.unlink(missing_ok=True)
