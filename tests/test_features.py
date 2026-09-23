import json
import statistics
from datetime import date
from pathlib import Path

import duckdb
import exchange_calendars as xcals
import pytest

from marketsignal.cli import main
from marketsignal.features import FeatureDataError, build_features, verify_manifest
from marketsignal.ingestion import ingest
from marketsignal.tiingo import TiingoProvider


def sessions(count: int = 32) -> list[date]:
    calendar = xcals.get_calendar("XNYS", start="2024-01-01", end="2024-04-01")
    return [session.date() for session in calendar.sessions_in_range("2024-01-02", "2024-03-01")][
        :count
    ]


def prices(
    ticker: str = "SPY", count: int = 32
) -> list[tuple[str, date, float, float, float, float, float, int]]:
    return [
        (ticker, day, 99.0 + i, 101.0 + i, 98.0 + i, 100.0 + i, 100.0 + i, 100 + i)
        for i, day in enumerate(sessions(count))
    ]


def write_prices(data_dir: Path, rows: list[tuple], *, malformed_schema: bool = False) -> Path:
    ticker = rows[0][0]
    output = data_dir / "processed" / "prices" / f"{ticker}.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    with duckdb.connect() as connection:
        if malformed_schema:
            connection.execute("CREATE TABLE prices (ticker VARCHAR, session_date DATE)")
            connection.execute("INSERT INTO prices VALUES (?, ?)", rows[0][:2])
        else:
            connection.execute(
                "CREATE TABLE prices (ticker VARCHAR, session_date DATE, open DOUBLE, "
                "high DOUBLE, low DOUBLE, close DOUBLE, adjusted_close DOUBLE, volume BIGINT)"
            )
            connection.executemany("INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        quoted = str(output).replace("'", "''")
        connection.execute(f"COPY prices TO '{quoted}' (FORMAT PARQUET)")
    return output


def read_table(path: Path) -> tuple[list[str], list[tuple]]:
    with duckdb.connect() as connection:
        result = connection.execute(
            "SELECT * FROM read_parquet(?) ORDER BY session_date", [str(path)]
        )
        return [column[0] for column in result.description], result.fetchall()


def test_feature_formulas_labels_and_manifest(tmp_path: Path) -> None:
    source = write_prices(tmp_path, prices())
    result = build_features(tmp_path, "SPY")
    columns, features = read_table(result.feature_path)
    label_columns, labels = read_table(result.label_path)
    assert result.feature_count == 32
    assert result.label_count == 27
    assert result.missing_sessions == ()
    assert columns == [
        "ticker",
        "session_date",
        "return_1",
        "return_5",
        "return_20",
        "volatility_5",
        "volatility_20",
        "ma_distance_20",
        "relative_volume_20",
    ]
    assert label_columns == [
        "ticker",
        "session_date",
        "label_end_date",
        "future_return_5",
        "target_positive_5",
    ]
    assert features[0][2:] == (None,) * 7
    assert features[19][4] is None
    t = 20
    expected_daily = [(100 + j) / (99 + j) - 1 for j in range(1, t + 1)]
    assert features[t][2] == pytest.approx(120 / 119 - 1)
    assert features[t][3] == pytest.approx(120 / 115 - 1)
    assert features[t][4] == pytest.approx(120 / 100 - 1)
    assert features[t][5] == pytest.approx(statistics.stdev(expected_daily[-5:]))
    assert features[t][6] == pytest.approx(statistics.stdev(expected_daily))
    assert features[t][7] == pytest.approx(120 / statistics.fmean(range(101, 121)) - 1)
    assert features[t][8] == pytest.approx(120 / statistics.fmean(range(100, 120)))
    assert labels[t][2] == sessions()[25]
    assert labels[t][3] == pytest.approx(125 / 120 - 1)
    assert labels[t][4] == 1
    assert labels[-1][1] == sessions()[26]
    assert all(row[1] not in {session for session in sessions()[27:]} for row in labels)

    manifest = verify_manifest(tmp_path, "SPY")
    assert manifest["source"]["path"] == str(source.relative_to(tmp_path))
    assert manifest["source"]["row_count"] == 32
    assert manifest["calendar"] == {"identifier": "XNYS", "library_version": xcals.__version__}
    assert manifest["feature_version"] == "v1"
    assert manifest["target_horizon_sessions"] == 5
    assert manifest["point_in_time_certified"] is False
    build_features(tmp_path, "SPY")
    assert read_table(result.feature_path)[1] == features
    assert read_table(result.label_path)[1] == labels


def test_future_change_does_not_change_current_features(tmp_path: Path) -> None:
    original = prices()
    write_prices(tmp_path, original)
    result = build_features(tmp_path, "SPY")
    before_features = read_table(result.feature_path)[1]
    before_labels = read_table(result.label_path)[1]
    revised = original.copy()
    row = list(revised[25])
    row[6] = 50.0
    row[7] = 9999
    revised[25] = tuple(row)
    write_prices(tmp_path, revised)
    build_features(tmp_path, "SPY")
    assert read_table(result.feature_path)[1][20] == before_features[20]
    after_labels = read_table(result.label_path)[1]
    assert after_labels[20][3] < 0
    assert after_labels[20][4] == 0
    assert before_labels[20][4] == 1


def test_zero_return_and_zero_prior_volume(tmp_path: Path) -> None:
    rows = prices()
    changed = []
    for i, row in enumerate(rows):
        record = list(row)
        if i < 20:
            record[7] = 0
        if i == 25:
            record[6] = 120.0
        changed.append(tuple(record))
    write_prices(tmp_path, changed)
    result = build_features(tmp_path, "SPY")
    assert read_table(result.feature_path)[1][20][8] is None
    label = read_table(result.label_path)[1][20]
    assert label[3] == 0.0
    assert label[4] == 0


def test_short_history_keeps_features_and_has_no_labels(tmp_path: Path) -> None:
    write_prices(tmp_path, prices(count=1))
    result = build_features(tmp_path, "SPY")
    assert result.feature_count == 1
    assert result.label_count == 0
    assert read_table(result.label_path)[1] == []
    assert read_table(result.feature_path)[1][0][2:] == (None,) * 7


def test_missing_session_invalidates_crossing_windows(tmp_path: Path) -> None:
    rows = prices()
    missing_date = rows[12][1]
    write_prices(tmp_path, rows[:12] + rows[13:])
    result = build_features(tmp_path, "SPY")
    assert result.missing_sessions == (missing_date,)
    features = read_table(result.feature_path)[1]
    labels = read_table(result.label_path)[1]
    assert features[12][2] is None  # The one-session return crosses the missing date.
    assert features[13][2] is not None
    assert features[15][3] is None  # The five-session window still crosses it.
    assert rows[8][1] not in {label[1] for label in labels}
    manifest = verify_manifest(tmp_path, "SPY")
    assert manifest["missing_expected_sessions"]["count"] == 1
    assert manifest["missing_expected_sessions"]["first"] == missing_date.isoformat()


@pytest.mark.parametrize("problem", ["duplicate", "unsorted", "non_session", "wrong_ticker"])
def test_invalid_rows_do_not_replace_outputs(tmp_path: Path, problem: str) -> None:
    original = prices()
    write_prices(tmp_path, original)
    result = build_features(tmp_path, "SPY")
    prior = [
        path.read_bytes() for path in (result.feature_path, result.label_path, result.manifest_path)
    ]
    broken = original.copy()
    if problem == "duplicate":
        broken.insert(5, broken[5])
    elif problem == "unsorted":
        broken[5], broken[6] = broken[6], broken[5]
    elif problem == "non_session":
        row = list(broken[5])
        row[1] = date(2024, 1, 6)
        broken[5] = tuple(row)
        broken.sort(key=lambda item: item[1])
    else:
        row = list(broken[5])
        row[0] = "QQQ"
        broken[5] = tuple(row)
    write_prices(tmp_path, broken)
    with pytest.raises(FeatureDataError):
        build_features(tmp_path, "SPY")
    assert [
        path.read_bytes() for path in (result.feature_path, result.label_path, result.manifest_path)
    ] == prior


def test_bad_schema_and_unsupported_symbol(tmp_path: Path) -> None:
    write_prices(tmp_path, prices(), malformed_schema=True)
    with pytest.raises(FeatureDataError, match="schema"):
        build_features(tmp_path, "SPY")
    write_prices(tmp_path, prices("AAPL"))
    with pytest.raises(FeatureDataError, match="calendar"):
        build_features(tmp_path, "AAPL")
    with pytest.raises(FeatureDataError, match="Cannot load exchange calendar"):
        build_features(tmp_path, "AAPL", "NOT_A_CALENDAR")
    assert build_features(tmp_path, "AAPL", "XNYS").feature_count == 32


def test_failed_calculation_and_interrupted_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_prices(tmp_path, prices())
    result = build_features(tmp_path, "SPY")
    prior = [
        path.read_bytes() for path in (result.feature_path, result.label_path, result.manifest_path)
    ]

    import marketsignal.features as feature_module

    original_write = feature_module._write_parquet
    calls = 0

    def fail_second_file(path: Path, schema: list, rows: list) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk full")
        original_write(path, schema, rows)

    monkeypatch.setattr(feature_module, "_write_parquet", fail_second_file)
    with pytest.raises(OSError, match="disk full"):
        build_features(tmp_path, "SPY")
    assert [
        path.read_bytes() for path in (result.feature_path, result.label_path, result.manifest_path)
    ] == prior
    assert not list((tmp_path / "features").glob(".*.parquet"))
    monkeypatch.undo()
    result.label_path.write_bytes(b"interrupted publication")
    with pytest.raises(FeatureDataError, match="does not match"):
        verify_manifest(tmp_path, "SPY")


def test_cli_runs_without_token_and_duckdb_join(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TIINGO_API_TOKEN", raising=False)
    write_prices(tmp_path, prices("SPY"))
    write_prices(tmp_path, prices("QQQ"))
    assert main(["features", "spy", "SPY", "QQQ", "--data-dir", str(tmp_path)]) == 0
    with duckdb.connect() as connection:
        count = connection.execute(
            "SELECT count(*) FROM read_parquet(?) AS f JOIN read_parquet(?) AS l "
            "USING (ticker, session_date) WHERE f.return_20 IS NOT NULL",
            [str(tmp_path / "features" / "SPY.parquet"), str(tmp_path / "labels" / "SPY.parquet")],
        ).fetchone()[0]
    assert count == 7
    assert json.loads((tmp_path / "features" / "QQQ.manifest.json").read_text())["ticker"] == "QQQ"


def test_build_uses_actual_ingestion_output_offline(tmp_path: Path) -> None:
    class FixtureProvider(TiingoProvider):
        def fetch_prices(self, ticker: str, start_date: date, end_date: date) -> bytes:
            records = [
                {
                    "date": f"{row[1]}T00:00:00.000Z",
                    "open": row[2],
                    "high": row[3],
                    "low": row[4],
                    "close": row[5],
                    "adjClose": row[6],
                    "volume": row[7],
                }
                for row in prices(ticker)
            ]
            return json.dumps(records).encode()

    observed_sessions = sessions()
    ingest(FixtureProvider(), "SPY", observed_sessions[0], observed_sessions[-1], tmp_path)
    result = build_features(tmp_path, "SPY")
    assert result.feature_count == len(observed_sessions)
    assert result.label_count == len(observed_sessions) - 5
    assert verify_manifest(tmp_path, "SPY")["source"]["row_count"] == len(observed_sessions)
