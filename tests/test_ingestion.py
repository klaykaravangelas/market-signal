import json
from datetime import date
from pathlib import Path

import duckdb
import pytest

from marketsignal.cli import main
from marketsignal.ingestion import ingest, normalize_tickers, replay
from marketsignal.prices import PriceDataError
from marketsignal.tiingo import TiingoProvider

START = date(2024, 1, 2)
END = date(2024, 1, 5)


def record(day: str, close: float = 100.0, **changes: object) -> dict[str, object]:
    result = {
        "date": f"{day}T00:00:00.000Z",
        "open": 99.0,
        "high": 101.0,
        "low": 98.0,
        "close": close,
        "adjClose": close - 1,
        "volume": 1000,
        "extraProviderField": "preserved",
    }
    result.update(changes)
    return result


class FakeProvider(TiingoProvider):
    def __init__(self, records: list[dict[str, object]]) -> None:
        super().__init__()
        self.records = records
        self.calls: list[tuple[str, date, date]] = []

    def fetch_prices(self, ticker: str, start_date: date, end_date: date) -> bytes:
        self.calls.append((ticker, start_date, end_date))
        return json.dumps(self.records).encode()


def read_rows(path: Path) -> list[tuple]:
    with duckdb.connect() as connection:
        return connection.execute(
            "SELECT ticker, session_date, close, adjusted_close, volume "
            "FROM read_parquet(?) ORDER BY session_date",
            [str(path)],
        ).fetchall()


def test_capture_merge_and_offline_replay(tmp_path: Path) -> None:
    provider = FakeProvider([record("2024-01-03"), record("2024-01-02")])
    output, rows = ingest(provider, "spy", START, END, tmp_path)
    assert [row.session_date for row in rows] == [START, date(2024, 1, 3)]
    assert read_rows(output) == [
        ("SPY", START, 100.0, 99.0, 1000),
        ("SPY", date(2024, 1, 3), 100.0, 99.0, 1000),
    ]
    captures = list((tmp_path / "raw" / "prices" / "tiingo").iterdir())
    assert len(captures) == 1
    capture = captures[0]
    assert (
        json.loads((capture / "response.json").read_text())[0]["extraProviderField"] == "preserved"
    )
    manifest = json.loads((capture / "manifest.json").read_text())
    assert manifest["status"] == "processed"
    assert manifest["processed_file"] == "processed/prices/SPY.parquet"
    assert "token" not in json.dumps(manifest).lower()

    provider.records = [record("2024-01-03", close=105.0), record("2024-01-04")]
    ingest(provider, "SPY", date(2024, 1, 3), END, tmp_path)
    assert [row[2] for row in read_rows(output)] == [100.0, 105.0, 100.0]
    replay(provider, capture, tmp_path)
    assert [row[2] for row in read_rows(output)] == [100.0, 100.0, 100.0]
    assert len(read_rows(output)) == 3
    assert len(provider.calls) == 2  # replay did not fetch


@pytest.mark.parametrize(
    "records",
    [
        [],
        [record("2024-01-02"), record("2024-01-02")],
        [record("2024-01-06")],
        [record("2024-01-02", close=0)],
        [record("2024-01-02", adjClose=float("nan"))],
        [record("2024-01-02", volume=-1)],
        [record("2024-01-02", volume=None)],
        [record("2024-01-02", high=None)],
    ],
)
def test_bad_provider_data_preserves_existing_file(tmp_path: Path, records: list[dict]) -> None:
    provider = FakeProvider([record("2024-01-02")])
    output, _ = ingest(provider, "SPY", START, END, tmp_path)
    original = output.read_bytes()
    provider.records = records
    with pytest.raises(PriceDataError):
        ingest(provider, "SPY", START, END, tmp_path)
    assert output.read_bytes() == original
    captures = list((tmp_path / "raw" / "prices" / "tiingo").iterdir())
    assert len(captures) == 2
    assert any(
        json.loads((path / "manifest.json").read_text())["status"] == "failed" for path in captures
    )


def test_provider_failure_does_not_create_capture(tmp_path: Path) -> None:
    class FailingProvider(FakeProvider):
        def fetch_prices(self, ticker: str, start_date: date, end_date: date) -> bytes:
            raise PriceDataError("rate limited")

    with pytest.raises(PriceDataError, match="rate limited"):
        ingest(FailingProvider([]), "SPY", START, END, tmp_path)
    assert not (tmp_path / "raw").exists()


def test_inputs_and_cli_fail_before_provider_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert normalize_tickers(["spy", " SPY ", "qqq"]) == ["SPY", "QQQ"]
    for symbols in ([], [""], ["../SPY"]):
        with pytest.raises(ValueError):
            normalize_tickers(symbols)
    monkeypatch.setenv("TIINGO_API_TOKEN", "secret")
    assert (
        main(
            [
                "ingest",
                "SPY",
                "--start",
                "2024-01-05",
                "--end",
                "2024-01-02",
                "--data-dir",
                str(tmp_path),
            ]
        )
        == 1
    )
    assert not (tmp_path / "raw").exists()
    with pytest.raises(SystemExit):
        main(["ingest", "SPY", "--start", "20240102", "--end", "2024-01-05"])


def test_tiingo_request_uses_header_and_decodes_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def read(self) -> bytes:
            return json.dumps([record("2024-01-02")]).encode()

    def fake_urlopen(request: object, timeout: int) -> Response:
        seen["url"] = request.full_url
        seen["auth"] = request.get_header("Authorization")
        seen["timeout"] = timeout
        return Response()

    monkeypatch.setattr("marketsignal.tiingo.urlopen", fake_urlopen)
    provider = TiingoProvider("secret")
    payload = provider.fetch_prices("SPY", START, END)
    assert "secret" not in seen["url"]
    assert seen["auth"] == "Token secret"
    assert "startDate=2024-01-02" in seen["url"]
    assert provider.decode_prices(payload, "SPY")[0].adjusted_close == 99.0


def test_cli_ingests_multiple_tickers_with_one_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider([record("2024-01-02")])
    provider.token = "test-token"
    monkeypatch.setattr("marketsignal.cli.TiingoProvider", lambda token: provider)
    monkeypatch.setenv("TIINGO_API_TOKEN", "test-token")
    result = main(
        [
            "ingest",
            "spy",
            "SPY",
            "QQQ",
            "--start",
            "2024-01-02",
            "--end",
            "2024-01-05",
            "--data-dir",
            str(tmp_path),
        ]
    )
    assert result == 0
    assert [call[0] for call in provider.calls] == ["SPY", "QQQ"]
    assert read_rows(tmp_path / "processed" / "prices" / "QQQ.parquet")[0][0] == "QQQ"
