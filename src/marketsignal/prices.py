"""Provider-independent daily price contract and validation."""

from dataclasses import dataclass
from datetime import date
from math import isfinite
from typing import Protocol


class PriceDataError(ValueError):
    """A provider response cannot form a valid price dataset."""


@dataclass(frozen=True, slots=True)
class PriceRow:
    ticker: str
    session_date: date
    open: float
    high: float
    low: float
    close: float
    adjusted_close: float
    volume: int


class MarketDataProvider(Protocol):
    name: str

    def fetch_prices(self, ticker: str, start_date: date, end_date: date) -> bytes:
        """Return the original provider response bytes for the requested range."""

    def decode_prices(self, payload: bytes, ticker: str) -> list[PriceRow]:
        """Convert provider fields to the normalized price contract."""


def validate_prices(
    rows: list[PriceRow], ticker: str, start_date: date, end_date: date
) -> list[PriceRow]:
    if not rows:
        raise PriceDataError(f"No prices returned for {ticker} in the requested range")
    seen: set[date] = set()
    for row in rows:
        if row.ticker != ticker or not start_date <= row.session_date <= end_date:
            raise PriceDataError("Provider returned a different ticker or a date outside the range")
        if row.session_date in seen:
            raise PriceDataError(f"Duplicate session date for {ticker}: {row.session_date}")
        seen.add(row.session_date)
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not isfinite(value)
            or value <= 0
            for value in (row.open, row.high, row.low, row.close, row.adjusted_close)
        ):
            raise PriceDataError(f"Invalid price for {ticker} on {row.session_date}")
        if isinstance(row.volume, bool) or not isinstance(row.volume, int) or row.volume < 0:
            raise PriceDataError(f"Invalid volume for {ticker} on {row.session_date}")
    return sorted(rows, key=lambda row: row.session_date)
