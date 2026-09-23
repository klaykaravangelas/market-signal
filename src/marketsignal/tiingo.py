"""Tiingo end-of-day provider adapter."""

import json
from datetime import date
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from marketsignal.prices import PriceDataError, PriceRow


class TiingoProvider:
    name = "tiingo"

    def __init__(self, token: str | None = None) -> None:
        self.token = token

    def fetch_prices(self, ticker: str, start_date: date, end_date: date) -> bytes:
        if not self.token:
            raise ValueError("Set TIINGO_API_TOKEN before fetching prices")
        query = urlencode({"startDate": start_date.isoformat(), "endDate": end_date.isoformat()})
        url = f"https://api.tiingo.com/tiingo/daily/{quote(ticker, safe='')}/prices?{query}"
        request = Request(
            url,
            headers={"Authorization": f"Token {self.token}", "Accept": "application/json"},
        )
        try:
            with urlopen(request, timeout=30) as response:
                payload = response.read()
        except HTTPError as exc:
            raise PriceDataError(f"Tiingo request failed with HTTP {exc.code}") from None
        except URLError as exc:
            raise PriceDataError("Tiingo request failed due to a network error") from exc
        if self.token.encode() in payload:
            raise PriceDataError(
                "Provider response contained the API token; refusing to persist it"
            )
        return payload

    def decode_prices(self, payload: bytes, ticker: str) -> list[PriceRow]:
        try:
            data = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PriceDataError("Tiingo returned invalid JSON") from exc
        if not isinstance(data, list):
            raise PriceDataError("Tiingo response must be a list of daily prices")
        rows = []
        for item in data:
            if not isinstance(item, dict):
                raise PriceDataError("Tiingo price record must be an object")
            try:
                timestamp = item["date"]
                if not isinstance(timestamp, str) or len(timestamp) < 10:
                    raise ValueError("invalid date")
                session_date = date.fromisoformat(timestamp[:10])
                prices = [
                    self._price(item[key]) for key in ("open", "high", "low", "close", "adjClose")
                ]
                volume = item["volume"]
                if isinstance(volume, bool) or not isinstance(volume, int):
                    raise ValueError("invalid volume")
            except (KeyError, TypeError, ValueError) as exc:
                raise PriceDataError("Tiingo price record has missing or invalid fields") from exc
            rows.append(PriceRow(ticker, session_date, *prices, volume))
        return rows

    @staticmethod
    def _price(value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("invalid price")
        return float(value)
