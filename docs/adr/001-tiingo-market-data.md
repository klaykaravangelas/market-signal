# ADR 001 — Tiingo for initial daily prices

## Status

Accepted for Spec 001

## Decision

Use Tiingo's end-of-day REST API as the first market-data adapter. Each user supplies their own `TIINGO_API_TOKEN`; downloaded data stays local and is ignored by Git. The application sends the token in an authorization header, not a URL or manifest.

Tiingo returns raw daily open, high, low, close, and volume, plus a separate `adjClose`. Its documentation says the adjusted fields incorporate split and dividend adjustments. We store raw OHLC and the provider's adjusted close without synthesizing adjustments. Historical values can be corrected or retrospectively adjusted, so raw response captures are retained.

The [EOD documentation](https://www.tiingo.com/documentation/end-of-day) describes the fields and date-range endpoint. Tiingo's [connection guide](https://www.tiingo.com/documentation/general/connecting) documents header authentication. Its [pricing page](https://www.tiingo.com/about/pricing) lists a free Starter tier, and the [general API documentation](https://www.tiingo.com/documentation/general) limits basic data to internal or personal use and permits applications that require users to supply their own tokens without distributing data.

## Alternatives and consequences

- Alpha Vantage's adjusted daily endpoint is marked premium in its current [API documentation](https://www.alphavantage.co/documentation/), so it is not the first adapter for this low-cost project.
- The code depends on a provider protocol, so Tiingo can be replaced without rewriting storage or validation.
- A token and network access are needed for a live run. Automated tests use local fixtures and do not require either.
