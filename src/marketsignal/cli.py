"""Command line entry point for local market data workflows."""

import argparse
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

from marketsignal.backtesting import BacktestError, backtest
from marketsignal.evaluation import EvaluationError, evaluate
from marketsignal.features import FeatureDataError, build_features
from marketsignal.ingestion import ingest, normalize_tickers, replay, validate_range
from marketsignal.prices import PriceDataError
from marketsignal.tiingo import TiingoProvider


def _date(value: str) -> date:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise argparse.ArgumentTypeError("Use an ISO date in YYYY-MM-DD format")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use an ISO date in YYYY-MM-DD format") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="marketsignal")
    subcommands = parser.add_subparsers(dest="command", required=True)
    fetch = subcommands.add_parser("ingest", help="Fetch and store daily prices")
    fetch.add_argument("tickers", nargs="+", help="Ticker symbols, for example SPY QQQ")
    fetch.add_argument("--start", type=_date, required=True)
    fetch.add_argument("--end", type=_date, required=True)
    fetch.add_argument("--data-dir", type=Path, default=Path("data"))
    repeat = subcommands.add_parser("replay", help="Reprocess one raw capture offline")
    repeat.add_argument("capture_dir", type=Path)
    repeat.add_argument("--data-dir", type=Path, default=Path("data"))
    feature_command = subcommands.add_parser("features", help="Build local features and labels")
    feature_command.add_argument("tickers", nargs="+", help="Ticker symbols, for example SPY QQQ")
    feature_command.add_argument("--data-dir", type=Path, default=Path("data"))
    feature_command.add_argument("--calendar", help="Explicit exchange calendar for other tickers")
    evaluation = subcommands.add_parser("evaluate", help="Evaluate local direction models")
    evaluation.add_argument("tickers", nargs="+", help="Ticker symbols, for example SPY QQQ")
    evaluation.add_argument("--first-validation-year", type=int, required=True)
    evaluation.add_argument("--folds", type=int, required=True)
    evaluation.add_argument("--data-dir", type=Path, default=Path("data"))
    backtest_command = subcommands.add_parser("backtest", help="Backtest a saved evaluation")
    backtest_command.add_argument("evaluation_run_id", help="Directory name under data/evaluations")
    backtest_command.add_argument("--data-dir", type=Path, default=Path("data"))
    backtest_command.add_argument("--fee-bps", type=float, default=1.0)
    backtest_command.add_argument("--slippage-bps", type=float, default=5.0)
    args = parser.parse_args(argv)
    try:
        if args.command == "backtest":
            output = backtest(
                args.data_dir, args.evaluation_run_id, args.fee_bps, args.slippage_bps
            )
            print(f"Backtest saved -> {output}")
            return 0
        if args.command == "evaluate":
            output = evaluate(args.data_dir, args.tickers, args.first_validation_year, args.folds)
            manifest = json.loads((output / "manifest.json").read_text())
            for ticker, counts in manifest["dropped_rows"].items():
                print(
                    f"{ticker}: dropped {counts['unknown_label']} unknown labels, "
                    f"{counts['null_or_nonfinite_predictor']} incomplete predictors"
                )
            print(f"Evaluation saved -> {output}")
            return 0
        if args.command == "features":
            tickers = normalize_tickers(args.tickers)
            failures = 0
            for ticker in tickers:
                try:
                    result = build_features(args.data_dir, ticker, args.calendar)
                    print(
                        f"{ticker}: {result.feature_count} feature rows, "
                        f"{result.label_count} labels, "
                        f"{len(result.missing_sessions)} missing sessions -> "
                        f"{result.manifest_path}"
                    )
                except (OSError, ValueError, FeatureDataError) as exc:
                    print(f"{ticker}: {exc}", file=sys.stderr)
                    failures += 1
            return 1 if failures else 0
        provider = TiingoProvider(os.environ.get("TIINGO_API_TOKEN"))
        if args.command == "ingest":
            tickers = normalize_tickers(args.tickers)
            validate_range(args.start, args.end)
            if not provider.token:
                raise ValueError("Set TIINGO_API_TOKEN before fetching prices")
            failures = 0
            for ticker in tickers:
                try:
                    output, rows = ingest(provider, ticker, args.start, args.end, args.data_dir)
                    print(
                        f"{ticker}: {len(rows)} rows, {rows[0].session_date} to "
                        f"{rows[-1].session_date} -> {output}"
                    )
                except (OSError, ValueError, PriceDataError) as exc:
                    print(f"{ticker}: {exc}", file=sys.stderr)
                    failures += 1
            return 1 if failures else 0
        output, rows = replay(provider, args.capture_dir, args.data_dir)
        print(f"Replayed {len(rows)} rows -> {output}")
        return 0
    except (OSError, KeyError, ValueError, PriceDataError, EvaluationError, BacktestError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
