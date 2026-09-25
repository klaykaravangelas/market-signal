"""Explain verified, saved validation predictions without refitting models."""

import html
import json
import math
import platform
import shutil
import statistics
from collections import defaultdict
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import uuid4

import duckdb
import numpy as np
import sklearn

from marketsignal.backtesting import STRATEGIES
from marketsignal.evaluation import (
    FEATURES,
    LEADERBOARD_SCHEMA,
    METRICS,
    MODELS,
    PREDICTION_SCHEMA,
    _scores,
)
from marketsignal.evaluation import FOLD_SCHEMA as EVALUATION_FOLD_SCHEMA
from marketsignal.features import FEATURE_VERSION, HORIZON, _sha256, _write_parquet
from marketsignal.ingestion import normalize_tickers
from marketsignal.tracking import (
    BACKTEST_SCHEMAS,
    EVALUATION_SCHEMAS,
    TrackingError,
    _backtest,
    _experiment,
    _finished,
    _mlflow,
    _rows,
    _run,
    _store,
)

VERSION = 1
BASELINES = ("always_positive", "training_prevalence")
FOLD_SCHEMA = [
    ("ticker", "VARCHAR"),
    ("fold_year", "INTEGER"),
    ("model", "VARCHAR"),
    ("training_rows", "INTEGER"),
    ("training_positive_count", "INTEGER"),
    ("training_positive_rate", "DOUBLE"),
    ("validation_rows", "INTEGER"),
    ("actual_positive_count", "INTEGER"),
    ("actual_positive_rate", "DOUBLE"),
    ("predicted_positive_count", "INTEGER"),
    ("predicted_positive_rate", "DOUBLE"),
    ("true_positive", "INTEGER"),
    ("false_positive", "INTEGER"),
    ("true_negative", "INTEGER"),
    ("false_negative", "INTEGER"),
    *[(f"probability_{name}", "DOUBLE") for name in ("min", "median", "mean", "max", "std")],
    *[
        (f"probability_{name}_fraction", "DOUBLE")
        for name in ("below_0_5", "equal_0_5", "above_0_5")
    ],
    *[(metric, "DOUBLE") for metric in METRICS],
    *[(f"saved_{metric}", "DOUBLE") for metric in METRICS],
    *[
        (f"{metric}_vs_{baseline}", "DOUBLE")
        for baseline in BASELINES
        for metric in ("accuracy_delta", "auc_delta", "brier_improvement")
    ],
]
BIN_SCHEMA = [
    ("ticker", "VARCHAR"),
    ("fold_year", "INTEGER"),
    ("model", "VARCHAR"),
    ("bin_index", "INTEGER"),
    ("bin_lower", "DOUBLE"),
    ("bin_upper", "DOUBLE"),
    ("count", "INTEGER"),
    ("mean_predicted_probability", "DOUBLE"),
    ("observed_positive_fraction", "DOUBLE"),
    ("sparse", "BOOLEAN"),
]
TRADING_SCHEMA = [
    ("ticker", "VARCHAR"),
    ("fold_year", "INTEGER"),
    ("strategy", "VARCHAR"),
    ("net_cumulative_return", "DOUBLE"),
    ("excess_vs_buy_and_hold", "DOUBLE"),
    ("exposure", "DOUBLE"),
    ("total_turnover", "DOUBLE"),
    ("fee_bps", "DOUBLE"),
    ("slippage_bps", "DOUBLE"),
    ("predicted_positive_rate", "DOUBLE"),
]
SCHEMAS = {
    "fold_diagnostics.parquet": FOLD_SCHEMA,
    "calibration_bins.parquet": BIN_SCHEMA,
}


class DiagnosticsError(ValueError):
    """Saved results cannot be explained faithfully."""


def _expected(manifest: dict) -> tuple[list[str], list[int], dict[tuple[str, int], dict]]:
    if manifest.get("feature_version") != FEATURE_VERSION or manifest.get("target") != {
        "column": "target_positive_5",
        "horizon_sessions": HORIZON,
        "positive_if": "future_return_5 > 0",
    }:
        raise DiagnosticsError("Unsupported evaluation feature or target version")
    if manifest.get("features") != list(FEATURES):
        raise DiagnosticsError("Unexpected evaluation feature order")
    config = manifest["configuration"]
    tickers = config["tickers"]
    start, count = config["first_validation_year"], config["folds"]
    if (
        not isinstance(tickers, list)
        or not tickers
        or normalize_tickers(tickers) != tickers
        or not isinstance(start, int)
        or not isinstance(count, int)
        or count < 1
        or not 1900 <= start <= 9999
        or start + count - 1 > 9999
    ):
        raise DiagnosticsError("Invalid evaluation fold configuration")
    years = list(range(start, start + count))
    details = manifest["folds"]
    by_key = {(item["ticker"], item["year"]): item for item in details}
    expected = {(ticker, year) for ticker in tickers for year in years}
    if len(details) != len(expected) or set(by_key) != expected:
        raise DiagnosticsError("Evaluation fold details differ from configuration")
    return tickers, years, by_key


def _validated_evaluation(data_dir: Path, run_id: str) -> dict:
    try:
        root, manifest, digest = _run(data_dir, "evaluations", run_id, EVALUATION_SCHEMAS)
        tickers, years, details = _expected(manifest)
        expected = {
            (ticker, year, model) for ticker in tickers for year in years for model in MODELS
        }
        saved = _rows(root / "fold_metrics.parquet", EVALUATION_FOLD_SCHEMA)
        metrics = {(r["ticker"], r["fold_year"], r["model"]): r for r in saved}
        if len(saved) != len(expected) or set(metrics) != expected:
            raise DiagnosticsError("Fold metric coverage differs from evaluation configuration")
        predictions = _rows(root / "predictions.parquet", PREDICTION_SCHEMA)
        grouped: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
        seen = set()
        for row in predictions:
            ticker, year, model = row["ticker"], row["fold_year"], row["model"]
            key = (ticker, year, model)
            session, end = row["session_date"], row["label_end_date"]
            if key not in expected or not isinstance(session, date) or not isinstance(end, date):
                raise DiagnosticsError("Prediction has an unexpected ticker, fold, model, or date")
            row_key = (*key, session)
            if row_key in seen:
                raise DiagnosticsError(f"Duplicate prediction key: {row_key}")
            seen.add(row_key)
            probability = row["probability_positive"]
            if probability is None or not math.isfinite(probability) or not 0 <= probability <= 1:
                raise DiagnosticsError(f"{key}: invalid probability")
            if row["actual"] not in (0, 1) or row["predicted"] != int(probability >= 0.5):
                raise DiagnosticsError(f"{key}: invalid actual or threshold prediction")
            if session.year != year or not session < end or end.year != year:
                raise DiagnosticsError(f"{key}: invalid validation or label end date")
            grouped[key].append(row)
        if set(grouped) != expected:
            raise DiagnosticsError("Prediction coverage differs from evaluation configuration")
        calculated = {}
        for ticker in tickers:
            for year in years:
                detail = details[ticker, year]
                reference = None
                for model in MODELS:
                    key = (ticker, year, model)
                    rows = sorted(grouped[key], key=lambda r: r["session_date"])
                    aligned = [(r["session_date"], r["label_end_date"], r["actual"]) for r in rows]
                    if reference is None:
                        reference = aligned
                    elif aligned != reference:
                        raise DiagnosticsError(f"{ticker} {year}: candidate dates or labels differ")
                    fold = metrics[key]
                    training = detail["training"]
                    validation = detail["validation"]
                    actual_count = sum(r["actual"] for r in rows)
                    if (
                        len(rows) != validation["rows"]
                        or len(rows) != fold["validation_rows"]
                        or actual_count != validation["positive"]
                        or actual_count != fold["validation_positive_count"]
                        or training["rows"] != fold["training_rows"]
                        or training["positive"] != fold["training_positive_count"]
                        or training["positive"] + training["negative"] != training["rows"]
                        or validation["positive"] + validation["negative"] != validation["rows"]
                        or rows[0]["session_date"].isoformat() != validation["first_session"]
                        or rows[-1]["session_date"].isoformat() != validation["last_session"]
                    ):
                        raise DiagnosticsError(f"{key}: prediction counts differ from saved fold")
                    if model == "always_positive" and any(
                        r["probability_positive"] != 1.0 for r in rows
                    ):
                        raise DiagnosticsError(f"{key}: always-positive baseline changed")
                    if model == "training_prevalence" and any(
                        not math.isclose(
                            r["probability_positive"],
                            training["positive"] / training["rows"],
                            rel_tol=0,
                            abs_tol=1e-10,
                        )
                        for r in rows
                    ):
                        raise DiagnosticsError(f"{key}: prevalence baseline changed")
                    scores = _scores(
                        np.asarray([r["actual"] for r in rows], dtype=int),
                        np.asarray([r["probability_positive"] for r in rows], dtype=float),
                    )
                    for name, value in scores.items():
                        if not math.isclose(value, fold[name], rel_tol=0, abs_tol=1e-10):
                            raise DiagnosticsError(f"{key}: saved {name} differs from predictions")
                    if not math.isclose(
                        actual_count / len(rows),
                        fold["validation_positive_rate"],
                        rel_tol=0,
                        abs_tol=1e-10,
                    ):
                        raise DiagnosticsError(
                            f"{key}: saved positive rate differs from predictions"
                        )
                    calculated[key] = (rows, scores)
        leaderboard = _rows(root / "leaderboard.parquet", LEADERBOARD_SCHEMA)
        by_leader = {(r["ticker"], r["model"]): r for r in leaderboard}
        if len(leaderboard) != len(tickers) * len(MODELS) or set(by_leader) != {
            (ticker, model) for ticker in tickers for model in MODELS
        }:
            raise DiagnosticsError("Leaderboard coverage differs from evaluation configuration")
        for (ticker, model), row in by_leader.items():
            if row["fold_count"] != len(years):
                raise DiagnosticsError(f"{ticker} {model}: leaderboard fold count differs")
            for metric in METRICS:
                values = [calculated[ticker, year, model][1][metric] for year in years]
                for suffix, value in (
                    ("mean", statistics.fmean(values)),
                    ("std", statistics.pstdev(values)),
                ):
                    if not math.isclose(row[f"{metric}_{suffix}"], value, rel_tol=0, abs_tol=1e-10):
                        raise DiagnosticsError(f"{ticker} {model}: leaderboard {metric} differs")
        return {
            "root": root,
            "manifest": manifest,
            "hash": digest,
            "tickers": tickers,
            "years": years,
            "details": details,
            "metrics": metrics,
            "calculated": calculated,
        }
    except DiagnosticsError:
        raise
    except (
        TrackingError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        ZeroDivisionError,
        duckdb.Error,
    ) as exc:
        raise DiagnosticsError(f"Invalid or altered evaluation {run_id}: {exc}") from exc


def _validated_backtest(data_dir: Path, run_id: str, evaluation: dict) -> dict:
    try:
        root, manifest, digest, rows = _backtest(
            data_dir, run_id, evaluation["manifest"]["run_id"], evaluation["hash"]
        )
        if manifest["evaluation_inputs"]["predictions"] != manifest_input_hash(
            evaluation, "predictions.parquet"
        ):
            raise DiagnosticsError("Backtest prediction hash differs from evaluation")
        config = manifest["configuration"]
        expected_folds = {
            (ticker, year) for ticker in evaluation["tickers"] for year in evaluation["years"]
        }
        windows = {(r["ticker"], r["year"]) for r in manifest["windows"]}
        if (
            windows != expected_folds
            or len(manifest["windows"]) != len(windows)
            or config["tickers"] != evaluation["tickers"]
            or manifest["strategies"] != list(STRATEGIES)
            or config["probability_threshold"] != 0.5
        ):
            raise DiagnosticsError("Backtest coverage or strategy differs from evaluation")
        for name in ("fee_bps", "slippage_bps"):
            if not math.isfinite(config[name]) or config[name] < 0:
                raise DiagnosticsError(f"Backtest has invalid {name}")
        for row in rows:
            if (
                any(
                    row[name] is None or not math.isfinite(row[name])
                    for name in (
                        "net_cumulative_return",
                        "excess_vs_buy_and_hold",
                        "exposure",
                        "total_turnover",
                    )
                )
                or not 0 <= row["exposure"] <= 1
            ):
                raise DiagnosticsError("Backtest has invalid trading context")
        return {"root": root, "manifest": manifest, "hash": digest, "rows": rows}
    except DiagnosticsError:
        raise
    except (TrackingError, OSError, KeyError, TypeError, ValueError, duckdb.Error) as exc:
        raise DiagnosticsError(f"Invalid or altered backtest {run_id}: {exc}") from exc


def manifest_input_hash(source: dict, name: str) -> str:
    return source["manifest"]["outputs"][name]["sha256"]


def _calculate(
    evaluation: dict, backtest: dict | None
) -> tuple[list[tuple], list[tuple], list[tuple]]:
    fold_rows: list[tuple] = []
    bin_rows: list[tuple] = []
    rate_by_key = {}
    for ticker in evaluation["tickers"]:
        for year in evaluation["years"]:
            for model in MODELS:
                key = (ticker, year, model)
                predictions, scores = evaluation["calculated"][key]
                saved = evaluation["metrics"][key]
                actual = [r["actual"] for r in predictions]
                predicted = [r["predicted"] for r in predictions]
                probabilities = [r["probability_positive"] for r in predictions]
                n = len(predictions)
                positive = sum(actual)
                positive_predictions = sum(predicted)
                rate_by_key[key] = positive_predictions / n
                tp = sum(a == 1 and p == 1 for a, p in zip(actual, predicted, strict=True))
                fp = sum(a == 0 and p == 1 for a, p in zip(actual, predicted, strict=True))
                tn = sum(a == 0 and p == 0 for a, p in zip(actual, predicted, strict=True))
                fn = sum(a == 1 and p == 0 for a, p in zip(actual, predicted, strict=True))
                values = {
                    "ticker": ticker,
                    "fold_year": year,
                    "model": model,
                    "training_rows": saved["training_rows"],
                    "training_positive_count": saved["training_positive_count"],
                    "training_positive_rate": saved["training_positive_count"]
                    / saved["training_rows"],
                    "validation_rows": n,
                    "actual_positive_count": positive,
                    "actual_positive_rate": positive / n,
                    "predicted_positive_count": positive_predictions,
                    "predicted_positive_rate": positive_predictions / n,
                    "true_positive": tp,
                    "false_positive": fp,
                    "true_negative": tn,
                    "false_negative": fn,
                    "probability_min": min(probabilities),
                    "probability_median": statistics.median(probabilities),
                    "probability_mean": statistics.fmean(probabilities),
                    "probability_max": max(probabilities),
                    "probability_std": statistics.pstdev(probabilities),
                    "probability_below_0_5_fraction": sum(p < 0.5 for p in probabilities) / n,
                    "probability_equal_0_5_fraction": sum(p == 0.5 for p in probabilities) / n,
                    "probability_above_0_5_fraction": sum(p > 0.5 for p in probabilities) / n,
                    **scores,
                    **{f"saved_{metric}": saved[metric] for metric in METRICS},
                }
                for baseline in BASELINES:
                    base = evaluation["calculated"][ticker, year, baseline][1]
                    values[f"accuracy_delta_vs_{baseline}"] = scores["accuracy"] - base["accuracy"]
                    values[f"auc_delta_vs_{baseline}"] = scores["roc_auc"] - base["roc_auc"]
                    values[f"brier_improvement_vs_{baseline}"] = (
                        base["brier_score"] - scores["brier_score"]
                    )
                fold_rows.append(tuple(values[name] for name, _ in FOLD_SCHEMA))
                bins: list[list[tuple[float, int]]] = [[] for _ in range(10)]
                for probability, label in zip(probabilities, actual, strict=True):
                    bins[min(int(probability * 10), 9)].append((probability, label))
                for index, contents in enumerate(bins):
                    count = len(contents)
                    bin_rows.append(
                        (
                            ticker,
                            year,
                            model,
                            index,
                            index / 10,
                            (index + 1) / 10,
                            count,
                            statistics.fmean(p for p, _ in contents) if count else None,
                            sum(label for _, label in contents) / count if count else None,
                            0 < count < 20,
                        )
                    )
    trading_rows: list[tuple] = []
    if backtest:
        config = backtest["manifest"]["configuration"]
        by_key = {(r["ticker"], r["fold_year"], r["strategy"]): r for r in backtest["rows"]}
        for ticker in evaluation["tickers"]:
            for year in evaluation["years"]:
                for strategy in STRATEGIES:
                    row = by_key[ticker, year, strategy]
                    trading_rows.append(
                        (
                            ticker,
                            year,
                            strategy,
                            row["net_cumulative_return"],
                            row["excess_vs_buy_and_hold"],
                            row["exposure"],
                            row["total_turnover"],
                            config["fee_bps"],
                            config["slippage_bps"],
                            rate_by_key.get((ticker, year, strategy)),
                        )
                    )
    return fold_rows, bin_rows, trading_rows


def _dict_rows(schema: list[tuple[str, str]], rows: list[tuple]) -> list[dict]:
    names = [name for name, _ in schema]
    return [dict(zip(names, row, strict=True)) for row in rows]


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _percentage_points(value: float) -> str:
    return f"{value * 100:+.1f} pp"


def _number(value: float | None, places: int = 3) -> str:
    return "—" if value is None else f"{value:.{places}f}"


def _signal_exposure_relation(rate: float | None, exposure: float) -> str:
    if rate is None:
        return "—"
    difference = abs(rate - exposure)
    if difference < 1e-12:
        return "same"
    if difference <= 0.05:
        return f"within 5 pp (Δ {difference * 100:.1f})"
    return f"differs by {difference * 100:.1f} pp"


def _chart(bins: list[dict], kind: str, title: str) -> str:
    width, height = 570, 190
    left, top, plot_w, plot_h = 44, 12, 500, 135
    escape = html.escape
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        f"<title>{escape(title)}</title>",
        '<line x1="44" y1="147" x2="544" y2="147" stroke="#64748b"/>',
        '<line x1="44" y1="12" x2="44" y2="147" stroke="#64748b"/>',
    ]
    if kind == "histogram":
        maximum = max((item["count"] for item in bins), default=1) or 1
        for item in bins:
            bar_h = item["count"] / maximum * plot_h
            x = left + item["bin_index"] * 50 + 5
            parts.append(
                f'<rect x="{x}" y="{top + plot_h - bar_h:.1f}" width="40" '
                f'height="{bar_h:.1f}" fill="#2563eb"/>'
            )
        ylabel = "Rows per bin"
    else:
        parts.append(
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top}" '
            'stroke="#64748b" stroke-dasharray="5 5"/>'
        )
        for item in bins:
            if item["count"]:
                x = left + item["mean_predicted_probability"] * plot_w
                y = top + (1 - item["observed_positive_fraction"]) * plot_h
                color = "#b45309" if item["sparse"] else "#0f766e"
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>')
        ylabel = "Observed positive rate"
    parts.extend(
        [
            f'<text x="8" y="20" font-size="11">{ylabel}</text>',
            '<text x="44" y="166" font-size="11">0</text>',
            '<text x="533" y="166" font-size="11">1</text>',
            '<text x="225" y="185" font-size="11">Predicted probability</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th scope='col'>{html.escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return (
        "<div class='table-wrap'><table><thead><tr>"
        f"{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _report(
    evaluation: dict,
    backtest: dict | None,
    folds: list[dict],
    bins: list[dict],
    trading: list[dict],
) -> str:
    fold_map = {(r["ticker"], r["fold_year"], r["model"]): r for r in folds}
    bins_map: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in bins:
        bins_map[row["ticker"], row["fold_year"], row["model"]].append(row)
    trading_map = {(r["ticker"], r["fold_year"], r["strategy"]): r for r in trading}
    sections = []
    for ticker in evaluation["tickers"]:
        for year in evaluation["years"]:
            summary = []
            cards = []
            for model in MODELS:
                row = fold_map[ticker, year, model]
                group_bins = bins_map[ticker, year, model]
                summary.append(
                    [
                        model,
                        str(row["validation_rows"]),
                        _percent(row["actual_positive_rate"]),
                        _percent(row["predicted_positive_rate"]),
                        _percent(row["accuracy"]),
                        _number(row["roc_auc"]),
                        _number(row["brier_score"]),
                    ]
                )
                deltas = []
                for baseline in BASELINES:
                    deltas.append(
                        [
                            baseline,
                            _percentage_points(row[f"accuracy_delta_vs_{baseline}"]),
                            _number(row[f"auc_delta_vs_{baseline}"]),
                            _number(row[f"brier_improvement_vs_{baseline}"]),
                        ]
                    )
                bin_table = [
                    [
                        f"{b['bin_lower']:.1f}–{b['bin_upper']:.1f}"
                        + (" (includes 1.0)" if b["bin_index"] == 9 else ""),
                        str(b["count"]),
                        _percent(b["mean_predicted_probability"]),
                        _percent(b["observed_positive_fraction"]),
                        "Yes" if b["sparse"] else "No",
                    ]
                    for b in group_bins
                ]
                metrics = [
                    [name, _number(row[name]), _number(row[f"saved_{name}"])] for name in METRICS
                ]
                note = (
                    "Constant-score baseline: ROC-AUC 0.5 does not show discrimination."
                    if model in BASELINES
                    else ""
                )
                cards.append(
                    f"<article class='card'><h4>{html.escape(model)}</h4>"
                    f"<p>{html.escape(note)}</p>"
                    f"<p><strong>Training:</strong> {row['training_rows']} rows, "
                    f"{row['training_positive_count']} positive "
                    f"({_percent(row['training_positive_rate'])}). "
                    f"<strong>Validation:</strong> {row['validation_rows']} rows, "
                    f"{row['actual_positive_count']} actually positive and "
                    f"{row['predicted_positive_count']} predicted positive.</p>"
                    f"<p><strong>Confusion counts at 0.5:</strong> TP {row['true_positive']}, "
                    f"FP {row['false_positive']}, TN {row['true_negative']}, "
                    f"FN {row['false_negative']}.</p>"
                    "<p><strong>Probability summary:</strong> "
                    f"min {_number(row['probability_min'])}, "
                    f"median {_number(row['probability_median'])}, "
                    f"mean {_number(row['probability_mean'])}, "
                    f"max {_number(row['probability_max'])}, SD {_number(row['probability_std'])}. "
                    "Below / equal / above 0.5: "
                    f"{_percent(row['probability_below_0_5_fraction'])} / "
                    f"{_percent(row['probability_equal_0_5_fraction'])} / "
                    f"{_percent(row['probability_above_0_5_fraction'])}.</p>"
                    "<h5>Predictive metrics: recomputed and saved</h5>"
                    + _table(["Metric", "Recomputed", "Saved"], metrics)
                    + "<h5>Difference versus baselines</h5>"
                    + _table(
                        ["Baseline", "Accuracy delta (pp)", "ROC-AUC delta", "Brier improvement"],
                        deltas,
                    )
                    + "<div class='charts'>"
                    + _chart(
                        group_bins, "histogram", f"{ticker} {year} {model} probability histogram"
                    )
                    + _chart(
                        group_bins, "reliability", f"{ticker} {year} {model} reliability diagram"
                    )
                    + "</div><h5>Probability bins</h5>"
                    + _table(
                        ["Bin", "Rows", "Mean predicted", "Observed positive", "Sparse (<20)"],
                        bin_table,
                    )
                    + "</article>"
                )
            trading_section = ""
            if backtest:
                trade_rows = []
                for strategy in STRATEGIES:
                    row = trading_map[ticker, year, strategy]
                    rate = row["predicted_positive_rate"]
                    relation = _signal_exposure_relation(rate, row["exposure"])
                    trade_rows.append(
                        [
                            strategy,
                            _percent(row["net_cumulative_return"]),
                            _percent(row["excess_vs_buy_and_hold"]),
                            _percent(row["exposure"]),
                            _number(row["total_turnover"]),
                            _percent(rate),
                            relation,
                        ]
                    )
                config = backtest["manifest"]["configuration"]
                trading_section = (
                    "<section class='trading'><h4>Trading context (separate backtest)</h4>"
                    f"<p>Fee {config['fee_bps']} bps and slippage {config['slippage_bps']} bps "
                    "per one-way position change. Buy and hold and all cash are shown below. "
                    "Predicted-positive rate can resemble fraction invested, but next-session "
                    "execution and fold boundaries mean they need not match. Trading returns "
                    "are not predictive accuracy.</p>"
                    + _table(
                        [
                            "Strategy",
                            "Net return",
                            "Vs buy and hold",
                            "Fraction invested",
                            "Turnover",
                            "Predicted positive",
                            "Signal / exposure",
                        ],
                        trade_rows,
                    )
                    + "</section>"
                )
            sections.append(
                f"<section class='fold'><h2>{html.escape(ticker)} · {year}</h2>"
                "<p>One chronological validation fold. All candidates use the same dates "
                "and outcomes.</p>"
                + _table(
                    [
                        "Candidate",
                        "Rows",
                        "Actual positive",
                        "Predicted positive",
                        "Accuracy",
                        "ROC-AUC",
                        "Brier",
                    ],
                    summary,
                )
                + "".join(cards)
                + trading_section
                + "</section>"
            )
    evaluation_id = html.escape(evaluation["manifest"]["run_id"])
    backtest_id = html.escape(backtest["manifest"]["run_id"]) if backtest else "None"
    return (
        """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MarketSignal model diagnostics</title><style>
body{font:16px/1.55 system-ui,sans-serif;background:#f4f7fb;color:#172033;margin:0}
main{max-width:1150px;margin:auto;padding:28px 20px 70px}
h1,h2,h3,h4,h5{line-height:1.2}h1{margin-bottom:4px}h2{margin-top:38px}
.intro,.fold{background:white;border:1px solid #d8e0ea;border-radius:12px;
padding:22px;margin:20px 0}
.card{border-top:1px solid #d8e0ea;padding:18px 0}
.card h4{font-size:1.2rem;margin-bottom:4px}
.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;
margin:12px 0 24px;font-size:.88rem}
th,td{border-bottom:1px solid #d8e0ea;padding:8px 10px;text-align:left;
white-space:nowrap}th{background:#eaf0f7}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}
svg{width:100%;background:#f8fafc;border:1px solid #d8e0ea;border-radius:7px}
.trading{background:#eff6ff;border-left:4px solid #2563eb;padding:14px;margin-top:20px}
.note{background:#fff7ed;border-left:4px solid #b45309;padding:12px 16px}
</style></head><body><main><header><h1>MarketSignal model diagnostics</h1>
<p>Saved out-of-sample validation predictions, explained one ticker and year at a time.
</p></header>
<section class="intro"><h2>How to read this report</h2>
<p><strong>Positive</strong> means the adjusted close was higher five exchange sessions
after the signal date. A probability at or above <strong>0.5</strong> becomes a positive
classification. These predictions are evaluated on later, held-out yearly folds.</p>
<p>In each confusion count, TP means a correctly predicted rise, FP a predicted rise
that did not happen, TN a correctly predicted non-rise, and FN a missed rise.</p>
<p><strong>Always positive</strong> assigns probability 1 to every row.
<strong>Training prevalence</strong> assigns the training set's positive fraction to every
row. They are simple reference points, not trained market forecasts. A positive accuracy
or ROC-AUC delta means the model improved on that baseline; a positive Brier improvement
means lower probability error. Accuracy differences are percentage points (pp).</p>
<p>For example, if 60 of 100 days are actually positive, an all-positive model gets 60
correct and 40 wrong without distinguishing the days. Its accuracy can be useful as a
benchmark, but it cannot identify which individual days will rise. A constant score's
ROC-AUC of 0.5 shows no discrimination.</p>
<p>Reliability diagrams compare mean predicted probability (horizontal) with observed
positive fraction (vertical). The diagonal is ideal agreement. Orange points have fewer
than 20 rows and are sparse. Empty bins have no observed rate. Brier score measures
overall probability error and does not by itself establish calibration.</p>
<p class="note">Five-session outcomes overlap across nearby rows, so rows are not
independent. Historical adjusted prices may reflect later provider corrections. A few
hundred rows can make calibration bins noisy. Using these years to design changes makes
them part of the research process; assess changes on a later untouched period. This is
not a trading recommendation or proof of future returns.</p>
<p><strong>Source evaluation:</strong> """
        + evaluation_id
        + """<br><strong>Source backtest:</strong> """
        + backtest_id
        + """</p></section>"""
        + "".join(sections)
        + "</main></body></html>\n"
    )


def _source_unchanged(source: dict) -> None:
    root, manifest = source["root"], source["manifest"]
    if _sha256(root / "manifest.json") != source["hash"]:
        raise DiagnosticsError(f"Source manifest changed during report creation: {root}")
    for name, entry in manifest["outputs"].items():
        if _sha256(root / name) != entry["sha256"]:
            raise DiagnosticsError(f"Source output changed during report creation: {root / name}")


def diagnose(data_dir: Path, evaluation_run_id: str, backtest_run_id: str | None = None) -> Path:
    """Validate saved results and atomically publish one local diagnostic report."""
    evaluation = _validated_evaluation(data_dir, evaluation_run_id)
    backtest = (
        _validated_backtest(data_dir, backtest_run_id, evaluation) if backtest_run_id else None
    )
    fold_rows, bin_rows, trading_rows = _calculate(evaluation, backtest)
    report = _report(
        evaluation,
        backtest,
        _dict_rows(FOLD_SCHEMA, fold_rows),
        _dict_rows(BIN_SCHEMA, bin_rows),
        _dict_rows(TRADING_SCHEMA, trading_rows),
    )
    root = data_dir / "diagnostics"
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC)
    run_id = timestamp.strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]
    final = root / run_id
    temporary = root / f".{run_id}.tmp"
    temporary.mkdir()
    try:
        _source_unchanged(evaluation)
        if backtest:
            _source_unchanged(backtest)
        _write_parquet(temporary / "fold_diagnostics.parquet", FOLD_SCHEMA, fold_rows)
        _write_parquet(temporary / "calibration_bins.parquet", BIN_SCHEMA, bin_rows)
        output_names = ["fold_diagnostics.parquet", "calibration_bins.parquet"]
        if backtest:
            _write_parquet(temporary / "trading_context.parquet", TRADING_SCHEMA, trading_rows)
            output_names.append("trading_context.parquet")
        (temporary / "report.html").write_text(report, encoding="utf-8")
        output_names.append("report.html")
        _source_unchanged(evaluation)
        if backtest:
            _source_unchanged(backtest)
        manifest = {
            "run_id": run_id,
            "generated_at_utc": timestamp.isoformat(),
            "version": VERSION,
            "calculation_version": VERSION,
            "report_version": VERSION,
            "evaluation_run_id": evaluation_run_id,
            "evaluation_manifest_sha256": evaluation["hash"],
            "evaluation_output_sha256": {
                name: entry["sha256"] for name, entry in evaluation["manifest"]["outputs"].items()
            },
            "backtest_run_id": backtest_run_id,
            "backtest_manifest_sha256": backtest["hash"] if backtest else None,
            "backtest_output_sha256": (
                {name: entry["sha256"] for name, entry in backtest["manifest"]["outputs"].items()}
                if backtest
                else None
            ),
            "threshold": 0.5,
            "bins": {
                "count": 10,
                "intervals": "[lower, upper), final bin includes 1.0",
                "sparse_below": 20,
            },
            "coverage": [
                {
                    "ticker": ticker,
                    "fold_year": year,
                    "models": list(MODELS),
                    "validation_rows": evaluation["details"][ticker, year]["validation"]["rows"],
                }
                for ticker in evaluation["tickers"]
                for year in evaluation["years"]
            ],
            "versions": {
                "python": platform.python_version(),
                "duckdb": duckdb.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "outputs": {
                name: {"path": name, "sha256": _sha256(temporary / name)} for name in output_names
            },
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.rename(final)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return final


def track_diagnostics(
    data_dir: Path,
    diagnostic_run_id: str,
    tracking_uri: str | None = None,
    artifact_root: Path | None = None,
) -> tuple[str, str, str]:
    """Import a verified local report into MLflow without changing the report."""
    try:
        root, manifest, digest = _run(data_dir, "diagnostics", diagnostic_run_id, SCHEMAS)
        outputs = manifest["outputs"]
        required = set(SCHEMAS) | {"report.html"}
        if (
            manifest["version"] != VERSION
            or manifest.get("calculation_version", manifest["version"]) != VERSION
            or manifest.get("report_version", manifest["version"]) != VERSION
            or not required <= outputs.keys()
        ):
            raise DiagnosticsError("Unsupported or incomplete diagnostic run")
        if bool(manifest["backtest_run_id"]) != ("trading_context.parquet" in outputs):
            raise DiagnosticsError("Diagnostic trading context differs from manifest")
        if "trading_context.parquet" in outputs:
            _run(
                data_dir,
                "diagnostics",
                diagnostic_run_id,
                {"trading_context.parquet": TRADING_SCHEMA},
            )
        evaluation = _run(
            data_dir, "evaluations", manifest["evaluation_run_id"], EVALUATION_SCHEMAS
        )
        if (
            evaluation[2] != manifest["evaluation_manifest_sha256"]
            or {name: entry["sha256"] for name, entry in evaluation[1]["outputs"].items()}
            != manifest["evaluation_output_sha256"]
        ):
            raise DiagnosticsError("Diagnostic evaluation source differs from manifest")
        if manifest["backtest_run_id"]:
            backtest = _run(data_dir, "backtests", manifest["backtest_run_id"], BACKTEST_SCHEMAS)
            if (
                backtest[2] != manifest["backtest_manifest_sha256"]
                or {name: entry["sha256"] for name, entry in backtest[1]["outputs"].items()}
                != manifest["backtest_output_sha256"]
            ):
                raise DiagnosticsError("Diagnostic backtest source differs from manifest")
        mlflow, client_class = _mlflow()
        uri, artifacts = _store(data_dir, tracking_uri, artifact_root)
        mlflow.set_tracking_uri(uri)
        client = client_class(tracking_uri=uri)
        experiment = _experiment(mlflow, "marketsignal-diagnostics", artifacts)
        existing = _finished(client, experiment, diagnostic_run_id, digest)
        if existing:
            return existing, uri, artifacts
        evaluation_mlflow_id = None
        evaluation_experiment = mlflow.get_experiment_by_name("marketsignal-evaluation")
        if evaluation_experiment:
            evaluation_mlflow_id = _finished(
                client,
                evaluation_experiment.experiment_id,
                manifest["evaluation_run_id"],
                manifest["evaluation_manifest_sha256"],
            )
        tags = {
            "source_run_id": diagnostic_run_id,
            "source_manifest_sha256": digest,
            "source_kind": "diagnostic",
            "evaluation_run_id": manifest["evaluation_run_id"],
            "evaluation_manifest_sha256": manifest["evaluation_manifest_sha256"],
        }
        if manifest["backtest_run_id"]:
            tags["backtest_run_id"] = manifest["backtest_run_id"]
            tags["backtest_manifest_sha256"] = manifest["backtest_manifest_sha256"]
        if evaluation_mlflow_id:
            tags["evaluation_mlflow_run_id"] = evaluation_mlflow_id
        with mlflow.start_run(
            experiment_id=experiment, run_name=diagnostic_run_id, tags=tags
        ) as run:
            for name in (
                "report.html",
                "manifest.json",
                "fold_diagnostics.parquet",
                "calibration_bins.parquet",
            ):
                mlflow.log_artifact(str(root / name))
            if "trading_context.parquet" in outputs:
                mlflow.log_artifact(str(root / "trading_context.parquet"))
            result = run.info.run_id
        if _finished(client, experiment, diagnostic_run_id, digest) != result:
            raise DiagnosticsError("Diagnostic MLflow import did not finish consistently")
        return result, uri, artifacts
    except DiagnosticsError:
        raise
    except TrackingError as exc:
        raise DiagnosticsError(str(exc)) from exc
    except Exception as exc:
        raise DiagnosticsError(
            f"Diagnostic MLflow import failed for {diagnostic_run_id}: {exc}"
        ) from exc
