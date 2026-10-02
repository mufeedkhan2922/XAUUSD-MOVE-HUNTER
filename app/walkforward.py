from __future__ import annotations

from typing import Any
import math
import pandas as pd

from app.backtest import run_backtest


def _slice(candles: list[dict[str, Any]], start: int, end: int) -> list[dict[str, Any]]:
    return candles[start:end]


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    return result.get("summary", {"trades": 0})


def _bootstrap_mean_ci(values: list[float], iterations: int = 1000) -> dict[str, float] | None:
    if len(values) < 10:
        return None
    # Deterministic pseudo-bootstrap so research reports are reproducible.
    import random
    rng = random.Random(20261002)
    means = []
    n = len(values)
    for _ in range(iterations):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return {
        "mean_r": round(sum(values) / n, 4),
        "ci95_low": round(means[int(0.025 * iterations)], 4),
        "ci95_high": round(means[int(0.975 * iterations) - 1], 4),
        "samples": n,
    }


def walk_forward(
    candles: list[dict[str, Any]],
    train_bars: int = 2000,
    test_bars: int = 500,
    step_bars: int = 500,
    thresholds: list[int] | None = None,
    horizon: int = 36,
    spread: float = 0.0,
    slippage: float = 0.0,
) -> dict[str, Any]:
    if thresholds is None:
        thresholds = [60, 65, 70, 75, 80]
    thresholds = sorted(set(int(x) for x in thresholds if 40 <= int(x) <= 100))
    if not thresholds:
        raise ValueError("thresholds must contain at least one value between 40 and 100")
    if train_bars < 200 or test_bars < 50 or step_bars < 50:
        raise ValueError("train_bars/test_bars/step_bars are too small for research")

    frame = pd.DataFrame(candles)
    if frame.empty:
        return {"windows": [], "summary": {"windows": 0, "error": "No candles supplied."}}
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    ordered = frame.to_dict(orient="records")

    windows = []
    cursor = 0
    while cursor + train_bars + test_bars <= len(ordered):
        train_end = cursor + train_bars
        test_end = train_end + test_bars
        train = _slice(ordered, cursor, train_end)
        test = _slice(ordered, train_end, test_end)

        train_results = {
            threshold: run_backtest(
                train, min_score=threshold, horizon=horizon,
                spread=spread, slippage=slippage
            )
            for threshold in thresholds
        }

        candidates = []
        for threshold, result in train_results.items():
            s = _summary(result)
            trades = int(s.get("trades", 0))
            expectancy = float(s.get("expectancy_r", 0.0))
            dd = float(s.get("max_drawdown_r", 0.0))
            # Require a meaningful sample when one exists; otherwise retain
            # the least-data candidate explicitly rather than silently inventing
            # a positive result.
            candidates.append((trades >= 3, expectancy, -dd, threshold, trades))

        viable = [x for x in candidates if x[0]]
        chosen = max(viable or candidates, key=lambda x: (x[1], x[2], x[3]))
        chosen_threshold = chosen[3]

        test_result = run_backtest(
            test, min_score=chosen_threshold, horizon=horizon,
            spread=spread, slippage=slippage
        )
        test_summary = _summary(test_result)
        test_trades = int(test_summary.get("trades", 0))
        test_r_values = []
        for trade in test_result.get("trades", []):
            outcome = trade.get("outcome")
            if outcome == "LOSS":
                test_r_values.append(-1.0)
            elif outcome == "WIN_1R":
                test_r_values.append(1.0)
            elif outcome == "WIN_2R":
                test_r_values.append(2.0)
            elif outcome == "WIN_3R":
                test_r_values.append(3.0)
            else:
                entry, exit_price, stop = trade.get("entry"), trade.get("exit_price"), trade.get("stop")
                if entry is not None and exit_price is not None and stop is not None:
                    risk = abs(float(entry) - float(stop))
                    sign = 1 if trade.get("direction") == "BULLISH" else -1
                    test_r_values.append(((float(exit_price) - float(entry)) * sign / risk) if risk else 0.0)

        windows.append({
            "train_start": str(train[0]["datetime"]),
            "train_end": str(train[-1]["datetime"]),
            "test_start": str(test[0]["datetime"]),
            "test_end": str(test[-1]["datetime"]),
            "selected_threshold": chosen_threshold,
            "training_candidates": [
                {"threshold": x[3], "trades": x[4], "expectancy_r": x[1], "drawdown_r": -x[2]}
                for x in candidates
            ],
            "train_summary": _summary(train_results[chosen_threshold]),
            "test_summary": test_summary,
            "test_return_distribution": {
                "mean_r": round(sum(test_r_values) / len(test_r_values), 4) if test_r_values else 0.0,
                "median_r": round(float(pd.Series(test_r_values).median()), 4) if test_r_values else 0.0,
                "ci95": _bootstrap_mean_ci(test_r_values),
            },
        })
        cursor += step_bars

    test_summaries = [w["test_summary"] for w in windows if w["test_summary"].get("trades", 0) > 0]
    total_trades = sum(int(s.get("trades", 0)) for s in test_summaries)
    total_net_r = sum(float(s.get("net_r", 0.0)) for s in test_summaries)
    positive_windows = sum(1 for s in test_summaries if float(s.get("expectancy_r", 0.0)) > 0)
    thresholds_used = [w["selected_threshold"] for w in windows]

    return {
        "summary": {
            "windows": len(windows),
            "test_windows_with_trades": len(test_summaries),
            "test_trades": total_trades,
            "test_net_r": round(total_net_r, 3),
            "positive_test_windows": positive_windows,
            "positive_test_window_pct": round(100 * positive_windows / len(test_summaries), 2) if test_summaries else 0.0,
            "threshold_stability": {
                "unique_thresholds": sorted(set(thresholds_used)),
                "most_common_threshold": max(set(thresholds_used), key=thresholds_used.count) if thresholds_used else None,
                "changes": sum(1 for a, b in zip(thresholds_used, thresholds_used[1:]) if a != b),
            },
            "research_quality": "INSUFFICIENT_SAMPLE" if total_trades < 30 else "PRELIMINARY" if total_trades < 100 else "RESEARCHABLE",
            "method": "rolling train/test walk-forward; thresholds selected only from training windows; no train/test candle overlap",
        },
        "windows": windows,
    }
