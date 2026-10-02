from __future__ import annotations

from typing import Any

import pandas as pd

from app.backtest import run_backtest


def _slice(candles: list[dict[str, Any]], start: int, end: int) -> list[dict[str, Any]]:
    return candles[start:end]


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    return result.get("summary", {"trades": 0})


def walk_forward(
    candles: list[dict[str, Any]],
    train_bars: int = 2000,
    test_bars: int = 500,
    step_bars: int = 500,
    thresholds: list[int] | None = None,
    horizon: int = 36,
) -> dict[str, Any]:
    if thresholds is None:
        thresholds = [60, 65, 70, 75, 80]

    frame = pd.DataFrame(candles)
    if frame.empty:
        return {"windows": [], "summary": {"windows": 0, "error": "No candles supplied."}}

    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    frame = frame.sort_values("datetime").reset_index(drop=True)
    ordered = frame.to_dict(orient="records")

    windows = []
    cursor = 0
    while cursor + train_bars + test_bars <= len(ordered):
        train = _slice(ordered, cursor, cursor + train_bars)
        test = _slice(ordered, cursor + train_bars, cursor + train_bars + test_bars)

        train_results = {
            threshold: run_backtest(train, min_score=threshold, horizon=horizon)
            for threshold in thresholds
        }

        # Select only from the training window. Ties are resolved by higher
        # expectancy, then lower drawdown, then higher threshold.
        candidates = []
        for threshold, result in train_results.items():
            s = _summary(result)
            trades = int(s.get("trades", 0))
            expectancy = float(s.get("expectancy_r", 0.0))
            dd = float(s.get("max_drawdown_r", 0.0))
            candidates.append((trades >= 3, expectancy, -dd, threshold))

        viable = [x for x in candidates if x[0]]
        chosen = max(viable or candidates, key=lambda x: (x[1], x[2], x[3]))
        chosen_threshold = chosen[3]

        test_result = run_backtest(test, min_score=chosen_threshold, horizon=horizon)
        windows.append({
            "train_start": str(train[0]["datetime"]),
            "train_end": str(train[-1]["datetime"]),
            "test_start": str(test[0]["datetime"]),
            "test_end": str(test[-1]["datetime"]),
            "selected_threshold": chosen_threshold,
            "train_summary": _summary(train_results[chosen_threshold]),
            "test_summary": _summary(test_result),
        })
        cursor += step_bars

    test_summaries = [w["test_summary"] for w in windows if w["test_summary"].get("trades", 0) > 0]
    total_trades = sum(int(s.get("trades", 0)) for s in test_summaries)
    total_net_r = sum(float(s.get("net_r", 0.0)) for s in test_summaries)
    positive_windows = sum(1 for s in test_summaries if float(s.get("expectancy_r", 0.0)) > 0)

    return {
        "summary": {
            "windows": len(windows),
            "test_windows_with_trades": len(test_summaries),
            "test_trades": total_trades,
            "test_net_r": round(total_net_r, 3),
            "positive_test_windows": positive_windows,
            "positive_test_window_pct": round(
                100 * positive_windows / len(test_summaries), 2
            ) if test_summaries else 0.0,
            "method": "rolling train/test walk-forward; threshold chosen only from each training window",
        },
        "windows": windows,
    }
