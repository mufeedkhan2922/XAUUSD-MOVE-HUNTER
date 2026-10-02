from __future__ import annotations

from typing import Any

import numpy as np

from app.backtest import run_backtest
from app.walkforward import walk_forward


def regime_label(candles: list[dict[str, Any]]) -> str:
    if len(candles) < 60:
        return "UNKNOWN"
    closes = np.array([float(x["close"]) for x in candles[-60:]], dtype=float)
    returns = np.diff(closes) / np.maximum(closes[:-1], 1e-12)
    trend = float(np.mean(returns[-20:]))
    vol = float(np.std(returns[-20:]))
    if vol > np.percentile(np.abs(returns), 80) * 1.5:
        return "HIGH_VOLATILITY"
    if abs(trend) < max(vol * 0.25, 1e-8):
        return "RANGE"
    return "TREND_UP" if trend > 0 else "TREND_DOWN"


def adaptive_research(
    candles: list[dict[str, Any]],
    thresholds: list[int] | None = None,
    horizon: int = 36,
    spread: float = 0.0,
    slippage: float = 0.0,
) -> dict[str, Any]:
    thresholds = thresholds or [55, 60, 65, 70, 75, 80, 85]
    wf = walk_forward(
        candles, thresholds=thresholds, horizon=horizon,
        spread=spread, slippage=slippage,
    )
    windows = wf.get("windows", [])
    stability = wf.get("summary", {}).get("threshold_stability", {})
    return {
        "current_regime": regime_label(candles),
        "walk_forward": wf,
        "adaptive_policy": {
            "candidate_thresholds": thresholds,
            "selected_threshold_by_window": [w["selected_threshold"] for w in windows],
            "parameter_stability": stability,
            "rule": "Thresholds are selected only from historical training windows; no future test data is used for selection.",
        },
        "anti_overfit": {
            "out_of_sample": True,
            "train_test_overlap": False,
            "minimum_researchable_test_trades": 100,
            "sample_warning": "Small samples can make adaptive results unstable.",
        },
    }


def failed_setup_analysis(
    candles: list[dict[str, Any]],
    min_score: int = 65,
    horizon: int = 36,
) -> dict[str, Any]:
    result = run_backtest(candles, min_score=min_score, horizon=horizon)
    trades = result.get("trades", [])
    failures = [t for t in trades if float(t.get("realized_r", 0)) <= 0]
    reasons: dict[str, int] = {}
    for trade in failures:
        for reason in trade.get("reasons", []):
            reasons[str(reason)] = reasons.get(str(reason), 0) + 1
    return {
        "failed_trades": len(failures),
        "common_failure_features": sorted(
            ({"feature": k, "count": v} for k, v in reasons.items()),
            key=lambda x: (-x["count"], x["feature"]),
        )[:20],
        "note": "Failure correlation is descriptive; it is not causal proof.",
    }
