from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from app.backtest import run_backtest


def _trades(result: dict[str, Any]) -> list[dict[str, Any]]:
    return list(result.get("trades", []))


def _group_stats(trades: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        value = t.get(key, "UNKNOWN")
        if isinstance(value, list):
            value = "|".join(map(str, value))
        groups[str(value)].append(float(t.get("realized_r", 0.0)))
    return {
        name: {
            "trades": len(values),
            "net_r": round(float(sum(values)), 3),
            "expectancy_r": round(float(np.mean(values)), 3) if values else 0.0,
            "win_rate_pct": round(100 * sum(v > 0 for v in values) / len(values), 2) if values else 0.0,
        }
        for name, values in sorted(groups.items())
    }


def research_report(
    candles: list[dict[str, Any]],
    min_score: int = 65,
    horizon: int = 36,
    spread: float = 0.0,
    slippage: float = 0.0,
) -> dict[str, Any]:
    """Produce a research report from the deterministic backtester.

    The report deliberately separates descriptive statistics from model
    selection. It never treats a high historical result as a guarantee.
    """
    result = run_backtest(
        candles, min_score=min_score, horizon=horizon,
        spread=spread, slippage=slippage,
    )
    trades = _trades(result)
    if not trades:
        return {
            "summary": result.get("summary", {}),
            "distributions": {},
            "session_statistics": {},
            "setup_statistics": {},
            "regime_statistics": {},
            "data_quality": {"candles": len(candles), "trades": 0},
        }

    frame = pd.DataFrame(candles)
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    frame["hour_utc"] = frame["datetime"].dt.hour
    frame["session"] = pd.cut(
        frame["hour_utc"],
        bins=[-1, 7, 12, 21, 23],
        labels=["ASIA", "LONDON", "NEW_YORK", "LATE"],
    )
    hour_map = frame["session"].to_dict()

    for t in trades:
        idx = int(t.get("signal_index", -1))
        t["session"] = str(hour_map.get(idx, "UNKNOWN"))
        reasons = t.get("reasons") or []
        t["setup_family"] = (
            "SWEEP_DISPLACEMENT" if any("sweep" in str(x).lower() for x in reasons)
            and "displacement" in " ".join(map(str, reasons)).lower()
            else "STRUCTURE_BREAK"
        )
        t["regime"] = (
            "HTF_ALIGNED" if sum("alignment" in str(x).lower() for x in reasons) >= 2
            else "PARTIAL_ALIGNMENT"
        )

    mfe = [float(t["mfe_r"]) for t in trades]
    mae = [float(t["mae_r"]) for t in trades]
    realized = [float(t["realized_r"]) for t in trades]

    return {
        "summary": result["summary"],
        "distributions": {
            "mfe_r": {
                "mean": round(float(np.mean(mfe)), 3),
                "median": round(float(np.median(mfe)), 3),
                "p75": round(float(np.percentile(mfe, 75)), 3),
                "p90": round(float(np.percentile(mfe, 90)), 3),
            },
            "mae_r": {
                "mean": round(float(np.mean(mae)), 3),
                "median": round(float(np.median(mae)), 3),
                "p75": round(float(np.percentile(mae, 75)), 3),
                "p90": round(float(np.percentile(mae, 90)), 3),
            },
            "realized_r": {
                "mean": round(float(np.mean(realized)), 3),
                "median": round(float(np.median(realized)), 3),
                "p10": round(float(np.percentile(realized, 10)), 3),
                "p90": round(float(np.percentile(realized, 90)), 3),
            },
        },
        "session_statistics": _group_stats(trades, "session"),
        "setup_statistics": _group_stats(trades, "setup_family"),
        "regime_statistics": _group_stats(trades, "regime"),
        "data_quality": {
            "candles": len(frame),
            "first_timestamp": str(frame.datetime.iloc[0]) if len(frame) else None,
            "last_timestamp": str(frame.datetime.iloc[-1]) if len(frame) else None,
            "trades": len(trades),
            "warning": "Descriptive backtest statistics; require out-of-sample validation before live use.",
        },
    }
