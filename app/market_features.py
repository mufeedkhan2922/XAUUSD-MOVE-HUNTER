from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    previous = frame["close"].shift(1)
    tr = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous).abs(),
            (frame["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["atr"] = atr(out)
    out["ema9"] = out["close"].ewm(span=9, adjust=False).mean()
    out["ema21"] = out["close"].ewm(span=21, adjust=False).mean()
    out["body"] = (out["close"] - out["open"]).abs()
    out["range"] = out["high"] - out["low"]
    out["body_ratio"] = np.where(out["range"] > 0, out["body"] / out["range"], 0.0)
    return out


def detect_fvg(frame: pd.DataFrame, lookback: int = 30) -> list[dict[str, Any]]:
    if len(frame) < 3:
        return []
    start = max(2, len(frame) - lookback)
    gaps: list[dict[str, Any]] = []
    for i in range(start, len(frame)):
        a, b, c = frame.iloc[i - 2], frame.iloc[i - 1], frame.iloc[i]
        if float(c.low) > float(a.high):
            gaps.append({
                "type": "BULLISH",
                "low": float(a.high),
                "high": float(c.low),
                "index": i,
                "datetime": str(c.datetime),
                "filled": bool(float(frame.iloc[-1].low) <= float(a.high)),
            })
        elif float(c.high) < float(a.low):
            gaps.append({
                "type": "BEARISH",
                "low": float(c.high),
                "high": float(a.low),
                "index": i,
                "datetime": str(c.datetime),
                "filled": bool(float(frame.iloc[-1].high) >= float(a.low)),
            })
    return gaps


def detect_order_blocks(frame: pd.DataFrame, lookback: int = 40) -> list[dict[str, Any]]:
    if len(frame) < 5:
        return []
    out: list[dict[str, Any]] = []
    start = max(1, len(frame) - lookback)
    for i in range(start, len(frame) - 1):
        candle = frame.iloc[i]
        nxt = frame.iloc[i + 1]
        body = abs(float(candle.close) - float(candle.open))
        next_body = abs(float(nxt.close) - float(nxt.open))
        if body == 0:
            continue
        if float(candle.close) < float(candle.open) and float(nxt.close) > float(nxt.open) and next_body >= body * 1.5:
            out.append({
                "type": "BULLISH",
                "low": float(candle.low),
                "high": float(candle.high),
                "datetime": str(candle.datetime),
                "index": i,
            })
        elif float(candle.close) > float(candle.open) and float(nxt.close) < float(nxt.open) and next_body >= body * 1.5:
            out.append({
                "type": "BEARISH",
                "low": float(candle.low),
                "high": float(candle.high),
                "datetime": str(candle.datetime),
                "index": i,
            })
    return out


def session_liquidity(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    local = frame.copy()
    local["date"] = local["datetime"].dt.date.astype(str)
    local["hour"] = local["datetime"].dt.hour
    latest_date = local["date"].iloc[-1]
    day = local[local["date"] == latest_date]

    def window(start: int, end: int) -> dict[str, float] | None:
        part = day[(day["hour"] >= start) & (day["hour"] < end)]
        if part.empty:
            return None
        return {"high": float(part.high.max()), "low": float(part.low.min())}

    previous = local[local["date"] < latest_date]
    prev_day = None
    if not previous.empty:
        prev = previous[previous["date"] == previous["date"].iloc[-1]]
        prev_day = {"high": float(prev.high.max()), "low": float(prev.low.min())}

    return {
        "previous_day": prev_day,
        "asia": window(0, 8),
        "london": window(8, 13),
        "new_york": window(13, 22),
    }


def compression_state(frame: pd.DataFrame, period: int = 20) -> dict[str, Any]:
    if len(frame) < period + 5:
        return {"compressed": False, "range_ratio": None}
    ranges = (frame["high"] - frame["low"]).tail(period)
    current_avg = float(ranges.mean())
    prior = (frame["high"] - frame["low"]).iloc[-period * 2:-period]
    prior_avg = float(prior.mean()) if len(prior) else 0.0
    ratio = current_avg / prior_avg if prior_avg else None
    return {
        "compressed": bool(ratio is not None and ratio <= 0.72),
        "range_ratio": round(ratio, 3) if ratio is not None else None,
    }


def expansion_state(frame: pd.DataFrame) -> dict[str, Any]:
    if len(frame) < 25:
        return {"expanding": False, "range_atr": None, "body_atr": None}
    last = frame.iloc[-1]
    a = float(frame["atr"].iloc[-1]) if "atr" in frame and not np.isnan(float(frame["atr"].iloc[-1])) else 0.0
    if a <= 0:
        return {"expanding": False, "range_atr": None, "body_atr": None}
    candle_range = float(last.high - last.low)
    body = abs(float(last.close - last.open))
    return {
        "expanding": bool(candle_range >= 1.5 * a and body >= 0.9 * a),
        "range_atr": round(candle_range / a, 3),
        "body_atr": round(body / a, 3),
    }
