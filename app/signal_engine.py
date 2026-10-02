from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

STATES = ("NO TRADE", "WATCH", "DEVELOPING", "VALID SETUP", "A+ EXPANSION SETUP")

def _atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    previous_close = frame["close"].shift(1)
    tr = pd.concat([
        frame["high"] - frame["low"],
        (frame["high"] - previous_close).abs(),
        (frame["low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def _resample(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        frame.set_index("datetime")
        .resample(rule)
        .agg(open=("open","first"), high=("high","max"), low=("low","min"), close=("close","last"))
        .dropna()
    )

def _context(frame: pd.DataFrame) -> dict[str, Any]:
    result = {}
    for rule, label in (("5min","5m"), ("15min","15m"), ("1h","1h")):
        bars = _resample(frame, rule)
        if len(bars) < 30:
            result[label] = {"bias":"UNKNOWN"}
            continue
        ema9 = bars.close.ewm(span=9, adjust=False).mean().iloc[-1]
        ema21 = bars.close.ewm(span=21, adjust=False).mean().iloc[-1]
        result[label] = {
            "bias": "BULLISH" if ema9 > ema21 else "BEARISH" if ema9 < ema21 else "NEUTRAL",
            "close": float(bars.close.iloc[-1]),
            "ema9": float(ema9),
            "ema21": float(ema21),
        }
    return result

def _liquidity(frame: pd.DataFrame) -> dict[str, float | None]:
    if len(frame) < 25:
        return {"recent_high":None,"recent_low":None,"sweep_high":None,"sweep_low":None}
    recent = frame.iloc[-21:-1]
    current = frame.iloc[-1]
    rh, rl = float(recent.high.max()), float(recent.low.min())
    return {
        "recent_high": rh,
        "recent_low": rl,
        "sweep_high": float(current.high) if current.high > rh and current.close < rh else None,
        "sweep_low": float(current.low) if current.low < rl and current.close > rl else None,
    }

def _structure(frame: pd.DataFrame) -> dict[str, Any]:
    if len(frame) < 20:
        return {"direction":"UNKNOWN","bos":False,"mss":False,"displacement":False}
    previous = frame.iloc[-6:-1]
    current = frame.iloc[-1]
    prior_high, prior_low = float(previous.high.max()), float(previous.low.min())
    bullish_bos, bearish_bos = float(current.close) > prior_high, float(current.close) < prior_low
    atr_series = _atr(frame)
    atr = float(atr_series.iloc[-1]) if not math.isnan(float(atr_series.iloc[-1])) else 0.0
    body = abs(float(current.close) - float(current.open))
    displacement = atr > 0 and body >= 1.2 * atr
    return {
        "direction": "BULLISH" if bullish_bos else "BEARISH" if bearish_bos else "NEUTRAL",
        "bos": bullish_bos or bearish_bos,
        "mss": (bullish_bos or bearish_bos) and displacement,
        "displacement": displacement,
        "body_atr_ratio": round(body / atr, 3) if atr else None,
    }

def _score(context: dict, liquidity: dict, structure: dict) -> tuple[int,list[str],str]:
    score, reasons = 0, []
    direction = structure.get("direction")
    b15, b1h = context.get("15m",{}).get("bias"), context.get("1h",{}).get("bias")
    if direction in ("BULLISH","BEARISH"):
        score += 20; reasons.append(f"{direction.lower()} structure break")
    if structure.get("mss"):
        score += 20; reasons.append("market-structure shift with displacement")
    if direction == "BULLISH" and liquidity.get("sweep_low") is not None:
        score += 25; reasons.append("sell-side liquidity sweep")
    if direction == "BEARISH" and liquidity.get("sweep_high") is not None:
        score += 25; reasons.append("buy-side liquidity sweep")
    if direction == b15:
        score += 15; reasons.append("5m direction aligns with 15m bias")
    if direction == b1h:
        score += 20; reasons.append("5m direction aligns with 1h bias")
    if direction == "NEUTRAL":
        return 0, reasons, "NO TRADE"
    state = "A+ EXPANSION SETUP" if score >= 85 else "VALID SETUP" if score >= 65 else "DEVELOPING" if score >= 40 else "WATCH"
    return min(score,100), reasons, state

def analyze_market(snapshot: dict) -> dict:
    frame = pd.DataFrame(snapshot.get("candles", []))
    if frame.empty:
        return {"state":"NO TRADE","score":0,"reason":"No candles available."}
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    for col in ("open","high","low","close"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["datetime","open","high","low","close"]).sort_values("datetime")
    context, liquidity, structure = _context(frame), _liquidity(frame), _structure(frame)
    score, reasons, state = _score(context, liquidity, structure)
    close = float(frame.close.iloc[-1])
    atr_series = _atr(frame)
    atr = float(atr_series.iloc[-1]) if not np.isnan(float(atr_series.iloc[-1])) else None
    direction = structure["direction"]
    entry = close if direction in ("BULLISH","BEARISH") else None
    invalidation, targets = None, []
    if entry is not None and atr:
        risk = max(0.8 * atr, 0.5)
        if direction == "BULLISH":
            invalidation, targets = entry-risk, [entry+risk, entry+2*risk, entry+3*risk]
        else:
            invalidation, targets = entry+risk, [entry-risk, entry-2*risk, entry-3*risk]
    return {
        "state":state, "score":score, "direction":direction, "price":close, "atr_5m":atr,
        "entry":entry, "invalidation":invalidation, "targets":targets,
        "move_potential":{"classification":"HIGH" if score>=85 else "MEDIUM" if score>=65 else "LOW",
                          "note":"Potential is a model estimate, not a guarantee."},
        "context":context, "liquidity":liquidity, "structure":structure,
        "reasons":reasons, "live_trading":False,
    }
