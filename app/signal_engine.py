from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from app.market_features import (
    add_indicators,
    compression_state,
    detect_fvg,
    detect_order_blocks,
    expansion_state,
    session_liquidity,
)


def _context(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for rule, label in (("5min", "5m"), ("15min", "15m"), ("1h", "1h")):
        bars = (
            frame.set_index("datetime")
            .resample(rule)
            .agg(open=("open","first"), high=("high","max"), low=("low","min"), close=("close","last"))
            .dropna()
        )
        if len(bars) < 30:
            result[label] = {"bias": "UNKNOWN"}
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


def _structure(frame: pd.DataFrame) -> dict[str, Any]:
    if len(frame) < 25:
        return {"direction":"UNKNOWN","bos":False,"mss":False,"displacement":False}

    previous = frame.iloc[-6:-1]
    current = frame.iloc[-1]
    prior_high, prior_low = float(previous.high.max()), float(previous.low.min())
    bullish_bos = float(current.close) > prior_high
    bearish_bos = float(current.close) < prior_low

    atr = float(frame.atr.iloc[-1]) if not np.isnan(float(frame.atr.iloc[-1])) else 0.0
    body = abs(float(current.close - current.open))
    displacement = atr > 0 and body >= 1.2 * atr

    return {
        "direction": "BULLISH" if bullish_bos else "BEARISH" if bearish_bos else "NEUTRAL",
        "bos": bool(bullish_bos or bearish_bos),
        "mss": bool((bullish_bos or bearish_bos) and displacement),
        "displacement": bool(displacement),
        "body_atr_ratio": round(body / atr, 3) if atr else None,
    }


def _liquidity(frame: pd.DataFrame) -> dict[str, Any]:
    recent = frame.iloc[-21:-1] if len(frame) >= 25 else frame.iloc[:-1]
    if recent.empty:
        return {}
    current = frame.iloc[-1]
    high, low = float(recent.high.max()), float(recent.low.min())
    return {
        "recent_high": high,
        "recent_low": low,
        "sweep_high": float(current.high) if current.high > high and current.close < high else None,
        "sweep_low": float(current.low) if current.low < low and current.close > low else None,
        "session_levels": session_liquidity(frame),
    }


def _score(context: dict, liquidity: dict, structure: dict, compression: dict, expansion: dict, fvg: list, blocks: list) -> tuple[int,list[str],str]:
    score = 0
    reasons: list[str] = []
    direction = structure.get("direction")
    b15 = context.get("15m", {}).get("bias")
    b1h = context.get("1h", {}).get("bias")

    if direction in ("BULLISH", "BEARISH"):
        score += 15
        reasons.append(f"{direction.lower()} structure break")
    if structure.get("mss"):
        score += 20
        reasons.append("displacement confirms structure shift")

    if direction == "BULLISH" and liquidity.get("sweep_low") is not None:
        score += 20
        reasons.append("sell-side liquidity sweep")
    if direction == "BEARISH" and liquidity.get("sweep_high") is not None:
        score += 20
        reasons.append("buy-side liquidity sweep")

    if direction == b15:
        score += 10
        reasons.append("15m bias alignment")
    if direction == b1h:
        score += 15
        reasons.append("1h bias alignment")

    matching_fvg = any(x["type"] == direction and not x["filled"] for x in fvg)
    matching_ob = any(x["type"] == direction for x in blocks)
    if matching_fvg:
        score += 10
        reasons.append("unfilled directional FVG")
    if matching_ob:
        score += 5
        reasons.append("directional order block")

    if compression.get("compressed"):
        score += 5
        reasons.append("recent volatility compression")
    if expansion.get("expanding"):
        score += 10
        reasons.append("current candle is expanding")

    if direction == "NEUTRAL":
        return 0, reasons, "NO TRADE"
    state = (
        "A+ EXPANSION SETUP" if score >= 85 else
        "VALID SETUP" if score >= 65 else
        "DEVELOPING" if score >= 40 else
        "WATCH"
    )
    return min(score, 100), reasons, state


def analyze_market(snapshot: dict[str, Any]) -> dict[str, Any]:
    frame = pd.DataFrame(snapshot.get("candles", []))
    if frame.empty:
        return {"state":"NO TRADE","score":0,"reason":"No candles available.","live_trading":False}

    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    for col in ("open","high","low","close"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["datetime","open","high","low","close"]).sort_values("datetime").reset_index(drop=True)
    frame = add_indicators(frame)

    context = _context(frame)
    liquidity = _liquidity(frame)
    structure = _structure(frame)
    compression = compression_state(frame)
    expansion = expansion_state(frame)
    fvgs = detect_fvg(frame)
    blocks = detect_order_blocks(frame)

    score, reasons, state = _score(context, liquidity, structure, compression, expansion, fvgs, blocks)

    price = float(frame.close.iloc[-1])
    atr = float(frame.atr.iloc[-1]) if not np.isnan(float(frame.atr.iloc[-1])) else None
    direction = structure["direction"]

    entry = price if direction in ("BULLISH", "BEARISH") and state in ("VALID SETUP","A+ EXPANSION SETUP") else None
    invalidation = None
    targets: list[float] = []

    if entry is not None and atr:
        risk = max(0.8 * atr, 0.5)
        if direction == "BULLISH":
            invalidation = entry - risk
            targets = [entry + risk, entry + 2*risk, entry + 3*risk]
        else:
            invalidation = entry + risk
            targets = [entry - risk, entry - 2*risk, entry - 3*risk]

    potential = "HIGH" if score >= 85 else "MEDIUM" if score >= 65 else "LOW"

    return {
        "state": state,
        "score": score,
        "direction": direction,
        "price": price,
        "atr_5m": atr,
        "entry": entry,
        "invalidation": invalidation,
        "targets": targets,
        "move_potential": {
            "classification": potential,
            "drivers": {
                "compression": compression,
                "expansion": expansion,
                "structure": structure,
            },
            "note": "Potential is a model estimate, not a guarantee.",
        },
        "context": context,
        "liquidity": liquidity,
        "fair_value_gaps": fvgs[-10:],
        "order_blocks": blocks[-10:],
        "reasons": reasons,
        "live_trading": False,
    }
