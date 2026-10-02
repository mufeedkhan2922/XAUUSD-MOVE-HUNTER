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
    indexed = frame.set_index("datetime")
    for rule, label in (("5min", "5m"), ("15min", "15m"), ("1h", "1h")):
        bars = indexed.resample(rule).agg(
            open=("open", "first"), high=("high", "max"),
            low=("low", "min"), close=("close", "last")
        ).dropna()
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
    if len(frame) < 35:
        return {"direction": "UNKNOWN", "bos": False, "mss": False,
                "choch": False, "displacement": False}

    current = frame.iloc[-1]
    previous = frame.iloc[-6:-1]
    prior_high, prior_low = float(previous.high.max()), float(previous.low.min())
    bullish_bos = float(current.close) > prior_high
    bearish_bos = float(current.close) < prior_low
    direction = "BULLISH" if bullish_bos else "BEARISH" if bearish_bos else "NEUTRAL"

    atr = float(current.atr) if not np.isnan(float(current.atr)) else 0.0
    body = abs(float(current.close - current.open))
    displacement = atr > 0 and body >= 1.2 * atr

    # A CHoCH is treated as a break against the short-term EMA regime;
    # MSS requires that break plus meaningful displacement.
    prior_ema_bias = "BULLISH" if float(frame.ema9.iloc[-2]) > float(frame.ema21.iloc[-2]) else "BEARISH"
    choch = direction in ("BULLISH", "BEARISH") and direction != prior_ema_bias
    mss = choch and displacement

    return {
        "direction": direction,
        "bos": bool(bullish_bos or bearish_bos),
        "choch": bool(choch),
        "mss": bool(mss),
        "displacement": bool(displacement),
        "body_atr_ratio": round(body / atr, 3) if atr else None,
    }


def _liquidity(frame: pd.DataFrame) -> dict[str, Any]:
    if len(frame) < 10:
        return {}

    current = frame.iloc[-1]
    recent = frame.iloc[-21:-1] if len(frame) >= 25 else frame.iloc[:-1]
    recent_high = float(recent.high.max())
    recent_low = float(recent.low.min())

    session = session_liquidity(frame)
    levels: dict[str, float] = {
        "recent_high": recent_high,
        "recent_low": recent_low,
    }
    for name, data in session.items():
        if data:
            levels[f"{name}_high"] = float(data["high"])
            levels[f"{name}_low"] = float(data["low"])

    # Equal highs/lows are approximate clusters within 0.15 ATR.
    atr = float(current.atr) if not np.isnan(float(current.atr)) else 0.0
    tolerance = max(0.15 * atr, 0.05)
    equal_high = equal_low = None
    if len(recent) >= 8:
        highs = recent.high.to_numpy(dtype=float)
        lows = recent.low.to_numpy(dtype=float)
        for value in highs[:-1]:
            if abs(value - highs[-1]) <= tolerance:
                equal_high = float((value + highs[-1]) / 2)
                break
        for value in lows[:-1]:
            if abs(value - lows[-1]) <= tolerance:
                equal_low = float((value + lows[-1]) / 2)
                break

    # Check the latest six completed candles for a sweep, not only the current bar.
    sweep_high = sweep_low = None
    sweep_age = None
    for age in range(0, min(6, len(frame) - 1)):
        bar = frame.iloc[-1 - age]
        before = frame.iloc[max(0, len(frame) - 21 - age):len(frame) - 1 - age]
        if before.empty:
            continue
        h, l = float(before.high.max()), float(before.low.min())
        if float(bar.high) > h and float(bar.close) < h:
            sweep_high, sweep_age = float(bar.high), age
            break
        if float(bar.low) < l and float(bar.close) > l:
            sweep_low, sweep_age = float(bar.low), age
            break

    # Explicit session/previous-day sweeps add context without forcing a trade.
    session_sweeps: list[str] = []
    for name, data in session.items():
        if not data:
            continue
        if float(current.high) > float(data["high"]) and float(current.close) < float(data["high"]):
            session_sweeps.append(f"{name}_high")
        if float(current.low) < float(data["low"]) and float(current.close) > float(data["low"]):
            session_sweeps.append(f"{name}_low")

    return {
        "levels": levels,
        "equal_high": equal_high,
        "equal_low": equal_low,
        "sweep_high": sweep_high,
        "sweep_low": sweep_low,
        "sweep_age_bars": sweep_age,
        "session_sweeps": session_sweeps,
        "session_levels": session,
    }


def _sequence(frame: pd.DataFrame, liquidity: dict[str, Any], structure: dict[str, Any],
              compression: dict[str, Any], expansion: dict[str, Any]) -> dict[str, Any]:
    direction = structure.get("direction")
    sweep = (
        direction == "BULLISH" and liquidity.get("sweep_low") is not None
    ) or (
        direction == "BEARISH" and liquidity.get("sweep_high") is not None
    )
    return {
        "compression": bool(compression.get("compressed")),
        "liquidity_event": bool(
            liquidity.get("equal_high") is not None or
            liquidity.get("equal_low") is not None or
            liquidity.get("sweep_high") is not None or
            liquidity.get("sweep_low") is not None
        ),
        "directional_sweep": bool(sweep),
        "displacement": bool(structure.get("displacement")),
        "expansion": bool(expansion.get("expanding")),
        "sequence_complete": bool(
            compression.get("compressed") and sweep and
            structure.get("displacement") and expansion.get("expanding")
        ),
    }


def _score(context: dict, liquidity: dict, structure: dict, compression: dict,
           expansion: dict, fvg: list, blocks: list, sequence: dict) -> tuple[int, list[str], str]:
    score = 0
    reasons: list[str] = []
    direction = structure.get("direction")
    b15 = context.get("15m", {}).get("bias")
    b1h = context.get("1h", {}).get("bias")

    if direction not in ("BULLISH", "BEARISH"):
        return 0, [], "NO TRADE"

    score += 12
    reasons.append(f"{direction.lower()} structure break")

    if structure.get("choch"):
        score += 10
        reasons.append("CHoCH against prior short-term regime")
    if structure.get("mss"):
        score += 18
        reasons.append("MSS confirmed by displacement")
    elif structure.get("displacement"):
        score += 12
        reasons.append("displacement")

    if direction == "BULLISH" and liquidity.get("sweep_low") is not None:
        score += 22
        reasons.append("sell-side liquidity sweep")
    if direction == "BEARISH" and liquidity.get("sweep_high") is not None:
        score += 22
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
        score += 8
        reasons.append("unfilled directional FVG")
    if matching_ob:
        score += 5
        reasons.append("directional order block")

    if compression.get("compressed"):
        score += 5
        reasons.append("compression before expansion")
    if expansion.get("expanding"):
        score += 8
        reasons.append("current candle expanding")

    if sequence.get("sequence_complete"):
        score += 10
        reasons.append("compression → sweep → displacement → expansion")

    # Reject weak breaks that have neither displacement nor a directional sweep.
    if not structure.get("displacement") and not (
        (direction == "BULLISH" and liquidity.get("sweep_low") is not None) or
        (direction == "BEARISH" and liquidity.get("sweep_high") is not None)
    ):
        score -= 18
        reasons.append("weak-break penalty")

    score = max(0, min(100, score))
    state = (
        "A+ EXPANSION SETUP" if score >= 85 and sequence["sequence_complete"] else
        "VALID SETUP" if score >= 68 else
        "DEVELOPING" if score >= 45 else
        "WATCH"
    )
    return score, reasons, state


def _move_potential(frame: pd.DataFrame, direction: str, score: int,
                    context: dict, liquidity: dict, sequence: dict,
                    atr: float | None) -> dict[str, Any]:
    if not atr or direction not in ("BULLISH", "BEARISH"):
        return {"classification": "LOW", "estimated_atr_multiple": 0.0, "drivers": []}

    sign = 1 if direction == "BULLISH" else -1
    price = float(frame.close.iloc[-1])
    candidates: list[float] = []
    for key in ("previous_day", "asia", "london", "new_york"):
        data = liquidity.get("session_levels", {}).get(key)
        if data:
            candidates.extend([float(data["high"]), float(data["low"])])
    directional = [x for x in candidates if (x - price) * sign > 0]
    nearest = min(directional, key=lambda x: abs(x - price)) if directional else None
    structure_room = abs(nearest - price) / atr if nearest is not None else 0.0

    htf = sum(context.get(x, {}).get("bias") == direction for x in ("15m", "1h"))
    estimate = 1.0 + score / 50.0 + 0.5 * htf
    if sequence["sequence_complete"]:
        estimate += 1.0
    if structure_room > 1.5:
        estimate += min(1.5, structure_room * 0.25)

    classification = "EXTREME" if estimate >= 5 else "HIGH" if estimate >= 3.5 else "MEDIUM" if estimate >= 2 else "LOW"
    return {
        "classification": classification,
        "estimated_atr_multiple": round(float(estimate), 2),
        "nearest_liquidity_distance_atr": round(float(structure_room), 2),
        "drivers": [
            "HTF alignment" if htf else "limited HTF alignment",
            "complete expansion sequence" if sequence["sequence_complete"] else "sequence incomplete",
            "room toward external liquidity" if structure_room > 1.5 else "limited nearby liquidity room",
        ],
        "note": "Model estimate only; it is not a guaranteed pip target.",
    }


def analyze_market(snapshot: dict[str, Any]) -> dict[str, Any]:
    frame = pd.DataFrame(snapshot.get("candles", []))
    if frame.empty:
        return {
            "state": "NO TRADE", "score": 0, "direction": "UNKNOWN",
            "price": None, "atr_5m": None, "entry": None,
            "invalidation": None, "targets": [],
            "move_potential": {"classification": "LOW", "estimated_atr_multiple": 0.0, "drivers": []},
            "sequence": {}, "context": {}, "liquidity": {}, "structure": {},
            "fair_value_gaps": [], "order_blocks": [],
            "reasons": ["No candles available."], "live_trading": False,
        }

    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    for col in ("open", "high", "low", "close"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["datetime", "open", "high", "low", "close"]).sort_values("datetime").reset_index(drop=True)
    frame = add_indicators(frame)

    context = _context(frame)
    liquidity = _liquidity(frame)
    structure = _structure(frame)
    compression = compression_state(frame)
    expansion = expansion_state(frame)
    fvgs = detect_fvg(frame)
    blocks = detect_order_blocks(frame)
    sequence = _sequence(frame, liquidity, structure, compression, expansion)

    score, reasons, state = _score(
        context, liquidity, structure, compression, expansion, fvgs, blocks, sequence
    )

    price = float(frame.close.iloc[-1])
    atr = float(frame.atr.iloc[-1]) if not np.isnan(float(frame.atr.iloc[-1])) else None
    direction = structure["direction"]

    # A live entry is allowed only after a meaningful break/shift. WATCH and
    # DEVELOPING states never produce a trade entry.
    entry = price if state in ("VALID SETUP", "A+ EXPANSION SETUP") else None
    invalidation = None
    targets: list[float] = []
    if entry is not None and atr:
        risk = max(0.8 * atr, 0.5)
        invalidation = entry - risk if direction == "BULLISH" else entry + risk
        sign = 1 if direction == "BULLISH" else -1
        targets = [entry + sign * risk, entry + sign * 2 * risk, entry + sign * 3 * risk]

    potential = _move_potential(frame, direction, score, context, liquidity, sequence, atr)

    return {
        "state": state,
        "score": score,
        "direction": direction,
        "price": price,
        "atr_5m": atr,
        "entry": entry,
        "invalidation": invalidation,
        "targets": targets,
        "move_potential": potential,
        "sequence": sequence,
        "context": context,
        "liquidity": liquidity,
        "structure": structure,
        "fair_value_gaps": fvgs[-10:],
        "order_blocks": blocks[-10:],
        "reasons": reasons,
        "live_trading": False,
    }
