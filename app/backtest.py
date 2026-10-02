from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.market_features import add_indicators, compression_state, detect_fvg, detect_order_blocks


@dataclass
class Trade:
    direction: str
    signal_index: int
    entry_index: int
    entry: float
    stop: float
    target_1: float
    target_2: float
    target_3: float
    score: int
    reasons: list[str]
    exit_index: int | None = None
    exit_price: float | None = None
    outcome: str = "OPEN"
    mfe_r: float = 0.0
    mae_r: float = 0.0
    realized_r: float = 0.0


def _context(frame: pd.DataFrame) -> dict[str, str]:
    result: dict[str, str] = {}
    indexed = frame.set_index("datetime")
    for rule, label in (("15min", "15m"), ("1h", "1h")):
        bars = indexed.resample(rule).agg(
            open=("open", "first"), high=("high", "max"),
            low=("low", "min"), close=("close", "last")
        ).dropna()
        if len(bars) < 30:
            result[label] = "UNKNOWN"
            continue
        e9 = bars.close.ewm(span=9, adjust=False).mean().iloc[-1]
        e21 = bars.close.ewm(span=21, adjust=False).mean().iloc[-1]
        result[label] = "BULLISH" if e9 > e21 else "BEARISH" if e9 < e21 else "NEUTRAL"
    return result


def _signal_at(frame: pd.DataFrame, i: int) -> dict[str, Any] | None:
    if i < 60:
        return None
    history = add_indicators(frame.iloc[: i + 1].copy())
    current = history.iloc[-1]
    previous = history.iloc[-6:-1]
    atr = float(current.atr) if not np.isnan(float(current.atr)) else 0.0
    if atr <= 0:
        return None

    high, low = float(previous.high.max()), float(previous.low.min())
    bullish_bos = float(current.close) > high
    bearish_bos = float(current.close) < low
    direction = "BULLISH" if bullish_bos else "BEARISH" if bearish_bos else "NEUTRAL"
    if direction == "NEUTRAL":
        return None

    body = abs(float(current.close - current.open))
    displacement = body >= 1.2 * atr
    recent = history.iloc[-21:-1]
    sweep_low = float(current.low) < float(recent.low.min()) and float(current.close) > float(recent.low.min())
    sweep_high = float(current.high) > float(recent.high.max()) and float(current.close) < float(recent.high.max())

    context = _context(history)
    score = 15 + (20 if displacement else 0)
    reasons = ["structure break"]
    if displacement:
        reasons.append("displacement")
    if direction == "BULLISH" and sweep_low:
        score += 20
        reasons.append("sell-side sweep")
    if direction == "BEARISH" and sweep_high:
        score += 20
        reasons.append("buy-side sweep")
    if context["15m"] == direction:
        score += 10
        reasons.append("15m alignment")
    if context["1h"] == direction:
        score += 15
        reasons.append("1h alignment")

    fvgs = detect_fvg(history)
    blocks = detect_order_blocks(history)
    if any(x["type"] == direction and not x["filled"] for x in fvgs):
        score += 10
        reasons.append("unfilled FVG")
    if any(x["type"] == direction for x in blocks):
        score += 5
        reasons.append("order block")

    compression = compression_state(history)
    if compression["compressed"]:
        score += 5
        reasons.append("compression")

    expansion = float(current.high - current.low) >= 1.5 * atr and body >= 0.9 * atr
    if expansion:
        score += 10
        reasons.append("expansion")

    score = min(score, 100)
    if score < 65:
        return None

    entry = float(current.close)
    # Volatility-aware initial risk. The minimum prevents unrealistically tiny
    # stops on very quiet synthetic/illiquid data.
    risk = max(0.8 * atr, 0.5)
    stop = entry - risk if direction == "BULLISH" else entry + risk
    sign = 1 if direction == "BULLISH" else -1

    return {
        "direction": direction,
        "score": score,
        "entry": entry,
        "stop": stop,
        "targets": [entry + sign * risk, entry + sign * 2 * risk, entry + sign * 3 * risk],
        "reasons": reasons,
    }


def _simulate_trade(
    frame: pd.DataFrame,
    signal_index: int,
    setup: dict[str, Any],
    horizon: int,
    spread: float,
    slippage: float,
) -> Trade:
    direction = setup["direction"]
    sign = 1 if direction == "BULLISH" else -1
    friction = spread / 2.0 + slippage
    entry = float(setup["entry"]) + sign * friction
    stop = float(setup["stop"])
    targets = tuple(float(x) for x in setup["targets"])
    risk = abs(entry - stop)
    if risk <= 0:
        raise ValueError("Invalid zero-risk trade.")

    # Partial ladder: 50% at T1, 30% at T2, 20% at T3.
    weights = (0.50, 0.30, 0.20)
    hit = [False, False, False]
    remaining = 1.0
    active_stop = stop
    trade = Trade(
        direction, signal_index, signal_index, entry, stop,
        targets[0], targets[1], targets[2], int(setup["score"]), list(setup["reasons"])
    )
    end = min(len(frame), signal_index + 1 + horizon)

    def execute(trigger: float, fraction: float) -> float:
        exit_price = trigger - sign * friction
        return ((exit_price - entry) * sign / risk) * fraction

    for j in range(signal_index + 1, end):
        bar = frame.iloc[j]
        high, low = float(bar.high), float(bar.low)
        favourable = high - entry if direction == "BULLISH" else entry - low
        adverse = low - entry if direction == "BULLISH" else entry - high
        trade.mfe_r = max(trade.mfe_r, favourable / risk)
        trade.mae_r = min(trade.mae_r, adverse / risk)

        stop_hit = low <= active_stop if direction == "BULLISH" else high >= active_stop
        target_hits = [
            (not hit[k]) and (high >= targets[k] if direction == "BULLISH" else low <= targets[k])
            for k in range(3)
        ]

        # OHLC cannot reveal intrabar order, so stop-first is conservative.
        if stop_hit:
            trade.realized_r += execute(active_stop, remaining)
            trade.exit_index, trade.exit_price = j, active_stop - sign * friction
            trade.outcome = "LOSS" if trade.realized_r <= 0 else "PARTIAL_WIN"
            return trade

        for k, touched in enumerate(target_hits):
            if not touched:
                continue
            fraction = min(weights[k], remaining)
            trade.realized_r += execute(targets[k], fraction)
            hit[k] = True
            remaining -= fraction
            if k == 0:
                active_stop = entry
            if remaining <= 1e-9:
                trade.exit_index, trade.exit_price = j, targets[k] - sign * friction
                trade.outcome = "WIN_3R" if k == 2 else "WIN_2R" if k == 1 else "WIN_1R"
                return trade

    trade.exit_index = end - 1 if end > signal_index + 1 else signal_index
    close = float(frame.iloc[trade.exit_index].close)
    trade.exit_price = close - sign * friction
    trade.realized_r += execute(close, remaining)
    trade.outcome = "TIMEOUT_WIN" if trade.realized_r > 0 else "TIMEOUT_LOSS"
    return trade


def run_backtest(
    candles: list[dict[str, Any]],
    min_score: int = 65,
    horizon: int = 36,
    spread: float = 0.0,
    slippage: float = 0.0,
    allow_overlap: bool = False,
) -> dict[str, Any]:
    if spread < 0 or slippage < 0:
        raise ValueError("spread and slippage must be non-negative.")

    frame = pd.DataFrame(candles)
    if frame.empty:
        return {"error": "No candles supplied.", "trades": []}

    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    for col in ("open", "high", "low", "close"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(
        subset=["datetime", "open", "high", "low", "close"]
    ).sort_values("datetime").reset_index(drop=True)

    trades: list[Trade] = []
    cooldown_until = -1
    active_until = -1
    for i in range(60, len(frame) - 2):
        if i <= cooldown_until:
            continue
        if not allow_overlap and i <= active_until:
            continue
        setup = _signal_at(frame, i)
        if not setup or setup["score"] < min_score:
            continue
        trade = _simulate_trade(frame, i, setup, horizon, spread, slippage)
        trades.append(trade)
        cooldown_until = i + 3
        active_until = trade.exit_index if trade.exit_index is not None else i

    if not trades:
        return {
            "summary": {
                "trades": 0,
                "message": "No setups met the selected threshold.",
                "threshold": min_score,
                "spread": spread,
                "slippage": slippage,
                "allow_overlap": allow_overlap,
            },
            "trades": [],
        }

    # Aggregate realized P&L from the partial target ladder.\n    r_results: list[float] = [float(t.realized_r) for t in trades]\n\n    wins = [x for x in r_results if x > 0]
    losses = [x for x in r_results if x <= 0]
    curve = np.cumsum(r_results)
    running_max = np.maximum.accumulate(curve)
    drawdown = curve - running_max

    by_direction: dict[str, dict[str, float]] = {}
    for direction in ("BULLISH", "BEARISH"):
        vals = [r for r, t in zip(r_results, trades) if t.direction == direction]
        by_direction[direction] = {
            "trades": len(vals),
            "win_rate_pct": round(100 * sum(v > 0 for v in vals) / len(vals), 2) if vals else 0.0,
            "net_r": round(float(sum(vals)), 3),
        }

    return {
        "summary": {
            "trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round(100 * len(wins) / len(trades), 2),
            "net_r": round(float(sum(r_results)), 3),
            "expectancy_r": round(float(np.mean(r_results)), 3),
            "profit_factor": round(float(sum(wins) / abs(sum(losses))), 3) if losses else None,
            "max_drawdown_r": round(float(abs(drawdown.min())), 3),
            "average_mfe_r": round(float(np.mean([t.mfe_r for t in trades])), 3),
            "average_mae_r": round(float(np.mean([t.mae_r for t in trades])), 3),
            "threshold": min_score,
            "horizon_bars": horizon,
            "spread": spread,
            "slippage": slippage,
            "allow_overlap": allow_overlap,
            "by_direction": by_direction,
            "note": "Deterministic OHLC research backtest. Trigger levels are kept separate from execution prices; entry/exit friction is modeled explicitly. Targets use a 50/30/20% partial-exit ladder with breakeven protection after T1. Same-bar stop/target conflicts use stop-first ordering."
        },
        "trades": [
            {
                "direction": t.direction,
                "score": t.score,
                "reasons": t.reasons,
                "signal_index": t.signal_index,
                "entry": t.entry,
                "stop": t.stop,
                "targets": [t.target_1, t.target_2, t.target_3],
                "exit_index": t.exit_index,
                "exit_price": t.exit_price,
                "outcome": t.outcome,
                "mfe_r": round(t.mfe_r, 3),
                "mae_r": round(t.mae_r, 3),
                "realized_r": round(t.realized_r, 3),
            }
            for t in trades
        ],
    }    # Aggregate realized P&L from the partial target ladder.
    r_results: list[float] = [float(t.realized_r) for t in trades]

    wins = [x for x in r_results if x > 0]
    losses = [x for x in r_results if x <= 0]
    curve = np.cumsum(r_results)
    running_max = np.maximum.accumulate(curve)
    drawdown = curve - running_max

    by_direction: dict[str, dict[str, float]] = {}
    for direction in ("BULLISH", "BEARISH"):
        vals = [r for r, t in zip(r_results, trades) if t.direction == direction]
        by_direction[direction] = {
            "trades": len(vals),
            "win_rate_pct": round(100 * sum(v > 0 for v in vals) / len(vals), 2) if vals else 0.0,
            "net_r": round(float(sum(vals)), 3),
        }

    return {
        "summary": {
            "trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round(100 * len(wins) / len(trades), 2),
            "net_r": round(float(sum(r_results)), 3),
            "expectancy_r": round(float(np.mean(r_results)), 3),
            "profit_factor": round(float(sum(wins) / abs(sum(losses))), 3) if losses else None,
            "max_drawdown_r": round(float(abs(drawdown.min())), 3),
            "average_mfe_r": round(float(np.mean([t.mfe_r for t in trades])), 3),
            "average_mae_r": round(float(np.mean([t.mae_r for t in trades])), 3),
            "threshold": min_score,
            "horizon_bars": horizon,
            "spread": spread,
            "slippage": slippage,
            "allow_overlap": allow_overlap,
            "by_direction": by_direction,
            "note": "Deterministic OHLC research backtest. Trigger levels are kept separate from execution prices; entry/exit friction is modeled explicitly. Targets use a 50/30/20% partial-exit ladder with breakeven protection after T1. Same-bar stop/target conflicts use stop-first ordering."
        },
        "trades": [
            {
                "direction": t.direction,
                "score": t.score,
                "reasons": t.reasons,
                "signal_index": t.signal_index,
                "entry": t.entry,
                "stop": t.stop,
                "targets": [t.target_1, t.target_2, t.target_3],
                "exit_index": t.exit_index,
                "exit_price": t.exit_price,
                "outcome": t.outcome,
                "mfe_r": round(t.mfe_r, 3),
                "mae_r": round(t.mae_r, 3),
                "realized_r": round(t.realized_r, 3),
            }
            for t in trades
        ],
    }
