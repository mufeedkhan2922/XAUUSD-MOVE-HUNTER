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
    # Model execution friction without moving the trigger levels themselves.
    # Entry is filled adversely; exits are filled adversely for the trader too.
    # This is still an OHLC approximation, but it avoids the earlier mistake of
    # shifting entry, stop, and targets by the same amount.
    half_spread = spread / 2.0
    entry_cost = half_spread + slippage
    exit_cost = half_spread + slippage
    raw_entry = float(setup["entry"])
    raw_stop = float(setup["stop"])
    raw_t1, raw_t2, raw_t3 = [float(x) for x in setup["targets"]]
    entry = raw_entry + sign * entry_cost
    stop_trigger = raw_stop
    t1_trigger, t2_trigger, t3_trigger = raw_t1, raw_t2, raw_t3
    risk = abs(entry - stop_trigger)
    if risk <= 0:
        raise ValueError("Invalid zero-risk trade.")

    # Executed exit prices include adverse friction. The trigger remains the
    # strategy level used to decide whether the bar touched stop/target.
    stop_exit = stop_trigger - sign * exit_cost
    t1_exit = t1_trigger - sign * exit_cost
    t2_exit = t2_trigger - sign * exit_cost
    t3_exit = t3_trigger - sign * exit_cost
    if risk <= 0:
        raise ValueError("Invalid zero-risk trade.")

    trade = Trade(
        direction, signal_index, signal_index, entry, stop_trigger, t1_trigger, t2_trigger, t3_trigger,
        int(setup["score"]), list(setup["reasons"])
    )
    end = min(len(frame), signal_index + 1 + horizon)

    for j in range(signal_index + 1, end):
        bar = frame.iloc[j]
        favourable = (float(bar.high) - entry) * sign
        adverse = (float(bar.low) - entry) * sign
        trade.mfe_r = max(trade.mfe_r, favourable / risk)
        trade.mae_r = min(trade.mae_r, adverse / risk)

        stop_hit = float(bar.low) <= stop_trigger if direction == "BULLISH" else float(bar.high) >= stop_trigger
        t3_hit = float(bar.high) >= t3_trigger if direction == "BULLISH" else float(bar.low) <= t3_trigger
        t2_hit = float(bar.high) >= t2_trigger if direction == "BULLISH" else float(bar.low) <= t2_trigger
        t1_hit = float(bar.high) >= t1_trigger if direction == "BULLISH" else float(bar.low) <= t1_trigger

        # With OHLC alone the intrabar path is unknowable. Stop-first is the
        # conservative assumption when stop and target are both touched.
        if stop_hit:
            trade.exit_index, trade.exit_price, trade.outcome = j, stop_exit, "LOSS"
            return trade
        if t3_hit:
            trade.exit_index, trade.exit_price, trade.outcome = j, t3_exit, "WIN_3R"
            return trade
        if t2_hit:
            trade.exit_index, trade.exit_price, trade.outcome = j, t2_exit, "WIN_2R"
            return trade
        if t1_hit:
            trade.exit_index, trade.exit_price, trade.outcome = j, t1_exit, "WIN_1R"
            return trade

    trade.exit_index = end - 1 if end > signal_index + 1 else signal_index
    trade.exit_price = float(frame.iloc[trade.exit_index].close) - sign * exit_cost
    pnl_r = ((trade.exit_price - entry) * sign) / risk
    trade.outcome = "TIMEOUT_WIN" if pnl_r > 0 else "TIMEOUT_LOSS"
    return trade


def run_backtest(
    candles: list[dict[str, Any]],
    min_score: int = 65,
    horizon: int = 36,
    spread: float = 0.0,
    slippage: float = 0.0,
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
    for i in range(60, len(frame) - 2):
        if i <= cooldown_until:
            continue
        setup = _signal_at(frame, i)
        if not setup or setup["score"] < min_score:
            continue
        trade = _simulate_trade(frame, i, setup, horizon, spread, slippage)
        trades.append(trade)
        cooldown_until = i + 3

    if not trades:
        return {
            "summary": {
                "trades": 0,
                "message": "No setups met the selected threshold.",
                "threshold": min_score,
                "spread": spread,
                "slippage": slippage,
            },
            "trades": [],
        }

    r_results: list[float] = []
    for t in trades:
        if t.outcome == "LOSS":
            r_results.append(-1.0)
        elif t.outcome == "WIN_3R":
            r_results.append(3.0)
        elif t.outcome == "WIN_2R":
            r_results.append(2.0)
        elif t.outcome == "WIN_1R":
            r_results.append(1.0)
        else:
            pnl = (
                (t.exit_price - t.entry)
                * (1 if t.direction == "BULLISH" else -1)
                / abs(t.entry - t.stop)
            )
            r_results.append(float(pnl))

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
            "by_direction": by_direction,
            "note": "Deterministic OHLC research backtest. Trigger levels are kept separate from execution prices; entry/exit friction is modeled explicitly. Same-bar stop/target conflicts use stop-first ordering."
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
            }
            for t in trades
        ],
    }
