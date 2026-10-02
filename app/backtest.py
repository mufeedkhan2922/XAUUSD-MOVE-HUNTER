from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.market_features import add_indicators, compression_state, detect_fvg, detect_order_blocks, session_liquidity


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
    exit_index: int | None = None
    exit_price: float | None = None
    outcome: str = "OPEN"
    mfe_r: float = 0.0
    mae_r: float = 0.0


def _context(frame: pd.DataFrame) -> dict[str, str]:
    result = {}
    for rule, label in (("15min", "15m"), ("1h", "1h")):
        bars = (
            frame.set_index("datetime")
            .resample(rule)
            .agg(open=("open","first"), high=("high","max"), low=("low","min"), close=("close","last"))
            .dropna()
        )
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
    history = frame.iloc[: i + 1].copy()
    history = add_indicators(history)
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


def _simulate_trade(frame: pd.DataFrame, signal_index: int, setup: dict[str, Any], horizon: int) -> Trade:
    direction = setup["direction"]
    entry = setup["entry"]
    stop = setup["stop"]
    t1, t2, t3 = setup["targets"]
    risk = abs(entry - stop)
    sign = 1 if direction == "BULLISH" else -1
    trade = Trade(direction, signal_index, signal_index, entry, stop, t1, t2, t3)
    end = min(len(frame), signal_index + 1 + horizon)

    for j in range(signal_index + 1, end):
        bar = frame.iloc[j]
        favourable = (float(bar.high) - entry) * sign
        adverse = (float(bar.low) - entry) * sign
        trade.mfe_r = max(trade.mfe_r, favourable / risk)
        trade.mae_r = min(trade.mae_r, adverse / risk)

        stop_hit = float(bar.low) <= stop if direction == "BULLISH" else float(bar.high) >= stop
        t3_hit = float(bar.high) >= t3 if direction == "BULLISH" else float(bar.low) <= t3
        t2_hit = float(bar.high) >= t2 if direction == "BULLISH" else float(bar.low) <= t2
        t1_hit = float(bar.high) >= t1 if direction == "BULLISH" else float(bar.low) <= t1

        # Conservative ordering when both stop and target are touched in one OHLC bar.
        if stop_hit:
            trade.exit_index, trade.exit_price, trade.outcome = j, stop, "LOSS"
            return trade
        if t3_hit:
            trade.exit_index, trade.exit_price, trade.outcome = j, t3, "WIN_3R"
            return trade
        if t2_hit:
            trade.exit_index, trade.exit_price, trade.outcome = j, t2, "WIN_2R"
            return trade
        if t1_hit:
            trade.exit_index, trade.exit_price, trade.outcome = j, t1, "WIN_1R"
            return trade

    trade.exit_index = end - 1 if end > signal_index + 1 else signal_index
    trade.exit_price = float(frame.iloc[trade.exit_index].close)
    pnl_r = ((trade.exit_price - entry) * sign) / risk
    trade.outcome = "TIMEOUT_WIN" if pnl_r > 0 else "TIMEOUT_LOSS"
    return trade


def run_backtest(candles: list[dict[str, Any]], min_score: int = 65, horizon: int = 36) -> dict[str, Any]:
    frame = pd.DataFrame(candles)
    if frame.empty:
        return {"error": "No candles supplied.", "trades": []}

    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    for col in ("open","high","low","close"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["datetime","open","high","low","close"]).sort_values("datetime").reset_index(drop=True)

    trades: list[Trade] = []
    cooldown_until = -1
    for i in range(60, len(frame) - 2):
        if i <= cooldown_until:
            continue
        setup = _signal_at(frame, i)
        if not setup or setup["score"] < min_score:
            continue
        trade = _simulate_trade(frame, i, setup, horizon)
        trades.append(trade)
        cooldown_until = i + 3

    if not trades:
        return {
            "summary": {"trades": 0, "message": "No setups met the selected threshold."},
            "trades": [],
        }

    r_results = []
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
            pnl = ((t.exit_price - t.entry) * (1 if t.direction == "BULLISH" else -1)) / abs(t.entry - t.stop)
            r_results.append(float(pnl))

    wins = [x for x in r_results if x > 0]
    losses = [x for x in r_results if x <= 0]
    curve = np.cumsum(r_results)
    running_max = np.maximum.accumulate(curve)
    drawdown = curve - running_max

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
            "note": "Backtest is deterministic and uses candle OHLC. Same-bar stop/target conflicts are resolved conservatively in favour of the stop.",
        },
        "trades": [
            {
                "direction": t.direction,
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
