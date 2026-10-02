from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from app.market_data import MarketDataError, get_market_snapshot
from app.signal_engine import analyze_market


class LiveMonitor:
    """Real-time signal lifecycle without live order execution."""

    STATES = {"NO TRADE", "WATCH", "DEVELOPING", "VALID SETUP", "A+ EXPANSION SETUP", "INVALIDATED"}

    def __init__(self) -> None:
        self.previous: dict[str, Any] | None = None
        self.current: dict[str, Any] | None = None

    def update(self, candles: int = 500) -> dict[str, Any]:
        snapshot = get_market_snapshot(candles)
        result = analyze_market(snapshot)
        previous_state = self.current.get("state") if self.current else None
        state = result.get("state", "NO TRADE")
        if previous_state in {"VALID SETUP", "A+ EXPANSION SETUP"} and state == "WATCH":
            result["state"] = "INVALIDATED"
        result["lifecycle"] = {
            "previous": previous_state,
            "current": result.get("state"),
            "changed": previous_state != result.get("state"),
        }
        self.previous = self.current
        self.current = result
        return result

    async def run(
        self,
        callback: Callable[[dict[str, Any]], Awaitable[None]],
        interval_seconds: int = 300,
        candles: int = 500,
    ) -> None:
        while True:
            try:
                result = self.update(candles)
                if result.get("lifecycle", {}).get("changed"):
                    await callback(result)
            except MarketDataError:
                pass
            await asyncio.sleep(max(30, interval_seconds))
