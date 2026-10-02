from __future__ import annotations

from typing import Any

from app.market_data import MarketDataError, get_market_snapshot
from app.paper import PaperBook
from app.signal_engine import analyze_market


class UnifiedEngine:
    """Single V2/V3/V4 orchestration layer.

    Research and adaptive analysis are separated from the live monitor, while
    the live path can only create paper positions.
    """

    def __init__(self, paper_book: PaperBook | None = None) -> None:
        self.paper = paper_book or PaperBook()
        self.previous_state: str | None = None

    def cycle(self, candles: int = 500, auto_paper: bool = False) -> dict[str, Any]:
        snapshot = get_market_snapshot(candles)
        signal = analyze_market(snapshot)

        previous = self.previous_state
        current = signal.get("state", "NO TRADE")
        invalidated = (
            previous in {"VALID SETUP", "A+ EXPANSION SETUP"}
            and current not in {"VALID SETUP", "A+ EXPANSION SETUP"}
        )
        if invalidated:
            signal["state"] = "INVALIDATED"
            current = "INVALIDATED"

        self.previous_state = current
        paper_action = None

        if auto_paper and signal.get("state") in {"VALID SETUP", "A+ EXPANSION SETUP"}:
            try:
                paper_action = self.paper.open(signal)
            except ValueError:
                paper_action = {"status": "not_opened", "reason": "paper position already open"}

        signal["lifecycle"] = {
            "previous": previous,
            "current": current,
            "changed": previous != current,
        }
        signal["paper_action"] = paper_action
        signal["paper_summary"] = self.paper.summary()
        return signal

    def mark_paper(self, price: float) -> dict[str, Any]:
        return {
            "updates": self.paper.mark(price),
            "summary": self.paper.summary(),
        }
