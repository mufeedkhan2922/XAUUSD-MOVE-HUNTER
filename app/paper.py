from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any


@dataclass
class PaperPosition:
    id: str
    direction: str
    entry: float
    stop: float
    targets: list[float]
    opened_at: str
    state: str = "OPEN"
    realized_r: float = 0.0


class PaperBook:
    def __init__(self) -> None:
        self.positions: dict[str, PaperPosition] = {}

    def open(self, setup: dict[str, Any]) -> dict[str, Any]:
        direction = str(setup.get("direction", "UNKNOWN"))
        if direction not in {"BULLISH", "BEARISH"}:
            raise ValueError("Paper position requires a directional setup.")
        if setup.get("entry") is None or setup.get("invalidation") is None:
            raise ValueError("Paper position requires entry and invalidation.")
        pid = f"P{len(self.positions) + 1:06d}"
        position = PaperPosition(
            id=pid, direction=direction, entry=float(setup["entry"]),
            stop=float(setup["invalidation"]),
            targets=[float(x) for x in setup.get("targets", [])],
            opened_at=datetime.now(timezone.utc).isoformat(),
        )
        self.positions[pid] = position
        return asdict(position)

    def mark(self, price: float) -> list[dict[str, Any]]:
        updates = []
        for position in self.positions.values():
            if position.state != "OPEN":
                continue
            hit_stop = price <= position.stop if position.direction == "BULLISH" else price >= position.stop
            if hit_stop:
                position.state = "STOPPED"
            elif position.targets and (
                price >= position.targets[-1] if position.direction == "BULLISH"
                else price <= position.targets[-1]
            ):
                position.state = "TARGET_COMPLETE"
            updates.append(asdict(position))
        return updates

    def snapshot(self) -> list[dict[str, Any]]:
        return [asdict(x) for x in self.positions.values()]
