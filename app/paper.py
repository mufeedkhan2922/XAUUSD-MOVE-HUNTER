from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json


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
    mfe_r: float = 0.0
    mae_r: float = 0.0
    current_price: float | None = None
    last_update: str | None = None


class PaperBook:
    """Persistent paper-trading ledger. It never submits broker orders."""

    def __init__(self, path: str = "data/paper_book.json") -> None:
        self.path = Path(path)
        self.positions: dict[str, PaperPosition] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.positions = {
                x["id"]: PaperPosition(**x) for x in raw.get("positions", [])
            }
        except (OSError, ValueError, TypeError, KeyError):
            self.positions = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"positions": [asdict(x) for x in self.positions.values()]}, indent=2),
            encoding="utf-8",
        )

    def open(self, setup: dict[str, Any]) -> dict[str, Any]:
        direction = str(setup.get("direction", "UNKNOWN"))
        if direction not in {"BULLISH", "BEARISH"}:
            raise ValueError("Paper position requires a directional setup.")
        if setup.get("entry") is None or setup.get("invalidation") is None:
            raise ValueError("Paper position requires entry and invalidation.")

        open_positions = [p for p in self.positions.values() if p.state == "OPEN"]
        if open_positions:
            raise ValueError("A paper position is already open.")

        pid = f"P{len(self.positions) + 1:06d}"
        now = datetime.now(timezone.utc).isoformat()
        position = PaperPosition(
            id=pid,
            direction=direction,
            entry=float(setup["entry"]),
            stop=float(setup["invalidation"]),
            targets=[float(x) for x in setup.get("targets", [])],
            opened_at=now,
            last_update=now,
        )
        self.positions[pid] = position
        self._save()
        return asdict(position)

    def mark(self, price: float) -> list[dict[str, Any]]:
        now = datetime.now(timezone.utc).isoformat()
        updates = []
        for position in self.positions.values():
            if position.state != "OPEN":
                continue

            price = float(price)
            position.current_price = price
            sign = 1 if position.direction == "BULLISH" else -1
            risk = abs(position.entry - position.stop)

            if risk > 0:
                excursion = (price - position.entry) * sign
                favourable = max(excursion, 0.0)
                adverse = max(-excursion, 0.0)
                position.mfe_r = max(position.mfe_r, favourable / risk)
                position.mae_r = max(position.mae_r, adverse / risk)

            hit_stop = price <= position.stop if position.direction == "BULLISH" else price >= position.stop
            hit_target = bool(position.targets) and (
                price >= position.targets[-1] if position.direction == "BULLISH"
                else price <= position.targets[-1]
            )

            if hit_stop:
                position.state = "STOPPED"
                position.realized_r = ((position.stop - position.entry) * sign / risk) if risk else 0.0
            elif hit_target:
                position.state = "TARGET_COMPLETE"
                position.realized_r = ((position.targets[-1] - position.entry) * sign / risk) if risk else 0.0

            position.last_update = now
            updates.append(asdict(position))

        self._save()
        return updates

    def summary(self) -> dict[str, Any]:
        positions = list(self.positions.values())
        closed = [p for p in positions if p.state != "OPEN"]
        realized = [p.realized_r for p in closed]
        return {
            "positions": len(positions),
            "open": sum(p.state == "OPEN" for p in positions),
            "closed": len(closed),
            "net_r": round(sum(realized), 3),
            "expectancy_r": round(sum(realized) / len(realized), 3) if realized else 0.0,
            "win_rate_pct": round(100 * sum(x > 0 for x in realized) / len(realized), 2) if realized else 0.0,
            "mfe_r": round(sum(p.mfe_r for p in positions) / len(positions), 3) if positions else 0.0,
            "mae_r": round(sum(p.mae_r for p in positions) / len(positions), 3) if positions else 0.0,
        }

    def snapshot(self) -> list[dict[str, Any]]:
        return [asdict(x) for x in self.positions.values()]
