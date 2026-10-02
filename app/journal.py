from __future__ import annotations

import json
from pathlib import Path
from typing import Any

JOURNAL_PATH = Path("data/journal.jsonl")


def record_setup(setup: dict[str, Any]) -> None:
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(setup, default=str) + "\n")


def read_setups(limit: int = 100) -> list[dict[str, Any]]:
    if not JOURNAL_PATH.exists():
        return []
    lines = JOURNAL_PATH.read_text(encoding="utf-8").splitlines()[-limit:]
    result = []
    for line in lines:
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return result
