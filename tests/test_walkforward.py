from datetime import datetime, timedelta, timezone

from app.walkforward import walk_forward


def make_candles(n=700):
    rows = []
    price = 100.0
    start = datetime(2026, 10, 2, tzinfo=timezone.utc)
    for i in range(n):
        open_price = price
        close = price + (0.04 if i % 2 == 0 else 0.02)
        high = close + 0.2
        low = open_price - 0.2
        rows.append({
            "datetime": (start + timedelta(minutes=5 * i)).isoformat().replace("+00:00", "Z"),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
        })
        price = close
    return rows


def test_walkforward_schema():
    result = walk_forward(make_candles(), train_bars=400, test_bars=150, step_bars=150)
    assert "summary" in result
    assert "windows" in result
    assert "research_quality" in result["summary"]
    assert "threshold_stability" in result["summary"]
