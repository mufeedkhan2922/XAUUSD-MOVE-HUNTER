import pandas as pd

from app.walkforward import walk_forward


def make_candles(n=700):
    rows = []
    price = 100.0
    for i in range(n):
        open_price = price
        close = price + (0.04 if i % 2 == 0 else 0.02)
        high = close + 0.2
        low = open_price - 0.2
        rows.append({
            "datetime": f"2026-10-02T{i//12:02d}:{(i%12)*5:02d}:00Z",
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
