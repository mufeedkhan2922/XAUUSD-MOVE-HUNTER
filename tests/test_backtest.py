import pandas as pd

from app.backtest import run_backtest


def make_trend_candles(n=180):
    rows = []
    price = 100.0
    for i in range(n):
        open_price = price
        if i < 100:
            close = price + (0.02 if i % 2 == 0 else 0.01)
        else:
            close = price + 0.9
        high = max(open_price, close) + 0.15
        low = min(open_price, close) - 0.15
        rows.append({
            "datetime": f"2026-10-02T{i//12:02d}:{(i%12)*5:02d}:00Z",
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
        })
        price = close
    return rows


def test_backtest_schema():
    result = run_backtest(make_trend_candles(), min_score=65, horizon=20)
    assert "summary" in result
    assert "trades" in result
    assert "trades" in result["summary"]


def test_empty_backtest():
    result = run_backtest([])
    assert "error" in result
