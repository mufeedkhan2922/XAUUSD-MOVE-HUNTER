from datetime import datetime, timedelta, timezone

from app.signal_engine import analyze_market


def make_candles(n=240):
    rows = []
    price = 2600.0
    start = datetime(2026, 10, 2, tzinfo=timezone.utc)
    for i in range(n):
        # Deterministic low-volatility stream with periodic directional candles.
        drift = 0.08 if i % 9 < 5 else -0.03
        if i > n - 8:
            drift = 2.0
        open_price = price
        close = price + drift
        high = max(open_price, close) + 0.8
        low = min(open_price, close) - 0.8
        rows.append({
            "datetime": (start + timedelta(minutes=5 * i)).isoformat().replace("+00:00", "Z"),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
        })
        price = close
    return rows


def test_signal_schema_contains_research_fields():
    result = analyze_market({"candles": make_candles()})
    assert result["live_trading"] is False
    assert "sequence" in result
    assert "structure" in result
    assert "move_potential" in result
    assert "liquidity" in result
    assert "classification" in result["move_potential"]


def test_empty_snapshot_is_safe():
    result = analyze_market({"candles": []})
    assert result["state"] == "NO TRADE"
    assert result["entry"] is None or "entry" not in result
