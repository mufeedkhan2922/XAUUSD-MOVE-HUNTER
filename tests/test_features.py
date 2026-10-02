import pandas as pd

from app.market_features import compression_state, detect_fvg, detect_order_blocks


def sample_frame():
    rows = [
        ("2026-10-02T10:00:00Z", 100, 101, 99, 100.5),
        ("2026-10-02T10:05:00Z", 100.5, 101, 100, 100.2),
        ("2026-10-02T10:10:00Z", 100.2, 102, 100.8, 101.8),
        ("2026-10-02T10:15:00Z", 101.8, 103, 101.7, 102.8),
        ("2026-10-02T10:20:00Z", 102.8, 103.2, 102.7, 103.0),
    ]
    return pd.DataFrame(rows, columns=["datetime","open","high","low","close"]).assign(
        datetime=lambda x: pd.to_datetime(x.datetime, utc=True)
    )


def test_fvg_detection():
    gaps = detect_fvg(sample_frame())
    assert any(g["type"] == "BULLISH" for g in gaps)


def test_order_block_detection():
    frame = sample_frame()
    blocks = detect_order_blocks(frame)
    assert isinstance(blocks, list)


def test_compression_has_stable_schema():
    result = compression_state(sample_frame())
    assert "compressed" in result
