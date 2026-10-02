import os
import time
from datetime import datetime, timezone

import httpx
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

class MarketDataError(RuntimeError):
    pass

_CACHE: dict[tuple[str, int], tuple[float, dict]] = {}


def _api_key() -> str:
    key = os.getenv("TWELVE_DATA_API_KEY", "").strip()
    if not key:
        raise MarketDataError("TWELVE_DATA_API_KEY is not configured.")
    return key


def _parse_time_series(payload: dict) -> pd.DataFrame:
    values = payload.get("values")
    if not values:
        raise MarketDataError(str(payload.get("message") or "No candle data returned."))
    frame = pd.DataFrame(values)
    required = {"datetime", "open", "high", "low", "close"}
    missing = required.difference(frame.columns)
    if missing:
        raise MarketDataError(f"Provider response is missing columns: {sorted(missing)}")
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    for column in ("open", "high", "low", "close", "volume"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["datetime", "open", "high", "low", "close"]).sort_values("datetime").reset_index(drop=True)


def get_market_snapshot(candles: int = 500) -> dict:
    key = _api_key()
    symbol = os.getenv("XAU_SYMBOL", "XAU/USD")
    cache_seconds = max(0, int(os.getenv("CANDLE_CACHE_SECONDS", "15")))
    cache_key = (symbol, candles)

    cached = _CACHE.get(cache_key)
    if cached and time.time() - cached[0] < cache_seconds:
        return {**cached[1], "cache": "hit"}

    params = {
        "symbol": symbol,
        "interval": "5min",
        "outputsize": min(max(candles, 100), 5000),
        "apikey": key,
        "format": "JSON",
    }
    try:
        response = httpx.get("https://api.twelvedata.com/time_series", params=params, timeout=15)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise MarketDataError(f"Market-data request failed: {exc}") from exc

    frame = _parse_time_series(payload)
    snapshot = {
        "symbol": symbol,
        "timeframe": "5m",
        "provider": "twelvedata",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "cache": "miss",
        "candles": frame.to_dict(orient="records"),
    }
    _CACHE[cache_key] = (time.time(), snapshot)
    return snapshot
