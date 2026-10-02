from fastapi import FastAPI, HTTPException
from app.market_data import MarketDataError, get_market_snapshot
from app.signal_engine import analyze_market

app = FastAPI(
    title="XAUUSD MOVE HUNTER",
    version="0.1.0",
    description="Research-first XAUUSD 5M expansion detection engine.",
)

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "xauusd-move-hunter", "live_trading_enabled": False}

@app.get("/api/v1/market")
def market(candles: int = 500) -> dict:
    try:
        return get_market_snapshot(candles)
    except MarketDataError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

@app.get("/api/v1/signal")
def signal(candles: int = 500) -> dict:
    try:
        snapshot = get_market_snapshot(candles)
    except MarketDataError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return analyze_market(snapshot)
