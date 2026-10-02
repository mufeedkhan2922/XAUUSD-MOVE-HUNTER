from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.backtest import run_backtest
from app.journal import read_setups, record_setup
from app.market_data import MarketDataError, get_market_snapshot
from app.signal_engine import analyze_market
from app.walkforward import walk_forward

app = FastAPI(
    title="XAUUSD MOVE HUNTER",
    version="0.3.0",
    description="Research-first XAUUSD 5M expansion detection engine.",
)


class BacktestRequest(BaseModel):
    candles: list[dict[str, Any]] = Field(default_factory=list, min_length=1)
    min_score: int = Field(default=65, ge=40, le=100)
    horizon: int = Field(default=36, ge=5, le=288)
    spread: float = Field(default=0.0, ge=0.0)
    slippage: float = Field(default=0.0, ge=0.0)


class WalkForwardRequest(BaseModel):
    candles: list[dict[str, Any]] = Field(default_factory=list, min_length=1)
    train_bars: int = Field(default=2000, ge=200, le=50000)
    test_bars: int = Field(default=500, ge=50, le=20000)
    step_bars: int = Field(default=500, ge=50, le=20000)
    thresholds: list[int] = Field(default_factory=lambda: [60, 65, 70, 75, 80])
    horizon: int = Field(default=36, ge=5, le=288)


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    path = Path(__file__).resolve().parent.parent / "dashboard" / "index.html"
    return path.read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "xauusd-move-hunter",
        "version": "0.3.0",
        "live_trading_enabled": False,
    }


@app.get("/api/v1/market")
def market(candles: int = 500) -> dict:
    try:
        return get_market_snapshot(candles)
    except MarketDataError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/v1/signal")
def signal(candles: int = 500, journal: bool = False) -> dict:
    try:
        snapshot = get_market_snapshot(candles)
    except MarketDataError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    result = analyze_market(snapshot)
    if journal:
        record_setup(result)
    return result


@app.post("/api/v1/backtest")
def backtest(request: BacktestRequest) -> dict:
    return run_backtest(
        request.candles,
        min_score=request.min_score,
        horizon=request.horizon,
        spread=request.spread,
        slippage=request.slippage,
    )


@app.post("/api/v1/walkforward")
def walkforward(request: WalkForwardRequest) -> dict:
    return walk_forward(
        request.candles,
        train_bars=request.train_bars,
        test_bars=request.test_bars,
        step_bars=request.step_bars,
        thresholds=request.thresholds,
        horizon=request.horizon,
    )


@app.get("/api/v1/journal")
def journal(limit: int = 100) -> dict:
    return {"setups": read_setups(max(1, min(limit, 500)))}
