# XAUUSD MOVE HUNTER

A research-first XAUUSD 5-minute move-detection and backtesting engine.

## Objective

Detect meaningful directional expansions using:

**Compression → Liquidity → Sweep → Displacement → Expansion**

The engine does not force a trade every candle. It produces:

- NO TRADE
- WATCH
- DEVELOPING
- VALID SETUP
- A+ EXPANSION SETUP

## Current architecture

- **5M** execution / signal timeframe
- **15M + 1H** context
- Liquidity sweeps and session/previous-day levels
- BOS / MSS structure
- ATR volatility and displacement
- FVG detection
- Order-block detection
- Compression and expansion detection
- Dynamic entry, invalidation and 1R/2R/3R targets
- Setup journal
- Deterministic OHLC backtester
- MFE / MAE
- Win rate, expectancy, profit factor and max drawdown in R
- GitHub Actions tests
- Live trading disabled

## API

Run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000`.

Endpoints:

- `GET /health`
- `GET /api/v1/market`
- `GET /api/v1/signal`
- `GET /api/v1/journal`
- `POST /api/v1/backtest`
- `GET /docs`

The backtest endpoint accepts candle OHLC data plus `min_score` and `horizon`. The simulator resolves same-bar stop/target conflicts conservatively in favour of the stop.

## Market data

V1 uses a Twelve Data adapter. Put the API key in `.env`; never commit credentials.

A short in-memory cache is used to reduce repeated provider requests.

## Research roadmap

**V0.3 — Backtest foundation**
- deterministic setup replay
- MFE/MAE
- expectancy
- profit factor
- drawdown
- setup threshold testing

**V0.4 — Robust research**
- walk-forward evaluation
- train/test separation
- regime buckets
- parameter sensitivity
- repeated-mistake analysis

**V0.5 — Live research dashboard**
- real-time setup lifecycle
- setup history
- paper-trade tracking
- performance dashboard
- alert layer

This project is research software, not financial advice and not a guarantee of trading performance.
