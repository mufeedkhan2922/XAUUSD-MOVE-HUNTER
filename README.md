# XAUUSD MOVE HUNTER

A research-first XAUUSD 5-minute move-detection engine.

## Objective

Detect meaningful directional expansions using:

**Compression → Liquidity → Sweep → Displacement → Expansion**

The engine does not force a trade every candle. It produces:

- NO TRADE
- WATCH
- DEVELOPING
- VALID SETUP
- A+ EXPANSION SETUP

## Architecture

- 5M execution / signal timeframe
- 15M + 1H context
- Liquidity: recent/session highs & lows and sweep detection
- Structure: BOS / MSS
- Displacement and volatility expansion
- Dynamic invalidation and target ladder
- Research journal and backtesting planned for V2
- Live trading intentionally disabled in V1

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Open `/docs` for the API.

## Market data

V1 uses a Twelve Data adapter. Put the API key in `.env`; never commit credentials.

This is research software, not financial advice and not a guarantee of trading performance.
