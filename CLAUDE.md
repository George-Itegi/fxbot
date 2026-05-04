# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**APEX TRADER** is an institutional-grade forex trading bot (Python) that combines Smart Money Concepts (SMC), multi-strategy consensus voting, XGBoost ML gating, and live MT5 execution. Current version: **4.2**.

## Prerequisites & Setup

- Python 3.11+, MetaTrader 5 terminal (AlgoTrading enabled), MySQL 8.0+
- API keys in `.env`: `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER` (required); `ANTHROPIC_API_KEY`, `ALPHAVANTAGE_KEY`, `FINNHUB_KEY`, Telegram tokens (optional)

```bash
pip install -r requirements.txt
mysql -u root < setup_xampp_mysql.sql   # Initialize apex_trader schema
```

## Running the Bot

```bash
# Live trading (MT5 connection, 30s scan cycles)
python main.py

# Backtesting
python -m backtest.run                              # All symbols, full history
python -m backtest.run --symbols EURUSD GBPUSD --days 90
python -m backtest.run --relaxed --store-db --days 180  # Collect ML training data
python -m backtest.run --train                      # Train XGBoost from backtest data
python -m backtest.run --use-model --days 60        # Backtest with AI gate active
```

## Architecture & Data Flow

### 1. Data Layer (`data_layer/`)
`master_scanner.py` is the main entry point — it aggregates all data sources into a `master_report` dict:

```
master_scan(symbol) →
  ├─ scan_symbol()       → tick data, delta, volume profile, VWAP, momentum
  ├─ scan_smc()          → market structure, order blocks, FVGs, liquidity sweeps
  ├─ fractal_alignment() → multi-TF confirmation (H4→H1→M15→M5→M1)
  └─ external data       → COT reports, Fear/Greed, news sentiment, intermarket correlations

Returns: master_report = {
    "final_score": 0–100,
    "combined_bias": "BULLISH" | "BEARISH" | "NEUTRAL",
    "market_state": "TRENDING_STRONG" | "BREAKOUT_ACCEPTED" | "RANGING" | ...,
    "recommendation": {"action": "BUY" | "SELL" | "SKIP"},
    ...smc_report, market_report, fractal_alignment...
}
```

### 2. Strategy Engine (`strategies/`)
Ten strategies grouped by concept (SMC_STRUCTURE, TREND_FOLLOWING, MEAN_REVERSION, ORDER_FLOW, OSCILLATOR). All implement the same interface:

```python
def evaluate(symbol, df_m1, df_m5, df_m15, df_h1,
             smc_report, market_report, df_h4, master_report) -> dict | None:
    # Returns {"direction": "BUY"|"SELL", "score": 0–100,
    #          "entry_price": float, "sl_pips": int, "tp1_pips": int,
    #          "confluence_factors": int} or None
```

`strategy_engine.py` runs the consensus gate:
1. Each strategy must clear its `STRATEGY_MIN_SCORES` threshold (65–72)
2. **Hard requirement**: Order flow imbalance OR volume surge must confirm
3. **Consensus gate**: Minimum 2 different strategy groups must agree
4. Correlated strategies are penalized to prevent echo-chamber consensus

Adding a new strategy: create `strategies/new_strategy.py`, add to `STRATEGY_GROUPS` in `strategy_engine.py`, add to `strategy_registry.py`, add its threshold to `STRATEGY_MIN_SCORES`, then backtest.

### 3. AI/ML Gate (`ai_engine/ml_gate.py`)
XGBoost regressor with 63 features predicts expected R-multiple (not win/loss). Retrains every 50 closed live trades.

- R ≥ 0.5 → TAKE | 0.0–0.5 → CAUTION | < 0.0 → SKIP
- Falls back to rule-based gates when no trained model exists

### 4. Risk Management (`risk_management/`)
Checks run in order before any trade:
1. Daily loss limit (8% hard stop)
2. Max 5 concurrent positions
3. 30-minute per-symbol cooldown
4. Correlation check: max 2 pairs in same direction per currency
5. Minimum 2:1 R:R ratio
6. Consecutive loss protection: after 3 losses, halve all sizes for 30 min
7. Spread gate (per-symbol `MAX_SPREAD` limit)

**Dynamic position sizing** (when enabled): 0.5% risk (score 65–75), 1.0% (75–85), 1.5% (85+, 3+ groups agree).

### 5. Execution (`execution/`)
`order_manager.py` handles MT5 order placement, partial TP (close 50% at 1R, move SL to breakeven, ATR-trail remainder), and trade sync. `manage_positions()` runs every 1 second in a background thread.

**Pip size constants** (must be consistent across all strategies, order_manager, and backtester):
- Standard forex: `0.0001`
- JPY pairs (≤3 digits): `0.01`
- Gold (XAUUSD): `0.1`
- Silver (XAGUSD): `0.01`
- Indices (US30, JP225, etc.): `1.0`

### 6. Configuration (`config/settings.py`)
All tunable parameters live here — never hardcode values elsewhere. Key thresholds:
- `MIN_AI_SCORE`: 85 (only very high-quality signals)
- `MIN_CONFLUENCE_COUNT`: 6
- `MIN_STRATEGY_SCORE`: 70
- `WATCHLIST`: 17 instruments (7 majors, 5 JPY crosses, 3 crosses, 2 commodities)
- `TIMEFRAMES`: H4 (trend) → H1 (structure) → M30 → M15 (bias) → M5 (confirm) → M1 (entry)

### 7. Database (`database/`)
MySQL (`apex_trader` schema). Tables: `trades`, `signals`, `backtest_trades`. Connection pool (max 32). `backtest_trades` feeds ML training.

### 8. Backtesting (`backtest/`)
Replays the full signal→risk→execution pipeline on MT5 historical data. Stores results to `backtest_trades` for ML training.

## Critical Implementation Details

**Type safety (v4.2 fix)**: Always convert numpy types to Python before string operations:
```python
direction = str(signal.get('direction', '')).upper()  # NOT direct .upper() on numpy value
```

**Fractal alignment gate**: H4+H1+M15 full alignment is too strict for intraday — allowed if `score ≥ 55` AND bias is non-neutral, or if market state is `TRENDING_STRONG`/`BREAKOUT_ACCEPTED`.

**Order block entries**: Enter only at the *edge* of an unmitigated OB zone with reversal confirmation. Entering mid-OB is low probability.

**Correlation risk example**: Long EURUSD + long EURGBP = EUR net +2. A third EUR-long is blocked (prevents "6-EUR-pair disaster").

**Session windows (UTC)**:
```
SYDNEY            21:00–00:00  # Thin/volatile
TOKYO             00:00–07:00  # Accumulation/ranging
LONDON_OPEN       07:00–08:00  # Manipulation (false breakouts, stop hunts)
LONDON_SESSION    08:00–12:00  # Expansion (strong directional)
NY_LONDON_OVERLAP 12:00–16:00  # Peak liquidity (highest probability)
NY_AFTERNOON      16:00–21:00  # Liquidation/reversals
```

## Debugging

| Symptom | Check |
|---|---|
| No trades firing | `MIN_AI_SCORE` (85), `MIN_CONFLUENCE_COUNT` (6), market state gates |
| Strategy not triggering | `STRATEGY_MIN_SCORES` (70+), institutional gate (OF or volume surge required) |
| DB errors | MySQL running port 3306, `apex_trader` schema exists, pool ≤ 32 |
| Wrong pip values | Verify pip_size constants match across strategies + order_manager + backtester |
| Re-entry blocked | Check `ALLOW_REENTRY`, `REENTRY_COOLDOWN_MINUTES`, `REENTRY_MIN_SCORE_INCREASE` |

Log markers: `🔍 SCAN CYCLE`, `🧭 SYMBOL` (score/bias), `✅ TRADE EXECUTED`, `🚫` (risk gate blocked).
