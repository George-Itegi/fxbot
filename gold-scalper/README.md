# GoldScalper v1.0 — Pure MQ5 Gold Scalper

Multi-regime gold scalping bot for MT5. No Python needed for live trading. Everything runs inside MQ5.

## Features

- **Regime Detection**: Hurst Exponent + ADX + ATR → classifies market as Trending, Mean-Reverting, Volatile, or Quiet
- **Order Flow Analysis**: Level 2 Depth of Market (DOM) → detects buy/sell imbalance
- **Spread Compression Signal**: When spread narrows below 50% of average, a big move is coming
- **Session Timing**: Only trades London/NY sessions (configurable)
- **ONNX ML**: Optional XGBoost model for confidence scoring (trained offline, runs in MQ5)
- **Virtual SL/TP**: Brokers can't see your stops (use with VPS for safety)
- **Physical SL Backup**: Safety net when running without VPS
- **Time Stops**: Auto-close if no profit after 30 seconds
- **Partial Close**: Close 70% at first TP, trail the rest
- **Trailing Stop**: Moves SL up as price moves in your favor

## Quick Start (5 minutes)

### 1. Copy files to MT5

Copy the entire `GoldScalper/` folder to your MT5 Experts directory:
```
<MT5 Data Folder>/MQL5/Experts/GoldScalper/
```

To find your MT5 data folder: Open MT5 → File → Open Data Folder

### 2. Compile

Open MetaEditor (F4 in MT5), open `GoldScalper.mq5`, press F7 to compile.

### 3. Attach to Chart

- Open XAUUSD M1 chart
- Drag GoldScalper from Navigator → Expert Advisors onto the chart
- In the EA properties:
  - ✅ Check "Allow Algo Trading"
  - ✅ Check "Allow DLL imports" (needed for ONNX)

### 4. Configure for Your Setup

**Without VPS (home PC):**
```
INP_VIRTUAL_SL   = false   ← Use physical SL (safety if MT5 disconnects)
INP_PHYSICAL_SL  = true    ← Broker holds your SL
INP_VIRTUAL_TP   = true    ← Virtual TP is safe (only profit, no risk)
```

**With VPS (24/5 uptime):**
```
INP_VIRTUAL_SL   = true    ← Maximum stealth (broker can't see SL)
INP_PHYSICAL_SL  = false   ← No physical SL needed
INP_VIRTUAL_TP   = true    ← Virtual TP
```

### 5. Start Trading

Click the "Algo Trading" button in MT5 toolbar (it should turn green). The bot will start analyzing and trading.

## Configuration Guide

### Risk Management
| Parameter | Default | Description |
|-----------|---------|-------------|
| INP_RISK_PCT | 1.0 | Risk % per trade (of account balance) |
| INP_MAX_SESSION_RISK | 3.0 | Max risk % in a single session |
| INP_MAX_TRADES_DAY | 30 | Maximum trades per day |
| INP_MAX_DRAWDOWN_PCT | 5.0 | Stop trading if daily drawdown exceeds this % |

### Session Timing (GMT)
| Parameter | Default | Description |
|-----------|---------|-------------|
| INP_LONDON_OPEN | 8 | London open hour (GMT) |
| INP_NY_OPEN | 13 | New York open hour (GMT) |
| INP_SESSION_BUFFER | 30 | Start looking N minutes before session |
| INP_ASIAN_SESSION | false | Trade Asian session? (usually choppy for gold) |

### Regime Detection
| Parameter | Default | Description |
|-----------|---------|-------------|
| INP_HURST_PERIOD | 100 | Lookback for Hurst exponent |
| INP_HURST_TREND | 0.6 | Hurst > this = trending |
| INP_HURST_MR | 0.4 | Hurst < this = mean-reverting |
| INP_ADX_TREND | 25 | ADX > this = trending |
| INP_ADX_QUIET | 15 | ADX < this = quiet (no trade) |

### Order Flow
| Parameter | Default | Description |
|-----------|---------|-------------|
| INP_OB_IMBALANCE_STRONG | 0.55 | Strong imbalance threshold |
| INP_OB_IMBALANCE_MODERATE | 0.40 | Moderate imbalance threshold |
| INP_DOM_LEVELS | 10 | DOM levels to analyze |
| INP_SPREAD_COMPRESS | 0.5 | Spread ratio < this = compression |

## ML Model (Optional but Recommended)

The bot works in **rule-based mode** by default. For better accuracy, train an ML model:

### Step 1: Export Training Data
1. Attach `ExportTickData.mq5` script to XAUUSD M1 chart
2. It exports `gold_features.csv` to `MQL5/Files/`
3. Copy the CSV file to the `ml/` directory

### Step 2: Train Model
```bash
cd ml/
pip install -r requirements.txt
python train_model.py
```

### Step 3: Deploy Model
1. Copy `GoldScalper.onnx` from `ml/` to `MQL5/Files/`
2. In GoldScalper config: `INP_USE_ML = true`
3. Re-attach EA to chart

## How It Works

### Trading Logic (per tick)

```
TICK ARRIVES
  │
  ├── Update regime (Hurst + ADX + ATR)
  ├── Update order flow (DOM imbalance)
  ├── Update spread monitoring
  │
  ├── Manage existing positions (virtual stops, trailing, time stops)
  │
  └── IF no position AND can trade:
        ├── Is regime tradeable? (NOT quiet)
        ├── Is session active? (London/NY)
        ├── Is spread acceptable? (< max)
        ├── Is there an order flow signal?
        ├── ML confidence > 65%? (if ML enabled)
        │
        └── ALL YES → OPEN TRADE
             Size: risk-based (X% of account)
             SL: 0.5× ATR (virtual or physical)
             TP: 1.0× ATR (virtual)
             Time stop: 30 seconds
             Partial close: 70% at +30 pips
```

### Regime Strategies

| Regime | SL | TP | Behavior |
|--------|----|----|----------|
| Trending | Normal 0.5× ATR | Normal 1.0× ATR | Momentum trades |
| Mean-Reverting | Normal | Tighter 0.6× ATR | Quick reversals |
| Volatile | Wider 1.5× ATR | Normal | Cautious entries |
| Quiet | N/A | N/A | **NO TRADING** |

## File Structure

```
MQL5/Experts/GoldScalper/
├── GoldScalper.mq5          ← Main EA (attach this to chart)
├── Config.mqh               ← All settings (change here)
├── RegimeDetector.mqh       ← Market regime classification
├── OrderFlowAnalyzer.mqh    ← Level 2 DOM analysis
├── SessionManager.mqh       ← Session timing
├── SpreadFilter.mqh         ← Spread monitoring
├── PositionSizer.mqh        ← Dynamic lot sizing
├── FeatureBuilder.mqh       ← ML feature vector builder
├── MLConfidence.mqh         ← ONNX model inference
└── TradeManager.mqh         ← Virtual stops + time stops + partial close

MQL5/Scripts/
└── ExportTickData.mq5       ← Data export for ML training

ml/
├── train_model.py           ← Train XGBoost → ONNX
└── requirements.txt         ← Python dependencies
```

## Broker Requirements

- ✅ MT5 platform
- ✅ XAUUSD symbol available
- ✅ Level 2 market depth (for order flow) — optional but recommended
- ✅ Low spread ECN account recommended (3-10 pips on gold)
- ⚠️ High spread accounts (30+ pips) will struggle — adjust INP_SPREAD_MAX

## Warnings

1. **TEST FIRST** on a demo account for at least 2 weeks before going live
2. **Without VPS**: Keep `INP_VIRTUAL_SL=false` and `INP_PHYSICAL_SL=true` for safety
3. **Gold is volatile**: Never risk more than 1-2% per trade
4. **Past performance ≠ future results**: This bot does not guarantee profits
5. **Brokers can change conditions**: Spread widening, slippage, and rejections can occur

## Performance Targets (Realistic)

| Metric | Conservative | Aggressive |
|--------|-------------|------------|
| Win rate | 55-62% | 60-68% |
| Avg win | 20-35 pips | 25-50 pips |
| Avg loss | 12-20 pips | 15-25 pips |
| Trades/day | 8-15 | 15-30 |
| Daily target | 0.5-1.5% | 1-3% |
