# Deriv Over/Under Bot

Binary options trading bot for Deriv synthetic indices using online machine learning.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Set your Deriv demo API token
export DERIV_API_TOKEN="your_demo_token_here"
# Get a free demo token from: https://app.deriv.com/account/api-token

# 3. Train the model on historical data
python -m models.warmup_trainer --symbol R_100 --samples 5000

# 4. Run the bot (paper mode — no real money)
python main.py --mode paper --symbol R_100

# 5. (Optional) Start the dashboard in another terminal
python -m dashboard.app
# Then open http://localhost:8050
```

## Architecture

```
Deriv WebSocket → Tick Aggregator → Feature Engine → Online Learner → Signal Generator → Risk Manager → Execution
```

## Strategy

- **Instrument:** Deriv Volatility 100 Index (R_100)
- **Contract:** Digit Over/Under binary options
- **Barrier:** Over 4 / Under 5
- **Model:** Online Logistic Regression (River)
- **Features:** 33 features from tick microstructure, digit distribution, volatility, and temporal patterns
- **Risk:** Quarter-Kelly sizing, 2% max per trade, circuit breaker after 5 consecutive losses

## Project Structure

```
deriv-bot/
├── config.py              # All settings and thresholds
├── main.py                # Entry point / orchestrator
├── data/
│   ├── deriv_ws.py        # Deriv WebSocket connection
│   ├── tick_aggregator.py # Tick buffering and rolling windows
│   └── feature_engine.py  # 33-feature computation
├── models/
│   ├── online_learner.py  # River online learning model
│   ├── drift_detector.py  # Concept drift detection
│   ├── model_persistence.py # Save/load model snapshots
│   └── warmup_trainer.py  # Offline batch training
├── trading/
│   ├── signal_generator.py # Model probability → trade decision
│   ├── risk_manager.py     # Position sizing and risk limits
│   ├── execution_engine.py # Paper/Live order execution
│   └── trade_logger.py     # Structured JSONL trade logging
├── dashboard/
│   └── app.py             # Live monitoring dashboard
└── utils/
    ├── logger.py          # Structured logging
    └── metrics.py         # Performance tracking (SQLite)
```

## Trading Modes

| Mode | Description | Risk |
|------|-------------|------|
| `paper` | Simulates trades, no real money | None |
| `live` | Places real orders on Deriv | REAL MONEY |

**Always start with paper mode.** Only switch to live after:
1. 500+ paper trades with positive P&L
2. Win rate > 55% sustained over 100+ trades
3. Maximum drawdown < 20%

## ⚠️ Disclaimer

- This is experimental software for educational purposes
- Deriv is not a regulated financial institution
- Binary options carry substantial risk of loss
- Past performance does not guarantee future results
- Never trade with money you cannot afford to lose
