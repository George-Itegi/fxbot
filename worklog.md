---
Task ID: 1
Agent: Main Agent
Task: Implement v4.0 signal quality fixes and all requested features

Work Log:
- Diagnosed root cause of all trades losing: tick classification using hardcoded 0.000015 threshold
  This meant 90%+ ticks for JPY/Gold/Oil/Indices were classified as NEUTRAL, making all order flow data useless
- Fixed tick_fetcher.py: spread-proportional threshold + consecutive tick comparison (3 methods)
- Fixed pip calculation for JP225/DE30/UK100 across ALL files (order_manager, risk_engine, ema_trend, momentum_velocity)
- Created correlation_manager.py: currency exposure groups, 7 correlation groups, max exposure limits
- Created trend_continuation.py: H4+H1+M15 pullback strategy for London/NY expansion
- Upgraded order_manager.py: ATR-adaptive trailing stop, break-even move, dynamic TP extension
- Fixed m1_momentum_scalp.py: ATR-based SL (was fixed 5p), v2.0
- Updated main.py: bias cross-validation, re-entry logic, consecutive loss tracking
- Fixed score inflation in market_scanner.py and ema_trend.py: deduplication cap
- Added consecutive loss protection in risk_engine.py
- Registered TREND_CONTINUATION in strategy_registry and strategy_engine

Stage Summary:
- 13 files changed, 1262 insertions, 192 deletions
- 9 strategies now active
- All changes pushed to GitHub (commit aeedc06)
- Key fix: tick classification now works for ALL symbol types
- Key fix: pip calculation standardized across all modules
