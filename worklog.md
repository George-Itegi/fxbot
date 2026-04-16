---
Task ID: 1
Agent: Super Z (Main Agent)
Task: Fix numpy.float64 .upper() error in APEX TRADER v4.1 parallel scan

Work Log:
- Analyzed entire fxbot codebase (~50+ Python files) to find source of `'numpy.float64' object has no attribute 'upper'` error
- Searched all .upper() calls — found 9 instances, all on `symbol` variable (always string from WATCHLIST)
- Searched all bias/direction field assignments in data layer (delta_calculator, volume_profile, vwap_calculator, tick_volume_surge, momentum_velocity) — all return proper strings
- Searched all strategy files (9 strategies) — all return proper string direction values
- Searched SMC modules (liquidity_sweeps, order_blocks, market_structure, htf_alignment) — all return proper strings
- Conclusion: The error is likely caused by numpy float leaking through pandas DataFrame operations into a field that eventually gets .upper() called on it
- Applied comprehensive defensive fixes:
  1. **main.py**: Added `import traceback` + full traceback in parallel scan error handler for exact debugging
  2. **main.py**: Added `str()` conversion for `direction` field from signal + `.upper()` sanitization
  3. **execution/order_manager.py**: `str(symbol).upper()` at both pip size functions
  4. **risk_management/risk_engine.py**: `str(symbol).upper()` at 3 locations
  5. **strategies/ema_trend.py**: `str(symbol).upper()`
  6. **strategies/strategy_engine.py**: Sanitize signal['direction'] with `str()` before appending
  7. **strategies/m1_momentum_scalp.py**: `str(m15_bias)` when assigning to direction
  8. **data_layer/tick_fetcher.py**: `str(symbol).upper()`
  9. **data_layer/momentum_velocity.py**: `str(symbol).upper()`
  10. **data_layer/market_scanner.py**: Explicit `str()` for all bias fields from dict.get() calls
  11. **data_layer/master_scanner.py**: Explicit `str()` for market_bias, smc_bias, pd_bias
  12. **data_layer/feature_store.py**: Complete rewrite of standardized dict with `str()`, `float()`, `int()`, `bool()` type casting for ALL fields

Stage Summary:
- All `.upper()` call sites now use `str(symbol).upper()` for safety
- All bias/direction fields are explicitly cast to `str()` when extracted from dicts
- Feature store now enforces type safety with explicit `str()`, `float()`, `int()`, `bool()` casting
- Strategy engine sanitizes signal['direction'] before use
- Main.py now logs full traceback for any parallel scan errors
- If error persists, full traceback will reveal exact location
