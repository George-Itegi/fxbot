# =============================================================
# backtest/run.py  v2.1
# CLI entry point: python -m backtest.run
# Upgraded: CLI arguments, parallel execution, ML training/testing
#
# v2.1 CHANGES:
#   --train          Train XGBoost model from backtest DB data
#   --use-model      Run backtest with trained model as additional gate
#   --clear-data     Clear all backtest DB tables for fresh start
#   --model-source   Choose training source (backtest/live/auto)
# =============================================================

import sys
import os
import datetime
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logger import get_logger
from core.connection import connect, disconnect

log = get_logger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="APEX TRADER — Backtesting Engine v2.1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m backtest.run                                  # Backtest all symbols (6 months)
  python -m backtest.run --symbols EURUSD GBPUSD           # Specific symbols only
  python -m backtest.run --days 90                        # Last 90 days
  python -m backtest.run --max-trades 50                  # Limit to 50 trades per symbol
  python -m backtest.run --strategies SMC_OB_REVERSAL TREND_CONTINUATION
  python -m backtest.run --no-partial-tp                  # Disable partial TP
  python -m backtest.run --no-trailing                    # Disable ATR trailing
  python -m backtest.run --no-dynamic-sizing              # Use fixed 1% risk
  python -m backtest.run --scan-every 30                  # Scan every 30 M1 bars

  # ── Data collection & ML training ──────────────────────
  python -m backtest.run --relaxed --store-db             # Collect training data
  python -m backtest.run --train                          # Train XGBoost from DB
  python -m backtest.run --use-model                      # Run with model active
  python -m backtest.run --relaxed --store-db --use-model # Collect + model combined
  python -m backtest.run --train --model-source backtest  # Force backtest data
  python -m backtest.run --clear-data                     # Clear DB for fresh start
  python -m backtest.run --model-status                   # Show model info
        """)

    # ── Backtest parameters ───────────────────────────────
    parser.add_argument(
        '--symbols', '-s', nargs='+', default=None,
        help='Symbols to backtest (default: all from config)')
    parser.add_argument(
        '--days', '-d', type=int, default=180,
        help='Number of days to backtest (default: 180)')
    parser.add_argument(
        '--max-trades', '-m', type=int, default=0,
        help='Max trades per symbol (0 = unlimited)')
    parser.add_argument(
        '--strategies', nargs='+', default=None,
        help='Specific strategies to test (default: all active)')
    parser.add_argument(
        '--scan-every', type=int, default=15,
        help='Scan every N M1 bars (default: 15 = M15)')

    # ── Feature toggles ───────────────────────────────────
    parser.add_argument(
        '--no-partial-tp', action='store_true',
        help='Disable partial TP (close 50%% at 1R)')
    parser.add_argument(
        '--no-trailing', action='store_true',
        help='Disable ATR trailing stop')
    parser.add_argument(
        '--no-dynamic-sizing', action='store_true',
        help='Disable dynamic position sizing (use fixed risk)')
    parser.add_argument(
        '--no-dynamic-tp', action='store_true',
        help='Disable dynamic TP extension')

    # ── Data management ───────────────────────────────────
    parser.add_argument(
        '--clear-cache', action='store_true',
        help='Clear cached data and re-download')
    parser.add_argument(
        '--clear-data', action='store_true',
        help='Clear all backtest DB tables (trades, signals, and all strategy '
             'feature tables) for a fresh start. Model files are NOT deleted.')
    parser.add_argument(
        '--balance', type=float, default=20000.0,
        help='Starting balance (default: 20000)')

    # ── Backtest modes ────────────────────────────────────
    parser.add_argument(
        '--relaxed', action='store_true',
        help='Relaxed mode: majority consensus (1 group), lower final_score gate (35), '
             'lower confluence min (4). Use for data collection / model training.')
    parser.add_argument(
        '--store-db', action='store_true',
        help='Store executed trades into MySQL for ML training. '
             'Blocked signals are NOT stored (they were never used in training).')
    parser.add_argument(
        '--parallel', action='store_true',
        help='Run all symbols in parallel on the same M1 timeline (like live trading). '
             'Much faster and more realistic than sequential execution.')
    parser.add_argument(
        '--no-limit', action='store_true',
        help='Remove max open position limits (no cap on total trades, '
             'multiple trades per symbol allowed). Use for data collection.')
    parser.add_argument(
        '--no-post-gates', action='store_true',
        help='Skip post-strategy gates (bias mismatch, consensus, confluence). '
             'Let L1 strategy models handle filtering instead. '
             'Requires --use-strategy-models to be effective.')

    # ── ML model commands ─────────────────────────────────
    model_group = parser.add_argument_group(
        'ML Gate v3.0 — Strategy-Informed ML',
        'Train and use the 60-feature ML gate to filter/replaces consensus gates')
    model_group.add_argument(
        '--train', action='store_true',
        help='Train ML Gate v3.0 from backtest DB data and exit. '
             'Requires --store-db data from a previous run.')
    model_group.add_argument(
        '--model-source', choices=['backtest', 'live', 'auto'],
        default='backtest',
        help='Training data source (default: backtest).')
    model_group.add_argument(
        '--use-model', action='store_true',
        help='Run backtest with ML Gate v3.0 replacing consensus gates. '
             'Collects ALL strategy scores as features for the model.')
    model_group.add_argument(
        '--model-status', action='store_true',
        help='Show model training status and exit.')

    # ── Layer 1 Strategy Model commands ──────────────────
    strat_model_group = parser.add_argument_group(
        'Layer 1 Strategy Models',
        'Per-strategy XGBoost models that replace hard-coded gates inside each strategy')
    strat_model_group.add_argument(
        '--use-strategy-models', action='store_true',
        help='Activate Layer 1 per-strategy models (PASS/REJECT before ML Gate). '
             'Requires trained strategy models (use --train-strategy-model first).')
    strat_model_group.add_argument(
        '--strategy-model-status', action='store_true',
        help='Show Layer 1 strategy model status and exit.')
    strat_model_group.add_argument(
        '--train-strategy-model', action='store_true',
        help='Train Layer 1 strategy model from backtest DB and exit.')
    strat_model_group.add_argument(
        '--train-strategy', type=str, default='VWAP_MEAN_REVERSION',
        help='Strategy to train for --train-strategy-model (default: VWAP_MEAN_REVERSION).')

    return parser.parse_args()


def _print_model_status():
    """Display ML Gate model status."""
    print("\n" + "="*60)
    print("  APEX TRADER — ML Gate v3.1 STATUS (Regression)")
    print("="*60)

    try:
        from ai_engine.ml_gate import is_model_trained, get_model_info

        if is_model_trained():
            info = get_model_info()
            print(f"\n  ML Gate v3.1 (Regression):")
            print(f"    Status:           TRAINED")
            print(f"    Version:          {info.get('version', '?')}")
            print(f"    Model type:       {info.get('model_type', '?')}")
            print(f"    Target:           {info.get('target', '?')}")
            print(f"    Features:         {info.get('n_features', 63)}")
            print(f"    Training trades:  {info.get('total_trades', '?')}")
            print(f"    Win rate:         {info.get('win_rate', '?')}%")
            print(f"    Mean R:           {info.get('mean_r', '?')}")
            print(f"    Median R:         {info.get('median_r', '?')}")
            print(f"    Val MAE:          {info.get('val_mae', '?')}")
            print(f"    Val R²:           {info.get('val_r2', '?')}")
            print(f"    Val Correlation:  {info.get('val_correlation', '?')}")
            print(f"    Model size:       {info.get('model_size_kb', '?')} KB")
            print(f"    Trained at:       {info.get('trained_at', '?')[:19]}")
            print(f"    TAKE threshold:   R >= {info.get('take_threshold', 0.5)}")
            print(f"    CAUTION threshold: R >= {info.get('caution_threshold', 0.0)}")

            top = info.get('top_features', [])
            if top:
                print(f"\n  Top 5 Features:")
                for fname, imp in top[:5]:
                    bar = '#' * int(imp * 200)
                    print(f"    {fname:30s} {imp:.4f}  {bar}")

            # Quintile calibration
            calibration = info.get('calibration', [])
            if calibration:
                print(f"\n  Quintile Calibration (predicted vs actual R):")
                for bucket in calibration:
                    for name, cal in bucket.items():
                        print(f"    {name}: pred_R={cal['predicted_r']:.3f} "
                              f"actual_R={cal['actual_mean_r']:.3f} "
                              f"WR={cal['win_rate_pct']:.1f}% "
                              f"(n={cal['count']})")
        else:
            print(f"\n  ML Gate v3.1 (Regression):")
            print(f"    Status:    NOT TRAINED")
            print(f"    Action:    Run --relaxed --store-db --no-limit, then --train")

        # DB stats
        try:
            from backtest.db_store import get_stats
            stats = get_stats()
            print(f"\n  Training Data in DB:")
            print(f"    Total trades:      {stats.get('total_trades', 0)}")
            print(f"    Total wins:        {stats.get('total_wins', 0)}")
            print(f"    Blocked signals:   {stats.get('total_blocked_signals', 0)}")
            wr = stats.get('win_rate', 0)
            print(f"    Overall win rate:  {wr}%")
        except Exception as e:
            print(f"\n  DB Stats:  Error loading: {e}")

        print("="*60 + "\n")

    except Exception as e:
        print(f"  Error: {e}")


def _train_model(model_source: str):
    """Train the ML Gate v3.1 regression model and display results."""
    print("\n" + "="*60)
    print("  APEX TRADER — ML Gate v3.1 TRAINING (Regression)")
    print("  (Predicts R-multiple per trade — not binary win/loss)")
    print("  (63 features: market quality, OF, VWAP, SMC, all 10 strategy scores, Fib)")
    print("="*60)

    try:
        from ai_engine.ml_gate import train_model

        print(f"\n  Training source: {model_source}")
        print(f"  Target: profit_r (R-multiple, continuous)")
        print(f"  This may take a moment...\n")

        result = train_model(source=model_source)

        if result['status'] == 'trained':
            print(f"  Training: SUCCESS")
            print(f"  Version:  {result.get('version', '3.1')}")
            print(f"  Type:     {result.get('model_type', 'XGBRegressor')}")
            print(f"  Trades:   {result.get('total_trades', 0)} "
                  f"({result.get('wins', 0)}W / {result.get('losses', 0)}L)")
            print(f"  WR:       {result.get('win_rate', 0)}%")
            print(f"  Features: {result.get('n_features', 63)}")
            print(f"")
            print(f"  R-multiple Distribution:")
            print(f"    Mean R:           {result.get('mean_r', 0)}")
            print(f"    Median R:         {result.get('median_r', 0)}")
            print(f"    Std Dev:          {result.get('std_r', 0)}")
            print(f"    Avg Win R:        {result.get('positive_mean_r', 0)}")
            print(f"    Avg Loss R:       {result.get('negative_mean_r', 0)}")
            print(f"")
            print(f"  Regression Metrics:")
            print(f"    Train MAE:        {result.get('train_mae', '?')}")
            print(f"    Val MAE:          {result.get('val_mae', '?')}")
            print(f"    Val RMSE:         {result.get('val_rmse', '?')}")
            print(f"    Val R²:           {result.get('val_r2', '?')}")
            print(f"    Val Correlation:  {result.get('val_correlation', '?')}")
            print(f"    Best Iteration:   {result.get('best_iteration', '?')}")
            print(f"    Model:            {result.get('model_size_kb', '?')} KB")
            print(f"    Path:             ai_engine/models/ml_gate_v3r.pkl")

            # Top features
            if 'top_features' in result:
                print(f"\n  Top 15 Features:")
                for fname, imp in result['top_features']:
                    bar = '#' * int(imp * 200)
                    print(f"    {fname:30s} {imp:.4f}  {bar}")

            # Quintile calibration
            calibration = result.get('calibration', [])
            if calibration:
                print(f"\n  Quintile Calibration (sorted by predicted R):")
                print(f"    {'Quintile':20s} {'Pred R':>8s} {'Actual R':>9s} {'WR%':>6s} {'N':>5s}")
                print(f"    {'-'*50}")
                for bucket in calibration:
                    for name, cal in bucket.items():
                        print(f"    {name:20s} {cal['predicted_r']:>8.3f} "
                              f"{cal['actual_mean_r']:>9.3f} "
                              f"{cal['win_rate_pct']:>5.1f}% "
                              f"{cal['count']:>5d}")

            print(f"\n  Execution thresholds:")
            print(f"    TAKE:    R >= {result.get('take_threshold', 0.5)}")
            print(f"    CAUTION: R >= {result.get('caution_threshold', 0.0)}")
            print(f"    SKIP:    R <  {result.get('caution_threshold', 0.0)}")
            print(f"\n  Model persists after restart — saved to disk.")
            print(f"  Use --use-model in your next backtest to activate.")
        else:
            print(f"  Training: {result['status'].upper()}")
            print(f"  Reason:   {result.get('reason', 'Unknown')}")
            print(f"\n  To fix this:")
            print(f"    1. Run: python -m backtest.run --relaxed --store-db --no-limit")
            print(f"       (collects trade data with rich features from all pairs)")
            print(f"    2. Then: python -m backtest.run --train")

        print("="*60 + "\n")

    except ImportError as e:
        print(f"  ERROR: Missing dependency — {e}")
        print(f"  Install: pip install xgboost scikit-learn joblib")
    except Exception as e:
        print(f"  ERROR: {e}")


def _clear_backtest_data():
    """Clear all backtest tables from the database."""
    print("\n" + "="*60)
    print("  APEX TRADER — CLEAR BACKTEST DATA")
    print("="*60)

    try:
        from database.db_manager import get_connection

        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        # All tables to clear
        tables = [
            'backtest_trades',
            'backtest_signals',
            'backtest_delta_div_features',
            'backtest_rsi_div_features',
            'backtest_liq_sweep_features',
            'backtest_ema_cross_features',
            'backtest_smc_ob_features',
            'backtest_structure_features',
            'backtest_trend_cont_features',
            'backtest_breakout_features',
            'backtest_fvg_features',
            'backtest_vwap_features',
        ]

        # Count before clearing
        print(f"\n  Current data:")
        total_rows = 0
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) as cnt FROM {table}")
                cnt = cursor.fetchone()['cnt']
                total_rows += cnt
                if cnt > 0:
                    print(f"    {table}:  {cnt} rows")
            except Exception:
                pass  # table doesn't exist yet

        if total_rows == 0:
            print(f"    (empty)")
            print(f"\n  No data to clear.")
            print("="*60 + "\n")
            conn.close()
            return

        # Clear all tables (TRUNCATE is faster than DELETE and resets AUTO_INCREMENT)
        for table in tables:
            try:
                cursor.execute(f"TRUNCATE TABLE {table}")
            except Exception:
                # Table might not exist yet — try DELETE instead
                try:
                    cursor.execute(f"DELETE FROM {table}")
                except Exception:
                    pass  # table doesn't exist, skip

        conn.commit()
        conn.close()

        print(f"\n  Cleared {total_rows} total rows across {len(tables)} tables")
        print(f"  DB is now ready for a fresh data collection run.")
        print(f"\n  Note: Your trained models are NOT deleted.")
        print(f"  Model files in ai_engine/models/ persist.")
        print(f"  To retrain after collecting new data: --train")
        print("="*60 + "\n")

    except Exception as e:
        print(f"  ERROR: {e}")
        print("="*60 + "\n")


def main():
    """Run backtests on all configured symbols."""
    args = parse_args()

    # ── Standalone commands (don't need MT5) ──────────────
    if args.model_status:
        _print_model_status()
        return

    if args.train:
        _train_model(args.model_source)
        return

    if args.clear_data:
        _clear_backtest_data()
        return

    if args.strategy_model_status:
        from ai_engine.train_strategy_model import _print_strategy_status
        _print_strategy_status()
        return

    if args.train_strategy_model:
        from ai_engine.train_strategy_model import _train_strategy
        _train_strategy(args.train_strategy)
        return

    # ── Backtest execution ────────────────────────────────
    # Apply CLI overrides to config
    from backtest import config as bt_config

    # Relaxed mode: auto-disable partial TP, trailing, and dynamic TP extension
    # to test with full trades running to completion
    if args.relaxed:
        bt_config.PARTIAL_TP_ENABLED = False
        bt_config.ATR_TRAIL_ENABLED = False
        bt_config.DYNAMIC_TP_EXTENSION_ENABLED = False

    if args.no_partial_tp:
        bt_config.PARTIAL_TP_ENABLED = False
    if args.no_trailing:
        bt_config.ATR_TRAIL_ENABLED = False
    if args.no_dynamic_sizing:
        bt_config.DYNAMIC_SIZING_ENABLED = False
    if args.no_dynamic_tp:
        bt_config.DYNAMIC_TP_EXTENSION_ENABLED = False

    # Clear cache if requested
    if args.clear_cache:
        import shutil
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'backtest', '.cache')
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print("  Cache cleared.")

    # Check model status if --use-model
    use_model = args.use_model
    model_loaded = False
    if use_model:
        from ai_engine.ml_gate import is_model_trained
        if not is_model_trained():
            print("\n  WARNING: --use-model specified but no trained model found!")
            print("  Run --relaxed --store-db --no-limit, then --train first.")
            print("  Continuing WITHOUT model filtering...\n")
            use_model = False
        else:
            model_loaded = True

    # Check strategy model status if --use-strategy-models
    use_strategy_models = args.use_strategy_models
    strat_models_loaded = False
    if use_strategy_models:
        from ai_engine.strategy_model import get_strategy_model_manager
        mgr = get_strategy_model_manager()
        if mgr._active:
            strat_models_loaded = True
        else:
            print("\n  WARNING: --use-strategy-models but no trained strategy models found!")
            print("  Run --train-strategy-model first.")
            print("  Continuing WITHOUT strategy model filtering...\n")
            use_strategy_models = False

    # Date range
    end_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1)
    start_date = end_date - datetime.timedelta(days=args.days)

    # Symbols
    from backtest.config import SYMBOLS, SCAN_EVERY_N_BARS, STRATEGIES_FILTER
    symbols = args.symbols if args.symbols else SYMBOLS
    strategies = args.strategies if args.strategies else STRATEGIES_FILTER

    scan_every = args.scan_every if args.scan_every != 15 else SCAN_EVERY_N_BARS

    print("\n" + "="*65)
    print("  APEX TRADER — BACKTESTING ENGINE v2.1")
    print(f"  Period: {start_date.date()} to {end_date.date()} ({args.days} days)")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Scan frequency: every {scan_every} M1 bars")
    print(f"  Balance: ${args.balance:,.2f}")
    mode_label = "RELAXED" if args.relaxed else "STRICT"
    # In relaxed mode, partial TP/trail/ext TP are auto-disabled
    show_partial = not args.relaxed and not args.no_partial_tp
    show_trail = not args.relaxed and not args.no_trailing
    show_dynsize = not args.no_dynamic_sizing
    show_exttp = not args.relaxed and not args.no_dynamic_tp
    print(f"  Features: PartialTP={show_partial} "
          f"Trail={show_trail} "
          f"DynamicSize={show_dynsize} "
          f"ExtTP={show_exttp}")
    if args.relaxed:
        print(f"  (Relaxed mode: PartialTP/Trail/ExtTP auto-disabled for full trade testing)")
    print(f"  Mode: {mode_label} | Store DB: {args.store_db} "
          f"| Model: {'ACTIVE' if model_loaded else 'OFF'}"
          f" | Strategy Models: {'ACTIVE' if strat_models_loaded else 'OFF'}")
    if model_loaded:
        print(f"  (ML Gate will filter trades — SKIP recommendation = trade blocked)")
    if strat_models_loaded:
        print(f"  (Layer 1 strategy models will filter signals before ML Gate)")
    print("="*65 + "\n")

    # Connect to MT5 for historical data
    from backtest.engine import run_backtest, BacktestConfig
    from backtest.report import print_summary, print_full_report

    if not connect():
        print("Cannot connect to MT5. Aborting.")
        return

    try:
        all_results = []

        if args.parallel:
            # ── Parallel mode: all symbols on same M1 timeline ──
            from backtest.engine import run_parallel_backtest
            log.info(f"\n  Running PARALLEL backtest ({len(symbols)} symbols on same timeline)...\n")
            all_results = run_parallel_backtest(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                scan_every=scan_every,
                relaxed_mode=args.relaxed,
                store_db=args.store_db,
                use_model=use_model,
                use_strategy_models=use_strategy_models,
                run_id=f"{mode_label.lower()}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}",
                max_trades_per_symbol=args.max_trades if args.max_trades > 0 else 9999,
                unlimited_positions=args.no_limit,
                no_post_gates=args.no_post_gates,
            )
        else:
            # ── Sequential mode: one symbol at a time (original) ──
            for i, symbol in enumerate(symbols):
                print(f"\n  [{i+1}/{len(symbols)}] Processing {symbol}...")

                config = BacktestConfig(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    scan_every_n_bars=scan_every,
                    max_trades=args.max_trades,
                    strategies_filter=strategies,
                    relaxed_mode=args.relaxed,
                    store_db=args.store_db,
                    use_model=use_model,
                    use_strategy_models=use_strategy_models,
                    run_id=f"{mode_label.lower()}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}",
                    unlimited_positions=args.no_limit,
                    no_post_gates=args.no_post_gates,
                )

                result = run_backtest(config)
                if result:
                    print_summary(result)
                    all_results.append(result)

        # Print full cross-symbol summary
        if all_results:
            print_full_report(all_results)

            # ── Model filtering stats ──────────────────────────
            total_model_blocked = sum(
                r.get('model_blocked', 0) for r in all_results)
            total_shadow = sum(
                r.get('shadow_trades', 0) for r in all_results)
            total_strat_rejected = sum(
                r.get('strat_model_rejected', 0) for r in all_results)
            total_strat_shadow = sum(
                r.get('strat_model_shadow_trades', 0) for r in all_results)
            if model_loaded and (total_model_blocked > 0 or total_shadow > 0):
                print(f"\n{'='*65}")
                print(f"  MODEL FILTERING SUMMARY")
                print(f"{'='*65}")
                print(f"  ML Gate (Layer 2):")
                print(f"    Trades SKIPPED by model (R < 0.0):   {total_model_blocked}")
                print(f"    Trades SHADOWED by model (0.0 <= R < 0.5): {total_shadow}")
                if strat_models_loaded and (total_strat_rejected > 0 or total_strat_shadow > 0):
                    print(f"  Strategy Models (Layer 1):")
                    print(f"    Signals REJECTED by strategy model: {total_strat_rejected}")
                    print(f"    L1 shadow trades simulated: {total_strat_shadow}")
                print(f"  Shadow trades are simulated and stored for ML training")
                print(f"{'='*65}")

            # Print DB stats if --store-db was used
            if args.store_db:
                try:
                    from backtest.db_store import get_stats
                    stats = get_stats()
                    print(f"\n{'='*65}")
                    print(f"  DATABASE STORAGE SUMMARY")
                    print(f"{'='*65}")
                    print(f"  Real trades stored:  {stats.get('total_trades', 0)}")
                    print(f"  Real trade wins:     {stats.get('total_wins', 0)}")
                    print(f"  Shadow trades stored: {stats.get('shadow_trades', 0)}")
                    print(f"  Shadow trade wins:    {stats.get('shadow_wins', 0)}")
                    print(f"  Total for training:   {stats.get('total_trades', 0) + stats.get('shadow_trades', 0)}")
                    if stats.get('by_strategy'):
                        print(f"\n  BY STRATEGY:")
                        for s in stats['by_strategy']:
                            wr = round(s['wins']/s['trades']*100, 1) if s['trades'] > 0 else 0
                            print(f"    {s['strategy']:30s} {s['trades']:4d} trades | {wr:5.1f}% WR | avg R: {s['avg_r']} | avg score: {s['avg_score']}")
                    print(f"{'='*65}")
                except Exception as e:
                    print(f"  [DB] Could not load stats: {e}")

            # Save results to file
            import json
            clean_results = []
            for r in all_results:
                clean = {}
                for k, v in r.items():
                    try:
                        json.dumps(v)
                        clean[k] = v
                    except (TypeError, ValueError):
                        clean[k] = str(v)
                clean_results.append(clean)

            output_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'backtest', 'results.json'
            )
            with open(output_path, 'w') as f:
                json.dump(clean_results, f, indent=2, default=str)
            print(f"\n  Results saved to: {output_path}")

            # Also save to download directory
            download_path = '/home/z/my-project/download/backtest_results.json'
            try:
                with open(download_path, 'w') as f:
                    json.dump(clean_results, f, indent=2, default=str)
                print(f"  Results also saved to: {download_path}")
            except Exception:
                pass

        else:
            print("\n  No results to report.")

    finally:
        disconnect()


if __name__ == "__main__":
    main()
