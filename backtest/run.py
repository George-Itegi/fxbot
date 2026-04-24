# =============================================================
# backtest/run.py  v3.0
# CLI entry point: python -m backtest.run
# v3.0 CHANGES (Strategy-Informed ML):
#   --collect-ml     Record trade features+outcomes into SignalModel history
#   --use-model      Filter trades using SignalModel v2 (60 features)
#   --walk-forward   Run walk-forward validation (train → test → slide)
#   --train-months   Training window size for walk-forward (default: 4)
#   --test-months    Test window size for walk-forward (default: 2)
#   --ml-threshold   WIN probability threshold (default: 0.62)
#   --seed-model     Bootstrap SignalModel from backtest DB before running
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
  python -m backtest.run --collect-ml                     # Record features→SignalModel
  python -m backtest.run --use-model                      # Run with SignalModel v2
  python -m backtest.run --walk-forward                   # Walk-forward validation
  python -m backtest.run --walk-forward --train-months 6 --test-months 2
  python -m backtest.run --seed-model                     # Bootstrap from backtest DB
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
        help='Clear all backtest_trades and backtest_signals from DB. '
             'Use before a fresh data collection run.')
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
        help='Store every trade and blocked signal into MySQL for ML training.')
    parser.add_argument(
        '--parallel', action='store_true',
        help='Run all symbols in parallel on the same M1 timeline (like live trading). '
             'Much faster and more realistic than sequential execution.')
    parser.add_argument(
        '--no-limit', action='store_true',
        help='Remove max open position limits (no cap on total trades, '
             'multiple trades per symbol allowed). Use for data collection.')

    # ── ML model commands ─────────────────────────────────
    model_group = parser.add_argument_group(
        'ML Model Training & Usage',
        'Train and use the XGBoost model to filter trades')
    model_group.add_argument(
        '--train', action='store_true',
        help='Train SignalModel from backtest DB data and exit.')
    model_group.add_argument(
        '--model-source', choices=['backtest', 'live', 'auto'],
        default='backtest',
        help='Training data source (default: backtest).')
    model_group.add_argument(
        '--use-model', action='store_true',
        help='Filter trades using SignalModel v2 (60 features, Strategy-Informed ML).')
    model_group.add_argument(
        '--collect-ml', action='store_true',
        help='Record trade features + outcomes into SignalModel history for training.')
    model_group.add_argument(
        '--seed-model', action='store_true',
        help='Bootstrap SignalModel from backtest DB before running.')
    model_group.add_argument(
        '--walk-forward', action='store_true',
        help='Run walk-forward validation: train ML on first N months, test on next M.')
    model_group.add_argument(
        '--train-months', type=int, default=4,
        help='Training window for walk-forward (default: 4 months).')
    model_group.add_argument(
        '--test-months', type=int, default=2,
        help='Test window for walk-forward (default: 2 months).')
    model_group.add_argument(
        '--ml-threshold', type=float, default=0.62,
        help='WIN probability threshold for ML gate (default: 0.62 = 62%%).')
    model_group.add_argument(
        '--model-status', action='store_true',
        help='Show SignalModel status and exit.')

    return parser.parse_args()


def _print_model_status():
    """Display model training status."""
    print("\n" + "="*60)
    print("  APEX TRADER — MODEL STATUS")
    print("="*60)

    try:
        from ai_engine.model_trainer import get_model_status
        from ai_engine.xgboost_classifier import is_model_trained

        status = get_model_status()

        # XGBoost
        xgb = status['xgboost']
        if xgb['trained']:
            print(f"\n  XGBoost Model:")
            print(f"    Status:    TRAINED")
            print(f"    File:      {xgb['path']}")
            print(f"    Size:      {xgb['size_kb']} KB")
            print(f"    Age:       {xgb.get('age_hours', '?')} hours ago")
        else:
            print(f"\n  XGBoost Model:")
            print(f"    Status:    NOT TRAINED")
            print(f"    Action:    Run --relaxed --store-db first, then --train")

        # LSTM
        lstm = status['lstm']
        if lstm['trained']:
            print(f"\n  LSTM Model:")
            print(f"    Status:    TRAINED")
            print(f"    Size:      {lstm['size_kb']} KB")
            print(f"    Age:       {lstm.get('age_hours', '?')} hours ago")
        else:
            print(f"\n  LSTM Model:")
            print(f"    Status:    NOT TRAINED (not recommended yet)")

        # DB stats
        try:
            from backtest.db_store import get_stats
            stats = get_stats()
            print(f"\n  Training Data in DB:")
            print(f"    Total trades:      {stats.get('total_trades', 0)}")
            print(f"    Total wins:        {stats.get('total_wins', 0)}")
            print(f"    Blocked signals:   {stats.get('total_blocked_signals', 0)}")
            print(f"    Executed signals:  {stats.get('total_executed_signals', 0)}")
            wr = stats.get('win_rate', 0)
            print(f"    Overall win rate:  {wr}%")
            if wr > 0:
                print(f"    Ready to train:    YES (50+ trades available)")
            else:
                print(f"    Ready to train:    NO (need 50+ trades)")
        except Exception as e:
            print(f"\n  DB Stats:  Error loading: {e}")

        print(f"\n  Models directory: {status['models_dir']}")
        print("="*60 + "\n")

    except Exception as e:
        print(f"  Error: {e}")


def _train_model(model_source: str):
    """Train the XGBoost model and display results."""
    print("\n" + "="*60)
    print("  APEX TRADER — XGBoost MODEL TRAINING")
    print("="*60)

    try:
        from ai_engine.model_trainer import train_xgboost

        print(f"\n  Training source: {model_source}")
        print(f"  This may take a moment...\n")

        result = train_xgboost(source=model_source)

        if result['status'] == 'trained':
            print(f"  Training: SUCCESS")
            print(f"  Source:   {result.get('source', model_source)}")
            print(f"  Trades:   {result.get('total_trades', 0)} "
                  f"({result.get('wins', 0)}W / {result.get('losses', 0)}L)")
            print(f"  WR:       {result.get('win_rate', 0)}%")
            print(f"  Train Acc:{result.get('train_accuracy', '?')}%")
            print(f"  Val Acc:  {result.get('val_accuracy', '?')}%")
            print(f"  Model:    {result.get('model_size_kb', '?')} KB")
            print(f"  Path:     ai_engine/models/xgb_model.pkl")

            # Top features
            if 'top_features' in result:
                print(f"\n  Top 10 Features:")
                for fname, imp in result['top_features']:
                    bar = '#' * int(imp * 100)
                    print(f"    {fname:25s} {imp:.4f}  {bar}")

            print(f"\n  Model persists after restart — saved to disk.")
            print(f"  Use --use-model in your next backtest to activate.")
        else:
            print(f"  Training: {result['status'].upper()}")
            print(f"  Reason:   {result.get('reason', 'Unknown')}")
            print(f"\n  To fix this:")
            print(f"    1. Run: python -m backtest.run --relaxed --store-db")
            print(f"       (collects trade data with rich features)")
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

        # Count before clearing
        cursor.execute("SELECT COUNT(*) as cnt FROM backtest_trades")
        trades_count = cursor.fetchone()['cnt']
        cursor.execute("SELECT COUNT(*) as cnt FROM backtest_signals")
        signals_count = cursor.fetchone()['cnt']

        print(f"\n  Current data:")
        print(f"    backtest_trades:   {trades_count} rows")
        print(f"    backtest_signals:  {signals_count} rows")

        if trades_count == 0 and signals_count == 0:
            print(f"\n  No data to clear.")
            print("="*60 + "\n")
            conn.close()
            return

        # Clear
        cursor.execute("DELETE FROM backtest_signals")
        cursor.execute("DELETE FROM backtest_trades")
        conn.commit()
        conn.close()

        print(f"\n  Cleared: {trades_count} trades + {signals_count} signals")
        print(f"  DB is now ready for a fresh data collection run.")
        print(f"\n  Note: Your trained model (if any) is NOT deleted.")
        print(f"  The model file (ai_engine/models/xgb_model.pkl) persists.")
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
        from ai_engine.xgboost_classifier import is_model_trained
        if not is_model_trained():
            print("\n  WARNING: --use-model specified but no trained model found!")
            print("  Run --train first to train the model.")
            print("  Continuing WITHOUT model filtering...\n")
            use_model = False
        else:
            model_loaded = True

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
          f"| Model: {'ACTIVE' if model_loaded else 'OFF'}")
    if model_loaded:
        print(f"  (Model will filter trades — SKIP recommendation = trade blocked)")
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
                run_id=f"{mode_label.lower()}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}",
                max_trades_per_symbol=args.max_trades if args.max_trades > 0 else 9999,
                unlimited_positions=args.no_limit,
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
                    use_model=args.use_model,
                    collect_ml_data=args.collect_ml or args.use_model,
                    ml_threshold=args.ml_threshold,
                    run_id=f"{mode_label.lower()}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}",
                    unlimited_positions=args.no_limit,
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
            if model_loaded and total_model_blocked > 0:
                print(f"\n{'='*65}")
                print(f"  MODEL FILTERING SUMMARY")
                print(f"{'='*65}")
                print(f"  Trades blocked by XGBoost model: {total_model_blocked}")
                print(f"  These trades had low win probability (SKIP recommendation)")
                print(f"{'='*65}")

            # Print DB stats if --store-db was used
            if args.store_db:
                try:
                    from backtest.db_store import get_stats
                    stats = get_stats()
                    print(f"\n{'='*65}")
                    print(f"  DATABASE STORAGE SUMMARY")
                    print(f"{'='*65}")
                    print(f"  Total trades stored:  {stats.get('total_trades', 0)}")
                    print(f"  Total wins:          {stats.get('total_wins', 0)}")
                    print(f"  Blocked signals:     {stats.get('total_blocked_signals', 0)}")
                    print(f"  Executed signals:    {stats.get('total_executed_signals', 0)}")
                    print(f"  Win rate:            {stats.get('win_rate', 0)}%")
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
