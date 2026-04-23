# =============================================================
# backtest/run.py  v2.0
# CLI entry point: python -m backtest.run
# Upgraded: CLI arguments, parallel execution, detailed logging
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
        description="APEX TRADER — Backtesting Engine v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m backtest.run                          # Backtest all symbols (6 months)
  python -m backtest.run --symbols EURUSD GBPUSD   # Specific symbols only
  python -m backtest.run --days 90                 # Last 90 days
  python -m backtest.run --max-trades 50           # Limit to 50 trades per symbol
  python -m backtest.run --strategies SMC_OB_REVERSAL TREND_CONTINUATION
  python -m backtest.run --no-partial-tp           # Disable partial TP
  python -m backtest.run --no-trailing             # Disable ATR trailing
  python -m backtest.run --no-dynamic-sizing       # Use fixed 1% risk
  python -m backtest.run --scan-every 30           # Scan every 30 M1 bars
        """)

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
    parser.add_argument(
        '--clear-cache', action='store_true',
        help='Clear cached data and re-download')
    parser.add_argument(
        '--balance', type=float, default=20000.0,
        help='Starting balance (default: 20000)')
    parser.add_argument(
        '--relaxed', action='store_true',
        help='Relaxed mode: majority consensus (1 group), lower final_score gate (35), '
             'lower confluence min (4). Use for data collection / model training.')
    parser.add_argument(
        '--store-db', action='store_true',
        help='Store every trade and blocked signal into MySQL for ML training.')

    return parser.parse_args()


def main():
    """Run backtests on all configured symbols."""
    args = parse_args()

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

    # Date range
    end_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1)
    start_date = end_date - datetime.timedelta(days=args.days)

    # Symbols
    from backtest.config import SYMBOLS, SCAN_EVERY_N_BARS, STRATEGIES_FILTER
    symbols = args.symbols if args.symbols else SYMBOLS
    strategies = args.strategies if args.strategies else STRATEGIES_FILTER

    scan_every = args.scan_every if args.scan_every != 15 else SCAN_EVERY_N_BARS

    print("\n" + "="*65)
    print("  APEX TRADER — BACKTESTING ENGINE v2.0")
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
    print(f"  Mode: {mode_label} | Store DB: {args.store_db}")
    print("="*65 + "\n")

    # Connect to MT5 for historical data
    from backtest.engine import run_backtest, BacktestConfig
    from backtest.report import print_summary, print_full_report

    if not connect():
        print("Cannot connect to MT5. Aborting.")
        return

    try:
        all_results = []

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
                run_id=f"{mode_label.lower()}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}",
            )

            result = run_backtest(config)
            if result:
                print_summary(result)
                all_results.append(result)

        # Print full cross-symbol summary
        if all_results:
            print_full_report(all_results)

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
