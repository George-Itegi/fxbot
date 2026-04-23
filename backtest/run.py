# =============================================================
# backtest/run.py
# CLI entry point: python -m backtest.run
# =============================================================

import sys
import os
import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logger import get_logger
from core.connection import connect, disconnect

log = get_logger(__name__)


def main():
    """Run backtests on all configured symbols."""
    print("\n" + "="*60)
    print("  APEX TRADER — BACKTESTING ENGINE v1.0")
    print("  Testing strategies on 6 months of historical data")
    print("="*60 + "\n")

    # Connect to MT5 for historical data
    from config.settings import WATCHLIST
    from backtest.config import (SYMBOLS, START_DATE, END_DATE,
                                  SCAN_EVERY_N_BARS)
    from backtest.engine import run_backtest, BacktestConfig
    from backtest.report import print_summary, print_full_report

    if not connect():
        print("Cannot connect to MT5. Aborting.")
        return

    try:
        all_results = []

        for symbol in SYMBOLS:
            config = BacktestConfig(
                symbol=symbol,
                start_date=START_DATE,
                end_date=END_DATE,
                scan_every_n_bars=SCAN_EVERY_N_BARS,
            )

            result = run_backtest(config)
            if result:
                print_summary(result)
                all_results.append(result)

        # Print full cross-symbol summary
        if all_results:
            print_full_report(all_results)

            # Save results to file
            import json
            # Convert non-serializable types
            clean_results = []
            for r in all_results:
                clean = {}
                for k, v in r.items():
                    if k == 'strategy_stats':
                        clean[k] = v  # dict is serializable
                    elif k == 'equity_curve':
                        pass  # Skip for now
                    else:
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

    finally:
        disconnect()


if __name__ == "__main__":
    main()
