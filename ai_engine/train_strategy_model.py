# =============================================================
# ai_engine/train_strategy_model.py  v1.0
# CLI entry point: python -m ai_engine.train_strategy_model
#
# Trains a per-strategy Layer 1 model from backtest DB data.
#
# Usage:
#   python -m ai_engine.train_strategy_model                          # Train VWAP (default)
#   python -m ai_engine.train_strategy_model --strategy VWAP_MEAN_REVERSION
#   python -m ai_engine.train_strategy_model --strategy BREAKOUT_MOMENTUM
#   python -m ai_engine.train_strategy_model --status                 # Show all model statuses
#   python -m ai_engine.train_strategy_model --train-all              # Train all with enough data
# =============================================================

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _print_strategy_status():
    """Display status of all strategy models."""
    print("\n" + "=" * 65)
    print("  APEX TRADER — Layer 1 Strategy Models STATUS")
    print("=" * 65)

    try:
        from ai_engine.strategy_model import get_strategy_model_manager

        mgr = get_strategy_model_manager()
        status = mgr.get_status()

        print(f"\n  Active models: {status['total_active']}/{len(status['strategies'])}")
        print()

        for name, info in status['strategies'].items():
            trained = info.get('trained', False)
            if trained:
                key = info.get('key', '?')
                trades = info.get('total_trades', '?')
                wr = info.get('win_rate', '?')
                mean_r = info.get('mean_r', '?')
                pass_count = info.get('pass_count', '?')
                reject_count = info.get('reject_count', '?')
                pass_wr = info.get('pass_wr', '?')
                reject_wr = info.get('reject_wr', '?')
                val_r2 = info.get('val_r2', '?')

                print(f"  {name:30s}  TRAINED")
                print(f"    Trades: {trades} | WR: {wr}% | Mean R: {mean_r}")
                print(f"    PASS: {pass_count} ({pass_wr}% WR) | REJECT: {reject_count} ({reject_wr}% WR)")
                print(f"    Val R²: {val_r2} | Size: {info.get('model_size_kb', '?')} KB")
                print(f"    Threshold: R >= {info.get('pass_threshold', 0.2)}")
                print()

                top = info.get('top_features', [])
                if top:
                    print(f"    Top 5 Features:")
                    for fname, imp in top[:5]:
                        bar = '#' * int(imp * 200)
                        print(f"      {fname:30s} {imp:.4f}  {bar}")
                    print()
            else:
                print(f"  {name:30s}  NOT TRAINED")
                print()

        # DB stats per strategy
        try:
            from database.db_manager import get_connection
            conn = get_connection()
            cursor = conn.cursor(dictionary=True)

            cursor.execute("""
                SELECT strategy, COUNT(*) as trades,
                       SUM(win) as wins,
                       ROUND(AVG(profit_r),3) as avg_r
                FROM backtest_trades
                WHERE source IN ('BACKTEST', 'SHADOW')
                  AND outcome IS NOT NULL
                  AND profit_r IS NOT NULL
                GROUP BY strategy
                ORDER BY trades DESC
            """)
            rows = cursor.fetchall()
            conn.close()

            if rows:
                print(f"\n  {'='*65}")
                print(f"  AVAILABLE TRAINING DATA (from DB):")
                print(f"  {'Strategy':30s} {'Trades':>8s} {'Wins':>6s} {'WR%':>7s} {'Avg R':>8s}")
                print(f"  {'-'*60}")
                for r in rows:
                    n = r['trades'] or 0
                    w = r['wins'] or 0
                    wr = round(w/n*100, 1) if n > 0 else 0
                    print(f"  {r['strategy']:30s} {n:>8d} {w:>6d} {wr:>6.1f}% {r['avg_r']:>8s}")
                    if n >= 50:
                        print(f"  {'  ^ MINIMUM FOR TRAINING':>45s}")
                print()

        except Exception as e:
            print(f"\n  DB Stats: Error loading: {e}")

    except Exception as e:
        print(f"  Error: {e}")

    print("=" * 65 + "\n")


def _train_strategy(strategy_name: str):
    """Train a specific strategy model."""
    print("\n" + "=" * 65)
    print(f"  APEX TRADER — Layer 1 Training: {strategy_name}")
    print("=" * 65)

    try:
        from ai_engine.strategy_model import get_strategy_model_manager
        from ai_engine.ml_gate import FEATURE_NAMES

        mgr = get_strategy_model_manager()

        print(f"\n  Strategy: {strategy_name}")
        print(f"  Features: {len(FEATURE_NAMES)} (same as ML Gate v3.3)")
        print(f"  Target:   profit_r (R-multiple, continuous)")
        print(f"  Layer:    1 (per-strategy)")
        print(f"  Training...\n")

        result = mgr.train_strategy(strategy_name)

        if result['status'] == 'trained':
            print(f"  Training: SUCCESS")
            print(f"  Version:  {result.get('version', '1.0')}")
            print(f"  Type:     {result.get('model_type', 'XGBRegressor')}")
            print(f"  Trades:   {result.get('total_trades', 0)} "
                  f"({result.get('wins', 0)}W / {result.get('losses', 0)}L)")
            print(f"  WR:       {result.get('win_rate', 0)}%")
            print(f"  Mean R:   {result.get('mean_r', 0)}")
            print(f"  Median R: {result.get('median_r', 0)}")
            print(f"")
            print(f"  Regression Metrics:")
            print(f"    Train MAE:        {result.get('train_mae', '?')}")
            print(f"    Val MAE:          {result.get('val_mae', '?')}")
            print(f"    Val RMSE:         {result.get('val_rmse', '?')}")
            print(f"    Val R²:           {result.get('val_r2', '?')}")
            print(f"    Val Correlation:  {result.get('val_correlation', '?')}")
            print(f"")
            print(f"  Selection Analysis (model's own filtering):")
            print(f"    PASS threshold:   R >= {result.get('pass_threshold', 0.2)}")
            print(f"    Trades PASSED:    {result.get('pass_count', 0)} "
                  f"({result.get('pass_wr', 0)}% WR)")
            print(f"    Trades REJECTED:  {result.get('reject_count', 0)} "
                  f"({result.get('reject_wr', 0)}% WR)")
            print(f"    Model size:       {result.get('model_size_kb', '?')} KB")
            print(f"    Best iteration:   {result.get('best_iteration', '?')}")

            # Top features
            if 'top_features' in result:
                print(f"\n  Top 10 Features:")
                for fname, imp in result['top_features'][:10]:
                    bar = '#' * int(imp * 200)
                    print(f"    {fname:30s} {imp:.4f}  {bar}")

            # Quintile calibration
            calibration = result.get('calibration', [])
            if calibration:
                print(f"\n  Quintile Calibration:")
                print(f"    {'Bucket':20s} {'Pred R':>8s} {'Actual R':>9s} {'WR%':>6s} {'N':>5s}")
                print(f"    {'-'*50}")
                for bucket in calibration:
                    for name, cal in bucket.items():
                        print(f"    {name:20s} {cal['predicted_r']:>8.3f} "
                              f"{cal['actual_mean_r']:>9.3f} "
                              f"{cal['win_rate_pct']:>5.1f}% "
                              f"{cal['count']:>5d}")

            print(f"\n  PASS vs REJECT comparison:")
            pass_wr = result.get('pass_wr', 0)
            reject_wr = result.get('reject_wr', 0)
            if pass_wr > reject_wr:
                diff = pass_wr - reject_wr
                print(f"    Model PASS has {diff:.1f}% HIGHER win rate than REJECT")
                print(f"    This means the model IS learning to filter bad trades")
            elif pass_wr < reject_wr:
                diff = reject_wr - pass_wr
                print(f"    WARNING: Model REJECT has {diff:.1f}% HIGHER win rate than PASS")
                print(f"    The model may be filtering GOOD trades — needs investigation")
            else:
                print(f"    No meaningful difference between PASS and REJECT")

            print(f"\n  Model saved to: ai_engine/models/{result.get('strategy_key', '?')}_strategy_model.pkl")
            print(f"  Use --use-strategy-models in backtest to activate")
        else:
            print(f"  Training: {result['status'].upper()}")
            print(f"  Reason:   {result.get('reason', 'Unknown')}")
            print(f"\n  To fix:")
            print(f"    1. Run: python -m backtest.run --relaxed --store-db --no-limit")
            print(f"    2. Wait for 50+ trades from {strategy_name}")
            print(f"    3. Then re-run this training command")

        print("=" * 65 + "\n")

    except ImportError as e:
        print(f"  ERROR: Missing dependency — {e}")
        print(f"  Install: pip install xgboost scikit-learn joblib")
    except Exception as e:
        print(f"  ERROR: {e}")


def _train_all_eligible():
    """Train all strategies that have enough data."""
    print("\n" + "=" * 65)
    print("  APEX TRADER — Layer 1: Train All Eligible Strategies")
    print("=" * 65)

    try:
        from database.db_manager import get_connection

        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT strategy, COUNT(*) as trades
            FROM backtest_trades
            WHERE source IN ('BACKTEST', 'SHADOW')
              AND outcome IS NOT NULL
              AND profit_r IS NOT NULL
            GROUP BY strategy
            HAVING trades >= 50
            ORDER BY trades ASC
        """)
        eligible = cursor.fetchall()
        conn.close()

        if not eligible:
            print("\n  No strategies have 50+ trades yet.")
            print("  Run: python -m backtest.run --relaxed --store-db --no-limit")
            print("=" * 65 + "\n")
            return

        print(f"\n  Eligible strategies ({len(eligible)}):")
        for r in eligible:
            print(f"    {r['strategy']:30s} {r['trades']} trades")
        print()

        from ai_engine.strategy_model import get_strategy_model_manager
        mgr = get_strategy_model_manager()

        for r in eligible:
            _train_strategy(r['strategy'])

    except Exception as e:
        print(f"  ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="APEX TRADER — Layer 1 Strategy Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '--strategy', '-s', type=str, default='VWAP_MEAN_REVERSION',
        help='Strategy to train (default: VWAP_MEAN_REVERSION)')
    parser.add_argument(
        '--status', action='store_true',
        help='Show status of all strategy models')
    parser.add_argument(
        '--train-all', action='store_true',
        help='Train all strategies with 50+ trades in DB')

    args = parser.parse_args()

    if args.status:
        _print_strategy_status()
    elif args.train_all:
        _train_all_eligible()
    else:
        _train_strategy(args.strategy)


if __name__ == "__main__":
    main()
