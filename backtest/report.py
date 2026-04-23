# =============================================================
# backtest/report.py
# Formats and prints backtest results.
# =============================================================

from core.logger import get_logger

log = get_logger(__name__)


def print_summary(summary: dict):
    """Print a formatted performance report for one symbol."""
    if not summary or summary.get('total_trades', 0) == 0:
        print(f"\n  {summary.get('symbol', '?')}: NO TRADES EXECUTED")
        print(f"  Signals found: {summary.get('signals_found', 0)}")
        print(f"  Blocked (consensus): {summary.get('signals_blocked_consensus', 0)}")
        print(f"  Blocked (gates): {summary.get('signals_blocked_gate', 0)}")
        return

    symbol = summary.get('symbol', '?')
    t = summary['total_trades']
    w = summary['wins']
    l = summary['losses']
    wr = summary['win_rate']
    pnl = summary['total_pnl']
    pf = summary['profit_factor']
    mdd = summary['max_drawdown']
    avg_w = summary['avg_win_pips']
    avg_l = summary['avg_loss_pips']
    bal = summary['final_balance']

    # Performance rating
    if wr >= 55 and pf >= 1.5 and pnl > 0:
        rating = "STRONG"
        icon = "A+"
    elif wr >= 50 and pf >= 1.3:
        rating = "PROFITABLE"
        icon = "A"
    elif wr >= 45 and pf >= 1.0:
        rating = "MARGINAL"
        icon = "B"
    else:
        rating = "UNPROFITABLE"
        icon = "C"

    print(f"\n{'='*60}")
    print(f"  BACKTEST RESULTS: {symbol} [{icon} {rating}]")
    print(f"{'='*60}")
    print(f"  Total Trades:    {t}")
    print(f"  Wins / Losses:   {w} / {l}")
    print(f"  Win Rate:        {wr}%")
    print(f"  Total P&L:       ${pnl:+,.2f}")
    print(f"  Profit Factor:   {pf}")
    print(f"  Max Drawdown:    ${mdd:,.2f}")
    print(f"  Avg Win:         +{avg_w} pips")
    print(f"  Avg Loss:        {avg_l} pips")
    print(f"  Final Balance:   ${bal:,.2f}")
    print(f"  Duration:        {summary.get('elapsed_seconds', 0)}s")
    print(f"")

    # Signal pipeline stats
    total_sig = (summary.get('signals_found', 0) +
                 summary.get('signals_blocked_consensus', 0) +
                 summary.get('signals_blocked_gate', 0))
    print(f"  SIGNAL PIPELINE:")
    print(f"    Total signals found:      {summary.get('signals_found', 0)}")
    print(f"    Blocked (consensus):     {summary.get('signals_blocked_consensus', 0)}")
    print(f"    Blocked (gates):         {summary.get('signals_blocked_gate', 0)}")
    print(f"    Executed:                {summary.get('trades_executed', 0)}")
    if total_sig > 0:
        exec_pct = summary.get('trades_executed', 0) / total_sig * 100
        print(f"    Execution rate:          {exec_pct:.1f}%")

    # Per-strategy breakdown
    strat_stats = summary.get('strategy_stats', {})
    if strat_stats:
        print(f"\n  STRATEGY BREAKDOWN:")
        print(f"  {'Strategy':<30} {'Trades':>6} {'Win%':>6} {'PnL':>10} {'AvgScore':>10}")
        print(f"  {'-'*65}")
        for name, s in sorted(strat_stats.items(), key=lambda x: x[1]['pnl'], reverse=True):
            strat_wr = (s['wins'] / s['trades'] * 100) if s['trades'] > 0 else 0
            avg_score = sum(s['scores']) / len(s['scores']) if s['scores'] else 0
            print(f"  {name:<30} {s['trades']:>6} {strat_wr:>5.1f}% "
                  f"${s['pnl']:>+9.2f} {avg_score:>9.1f}")

    print(f"{'='*60}\n")


def print_full_report(all_results: list):
    """Print summary across all symbols."""
    if not all_results:
        print("No results to report.")
        return

    print(f"\n{'#'*70}")
    print(f"  APEX TRADER — BACKTEST SUMMARY")
    print(f"  Symbols tested: {len(all_results)}")
    print(f"{'#'*70}")

    total_trades = 0
    total_wins = 0
    total_pnl = 0.0
    profitable = 0

    print(f"\n  {'Symbol':<10} {'Trades':>7} {'Win%':>7} {'P&L':>12} {'PF':>6} "
          f"{'MaxDD':>10} {'Rating':>12}")
    print(f"  {'-'*75}")

    for r in sorted(all_results, key=lambda x: x.get('total_pnl', 0), reverse=True):
        sym = r.get('symbol', '?')
        t = r.get('total_trades', 0)
        wr = r.get('win_rate', 0)
        pnl = r.get('total_pnl', 0)
        pf = r.get('profit_factor', 0)
        mdd = r.get('max_drawdown', 0)
        bal = r.get('final_balance', 0)

        if t == 0:
            print(f"  {sym:<10} {'NO TRADES':>7}")
            continue

        if wr >= 55 and pf >= 1.5 and pnl > 0:
            rating = "STRONG"
        elif wr >= 50 and pf >= 1.3:
            rating = "PROFITABLE"
        elif wr >= 45 and pf >= 1.0:
            rating = "MARGINAL"
        else:
            rating = "UNPROFITABLE"

        if pnl > 0:
            profitable += 1

        total_trades += t
        total_wins += r.get('wins', 0)
        total_pnl += pnl

        print(f"  {sym:<10} {t:>7} {wr:>6.1f}% ${pnl:>+11.2f} {pf:>5.2f} "
              f"${mdd:>9.2f} {rating:>12}")

    # Overall stats
    overall_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    print(f"  {'-'*75}")
    print(f"  {'TOTAL':<10} {total_trades:>7} {overall_wr:>6.1f}% ${total_pnl:>+11.2f}")
    print(f"  Profitable symbols: {profitable}/{len(all_results)}")

    # Recommendations
    print(f"\n  RECOMMENDATIONS:")
    print(f"  {'-'*60}")
    for r in sorted(all_results, key=lambda x: x.get('total_pnl', 0), reverse=True):
        sym = r.get('symbol', '?')
        t = r.get('total_trades', 0)
        pnl = r.get('total_pnl', 0)
        wr = r.get('win_rate', 0)

        if t == 0:
            print(f"  {sym}: No trades — strategies too strict or no setups")
        elif pnl > 0 and wr >= 55:
            print(f"  {sym}: KEEP — profitable ({wr}% WR, ${pnl:+.2f})")
        elif pnl > 0:
            print(f"  {sym}: MONITOR — marginally profitable ({wr}% WR)")
        else:
            print(f"  {sym}: DISABLE — losing ({wr}% WR, ${pnl:+.2f})")
