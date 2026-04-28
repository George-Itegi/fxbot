# =============================================================
# backtest/report.py  v2.0
# Formats and prints backtest results.
# Upgraded: per-session stats, conviction stats, partial TP stats,
#           R-multiple stats, Sharpe ratio, outcome breakdown.
# =============================================================

from core.logger import get_logger

log = get_logger(__name__)


def print_summary(summary: dict):
    """Print a formatted performance report for one symbol."""
    if not summary or summary.get('total_trades', 0) == 0:
        print(f"\n  {summary.get('symbol', '?')}: NO TRADES EXECUTED")
        _print_gate_breakdown(summary)
        return

    symbol = summary.get('symbol', '?')
    t = summary['total_trades']
    w = summary['wins']
    l = summary['losses']
    wr = summary['win_rate']
    pnl = summary['total_pnl']
    pf = summary['profit_factor']
    mdd = summary['max_drawdown']
    mdd_pct = summary.get('max_drawdown_pct', 0)
    avg_w = summary['avg_win_pips']
    avg_l = summary['avg_loss_pips']
    avg_w_r = summary.get('avg_win_r', 0)
    avg_l_r = summary.get('avg_loss_r', 0)
    ev = summary.get('ev_per_trade', 0)
    sharpe = summary.get('sharpe_ratio', 0)
    ret_pct = summary.get('return_pct', 0)
    mfe = summary.get('mfe_pips', 0)
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

    print(f"\n{'='*65}")
    print(f"  BACKTEST v2.0 RESULTS: {symbol} [{icon} {rating}]")
    print(f"{'='*65}")
    print(f"  Total Trades:    {t}")
    print(f"  Wins / Losses:   {w} / {l}")
    print(f"  Win Rate:        {wr}%")
    print(f"  Total P&L:       ${pnl:+,.2f}")
    print(f"  Return:          {ret_pct:+.2f}%")
    print(f"  Profit Factor:   {pf}")
    print(f"  Expected Value:  {ev:+.3f}R per trade")
    print(f"  Sharpe Ratio:    {sharpe}")
    print(f"  Max Drawdown:    ${mdd:,.2f} ({mdd_pct:.1f}%)")
    print(f"  Avg Win:         +{avg_w} pips ({avg_w_r:+.2f}R)")
    print(f"  Avg Loss:        {avg_l} pips ({avg_l_r:+.2f}R)")
    print(f"  Max Favorable:   +{mfe} pips")
    print(f"  Final Balance:   ${bal:,.2f}")
    print(f"  Duration:        {summary.get('elapsed_seconds', 0)}s")
    print(f"")

    # Signal pipeline stats
    _print_gate_breakdown(summary)

    # Partial TP stats
    pt = summary.get('partial_tp_stats', {})
    if pt.get('trades_triggered', 0) > 0:
        print(f"\n  PARTIAL TP STATISTICS:")
        print(f"    Trades triggered:    {pt['trades_triggered']}")
        print(f"    Wins after partial:  {pt.get('wins_after_partial', 0)}")
        print(f"    Win rate after:      {pt.get('win_rate_after_partial', 0)}%")
        print(f"    Total profit locked: ${pt.get('total_partial_profit', 0):+.2f}")

    # Outcome breakdown
    outcomes = summary.get('outcome_counts', {})
    if outcomes:
        print(f"\n  OUTCOME BREAKDOWN:")
        for outcome, count in sorted(outcomes.items(), key=lambda x: -x[1]):
            print(f"    {outcome:<25} {count:>5}")

    # Per-strategy breakdown
    strat_stats = summary.get('strategy_stats', {})
    if strat_stats:
        print(f"\n  STRATEGY BREAKDOWN:")
        print(f"  {'Strategy':<30} {'Trades':>6} {'Win%':>6} {'PnL':>10} "
              f"{'AvgR':>7} {'AvgScr':>7} {'PtrTP':>6} {'Trail':>6}")
        print(f"  {'-'*80}")
        for name, s in sorted(strat_stats.items(), key=lambda x: x[1]['pnl'], reverse=True):
            strat_wr = (s['wins'] / s['trades'] * 100) if s['trades'] > 0 else 0
            avg_score = sum(s['scores']) / len(s['scores']) if s['scores'] else 0
            print(f"  {name:<30} {s['trades']:>6} {strat_wr:>5.1f}% "
                  f"${s['pnl']:>+9.2f} {s.get('avg_r',0):>+6.2f} "
                  f"{avg_score:>6.1f} "
                  f"{s.get('partial_tp_count',0):>6} "
                  f"{s.get('trail_count',0):>6}")

    # Per-session breakdown
    session_stats = summary.get('session_stats', {})
    if session_stats:
        print(f"\n  SESSION BREAKDOWN:")
        print(f"  {'Session':<20} {'Trades':>6} {'Win%':>6} {'PnL':>10} {'Pips':>10}")
        print(f"  {'-'*55}")
        for sess, s in sorted(session_stats.items(), key=lambda x: x[1]['pnl'], reverse=True):
            sess_wr = (s['wins'] / s['trades'] * 100) if s['trades'] > 0 else 0
            print(f"  {sess:<20} {s['trades']:>6} {sess_wr:>5.1f}% "
                  f"${s['pnl']:>+9.2f} {s['pips']:>+9.1f}")

    # Per-market-state breakdown
    state_stats = summary.get('state_stats', {})
    if state_stats:
        print(f"\n  MARKET STATE BREAKDOWN:")
        print(f"  {'State':<25} {'Trades':>6} {'Win%':>6} {'PnL':>10} {'Pips':>10}")
        print(f"  {'-'*60}")
        for state, s in sorted(state_stats.items(), key=lambda x: x[1]['pnl'], reverse=True):
            state_wr = (s['wins'] / s['trades'] * 100) if s['trades'] > 0 else 0
            print(f"  {state:<25} {s['trades']:>6} {state_wr:>5.1f}% "
                  f"${s['pnl']:>+9.2f} {s['pips']:>+9.1f}")

    # Conviction breakdown
    conv_stats = summary.get('conviction_stats', {})
    if conv_stats:
        print(f"\n  CONVICTION BREAKDOWN:")
        print(f"  {'Conviction':<15} {'Trades':>6} {'Win%':>6} {'PnL':>10}")
        print(f"  {'-'*40}")
        for conv, s in sorted(conv_stats.items(), key=lambda x: x[1]['pnl'], reverse=True):
            conv_wr = (s['wins'] / s['trades'] * 100) if s['trades'] > 0 else 0
            print(f"  {conv:<15} {s['trades']:>6} {conv_wr:>5.1f}% "
                  f"${s['pnl']:>+9.2f}")

    print(f"{'='*65}\n")


def print_full_report(all_results: list):
    """Print summary across all symbols."""
    if not all_results:
        print("No results to report.")
        return

    print(f"\n{'#'*80}")
    print(f"  APEX TRADER — BACKTEST v2.0 SUMMARY")
    print(f"  Symbols tested: {len(all_results)}")
    print(f"{'#'*80}")

    total_trades = 0
    total_wins = 0
    total_pnl = 0.0
    total_partial_profit = 0.0
    profitable = 0

    print(f"\n  {'Symbol':<10} {'Trades':>7} {'Win%':>7} {'P&L':>12} {'PF':>6} "
          f"{'MaxDD':>10} {'EV/R':>7} {'Sharpe':>7} {'Rating':>12}")
    print(f"  {'-'*95}")

    for r in sorted(all_results, key=lambda x: x.get('total_pnl', 0), reverse=True):
        sym = r.get('symbol', '?')
        t = r.get('total_trades', 0)
        wr = r.get('win_rate', 0)
        pnl = r.get('total_pnl', 0)
        pf = r.get('profit_factor', 0)
        mdd = r.get('max_drawdown', 0)
        ev = r.get('ev_per_trade', 0)
        sharpe = r.get('sharpe_ratio', 0)

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
        pt = r.get('partial_tp_stats', {})
        total_partial_profit += pt.get('total_partial_profit', 0)

        print(f"  {sym:<10} {t:>7} {wr:>6.1f}% ${pnl:>+11.2f} {pf:>5.2f} "
              f"${mdd:>9.2f} {ev:>+6.3f} {sharpe:>6.2f} {rating:>12}")

    # Overall stats
    overall_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    print(f"  {'-'*95}")
    print(f"  {'TOTAL':<10} {total_trades:>7} {overall_wr:>6.1f}% "
          f"${total_pnl:>+11.2f}")
    print(f"  Profitable symbols: {profitable}/{len(all_results)}")
    print(f"  Partial TP profit (banked): ${total_partial_profit:+,.2f}")

    # Recommendations
    print(f"\n  RECOMMENDATIONS:")
    print(f"  {'-'*65}")
    for r in sorted(all_results, key=lambda x: x.get('total_pnl', 0), reverse=True):
        sym = r.get('symbol', '?')
        t = r.get('total_trades', 0)
        pnl = r.get('total_pnl', 0)
        wr = r.get('win_rate', 0)
        ev = r.get('ev_per_trade', 0)
        pt = r.get('partial_tp_stats', {})

        if t == 0:
            print(f"  {sym}: No trades — strategies too strict or no setups")
        elif pnl > 0 and wr >= 55 and ev > 0:
            print(f"  {sym}: KEEP — strong edge (WR={wr}%, EV={ev:+.3f}R, ${pnl:+.2f})")
        elif pnl > 0:
            print(f"  {sym}: MONITOR — marginally profitable (WR={wr}%, EV={ev:+.3f}R)")
        elif pnl > 0 and pt.get('total_partial_profit', 0) > abs(pnl):
            print(f"  {sym}: REVIEW — partial TP saved it (${pt.get('total_partial_profit',0):+.2f} banked)")
        else:
            print(f"  {sym}: DISABLE — losing (WR={wr}%, EV={ev:+.3f}R, ${pnl:+.2f})")

    # Key insights
    print(f"\n  KEY INSIGHTS:")
    print(f"  {'-'*65}")
    # Best strategy across all symbols
    all_strat_stats = {}
    for r in all_results:
        for name, s in r.get('strategy_stats', {}).items():
            if name not in all_strat_stats:
                all_strat_stats[name] = {'trades': 0, 'wins': 0, 'pnl': 0.0, 'scores': []}
            all_strat_stats[name]['trades'] += s['trades']
            all_strat_stats[name]['wins'] += s['wins']
            all_strat_stats[name]['pnl'] += s['pnl']
            all_strat_stats[name]['scores'].extend(s.get('scores', []))

    if all_strat_stats:
        print(f"\n  BEST STRATEGIES (across all symbols):")
        sorted_strats = sorted(all_strat_stats.items(), key=lambda x: x[1]['pnl'], reverse=True)
        for name, s in sorted_strats[:5]:
            wr = (s['wins'] / s['trades'] * 100) if s['trades'] > 0 else 0
            avg_score = sum(s['scores']) / len(s['scores']) if s['scores'] else 0
            print(f"    {name:<30} {s['trades']:>5} trades  {wr:>5.1f}% WR  "
                  f"${s['pnl']:>+9.2f}  avg_score={avg_score:.1f}")

    # Best sessions
    all_sess_stats = {}
    for r in all_results:
        for sess, s in r.get('session_stats', {}).items():
            if sess not in all_sess_stats:
                all_sess_stats[sess] = {'trades': 0, 'wins': 0, 'pnl': 0.0}
            all_sess_stats[sess]['trades'] += s['trades']
            all_sess_stats[sess]['wins'] += s['wins']
            all_sess_stats[sess]['pnl'] += s['pnl']

    if all_sess_stats:
        print(f"\n  BEST SESSIONS:")
        sorted_sess = sorted(all_sess_stats.items(), key=lambda x: x[1]['pnl'], reverse=True)
        for sess, s in sorted_sess[:5]:
            wr = (s['wins'] / s['trades'] * 100) if s['trades'] > 0 else 0
            print(f"    {sess:<25} {s['trades']:>5} trades  {wr:>5.1f}% WR  ${s['pnl']:>+9.2f}")


def _print_gate_breakdown(summary: dict):
    """Print detailed per-gate rejection breakdown."""
    if not summary:
        return

    # Gate counters (pre-strategy gates)
    g_score = summary.get('signals_blocked_score', 0)
    g_inst = summary.get('signals_blocked_gate', 0)
    g_choppy = summary.get('signals_blocked_choppy', 0)
    g_bias = summary.get('signals_blocked_bias', 0)
    g_consensus = summary.get('signals_blocked_consensus', 0)
    g_confluence = summary.get('signals_blocked_confluence', 0)
    g_no_strat = summary.get('signals_no_strategy', 0)
    g_l1_reject = summary.get('strat_model_rejected', 0)
    g_l2_skip = summary.get('model_blocked', 0)
    g_l2_caution = summary.get('model_caution', 0)
    signals_found = summary.get('signals_found', 0)
    executed = summary.get('trades_executed', 0)

    # Total pre-strategy gate blocks
    total_pre_gates = g_score + g_inst + g_choppy

    # Total post-strategy gate blocks
    total_post_gates = g_bias + g_consensus + g_confluence + g_no_strat

    # Total ML blocks
    total_ml = g_l1_reject + g_l2_skip + g_l2_caution

    # Grand total of all blocks + signals
    total_all = total_pre_gates + total_post_gates + signals_found + total_ml

    print(f"  SIGNAL PIPELINE:")
    print(f"    Scan bars evaluated:       {total_pre_gates + g_no_strat + total_post_gates + signals_found + total_ml}")

    # Pre-strategy gates
    print(f"")
    print(f"    PRE-STRATEGY GATES:")
    print(f"      Gate 0 (score < min):    {g_score:>6}")
    print(f"      Gate 1 (no institution): {g_inst:>6}  {'<<< BOTTLENECK' if g_inst > total_pre_gates * 0.6 and total_pre_gates > 0 else ''}")
    print(f"      Gate 2 (choppy market):  {g_choppy:>6}")
    print(f"      Subtotal:                 {total_pre_gates:>6}")

    # Post-strategy gates
    print(f"")
    print(f"    POST-STRATEGY GATES:")
    print(f"      No qualifying signals:    {g_no_strat:>6}")
    print(f"      Gate 3 (bias mismatch):  {g_bias:>6}")
    print(f"      Gate 4 (no consensus):   {g_consensus:>6}  {'<<< BOTTLENECK' if g_consensus > total_post_gates * 0.6 and total_post_gates > 0 else ''}")
    print(f"      Gate 5 (low confluence): {g_confluence:>6}")
    print(f"      Subtotal:                 {total_post_gates:>6}")

    # ML layer
    print(f"")
    print(f"    ML LAYERS:")
    print(f"      Signals reached ML:      {signals_found:>6}")
    print(f"      L1 Strategy REJECTED:    {g_l1_reject:>6}")
    print(f"      L2 ML Gate CAUTION:      {g_l2_caution:>6}  (shadowed, not executed)")
    print(f"      L2 ML Gate SKIP:         {g_l2_skip:>6}  (shadowed, not executed)")
    ml_survived = signals_found - g_l1_reject - g_l2_skip - g_l2_caution
    print(f"      ML Survived:             {ml_survived:>6}")
    print(f"")
    print(f"    FINAL OUTCOME:")
    print(f"      Executed:                {executed:>6}")
    if total_all > 0:
        exec_pct = executed / total_all * 100
        print(f"      Execution rate:           {exec_pct:>5.1f}%")
