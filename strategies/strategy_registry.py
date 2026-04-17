# =============================================================
# strategies/strategy_registry.py
# Tracks every strategy's lifecycle, performance, and phase.
# VIRTUAL → PAPER_TRADING → LIVE_ACTIVE → DEGRADING → RETIRED
#
# Session names aligned with institutional market behaviors:
#   SYDNEY            21:00-00:00 UTC  (Price Discovery)
#   TOKYO             00:00-07:00 UTC  (Accumulation)
#   LONDON_OPEN       07:00-08:00 UTC  (Manipulation)
#   LONDON_SESSION    08:00-12:00 UTC  (Expansion)
#   NY_LONDON_OVERLAP 12:00-16:00 UTC  (Distribution)
#   NY_AFTERNOON      16:00-21:00 UTC  (Late Distribution)
# =============================================================

from datetime import datetime

# Strategy lifecycle phases
PHASE_VIRTUAL      = "VIRTUAL"
PHASE_PAPER        = "PAPER_TRADING"
PHASE_LIVE         = "LIVE_ACTIVE"
PHASE_DEGRADING    = "DEGRADING"
PHASE_RETIRED      = "RETIRED"

# Promotion thresholds
VIRTUAL_MIN_TRADES = 50
VIRTUAL_MIN_WINRATE= 58.0
PAPER_MIN_TRADES   = 30
PAPER_MIN_WINRATE  = 60.0
DEGRADING_THRESHOLD= 45.0  # Win rate below this = degrading

# Registry — all known strategies
REGISTRY = {
    "EMA_TREND_MTF": {
        "name":        "EMA Trend Multi-Timeframe",
        "file":        "strategies/ema_trend.py",
        "version":     "2.0",
        "phase":       PHASE_PAPER,
        "status":      "ACTIVE",
        "best_state":  ["TRENDING_STRONG", "BREAKOUT_ACCEPTED"],
        "best_session":["LONDON_SESSION", "NY_LONDON_OVERLAP", "NY_AFTERNOON", "LONDON_OPEN"],
        "total_trades": 0,
        "wins":        0,
        "losses":      0,
        "total_pnl":   0.0,
        "win_rate":    0.0,
        "created_at":  "2026-04-14",
        "promoted_to_paper": "2026-04-14",
        "promoted_to_live":  None,
        "notes":       "Expansion/Distribution — ride the daily trend (London+NY)",
    },
    "SMC_OB_REVERSAL": {
        "name":        "SMC Order Block Reversal",
        "file":        "strategies/smc_ob_reversal.py",
        "version":     "1.0",
        "phase":       PHASE_PAPER,
        "status":      "ACTIVE",
        "best_state":  ["TRENDING_STRONG"],
        "best_session":["LONDON_OPEN", "LONDON_SESSION", "NY_LONDON_OVERLAP", "NY_AFTERNOON"],
        "total_trades": 0,
        "wins":        0,
        "losses":      0,
        "total_pnl":   0.0,
        "win_rate":    0.0,
        "created_at":  "2026-04-14",
        "promoted_to_paper": "2026-04-14",
        "promoted_to_live":  None,
        "notes":       "Manipulation — catches Judas Swing reversals at London open",
    },
    "LIQUIDITY_SWEEP_ENTRY": {
        "name":        "Liquidity Sweep Entry",
        "file":        "strategies/liquidity_sweep_entry.py",
        "version":     "1.0",
        "phase":       PHASE_PAPER,
        "status":      "ACTIVE",
        "best_state":  ["BREAKOUT_ACCEPTED"],
        "best_session":["LONDON_OPEN", "LONDON_SESSION", "NY_LONDON_OVERLAP", "NY_AFTERNOON"],
        "total_trades": 0,
        "wins":        0,
        "losses":      0,
        "total_pnl":   0.0,
        "win_rate":    0.0,
        "created_at":  "2026-04-14",
        "promoted_to_paper": "2026-04-14",
        "promoted_to_live":  None,
        "notes":       "Manipulation — stop hunt reversal after Judas Swing",
    },
    "VWAP_MEAN_REVERSION": {
        "name":        "VWAP Mean Reversion",
        "file":        "strategies/vwap_mean_reversion.py",
        "version":     "1.0",
        "phase":       PHASE_PAPER,
        "status":      "ACTIVE",
        "best_state":  ["BALANCED"],
        "best_session":["TOKYO", "LONDON_OPEN", "LONDON_SESSION", "NY_AFTERNOON"],
        "total_trades": 0,
        "wins":        0,
        "losses":      0,
        "total_pnl":   0.0,
        "win_rate":    0.0,
        "created_at":  "2026-04-14",
        "promoted_to_paper": "2026-04-14",
        "promoted_to_live":  None,
        "notes":       "Accumulation — range-bound mean reversion (Tokyo/range markets)",
    },
    "ORDER_FLOW_EXHAUSTION": {
        "name":        "Order Flow Exhaustion",
        "file":        "strategies/order_flow_exhaustion.py",
        "version":     "1.0",
        "phase":       PHASE_VIRTUAL,
        "status":      "ACTIVE",
        "best_state":  ["BREAKOUT_REJECTED", "REVERSAL_RISK"],
        "best_session":["LONDON_OPEN", "LONDON_SESSION", "NY_LONDON_OVERLAP", "NY_AFTERNOON", "TOKYO"],
        "total_trades": 0,
        "wins":        0,
        "losses":      0,
        "total_pnl":   0.0,
        "win_rate":    0.0,
        "created_at":  "2026-04-16",
        "promoted_to_paper": None,
        "promoted_to_live":  None,
        "notes":       "Distribution — catches flow exhaustion at NY liquidation",
    },
    "M1_MOMENTUM_SCALP": {
        "name":        "M1 Momentum Scalp",
        "file":        "strategies/m1_momentum_scalp.py",
        "version":     "1.0",
        "phase":       PHASE_PAPER,
        "status":      "ACTIVE",
        "best_state":  ["TRENDING_STRONG", "BREAKOUT_ACCEPTED"],
        "best_session":["LONDON_SESSION", "NY_LONDON_OVERLAP", "NY_AFTERNOON", "LONDON_OPEN"],
        "total_trades": 0,
        "wins":        0,
        "losses":      0,
        "total_pnl":   0.0,
        "win_rate":    0.0,
        "created_at":  "2026-04-17",
        "promoted_to_paper": "2026-04-17",
        "promoted_to_live":  None,
        "notes":       "Expansion/Distribution — momentum scalp on institutional moves (London+NY)",
    },
    "OPENING_RANGE_BREAKOUT": {
        "name":        "Opening Range Breakout",
        "file":        "strategies/opening_range_breakout.py",
        "version":     "1.0",
        "phase":       PHASE_PAPER,
        "status":      "ACTIVE",
        "best_state":  ["TRENDING_STRONG", "BALANCED"],
        "best_session":["LONDON_OPEN", "LONDON_SESSION", "NY_LONDON_OVERLAP"],
        "total_trades": 0,
        "wins":        0,
        "losses":      0,
        "total_pnl":   0.0,
        "win_rate":    0.0,
        "created_at":  "2026-04-17",
        "promoted_to_paper": "2026-04-17",
        "promoted_to_live":  None,
        "notes":       "Manipulation/Expansion — ORB on London open range, retest entry",
    },
    "DELTA_DIVERGENCE": {
        "name":        "Delta Divergence",
        "file":        "strategies/delta_divergence.py",
        "version":     "1.0",
        "phase":       PHASE_PAPER,
        "status":      "ACTIVE",
        "best_state":  ["BREAKOUT_REJECTED", "REVERSAL_RISK", "BALANCED"],
        "best_session":["LONDON_OPEN", "LONDON_SESSION", "NY_LONDON_OVERLAP", "NY_AFTERNOON"],
        "total_trades": 0,
        "wins":        0,
        "losses":      0,
        "total_pnl":   0.0,
        "win_rate":    0.0,
        "created_at":  "2026-04-17",
        "promoted_to_paper": "2026-04-17",
        "promoted_to_live":  None,
        "notes":       "Manipulation/Distribution — catches fake breakouts and trapped retail",
    },
    "TREND_CONTINUATION": {
        "name":        "Trend Continuation",
        "file":        "strategies/trend_continuation.py",
        "version":     "1.0",
        "phase":       PHASE_PAPER,
        "status":      "ACTIVE",
        "best_state":  ["TRENDING_STRONG", "BREAKOUT_ACCEPTED"],
        "best_session":["LONDON_SESSION", "NY_LONDON_OVERLAP", "LONDON_OPEN", "NY_AFTERNOON"],
        "total_trades": 0,
        "wins":        0,
        "losses":      0,
        "total_pnl":   0.0,
        "win_rate":    0.0,
        "created_at":  "2026-04-17",
        "promoted_to_paper": "2026-04-17",
        "promoted_to_live":  None,
        "notes":       "Expansion — rides London/NY trend after pullback to EMA (H4+H1+M15+M5)",
    },
    "SMART_MONEY_FOOTPRINT": {
        "name":        "Smart Money Footprint",
        "file":        "strategies/smart_money_footprint.py",
        "version":     "1.0",
        "phase":       PHASE_PAPER,
        "status":      "ACTIVE",
        "best_state":  ["TRENDING_STRONG", "BREAKOUT_REJECTED", "REVERSAL_RISK", "BALANCED"],
        "best_session":["LONDON_SESSION", "NY_LONDON_OVERLAP", "NY_SESSION"],
        "total_trades": 0,
        "wins":        0,
        "losses":      0,
        "total_pnl":   0.0,
        "win_rate":    0.0,
        "created_at":  "2026-04-17",
        "promoted_to_paper": "2026-04-17",
        "promoted_to_live":  None,
        "notes":       "Institutional Alpha — order flow divergence, absorption, stop hunts, volume nodes",
    },
}


def get_active_strategies(phase: str = None) -> list:
    """Return list of active strategy names, optionally filtered by phase."""
    result = []
    for name, info in REGISTRY.items():
        if info['status'] != 'ACTIVE':
            continue
        if phase and info['phase'] != phase:
            continue
        result.append(name)
    return result


def update_performance(strategy_name: str,
                       won: bool, pnl: float):
    """Update win/loss stats after a trade closes."""
    if strategy_name not in REGISTRY:
        return
    s = REGISTRY[strategy_name]
    s['total_trades'] += 1
    if won:
        s['wins'] += 1
    else:
        s['losses'] += 1
    s['total_pnl']  = round(s['total_pnl'] + pnl, 2)
    s['win_rate']   = round(s['wins'] / s['total_trades'] * 100, 1)

    # Auto-detect degrading
    if s['total_trades'] >= 20 and s['win_rate'] < DEGRADING_THRESHOLD:
        s['phase']  = PHASE_DEGRADING
        s['status'] = 'DEGRADING'
        print(f"[REGISTRY] ⚠️ {strategy_name} DEGRADING — "
              f"win rate {s['win_rate']}%")


def get_summary() -> str:
    """Print a clean summary of all strategies."""
    lines = ["\n=== STRATEGY REGISTRY ==="]
    for name, s in REGISTRY.items():
        lines.append(
            f"  {name:<28} | Phase: {s['phase']:<14}"
            f" | WR: {s['win_rate']:5.1f}%"
            f" | Trades: {s['total_trades']:3d}"
            f" | PnL: ${s['total_pnl']:+.2f}"
        )
    lines.append("=" * 55)
    return "\n".join(lines)
