# =============================================================
# ai_engine/phase_manager.py
# PURPOSE: Manages strategy lifecycle across all 3 phases.
# VIRTUAL → PAPER_TRADING → LIVE_ACTIVE → DEGRADING → RETIRED
# Checks thresholds and promotes/demotes strategies automatically.
# =============================================================

from datetime import datetime, timezone
from core.logger import get_logger
from strategies.strategy_registry import (
    REGISTRY, PHASE_VIRTUAL, PHASE_PAPER, PHASE_LIVE,
    PHASE_DEGRADING, PHASE_RETIRED,
    VIRTUAL_MIN_TRADES, VIRTUAL_MIN_WINRATE,
    PAPER_MIN_TRADES, PAPER_MIN_WINRATE,
    DEGRADING_THRESHOLD
)

log = get_logger(__name__)

# Live promotion threshold — stricter than demo
LIVE_MIN_MONTHS     = 1     # Must have 1 month of demo data
LIVE_MIN_WINRATE    = 62.0  # 62% win rate in demo
LIVE_MIN_TRADES     = 30    # At least 30 demo trades


def check_all_promotions():
    """
    Check every strategy in registry and promote/demote as needed.
    Call this after every batch of trades completes.
    """
    now = datetime.now(timezone.utc).isoformat()
    for name, info in REGISTRY.items():
        phase    = info['phase']
        wr       = info['win_rate']
        trades   = info['total_trades']

        # VIRTUAL → PAPER check
        if phase == PHASE_VIRTUAL:
            if trades >= VIRTUAL_MIN_TRADES and wr >= VIRTUAL_MIN_WINRATE:
                _promote(name, PHASE_PAPER,
                         f"Win rate {wr}% over {trades} virtual trades")
            elif trades >= VIRTUAL_MIN_TRADES and wr < VIRTUAL_MIN_WINRATE:
                _retire(name,
                        f"Virtual failed: {wr}% < {VIRTUAL_MIN_WINRATE}%")

        # PAPER → LIVE check
        elif phase == PHASE_PAPER:
            if trades >= LIVE_MIN_TRADES and wr >= LIVE_MIN_WINRATE:
                _promote(name, PHASE_LIVE,
                         f"Demo win rate {wr}% over {trades} trades")
            elif trades >= LIVE_MIN_TRADES * 2 and wr < PAPER_MIN_WINRATE:
                _demote(name, PHASE_VIRTUAL,
                        f"Demo failing: {wr}% — back to virtual")

        # LIVE → check for degradation
        elif phase == PHASE_LIVE:
            if trades >= 20 and wr < DEGRADING_THRESHOLD:
                _demote(name, PHASE_DEGRADING,
                        f"Live degrading: {wr}% < {DEGRADING_THRESHOLD}%")

        # DEGRADING → either recover or retire
        elif phase == PHASE_DEGRADING:
            if wr >= PAPER_MIN_WINRATE:
                _promote(name, PHASE_LIVE,
                         f"Recovered to {wr}%")
            elif trades >= 50 and wr < DEGRADING_THRESHOLD:
                _retire(name,
                        f"Degrading not recovering: {wr}%")

def _promote(name: str, new_phase: str, reason: str):
    """Promote a strategy to a higher phase."""
    old_phase = REGISTRY[name]['phase']
    REGISTRY[name]['phase']  = new_phase
    REGISTRY[name]['status'] = 'ACTIVE'
    now = datetime.now(timezone.utc).isoformat()
    if new_phase == PHASE_PAPER:
        REGISTRY[name]['promoted_to_paper'] = now
    elif new_phase == PHASE_LIVE:
        REGISTRY[name]['promoted_to_live'] = now
    log.info(f"[PHASE] ✅ PROMOTED: {name}"
             f" {old_phase} → {new_phase} | {reason}")
    _log_phase_change(name, old_phase, new_phase, reason)


def _demote(name: str, new_phase: str, reason: str):
    """Demote a strategy to a lower phase for retraining."""
    old_phase = REGISTRY[name]['phase']
    REGISTRY[name]['phase']        = new_phase
    REGISTRY[name]['status']       = 'ACTIVE'
    REGISTRY[name]['total_trades'] = 0
    REGISTRY[name]['wins']         = 0
    REGISTRY[name]['losses']       = 0
    REGISTRY[name]['win_rate']     = 0.0
    log.warning(f"[PHASE] ⬇️ DEMOTED: {name}"
                f" {old_phase} → {new_phase} | {reason}")
    _log_phase_change(name, old_phase, new_phase, reason)


def _retire(name: str, reason: str):
    """Retire a strategy permanently."""
    old_phase = REGISTRY[name]['phase']
    REGISTRY[name]['phase']  = PHASE_RETIRED
    REGISTRY[name]['status'] = 'RETIRED'
    log.warning(f"[PHASE] 🪦 RETIRED: {name} | {reason}")
    _log_phase_change(name, old_phase, PHASE_RETIRED, reason)


def _log_phase_change(name: str, old: str,
                      new: str, reason: str):
    """Log phase change to database."""
    try:
        from database.db_manager import get_connection
        conn = get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS phase_log (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp  TEXT,
                strategy   TEXT,
                old_phase  TEXT,
                new_phase  TEXT,
                reason     TEXT
            )
        """)
        conn.execute("""
            INSERT INTO phase_log
            (timestamp, strategy, old_phase, new_phase, reason)
            VALUES (?, ?, ?, ?, ?)
        """, (datetime.now(timezone.utc).isoformat(),
              name, old, new, reason))
        conn.commit()
        conn.close()
    except Exception as e:
        log.error(f"[PHASE] DB log failed: {e}")


def get_strategies_for_phase(phase: str) -> list:
    """Return active strategy names for a given phase."""
    return [n for n, info in REGISTRY.items()
            if info['phase'] == phase
            and info['status'] == 'ACTIVE']


def get_phase_summary() -> str:
    """Print phase summary of all strategies."""
    lines = ["\n=== PHASE MANAGER STATUS ==="]
    for name, info in REGISTRY.items():
        lines.append(
            f"  {name:<28}"
            f" | {info['phase']:<16}"
            f" | WR:{info['win_rate']:5.1f}%"
            f" | Trades:{info['total_trades']:3d}"
            f" | {info['status']}"
        )
    return "\n".join(lines)


# =============================================================
# STANDALONE TEST
# =============================================================
if __name__ == "__main__":
    print(get_phase_summary())
    print("\nChecking promotions...")
    check_all_promotions()
    print(get_phase_summary())
