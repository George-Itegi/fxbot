# =============================================================
# rpde/safety_guards.py  —  Human Safety Rails (Phase 3)
#
# NON-OVERRIDABLE safety guards that sit between the AI decision
# engine and trade execution. The AI can NEVER bypass these rules.
# They represent the human's final safety net for real capital.
#
# Guard severity:
#   HARD  → System-wide SHUTDOWN. No new trades until manual reset.
#   SOFT  → Skip this specific trade. System continues operating.
#
# Guards are checked in priority order. First failing guard returns
# immediately (no point running cheaper checks after a HARD fail).
#
# ⚠️  IMPORTANT: These rules are sacrosanct. No AI override, no
#     config flag, no "just this once" exception. They exist
#     because real money is on the line.
# =============================================================

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any

from core.logger import get_logger

log = get_logger(__name__)

# ── Config imports with safe fallbacks ──────────────────────────
# These live in rpde/config.py. We define fallback defaults here
# so the safety system is self-contained even if config is missing.
# The fallbacks are CONSERVATIVE — they protect capital first.

try:
    from rpde.config import (
        SAFETY_MAX_DAILY_LOSS_PCT,
        SAFETY_MAX_WEEKLY_LOSS_PCT,
        SAFETY_MAX_POSITIONS,
        SAFETY_MAX_PER_PAIR,
        SAFETY_MAX_CONSECUTIVE_LOSSES,
        SAFETY_MARGIN_LEVEL_MIN,
        SAFETY_EQUITY_MIN_PCT,
        SAFETY_NEWS_BUFFER_MINUTES,
        SAFETY_MEDIUM_NEWS_BUFFER_MINUTES,
        SAFETY_FRIDAY_CLOSE_HOUR,
        SAFETY_MONDAY_OPEN_MINUTE,
        SAFETY_SPREAD_MULTIPLIER,
        SAFETY_ATR_EXTREME_MULTIPLIER,
        SAFETY_COOLDOWN_AFTER_SHUTDOWN_HOURS,
        SAFETY_FREE_MARGIN_BUFFER,
        SPREAD_LIMITS,
    )
except ImportError:
    log.warning(
        "[SAFETY] rpde.config safety constants not found — "
        "using conservative built-in defaults"
    )
    SAFETY_MAX_DAILY_LOSS_PCT = 3.0
    SAFETY_MAX_WEEKLY_LOSS_PCT = 5.0
    SAFETY_MAX_POSITIONS = 5
    SAFETY_MAX_PER_PAIR = 2
    SAFETY_MAX_CONSECUTIVE_LOSSES = 5
    SAFETY_MARGIN_LEVEL_MIN = 150.0
    SAFETY_EQUITY_MIN_PCT = 50.0
    SAFETY_NEWS_BUFFER_MINUTES = 15
    SAFETY_MEDIUM_NEWS_BUFFER_MINUTES = 5
    SAFETY_FRIDAY_CLOSE_HOUR = 20
    SAFETY_MONDAY_OPEN_MINUTE = 5
    SAFETY_SPREAD_MULTIPLIER = 3.0
    SAFETY_ATR_EXTREME_MULTIPLIER = 3.0
    SAFETY_COOLDOWN_AFTER_SHUTDOWN_HOURS = 2
    SAFETY_FREE_MARGIN_BUFFER = 2.0
    SPREAD_LIMITS = {
        "EURUSD": 2.0,
        "GBPUSD": 3.0,
        "EURJPY": 3.0,
        "GBPJPY": 4.0,
        "CHFJPY": 3.5,
        "CADJPY": 4.0,
        "AUDJPY": 3.5,
        "AUDUSD": 2.0,
        "XAGUSD": 5.0,
        "AUDCAD": 4.0,
        "DEFAULT": 4.0,
    }


# ── Low-quality session windows (UTC) ──────────────────────────
# Edges of session overlaps where liquidity thins out and spreads
# widen unpredictably. Trading here is coin-flip territory.
_LOW_LIQUIDITY_WINDOWS = [
    # (start_hour_utc, end_hour_utc, description)
    (21, 22, "Sydney open — thin liquidity, weekend gaps settling"),
    (22, 23, "Sydney session — minimal institutional flow"),
    (23, 0,  "Sydney/Tokyo transition — widest spreads of the day"),
    (6,  7,  "Pre-London — Tokyo closing, manipulation phase begins"),
    (16, 17, "Post-overlap — NY lunch, liquidity drain starts"),
    (20, 21, "NY afternoon close — position squaring, erratic moves"),
]


# ================================================================
# Data structures
# ================================================================

@dataclass
class SafetyGuardResult:
    """
    Result from a single safety guard check.

    Attributes:
        passed:     True if the guard allows the trade, False otherwise.
        guard_name: Human-readable name of the guard that was evaluated.
        severity:   "SOFT" (skip this trade) or "HARD" (shut down system).
        message:    Human-readable explanation of the decision.
        metadata:   Guard-specific details for diagnostics and logging.
    """
    passed: bool
    guard_name: str
    severity: str
    message: str
    metadata: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.passed


# ================================================================
# Safety Guard System
# ================================================================

class SafetyGuardSystem:
    """
    Non-overridable safety system for trade execution.

    Every trade must pass ALL soft guards before execution.
    Hard guards trigger immediate system-wide shutdown.

    The system is thread-safe for use in multi-threaded environments
    (e.g., the execution loop running alongside position management).

    Usage:
        safety = SafetyGuardSystem()
        result = safety.check(trade_request, account_state, market_state)
        if not result.passed:
            if result.severity == "HARD":
                # Shut down the trading loop entirely
                pass
            else:
                # Skip this trade, continue scanning
                pass
    """

    # ── Per-pair position tracker ──
    # {pair: count} — maintained externally via update_positions()
    _pair_position_counts: Dict[str, int]

    def __init__(self, config_override: dict = None):
        """
        Initialize the safety guard system.

        Args:
            config_override: Optional dict to override specific config
                             values for testing. Keys must match the
                             SAFETY_* constants. NOT for production use.
        """
        self._config_override = config_override or {}

        # Guard check functions — ordered by priority.
        # HARD guards first (most critical checks), then SOFT guards.
        self._guards: List = [
            self._check_margin_call,         # ⑥ HARD — most critical, check first
            self._check_max_drawdown,        # ① HARD — capital preservation
            self._check_position_limit,      # ③ SOFT — exposure control
            self._check_spread_filter,       # ② SOFT — execution quality
            self._check_news_filter,         # ④ SOFT — event risk
            self._check_weekend_filter,      # ⑤ SOFT — session validity
            self._check_consecutive_losses,  # Extra SOFT — psychological/tilt guard
            self._check_session_quality,     # Extra SOFT — liquidity guard
            self._check_volatility_extreme,  # Extra SOFT — risk-of-ruin guard
        ]

        # Shutdown state
        self._shutdown_flag: bool = False
        self._shutdown_reason: Optional[str] = None
        self._shutdown_time: Optional[datetime] = None

        # Guard execution log (circular buffer, bounded)
        self._guard_log: List[dict] = []
        self._max_log_size: int = 500

        # Per-pair position tracking
        self._pair_position_counts: Dict[str, int] = {}

        # Thread lock for all state mutations
        self._lock = threading.RLock()

        log.info(
            "[SAFETY] Guard system initialized — "
            f"{len(self._guards)} guards active "
            f"(daily_limit={self._cfg('SAFETY_MAX_DAILY_LOSS_PCT')}%, "
            f"weekly_limit={self._cfg('SAFETY_MAX_WEEKLY_LOSS_PCT')}%, "
            f"max_positions={self._cfg('SAFETY_MAX_POSITIONS')})"
        )

    # ── Config helper ──────────────────────────────────────────

    def _cfg(self, key: str) -> Any:
        """Resolve a config value: override → rpde.config → fallback."""
        if key in self._config_override:
            return self._config_override[key]
        # Look up from module-level globals (where the imports landed)
        return globals().get(key)

    # ── Public API ─────────────────────────────────────────────

    def check(self, trade_request: dict, account_state: dict,
              market_state: dict) -> SafetyGuardResult:
        """
        Run all safety guards against a proposed trade.

        Guards are evaluated in priority order. If the system is
        already shut down, returns immediately with a HARD failure.

        Args:
            trade_request:  Proposed trade details (pair, direction, size, etc.)
            account_state:  Current account metrics (balance, equity, margin, etc.)
            market_state:   Current market conditions (spread, session, news, etc.)

        Returns:
            SafetyGuardResult with pass/fail, severity, and details.
            If passed=True, the trade is safe to execute.
        """
        with self._lock:
            # 0. Is the system already shut down?
            if self._shutdown_flag:
                result = SafetyGuardResult(
                    passed=False,
                    guard_name="system_shutdown",
                    severity="HARD",
                    message=(
                        f"System is SHUT DOWN: {self._shutdown_reason}. "
                        "Manual reset required."
                    ),
                    metadata={
                        "shutdown_reason": self._shutdown_reason,
                        "shutdown_time": (
                            self._shutdown_time.isoformat()
                            if self._shutdown_time else None
                        ),
                    },
                )
                self._record_result(result, trade_request)
                return result

            # 1. Check cooldown period after shutdown (even if reset)
            if self._shutdown_time and self._cfg("SAFETY_COOLDOWN_AFTER_SHUTDOWN_HOURS"):
                cooldown_hours = self._cfg("SAFETY_COOLDOWN_AFTER_SHUTDOWN_HOURS")
                elapsed = (datetime.now(timezone.utc) - self._shutdown_time).total_seconds() / 3600
                if elapsed < cooldown_hours:
                    remaining = cooldown_hours - elapsed
                    result = SafetyGuardResult(
                        passed=False,
                        guard_name="shutdown_cooldown",
                        severity="HARD",
                        message=(
                            f"Post-shutdown cooldown active: "
                            f"{remaining:.1f}h remaining "
                            f"(of {cooldown_hours}h cooldown period)"
                        ),
                        metadata={
                            "cooldown_hours": cooldown_hours,
                            "remaining_hours": round(remaining, 2),
                            "shutdown_time": self._shutdown_time.isoformat(),
                        },
                    )
                    self._record_result(result, trade_request)
                    return result

            # 2. Run guards in order — first fail returns immediately
            pair = str(trade_request.get("pair", "UNKNOWN")).upper()
            for guard_fn in self._guards:
                try:
                    result = guard_fn(trade_request, account_state, market_state)
                except Exception as exc:
                    # A guard crashing must NEVER block the system.
                    # Log the error and PASS the guard (fail-open for
                    # bugs, fail-closed only for intentional rules).
                    log.error(
                        f"[SAFETY] Guard {guard_fn.__name__} raised "
                        f"{type(exc).__name__}: {exc} — "
                        "FAIL-OPEN (allowing trade due to guard error)"
                    )
                    result = SafetyGuardResult(
                        passed=True,
                        guard_name=guard_fn.__name__,
                        severity="SOFT",
                        message=f"Guard error (fail-open): {exc}",
                        metadata={"error": str(exc), "error_type": type(exc).__name__},
                    )

                self._record_result(result, trade_request)

                if not result.passed:
                    # HARD → trigger shutdown, then return
                    if result.severity == "HARD":
                        self._trigger_shutdown(result.message)

                    level = log.warning if result.severity == "SOFT" else log.critical
                    level(
                        f"[SAFETY] {result.severity} BLOCK: "
                        f"{result.guard_name} — {result.message}"
                    )
                    return result

            # All guards passed
            result = SafetyGuardResult(
                passed=True,
                guard_name="all_guards",
                severity="SOFT",
                message=f"All {len(self._guards)} safety guards PASSED for {pair}",
                metadata={
                    "pair": pair,
                    "direction": trade_request.get("direction"),
                    "size_r": trade_request.get("size_r"),
                },
            )
            self._record_result(result, trade_request)
            log.debug(f"[SAFETY] ✓ All guards passed for {pair} "
                      f"{trade_request.get('direction')}")
            return result

    def is_shutdown(self) -> bool:
        """Check if the safety system has triggered a shutdown."""
        with self._lock:
            return self._shutdown_flag

    def get_shutdown_reason(self) -> Optional[str]:
        """Get the reason for the current shutdown, if any."""
        with self._lock:
            return self._shutdown_reason

    def force_shutdown(self, reason: str) -> None:
        """
        Manually trigger a system shutdown.

        This is the human emergency brake. Call this from any
        monitoring system, dashboard button, or alert handler.

        Args:
            reason: Human-readable explanation for the shutdown.
        """
        with self._lock:
            self._trigger_shutdown(reason)

    def reset_shutdown(self) -> bool:
        """
        Manually reset the shutdown flag.

        ⚠️  HUMAN ONLY — This must be called by a human operator,
        never by the AI system. The post-shutdown cooldown period
        still applies even after reset.

        Returns:
            True if shutdown was active and has been reset.
            False if no shutdown was active.
        """
        with self._lock:
            if not self._shutdown_flag:
                return False
            was_shutdown = self._shutdown_flag
            self._shutdown_flag = False
            log.warning(
                f"[SAFETY] ⚠️ MANUAL RESET: System shutdown cleared "
                f"(was: {self._shutdown_reason}). "
                f"Post-shutdown cooldown may still apply."
            )
            # Keep shutdown_reason and shutdown_time for cooldown calc
            return was_shutdown

    def get_guard_history(self, limit: int = 50) -> List[dict]:
        """
        Get the recent history of guard check results.

        Args:
            limit: Maximum number of entries to return (most recent first).

        Returns:
            List of guard check result dicts, newest first.
        """
        with self._lock:
            return list(self._guard_log[-limit:])

    def get_summary(self) -> dict:
        """
        Get aggregate statistics of guard performance.

        Returns:
            Dict with pass/fail counts per guard, total checks,
            shutdown status, and failure rates.
        """
        with self._lock:
            guard_stats: Dict[str, dict] = {}
            total = len(self._guard_log)

            for entry in self._guard_log:
                name = entry.get("guard_name", "unknown")
                if name not in guard_stats:
                    guard_stats[name] = {"passed": 0, "failed_soft": 0, "failed_hard": 0}
                if entry.get("passed"):
                    guard_stats[name]["passed"] += 1
                elif entry.get("severity") == "HARD":
                    guard_stats[name]["failed_hard"] += 1
                else:
                    guard_stats[name]["failed_soft"] += 1

            return {
                "total_checks": total,
                "is_shutdown": self._shutdown_flag,
                "shutdown_reason": self._shutdown_reason,
                "shutdown_time": (
                    self._shutdown_time.isoformat()
                    if self._shutdown_time else None
                ),
                "guard_breakdown": guard_stats,
            }

    def update_positions(self, pair: str, count: int) -> None:
        """
        Update the per-pair position counter.

        Call this whenever positions are opened or closed so
        the position limit guard has accurate data.

        Args:
            pair:   Trading pair symbol (e.g., "EURJPY").
            count:  Current number of open positions for this pair.
        """
        with self._lock:
            self._pair_position_counts[pair.upper()] = count

    # ── Internal methods ───────────────────────────────────────

    def _trigger_shutdown(self, reason: str) -> None:
        """Set the shutdown flag and log the event at CRITICAL level."""
        self._shutdown_flag = True
        self._shutdown_reason = reason
        self._shutdown_time = datetime.now(timezone.utc)
        log.critical(
            f"[SAFETY] ═══════════════════════════════════════════\n"
            f"[SAFETY] 🔴 SYSTEM SHUTDOWN TRIGGERED\n"
            f"[SAFETY] Reason: {reason}\n"
            f"[SAFETY] Time:   {self._shutdown_time.isoformat()}\n"
            f"[SAFETY] ═══════════════════════════════════════════\n"
            f"[SAFETY] ACTION REQUIRED: No new trades will be\n"
            f"[SAFETY] executed until a human performs reset_shutdown()."
        )

    def _record_result(self, result: SafetyGuardResult,
                       trade_request: dict) -> None:
        """Append a guard result to the internal log."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "guard_name": result.guard_name,
            "passed": result.passed,
            "severity": result.severity,
            "message": result.message,
            "metadata": result.metadata,
            "pair": str(trade_request.get("pair", "UNKNOWN")).upper(),
            "direction": trade_request.get("direction"),
        }
        self._guard_log.append(entry)
        # Trim to max size
        if len(self._guard_log) > self._max_log_size:
            self._guard_log = self._guard_log[-self._max_log_size:]

    # ════════════════════════════════════════════════════════════
    # HARD GUARDS — System shutdown on failure
    # ════════════════════════════════════════════════════════════

    def _check_margin_call(self, trade_request: dict,
                           account_state: dict,
                           market_state: dict) -> SafetyGuardResult:
        """
        ⑥ Margin Call Protection (HARD)

        Three tiers of margin protection:
          - margin_level < SAFETY_MARGIN_LEVEL_MIN% → SHUT DOWN
          - equity < SAFETY_EQUITY_MIN_PCT% of balance → SHUT DOWN
          - free_margin < required_margin * SAFETY_FREE_MARGIN_BUFFER → SKIP (soft)

        These thresholds are non-negotiable. If you're close to a
        margin call, the ONLY correct action is to stop trading.
        """
        equity = account_state.get("equity", 0)
        balance = account_state.get("balance", 0)
        margin = account_state.get("margin", 0)
        free_margin = account_state.get("free_margin", 0)
        margin_level = account_state.get("margin_level", 0)
        estimated_margin = trade_request.get("estimated_margin", 0)

        # Guard: Missing data — fail open (don't shut down on null data)
        if balance is None or balance <= 0:
            log.warning("[SAFETY] Margin call guard: balance missing/zero — "
                        "fail-open (allowing trade)")
            return SafetyGuardResult(
                passed=True,
                guard_name="margin_call",
                severity="HARD",
                message="Balance data missing — cannot evaluate margin safety (fail-open)",
                metadata={"data_issue": "balance missing"},
            )

        # Check 1: Margin level percentage
        # margin_level = (equity / margin) * 100
        min_margin_level = self._cfg("SAFETY_MARGIN_LEVEL_MIN")
        if margin_level is not None and margin_level > 0 and margin_level < min_margin_level:
            return SafetyGuardResult(
                passed=False,
                guard_name="margin_call",
                severity="HARD",
                message=(
                    f"Margin level CRITICAL: {margin_level:.1f}% "
                    f"< minimum {min_margin_level:.1f}%"
                ),
                metadata={
                    "check": "margin_level",
                    "margin_level": margin_level,
                    "threshold": min_margin_level,
                    "equity": equity,
                    "margin": margin,
                },
            )

        # Check 2: Equity vs balance percentage
        # If equity drops below X% of balance, we're in deep trouble
        min_equity_pct = self._cfg("SAFETY_EQUITY_MIN_PCT")
        equity_pct = (equity / balance * 100) if balance > 0 else 100
        if equity_pct < min_equity_pct:
            return SafetyGuardResult(
                passed=False,
                guard_name="margin_call",
                severity="HARD",
                message=(
                    f"Equity CRITICAL: ${equity:.2f} is {equity_pct:.1f}% "
                    f"of balance ${balance:.2f} (minimum {min_equity_pct:.1f}%)"
                ),
                metadata={
                    "check": "equity_pct",
                    "equity": equity,
                    "balance": balance,
                    "equity_pct": round(equity_pct, 2),
                    "threshold": min_equity_pct,
                },
            )

        # Check 3: Free margin vs estimated required margin (soft warning)
        buffer = self._cfg("SAFETY_FREE_MARGIN_BUFFER")
        if estimated_margin and estimated_margin > 0 and free_margin is not None:
            required_total = estimated_margin * buffer
            if free_margin < required_total:
                return SafetyGuardResult(
                    passed=False,
                    guard_name="margin_call",
                    severity="SOFT",
                    message=(
                        f"Insufficient free margin: ${free_margin:.2f} "
                        f"< ${required_total:.2f} required "
                        f"(estimated_margin ${estimated_margin:.2f} × {buffer}x buffer)"
                    ),
                    metadata={
                        "check": "free_margin_buffer",
                        "free_margin": free_margin,
                        "estimated_margin": estimated_margin,
                        "buffer": buffer,
                        "required": required_total,
                    },
                )

        return SafetyGuardResult(
            passed=True,
            guard_name="margin_call",
            severity="HARD",
            message="Margin safety checks passed",
            metadata={
                "margin_level": margin_level,
                "equity_pct": round(equity_pct, 2),
                "free_margin": free_margin,
            },
        )

    def _check_max_drawdown(self, trade_request: dict,
                            account_state: dict,
                            market_state: dict) -> SafetyGuardResult:
        """
        ① Max Drawdown Guard (HARD)

        If cumulative losses exceed configured thresholds, the system
        shuts down to prevent further damage.

        Checks:
          - Daily P&L:   loss > SAFETY_MAX_DAILY_LOSS_PCT%   → SHUT DOWN
          - Weekly P&L:  loss > SAFETY_MAX_WEEKLY_LOSS_PCT%  → SHUT DOWN

        Drawdown limits are asymmetric — the daily limit is tighter
        because a bad day can cascade into margin calls quickly.
        The weekly limit is a circuit breaker for sustained losing.
        """
        daily_pnl_pct = account_state.get("daily_pnl_pct")
        weekly_pnl_pct = account_state.get("weekly_pnl_pct")
        daily_pnl = account_state.get("daily_pnl")
        weekly_pnl = account_state.get("weekly_pnl")
        balance = account_state.get("balance", 0)

        # Daily drawdown check
        max_daily = self._cfg("SAFETY_MAX_DAILY_LOSS_PCT")
        if daily_pnl_pct is not None and daily_pnl_pct < -max_daily:
            return SafetyGuardResult(
                passed=False,
                guard_name="max_drawdown",
                severity="HARD",
                message=(
                    f"Daily drawdown EXCEEDED: {daily_pnl_pct:.2f}% "
                    f"(limit: -{max_daily}%) — "
                    f"P&L: ${daily_pnl:.2f} on ${balance:.2f} balance"
                ),
                metadata={
                    "check": "daily",
                    "daily_pnl_pct": daily_pnl_pct,
                    "daily_pnl": daily_pnl,
                    "threshold": -max_daily,
                    "balance": balance,
                },
            )

        # Weekly drawdown check
        max_weekly = self._cfg("SAFETY_MAX_WEEKLY_LOSS_PCT")
        if weekly_pnl_pct is not None and weekly_pnl_pct < -max_weekly:
            return SafetyGuardResult(
                passed=False,
                guard_name="max_drawdown",
                severity="HARD",
                message=(
                    f"Weekly drawdown EXCEEDED: {weekly_pnl_pct:.2f}% "
                    f"(limit: -{max_weekly}%) — "
                    f"P&L: ${weekly_pnl:.2f} on ${balance:.2f} balance"
                ),
                metadata={
                    "check": "weekly",
                    "weekly_pnl_pct": weekly_pnl_pct,
                    "weekly_pnl": weekly_pnl,
                    "threshold": -max_weekly,
                    "balance": balance,
                },
            )

        return SafetyGuardResult(
            passed=True,
            guard_name="max_drawdown",
            severity="HARD",
            message=(
                f"Drawdown within limits "
                f"(daily: {daily_pnl_pct:.2f}%, weekly: {weekly_pnl_pct:.2f}%)"
            ),
            metadata={
                "daily_pnl_pct": daily_pnl_pct,
                "weekly_pnl_pct": weekly_pnl_pct,
            },
        )

    # ════════════════════════════════════════════════════════════
    # SOFT GUARDS — Skip this trade, system continues
    # ════════════════════════════════════════════════════════════

    def _check_position_limit(self, trade_request: dict,
                              account_state: dict,
                              market_state: dict) -> SafetyGuardResult:
        """
        ③ Position Limit (SOFT)

        Prevents overexposure by capping total open positions and
        per-pair concentration.

        Checks:
          - Total open positions >= SAFETY_MAX_POSITIONS  → SKIP
          - Same-pair positions >= SAFETY_MAX_PER_PAIR     → SKIP
        """
        pair = str(trade_request.get("pair", "")).upper()
        open_positions = account_state.get("open_positions", 0)
        max_positions = self._cfg("SAFETY_MAX_POSITIONS")
        max_per_pair = self._cfg("SAFETY_MAX_PER_PAIR")

        # Total position count
        if open_positions >= max_positions:
            return SafetyGuardResult(
                passed=False,
                guard_name="position_limit",
                severity="SOFT",
                message=(
                    f"Max positions reached: {open_positions}/{max_positions} "
                    f"— cannot open new trade on {pair}"
                ),
                metadata={
                    "check": "total_positions",
                    "open_positions": open_positions,
                    "max_positions": max_positions,
                    "pair": pair,
                },
            )

        # Per-pair position count
        pair_count = self._pair_position_counts.get(pair, 0)
        if pair_count >= max_per_pair:
            return SafetyGuardResult(
                passed=False,
                guard_name="position_limit",
                severity="SOFT",
                message=(
                    f"Max per-pair positions: {pair_count}/{max_per_pair} "
                    f"on {pair} — already fully exposed"
                ),
                metadata={
                    "check": "per_pair",
                    "pair": pair,
                    "pair_count": pair_count,
                    "max_per_pair": max_per_pair,
                },
            )

        return SafetyGuardResult(
            passed=True,
            guard_name="position_limit",
            severity="SOFT",
            message=f"Position limits OK: {open_positions}/{max_positions} total, "
                    f"{pair_count}/{max_per_pair} on {pair}",
            metadata={
                "open_positions": open_positions,
                "pair": pair,
                "pair_count": pair_count,
            },
        )

    def _check_spread_filter(self, trade_request: dict,
                             account_state: dict,
                             market_state: dict) -> SafetyGuardResult:
        """
        ② Spread Filter (SOFT)

        Two spread checks to prevent execution during illiquid periods:

        1. Static: current spread > max_allowed_spread for the pair
        2. Dynamic: current spread > SAFETY_SPREAD_MULTIPLIER × average_spread

        Wide spreads eat into profits and indicate low liquidity.
        Never enter a trade when the market is thin — the fill
        will be worse than expected and slippage will be brutal.
        """
        pair = str(trade_request.get("pair", "")).upper()
        current_spread = market_state.get("current_spread")
        average_spread = market_state.get("average_spread")

        # Missing spread data — fail open
        if current_spread is None:
            log.warning(f"[SAFETY] Spread filter: no spread data for {pair} — fail-open")
            return SafetyGuardResult(
                passed=True,
                guard_name="spread_filter",
                severity="SOFT",
                message=f"No spread data for {pair} — fail-open",
                metadata={"pair": pair, "data_issue": "current_spread missing"},
            )

        # Static spread check — per-pair limit from config
        spread_limits = self._cfg("SPREAD_LIMITS") or {}
        max_spread = spread_limits.get(pair, spread_limits.get("DEFAULT", 4.0))

        if current_spread > max_spread:
            return SafetyGuardResult(
                passed=False,
                guard_name="spread_filter",
                severity="SOFT",
                message=(
                    f"Spread too wide on {pair}: {current_spread:.1f} pips "
                    f"> max {max_spread:.1f} pips"
                ),
                metadata={
                    "check": "static",
                    "pair": pair,
                    "current_spread": current_spread,
                    "max_spread": max_spread,
                },
            )

        # Dynamic spread check — relative to recent average
        spread_multiplier = self._cfg("SAFETY_SPREAD_MULTIPLIER")
        if average_spread is not None and average_spread > 0:
            dynamic_limit = average_spread * spread_multiplier
            if current_spread > dynamic_limit:
                return SafetyGuardResult(
                    passed=False,
                    guard_name="spread_filter",
                    severity="SOFT",
                    message=(
                        f"Spread abnormal on {pair}: {current_spread:.1f} pips "
                        f"> {spread_multiplier}x average ({average_spread:.1f} pips = "
                        f"{dynamic_limit:.1f} pips limit)"
                    ),
                    metadata={
                        "check": "dynamic",
                        "pair": pair,
                        "current_spread": current_spread,
                        "average_spread": average_spread,
                        "multiplier": spread_multiplier,
                        "dynamic_limit": dynamic_limit,
                    },
                )

        return SafetyGuardResult(
            passed=True,
            guard_name="spread_filter",
            severity="SOFT",
            message=f"Spread OK on {pair}: {current_spread:.1f} pips "
                    f"(max: {max_spread:.1f})",
            metadata={
                "pair": pair,
                "current_spread": current_spread,
                "max_spread": max_spread,
                "average_spread": average_spread,
            },
        )

    def _check_news_filter(self, trade_request: dict,
                           account_state: dict,
                           market_state: dict) -> SafetyGuardResult:
        """
        ④ News Filter (SOFT)

        Prevents trading around scheduled high-impact events where
        spreads explode, slippage is extreme, and direction is random.

        Checks:
          - High-impact event within SAFETY_NEWS_BUFFER_MINUTES → SKIP
          - Medium-impact event within SAFETY_MEDIUM_NEWS_BUFFER_MINUTES → SKIP

        If no news data is available (next_high_impact_news_minutes is
        None), the guard PASSES — we don't block trading just because
        we don't have a news feed connected.
        """
        pair = str(trade_request.get("pair", "")).upper()
        high_impact_mins = market_state.get("next_high_impact_news_minutes")
        medium_impact_mins = market_state.get("next_medium_impact_news_minutes")

        # High-impact news check
        news_buffer = self._cfg("SAFETY_NEWS_BUFFER_MINUTES")
        if high_impact_mins is not None and high_impact_mins <= news_buffer:
            return SafetyGuardResult(
                passed=False,
                guard_name="news_filter",
                severity="SOFT",
                message=(
                    f"High-impact news in {high_impact_mins:.0f} min "
                    f"(buffer: {news_buffer} min) — skipping {pair}"
                ),
                metadata={
                    "check": "high_impact",
                    "pair": pair,
                    "minutes_to_news": high_impact_mins,
                    "buffer_minutes": news_buffer,
                },
            )

        # Medium-impact news check (stricter buffer)
        medium_buffer = self._cfg("SAFETY_MEDIUM_NEWS_BUFFER_MINUTES")
        if medium_impact_mins is not None and medium_impact_mins <= medium_buffer:
            return SafetyGuardResult(
                passed=False,
                guard_name="news_filter",
                severity="SOFT",
                message=(
                    f"Medium-impact news in {medium_impact_mins:.0f} min "
                    f"(buffer: {medium_buffer} min) — skipping {pair}"
                ),
                metadata={
                    "check": "medium_impact",
                    "pair": pair,
                    "minutes_to_news": medium_impact_mins,
                    "buffer_minutes": medium_buffer,
                },
            )

        return SafetyGuardResult(
            passed=True,
            guard_name="news_filter",
            severity="SOFT",
            message=f"No imminent news events for {pair}",
            metadata={
                "pair": pair,
                "next_high_impact_mins": high_impact_mins,
                "next_medium_impact_mins": medium_impact_mins,
            },
        )

    def _check_weekend_filter(self, trade_request: dict,
                              account_state: dict,
                              market_state: dict) -> SafetyGuardResult:
        """
        ⑤ Weekend Filter (SOFT)

        Forex markets close over the weekend and reopen with gap risk.
        Trading near close or during weekend is dangerous.

        Rules:
          - Friday 20:00+ UTC → No new trades (markets thinning)
          - Saturday/Sunday  → No trading (markets closed)
          - Monday 00:00-00:05 UTC → No trading (spread too wide on open)

        Also respects market_state['is_weekend'] if provided.
        """
        hour_utc = market_state.get("hour_utc")
        day_of_week = market_state.get("day_of_week")
        is_weekend = market_state.get("is_weekend")

        # Use explicit weekend flag if available
        if is_weekend is True:
            return SafetyGuardResult(
                passed=False,
                guard_name="weekend_filter",
                severity="SOFT",
                message="Weekend — markets are closed or closing",
                metadata={"check": "is_weekend_flag", "is_weekend": True},
            )

        # Derive from current time if not provided in market_state
        if hour_utc is None or day_of_week is None:
            now = datetime.now(timezone.utc)
            hour_utc = hour_utc if hour_utc is not None else now.hour
            day_of_week = day_of_week if day_of_week is not None else now.weekday()

        friday_close_hour = self._cfg("SAFETY_FRIDAY_CLOSE_HOUR")
        monday_open_minute = self._cfg("SAFETY_MONDAY_OPEN_MINUTE")

        # Friday late trading — no new positions after Friday close
        # Monday=0, Tuesday=1, ..., Friday=4, Saturday=5, Sunday=6
        if day_of_week == 4 and hour_utc >= friday_close_hour:  # Friday
            return SafetyGuardResult(
                passed=False,
                guard_name="weekend_filter",
                severity="SOFT",
                message=(
                    f"Friday after {friday_close_hour}:00 UTC — "
                    f"weekend approaching, no new trades"
                ),
                metadata={
                    "check": "friday_close",
                    "day_of_week": "Friday",
                    "hour_utc": hour_utc,
                    "threshold_hour": friday_close_hour,
                },
            )

        # Saturday — markets fully closed
        if day_of_week == 5:  # Saturday
            return SafetyGuardResult(
                passed=False,
                guard_name="weekend_filter",
                severity="SOFT",
                message="Saturday — forex markets are closed",
                metadata={"check": "saturday", "day_of_week": "Saturday"},
            )

        # Sunday — markets still closed (forex opens Sunday ~22:00 UTC)
        if day_of_week == 6:  # Sunday
            return SafetyGuardResult(
                passed=False,
                guard_name="weekend_filter",
                severity="SOFT",
                message="Sunday — forex markets are closed (open ~22:00 UTC)",
                metadata={"check": "sunday", "day_of_week": "Sunday"},
            )

        # Monday early — spreads are absurdly wide on open
        if day_of_week == 0 and hour_utc == 0:
            now = datetime.now(timezone.utc)
            if now.minute < monday_open_minute:
                return SafetyGuardResult(
                    passed=False,
                    guard_name="weekend_filter",
                    severity="SOFT",
                    message=(
                        f"Monday open — waiting {monday_open_minute} min "
                        f"for spreads to normalize"
                    ),
                    metadata={
                        "check": "monday_open",
                        "day_of_week": "Monday",
                        "hour_utc": 0,
                        "minute_utc": now.minute,
                        "wait_until_minute": monday_open_minute,
                    },
                )

        return SafetyGuardResult(
            passed=True,
            guard_name="weekend_filter",
            severity="SOFT",
            message="Weekend filter passed — within trading hours",
            metadata={
                "day_of_week": day_of_week,
                "hour_utc": hour_utc,
            },
        )

    def _check_consecutive_losses(self, trade_request: dict,
                                  account_state: dict,
                                  market_state: dict) -> SafetyGuardResult:
        """
        Consecutive Losses Guard (SOFT)

        After SAFETY_MAX_CONSECUTIVE_LOSSES losses in a row, skip
        the next signal. This prevents tilt-trading and gives the
        system (and the human) a chance to cool down.

        The consecutive_losses counter in account_state tracks the
        current streak. A single win resets it to 0.

        This is a psychological guard as much as a financial one —
        after 5 losses, the edge is likely gone (regime change,
        spread manipulation, or just bad luck). Either way, wait.
        """
        consecutive_losses = account_state.get("consecutive_losses", 0)
        max_consecutive = self._cfg("SAFETY_MAX_CONSECUTIVE_LOSSES")
        pair = str(trade_request.get("pair", "")).upper()

        if consecutive_losses >= max_consecutive:
            return SafetyGuardResult(
                passed=False,
                guard_name="consecutive_losses",
                severity="SOFT",
                message=(
                    f"Consecutive loss cooldown: {consecutive_losses} losses "
                    f"(limit: {max_consecutive}) — skipping {pair}"
                ),
                metadata={
                    "pair": pair,
                    "consecutive_losses": consecutive_losses,
                    "threshold": max_consecutive,
                },
            )

        return SafetyGuardResult(
            passed=True,
            guard_name="consecutive_losses",
            severity="SOFT",
            message=f"Consecutive losses: {consecutive_losses} "
                    f"(limit: {max_consecutive}) — OK",
            metadata={
                "consecutive_losses": consecutive_losses,
                "threshold": max_consecutive,
            },
        )

    def _check_session_quality(self, trade_request: dict,
                               account_state: dict,
                               market_state: dict) -> SafetyGuardResult:
        """
        Session Quality Guard (SOFT)

        Avoids trading during low-liquidity session windows where
        spreads widen, fills are poor, and institutional flow is absent.

        Known low-quality windows:
          - Sydney open (21:00-23:00 UTC): Thin, weekend gaps settling
          - Pre-London (06:00-07:00 UTC): Tokyo closing, manipulation prep
          - Post-overlap (16:00-17:00 UTC): NY lunch, liquidity drain
          - NY afternoon close (20:00-21:00 UTC): Squaring, erratic

        If session data is not available, the guard passes (fail-open).
        """
        session = market_state.get("session")
        hour_utc = market_state.get("hour_utc")
        pair = str(trade_request.get("pair", "")).upper()

        # If no time data, pass (we can't evaluate without it)
        if hour_utc is None:
            now = datetime.now(timezone.utc)
            hour_utc = now.hour

        # Check against known low-liquidity windows
        for window_start, window_end, description in _LOW_LIQUIDITY_WINDOWS:
            if window_start <= window_end:
                # Normal range (e.g., 21-22)
                in_window = window_start <= hour_utc < window_end
            else:
                # Wraparound midnight (e.g., 23-0)
                in_window = hour_utc >= window_start or hour_utc < window_end

            if in_window:
                return SafetyGuardResult(
                    passed=False,
                    guard_name="session_quality",
                    severity="SOFT",
                    message=(
                        f"Low-liquidity window ({hour_utc}:00 UTC): "
                        f"{description} — skipping {pair}"
                    ),
                    metadata={
                        "pair": pair,
                        "hour_utc": hour_utc,
                        "session": session,
                        "window_description": description,
                        "window_start": window_start,
                        "window_end": window_end,
                    },
                )

        return SafetyGuardResult(
            passed=True,
            guard_name="session_quality",
            severity="SOFT",
            message=f"Session quality OK: {session or 'unknown'} at {hour_utc}:00 UTC",
            metadata={
                "session": session,
                "hour_utc": hour_utc,
            },
        )

    def _check_volatility_extreme(self, trade_request: dict,
                                  account_state: dict,
                                  market_state: dict) -> SafetyGuardResult:
        """
        Volatility Extreme Guard (SOFT)

        If current ATR is > SAFETY_ATR_EXTREME_MULTIPLIER × average ATR,
        the market is in an extreme volatility regime. This means:
          - Stop losses are more likely to be hit by noise
          - Position sizing calculations are inaccurate
          - The market may be in crisis mode (flash crash, etc.)

        Extreme volatility is NOT the same as a trending market.
        It's when price is moving erratically with no clear direction.

        If ATR data is unavailable, the guard passes (fail-open).
        """
        pair = str(trade_request.get("pair", "")).upper()
        atr = market_state.get("atr")
        average_atr = market_state.get("average_atr")
        multiplier = self._cfg("SAFETY_ATR_EXTREME_MULTIPLIER")

        # Missing ATR data — fail open
        if atr is None or average_atr is None or average_atr <= 0:
            log.debug(f"[SAFETY] Volatility guard: no ATR data for {pair} — fail-open")
            return SafetyGuardResult(
                passed=True,
                guard_name="volatility_extreme",
                severity="SOFT",
                message=f"No ATR data for {pair} — fail-open",
                metadata={"pair": pair, "data_issue": "atr missing"},
            )

        atr_ratio = atr / average_atr
        if atr_ratio > multiplier:
            return SafetyGuardResult(
                passed=False,
                guard_name="volatility_extreme",
                severity="SOFT",
                message=(
                    f"Extreme volatility on {pair}: "
                    f"ATR {atr:.5f} is {atr_ratio:.1f}x the average "
                    f"({average_atr:.5f}) — threshold {multiplier}x"
                ),
                metadata={
                    "pair": pair,
                    "atr": atr,
                    "average_atr": average_atr,
                    "atr_ratio": round(atr_ratio, 2),
                    "threshold": multiplier,
                },
            )

        return SafetyGuardResult(
            passed=True,
            guard_name="volatility_extreme",
            severity="SOFT",
            message=f"Volatility normal on {pair}: "
                    f"ATR {atr:.5f} is {atr_ratio:.1f}x average "
                    f"(threshold: {multiplier}x)",
            metadata={
                "pair": pair,
                "atr": atr,
                "average_atr": average_atr,
                "atr_ratio": round(atr_ratio, 2),
            },
        )


# ================================================================
# Module-level convenience function
# ================================================================

# Singleton instance for one-shot checks (stateless convenience)
_default_instance: Optional[SafetyGuardSystem] = None


def _get_default_instance() -> SafetyGuardSystem:
    """Lazy-init a shared SafetyGuardSystem for convenience functions."""
    global _default_instance
    if _default_instance is None:
        _default_instance = SafetyGuardSystem()
    return _default_instance


def check_trade_safety(trade_request: dict,
                       account_state: dict,
                       market_state: dict) -> dict:
    """
    One-shot safety check without managing a SafetyGuardSystem instance.

    Uses a shared singleton internally. For production systems that
    need state tracking (shutdown flags, guard history, position
    tracking), use SafetyGuardSystem directly instead.

    Args:
        trade_request:  Proposed trade details.
        account_state:  Current account metrics.
        market_state:   Current market conditions.

    Returns:
        Dict with keys:
          - "approved":  bool — True if trade is safe to execute
          - "reason":    str  — Human-readable explanation
          - "severity":  str  — "SOFT" or "HARD"
          - "guard":     str  — Name of the guard that decided
          - "metadata":  dict — Guard-specific details

    Example:
        >>> result = check_trade_safety(trade_req, acct_state, mkt_state)
        >>> if not result["approved"]:
        ...     print(f"Blocked by {result['guard']}: {result['reason']}")
    """
    instance = _get_default_instance()
    result = instance.check(trade_request, account_state, market_state)
    return {
        "approved": result.passed,
        "reason": result.message,
        "severity": result.severity,
        "guard": result.guard_name,
        "metadata": result.metadata,
    }


def is_system_shutdown() -> bool:
    """
    Quick check if the safety system has triggered a shutdown.

    Returns:
        True if the system is in shutdown state.
    """
    return _get_default_instance().is_shutdown()


def force_safety_shutdown(reason: str) -> None:
    """
    Emergency shutdown trigger. Call from monitoring/alerting systems.

    Args:
        reason: Human-readable explanation for the shutdown.
    """
    _get_default_instance().force_shutdown(reason)
