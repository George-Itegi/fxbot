"""
Risk Manager v4
================
The most important module in the entire bot.
Risk management is what separates surviving bots from blown accounts.

Key changes from v3:
- Circuit breaker auto-resets after 90 seconds
- Raised MAX_CONSECUTIVE_LOSSES to 7 (was 5)
- Gentler stake reduction during losing streaks (15% per loss, not 50%)
- Bankroll is updated from real Deriv demo account balance
"""

import time
from dataclasses import dataclass
from typing import Optional

from config import (MAX_BANKROLL_PER_TRADE, MAX_DAILY_LOSS,
                    MAX_CONSECUTIVE_LOSSES, MAX_OPEN_POSITIONS,
                    SESSION_TIME_LIMIT_MINUTES, MIN_STAKE,
                    CIRCUIT_BREAKER_COOLDOWN_SEC)
from trading.signal_generator import Signal
from utils.logger import setup_logger

logger = setup_logger("trading.risk_manager")


@dataclass
class RiskDecision:
    """Result of a risk check."""
    approved: bool
    reason: str
    adjusted_stake: float
    checks: dict


class RiskManager:
    """
    Gate keeper for all trades.

    Hard rules (non-negotiable):
    - Max 2% of bankroll per trade
    - Max 10% daily loss -> halt trading
    - 7 consecutive losses -> circuit breaker (auto-resets after 90s)
    - Only 1 open position at a time
    - Max 8-hour session

    Soft rules (auto-adjust instead of block):
    - Position size too large -> reduce to max allowed
    - Losing streak -> reduce stake by 15% per loss (not 50%)

    Circuit Breaker v4:
    - Activates after MAX_CONSECUTIVE_LOSSES (7)
    - AUTO-RESETS after 90 seconds
    - After reset, consecutive_losses is reset to 0
    - This prevents permanent blocking while still protecting against blowouts
    """

    # Stake reduction per consecutive loss (15% per loss, not 50%)
    STAKE_REDUCTION_PER_LOSS = 0.15

    def __init__(self, initial_bankroll: float):
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll

        # Session tracking
        self.session_start = time.time()
        self._session_active = True
        self._shutdown_done = False

        # Daily tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self._last_day = time.strftime("%Y-%m-%d")

        # Streak tracking
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.max_consecutive_losses = 0

        # Position tracking
        self.open_positions = 0
        self.total_trades = 0

        # State flags
        self.circuit_breaker_active = False
        self._circuit_breaker_activated_at: float = 0.0
        self.daily_loss_limit_hit = False
        self.session_expired = False
        self.model_drift_active = False

        logger.info(f"RiskManager initialized: bankroll=${initial_bankroll:.2f}")

    def update_bankroll(self, balance: float):
        """
        Update bankroll from real Deriv account balance.
        Called when we receive balance updates from the API.
        """
        old = self.bankroll
        self.bankroll = balance
        if abs(old - balance) > 0.01:
            logger.info(f"Bankroll updated: ${old:.2f} -> ${balance:.2f} (from Deriv API)")

    def can_trade(self, signal: Signal) -> RiskDecision:
        """
        MASTER GATE - all trades must pass through here.

        Position size is a SOFT rule:
        - If stake > max_allowed: auto-adjust down to max_allowed
        - Only block if adjusted stake < MIN_STAKE

        Circuit breaker auto-resets after cooldown period.
        """
        # ─── Auto-reset circuit breaker ───
        if self.circuit_breaker_active:
            elapsed = time.time() - self._circuit_breaker_activated_at
            if elapsed >= CIRCUIT_BREAKER_COOLDOWN_SEC:
                self.circuit_breaker_active = False
                self.consecutive_losses = 0
                logger.info(
                    f"Circuit breaker AUTO-RESET after {elapsed:.0f}s cooldown. "
                    f"Resuming trading."
                )

        checks = {}
        reasons = []

        # ─── 1. Circuit Breaker Check ───
        checks["circuit_breaker"] = not self.circuit_breaker_active
        if not checks["circuit_breaker"]:
            remaining = CIRCUIT_BREAKER_COOLDOWN_SEC - (time.time() - self._circuit_breaker_activated_at)
            reasons.append(f"Circuit breaker active ({remaining:.0f}s remaining)")

        # ─── 2. Daily Loss Limit ───
        checks["daily_loss_limit"] = not self.daily_loss_limit_hit
        if not checks["daily_loss_limit"]:
            reasons.append(f"Daily loss limit hit (${self.daily_pnl:.2f})")

        # ─── 3. Session Time ───
        session_minutes = (time.time() - self.session_start) / 60
        # 0 = unlimited session (demo training mode)
        if SESSION_TIME_LIMIT_MINUTES > 0:
            self.session_expired = session_minutes >= SESSION_TIME_LIMIT_MINUTES
        else:
            self.session_expired = False
        checks["session_time"] = not self.session_expired
        if not checks["session_time"]:
            reasons.append(f"Session expired ({session_minutes:.0f}min)")

        # ─── 4. Open Positions ───
        checks["open_positions"] = self.open_positions < MAX_OPEN_POSITIONS
        if not checks["open_positions"]:
            reasons.append(f"Too many open positions ({self.open_positions})")

        # ─── 5. Position Size (SOFT RULE - auto-adjust) ───
        max_allowed = self.bankroll * MAX_BANKROLL_PER_TRADE
        checks["position_size"] = signal.stake <= max_allowed
        if not checks["position_size"]:
            # SOFT: Don't block, just adjust down
            if max_allowed >= MIN_STAKE:
                checks["position_size"] = True  # Not a blocker anymore
                logger.info(
                    f"Stake auto-adjusted: ${signal.stake:.2f} -> ${max_allowed:.2f} "
                    f"(max {MAX_BANKROLL_PER_TRADE:.0%} of bankroll)"
                )
            else:
                reasons.append(
                    f"Max allowed ${max_allowed:.2f} < min stake ${MIN_STAKE:.2f}"
                )

        # ─── 6. Consecutive Losses ───
        checks["consecutive_losses"] = self.consecutive_losses < MAX_CONSECUTIVE_LOSSES
        if not checks["consecutive_losses"]:
            reasons.append(f"Max consecutive losses ({self.consecutive_losses})")

        # ─── 7. Model Drift ───
        checks["model_drift"] = not self.model_drift_active
        if not checks["model_drift"]:
            reasons.append("Model in drift state")

        # ─── 8. Expected Value ───
        checks["positive_ev"] = signal.expected_value > 0
        if not checks["positive_ev"]:
            reasons.append(f"Negative EV ({signal.expected_value:.4f})")

        # ─── 9. Minimum Bankroll ───
        checks["min_bankroll"] = self.bankroll >= MIN_STAKE
        if not checks["min_bankroll"]:
            reasons.append(f"Insufficient bankroll (${self.bankroll:.2f})")

        # ─── Evaluate ───
        approved = all(checks.values())

        if approved:
            adjusted_stake = self._adjust_stake(signal.stake, is_martingale=signal.is_martingale)
            reason = "All checks passed"
            if signal.is_martingale:
                reason += " (martingale recovery — loss reduction bypassed)"
            logger.info(f"RISK APPROVED: stake=${adjusted_stake:.2f}")
        else:
            adjusted_stake = 0.0
            reason = "; ".join(reasons)
            logger.warning(f"RISK BLOCKED: {reason}")

        return RiskDecision(
            approved=approved,
            reason=reason,
            adjusted_stake=adjusted_stake,
            checks=checks,
        )

    def _adjust_stake(self, requested_stake: float, is_martingale: bool = False) -> float:
        """
        Adjust stake based on current conditions.
        Gentler reduction during losing streaks: 15% per consecutive loss
        (not the previous 50% per loss which was too aggressive).

        MARTINGALE BYPASS: During martingale recovery, the stake is 2x the
        last lost stake by design — reducing it defeats the recovery strategy.
        So we skip the loss-based reduction and only enforce the hard cap.
        """
        stake = requested_stake

        # ─── Martingale recovery: only apply hard cap, no loss reduction ───
        if not is_martingale:
            # Gradual reduction: 15% less per consecutive loss
            if self.consecutive_losses >= 2:
                reduction = max(0.25, 1.0 - (self.STAKE_REDUCTION_PER_LOSS * self.consecutive_losses))
                stake *= reduction
                logger.info(
                    f"Stake reduced to {reduction:.0%} due to {self.consecutive_losses} "
                    f"consecutive losses"
                )

        # Cap at max allowed per trade (2% bankroll) — ALWAYS enforced, even martingale
        max_allowed = self.bankroll * MAX_BANKROLL_PER_TRADE
        stake = min(stake, max_allowed)

        # Floor at minimum
        stake = max(stake, MIN_STAKE)

        return round(stake, 2)

    def record_outcome(self, won: bool, stake: float, payout_received: float):
        """Record the outcome of a completed trade."""
        self.total_trades += 1
        self.daily_trades += 1

        if won:
            profit = payout_received - stake
            self.daily_pnl += profit
            self.bankroll += profit
            self.consecutive_losses = 0
            self.consecutive_wins += 1
            logger.info(f"Trade WON: +${profit:.2f} (bankroll: ${self.bankroll:.2f})")
        else:
            self.daily_pnl -= stake
            self.bankroll -= stake
            self.consecutive_wins = 0
            self.consecutive_losses += 1
            self.max_consecutive_losses = max(
                self.max_consecutive_losses, self.consecutive_losses
            )
            logger.info(f"Trade LOST: -${stake:.2f} (bankroll: ${self.bankroll:.2f})")

        # Update open positions
        self.open_positions = max(0, self.open_positions - 1)

        # Check circuit breaker
        if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            if not self.circuit_breaker_active:
                self.circuit_breaker_active = True
                self._circuit_breaker_activated_at = time.time()
                logger.warning(
                    f"CIRCUIT BREAKER ACTIVATED: "
                    f"{self.consecutive_losses} consecutive losses. "
                    f"Auto-resets in {CIRCUIT_BREAKER_COOLDOWN_SEC}s."
                )

        # Check daily loss limit
        max_daily_loss = self.initial_bankroll * MAX_DAILY_LOSS
        if self.daily_pnl <= -max_daily_loss:
            self.daily_loss_limit_hit = True
            logger.error(
                f"DAILY LOSS LIMIT HIT: "
                f"${self.daily_pnl:.2f} (limit: ${-max_daily_loss:.2f})"
            )

    def set_drift_state(self, is_drifting: bool):
        """Update model drift state from drift detector."""
        self.model_drift_active = is_drifting

    def reset_circuit_breaker(self):
        """Manually reset circuit breaker."""
        self.circuit_breaker_active = False
        self.consecutive_losses = 0
        logger.info("Circuit breaker manually reset")

    def reset_daily(self):
        """Reset daily counters."""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_loss_limit_hit = False
        self._last_day = time.strftime("%Y-%m-%d")
        logger.info("Daily risk counters reset")

    def new_session(self, bankroll: float = None):
        """Start a new trading session."""
        if bankroll:
            self.bankroll = bankroll
        self.session_start = time.time()
        self.session_expired = False
        self.circuit_breaker_active = False
        self._circuit_breaker_activated_at = 0.0
        self.open_positions = 0
        logger.info(f"New session started: bankroll=${self.bankroll:.2f}")

    @property
    def total_pnl(self) -> float:
        return self.bankroll - self.initial_bankroll

    @property
    def roi(self) -> float:
        if self.initial_bankroll == 0:
            return 0.0
        return self.total_pnl / self.initial_bankroll

    def summary(self) -> dict:
        # Check auto-reset for display purposes
        if self.circuit_breaker_active:
            elapsed = time.time() - self._circuit_breaker_activated_at
            remaining = max(0, CIRCUIT_BREAKER_COOLDOWN_SEC - elapsed)
        else:
            remaining = 0

        return {
            "bankroll": round(self.bankroll, 2),
            "initial_bankroll": round(self.initial_bankroll, 2),
            "total_pnl": round(self.total_pnl, 2),
            "roi": round(self.roi * 100, 1),
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_trades": self.daily_trades,
            "total_trades": self.total_trades,
            "consecutive_losses": self.consecutive_losses,
            "consecutive_wins": self.consecutive_wins,
            "circuit_breaker": self.circuit_breaker_active,
            "circuit_breaker_remaining_sec": round(remaining, 0),
            "daily_limit_hit": self.daily_loss_limit_hit,
            "session_expired": self.session_expired,
            "model_drift": self.model_drift_active,
            "session_minutes": round((time.time() - self.session_start) / 60, 0),
        }
