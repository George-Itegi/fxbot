"""
Stake Manager — Dynamic Stake Sizing with Martingale Recovery
==============================================================
Solves the 50% win rate problem:

At 50% win rate with ~89% payout, you're LOSING money:
  EV = 0.5 * 0.89 - 0.5 * 1.0 = -$0.055 per dollar risked
  Breakeven win rate = 1 / (1 + payout) ≈ 52.9%

This module implements THREE strategies working together:

1. CONFIDENCE SCALING:
   - All 3 models agree (100%) → stake * 3.0
   - 2/3 models agree (67%)    → stake * 1.0
   - Less than 2/3 agree       → skip trade entirely
   This means we trade MORE when our edge is strongest.

2. MARTINGALE 2x ON LOSS (CONSERVATIVE):
   - After a loss: NEXT stake = 2x the lost stake
   - After a win:  Reset stake back to normal base
   - Max consecutive martingale steps: 2 (2^2 = 4x base → hard cap)
   - Hard cap: never exceed $20 (MAX_MARTINGALE_STAKE)
   - Direction MUST persist: if you lost on Over, you double down on Over
   - If max martingale steps reached, reset to base stake (take the loss)
   - Worst case: $5 + $10 + $20 = $35 total exposure

3. WIN STREAK COMPOUNDING:
   - Consecutive wins → slight boost (1.1x per win, capped at 2.0x)
   - Hard cap prevents runaway stakes

The combined formula:
  final_stake = base_stake × confidence_multiplier × martingale_factor × streak_factor

All factors are capped so the final stake NEVER exceeds:
  - MAX_STAKE ($5.00) absolute limit (normal mode)
  - MAX_MARTINGALE_STAKE ($20.00) absolute limit (recovery mode)
  - MAX_MARTINGALE_STEPS (2) consecutive doublings max
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from config import (MIN_STAKE, MAX_STAKE, MAX_BANKROLL_PER_TRADE,
                    MIN_CONFIDENCE)
from trading.signal_generator import Signal
from utils.logger import setup_logger

logger = setup_logger("trading.stake_manager")


@dataclass
class StakeState:
    """Tracks the state used for dynamic stake calculations."""
    # Drawdown tracking
    peak_bankroll: float = 0.0          # Highest bankroll seen
    current_drawdown_pct: float = 0.0   # Current drawdown from peak (0-1)

    # Streak tracking
    consecutive_wins: int = 0
    consecutive_losses: int = 0

    # ─── Martingale tracking ───
    martingale_step: int = 0            # Current martingale step (0 = no martingale active)
    last_lost_stake: float = 0.0        # The stake that was lost (to double from)
    martingale_base_stake: float = 0.0  # The base stake before martingale started
    martingale_direction: Optional[str] = None  # Direction of FIRST loss — must persist!
    martingale_market: Optional[str] = None     # v8.1: Market where martingale started — MUST stay here!
    martingale_barrier: Optional[int] = None    # v9: Barrier of FIRST loss — must persist!

    # Recent trade history (for win rate calculation)
    recent_results: list = field(default_factory=list)
    max_recent_results: int = 50

    # Last calculated stake info (for logging)
    last_stake_breakdown: dict = field(default_factory=dict)


class StakeManager:
    """
    Dynamic stake sizing with Martingale 2x on loss (CONSERVATIVE).

    The strategy:
    - After a LOSS: double the next stake (Martingale 2x)
    - After a WIN: reset stake to normal base
    - Max 2 consecutive martingale steps ($5→$10→$20 then RESET)
    - Direction MUST persist during recovery (no switching)
    - Hard caps prevent runaway stakes

    Worst case per chain: $5 + $10 + $20 = $35 total exposure
    (Previously: $5 + $10 + $20 + $40 + $50 + $50 = $175 — way too dangerous)
    """

    # ─── Configuration ───

    # Confidence multiplier bounds
    CONFIDENCE_MIN_MULT = 0.5    # At MIN_CONFIDENCE (56%): 0.5x stake
    CONFIDENCE_MAX_MULT = 3.0    # At 80%+ confidence: 3.0x stake

    # Agreement multiplier (THIS IS THE BIGGEST FACTOR)
    AGREEMENT_ALL_3 = 3.0       # All 3 models agree → 3x stake
    AGREEMENT_2_OF_3 = 1.0      # 2 of 3 agree → normal stake
    AGREEMENT_BELOW = 0.0       # Below 67% → SKIP (handled in signal_gen)

    # ─── Martingale Settings ───
    # v8: Adjusted for 85% payout. 2x doesn't recover — need 2.35x.
    # Step 1: $0.35 → Step 2: $0.82 → Step 3: $1.93
    # If win at step 3: $1.93 × 0.85 = $1.64 profit vs $1.17 total loss = +$0.47 net
    MARTINGALE_MULTIPLIER = 2.35      # Adjusted for 85% payout (2x doesn't recover)
    MAX_MARTINGALE_STEPS = 3          # Max 3 steps (trusted setups, max 3-4 consecutive losses)
    MAX_MARTINGALE_STAKE = 10.0       # Absolute max for martingale ($10 — tighter with quality setups)

    # Win streak compounding
    STREAK_BOOST_PER_WIN = 0.10     # +10% per consecutive win
    STREAK_BOOST_CAP = 2.0          # Cap at 2x (5 consecutive wins)

    # Hard safety caps
    MAX_NORMAL_STAKE_PCT = 0.05     # Max 5% of bankroll normally

    def __init__(self, initial_bankroll: float = 100.0):
        self.state = StakeState(peak_bankroll=initial_bankroll)
        self._initial_bankroll = initial_bankroll
        logger.info(
            f"StakeManager initialized: bankroll=${initial_bankroll:.2f}, "
            f"martingale={self.MARTINGALE_MULTIPLIER}x, max_steps={self.MAX_MARTINGALE_STEPS}, "
            f"max_martingale_stake=${self.MAX_MARTINGALE_STAKE}, "
            f"breakeven_wr={self._breakeven_win_rate(0.85):.1%}"
        )

    # ─── Public API ───

    def calculate_stake(self, signal: Signal, bankroll: float,
                        payout: float = 0.89) -> float:
        """
        Calculate the optimal stake for a given signal.

        MARTINGALE PRIORITY:
        If martingale is active (we lost the last trade), the next stake
        is simply 2x the lost stake. This is the classic martingale recovery:
          Lose $5 → next = $10
          Lose $10 → next = $20
          Lose $20 → next = $40
        Capped at MAX_MARTINGALE_STEPS and MAX_MARTINGALE_STAKE.

        NORMAL MODE:
        Combines Kelly criterion, confidence, agreement, streak, and EV factors.
        """
        if bankroll < MIN_STAKE:
            self.state.last_stake_breakdown = {"reason": "bankroll_too_low"}
            return MIN_STAKE

        # Update drawdown state
        self._update_drawdown(bankroll)

        # ─── MARTINGALE RECOVERY (top priority) ───
        # If we lost the last trade, double the stake to recover.
        # This is SIMPLE and DIRECT: 2x the last lost stake, not a complex formula.
        if self.state.martingale_step > 0:
            recovery_stake = self.state.last_lost_stake * self.MARTINGALE_MULTIPLIER

            # Safety caps for martingale
            max_martingale = min(self.MAX_MARTINGALE_STAKE, bankroll * 0.10)
            recovery_stake = min(recovery_stake, max_martingale)
            recovery_stake = max(recovery_stake, MIN_STAKE)
            recovery_stake = round(recovery_stake, 2)

            self.state.last_stake_breakdown = {
                "mode": "martingale_recovery",
                "last_lost_stake": round(self.state.last_lost_stake, 2),
                "martingale_step": self.state.martingale_step,
                "recovery_stake": recovery_stake,
                "final_stake": recovery_stake,
                "drawdown_pct": round(self.state.current_drawdown_pct * 100, 1),
            }
            return recovery_stake

        # ─── NORMAL STAKE CALCULATION ───
        # 1. Base Stake (Kelly)
        base_stake = signal.kelly_fraction * bankroll
        base_stake = max(base_stake, MIN_STAKE)

        # 2. Confidence Multiplier
        confidence_mult = self._confidence_multiplier(signal.confidence)

        # 3. Setup Quality Multiplier (replaces agreement — now single model)
        setup_mult = self._setup_quality_multiplier(signal.setup_score)

        # 4. Streak Factor
        streak_factor = self._streak_factor()

        # 5. EV Bonus
        ev_factor = self._ev_factor(signal.expected_value, payout)

        # ─── Combine All Factors ───
        combined_mult = confidence_mult * setup_mult * streak_factor * ev_factor

        stake = base_stake * combined_mult

        # ─── Hard Safety Caps ───
        # Normal cap: max 5% of bankroll
        max_stake = bankroll * self.MAX_NORMAL_STAKE_PCT
        stake = min(stake, max_stake)

        # Absolute cap
        stake = min(stake, MAX_STAKE)

        # Floor
        stake = max(stake, MIN_STAKE)

        # Round to 2 decimal places
        stake = round(stake, 2)

        # Store breakdown for logging
        self.state.last_stake_breakdown = {
            "mode": "normal",
            "base_stake": round(base_stake, 2),
            "confidence_mult": round(confidence_mult, 2),
            "setup_mult": round(setup_mult, 2),
            "streak_factor": round(streak_factor, 2),
            "ev_factor": round(ev_factor, 2),
            "combined_mult": round(combined_mult, 2),
            "final_stake": stake,
            "martingale_step": 0,
            "recovery_mode": False,
            "drawdown_pct": round(self.state.current_drawdown_pct * 100, 1),
        }

        return stake

    def record_outcome(self, won: bool, stake: float, payout: float,
                       bankroll: float, direction: str = None,
                       symbol: str = None, barrier: int = None):
        """
        Record a trade outcome to update martingale and streak tracking.

        Call this AFTER every trade settlement.
        
        Args:
            direction: The direction of the trade (e.g., "DIGITOVER" or "DIGITUNDER").
                       Required for martingale direction persistence.
            symbol: The market symbol (e.g., "1HZ100V").
                    v8.1: Required for martingale market persistence.
            barrier: The barrier value (e.g., 7 for Over 7).
                     v9: Required for martingale barrier persistence.
        """
        if won:
            self.state.consecutive_wins += 1
            self.state.consecutive_losses = 0

            # ─── WIN: Reset martingale ───
            self.state.martingale_step = 0
            self.state.last_lost_stake = 0.0
            self.state.martingale_base_stake = 0.0
            self.state.martingale_direction = None
            self.state.martingale_market = None
            self.state.martingale_barrier = None  # v9
        else:
            self.state.consecutive_losses += 1
            self.state.consecutive_wins = 0

            # ─── LOSS: Activate martingale ───
            # Next stake = 2.35x the lost stake
            if self.state.martingale_step == 0:
                # First loss — record the base stake, direction, market, AND barrier
                self.state.martingale_base_stake = stake
                self.state.martingale_direction = direction
                self.state.martingale_market = symbol
                self.state.martingale_barrier = barrier  # v9
                logger.info(
                    f"MARTINGALE STARTED: market={symbol}, direction={direction}, "
                    f"barrier={barrier}, lost ${stake:.2f}"
                )

            self.state.last_lost_stake = stake
            self.state.martingale_step += 1

            # If we've hit max martingale steps, reset
            if self.state.martingale_step > self.MAX_MARTINGALE_STEPS:
                logger.warning(
                    f"MARTINGALE MAX REACHED ({self.MAX_MARTINGALE_STEPS} steps). "
                    f"Lost ${stake:.2f} on {self.state.martingale_direction} @ {self.state.martingale_market}. "
                    f"Resetting to base stake — taking the loss."
                )
                self.state.martingale_step = 0
                self.state.last_lost_stake = 0.0
                self.state.martingale_base_stake = 0.0
                self.state.martingale_direction = None
                self.state.martingale_market = None
                self.state.martingale_barrier = None  # v9
            else:
                next_stake = min(stake * self.MARTINGALE_MULTIPLIER, self.MAX_MARTINGALE_STAKE)
                logger.info(
                    f"MARTINGALE STEP {self.state.martingale_step}/{self.MAX_MARTINGALE_STEPS}: "
                    f"Lost ${stake:.2f} on {direction} @ {symbol}, next stake=${next_stake:.2f} "
                    f"(must stay {self.state.martingale_direction} on {self.state.martingale_market})"
                )

        # Track recent results for win rate
        self.state.recent_results.append(won)
        if len(self.state.recent_results) > self.state.max_recent_results:
            self.state.recent_results = self.state.recent_results[-self.state.max_recent_results:]

        # Update drawdown
        self._update_drawdown(bankroll)

    def get_recent_win_rate(self) -> float:
        """Get win rate from recent trades."""
        if not self.state.recent_results:
            return 0.5  # No data → assume 50%
        wins = sum(1 for r in self.state.recent_results if r)
        return wins / len(self.state.recent_results)

    def is_profitable_at_current_win_rate(self, payout: float = 0.89) -> dict:
        """Check if the current win rate is profitable given the payout."""
        wr = self.get_recent_win_rate()
        breakeven = self._breakeven_win_rate(payout)
        ev_per_dollar = wr * payout - (1 - wr) * 1.0

        return {
            "win_rate": round(wr, 3),
            "breakeven_wr": round(breakeven, 3),
            "is_profitable": wr > breakeven,
            "ev_per_dollar": round(ev_per_dollar, 4),
            "recent_trades": len(self.state.recent_results),
            "payout": round(payout, 3),
            "gap": round(wr - breakeven, 3),
        }

    def summary(self) -> dict:
        """Return full state summary for logging."""
        return {
            "peak_bankroll": round(self.state.peak_bankroll, 2),
            "current_drawdown_pct": round(self.state.current_drawdown_pct * 100, 1),
            "consecutive_wins": self.state.consecutive_wins,
            "consecutive_losses": self.state.consecutive_losses,
            "martingale_step": self.state.martingale_step,
            "martingale_direction": self.state.martingale_direction,
            "martingale_market": self.state.martingale_market,
            "martingale_barrier": self.state.martingale_barrier,  # v9
            "martingale_next_mult": round(self.MARTINGALE_MULTIPLIER ** self.state.martingale_step, 2) if self.state.martingale_step > 0 else 1,
            "recent_win_rate": round(self.get_recent_win_rate(), 3),
            "recent_trades": len(self.state.recent_results),
            "last_breakdown": self.state.last_stake_breakdown,
        }

    # ─── Private Methods ───

    def _martingale_factor(self) -> float:
        """
        Martingale 2x on loss.

        After a loss, the next stake is 2x the lost stake.
        - Step 0 (no loss): 1.0x
        - Step 1 (1 loss): 2.0x
        - Step 2 (2 losses): 4.0x
        - Step 3 (3 losses): 8.0x
        - Step 4 (4 losses): 16.0x
        - Step 5 (5 losses): 32.0x → MAX, then reset

        This is applied ON TOP of the base stake calculation.
        """
        if self.state.martingale_step == 0:
            return 1.0

        factor = self.MARTINGALE_MULTIPLIER ** self.state.martingale_step
        return factor

    def _confidence_multiplier(self, confidence: float) -> float:
        """Scale stake by model confidence."""
        if confidence <= MIN_CONFIDENCE:
            return self.CONFIDENCE_MIN_MULT

        if confidence >= 0.80:
            return self.CONFIDENCE_MAX_MULT

        range_size = 0.80 - MIN_CONFIDENCE
        position = (confidence - MIN_CONFIDENCE) / range_size

        mult = self.CONFIDENCE_MIN_MULT + position * (self.CONFIDENCE_MAX_MULT - self.CONFIDENCE_MIN_MULT)
        return mult

    def _setup_quality_multiplier(self, setup_score: float) -> float:
        """Scale stake by setup quality score."""
        # Score 0.60 = 1.0x (minimum), 0.75 = 1.5x, 0.90+ = 2.0x
        if setup_score >= 0.90:
            return 2.0
        elif setup_score >= 0.60:
            position = (setup_score - 0.60) / (0.90 - 0.60)
            return 1.0 + position * 1.0  # 1.0x to 2.0x
        else:
            return 0.5  # Below minimum — shouldn't trade but safety net

    def _streak_factor(self) -> float:
        """Win streak compounding."""
        if self.state.consecutive_wins <= 0:
            return 1.0

        boost = 1.0 + (self.STREAK_BOOST_PER_WIN * self.state.consecutive_wins)
        return min(boost, self.STREAK_BOOST_CAP)

    def _ev_factor(self, ev: float, payout: float) -> float:
        """Boost stake for high-EV signals."""
        if ev >= 0.20:
            return 2.0
        elif ev >= 0.10:
            return 1.0 + 0.5 * ((ev - 0.10) / 0.10)
        elif ev >= 0.01:
            return 0.8 + 0.2 * ((ev - 0.01) / 0.09)
        else:
            return 0.8

    def _update_drawdown(self, bankroll: float):
        """Update drawdown tracking from current bankroll."""
        if bankroll > self.state.peak_bankroll:
            self.state.peak_bankroll = bankroll

        if self.state.peak_bankroll > 0:
            dd = (self.state.peak_bankroll - bankroll) / self.state.peak_bankroll
            self.state.current_drawdown_pct = max(0.0, dd)
        else:
            self.state.current_drawdown_pct = 0.0

    @staticmethod
    def _breakeven_win_rate(payout: float) -> float:
        """Calculate the breakeven win rate for a given payout."""
        if payout <= 0:
            return 1.0
        return 1.0 / (1.0 + payout)
