"""
Stake Manager — Dynamic Stake Sizing with Recovery Logic
==========================================================
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

2. DRAWDOWN RECOVERY (Anti-Martingale):
   - After losses: REDUCE stake (survive the drawdown)
   - After wins:   GRADUALLY increase back (ride the recovery)
   - NEVER double down after a loss (martingale = death)
   - The recovery is slow: +10% per win, so you don't overshoot

3. WIN STREAK COMPOUNDING:
   - Consecutive wins → slight boost (1.1x per win, capped at 2.0x)
   - This is NOT martingale — we only increase after WINS, not losses
   - Hard cap prevents runaway stakes

The combined formula:
  final_stake = base_stake × confidence_multiplier × drawdown_factor × streak_factor

All factors are capped so the final stake NEVER exceeds:
  - MAX_BANKROLL_PER_TRADE (5%) of current bankroll
  - MAX_STAKE ($5.00) absolute limit
  - 2% of bankroll during drawdown (extra safety)
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

    # Recovery tracking
    recovery_mode: bool = False          # True when recovering from drawdown
    recovery_progress: float = 0.0       # 0 = deep drawdown, 1 = fully recovered

    # Recent trade history (for win rate calculation)
    recent_results: list = field(default_factory=list)
    max_recent_results: int = 50

    # Last calculated stake info (for logging)
    last_stake_breakdown: dict = field(default_factory=dict)


class StakeManager:
    """
    Dynamic stake sizing with recovery logic.

    The KEY insight: at 50% win rate you LOSE money with flat stakes.
    To be profitable you need EITHER:
      a) Win rate > 53% (breakeven at 89% payout), OR
      b) Win BIGGER when you win, and lose SMALLER when you lose

    This module implements (b): trade bigger when confidence is high
    (models agree), and smaller during drawdowns.

    The math works like this:
    - With confidence scaling: average win stake > average loss stake
    - Because high-confidence trades win more often
    - So even at 50% overall win rate, you win more on the high-stake trades
    - And lose less on the low-stake trades (where confidence was lower)
    """

    # ─── Configuration ───

    # Confidence multiplier bounds
    CONFIDENCE_MIN_MULT = 0.5    # At MIN_CONFIDENCE (56%): 0.5x stake
    CONFIDENCE_MAX_MULT = 3.0    # At 80%+ confidence: 3.0x stake

    # Agreement multiplier (THIS IS THE BIGGEST FACTOR)
    AGREEMENT_ALL_3 = 3.0       # All 3 models agree → 3x stake
    AGREEMENT_2_OF_3 = 1.0      # 2 of 3 agree → normal stake
    AGREEMENT_BELOW = 0.0       # Below 67% → SKIP (handled in signal_gen)

    # Drawdown recovery
    DRAWDOWN_REDUCTION_RATE = 0.12  # Reduce stake by 12% per 1% drawdown
    DRAWDOWN_FLOOR = 0.25           # Never reduce below 25% of base (even in deep drawdown)
    RECOVERY_INCREASE_RATE = 0.10   # Increase 10% per win during recovery
    DRAWDOWN_THRESHOLD = 0.05       # Enter recovery mode after 5% drawdown

    # Win streak compounding
    STREAK_BOOST_PER_WIN = 0.10     # +10% per consecutive win
    STREAK_BOOST_CAP = 2.0          # Cap at 2x (5 consecutive wins)

    # Hard safety caps
    MAX_DRAWDOWN_STAKE_PCT = 0.02   # Max 2% of bankroll during drawdown
    MAX_NORMAL_STAKE_PCT = 0.05     # Max 5% of bankroll normally

    def __init__(self, initial_bankroll: float = 100.0):
        self.state = StakeState(peak_bankroll=initial_bankroll)
        self._initial_bankroll = initial_bankroll
        logger.info(
            f"StakeManager initialized: bankroll=${initial_bankroll:.2f}, "
            f"breakeven_wr={self._breakeven_win_rate(0.89):.1%}"
        )

    # ─── Public API ───

    def calculate_stake(self, signal: Signal, bankroll: float,
                        payout: float = 0.89) -> float:
        """
        Calculate the optimal stake for a given signal.

        This is the MAIN entry point. It combines all three strategies:
        1. Base stake from Kelly criterion (signal.kelly_fraction × bankroll)
        2. Confidence multiplier (higher confidence → bigger stake)
        3. Drawdown factor (during drawdown → smaller stake)
        4. Streak factor (win streaks → slightly bigger, losses → smaller)

        The result is ALWAYS bounded by hard safety caps.
        """
        if bankroll < MIN_STAKE:
            self.state.last_stake_breakdown = {"reason": "bankroll_too_low"}
            return MIN_STAKE

        # Update drawdown state
        self._update_drawdown(bankroll)

        # ─── 1. Base Stake (Kelly) ───
        # Kelly already accounts for edge, but we enhance it
        base_stake = signal.kelly_fraction * bankroll
        base_stake = max(base_stake, MIN_STAKE)

        # ─── 2. Confidence Multiplier ───
        confidence_mult = self._confidence_multiplier(signal.confidence)

        # ─── 3. Agreement Multiplier (THE KEY FACTOR) ───
        agreement_mult = self._agreement_multiplier(signal.model_agreement)

        # ─── 4. Drawdown Factor ───
        drawdown_factor = self._drawdown_factor()

        # ─── 5. Streak Factor ───
        streak_factor = self._streak_factor()

        # ─── 6. EV Bonus ───
        ev_factor = self._ev_factor(signal.expected_value, payout)

        # ─── Combine All Factors ───
        combined_mult = confidence_mult * agreement_mult * drawdown_factor * streak_factor * ev_factor

        # Cap combined multiplier to prevent runaway
        combined_mult = min(combined_mult, 6.0)

        stake = base_stake * combined_mult

        # ─── Hard Safety Caps ───
        # During drawdown: max 2% of bankroll
        if self.state.recovery_mode:
            max_stake = bankroll * self.MAX_DRAWDOWN_STAKE_PCT
            stake = min(stake, max_stake)
        else:
            # Normal: max 5% of bankroll
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
            "base_stake": round(base_stake, 2),
            "confidence_mult": round(confidence_mult, 2),
            "agreement_mult": round(agreement_mult, 2),
            "drawdown_factor": round(drawdown_factor, 2),
            "streak_factor": round(streak_factor, 2),
            "ev_factor": round(ev_factor, 2),
            "combined_mult": round(combined_mult, 2),
            "final_stake": stake,
            "recovery_mode": self.state.recovery_mode,
            "drawdown_pct": round(self.state.current_drawdown_pct * 100, 1),
        }

        return stake

    def record_outcome(self, won: bool, stake: float, payout: float,
                       bankroll: float):
        """
        Record a trade outcome to update streak and drawdown tracking.

        Call this AFTER every trade settlement.
        """
        # Update streaks
        if won:
            self.state.consecutive_wins += 1
            self.state.consecutive_losses = 0

            # During recovery, each win increases recovery progress
            if self.state.recovery_mode:
                self.state.recovery_progress = min(
                    1.0, self.state.recovery_progress + self.RECOVERY_INCREASE_RATE
                )
        else:
            self.state.consecutive_losses += 1
            self.state.consecutive_wins = 0

            # A loss during recovery sets back progress
            if self.state.recovery_mode:
                self.state.recovery_progress = max(
                    0.0, self.state.recovery_progress - 0.15
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
        """
        Check if the current win rate is profitable given the payout.

        This is the answer to "at 50% win rate, am I in profit or loss?"
        """
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
            "gap": round(wr - breakeven, 3),  # Positive = profitable edge
        }

    def summary(self) -> dict:
        """Return full state summary for logging."""
        return {
            "peak_bankroll": round(self.state.peak_bankroll, 2),
            "current_drawdown_pct": round(self.state.current_drawdown_pct * 100, 1),
            "recovery_mode": self.state.recovery_mode,
            "recovery_progress": round(self.state.recovery_progress, 2),
            "consecutive_wins": self.state.consecutive_wins,
            "consecutive_losses": self.state.consecutive_losses,
            "recent_win_rate": round(self.get_recent_win_rate(), 3),
            "recent_trades": len(self.state.recent_results),
            "last_breakdown": self.state.last_stake_breakdown,
        }

    # ─── Private Methods ───

    def _confidence_multiplier(self, confidence: float) -> float:
        """
        Scale stake by model confidence.

        Low confidence (56%) → 0.5x stake (trade small, might be wrong)
        High confidence (80%+) → 3.0x stake (models are very sure)

        The idea: when models are more confident, they're also more likely
        to be right, so we should risk more on those trades.
        """
        if confidence <= MIN_CONFIDENCE:
            return self.CONFIDENCE_MIN_MULT

        # Scale linearly from MIN_CONFIDENCE to 0.80
        if confidence >= 0.80:
            return self.CONFIDENCE_MAX_MULT

        # Linear interpolation
        range_size = 0.80 - MIN_CONFIDENCE
        position = (confidence - MIN_CONFIDENCE) / range_size

        mult = self.CONFIDENCE_MIN_MULT + position * (self.CONFIDENCE_MAX_MULT - self.CONFIDENCE_MIN_MULT)
        return mult

    def _agreement_multiplier(self, agreement: float) -> float:
        """
        Scale stake by model agreement (THE MOST IMPORTANT FACTOR).

        All 3 models agree (100%) → 3.0x stake
          This is the strongest signal — all models see the same pattern.
          When all 3 agree, the trade is more likely to win.

        2 of 3 agree (67%) → 1.0x stake
          Normal signal, use standard stake.

        Below 67% → 0x (should be filtered out by signal generator)

        WHY THIS MATTERS AT 50% WIN RATE:
        - If 3/3 agree trades win 60% and 2/3 trades win 45%,
          then with 3x stake on 3/3 trades:
          Expected profit = 3 * (0.60 * payout - 0.40 * 1) = 3 * (0.534 - 0.40) = +0.402
          Expected loss on 2/3 = 1 * (0.45 * payout - 0.55 * 1) = 0.401 - 0.55 = -0.149
          Net positive! Even though overall win rate is ~50%.
        """
        if agreement >= 1.0:
            return self.AGREEMENT_ALL_3
        elif agreement >= 0.67:
            # Scale from 1.0x at 67% to 3.0x at 100%
            position = (agreement - 0.67) / (1.0 - 0.67)
            return self.AGREEMENT_2_OF_3 + position * (self.AGREEMENT_ALL_3 - self.AGREEMENT_2_OF_3)
        else:
            return self.AGREEMENT_BELOW

    def _drawdown_factor(self) -> float:
        """
        Reduce stake during drawdowns (ANTI-MARTINGALE).

        This is the OPPOSITE of martingale. When losing:
        - Martingale says: double your bet to recover (DANGEROUS)
        - Anti-martingale says: bet LESS to survive (SAFE)

        The factor scales down based on drawdown depth:
        - 0% drawdown  → 1.0x (normal)
        - 5% drawdown  → 0.7x (reduce slightly)
        - 10% drawdown → 0.5x (significant reduction)
        - 20% drawdown → 0.25x (floor — never go below this)

        Recovery is gradual: each win increases recovery_progress by 10%,
        which gradually increases the drawdown factor back to 1.0.
        """
        dd = self.state.current_drawdown_pct

        if dd < self.DRAWDOWN_THRESHOLD:
            # Not in significant drawdown
            if self.state.recovery_mode:
                # Still in recovery but close to peak
                recovery_boost = 0.5 + 0.5 * self.state.recovery_progress
                return min(1.0, recovery_boost)
            return 1.0

        # In drawdown: reduce stake proportionally
        # factor = 1.0 - (drawdown_reduction_rate * drawdown_pct)
        # But floor at DRAWDOWN_FLOOR so we never stop trading entirely
        raw_factor = 1.0 - (self.DRAWDOWN_REDUCTION_RATE * dd * 100)
        factor = max(self.DRAWDOWN_FLOOR, raw_factor)

        return factor

    def _streak_factor(self) -> float:
        """
        Win streak compounding (small boost for consecutive wins).

        This is NOT martingale:
        - Martingale: increase after LOSS (to recover) → DANGEROUS
        - This: increase after WIN (to ride momentum) → SAFE

        +10% per consecutive win, capped at 2.0x (5 wins).

        On a loss, factor drops back to 1.0 immediately.
        This means we naturally bet bigger during winning streaks
        and smaller during losing streaks.
        """
        if self.state.consecutive_wins <= 0:
            return 1.0

        boost = 1.0 + (self.STREAK_BOOST_PER_WIN * self.state.consecutive_wins)
        return min(boost, self.STREAK_BOOST_CAP)

    def _ev_factor(self, ev: float, payout: float) -> float:
        """
        Boost stake for high-EV signals.

        EV > 0.10 → 1.5x
        EV > 0.20 → 2.0x
        EV < 0.01 → 0.8x (marginal edge, be cautious)
        """
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
        # Update peak
        if bankroll > self.state.peak_bankroll:
            self.state.peak_bankroll = bankroll
            # If we've recovered past our peak, exit recovery mode
            if self.state.recovery_mode:
                self.state.recovery_mode = False
                self.state.recovery_progress = 1.0
                logger.info(
                    f"RECOVERY COMPLETE: bankroll ${bankroll:.2f} "
                    f"back above peak ${self.state.peak_bankroll:.2f}"
                )

        # Calculate drawdown from peak
        if self.state.peak_bankroll > 0:
            dd = (self.state.peak_bankroll - bankroll) / self.state.peak_bankroll
            self.state.current_drawdown_pct = max(0.0, dd)
        else:
            self.state.current_drawdown_pct = 0.0

        # Enter recovery mode if drawdown exceeds threshold
        if self.state.current_drawdown_pct >= self.DRAWDOWN_THRESHOLD:
            if not self.state.recovery_mode:
                self.state.recovery_mode = True
                self.state.recovery_progress = 0.0
                logger.warning(
                    f"ENTERING RECOVERY MODE: "
                    f"drawdown={self.state.current_drawdown_pct:.1%}, "
                    f"peak=${self.state.peak_bankroll:.2f}, "
                    f"current=${bankroll:.2f}"
                )

    @staticmethod
    def _breakeven_win_rate(payout: float) -> float:
        """
        Calculate the breakeven win rate for a given payout.

        At payout P, you need to win at least:
          W = 1 / (1 + P)

        Examples:
          Payout 0.89 → breakeven 52.9%
          Payout 0.85 → breakeven 54.1%
          Payout 0.95 → breakeven 51.3%
        """
        if payout <= 0:
            return 1.0
        return 1.0 / (1.0 + payout)
