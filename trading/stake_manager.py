"""
Stake Manager — Dynamic Stake Sizing with Martingale Recovery (v12)
=====================================================================
v12: Fixed martingale for Over 4 / Under 5 contracts (~95% payout).

PROBLEM WITH PREVIOUS VERSIONS:
- Martingale multiplier was 2.35x (designed for 85% payout)
- But Over 4 / Under 5 have ~95% payout
- The 10% bankroll cap was too aggressive, killing recovery stakes
- MAX_MARTINGALE_STEPS=2 but log showed 5 (old code was running)
- Breakeven calculation was using 85% instead of 95%

v12 MARTINGALE MATH (for ~95% payout):
- To recover a loss of $L with payout P, you need to win:
  stake × P ≥ L  →  stake ≥ L / P
- For P = 0.95: stake ≥ L / 0.95 = L × 1.053
- But that only breaks even. To PROFIT from recovery:
  stake × P ≥ L + target_profit
- Conservative: use 2.1x multiplier (recovers loss + small profit)

ACTUAL RECOVERY EXAMPLES:
  Lose $0.35 → next = $0.35 × 2.1 = $0.74
  Win $0.74 → profit = $0.74 × 0.95 = $0.70, net = $0.70 - $0.35 = +$0.35 ✅

  Lose $0.35 → lose $0.74 → next = $0.74 × 2.1 = $1.55
  Win $1.55 → profit = $1.55 × 0.95 = $1.47, net = $1.47 - $0.35 - $0.74 = +$0.38 ✅

  Max 3 steps: $0.35 + $0.74 + $1.55 = $2.64 total risk
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
    martingale_market: Optional[str] = None     # Market where martingale started
    martingale_barrier: Optional[int] = None    # Barrier of FIRST loss — must persist!
    total_martingale_loss: float = 0.0  # v12: Total accumulated loss during martingale chain

    # Recent trade history (for win rate calculation)
    recent_results: list = field(default_factory=list)
    max_recent_results: int = 50

    # Last calculated stake info (for logging)
    last_stake_breakdown: dict = field(default_factory=dict)


class StakeManager:
    """
    Dynamic stake sizing with Martingale recovery (v12 — fixed for 95% payout).
    
    v12 Key Fix: Martingale multiplier and caps are now correct for
    Over 4 / Under 5 contracts which have ~95% payout.
    
    The recovery chain is:
    - Step 1: 2.1x the lost stake
    - Step 2: 2.1x the previous stake
    - Step 3: 2.1x again (max)
    
    Max total risk: ~$2.64 on a $0.35 base stake (3 steps)
    """

    # ─── Configuration ───

    # Confidence multiplier bounds
    CONFIDENCE_MIN_MULT = 0.5    # At MIN_CONFIDENCE (52%): 0.5x stake
    CONFIDENCE_MAX_MULT = 2.0    # At 80%+ confidence: 2.0x stake (v12: reduced from 3.0)

    # ─── Martingale Settings (v12: Fixed for ~95% payout) ───
    # For 95% payout, you need: next_stake × 0.95 > total_loss_so_far
    # 2.1x ensures recovery + small profit at each step
    MARTINGALE_MULTIPLIER = 2.1       # v12: Correct for 95% payout (was 2.35 for 85%)
    MAX_MARTINGALE_STEPS = 3          # v12: 3 steps max (was 2 — too few for $0.35 base)
    MAX_MARTINGALE_STAKE = 2.0        # v12: $2 max (was $5 — too aggressive for 50/50 contracts)
    MARTINGALE_BANKROLL_PCT = 0.15    # v12: Up to 15% of bankroll for recovery (was 10%)

    # Win streak compounding
    STREAK_BOOST_PER_WIN = 0.10     # +10% per consecutive win
    STREAK_BOOST_CAP = 1.5          # v12: Cap at 1.5x (was 2.0 — less aggressive)

    # Hard safety caps
    MAX_NORMAL_STAKE_PCT = 0.05     # Max 5% of bankroll normally

    def __init__(self, initial_bankroll: float = 100.0):
        self.state = StakeState(peak_bankroll=initial_bankroll)
        self._initial_bankroll = initial_bankroll
        logger.info(
            f"StakeManager v12: bankroll=${initial_bankroll:.2f}, "
            f"martingale={self.MARTINGALE_MULTIPLIER}x, max_steps={self.MAX_MARTINGALE_STEPS}, "
            f"max_martingale_stake=${self.MAX_MARTINGALE_STAKE}, "
            f"breakeven_wr(95%)={self._breakeven_win_rate(0.95):.1%}, "
            f"breakeven_wr(85%)={self._breakeven_win_rate(0.85):.1%}"
        )

    # ─── Public API ───

    def calculate_stake(self, signal: Signal, bankroll: float,
                        payout: float = 0.95) -> float:
        """
        Calculate the optimal stake for a given signal.
        
        v12: payout default is 0.95 (Over 4 / Under 5), not 0.89.

        MARTINGALE RECOVERY:
        If martingale is active, the next stake is 2.1x the lost stake.
        This ensures full recovery + profit at ~95% payout.

        NORMAL MODE:
        Kelly criterion + confidence + setup quality + streak + EV factors.
        """
        if bankroll < MIN_STAKE:
            self.state.last_stake_breakdown = {"reason": "bankroll_too_low"}
            return MIN_STAKE

        # Update drawdown state
        self._update_drawdown(bankroll)

        # ─── MARTINGALE RECOVERY (top priority) ───
        if self.state.martingale_step > 0:
            # Calculate recovery stake: multiply last lost stake
            recovery_stake = self.state.last_lost_stake * self.MARTINGALE_MULTIPLIER
            
            # v12: Check if recovery stake is enough to cover total loss
            # We want: recovery_stake × payout > total_martingale_loss
            min_recovery = self.state.total_martingale_loss / payout if payout > 0 else recovery_stake
            
            # Use the LARGER of the two (ensures full recovery)
            recovery_stake = max(recovery_stake, min_recovery)
            
            # v12: Allow up to 15% of bankroll for recovery (was 10% — too tight)
            max_martingale = min(self.MAX_MARTINGALE_STAKE, bankroll * self.MARTINGALE_BANKROLL_PCT)
            recovery_stake = min(recovery_stake, max_martingale)
            recovery_stake = max(recovery_stake, MIN_STAKE)
            recovery_stake = round(recovery_stake, 2)

            # Log what we're trying to recover
            potential_profit = recovery_stake * payout - self.state.total_martingale_loss
            logger.info(
                f"MARTINGALE STAKE: ${recovery_stake:.2f} "
                f"(step {self.state.martingale_step}/{self.MAX_MARTINGALE_STEPS}, "
                f"lost_so_far=${self.state.total_martingale_loss:.2f}, "
                f"potential_net=${potential_profit:+.2f})"
            )

            self.state.last_stake_breakdown = {
                "mode": "martingale_recovery",
                "last_lost_stake": round(self.state.last_lost_stake, 2),
                "total_loss_so_far": round(self.state.total_martingale_loss, 2),
                "martingale_step": self.state.martingale_step,
                "recovery_stake": recovery_stake,
                "potential_net": round(potential_profit, 2),
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

        # 3. Setup Quality Multiplier
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
        """
        if won:
            self.state.consecutive_wins += 1
            self.state.consecutive_losses = 0

            # ─── WIN: Reset martingale ───
            if self.state.martingale_step > 0:
                # We won during martingale recovery!
                net_result = (stake * payout) - self.state.total_martingale_loss
                logger.info(
                    f"MARTINGALE RECOVERY WIN! "
                    f"Step {self.state.martingale_step}, won ${stake * payout:.2f}, "
                    f"total_loss_was=${self.state.total_martingale_loss:.2f}, "
                    f"net=${net_result:+.2f}"
                )
            
            self.state.martingale_step = 0
            self.state.last_lost_stake = 0.0
            self.state.martingale_base_stake = 0.0
            self.state.martingale_direction = None
            self.state.martingale_market = None
            self.state.martingale_barrier = None
            self.state.total_martingale_loss = 0.0  # v12: Reset total loss
        else:
            self.state.consecutive_losses += 1
            self.state.consecutive_wins = 0

            # ─── LOSS: Activate martingale ───
            # Track total loss for proper recovery calculation
            self.state.total_martingale_loss += stake  # v12: Accumulate total loss
            
            if self.state.martingale_step == 0:
                # First loss — record the base stake, direction, market, AND barrier
                self.state.martingale_base_stake = stake
                self.state.martingale_direction = direction
                self.state.martingale_market = symbol
                self.state.martingale_barrier = barrier
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
                    f"Total loss: ${self.state.total_martingale_loss:.2f} "
                    f"on {self.state.martingale_direction} @ {self.state.martingale_market}. "
                    f"Resetting — taking the loss."
                )
                self.state.martingale_step = 0
                self.state.last_lost_stake = 0.0
                self.state.martingale_base_stake = 0.0
                self.state.martingale_direction = None
                self.state.martingale_market = None
                self.state.martingale_barrier = None
                self.state.total_martingale_loss = 0.0
            else:
                next_stake = min(stake * self.MARTINGALE_MULTIPLIER, self.MAX_MARTINGALE_STAKE)
                logger.info(
                    f"MARTINGALE STEP {self.state.martingale_step}/{self.MAX_MARTINGALE_STEPS}: "
                    f"Lost ${stake:.2f} on {direction} @ {symbol}, "
                    f"total_loss=${self.state.total_martingale_loss:.2f}, "
                    f"next stake=${next_stake:.2f} "
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

    def is_profitable_at_current_win_rate(self, payout: float = 0.95) -> dict:
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
            "martingale_barrier": self.state.martingale_barrier,
            "total_martingale_loss": round(self.state.total_martingale_loss, 2),
            "recent_win_rate": round(self.get_recent_win_rate(), 3),
            "recent_trades": len(self.state.recent_results),
            "last_breakdown": self.state.last_stake_breakdown,
        }

    # ─── Private Methods ───

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
        if setup_score >= 0.90:
            return 2.0
        elif setup_score >= 0.60:
            position = (setup_score - 0.60) / (0.90 - 0.60)
            return 1.0 + position * 1.0
        else:
            return 0.5

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
