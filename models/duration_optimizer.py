"""
Dynamic Duration Optimizer
===========================
Automatically selects the BEST contract duration (number of ticks)
for Over/Under trades based on real-time performance tracking.

Why this matters:
- Different market conditions favor different contract lengths
- A 5-tick contract might win 58% of the time, while a 3-tick wins 55%
  and a 7-tick wins 61% — the bot should pick 7!
- During high volatility, shorter durations may be better (less exposure)
- During trending periods, longer durations may capture the trend

How it works:
1. Tracks win rate per duration (1-10 ticks) in a rolling window
2. Tracks payout per duration (Deriv offers different payouts per duration)
3. Computes Expected Value = win_rate * payout - (1 - win_rate) * stake
4. Selects the duration with the HIGHEST expected value
5. Requires minimum sample size before trusting a duration
6. Gradually explores all durations (epsilon-greedy exploration)

Deriv API supports durations from 1 to 10 ticks for digit contracts.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from config import CONTRACT_DURATION, MIN_DURATION, MAX_DURATION
from utils.logger import setup_logger

logger = setup_logger("models.duration_optimizer")


@dataclass
class DurationStats:
    """Statistics for a single contract duration."""
    duration: int                   # Number of ticks
    wins: int = 0                   # Total wins
    losses: int = 0                 # Total losses
    total_payout: float = 0.0       # Total payout received
    total_stake: float = 0.0        # Total stake wagered
    recent_results: deque = field(default_factory=lambda: deque(maxlen=50))  # Rolling window
    
    @property
    def total_trades(self) -> int:
        return self.wins + self.losses
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.5  # Prior
        return self.wins / self.total_trades
    
    @property
    def recent_win_rate(self) -> float:
        """Win rate from the last N trades (more responsive to current conditions)."""
        if not self.recent_results:
            return 0.5
        wins = sum(1 for r in self.recent_results if r["won"])
        return wins / len(self.recent_results)
    
    @property
    def avg_payout_ratio(self) -> float:
        """Average payout as a ratio of stake (e.g., 0.85 means $0.85 per $1 staked)."""
        if self.wins == 0:
            return 0.0
        return self.total_payout / self.total_stake if self.total_stake > 0 else 0.0
    
    @property
    def expected_value(self) -> float:
        """
        Expected value per dollar staked.
        EV = P(win) * payout_ratio - P(lose) * 1
        Positive EV = profitable duration
        """
        wr = self.recent_win_rate  # Use recent win rate (more adaptive)
        payout = self.avg_payout_ratio if self.avg_payout_ratio > 0 else 0.85  # Default Deriv payout
        return wr * payout - (1 - wr)
    
    @property
    def confidence(self) -> float:
        """
        How confident we are in this duration's stats.
        Based on sample size — more trades = more confidence.
        Uses a Bayesian-inspired approach with a minimum sample threshold.
        """
        n = self.total_trades
        min_samples = 10  # Need at least 10 trades to start trusting
        if n < min_samples:
            return n / min_samples * 0.5  # 0 to 0.5 for under-sampled
        # Confidence approaches 1.0 as sample size grows
        return min(1.0, min_samples / n + 0.5 * (1 - min_samples / n))


class DurationOptimizer:
    """
    Dynamically selects the optimal contract duration (tick count).
    
    Strategy:
    1. EXPLORE: Initially trade all durations to gather data
    2. EXPLOIT: After enough data, pick the best duration
    3. ADAPT: If market conditions change, re-explore
    
    Uses epsilon-greedy exploration:
    - With probability epsilon, pick a random duration (explore)
    - With probability 1-epsilon, pick the best duration (exploit)
    - Epsilon decreases over time as we gather more data
    - Epsilon increases when drift is detected (re-explore)
    """
    
    def __init__(self,
                 min_duration: int = None,
                 max_duration: int = None,
                 default_duration: int = None,
                 exploration_rate: float = 0.15):
        """
        Args:
            min_duration: Minimum contract duration in ticks
            max_duration: Maximum contract duration in ticks
            default_duration: Starting duration (before we have data)
            exploration_rate: Initial epsilon for epsilon-greedy (0.15 = 15% explore)
        """
        self.min_duration = min_duration or MIN_DURATION
        self.max_duration = max_duration or MAX_DURATION
        self.default_duration = default_duration or CONTRACT_DURATION
        self.epsilon = exploration_rate
        
        # Stats for each duration
        self.stats: dict[int, DurationStats] = {}
        for d in range(self.min_duration, self.max_duration + 1):
            self.stats[d] = DurationStats(duration=d)
        
        # Current best duration (updated after each trade result)
        self._current_best: int = self.default_duration
        
        # Exploration tracking
        self._total_selections = 0
        self._exploration_count = 0
        self._last_drift_time = 0.0
        self._drift_mode = False
        
        logger.info(
            f"DurationOptimizer initialized: "
            f"range={self.min_duration}-{self.max_duration} ticks, "
            f"default={self.default_duration}, "
            f"epsilon={self.epsilon:.0%}"
        )
    
    def select_duration(self, force_duration: int = None) -> int:
        """
        Select the best contract duration for the next trade.
        
        Args:
            force_duration: If provided, skip optimization and use this value
        
        Returns:
            Number of ticks for the contract duration
        """
        if force_duration is not None:
            return force_duration
        
        self._total_selections += 1
        
        # ─── Epsilon-Greedy Selection ───
        import random
        
        # Adjust epsilon based on drift state
        effective_epsilon = self.epsilon
        if self._drift_mode:
            effective_epsilon = min(0.5, self.epsilon * 3)  # Explore more during drift
        
        # Decay epsilon over time (but keep a minimum)
        decayed_epsilon = max(0.05, effective_epsilon * (0.995 ** self._total_selections))
        
        if random.random() < decayed_epsilon:
            # EXPLORE: Pick a random duration
            duration = random.randint(self.min_duration, self.max_duration)
            self._exploration_count += 1
            logger.debug(f"Duration EXPLORING: {duration} ticks (epsilon={decayed_epsilon:.3f})")
            return duration
        
        # EXPLOIT: Pick the duration with highest expected value
        best_duration = self._get_best_duration()
        return best_duration
    
    def _get_best_duration(self) -> int:
        """
        Select the duration with the highest expected value.
        Requires minimum sample size before considering a duration.
        """
        candidates = []
        
        for duration, stats in self.stats.items():
            # Only consider durations with enough trades
            if stats.total_trades < 3:
                continue
            
            # Score = EV * confidence (trust high-EV durations more when we have data)
            score = stats.expected_value * stats.confidence
            candidates.append((duration, score, stats))
        
        if not candidates:
            # No data yet — use default
            return self.default_duration
        
        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        best_duration = candidates[0][0]
        best_ev = candidates[0][2].expected_value
        best_wr = candidates[0][2].recent_win_rate
        
        if best_duration != self._current_best:
            old_best = self._current_best
            self._current_best = best_duration
            logger.info(
                f"Duration SHIFT: {old_best} → {best_duration} ticks "
                f"(EV={best_ev:+.3f}, WR={best_wr:.1%}, "
                f"trades={candidates[0][2].total_trades})"
            )
        
        return best_duration
    
    def record_result(self, duration: int, won: bool, payout: float, stake: float):
        """
        Record the outcome of a trade at a given duration.
        
        Args:
            duration: Contract duration in ticks
            won: Whether the trade won
            payout: Payout amount (total, including stake)
            stake: Stake amount
        """
        if duration not in self.stats:
            logger.warning(f"Unknown duration {duration}, ignoring result")
            return
        
        stats = self.stats[duration]
        
        if won:
            stats.wins += 1
            stats.total_payout += payout
        else:
            stats.losses += 1
        
        stats.total_stake += stake
        stats.recent_results.append({"won": won, "payout": payout, "stake": stake, "time": time.time()})
    
    def on_drift_detected(self):
        """
        Called when concept drift is detected.
        Increases exploration to find durations that work in the new regime.
        """
        self._last_drift_time = time.time()
        self._drift_mode = True
        
        # Boost exploration rate temporarily
        self.epsilon = min(0.4, self.epsilon * 2)
        
        logger.info(
            f"Drift detected → boosting duration exploration (epsilon={self.epsilon:.0%})"
        )
    
    def on_drift_resolved(self):
        """Called when drift has passed. Returns to normal exploration."""
        self._drift_mode = False
        self.epsilon = max(0.05, self.epsilon / 2)
        logger.info("Drift resolved → reducing duration exploration")
    
    def get_recommendation(self) -> dict:
        """
        Get current duration recommendation with explanation.
        Useful for logging and dashboard display.
        """
        best = self._get_best_duration()
        stats = self.stats.get(best)
        
        return {
            "recommended_duration": best,
            "expected_value": stats.expected_value if stats else 0,
            "win_rate": stats.recent_win_rate if stats else 0.5,
            "confidence": stats.confidence if stats else 0,
            "total_trades": stats.total_trades if stats else 0,
            "exploration_rate": self.epsilon,
            "drift_mode": self._drift_mode,
            "all_durations": {
                d: {
                    "ev": round(s.expected_value, 4),
                    "win_rate": round(s.recent_win_rate, 3),
                    "trades": s.total_trades,
                    "confidence": round(s.confidence, 3),
                }
                for d, s in self.stats.items()
            },
        }
    
    def summary(self) -> dict:
        """Get summary of duration optimizer state."""
        rec = self.get_recommendation()
        return {
            "best_duration": rec["recommended_duration"],
            "best_ev": rec["expected_value"],
            "exploration_rate": round(self.epsilon, 3),
            "drift_mode": self._drift_mode,
            "total_selections": self._total_selections,
            "exploration_count": self._exploration_count,
            "duration_stats": rec["all_durations"],
        }
