"""
Signal Generator
=================
Converts model probabilities into trade decisions.
This is where most bots fail — they trade too often.
The key: ONLY trade when your edge is CLEAR.
"""

import time
from dataclasses import dataclass
from typing import Optional

from config import (MIN_CONFIDENCE, MIN_EDGE_THRESHOLD, OVER_BARRIER,
                    UNDER_BARRIER, CONTRACT_TYPE_OVER, CONTRACT_TYPE_UNDER,
                    CONTRACT_DURATION, KELLY_FRACTION, COOLDOWN_AFTER_LOSS_TICKS,
                    MIN_TRADE_INTERVAL_SEC, MAX_DAILY_TRADES)
from models.online_learner import Prediction
from utils.logger import setup_logger

logger = setup_logger("trading.signal_generator")


@dataclass
class Signal:
    """Generated trade signal."""
    direction: str          # "DIGITOVER" or "DIGITUNDER"
    barrier: int            # The digit barrier
    confidence: float       # Model confidence (0-1)
    expected_value: float   # Expected profit per dollar risked
    kelly_fraction: float   # Kelly-optimal fraction of bankroll
    stake: float            # Actual stake amount in USD
    contract_duration: int  # Number of ticks
    timestamp: float
    features_snapshot: dict  # Features at signal time (for learning later)
    reason: str             # Human-readable reason for the signal


class SignalGenerator:
    """
    Converts model predictions + market data into trade signals.
    
    Decision flow:
    1. Get model prediction (probability of Over)
    2. Check current payout for both Over and Under
    3. Calculate expected value for each direction
    4. If EV > threshold AND confidence > threshold → generate signal
    5. Otherwise → no trade (this is the most common outcome)
    """
    
    def __init__(self):
        self._last_signal_time = 0.0
        self._signals_generated = 0
        self._signals_skipped = 0
        self._skip_reasons = {}
    
    def generate(self, prediction: Prediction, features: dict,
                 payout: float, bankroll: float,
                 model_in_drift: bool = False) -> Optional[Signal]:
        """
        Generate a trade signal from model prediction + current payout.
        
        Args:
            prediction: Model's output (probabilities, confidence)
            features: Feature dict at prediction time (stored for later learning)
            payout: Current payout ratio for the contract (e.g., 0.85 = 85%)
            bankroll: Current account balance
            model_in_drift: Whether the model is in drift state
        
        Returns:
            Signal if conditions met, None otherwise.
        """
        # ─── Pre-trade Checks ───
        
        # Check drift
        if model_in_drift:
            self._skip("model_drift")
            return None
        
        # Check minimum confidence
        if prediction.confidence < MIN_CONFIDENCE:
            self._skip("low_confidence")
            return None
        
        # Check cooldown
        time_since_last = time.time() - self._last_signal_time
        if time_since_last < MIN_TRADE_INTERVAL_SEC:
            self._skip("cooldown")
            return None
        
        # Check daily trade limit
        if self._signals_generated >= MAX_DAILY_TRADES:
            self._skip("daily_limit")
            return None
        
        # ─── Calculate Expected Value ───
        # EV = P(win) × payout - P(lose) × 1
        # (expressed per dollar risked)
        
        ev_over = prediction.prob_over * payout - prediction.prob_under * 1.0
        ev_under = prediction.prob_under * payout - prediction.prob_over * 1.0
        
        signal = None
        
        if ev_over > MIN_EDGE_THRESHOLD and prediction.prob_over >= MIN_CONFIDENCE:
            kelly = self._kelly_fraction(prediction.prob_over, payout)
            stake = self._calculate_stake(kelly, bankroll)
            reason = (
                f"Over: prob={prediction.prob_over:.2%}, "
                f"EV={ev_over:+.3f}, payout={payout:.0%}"
            )
            signal = Signal(
                direction=CONTRACT_TYPE_OVER,
                barrier=OVER_BARRIER,
                confidence=prediction.prob_over,
                expected_value=round(ev_over, 4),
                kelly_fraction=kelly,
                stake=stake,
                contract_duration=CONTRACT_DURATION,
                timestamp=time.time(),
                features_snapshot=features.copy(),
                reason=reason,
            )
        
        elif ev_under > MIN_EDGE_THRESHOLD and prediction.prob_under >= MIN_CONFIDENCE:
            kelly = self._kelly_fraction(prediction.prob_under, payout)
            stake = self._calculate_stake(kelly, bankroll)
            reason = (
                f"Under: prob={prediction.prob_under:.2%}, "
                f"EV={ev_under:+.3f}, payout={payout:.0%}"
            )
            signal = Signal(
                direction=CONTRACT_TYPE_UNDER,
                barrier=UNDER_BARRIER,
                confidence=prediction.prob_under,
                expected_value=round(ev_under, 4),
                kelly_fraction=kelly,
                stake=stake,
                contract_duration=CONTRACT_DURATION,
                timestamp=time.time(),
                features_snapshot=features.copy(),
                reason=reason,
            )
        else:
            self._skip("insufficient_edge")
        
        if signal:
            self._signals_generated += 1
            self._last_signal_time = time.time()
            logger.info(f"📈 SIGNAL: {signal.reason}")
        else:
            self._signals_skipped += 1
        
        return signal
    
    def _kelly_fraction(self, win_prob: float, payout: float) -> float:
        """
        Kelly Criterion: f* = (b×p - q) / b
        where b = payout ratio, p = win prob, q = 1-p
        
        Returns fractional Kelly (divided by KELLY_FRACTION for safety).
        """
        b = payout
        p = win_prob
        q = 1 - p
        
        if b <= 0:
            return 0.0
        
        kelly = (b * p - q) / b
        
        # Apply fractional Kelly for safety
        fractional = kelly * KELLY_FRACTION
        
        # Clamp to safe range
        return max(0.01, min(fractional, 0.05))
    
    def _calculate_stake(self, kelly_fraction: float, bankroll: float) -> float:
        """Convert Kelly fraction to actual stake amount."""
        from config import MIN_STAKE, MAX_STAKE
        stake = kelly_fraction * bankroll
        return max(MIN_STAKE, min(stake, MAX_STAKE))
    
    def _skip(self, reason: str):
        """Track why a signal was skipped."""
        self._skip_reasons[reason] = self._skip_reasons.get(reason, 0) + 1
    
    def reset_daily(self):
        """Reset daily counters (call at start of each day)."""
        self._signals_generated = 0
        self._skip_reasons = {}
        logger.info("Daily signal counters reset")
    
    def summary(self) -> dict:
        return {
            "signals_generated": self._signals_generated,
            "signals_skipped": self._signals_skipped,
            "skip_rate": (
                self._signals_skipped / (self._signals_generated + self._signals_skipped)
                if (self._signals_generated + self._signals_skipped) > 0 else 0
            ),
            "skip_reasons": self._skip_reasons,
        }
