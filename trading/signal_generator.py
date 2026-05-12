"""
Signal Generator (v5 — Dynamic Barrier Selection)
==================================================
FIXED confidence + FIXED EV + Dynamic barriers.

v5 Changes from v4.1:
- Confidence = ACTUAL observed win probability (not inflated mapping)
  Old: confidence = 0.50 + setup_score * 0.40 (always 70-90% = wrong!)
  New: confidence = observed_prob for the chosen barrier
- EV = observed_prob * (1 + payout_rate) - 1 (using REAL payout for barrier)
- Barrier is now DYNAMIC (from SetupDetector) — not always Over 4 / Under 5
- ML disagreement REDUCES confidence by 20% (doesn't block entirely)
- Minimum EV of 5% required to trade
- Minimum trade interval raised to 10 seconds
"""

import time
from dataclasses import dataclass
from typing import Optional

from config import (MIN_CONFIDENCE, MIN_EDGE_THRESHOLD, OVER_BARRIER,
                    UNDER_BARRIER, CONTRACT_TYPE_OVER, CONTRACT_TYPE_UNDER,
                    CONTRACT_DURATION, DYNAMIC_DURATION, KELLY_FRACTION,
                    MIN_TRADE_INTERVAL_SEC, MAX_DAILY_TRADES,
                    MIN_SETUP_SCORE, FORCE_TRADE_MIN_SETUP_SCORE,
                    MARTINGALE_MIN_CONFIDENCE, MARTINGALE_MIN_SETUP_SCORE,
                    ML_CONFIRMATION_WEIGHT, ML_DISAGREEMENT_BLOCKS,
                    MIN_DIGIT_FREQUENCY_EDGE, MIN_EV_FOR_TRADE,
                    DYNAMIC_BARRIERS)
from trading.setup_detector import Setup
from models.online_learner import Prediction
from utils.logger import setup_logger

logger = setup_logger("trading.signal_generator")


@dataclass
class Signal:
    """Generated trade signal."""
    direction: str          # "DIGITOVER" or "DIGITUNDER"
    barrier: int            # The digit barrier (DYNAMIC in v9!)
    confidence: float       # ACTUAL observed win probability (not inflated!)
    expected_value: float   # Expected profit per dollar staked
    kelly_fraction: float   # Kelly-optimal fraction of bankroll
    stake: float            # Actual stake amount in USD
    contract_duration: int  # Number of ticks
    timestamp: float
    features_snapshot: dict
    reason: str
    setup_score: float
    model_agreement: float
    is_martingale: bool = False
    # v9: Additional info for transparency
    natural_prob: float = 0.5       # Natural probability for this barrier
    observed_prob: float = 0.5      # Observed probability from digit frequencies
    payout_rate: float = 0.95       # Payout rate for this barrier
    z_score: float = 0.0           # Statistical significance


class SignalGenerator:
    """
    Converts setup + ML confirmation into trade signals.
    
    v5 Decision Flow:
    1. Check setup (from SetupDetector) — no setup = no trade
    2. Direction and barrier from setup (DYNAMIC — not always Over 4 / Under 5)
    3. Confidence = observed win probability for the chosen barrier
    4. ML confirmation: if agrees, boost; if disagrees, reduce by 20%
    5. EV = observed_prob * (1 + payout_rate) - 1
    6. Only trade if EV > 5% and setup is significant
    """
    
    def __init__(self, dynamic_barriers=True):
        self._last_signal_time = 0.0
        self._signals_generated = 0
        self._signals_skipped = 0
        self._skip_reasons = {}
        self._dynamic_barriers = dynamic_barriers
    
    def generate(self, setup: Setup, prediction: Prediction,
                 features: dict, payout: float, bankroll: float,
                 duration: int = 5,
                 model_in_drift: bool = False,
                 is_martingale: bool = False,
                 martingale_direction: str = None,
                 martingale_barrier: int = None) -> Optional[Signal]:
        """
        Generate a trade signal from setup + ML confirmation.
        
        v5 Key Changes:
        - Confidence = observed_prob (the REAL probability, not inflated)
        - EV uses the ACTUAL payout rate for the chosen barrier
        - Barrier is DYNAMIC (from setup), not always Over 4 / Under 5
        """
        # ─── SETUP GATE ───
        if not setup.active:
            self._skip("no_valid_setup")
            return None
        
        # ─── MARTINGALE GATE ───
        if is_martingale:
            if setup.setup_score < MARTINGALE_MIN_SETUP_SCORE:
                self._skip("martingale_weak_setup")
                logger.info(
                    f"MARTINGALE GATE: Setup score {setup.setup_score:.2f} "
                    f"< {MARTINGALE_MIN_SETUP_SCORE} — not recovering on weak setup"
                )
                return None
            
            # Direction must match the original losing trade
            if martingale_direction and setup.direction != martingale_direction:
                self._skip("martingale_direction_mismatch")
                logger.info(
                    f"MARTINGALE DIRECTION: Setup says {setup.direction} "
                    f"but martingale requires {martingale_direction} — skipping"
                )
                return None
            
            # v9: Barrier must also match (or be compatible) for martingale
            # If we lost on Over 7, we should recover on Over 7 (same barrier)
            if martingale_barrier is not None and setup.barrier != martingale_barrier:
                # Allow recovery on the same direction but different barrier if it has better EV
                # This gives the bot flexibility to find a better recovery opportunity
                pass  # Allow different barrier for now — the direction match is more important
        
        # ─── DIRECTION AND BARRIER FROM SETUP ───
        direction = setup.direction
        barrier = setup.barrier  # v9: DYNAMIC barrier from setup!
        
        # ─── CONFIDENCE = OBSERVED WIN PROBABILITY (v9: FIXED!) ───
        # The OLD system mapped setup_score to [0.50, 0.90], always inflating confidence.
        # Now confidence = the ACTUAL observed probability for this barrier.
        # For Over 8: might be 0.14 (14% observed vs 10% natural = strong edge)
        # For Over 4: might be 0.55 (55% observed vs 50% natural = weak edge)
        confidence = setup.observed_prob
        
        # ─── ML CONFIRMATION ───
        # The Logistic Regression model confirms or disputes the direction.
        # v9: ML disagreement REDUCES confidence by 20%, doesn't block entirely.
        # The ML model was trained on Over 4 / Under 5 data, so it's less reliable
        # for other barriers. But its opinion still matters.
        ml_agrees = (
            (direction == CONTRACT_TYPE_OVER and prediction.prob_over > prediction.prob_under) or
            (direction == CONTRACT_TYPE_UNDER and prediction.prob_under > prediction.prob_over)
        )
        
        if ML_DISAGREEMENT_BLOCKS and not ml_agrees and not is_martingale:
            # Old behavior: block on ML disagreement
            self._skip("ml_disagrees_blocked")
            logger.info(
                f"ML DISAGREES BLOCKED: Setup says {direction.replace('DIGIT','')} "
                f"but ML says {'Over' if prediction.prob_over > prediction.prob_under else 'Under'} "
                f"— skipping"
            )
            return None
        
        if ml_agrees:
            # ML agrees — small confidence boost
            ml_prob = prediction.prob_over if direction == CONTRACT_TYPE_OVER else prediction.prob_under
            confidence = confidence * (1 - ML_CONFIRMATION_WEIGHT) + ml_prob * ML_CONFIRMATION_WEIGHT
            ml_status = "ML_AGREES"
        else:
            # ML disagrees — reduce confidence by 20% (but don't block)
            confidence = confidence * 0.80
            ml_status = "ML_DISAGREES(-20%)"
        
        # Clamp confidence
        confidence = max(0.05, min(0.99, confidence))
        
        # ─── EV CALCULATION (v9: FIXED!) ───
        # EV = observed_prob * (1 + payout_rate) - 1
        # Uses the ACTUAL payout rate for this barrier, not a generic value.
        payout_rate = setup.payout_rate
        ev = confidence * (1 + payout_rate) - 1.0
        
        # ─── COOLDOWN CHECK ───
        time_since_last = time.time() - self._last_signal_time
        if time_since_last < MIN_TRADE_INTERVAL_SEC:
            self._skip("cooldown")
            return None
        
        # ─── DAILY TRADE LIMIT ───
        if MAX_DAILY_TRADES > 0 and self._signals_generated >= MAX_DAILY_TRADES:
            self._skip("daily_limit")
            return None
        
        # ─── CONFIDENCE CHECK ───
        if confidence < MIN_CONFIDENCE:
            self._skip("low_confidence")
            return None
        
        # ─── EV CHECK (v9: Must be positive AND > minimum threshold) ───
        if ev < MIN_EV_FOR_TRADE:
            self._skip("ev_below_minimum")
            if ev > 0:
                logger.debug(
                    f"EV TOO LOW: {ev:+.1%} < {MIN_EV_FOR_TRADE:.0%} minimum "
                    f"for {direction.replace('DIGIT','')}{barrier}"
                )
            return None
        
        if ev < MIN_EDGE_THRESHOLD:
            self._skip("negative_ev")
            return None
        
        # ─── DRIFT CHECK ───
        if model_in_drift and not ml_agrees and setup.setup_score < FORCE_TRADE_MIN_SETUP_SCORE:
            self._skip("drift_no_ml_support")
            return None
        
        # ─── GENERATE SIGNAL ───
        kelly = self._kelly_fraction(confidence, payout_rate)
        stake = self._calculate_stake(
            kelly, bankroll,
            confidence=confidence,
            setup_score=setup.setup_score,
            ev=ev,
        )
        
        contract_duration = max(2, duration)  # Minimum 2 ticks
        
        dir_label = "Over" if direction == CONTRACT_TYPE_OVER else "Under"
        reason = (
            f"{dir_label}{barrier}: obs_prob={setup.observed_prob:.1%} "
            f"(natural={setup.natural_prob:.0%}) "
            f"conf={confidence:.0%} EV={ev:+.1%} "
            f"payout={payout_rate:.1%} z={setup.z_score:.1f} "
            f"dur={contract_duration}t "
            f"{ml_status} "
            f"setup={setup.setup_score:.2f} "
            f"stake=${stake:.2f}"
        )
        
        signal = Signal(
            direction=direction,
            barrier=barrier,
            confidence=round(confidence, 4),
            expected_value=round(ev, 4),
            kelly_fraction=kelly,
            stake=stake,
            contract_duration=contract_duration,
            timestamp=time.time(),
            features_snapshot=features.copy(),
            reason=reason,
            setup_score=setup.setup_score,
            model_agreement=1.0 if ml_agrees else 0.5,
            natural_prob=setup.natural_prob,
            observed_prob=setup.observed_prob,
            payout_rate=payout_rate,
            z_score=setup.z_score,
        )
        
        self._signals_generated += 1
        self._last_signal_time = time.time()
        logger.info(f"SIGNAL: {signal.reason}")
        return signal
    
    def _kelly_fraction(self, win_prob: float, payout_rate: float) -> float:
        """Kelly Criterion: f* = (b*p - q) / b"""
        from config import MAX_BANKROLL_PER_TRADE
        
        b = payout_rate  # You win $payout_rate per $1 staked
        p = win_prob
        q = 1 - p
        
        if b <= 0:
            return 0.0
        
        kelly = (b * p - q) / b
        fractional = kelly * KELLY_FRACTION
        return max(0.01, min(fractional, MAX_BANKROLL_PER_TRADE))
    
    def _calculate_stake(self, kelly_fraction: float, bankroll: float,
                          confidence: float = 0.5, setup_score: float = 0.5,
                          ev: float = 0.0) -> float:
        """
        Convert Kelly fraction to stake with quality-based sizing.
        
        v9: For low-probability barriers (e.g., Over 8), the stake should be
        SMALLER because the variance is much higher. A 10% probability bet
        with 900% payout can have long losing streaks.
        """
        from config import MIN_STAKE, MAX_STAKE, MAX_BANKROLL_PER_TRADE, MIN_CONFIDENCE
        
        stake = kelly_fraction * bankroll
        
        # Setup quality boost
        if setup_score >= MIN_SETUP_SCORE:
            setup_range = min(1.0, (setup_score - MIN_SETUP_SCORE) / (1.0 - MIN_SETUP_SCORE))
            setup_boost = 1.0 + setup_range * 0.5  # 1.0x to 1.5x
        else:
            setup_boost = 0.5
        
        # EV boost (higher EV = more stake, but capped conservatively)
        if ev >= 0.20:
            ev_boost = 1.5
        elif ev >= 0.10:
            ev_boost = 1.0 + 0.5 * (ev - 0.10) / 0.10
        elif ev >= MIN_EV_FOR_TRADE:
            ev_boost = 0.8 + 0.2 * (ev - MIN_EV_FOR_TRADE) / (0.10 - MIN_EV_FOR_TRADE)
        else:
            ev_boost = 0.5
        
        # Variance adjustment: lower probability = higher variance = smaller stake
        # For a barrier with 10% natural prob, stake should be ~1/3 of Over 4 stake
        # This prevents blowing up the bankroll on long losing streaks
        if confidence < 0.20:
            variance_adj = 0.25  # Very low prob — small stakes
        elif confidence < 0.30:
            variance_adj = 0.40
        elif confidence < 0.50:
            variance_adj = 0.60
        else:
            variance_adj = 1.0  # 50%+ prob — normal sizing
        
        # Combined
        total_boost = setup_boost * ev_boost * variance_adj
        stake = stake * total_boost
        
        # Hard limits
        max_allowed = bankroll * MAX_BANKROLL_PER_TRADE
        stake = min(stake, max_allowed)
        stake = min(stake, MAX_STAKE)
        stake = max(stake, MIN_STAKE)
        return round(stake, 2)
    
    def _skip(self, reason: str):
        """Track why a signal was skipped."""
        self._skip_reasons[reason] = self._skip_reasons.get(reason, 0) + 1
        self._signals_skipped += 1
    
    def reset_daily(self):
        """Reset daily counters."""
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
