"""
Signal Generator (v6 — Conservative Edge Detection)
====================================================
v10: ML disagreement BLOCKS, confidence uses Bayesian-adjusted prob, fixed 5t.

v6 Changes from v5:
- Confidence = BAYESIAN-ADJUSTED observed probability (not raw)
  Raw observed on Over 8 might be 40%, but Bayesian-adjusted is 30%
  This prevents overconfident trading on noisy observations
- ML disagreement now BLOCKS trades again (v9's 20% reduction was too weak)
  On synthetic indices, we need EVERY confirmation we can get
- EV uses Bayesian-adjusted probability (not raw observed)
- Fixed 5-tick duration — no more 2t/3t/5t chaos
- Higher minimums: 35% confidence, 8% EV
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
    barrier: int            # The digit barrier
    confidence: float       # Bayesian-adjusted observed win probability
    expected_value: float   # Expected profit per dollar staked (using adjusted prob)
    kelly_fraction: float   # Kelly-optimal fraction of bankroll
    stake: float            # Actual stake amount in USD
    contract_duration: int  # Number of ticks (always 5 in v10)
    timestamp: float
    features_snapshot: dict
    reason: str
    setup_score: float
    model_agreement: float
    is_martingale: bool = False
    # v10: Additional info
    natural_prob: float = 0.5       # Natural probability for this barrier
    observed_prob: float = 0.5      # Raw observed probability from digit frequencies
    adjusted_prob: float = 0.5      # v10: Bayesian-adjusted probability (used for confidence)
    payout_rate: float = 0.95       # Payout rate for this barrier
    z_score: float = 0.0           # Statistical significance


class SignalGenerator:
    """
    Converts setup + ML confirmation into trade signals.
    
    v6 Decision Flow:
    1. Check setup (from SetupDetector) — no setup = no trade
    2. Direction and barrier from setup (MODERATE barriers only)
    3. Confidence = Bayesian-adjusted observed probability (not raw!)
    4. ML confirmation: if BLOCKS on disagreement (v10: restored)
    5. EV = adjusted_prob * (1 + payout_rate) - 1
    6. Only trade if EV > 8% and setup is significant
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
        
        v6 Key Changes:
        - Confidence = Bayesian-adjusted probability (not raw observed)
        - EV uses adjusted probability
        - ML disagreement BLOCKS (not 20% reduction)
        - Fixed 5-tick duration
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
            
            # v10: Barrier must also match for martingale
            # If we lost on Over 5, we should recover on Over 5 (same barrier)
            if martingale_barrier is not None and setup.barrier != martingale_barrier:
                self._skip("martingale_barrier_mismatch")
                logger.info(
                    f"MARTINGALE BARRIER: Setup says {setup.direction}{setup.barrier} "
                    f"but martingale requires barrier {martingale_barrier} — skipping"
                )
                return None
        
        # ─── DIRECTION AND BARRIER FROM SETUP ───
        direction = setup.direction
        barrier = setup.barrier
        
        # ─── CONFIDENCE = BAYESIAN-ADJUSTED PROBABILITY (v10) ───
        # v9 used raw observed_prob which was noisy (40% on Over 8 when natural is 10%).
        # v10 uses the Bayesian-adjusted probability which shrinks toward natural.
        # For Over 5 (40% natural) with 50% observed: adjusted = ~47%
        # For Over 6 (30% natural) with 40% observed: adjusted = ~37%
        confidence = setup.adjusted_prob
        
        # ─── ML CONFIRMATION ───
        # v10: ML disagreement now BLOCKS trades (restored from v8).
        # v9's 20% reduction was too weak — it allowed trades that the ML model
        # disagreed with, and those trades lost at a very high rate.
        ml_agrees = (
            (direction == CONTRACT_TYPE_OVER and prediction.prob_over > prediction.prob_under) or
            (direction == CONTRACT_TYPE_UNDER and prediction.prob_under > prediction.prob_over)
        )
        
        if ML_DISAGREEMENT_BLOCKS and not ml_agrees and not is_martingale:
            # v10: Block on ML disagreement (except during martingale recovery)
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
            # v10: During martingale, ML disagreement still reduces confidence
            confidence = confidence * 0.80
            ml_status = "ML_DISAGREES(-20%[martingale])"
        
        # Clamp confidence
        confidence = max(0.05, min(0.99, confidence))
        
        # ─── EV CALCULATION (v10: uses Bayesian-adjusted confidence) ───
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
        
        # ─── EV CHECK (v10: Must be positive AND > 8% minimum) ───
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
        
        # v10: Fixed 5-tick duration
        contract_duration = 5
        
        dir_label = "Over" if direction == CONTRACT_TYPE_OVER else "Under"
        reason = (
            f"{dir_label}{barrier}: obs={setup.observed_prob:.1%} "
            f"adj={setup.adjusted_prob:.1%} (natural={setup.natural_prob:.0%}) "
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
            adjusted_prob=setup.adjusted_prob,
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
        
        v10: Simpler sizing — moderate barriers have manageable variance,
        so we don't need the complex variance adjustment from v9.
        """
        from config import MIN_STAKE, MAX_STAKE, MAX_BANKROLL_PER_TRADE
        
        stake = kelly_fraction * bankroll
        
        # Setup quality boost
        if setup_score >= MIN_SETUP_SCORE:
            setup_range = min(1.0, (setup_score - MIN_SETUP_SCORE) / (1.0 - MIN_SETUP_SCORE))
            setup_boost = 1.0 + setup_range * 0.3  # 1.0x to 1.3x (v10: less aggressive)
        else:
            setup_boost = 0.5
        
        # EV boost
        if ev >= 0.20:
            ev_boost = 1.3
        elif ev >= MIN_EV_FOR_TRADE:
            ev_boost = 0.8 + 0.5 * (ev - MIN_EV_FOR_TRADE) / (0.20 - MIN_EV_FOR_TRADE)
        else:
            ev_boost = 0.5
        
        # Combined
        total_boost = setup_boost * ev_boost
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
