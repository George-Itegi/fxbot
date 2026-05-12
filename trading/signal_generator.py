"""
Signal Generator (v4 — Structured Quality Trading)
====================================================
RULE-BASED primary + ML confirmation.

The decision flow now mirrors the manual trading process:
1. SetupDetector checks trend + digit frequency → is there a good setup?
2. Direction is determined by the SETUP (not the ML model)
3. ML (Logistic Regression) CONFIRMS or adjusts confidence
4. Observation phase determines duration (watch digits 20-30s)
5. If setup is good → trade with confidence, let martingale recover

Key changes from v3:
- Digit frequency Over/Under split is PRIMARY direction signal
- ML model is a CONFIRMATION signal, not the decision-maker
- Setup quality score replaces model agreement as the main quality gate
- Observation phase determines duration
- Less filtering within a valid setup (trust the setup)
- Model agreement is always 1.0 (single model)
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
                    TREND_CONFIDENCE_REDUCTION, TREND_SIGNAL_SCORE_REDUCTION,
                    ML_CONFIRMATION_WEIGHT, ML_DISAGREEMENT_PENALTY,
                    MIN_DIGIT_FREQUENCY_EDGE)
from trading.setup_detector import Setup
from models.online_learner import Prediction
from utils.logger import setup_logger

logger = setup_logger("trading.signal_generator")


@dataclass
class Signal:
    """Generated trade signal."""
    direction: str          # "DIGITOVER" or "DIGITUNDER"
    barrier: int            # The digit barrier
    confidence: float       # Combined confidence (rule-based + ML confirmation)
    expected_value: float   # Expected profit per dollar risked
    kelly_fraction: float   # Kelly-optimal fraction of bankroll
    stake: float            # Actual stake amount in USD
    contract_duration: int  # Number of ticks (determined by observation phase)
    timestamp: float
    features_snapshot: dict  # Features at signal time
    reason: str             # Human-readable reason
    setup_score: float      # Setup quality score (0-1)
    model_agreement: float  # Always 1.0 now (single model)
    is_martingale: bool = False


class SignalGenerator:
    """
    Converts setup + ML confirmation into trade signals.
    
    v4 Decision Flow:
    1. Check setup (from SetupDetector) — no setup = no trade
    2. Determine direction from setup (trend + digit frequency aligned)
    3. Get ML model prediction as CONFIRMATION
    4. Adjust confidence based on ML agreement/disagreement
    5. Calculate EV and generate signal if conditions met
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
                 martingale_direction: str = None) -> Optional[Signal]:
        """
        Generate a trade signal from setup + ML confirmation.
        
        Args:
            setup: SetupDetector's evaluation of the market
            prediction: ML model's prediction (confirmation signal)
            features: Feature dict at prediction time
            payout: Current payout ratio
            bankroll: Current account balance
            duration: Contract duration in ticks (from observation phase)
            model_in_drift: Whether the model is in drift state
            is_martingale: Whether this is a martingale recovery trade
            martingale_direction: Required direction for martingale recovery
        
        Returns:
            Signal if conditions met, None otherwise.
        """
        # ─── SETUP GATE (FIRST CHECK) ───
        # No valid setup = no trade. Period.
        if not setup.active:
            self._skip("no_valid_setup")
            return None
        
        # ─── MARTINGALE GATE ───
        # During martingale, setup must STILL be valid and direction must persist
        if is_martingale:
            # Setup must still meet minimum quality
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
            
            # Confidence must be reasonable (lowered from 80% since we trust setups)
            best_conf = max(prediction.prob_over, prediction.prob_under)
            if best_conf < MARTINGALE_MIN_CONFIDENCE:
                self._skip("martingale_low_confidence")
                logger.info(
                    f"MARTINGALE GATE: ML confidence {best_conf:.0%} "
                    f"< {MARTINGALE_MIN_CONFIDENCE:.0%} — waiting"
                )
                return None
        
        # ─── DIRECTION from SETUP ───
        # The setup determines direction (trend + frequency aligned)
        direction = setup.direction
        if direction == CONTRACT_TYPE_OVER:
            barrier = OVER_BARRIER
        else:
            barrier = UNDER_BARRIER
        
        # ─── ML CONFIRMATION ───
        # The Logistic Regression model confirms or disputes the direction.
        # Rules are PRIMARY — ML only adjusts confidence.
        ml_agrees = (
            (direction == CONTRACT_TYPE_OVER and prediction.prob_over > prediction.prob_under) or
            (direction == CONTRACT_TYPE_UNDER and prediction.prob_under > prediction.prob_over)
        )
        
        # Base confidence from the setup quality
        # High setup score = high base confidence
        base_confidence = 0.50 + (setup.setup_score * 0.40)  # 0.50 to 0.90
        
        # Adjust confidence based on ML confirmation
        if ml_agrees:
            # ML agrees — boost confidence
            ml_prob = prediction.prob_over if direction == CONTRACT_TYPE_OVER else prediction.prob_under
            # Blend: base confidence weighted by (1 - ML_CONFIRMATION_WEIGHT) + ML weighted
            confidence = (
                base_confidence * (1 - ML_CONFIRMATION_WEIGHT) +
                ml_prob * ML_CONFIRMATION_WEIGHT
            )
            ml_status = "ML_AGREES"
        else:
            # ML disagrees — reduce confidence
            confidence = base_confidence - ML_DISAGREEMENT_PENALTY
            ml_status = "ML_DISAGREES"
        
        # Clamp confidence
        confidence = max(0.40, min(0.95, confidence))
        
        # ─── Calculate Expected Value ───
        if direction == CONTRACT_TYPE_OVER:
            ev = confidence * payout - (1 - confidence) * 1.0
        else:
            ev = confidence * payout - (1 - confidence) * 1.0
        
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
        effective_min_confidence = max(0.45, MIN_CONFIDENCE - TREND_CONFIDENCE_REDUCTION)
        if confidence < effective_min_confidence:
            self._skip("low_confidence")
            return None
        
        # ─── EV CHECK ───
        if ev < MIN_EDGE_THRESHOLD:
            self._skip("negative_ev")
            return None
        
        # ─── DRIFT CHECK ───
        # With rule-based primary, drift matters less — but still skip if model is very confused
        if model_in_drift and not ml_agrees and setup.setup_score < FORCE_TRADE_MIN_SETUP_SCORE:
            self._skip("drift_no_ml_support")
            return None
        
        # ─── GENERATE SIGNAL ───
        kelly = self._kelly_fraction(confidence, payout)
        stake = self._calculate_stake(
            kelly, bankroll,
            confidence=confidence,
            setup_score=setup.setup_score,
            ev=ev,
        )
        
        # Duration from observation phase
        contract_duration = duration
        if contract_duration < 2:
            contract_duration = 2  # Minimum 2 ticks
        
        direction_label = "Over" if direction == CONTRACT_TYPE_OVER else "Under"
        reason = (
            f"{direction_label}: setup={setup.setup_score:.2f} "
            f"conf={confidence:.0%} EV={ev:+.3f} "
            f"payout={payout:.0%} dur={contract_duration}t "
            f"{ml_status} "
            f"Over={setup.over_freq:.1%} Under={setup.under_freq:.1%} "
            f"edge={setup.freq_edge:.1%} "
            f"trend={'UP' if setup.trend_regime==1 else 'DOWN'}({setup.trend_strength:.1f}) "
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
            model_agreement=1.0,  # Single model = always 1.0
        )
        
        self._signals_generated += 1
        self._last_signal_time = time.time()
        logger.info(f"SIGNAL: {signal.reason}")
        return signal
    
    def _kelly_fraction(self, win_prob: float, payout: float) -> float:
        """Kelly Criterion: f* = (b*p - q) / b"""
        from config import MAX_BANKROLL_PER_TRADE
        
        b = payout
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
        Convert Kelly fraction to stake with setup-quality-based sizing.
        
        Higher setup score = higher stake (we trust good setups).
        """
        from config import MIN_STAKE, MAX_STAKE, MAX_BANKROLL_PER_TRADE, MIN_CONFIDENCE
        
        # Base stake from Kelly
        stake = kelly_fraction * bankroll
        
        # Setup quality boost: score 0.60 = 1.0x, 0.80 = 1.5x, 1.0 = 2.0x
        if setup_score >= MIN_SETUP_SCORE:
            setup_range = min(1.0, (setup_score - MIN_SETUP_SCORE) / (1.0 - MIN_SETUP_SCORE))
            setup_boost = 1.0 + setup_range * 1.0  # 1.0x to 2.0x
        else:
            setup_boost = 0.5
        
        # Confidence boost
        if confidence > MIN_CONFIDENCE:
            conf_range = min(1.0, (confidence - MIN_CONFIDENCE) / (0.90 - MIN_CONFIDENCE))
            confidence_boost = 1.0 + conf_range * 1.5  # 1.0x to 2.5x
        else:
            confidence_boost = 0.8
        
        # EV boost
        if ev > MIN_EDGE_THRESHOLD:
            ev_range = min(1.0, (ev - MIN_EDGE_THRESHOLD) / 0.20)
            ev_boost = 1.0 + ev_range * 0.5  # 1.0x to 1.5x
        else:
            ev_boost = 0.9
        
        # Combined (capped at 6x total)
        total_boost = min(6.0, setup_boost * confidence_boost * ev_boost)
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
