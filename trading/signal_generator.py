"""
Signal Generator (v2 — Dynamic Duration)
==========================================
Converts model probabilities into trade decisions.
Now supports DYNAMIC contract duration via DurationOptimizer.

This is where most bots fail — they trade too often.
The key: ONLY trade when your edge is CLEAR.
"""

import time
from dataclasses import dataclass
from typing import Optional

from config import (MIN_CONFIDENCE, MIN_EDGE_THRESHOLD, OVER_BARRIER,
                    UNDER_BARRIER, CONTRACT_TYPE_OVER, CONTRACT_TYPE_UNDER,
                    CONTRACT_DURATION, KELLY_FRACTION, COOLDOWN_AFTER_LOSS_TICKS,
                    MIN_TRADE_INTERVAL_SEC, MAX_DAILY_TRADES,
                    DYNAMIC_DURATION)
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
    contract_duration: int  # Number of ticks (DYNAMIC — set by DurationOptimizer)
    timestamp: float
    features_snapshot: dict  # Features at signal time (for learning later)
    reason: str             # Human-readable reason for the signal
    model_agreement: float  # Fraction of sub-models that agree (ensemble)


class SignalGenerator:
    """
    Converts model predictions + market data into trade signals.
    
    Decision flow:
    1. Get model prediction (probability of Over)
    2. Check current payout for both Over and Under
    3. Calculate expected value for each direction
    4. If EV > threshold AND confidence > threshold → generate signal
    5. Otherwise → no trade (this is the most common outcome)
    
    v2 additions:
    - Dynamic contract duration via DurationOptimizer
    - Model agreement score from ensemble
    - Enhanced signal reason logging
    """
    
    def __init__(self, duration_optimizer=None, dynamic_barriers=True,
                 min_model_agreement=0.67):
        self._last_signal_time = 0.0
        self._signals_generated = 0
        self._signals_skipped = 0
        self._skip_reasons = {}
        self._duration_optimizer = duration_optimizer
        self._dynamic_barriers = dynamic_barriers
        self._min_model_agreement = min_model_agreement
    
    def generate(self, prediction: Prediction, features: dict,
                 payout: float, bankroll: float,
                 model_in_drift: bool = False) -> Optional[Signal]:
        """
        Generate a trade signal from model prediction + current payout.
        
        RULE: When ALL ensemble models agree 100%, a trade MUST occur.
        This overrides confidence thresholds and EV thresholds.
        The models agreeing 100% is the strongest possible signal.
        
        Args:
            prediction: Model's output (probabilities, confidence)
            features: Feature dict at prediction time (stored for later learning)
            payout: Current payout ratio for the contract (e.g., 0.85 = 85%)
            bankroll: Current account balance
            model_in_drift: Whether the model is in drift state
        
        Returns:
            Signal if conditions met, None otherwise.
        """
        # ─── FORCE TRADE: All 3 models agree 100% ───
        # This is the STRONGEST signal — override everything except cooldown
        all_models_agree = prediction.model_agreement >= 1.0
        
        if all_models_agree:
            # Even with 100% agreement, respect minimum cooldown
            time_since_last = time.time() - self._last_signal_time
            if time_since_last < 1:  # Only 1s cooldown for 100% agreement
                self._skip("cooldown")
                return None
            
            # Force a trade — calculate which direction
            ev_over = prediction.prob_over * payout - prediction.prob_under * 1.0
            ev_under = prediction.prob_under * payout - prediction.prob_over * 1.0
            
            # Select duration
            if self._duration_optimizer and DYNAMIC_DURATION:
                duration = self._duration_optimizer.select_duration()
            else:
                duration = CONTRACT_DURATION
            
            # Determine direction — go with the ensemble majority
            if prediction.prob_over >= prediction.prob_under:
                direction = CONTRACT_TYPE_OVER
                confidence = prediction.prob_over
                ev = ev_over
                barrier = OVER_BARRIER
            else:
                direction = CONTRACT_TYPE_UNDER
                confidence = prediction.prob_under
                ev = ev_under
                barrier = UNDER_BARRIER
            
            kelly = self._kelly_fraction(confidence, payout)
            stake = self._calculate_stake(kelly, bankroll,
                                          confidence=confidence,
                                          agreement=prediction.model_agreement,
                                          ev=max(ev, 0.01))
            
            reason = (
                f"FORCED (100% AGREEMENT): {direction.replace('DIGIT', '').title()} "
                f"prob={confidence:.2%}, EV={ev:+.3f}, payout={payout:.0%}, "
                f"dur={duration}t, agree=100%, stake=${stake:.2f}"
            )
            
            signal = Signal(
                direction=direction,
                barrier=barrier,
                confidence=confidence,
                expected_value=round(max(ev, 0.01), 4),
                kelly_fraction=kelly,
                stake=stake,
                contract_duration=duration,
                timestamp=time.time(),
                features_snapshot=features.copy(),
                reason=reason,
                model_agreement=prediction.model_agreement,
            )
            
            self._signals_generated += 1
            self._last_signal_time = time.time()
            logger.info(f"SIGNAL: {signal.reason}")
            return signal
        
        # ─── Normal Pre-trade Checks (when NOT 100% agreement) ───
        
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
        
        # Check daily trade limit (0 = unlimited)
        if MAX_DAILY_TRADES > 0 and self._signals_generated >= MAX_DAILY_TRADES:
            self._skip("daily_limit")
            return None
        
        # Check model agreement (ensemble must agree enough)
        if prediction.model_agreement < self._min_model_agreement:
            self._skip("low_agreement")
            return None
        
        # ─── Calculate Expected Value ───
        # EV = P(win) * payout - P(lose) * 1
        # (expressed per dollar risked)
        
        ev_over = prediction.prob_over * payout - prediction.prob_under * 1.0
        ev_under = prediction.prob_under * payout - prediction.prob_over * 1.0
        
        # ─── Select Contract Duration ───
        if self._duration_optimizer and DYNAMIC_DURATION:
            duration = self._duration_optimizer.select_duration()
        else:
            duration = CONTRACT_DURATION
        
        signal = None
        
        if ev_over > MIN_EDGE_THRESHOLD and prediction.prob_over >= MIN_CONFIDENCE:
            kelly = self._kelly_fraction(prediction.prob_over, payout)
            stake = self._calculate_stake(kelly, bankroll,
                                          confidence=prediction.prob_over,
                                          agreement=prediction.model_agreement,
                                          ev=ev_over)
            reason = (
                f"Over: prob={prediction.prob_over:.2%}, "
                f"EV={ev_over:+.3f}, payout={payout:.0%}, "
                f"dur={duration}t, agree={prediction.model_agreement:.0%}, "
                f"stake=${stake:.2f}"
            )
            signal = Signal(
                direction=CONTRACT_TYPE_OVER,
                barrier=OVER_BARRIER,
                confidence=prediction.prob_over,
                expected_value=round(ev_over, 4),
                kelly_fraction=kelly,
                stake=stake,
                contract_duration=duration,
                timestamp=time.time(),
                features_snapshot=features.copy(),
                reason=reason,
                model_agreement=prediction.model_agreement,
            )
        
        elif ev_under > MIN_EDGE_THRESHOLD and prediction.prob_under >= MIN_CONFIDENCE:
            kelly = self._kelly_fraction(prediction.prob_under, payout)
            stake = self._calculate_stake(kelly, bankroll,
                                          confidence=prediction.prob_under,
                                          agreement=prediction.model_agreement,
                                          ev=ev_under)
            reason = (
                f"Under: prob={prediction.prob_under:.2%}, "
                f"EV={ev_under:+.3f}, payout={payout:.0%}, "
                f"dur={duration}t, agree={prediction.model_agreement:.0%}, "
                f"stake=${stake:.2f}"
            )
            signal = Signal(
                direction=CONTRACT_TYPE_UNDER,
                barrier=UNDER_BARRIER,
                confidence=prediction.prob_under,
                expected_value=round(ev_under, 4),
                kelly_fraction=kelly,
                stake=stake,
                contract_duration=duration,
                timestamp=time.time(),
                features_snapshot=features.copy(),
                reason=reason,
                model_agreement=prediction.model_agreement,
            )
        else:
            self._skip("insufficient_edge")
        
        if signal:
            self._signals_generated += 1
            self._last_signal_time = time.time()
            logger.info(f"SIGNAL: {signal.reason}")
        else:
            self._signals_skipped += 1
        
        return signal
    
    def _kelly_fraction(self, win_prob: float, payout: float) -> float:
        """
        Kelly Criterion: f* = (b*p - q) / b
        where b = payout ratio, p = win prob, q = 1-p
        
        Returns fractional Kelly (divided by KELLY_FRACTION for safety).
        
        v2: Upper clamp is now MAX_BANKROLL_PER_TRADE so the risk manager
        doesn't have to reject every signal. Kelly is a SUGGESTION — the
        hard cap is the risk manager's 2% rule.
        """
        from config import MAX_BANKROLL_PER_TRADE
        
        b = payout
        p = win_prob
        q = 1 - p
        
        if b <= 0:
            return 0.0
        
        kelly = (b * p - q) / b
        
        # Apply fractional Kelly for safety
        fractional = kelly * KELLY_FRACTION
        
        # Clamp: floor at 1%, ceiling at max bankroll per trade (2%)
        # This ensures Kelly never suggests more than the risk manager allows
        return max(0.01, min(fractional, MAX_BANKROLL_PER_TRADE))
    
    def _calculate_stake(self, kelly_fraction: float, bankroll: float,
                          confidence: float = 0.5, agreement: float = 0.67,
                          ev: float = 0.0) -> float:
        """
        Convert Kelly fraction to actual stake amount with DYNAMIC SIZING.
        
        The more the models agree and the higher the confidence, the bigger
        the stake. When only 2/3 models agree or confidence is low, use the
        minimum stake.
        
        Scaling factors:
          confidence_boost: 1.0x at MIN_CONFIDENCE → up to 3.0x at 80%+
          agreement_boost:  1.0x at 67% (2/3 agree) → 2.0x at 100% (3/3 agree)
          ev_boost:         1.0x at MIN_EDGE → up to 2.0x at high EV
        
        Combined boost is capped at 8x so even a perfect signal can't
        exceed MAX_STAKE or the 2% bankroll rule.
        """
        from config import MIN_STAKE, MAX_STAKE, MAX_BANKROLL_PER_TRADE, MIN_CONFIDENCE, MIN_EDGE_THRESHOLD
        
        # Base stake from Kelly
        stake = kelly_fraction * bankroll
        
        # ─── Dynamic boost factors ───
        
        # Confidence boost: scale from 1x (at MIN_CONFIDENCE) to 3x (at 80%+)
        # 56% conf → 1.0x, 65% → 1.6x, 70% → 2.1x, 75% → 2.5x, 80%+ → 3.0x
        if confidence > MIN_CONFIDENCE:
            conf_range = min(1.0, (confidence - MIN_CONFIDENCE) / (0.80 - MIN_CONFIDENCE))
            confidence_boost = 1.0 + 2.0 * conf_range
        else:
            confidence_boost = 1.0
        
        # Agreement boost: 67% (2/3 agree) → 1.0x, 100% (3/3 agree) → 2.0x
        # This is the KEY factor — all 3 models agreeing is a strong signal
        if agreement >= 1.0:
            agreement_boost = 2.0   # All 3 agree → double stake
        elif agreement >= 0.67:
            # Scale from 1.0x at 67% to 2.0x at 100%
            agreement_boost = 1.0 + (agreement - 0.67) / (1.0 - 0.67) * 1.0
        else:
            agreement_boost = 0.5   # Less than 2/3 agree → reduce stake
        
        # EV boost: higher expected value → slightly bigger stake
        # 0.01 EV → 1.0x, 0.10 EV → 1.5x, 0.20+ EV → 2.0x
        if ev > MIN_EDGE_THRESHOLD:
            ev_range = min(1.0, (ev - MIN_EDGE_THRESHOLD) / 0.20)
            ev_boost = 1.0 + 1.0 * ev_range
        else:
            ev_boost = 1.0
        
        # Combined boost (capped at 8x total)
        total_boost = min(8.0, confidence_boost * agreement_boost * ev_boost)
        
        # Apply boost to stake
        stake = stake * total_boost
        
        # ─── Hard limits (these are ALWAYS enforced) ───
        # Hard cap: never exceed MAX_BANKROLL_PER_TRADE of bankroll
        max_allowed = bankroll * MAX_BANKROLL_PER_TRADE
        stake = min(stake, max_allowed)
        # Also cap at absolute MAX_STAKE
        stake = min(stake, MAX_STAKE)
        # Floor at MIN_STAKE
        stake = max(stake, MIN_STAKE)
        return round(stake, 2)
    
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
