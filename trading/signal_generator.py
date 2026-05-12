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
                    DYNAMIC_DURATION, FORCE_TRADE_MIN_CONFIDENCE,
                    FORCE_TRADE_MIN_EV, MIN_SIGNAL_SCORE,
                    MARTINGALE_MIN_CONFIDENCE, MARTINGALE_MIN_AGREEMENT,
                    TREND_SLOPE_TSTAT_THRESHOLD, TREND_CONFIDENCE_REDUCTION,
                    TREND_SIGNAL_SCORE_REDUCTION)
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
    is_martingale: bool = False  # True if this is a martingale recovery trade


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
    
    v3 additions:
    - Trend bias: lower confidence threshold for trend-aligned trades
    - Uptrend → easier to trade Over; Downtrend → easier to trade Under
    - Ranging markets trade normally (no penalty, no restriction)
    - This is a BIAS, not a gate — counter-trend trades still happen normally
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
                 model_in_drift: bool = False,
                 is_martingale: bool = False) -> Optional[Signal]:
        """
        Generate a trade signal from model prediction + current payout.
        
        CONFIDENCE-WEIGHTED AGREEMENT:
        Signals are scored by confidence * agreement. A trade must meet
        MIN_SIGNAL_SCORE to be taken. This prevents:
        - 100% agreement at 61% confidence (weak — models barely agree)
        - 67% agreement at 58% confidence (weak — 2/3 models barely leaning)
        Only trade when models are BOTH confident AND in agreement.
        
        MARTINGALE CONFIDENCE GATE:
        During martingale recovery, the bar is MUCH higher:
        - Must have 80%+ confidence AND 100% agreement
        - No more doubling down on weak 68% confidence signals
        
        Args:
            prediction: Model's output (probabilities, confidence)
            features: Feature dict at prediction time (stored for later learning)
            payout: Current payout ratio for the contract (e.g., 0.85 = 85%)
            bankroll: Current account balance
            model_in_drift: Whether the model is in drift state
            is_martingale: Whether this is a martingale recovery trade
        
        Returns:
            Signal if conditions met, None otherwise.
        """
        # ─── MARTINGALE CONFIDENCE GATE ───
        # During martingale recovery, the model must be MUCH more confident.
        # No more 68% confidence martingale trades — that's reckless.
        # Requires BOTH: 80%+ confidence AND 100% model agreement.
        if is_martingale:
            best_conf = max(prediction.prob_over, prediction.prob_under)
            if best_conf < MARTINGALE_MIN_CONFIDENCE:
                self._skip("martingale_low_confidence")
                logger.info(
                    f"MARTINGALE GATE: Rejected — confidence {best_conf:.0%} "
                    f"< {MARTINGALE_MIN_CONFIDENCE:.0%} minimum. "
                    f"Waiting for stronger signal before doubling down."
                )
                return None
            if prediction.model_agreement < MARTINGALE_MIN_AGREEMENT:
                self._skip("martingale_low_agreement")
                logger.info(
                    f"MARTINGALE GATE: Rejected — agreement {prediction.model_agreement:.0%} "
                    f"< {MARTINGALE_MIN_AGREEMENT:.0%} minimum. "
                    f"All models must agree to double down."
                )
                return None
        
        # ─── FORCE TRADE: All 3 models agree 100% WITH REAL confidence ───
        # PROBLEM: With 3 binary votes, models at 54%/46% ALL vote the same way
        # → fake 100% "agreement" on every tick.
        # FIX: Require (a) ensemble confidence >= 60% AND (b) positive EV.
        # This ensures forced trades only happen when models are MEANINGFULLY aligned.
        all_models_agree = prediction.model_agreement >= 1.0
        
        if all_models_agree:
            # Calculate EV for both directions
            ev_over = prediction.prob_over * payout - prediction.prob_under * 1.0
            ev_under = prediction.prob_under * payout - prediction.prob_over * 1.0
            
            # Pick best direction
            if prediction.prob_over >= prediction.prob_under:
                best_confidence = prediction.prob_over
                best_ev = ev_over
            else:
                best_confidence = prediction.prob_under
                best_ev = ev_under
            
            # ─── GUARD: Fake agreement check ───
            # If confidence is low (e.g., 54%), 100% vote agreement is MEANINGLESS.
            # Models are barely leaning — don't force a trade.
            if best_confidence < FORCE_TRADE_MIN_CONFIDENCE:
                self._skip("forced_low_confidence")
                return None
            
            # ─── GUARD: Negative EV ───
            # Even with 100% agreement, don't trade if EV is negative.
            # This happens when confidence is just above threshold but payout is low.
            if best_ev < FORCE_TRADE_MIN_EV:
                self._skip("forced_negative_ev")
                return None
            
            # Even with 100% agreement, respect minimum cooldown
            time_since_last = time.time() - self._last_signal_time
            if time_since_last < MIN_TRADE_INTERVAL_SEC:
                self._skip("cooldown")
                return None
            
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
                                          ev=ev)
            
            # Check trend for logging context (not used for gating — forced trades always go through)
            tr = features.get("trend_regime", 0)
            trend_ctx = f", TREND={'UPTREND' if tr == 1 else 'DOWNTREND' if tr == -1 else 'ranging'}" if tr != 0 else ""
            reason = (
                f"FORCED (100% AGREEMENT): {direction.replace('DIGIT', '').title()} "
                f"prob={confidence:.2%}, EV={ev:+.3f}, payout={payout:.0%}, "
                f"dur={duration}t, agree=100%, stake=${stake:.2f}{trend_ctx}"
            )
            
            signal = Signal(
                direction=direction,
                barrier=barrier,
                confidence=confidence,
                expected_value=round(ev, 4),
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
        
        # ─── TREND BIAS ───
        # Linear regression slope detects market trend direction.
        # When a trend is detected, lower confidence threshold for
        # trend-aligned trades. This is a BIAS, not a restriction:
        # - Uptrend → easier to trade Over (lower confidence needed)
        # - Downtrend → easier to trade Under (lower confidence needed)
        # - Ranging → no change (trade normally)
        # Counter-trend trades are NOT blocked — they just use normal thresholds.
        trend_regime = features.get("trend_regime", 0)   # 1=up, -1=down, 0=ranging
        tstat_200 = features.get("slope_tstat_200", 0.0)
        tstat_50 = features.get("slope_tstat_50", 0.0)
        
        # Is the prediction trend-aligned?
        # Uptrend + predicting Over = aligned. Downtrend + predicting Under = aligned.
        is_trend_aligned = False
        trend_label = "ranging"
        if trend_regime == 1 and prediction.prob_over > prediction.prob_under:
            is_trend_aligned = True
            trend_label = "UPTREND"
        elif trend_regime == -1 and prediction.prob_under > prediction.prob_over:
            is_trend_aligned = True
            trend_label = "DOWNTREND"
        elif trend_regime == 1:
            trend_label = "UPTREND"
        elif trend_regime == -1:
            trend_label = "DOWNTREND"
        
        # Lower confidence threshold for trend-aligned predictions
        effective_min_confidence = MIN_CONFIDENCE
        effective_signal_score = MIN_SIGNAL_SCORE
        if is_trend_aligned:
            effective_min_confidence = max(0.50, MIN_CONFIDENCE - TREND_CONFIDENCE_REDUCTION)
            effective_signal_score = max(0.50, MIN_SIGNAL_SCORE - TREND_SIGNAL_SCORE_REDUCTION)
        
        # Check minimum confidence (with trend bias applied)
        if prediction.confidence < effective_min_confidence:
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
        
        # ─── Confidence-Weighted Agreement Score ───
        # Agreement alone isn't enough. 100% agreement at 61% confidence is WEAK.
        # The signal score = confidence * agreement.
        # Both must be high — confident models that agree → strong signal.
        # 100% agree at 72% confidence → score = 0.72 → TRADE
        # 100% agree at 61% confidence → score = 0.61 → SKIP (below 0.65)
        # 67% agree at 70% confidence → score = 0.47 → SKIP
        #
        # Trend bias lowers the signal_score threshold for trend-aligned trades:
        # Uptrend + Over at 67% agree × 90% conf = 0.60 → passes 0.60 threshold
        signal_score = prediction.confidence * prediction.model_agreement
        if signal_score < effective_signal_score:
            self._skip("low_signal_score")
            return None
        
        # ─── Calculate Expected Value ───
        # EV = P(win) * payout - P(lose) * 1
        # (expressed per dollar risked)
        
        ev_over = prediction.prob_over * payout - prediction.prob_under * 1.0
        ev_under = prediction.prob_under * payout - prediction.prob_over * 1.0
        
        # ─── Direction-specific confidence thresholds ───
        # In an uptrend, Over trades need less confidence. In a downtrend,
        # Under trades need less confidence. Counter-trend uses normal threshold.
        over_min_conf = MIN_CONFIDENCE
        under_min_conf = MIN_CONFIDENCE
        if trend_regime == 1:   # Uptrend
            over_min_conf = max(0.50, MIN_CONFIDENCE - TREND_CONFIDENCE_REDUCTION)
        elif trend_regime == -1:  # Downtrend
            under_min_conf = max(0.50, MIN_CONFIDENCE - TREND_CONFIDENCE_REDUCTION)
        
        # ─── Select Contract Duration ───
        if self._duration_optimizer and DYNAMIC_DURATION:
            duration = self._duration_optimizer.select_duration()
        else:
            duration = CONTRACT_DURATION
        
        signal = None
        
        if ev_over > MIN_EDGE_THRESHOLD and prediction.prob_over >= over_min_conf:
            kelly = self._kelly_fraction(prediction.prob_over, payout)
            stake = self._calculate_stake(kelly, bankroll,
                                          confidence=prediction.prob_over,
                                          agreement=prediction.model_agreement,
                                          ev=ev_over)
            trend_info = f", TREND={trend_label}" if trend_regime != 0 else ""
            reason = (
                f"Over: prob={prediction.prob_over:.2%}, "
                f"EV={ev_over:+.3f}, payout={payout:.0%}, "
                f"dur={duration}t, agree={prediction.model_agreement:.0%}, "
                f"stake=${stake:.2f}{trend_info}"
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
        
        elif ev_under > MIN_EDGE_THRESHOLD and prediction.prob_under >= under_min_conf:
            kelly = self._kelly_fraction(prediction.prob_under, payout)
            stake = self._calculate_stake(kelly, bankroll,
                                          confidence=prediction.prob_under,
                                          agreement=prediction.model_agreement,
                                          ev=ev_under)
            trend_info = f", TREND={trend_label}" if trend_regime != 0 else ""
            reason = (
                f"Under: prob={prediction.prob_under:.2%}, "
                f"EV={ev_under:+.3f}, payout={payout:.0%}, "
                f"dur={duration}t, agree={prediction.model_agreement:.0%}, "
                f"stake=${stake:.2f}{trend_info}"
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
