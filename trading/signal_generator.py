"""
Signal Generator (v3 — Trend Requirement + Dynamic Duration)
==============================================================
Converts model probabilities into trade decisions.
Dynamic duration via DurationOptimizer. ONLY trades in strong trends.

This is where most bots fail — they trade too often.
The key: ONLY trade when your edge is CLEAR AND the trend is strong.
"""

import time
from dataclasses import dataclass
from typing import Optional

from config import (MIN_CONFIDENCE, MIN_EDGE_THRESHOLD, OVER_BARRIER,
                    UNDER_BARRIER, CONTRACT_TYPE_OVER, CONTRACT_TYPE_UNDER,
                    CONTRACT_DURATION, DYNAMIC_DURATION, KELLY_FRACTION,
                    COOLDOWN_AFTER_LOSS_TICKS,
                    MIN_TRADE_INTERVAL_SEC, MAX_DAILY_TRADES,
                    FORCE_TRADE_MIN_CONFIDENCE,
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
    4. TREND GATE: Only trade if a strong trend is confirmed
       - Uptrend → ONLY Over trades allowed
       - Downtrend → ONLY Under trades allowed
       - Ranging → NO TRADE (skip entirely)
    5. If EV > threshold AND confidence > threshold → generate signal
    6. Otherwise → no trade (this is the most common outcome)
    
    v3 changes from v2:
    - FIXED 1-tick duration (DurationOptimizer removed — same payout, shorter = better)
    - Trend is now a REQUIREMENT, not a bias:
      - No trend = no trade at all
      - Uptrend = only Over, Downtrend = only Under
    - Three-window trend confirmation (50, 200, 500 ticks)
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
                 is_martingale: bool = False,
                 martingale_direction: str = None) -> Optional[Signal]:
        """
        Generate a trade signal from model prediction + current payout.
        
        TREND REQUIREMENT:
        Trades ONLY happen in strong trends. No trend = no trade.
        - Uptrend (trend_regime=1) → ONLY Over trades
        - Downtrend (trend_regime=-1) → ONLY Under trades
        - Ranging (trend_regime=0) → NO TRADE, skip entirely
        
        CONFIDENCE-WEIGHTED AGREEMENT:
        Signals are scored by confidence * agreement. A trade must meet
        MIN_SIGNAL_SCORE to be taken.
        
        MARTINGALE CONFIDENCE GATE:
        During martingale recovery, the bar is MUCH higher:
        - Must have 80%+ confidence AND 100% agreement
        - Must maintain SAME direction as the original losing trade
        - No more doubling down on weak signals or switching directions
        
        Args:
            prediction: Model's output (probabilities, confidence)
            features: Feature dict at prediction time (stored for later learning)
            payout: Current payout ratio for the contract (e.g., 0.85 = 85%)
            bankroll: Current account balance
            model_in_drift: Whether the model is in drift state
            is_martingale: Whether this is a martingale recovery trade
            martingale_direction: Required direction for martingale recovery
        
        Returns:
            Signal if conditions met, None otherwise.
        """
        # ─── TREND REQUIREMENT (FIRST CHECK — GATE EVERYTHING) ───
        # This is the most important filter. No trend = no trade.
        # Uptrend → only Over. Downtrend → only Under. Ranging → skip.
        trend_regime = features.get("trend_regime", 0)   # 1=up, -1=down, 0=ranging
        tstat_50 = features.get("slope_tstat_50", 0.0)
        tstat_200 = features.get("slope_tstat_200", 0.0)
        tstat_500 = features.get("slope_tstat_500", 0.0)
        
        if trend_regime == 0:
            self._skip("no_trend_ranging")
            # Log occasionally (not every tick — too noisy)
            if self._signals_skipped % 50 == 0:
                logger.info(
                    f"TREND GATE: No strong trend (t50={tstat_50:.1f}, "
                    f"t200={tstat_200:.1f}, t500={tstat_500:.1f}) — "
                    f"skipping trade. Need all 3 windows to agree at t>{TREND_SLOPE_TSTAT_THRESHOLD}"
                )
            return None
        
        # Determine which direction the trend allows
        if trend_regime == 1:
            allowed_direction = CONTRACT_TYPE_OVER
            trend_label = "UPTREND"
        else:  # trend_regime == -1
            allowed_direction = CONTRACT_TYPE_UNDER
            trend_label = "DOWNTREND"
        
        # ─── MARTINGALE CONFIDENCE GATE ───
        # During martingale recovery, the model must be MUCH more confident.
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
            
            # ─── MARTINGALE DIRECTION PERSISTENCE ───
            # During martingale, the direction MUST match the original losing trade.
            if martingale_direction:
                if martingale_direction != allowed_direction:
                    # Martingale wants a direction that the trend doesn't support
                    self._skip("martingale_vs_trend")
                    logger.info(
                        f"MARTINGALE + TREND CONFLICT: Martingale requires "
                        f"{martingale_direction} but trend says {allowed_direction} "
                        f"({trend_label}). Skipping — won't fight the trend."
                    )
                    return None
                if prediction.prob_over >= prediction.prob_under:
                    model_direction = CONTRACT_TYPE_OVER
                else:
                    model_direction = CONTRACT_TYPE_UNDER
                
                if model_direction != martingale_direction:
                    self._skip("martingale_direction_mismatch")
                    logger.info(
                        f"MARTINGALE DIRECTION GATE: Rejected — model wants {model_direction} "
                        f"but martingale requires {martingale_direction}. "
                        f"Must recover in the SAME direction as the original loss."
                    )
                    return None
        
        # ─── FORCE TRADE: All 3 models agree 100% WITH REAL confidence ───
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
            if best_confidence < FORCE_TRADE_MIN_CONFIDENCE:
                self._skip("forced_low_confidence")
                return None
            
            # ─── GUARD: Negative EV ───
            if best_ev < FORCE_TRADE_MIN_EV:
                self._skip("forced_negative_ev")
                return None
            
            # ─── TREND GATE: Even forced trades must respect the trend ───
            # Uptrend → only Over, Downtrend → only Under
            if trend_regime == 1 and prediction.prob_under > prediction.prob_over:
                # Trend says Over but model wants Under — skip
                self._skip("forced_counter_trend")
                return None
            elif trend_regime == -1 and prediction.prob_over > prediction.prob_under:
                # Trend says Under but model wants Over — skip
                self._skip("forced_counter_trend")
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
            
            # Determine direction — MUST follow the trend
            if trend_regime == 1:
                direction = CONTRACT_TYPE_OVER
                confidence = prediction.prob_over
                ev = ev_over
                barrier = OVER_BARRIER
            else:  # trend_regime == -1
                direction = CONTRACT_TYPE_UNDER
                confidence = prediction.prob_under
                ev = ev_under
                barrier = UNDER_BARRIER
            
            kelly = self._kelly_fraction(confidence, payout)
            stake = self._calculate_stake(kelly, bankroll,
                                          confidence=confidence,
                                          agreement=prediction.model_agreement,
                                          ev=ev)
            
            reason = (
                f"FORCED (100% AGREEMENT + {trend_label}): {direction.replace('DIGIT', '').title()} "
                f"prob={confidence:.2%}, EV={ev:+.3f}, payout={payout:.0%}, "
                f"dur={duration}t, agree=100%, stake=${stake:.2f}, "
                f"t50={tstat_50:.1f} t200={tstat_200:.1f} t500={tstat_500:.1f}"
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
        
        # ─── TREND DIRECTION ENFORCEMENT ───
        # Uptrend → ONLY allow Over trades
        # Downtrend → ONLY allow Under trades
        # This is a REQUIREMENT — counter-trend trades are NOT allowed.
        
        # Lower confidence threshold for trend-aligned trades (small bonus)
        effective_min_confidence = MIN_CONFIDENCE
        effective_signal_score = MIN_SIGNAL_SCORE
        # Since we already confirmed trend_regime != 0, the trade IS trend-aligned
        effective_min_confidence = max(0.50, MIN_CONFIDENCE - TREND_CONFIDENCE_REDUCTION)
        effective_signal_score = max(0.50, MIN_SIGNAL_SCORE - TREND_SIGNAL_SCORE_REDUCTION)
        
        # Check minimum confidence (with trend bonus applied)
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
        signal_score = prediction.confidence * prediction.model_agreement
        if signal_score < effective_signal_score:
            self._skip("low_signal_score")
            return None
        
        # ─── Calculate Expected Value ───
        ev_over = prediction.prob_over * payout - prediction.prob_under * 1.0
        ev_under = prediction.prob_under * payout - prediction.prob_over * 1.0
        
        # Select contract duration
        if self._duration_optimizer and DYNAMIC_DURATION:
            duration = self._duration_optimizer.select_duration()
        else:
            duration = CONTRACT_DURATION
        
        signal = None
        
        # ─── TREND-ALIGNED DIRECTION ONLY ───
        # Uptrend → only evaluate Over. Downtrend → only evaluate Under.
        if trend_regime == 1:
            # UPTREND: Only Over trades allowed
            over_min_conf = max(0.50, MIN_CONFIDENCE - TREND_CONFIDENCE_REDUCTION)
            if ev_over > MIN_EDGE_THRESHOLD and prediction.prob_over >= over_min_conf:
                kelly = self._kelly_fraction(prediction.prob_over, payout)
                stake = self._calculate_stake(kelly, bankroll,
                                              confidence=prediction.prob_over,
                                              agreement=prediction.model_agreement,
                                              ev=ev_over)
                reason = (
                    f"Over: prob={prediction.prob_over:.2%}, "
                    f"EV={ev_over:+.3f}, payout={payout:.0%}, "
                    f"dur={duration}t, agree={prediction.model_agreement:.0%}, "
                    f"stake=${stake:.2f}, TREND={trend_label}, "
                    f"t50={tstat_50:.1f} t200={tstat_200:.1f} t500={tstat_500:.1f}"
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
            else:
                self._skip("uptrend_over_insufficient_edge")
        
        elif trend_regime == -1:
            # DOWNTREND: Only Under trades allowed
            under_min_conf = max(0.50, MIN_CONFIDENCE - TREND_CONFIDENCE_REDUCTION)
            if ev_under > MIN_EDGE_THRESHOLD and prediction.prob_under >= under_min_conf:
                kelly = self._kelly_fraction(prediction.prob_under, payout)
                stake = self._calculate_stake(kelly, bankroll,
                                              confidence=prediction.prob_under,
                                              agreement=prediction.model_agreement,
                                              ev=ev_under)
                reason = (
                    f"Under: prob={prediction.prob_under:.2%}, "
                    f"EV={ev_under:+.3f}, payout={payout:.0%}, "
                    f"dur={duration}t, agree={prediction.model_agreement:.0%}, "
                    f"stake=${stake:.2f}, TREND={trend_label}, "
                    f"t50={tstat_50:.1f} t200={tstat_200:.1f} t500={tstat_500:.1f}"
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
                self._skip("downtrend_under_insufficient_edge")
        
        if signal:
            self._signals_generated += 1
            self._last_signal_time = time.time()
            logger.info(f"SIGNAL: {signal.reason}")
        # Note: _signals_skipped is incremented inside _skip() calls
        
        return signal
    
    def _kelly_fraction(self, win_prob: float, payout: float) -> float:
        """
        Kelly Criterion: f* = (b*p - q) / b
        where b = payout ratio, p = win prob, q = 1-p
        
        Returns fractional Kelly (divided by KELLY_FRACTION for safety).
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
        
        # Clamp: floor at 1%, ceiling at max bankroll per trade
        return max(0.01, min(fractional, MAX_BANKROLL_PER_TRADE))
    
    def _calculate_stake(self, kelly_fraction: float, bankroll: float,
                          confidence: float = 0.5, agreement: float = 0.67,
                          ev: float = 0.0) -> float:
        """
        Convert Kelly fraction to actual stake amount with DYNAMIC SIZING.
        """
        from config import MIN_STAKE, MAX_STAKE, MAX_BANKROLL_PER_TRADE, MIN_CONFIDENCE, MIN_EDGE_THRESHOLD
        
        # Base stake from Kelly
        stake = kelly_fraction * bankroll
        
        # ─── Dynamic boost factors ───
        
        # Confidence boost: scale from 1x (at MIN_CONFIDENCE) to 3x (at 80%+)
        if confidence > MIN_CONFIDENCE:
            conf_range = min(1.0, (confidence - MIN_CONFIDENCE) / (0.80 - MIN_CONFIDENCE))
            confidence_boost = 1.0 + 2.0 * conf_range
        else:
            confidence_boost = 1.0
        
        # Agreement boost: 67% → 1.0x, 100% → 2.0x
        if agreement >= 1.0:
            agreement_boost = 2.0
        elif agreement >= 0.67:
            agreement_boost = 1.0 + (agreement - 0.67) / (1.0 - 0.67) * 1.0
        else:
            agreement_boost = 0.5
        
        # EV boost: higher expected value → slightly bigger stake
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
