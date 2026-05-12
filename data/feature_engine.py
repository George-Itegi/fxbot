"""
Feature Engine (v3 — Digit-Centric Features)
=============================================
Computes 50+ features for Over/Under prediction from tick data.
Features are returned as a flat dict suitable for River/scikit-learn.

v3 additions (digit-centric):
- Current digit as a direct feature
- Over/Under streak tracking
- Digit gap between consecutive digits
- Digit autocorrelation
- Over/Under reversal rate
- Digit moving average and velocity

v2 additions:
- Markov chain transition features (what digit follows what digit)
- Digit n-gram features (pairs, triples)
- Improved temporal features
"""

import math
import time
from typing import Optional

from data.tick_aggregator import TickAggregator
from config import TICK_WINDOWS, OVER_BARRIER, UNDER_BARRIER
from utils.logger import setup_logger

logger = setup_logger("data.feature_engine")


class FeatureEngine:
    """
    Computes all features needed for the Over/Under model.
    
    Feature groups:
    1. Digit Distribution (16 features)
    2. Volatility (6 features)
    3. Momentum (7 features)
    4. Temporal (6 features)
    5. Pattern (5 features)
    6. Derived/Interaction (5 features)
    7. Markov Chain Transitions (6 features)
    8. Digit N-grams (5 features)
    9. Digit-Centric (9 features)              ← NEW v3
    
    Total: ~65 features
    
    IMPORTANT: For Deriv Over/Under contracts, the LAST DIGIT is the
    ONLY thing that determines the outcome. Group 9 features are the
    most directly relevant because they focus purely on the last digit
    pattern, not the price level.
    """
    
    def __init__(self, aggregator: TickAggregator):
        self.agg = aggregator
        self.last_features: Optional[dict] = None
        
        # Markov chain transition matrix
        # Counts of (digit_t) → (digit_t+1) transitions
        self._transition_counts: dict[int, dict[int, int]] = {}
        for d in range(10):
            self._transition_counts[d] = {d2: 0 for d2 in range(10)}
        self._transition_total: dict[int, int] = {d: 0 for d in range(10)}
        
        # Last digit tracking for transition updates
        self._last_digit: Optional[int] = None
        
        # Digit pair tracking (bigrams)
        self._digit_pairs: dict[str, int] = {}  # "3_5" → count
        self._pair_total = 0
        
        # Over/Under streak tracking (for digit-centric features)
        self._over_streak: int = 0    # Consecutive Over (digit > 4) results
        self._under_streak: int = 0   # Consecutive Under (digit < 5) results
        self._last_was_over: Optional[bool] = None  # Was the last digit Over?
        
        # Digit gap tracking (for digit velocity)
        self._last_digit_value: Optional[int] = None  # For computing digit gap
    
    def compute_features(self) -> Optional[dict]:
        """
        Compute full feature vector from current tick buffers.
        Returns None if aggregator doesn't have enough data yet.
        """
        if not self.agg.is_warm("short"):
            return None
        
        # Update Markov transition matrix with latest tick
        self._update_markov()
        
        features = {}
        
        # ─── 1. Digit Distribution Features ───
        features.update(self._digit_features())
        
        # ─── 2. Volatility Features ───
        features.update(self._volatility_features())
        
        # ─── 3. Momentum Features ───
        features.update(self._momentum_features())
        
        # ─── 4. Temporal Features ───
        features.update(self._temporal_features())
        
        # ─── 5. Pattern Features ───
        features.update(self._pattern_features())
        
        # ─── 6. Derived/Interaction Features ───
        features.update(self._derived_features(features))
        
        # ─── 7. Markov Chain Transition Features ───
        features.update(self._markov_features())
        
        # ─── 8. Digit N-gram Features ───
        features.update(self._ngram_features())
        
        # ─── 9. Digit-Centric Features ───
        features.update(self._digit_centric_features())
        
        # ─── 10. Trend Slope Features ───
        features.update(self._trend_features())
        
        self.last_features = features
        return features
    
    def _update_markov(self):
        """Update Markov transition counts with the latest digit from aggregator."""
        current_digit = self.agg.last_tick.digit if self.agg.last_tick else None
        if current_digit is not None:
            self.update_markov(current_digit)

    def update_markov(self, digit: int):
        """
        Public method to update Markov transition counts with a specific digit.
        Called on every tick (even throttled ones) so no transitions are missed.
        """
        if self._last_digit is not None:
            # Record transition: last_digit → current_digit
            self._transition_counts[self._last_digit][digit] += 1
            self._transition_total[self._last_digit] += 1
            
            # Record pair (bigram)
            pair_key = f"{self._last_digit}_{digit}"
            self._digit_pairs[pair_key] = self._digit_pairs.get(pair_key, 0) + 1
            self._pair_total += 1
        
        self._last_digit = digit

    def update_streaks(self, digit: int):
        """
        Update streak tracking with a new digit.
        Tracks Over/Under streaks (consecutive digit > 4 or < 5)
        separately from same-digit streaks.
        """
        is_over = digit > OVER_BARRIER
        
        if is_over:
            if self._last_was_over:
                self._over_streak += 1
                self._under_streak = 0
            else:
                self._over_streak = 1
                self._under_streak = 0
        else:
            if not self._last_was_over:
                self._under_streak += 1
                self._over_streak = 0
            else:
                self._under_streak = 1
                self._over_streak = 0
        
        self._last_was_over = is_over
    
    def _digit_features(self) -> dict:
        """Digit distribution and barrier hit rates."""
        features = {}
        
        # Digit frequency for each digit (0-9) in short window
        dist_short = self.agg.digit_distribution("short")
        for d in range(10):
            features[f"digit_freq_{d}_s"] = dist_short.get(d, 0.1)
        
        # Barrier-specific hit rates across multiple windows
        features["barrier_hit_over_short"] = self.agg.barrier_hit_rate(
            OVER_BARRIER, "short"
        )
        features["barrier_hit_over_medium"] = self.agg.barrier_hit_rate(
            OVER_BARRIER, "medium"
        )
        features["barrier_hit_over_long"] = self.agg.barrier_hit_rate(
            OVER_BARRIER, "long"
        )
        features["barrier_hit_under_short"] = self.agg.barrier_hit_rate(
            UNDER_BARRIER, "short"
        )
        features["barrier_hit_under_medium"] = self.agg.barrier_hit_rate(
            UNDER_BARRIER, "medium"
        )
        
        # Change in barrier hit rate (momentum of hit rate)
        hit_short = features["barrier_hit_over_short"]
        hit_medium = features["barrier_hit_over_medium"]
        features["barrier_rate_momentum"] = hit_short - hit_medium
        
        return features
    
    def _volatility_features(self) -> dict:
        """Price volatility metrics."""
        features = {}
        
        # Standard deviation across windows
        features["price_std_short"] = self.agg.price_std("short")
        features["price_std_medium"] = self.agg.price_std("medium")
        
        # Price range
        features["price_range_short"] = self.agg.price_range("short")
        features["price_range_medium"] = self.agg.price_range("medium")
        
        # Volatility ratio (short/long) — regime indicator
        std_long = self.agg.price_std("long")
        if std_long > 0:
            features["volatility_ratio"] = features["price_std_short"] / std_long
        else:
            features["volatility_ratio"] = 1.0
        
        # Coefficient of variation
        ticks = self.agg.get_window("short")
        if ticks:
            mean_price = sum(t.quote for t in ticks) / len(ticks)
            if mean_price > 0:
                features["cv_short"] = features["price_std_short"] / mean_price
            else:
                features["cv_short"] = 0.0
        else:
            features["cv_short"] = 0.0
        
        return features
    
    def _momentum_features(self) -> dict:
        """Price direction and momentum metrics."""
        features = {}
        
        ticks_micro = self.agg.get_window("micro")
        ticks_short = self.agg.get_window("short")
        ticks_medium = self.agg.get_window("medium")
        
        # Price changes
        if len(ticks_micro) >= 2:
            features["price_change_micro"] = (
                ticks_micro[-1].quote - ticks_micro[0].quote
            )
        else:
            features["price_change_micro"] = 0.0
        
        if len(ticks_short) >= 2:
            features["price_change_short"] = (
                ticks_short[-1].quote - ticks_short[0].quote
            )
        else:
            features["price_change_short"] = 0.0
        
        if len(ticks_medium) >= 2:
            features["price_change_medium"] = (
                ticks_medium[-1].quote - ticks_medium[0].quote
            )
        else:
            features["price_change_medium"] = 0.0
        
        # Direction bias
        features["direction_bias_short"] = self.agg.direction_bias("short")
        features["direction_bias_medium"] = self.agg.direction_bias("medium")
        
        # Consecutive same-direction run
        if ticks_short:
            last_dir = ticks_short[-1].direction
            consec = 0
            for t in reversed(ticks_short):
                if t.direction == last_dir and last_dir != 0:
                    consec += 1
                else:
                    break
            features["consecutive_direction"] = consec
        else:
            features["consecutive_direction"] = 0
        
        # Mean reversion score (how far from rolling mean)
        if ticks_short:
            mean_q = sum(t.quote for t in ticks_short) / len(ticks_short)
            last_q = ticks_short[-1].quote
            features["mean_reversion"] = (last_q - mean_q) / features["price_std_short"] if features.get("price_std_short", 0) > 0 else 0.0
        else:
            features["mean_reversion"] = 0.0
        
        return features
    
    def _temporal_features(self) -> dict:
        """Timing and frequency features."""
        features = {}
        
        ticks = self.agg.get_window("short")
        
        # Time since last tick
        if self.agg.last_tick:
            features["seconds_since_last_tick"] = max(
                0, time.time() - self.agg.last_tick.epoch
            ) if hasattr(self, '_time_ref') else 0.0
        else:
            features["seconds_since_last_tick"] = 0.0
        
        # Tick rate (ticks per second) across windows
        features["tick_rate_micro"] = self.agg.tick_rate("micro")
        features["tick_rate_short"] = self.agg.tick_rate("short")
        features["tick_rate_medium"] = self.agg.tick_rate("medium")
        
        # Tick rate ratio (short/medium) — acceleration indicator
        rate_medium = features["tick_rate_medium"]
        if rate_medium > 0:
            features["tick_rate_ratio"] = features["tick_rate_short"] / rate_medium
        else:
            features["tick_rate_ratio"] = 1.0
        
        # Inter-tick time variability (std of gaps)
        if len(ticks) >= 3:
            gaps = [ticks[i].epoch - ticks[i-1].epoch 
                    for i in range(1, len(ticks)) if ticks[i].epoch - ticks[i-1].epoch > 0]
            if gaps:
                mean_gap = sum(gaps) / len(gaps)
                var_gap = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
                features["tick_gap_std"] = var_gap ** 0.5
            else:
                features["tick_gap_std"] = 0.0
        else:
            features["tick_gap_std"] = 0.0
        
        return features
    
    def _pattern_features(self) -> dict:
        """Digit pattern features."""
        features = {}
        
        # Run length of same digit
        features["digit_run_length"] = self.agg.consecutive_same_digit()
        
        # Alternation ratio (how often consecutive digits differ)
        ticks = self.agg.get_window("short")
        if len(ticks) >= 2:
            alternations = sum(
                1 for i in range(1, len(ticks))
                if ticks[i].digit != ticks[i-1].digit
            )
            features["alternation_ratio"] = alternations / (len(ticks) - 1)
        else:
            features["alternation_ratio"] = 0.5
        
        # Shannon entropy of digit distribution
        features["entropy_short"] = self.agg.entropy("short")
        features["entropy_medium"] = self.agg.entropy("medium")
        
        # Entropy change (is predictability increasing or decreasing?)
        features["entropy_change"] = features["entropy_short"] - features["entropy_medium"]
        
        return features
    
    def _derived_features(self, base: dict) -> dict:
        """Interaction and derived features."""
        features = {}
        
        # Volatility x Entropy interaction
        vol = base.get("volatility_ratio", 1.0)
        ent = base.get("entropy_short", 3.0)
        features["vol_x_entropy"] = vol * ent
        
        # Barrier hit rate x volatility interaction
        hit = base.get("barrier_hit_over_short", 0.5)
        features["hit_x_vol"] = hit * vol
        
        # Regime label (0=low vol, 1=medium, 2=high vol)
        if vol < 0.8:
            features["regime"] = 0
        elif vol < 1.3:
            features["regime"] = 1
        else:
            features["regime"] = 2
        
        # Confidence feature: inverse entropy (lower entropy = more predictable)
        features["predictability"] = 1.0 / (ent + 0.01)
        
        # Digit bias toward upper half (5-9)
        dist = self.agg.digit_distribution("short")
        upper = sum(dist.get(d, 0) for d in range(5, 10))
        features["upper_digit_bias"] = upper
        
        return features
    
    def _markov_features(self) -> dict:
        """
        Markov chain transition features.
        
        These capture the PREDICTIVE POWER of digit transitions.
        If digit 3 is followed by digit 7 at 15% rate (vs 10% uniform),
        the model can learn this pattern.
        
        Key insight: The current digit gives us a conditional probability
        for the NEXT digit. This is MORE informative than the unconditional
        digit distribution alone.
        """
        features = {}
        
        current_digit = self.agg.last_tick.digit if self.agg.last_tick else 0
        
        # Feature 1: P(over | current_digit) — probability of over GIVEN the current digit
        # This is the KEY Markov feature for Over/Under prediction
        if self._transition_total.get(current_digit, 0) >= 5:
            over_count = sum(
                self._transition_counts[current_digit].get(d, 0)
                for d in range(OVER_BARRIER + 1, 10)
            )
            total = self._transition_total[current_digit]
            features["markov_p_over_given_current"] = over_count / total if total > 0 else 0.5
        else:
            features["markov_p_over_given_current"] = 0.5  # Prior
        
        # Feature 2: P(under | current_digit)
        if self._transition_total.get(current_digit, 0) >= 5:
            under_count = sum(
                self._transition_counts[current_digit].get(d, 0)
                for d in range(0, UNDER_BARRIER)
            )
            total = self._transition_total[current_digit]
            features["markov_p_under_given_current"] = under_count / total if total > 0 else 0.5
        else:
            features["markov_p_under_given_current"] = 0.5
        
        # Feature 3: Markov departure from uniform — how much does the transition
        # probability deviate from 0.5? Higher = more predictive structure
        p_over = features["markov_p_over_given_current"]
        features["markov_edge"] = abs(p_over - 0.5)  # 0 = no edge, 0.5 = max edge
        
        # Feature 4: Most likely next digit (argmax of transition row)
        if self._transition_total.get(current_digit, 0) >= 5:
            row = self._transition_counts[current_digit]
            most_likely = max(row, key=row.get)
            features["markov_most_likely_next"] = most_likely
            features["markov_most_likely_prob"] = row[most_likely] / self._transition_total[current_digit]
        else:
            features["markov_most_likely_next"] = 5  # Uniform prior
            features["markov_most_likely_prob"] = 0.1
        
        return features
    
    def _ngram_features(self) -> dict:
        """
        Digit n-gram (bigram) features.
        
        Captures patterns in digit PAIRS, e.g., "3 followed by 7"
        or "9 followed by 1". These are essentially 2nd-order Markov features.
        """
        features = {}
        
        # Get the last 2 digits
        digits = list(self.agg.digit_buffer)
        
        if len(digits) >= 2:
            current_pair = f"{digits[-2]}_{digits[-1]}"
            
            # Feature 1: How often has this pair occurred?
            pair_count = self._digit_pairs.get(current_pair, 0)
            pair_freq = pair_count / max(self._pair_total, 1)
            features["bigram_freq"] = pair_freq
            
            # Feature 2: Is this pair above or below uniform expectation?
            # Uniform: each pair = 1/100 = 0.01. Above 0.01 = overrepresented.
            features["bigram_surprise"] = pair_freq / 0.01  # >1 = more common than expected
            
            # Feature 3: What digit typically follows this pair?
            # This is a 2nd-order Markov feature (conditioned on 2 previous digits)
            # We'd need a 3-gram table for full implementation, but we can
            # approximate by looking at the transition from the last digit
            # and weighting by pair frequency
            
        else:
            features["bigram_freq"] = 0.01
            features["bigram_surprise"] = 1.0
        
        # Feature 4: Last 3 digits as a pattern index
        # Encodes the last 3 digits as a single number (0-999)
        # This gives the tree model an easy way to split on specific patterns
        if len(digits) >= 3:
            features["trigram_index"] = digits[-3] * 100 + digits[-2] * 10 + digits[-1]
        else:
            features["trigram_index"] = 0
        
        # Feature 5: Pair reversal rate — how often does the same pair
        # appear in reversed order? (e.g., 3_7 vs 7_3)
        if len(digits) >= 2:
            reversed_pair = f"{digits[-1]}_{digits[-2]}"
            fwd = self._digit_pairs.get(current_pair if len(digits) >= 2 else "", 0)
            rev = self._digit_pairs.get(reversed_pair, 0)
            total = fwd + rev
            features["pair_asymmetry"] = (fwd - rev) / max(total, 1) if total > 0 else 0.0
        else:
            features["pair_asymmetry"] = 0.0
        
        return features
    
    def _digit_centric_features(self) -> dict:
        """
        Digit-Centric Features (v3 — THE MOST IMPORTANT GROUP)
        ========================================================
        These features focus PURELY on the last digit pattern.
        
        For Deriv Over/Under contracts, the ONLY thing that determines
        the outcome is whether the last digit > 4 (Over) or < 5 (Under).
        The actual price level doesn't matter AT ALL.
        
        These features capture patterns that are DIRECTLY about the digit
        that will resolve the contract, not the price that contains it.
        """
        features = {}
        
        digits = list(self.agg.digit_buffer)
        current_digit = self.agg.last_tick.digit if self.agg.last_tick else 5
        
        # ─── Feature 1: Current Digit (0-9) ───
        # The RAW value of the last digit. This was MISSING before —
        # the model had to infer the digit from frequency distributions,
        # but now it gets it directly. Tree models can split on this
        # to learn "digit 9 → Over, digit 1 → Under" directly.
        features["current_digit"] = current_digit
        
        # ─── Feature 2: Over/Under Streak ───
        # How many consecutive Over or Under results have we seen?
        # This is DIFFERENT from digit_run_length (same digit repeating).
        # Over streak = consecutive digits > 4
        # Under streak = consecutive digits < 5
        # A long Over streak might mean:
        #   - Mean reversion is due → predict Under
        #   - Or momentum continues → predict Over
        # The MODEL learns which one is true from the data.
        if self._last_was_over is True:
            features["ou_streak"] = self._over_streak    # Positive = Over streak
        elif self._last_was_over is False:
            features["ou_streak"] = -self._under_streak  # Negative = Under streak
        else:
            features["ou_streak"] = 0  # No data yet
        
        # ─── Feature 3: Digit Gap ───
        # Difference between the current digit and the previous digit.
        # Large positive gap (e.g., 1 → 8, gap=+7) = sudden jump high
        # Large negative gap (e.g., 9 → 2, gap=-7) = sudden drop
        # The model can learn that big gaps often reverse (mean reversion
        # in digit space) or that they indicate a volatile period.
        if len(digits) >= 2:
            features["digit_gap"] = digits[-1] - digits[-2]
        else:
            features["digit_gap"] = 0
        
        # ─── Feature 4: Digit Autocorrelation ───
        # Correlation between consecutive digits in the short window.
        # Positive autocorrelation: high digits follow high digits, 
        # low follow low → the digit sequence has MOMENTUM.
        # Negative autocorrelation: high follows low and vice versa →
        # the digit sequence OSCILLATES (mean reversion).
        # Near zero: digits are independent (random, no edge).
        if len(digits) >= 10:
            recent = digits[-50:] if len(digits) >= 50 else digits
            n = len(recent) - 1
            if n > 0:
                mean_d = sum(recent) / len(recent)
                num = sum((recent[i] - mean_d) * (recent[i+1] - mean_d) for i in range(n))
                den1 = sum((recent[i] - mean_d) ** 2 for i in range(n))
                den2 = sum((recent[i+1] - mean_d) ** 2 for i in range(n))
                denom = (den1 * den2) ** 0.5
                features["digit_autocorr"] = num / denom if denom > 0 else 0.0
            else:
                features["digit_autocorr"] = 0.0
        else:
            features["digit_autocorr"] = 0.0
        
        # ─── Feature 5: Over/Under Reversal Rate ───
        # How often does Over follow Under (or vice versa) in recent ticks?
        # High reversal rate (>0.6) = digits are oscillating O/U/O/U
        # Low reversal rate (<0.4) = digits cluster in Over or Under runs
        # This tells the model whether to expect a reversal or continuation.
        if len(digits) >= 10:
            recent = digits[-30:] if len(digits) >= 30 else digits
            ou_sequence = [1 if d > OVER_BARRIER else 0 for d in recent]
            reversals = sum(
                1 for i in range(1, len(ou_sequence))
                if ou_sequence[i] != ou_sequence[i-1]
            )
            features["ou_reversal_rate"] = reversals / (len(ou_sequence) - 1)
        else:
            features["ou_reversal_rate"] = 0.5
        
        # ─── Feature 6: Digit Moving Average ───
        # Average of the last N digit values. If the average is > 4.5,
        # digits are trending high → Over has an edge. If < 4.5,
        # digits are trending low → Under has an edge.
        # This is a SMOOTHER version of the digit frequency features.
        if len(digits) >= 5:
            features["digit_ma_short"] = sum(digits[-5:]) / 5.0
        else:
            features["digit_ma_short"] = 4.5
        
        if len(digits) >= 20:
            features["digit_ma_long"] = sum(digits[-20:]) / 20.0
        else:
            features["digit_ma_long"] = 4.5
        
        # ─── Feature 7: Digit Velocity ───
        # Rate of change of the digit moving average.
        # If digit_ma_short > digit_ma_long → digits are trending UP
        # If digit_ma_short < digit_ma_long → digits are trending DOWN
        # This captures MOMENTUM in digit space specifically.
        features["digit_velocity"] = features["digit_ma_short"] - features["digit_ma_long"]
        
        # ─── Feature 8: Current Digit Percentile ───
        # Where does the current digit sit in the recent distribution?
        # If digit 8 appears and it's at the 90th percentile, it's
        # unusually high → might mean revert down. If digit 3 is at
        # the 10th percentile, it's unusually low → might mean revert up.
        if len(digits) >= 20:
            recent = digits[-50:] if len(digits) >= 50 else digits
            below = sum(1 for d in recent if d < current_digit)
            features["digit_percentile"] = below / len(recent)
        else:
            features["digit_percentile"] = 0.5
        
        # ─── Feature 9: Last 2 Digits Encoded ───
        # The last 2 digits as a single number (0-99).
        # This gives tree models an efficient way to split on
        # specific 2-digit patterns. For example, if "73" (digit 7
        # followed by digit 3) often leads to Over, the tree can
        # learn "if last_2_digits == 73 → predict Over".
        if len(digits) >= 2:
            features["last_2_digits"] = digits[-2] * 10 + digits[-1]
        else:
            features["last_2_digits"] = 50
        
        return features
    
    def _trend_features(self) -> dict:
        """
        Trend Slope Features (v4 — Linear Regression on Price)
        =========================================================
        Computes linear regression slope of prices over 50-tick and 200-tick
        windows to detect market trend direction.

        For Deriv Over/Under contracts, the trend of the PRICE matters because:
        - Uptrend → higher digits more likely → Over contracts have an edge
        - Downtrend → lower digits more likely → Under contracts have an edge
        - Ranging → no directional bias → trade normally

        The slope is expressed as a t-statistic (slope / standard error),
        which is market-independent and comparable across instruments.
        A t-statistic > 2.0 means the trend is statistically significant.

        Uses windows of 50 (short, responsive) and 200 (medium, stable)
        ticks. Window of 10 is too small to show meaningful trends.

        The signal generator uses these to BIAS trade selection (lower
        confidence threshold for trend-aligned trades), NOT to restrict
        trades. Ranging markets trade normally with no penalty.
        """
        features = {}

        for window_name, n_ticks in [("short", 50), ("medium", 200)]:
            ticks = self.agg.get_window(window_name)
            n = len(ticks)

            if n < 20:
                features[f"slope_{n_ticks}"] = 0.0
                features[f"slope_tstat_{n_ticks}"] = 0.0
                features[f"slope_r2_{n_ticks}"] = 0.0
                continue

            # ─── Linear Regression: y = a + b*x ───
            # x = tick index (0, 1, ..., n-1), y = price
            prices = [t.quote for t in ticks]
            x_mean = (n - 1) / 2.0
            y_mean = sum(prices) / n

            # Sum of squares: SS_xy and SS_xx
            ss_xy = sum((i - x_mean) * (prices[i] - y_mean) for i in range(n))
            ss_xx = sum((i - x_mean) ** 2 for i in range(n))

            if ss_xx == 0 or n < 3:
                features[f"slope_{n_ticks}"] = 0.0
                features[f"slope_tstat_{n_ticks}"] = 0.0
                features[f"slope_r2_{n_ticks}"] = 0.0
                continue

            slope = ss_xy / ss_xx

            # ─── Standard error of the slope ───
            # SE(slope) = sqrt(SS_res / (n-2) / SS_xx)
            # where SS_res = sum of squared residuals
            y_pred = [y_mean + slope * (i - x_mean) for i in range(n)]
            ss_res = sum((prices[i] - y_pred[i]) ** 2 for i in range(n))

            if ss_res > 0 and n > 2:
                se_slope = (ss_res / (n - 2) / ss_xx) ** 0.5
                t_stat = slope / se_slope
            else:
                se_slope = 0.0
                t_stat = 0.0

            # ─── Normalized slope (% change per tick relative to mean price) ───
            norm_slope = slope / y_mean if y_mean > 0 else 0.0

            # ─── R-squared (coefficient of determination) ───
            ss_tot = sum((p - y_mean) ** 2 for p in prices)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            features[f"slope_{n_ticks}"] = norm_slope
            features[f"slope_tstat_{n_ticks}"] = t_stat
            features[f"slope_r2_{n_ticks}"] = max(0.0, r_squared)  # Clamp negative R²

        # ─── Trend Regime Classification ───
        # Primary: 200-tick slope (more stable, less noise)
        # Confirmation: 50-tick slope (more responsive, catches turns earlier)
        # Both must agree for a trend classification.
        # Ranging = no agreement or neither is significant.
        tstat_50 = features.get("slope_tstat_50", 0.0)
        tstat_200 = features.get("slope_tstat_200", 0.0)

        if tstat_200 > 2.0 and tstat_50 > 0:
            # Both agree: uptrend (200 is significantly positive, 50 is also positive)
            features["trend_regime"] = 1   # Uptrend
        elif tstat_200 < -2.0 and tstat_50 < 0:
            # Both agree: downtrend (200 is significantly negative, 50 is also negative)
            features["trend_regime"] = -1  # Downtrend
        else:
            features["trend_regime"] = 0   # Ranging / no clear trend

        return features

    def get_feature_names(self) -> list:
        """
        Get ordered list of all feature names.
        Call after at least one compute_features() call.
        """
        if self.last_features:
            return list(self.last_features.keys())
        return []
    
    def create_label(self, ticks_ahead: int = 5, barrier: int = OVER_BARRIER, 
                     over: bool = True) -> Optional[int]:
        """
        Create training label from future ticks.
        
        Returns:
            1 if the condition was met in the next N ticks, 0 otherwise.
        
        This should be called AFTER the future ticks have arrived.
        Store the tick index at prediction time, then check if any
        of the next N ticks had digit > barrier.
        """
        # This is used by the training pipeline — see warmup_trainer.py
        # For online learning, the label is determined after the contract settles
        pass
