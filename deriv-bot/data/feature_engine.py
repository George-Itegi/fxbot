"""
Feature Engine
===============
Computes 30+ features for Over/Under prediction from tick data.
Features are returned as a flat dict suitable for River/scikit-learn.
"""

import math
from typing import Optional

from data.tick_aggregator import TickAggregator
from config import TICK_WINDOWS, OVER_BARRIER, UNDER_BARRIER
from utils.logger import setup_logger

logger = setup_logger("data.feature_engine")


class FeatureEngine:
    """
    Computes all features needed for the Over/Under model.
    
    Feature groups:
    1. Digit Distribution (10 features)
    2. Volatility (5 features)
    3. Momentum (5 features)
    4. Temporal (4 features)
    5. Pattern (4 features)
    6. Derived/Interaction (5 features)
    
    Total: ~33 features
    """
    
    def __init__(self, aggregator: TickAggregator):
        self.agg = aggregator
        self.last_features: Optional[dict] = None
    
    def compute_features(self) -> Optional[dict]:
        """
        Compute full feature vector from current tick buffers.
        Returns None if aggregator doesn't have enough data yet.
        """
        if not self.agg.is_warm("short"):
            return None
        
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
        
        self.last_features = features
        return features
    
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
        
        # Volatility × Entropy interaction
        vol = base.get("volatility_ratio", 1.0)
        ent = base.get("entropy_short", 3.0)
        features["vol_x_entropy"] = vol * ent
        
        # Barrier hit rate × volatility interaction
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


# Need time module for temporal features
import time
