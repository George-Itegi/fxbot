"""
Tick Aggregator
================
Buffers raw ticks into rolling windows for feature computation.
Uses collections.deque for O(1) append/discard at both ends.

CRITICAL: Decimal places must match Deriv's settings per symbol.
The "last digit" that Deriv uses to resolve Over/Under contracts
depends on the number of decimal places (pip_size) of the instrument.
"""

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

from config import TICK_WINDOWS
from utils.logger import setup_logger

logger = setup_logger("data.tick_aggregator")


@dataclass
class Tick:
    """Single tick data point."""
    epoch: float          # Unix timestamp
    quote: float          # Price
    symbol: str           # Instrument symbol
    digit: int            # Last digit of the price (at correct decimal precision)
    direction: int        # 1=up, -1=down, 0=same
    tick_index: int = 0   # Monotonically increasing counter


class TickAggregator:
    """
    Maintains rolling buffers of ticks across multiple time windows.
    
    Features:
    - O(1) tick insertion via deque
    - Multiple window sizes for multi-scale analysis
    - Pre-computed statistics per window
    - Digit extraction with CORRECT decimal places per symbol
    - Auto-detection of decimal places from Deriv tick pip_size
    
    DECIMAL PLACES REFERENCE (verify with Deriv — these can change):
        R_10  (Volatility 10)  → 2 decimal places
        R_25  (Volatility 25)  → 2 decimal places
        R_50  (Volatility 50)  → 2 decimal places
        R_75  (Volatility 75)  → 4 decimal places
        R_100 (Volatility 100) → 2 decimal places
    """
    
    def __init__(self, symbol: str, decimal_places: int = None):
        self.symbol = symbol
        self.tick_count = 0
        self.last_tick: Optional[Tick] = None
        
        # ─── Decimal places for digit extraction ───
        # This is CRITICAL: Deriv resolves Over/Under based on the LAST digit
        # of the price at the correct decimal precision. If we use the wrong
        # number of decimal places, we extract the WRONG digit and the
        # entire bot's predictions and learning are garbage.
        #
        # Source priority:
        #   1. Explicitly passed decimal_places (from Deriv active_symbols API)
        #   2. Auto-detected from pip_size in tick responses
        #   3. Fallback from config SYMBOLS dict
        #   4. Default: 2
        self.decimal_places: Optional[int] = decimal_places
        self._decimal_places_logged = False  # Only log once when first detected
        
        # Rolling buffers — one deque per window size
        self.windows: dict[str, deque] = {}
        for name, size in TICK_WINDOWS.items():
            self.windows[name] = deque(maxlen=size)
        
        # Digit tracking buffers (for digit distribution analysis)
        self.digit_buffer: deque = deque(maxlen=TICK_WINDOWS["long"])
        
        # Barrier hit tracking
        self.barrier_hits: dict[int, deque] = {}  # barrier -> deque of bools
        for barrier in range(10):
            self.barrier_hits[barrier] = deque(maxlen=TICK_WINDOWS["long"])
        
        # ─── Digit extraction verification ───
        # Store raw price strings for the first few ticks to verify
        # that our decimal places match what Deriv actually shows
        self._verification_samples: list[dict] = []
        
        logger.info(f"TickAggregator initialized for {symbol} "
                     f"(windows: {list(TICK_WINDOWS.values())}, "
                     f"decimal_places: {decimal_places or 'auto-detect'})")
    
    def add_tick(self, epoch: float, quote: float, decimal_places: int = None) -> Tick:
        """
        Add a new tick and return the Tick object.
        
        Args:
            epoch: Unix timestamp from Deriv
            quote: Price quote from Deriv
            decimal_places: Number of decimal places for this symbol.
                           If provided, updates our cached value.
                           If not provided, uses cached or auto-detects.
        """
        # ─── Auto-detect / update decimal places ───
        if decimal_places is not None:
            if self.decimal_places is None or self.decimal_places != decimal_places:
                old_dp = self.decimal_places
                self.decimal_places = decimal_places
                if not self._decimal_places_logged:
                    logger.info(
                        f"  {self.symbol}: decimal_places={decimal_places} "
                        f"(was: {old_dp}, source: tick pip_size)"
                    )
                    self._decimal_places_logged = True
        
        # ─── Extract last digit using correct precision ───
        dp = self.decimal_places or 2  # Default to 2 if not yet detected
        price_str = f"{quote:.{dp}f}"
        digit = int(price_str[-1])
        
        # ─── Validation: log first few extractions for verification ───
        if self.tick_count < 5:
            sample = {
                "raw_quote": quote,
                "decimal_places": dp,
                "formatted": price_str,
                "last_digit": digit,
            }
            self._verification_samples.append(sample)
            logger.info(
                f"  Tick #{self.tick_count}: quote={quote}, dp={dp}, "
                f"formatted=\"{price_str}\", last_digit={digit}"
            )
        
        # ─── Cross-check: verify digit is in valid range ───
        if digit < 0 or digit > 9:
            logger.warning(
                f"Invalid digit {digit} extracted from quote={quote}, dp={dp}, "
                f"formatted=\"{price_str}\" — defaulting to 0"
            )
            digit = max(0, min(digit, 9))
        
        # Determine direction
        if self.last_tick is not None:
            direction = 1 if quote > self.last_tick.quote else (
                -1 if quote < self.last_tick.quote else 0
            )
        else:
            direction = 0
        
        tick = Tick(
            epoch=epoch,
            quote=quote,
            symbol=self.symbol,
            digit=digit,
            direction=direction,
            tick_index=self.tick_count,
        )
        
        self.tick_count += 1
        
        # Add to all windows
        for window in self.windows.values():
            window.append(tick)
        
        # Track digits
        self.digit_buffer.append(digit)
        
        # Track barrier hits (digit > barrier)
        for barrier in range(10):
            self.barrier_hits[barrier].append(digit > barrier)
        
        self.last_tick = tick
        return tick
    
    def get_window(self, name: str) -> list:
        """Get all ticks in a named window."""
        return list(self.windows.get(name, []))
    
    def get_window_size(self, name: str) -> int:
        """Get current size of a window (may be less than maxlen)."""
        return len(self.windows.get(name, []))
    
    def is_warm(self, min_window: str = "medium") -> bool:
        """Check if we have enough data for feature computation."""
        min_ticks = TICK_WINDOWS.get(min_window, 200)
        return self.tick_count >= min_ticks
    
    # ─── Pre-computed Statistics ───
    
    def digit_distribution(self, window: str = "short") -> dict[int, float]:
        """
        Get frequency distribution of digits 0-9 in a window.
        Returns dict of {digit: frequency}.
        """
        ticks = self.windows.get(window, [])
        if not ticks:
            return {d: 0.1 for d in range(10)}  # Uniform prior
        
        counts = {d: 0 for d in range(10)}
        for t in ticks:
            counts[t.digit] += 1
        
        total = len(ticks)
        return {d: c / total for d, c in counts.items()}
    
    def barrier_hit_rate(self, barrier: int, window: str = "short") -> float:
        """
        Get the rate of digits > barrier in a window.
        This is the key metric for Over/Under prediction.
        """
        hits = self.barrier_hits.get(barrier, deque())
        n = min(len(hits), TICK_WINDOWS.get(window, 50))
        
        if n == 0:
            return 0.5  # Assume uniform
        
        recent = list(hits)[-n:]
        return sum(recent) / len(recent)
    
    def price_std(self, window: str = "short") -> float:
        """Standard deviation of prices in a window."""
        ticks = self.windows.get(window, [])
        if len(ticks) < 2:
            return 0.0
        quotes = [t.quote for t in ticks]
        mean = sum(quotes) / len(quotes)
        variance = sum((q - mean) ** 2 for q in quotes) / (len(quotes) - 1)
        return variance ** 0.5
    
    def price_range(self, window: str = "short") -> float:
        """Max - Min price in a window."""
        ticks = self.windows.get(window, [])
        if not ticks:
            return 0.0
        quotes = [t.quote for t in ticks]
        return max(quotes) - min(quotes)
    
    def tick_rate(self, window: str = "short") -> float:
        """
        Average ticks per second in a window.
        Measures how fast the market is moving.
        """
        ticks = self.windows.get(window, [])
        if len(ticks) < 2:
            return 0.0
        duration = ticks[-1].epoch - ticks[0].epoch
        if duration <= 0:
            return float(len(ticks))
        return (len(ticks) - 1) / duration
    
    def direction_bias(self, window: str = "short") -> float:
        """
        Ratio of up-ticks to total ticks.
        >0.5 = bullish bias, <0.5 = bearish bias.
        """
        ticks = self.windows.get(window, [])
        if not ticks:
            return 0.5
        up = sum(1 for t in ticks if t.direction == 1)
        return up / len(ticks)
    
    def consecutive_same_digit(self) -> int:
        """Count of consecutive identical digits at the end of buffer."""
        if len(self.digit_buffer) < 2:
            return len(self.digit_buffer)
        
        count = 1
        last = self.digit_buffer[-1]
        for d in reversed(list(self.digit_buffer)[:-1]):
            if d == last:
                count += 1
            else:
                break
        return count
    
    def entropy(self, window: str = "short") -> float:
        """
        Shannon entropy of digit distribution.
        Higher = more uniform/random.
        Lower = more predictable digit patterns.
        """
        dist = self.digit_distribution(window)
        entropy = 0.0
        for p in dist.values():
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy
    
    def summary(self) -> dict:
        """Get current aggregator state summary."""
        return {
            "symbol": self.symbol,
            "total_ticks": self.tick_count,
            "last_quote": self.last_tick.quote if self.last_tick else 0,
            "last_digit": self.last_tick.digit if self.last_tick else -1,
            "decimal_places": self.decimal_places,
            "is_warm": self.is_warm(),
            "windows": {
                name: len(w) for name, w in self.windows.items()
            },
        }
