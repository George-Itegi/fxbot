"""
Setup Detector — Market Setup Quality Assessment (v8)
======================================================
Encapsulates the trader's manual decision process:

1. Is there a strong trend? (3-window linear regression slope)
2. Is there a clear digit frequency edge? (Over vs Under frequency)
3. Are all signals aligned? (trend + frequency agree on direction)
4. How strong is the setup? (composite quality score 0-1)

This replaces the old "model agreement" system with something the trader
can VERIFY: "Trend is up, Over-frequency is 57%, all 3 windows agree,
setup score is 0.82 — I understand why the bot wants to trade Over."

The setup score drives EVERYTHING:
- High setup score (0.75+) -> trade with confidence, allow martingale
- Medium setup score (0.60-0.75) -> trade normally
- Low setup score (<0.60) -> NO TRADE, wait for better setup
"""

import time
from dataclasses import dataclass
from typing import Optional

from config import (
    OVER_BARRIER, UNDER_BARRIER,
    TREND_SLOPE_TSTAT_THRESHOLD,
    MIN_DIGIT_FREQUENCY_EDGE,
    DIGIT_FREQ_WINDOW_AGREEMENT,
    MIN_SETUP_SCORE,
    PROFIT_TARGET_PER_MARKET,
    OBSERVATION_PERIOD_SEC,
    MIN_OBSERVATION_TICKS,
    CONTRACT_TYPE_OVER, CONTRACT_TYPE_UNDER,
)
from utils.logger import setup_logger

logger = setup_logger("trading.setup_detector")


@dataclass
class Setup:
    """Detected market setup — the basis for a trade decision."""
    active: bool = False               # Is there a valid setup right now?
    direction: str = ""                # "DIGITOVER" or "DIGITUNDER"
    trend_regime: int = 0              # 1=uptrend, -1=downtrend, 0=ranging
    trend_strength: float = 0.0        # Min t-stat across 200/500 windows
    over_freq: float = 0.5             # Average Over-frequency across windows
    under_freq: float = 0.5            # Average Under-frequency across windows
    freq_edge: float = 0.0             # |over_freq - under_freq| — digit frequency edge
    freq_direction: int = 0            # 1=Over dominant, -1=Under dominant
    window_agreement: int = 0          # How many windows agree on freq direction (0-3)
    trend_freq_aligned: bool = False   # Does trend direction match frequency direction?
    setup_score: float = 0.0           # Composite quality score (0-1)
    reason: str = ""                   # Human-readable explanation
    detected_at: float = 0.0           # When was this setup detected?
    observation_complete: bool = False  # Has the observation phase finished?
    observed_duration: int = 5         # Duration determined by observation (ticks)


@dataclass
class MarketSession:
    """Tracks a trading session on one market."""
    symbol: str
    session_start: float = 0.0         # When did this session start?
    session_pnl: float = 0.0           # P&L for this session
    session_trades: int = 0            # Number of trades in this session
    session_wins: int = 0              # Wins in this session
    profit_target_reached: bool = False # Has the profit target been reached?
    setup_broken: bool = False         # Has the setup broken?
    last_trade_time: float = 0.0       # When was the last trade in this session?
    current_setup: Optional[Setup] = None  # Current active setup


class ObservationTracker:
    """
    Tracks digit movement during the observation phase.
    
    When a setup is detected, we WATCH the market for 20-30 seconds.
    During this time, we track:
    - When a non-dominant digit appears, how many ticks until a dominant digit appears?
    - This tells us the OPTIMAL tick duration for the contract.
    
    Example: Setup says Over (digits 5-9 dominant).
    - Digit 3 appears (Under)
    - 2 ticks later, digit 7 appears (Over)
    - So duration from Under -> Over = 2 ticks
    - Average over observation = 3 ticks -> use 3t duration
    """
    
    def __init__(self, observation_sec: float = OBSERVATION_PERIOD_SEC,
                 min_ticks: int = MIN_OBSERVATION_TICKS):
        self._observation_sec = observation_sec
        self._min_ticks = min_ticks
        self._start_time: float = 0.0
        self._tick_count: int = 0
        self._direction: int = 0  # 1=Over dominant, -1=Under dominant
        self._flip_durations: list = []  # How many ticks from non-dominant to dominant
        self._last_non_dominant_tick: int = 0  # Tick index when last non-dominant appeared
        self._in_non_dominant: bool = False
        self._complete: bool = False
    
    def start(self, direction: int):
        """Start observation phase. direction: 1=Over, -1=Under."""
        self._start_time = time.time()
        self._tick_count = 0
        self._direction = direction
        self._flip_durations = []
        self._last_non_dominant_tick = 0
        self._in_non_dominant = False
        self._complete = False
    
    def observe_tick(self, digit: int):
        """
        Process a tick during the observation phase.
        Returns True if observation is complete.
        """
        if self._complete:
            return True
        
        self._tick_count += 1
        is_dominant = (digit > OVER_BARRIER) if self._direction == 1 else (digit < UNDER_BARRIER)
        
        if not is_dominant:
            # Non-dominant digit appeared — start counting
            if not self._in_non_dominant:
                self._in_non_dominant = True
                self._last_non_dominant_tick = self._tick_count
        else:
            # Dominant digit appeared
            if self._in_non_dominant:
                # Flipped from non-dominant to dominant — record the duration
                flip_duration = self._tick_count - self._last_non_dominant_tick
                if flip_duration > 0:
                    self._flip_durations.append(flip_duration)
                self._in_non_dominant = False
        
        # Check if observation is complete
        elapsed = time.time() - self._start_time
        if elapsed >= self._observation_sec and self._tick_count >= self._min_ticks:
            self._complete = True
        
        return self._complete
    
    def get_recommended_duration(self) -> int:
        """
        Determine the optimal tick duration from observation data.
        
        Uses the MODE (most common) flip duration, not the average.
        If most flips happen in 2 ticks, use 2t — that's the most reliable.
        Falls back to average if no clear mode.
        Minimum duration: 2 ticks (1t is too noisy).
        """
        if not self._flip_durations:
            return 5  # Default if no data
        
        from collections import Counter
        counts = Counter(self._flip_durations)
        mode_duration = counts.most_common(1)[0][0]
        
        # Also compute average for context
        avg_duration = sum(self._flip_durations) / len(self._flip_durations)
        
        # Use mode if we have enough data, else use average
        if len(self._flip_durations) >= 5:
            recommended = mode_duration
        else:
            recommended = round(avg_duration)
        
        # Enforce bounds
        recommended = max(2, min(10, recommended))  # 2-10 ticks
        
        return recommended
    
    def summary(self) -> dict:
        return {
            "ticks_observed": self._tick_count,
            "flip_count": len(self._flip_durations),
            "flip_durations": self._flip_durations[-10:],  # Last 10
            "recommended_duration": self.get_recommended_duration() if self._flip_durations else None,
            "complete": self._complete,
            "elapsed_sec": round(time.time() - self._start_time, 1) if self._start_time > 0 else 0,
        }


class SetupDetector:
    """
    Detects and scores market setups for quality trading.
    
    A "setup" is when:
    1. A strong trend is detected (3-window agreement at 3-sigma)
    2. Digit frequency shows a clear Over/Under edge
    3. Trend direction and frequency direction are ALIGNED
    4. The setup quality score meets the minimum threshold
    
    This mirrors the manual trading process:
    - "Is the market trending?" -> Trend check
    - "Which side has more digits?" -> Digit frequency check  
    - "Is it Over or Under dominant?" -> Direction determination
    - "How strong is this setup?" -> Quality scoring
    """
    
    def __init__(self):
        self._observation_trackers: dict[str, ObservationTracker] = {}
        self._sessions: dict[str, MarketSession] = {}
        self._setup_cache: dict[str, Setup] = {}
        
        logger.info("SetupDetector initialized: trend + digit frequency quality system")
    
    def evaluate(self, symbol: str, features: dict) -> Setup:
        """
        Evaluate the current market setup.
        
        This is the MAIN method — called on every tick for each market.
        Returns a Setup object describing the current market state.
        """
        setup = Setup()
        
        # ─── Step 1: Trend Check ───
        trend_regime = features.get("trend_regime", 0)
        tstat_50 = features.get("slope_tstat_50", 0.0)
        tstat_200 = features.get("slope_tstat_200", 0.0)
        tstat_500 = features.get("slope_tstat_500", 0.0)
        
        setup.trend_regime = trend_regime
        
        if trend_regime == 0:
            setup.active = False
            setup.reason = f"No trend (t50={tstat_50:.1f}, t200={tstat_200:.1f}, t500={tstat_500:.1f})"
            self._check_session_break(symbol, setup)
            self._setup_cache[symbol] = setup
            return setup
        
        # Trend strength = minimum of the significant t-stats
        if trend_regime == 1:
            setup.trend_strength = min(tstat_200, tstat_500)
            setup.reason = f"Uptrend (t50={tstat_50:.1f}, t200={tstat_200:.1f}, t500={tstat_500:.1f})"
        else:
            setup.trend_strength = min(abs(tstat_200), abs(tstat_500))
            setup.reason = f"Downtrend (t50={tstat_50:.1f}, t200={tstat_200:.1f}, t500={tstat_500:.1f})"
        
        # ─── Step 2: Digit Frequency Check ───
        over_freq_short = features.get("over_freq_short", 0.5)
        over_freq_medium = features.get("over_freq_medium", 0.5)
        over_freq_long = features.get("over_freq_trend_long", 0.5)
        
        under_freq_short = features.get("under_freq_short", 0.5)
        under_freq_medium = features.get("under_freq_medium", 0.5)
        under_freq_long = features.get("under_freq_trend_long", 0.5)
        
        # Average across windows (weighted toward medium and long)
        setup.over_freq = (over_freq_short * 0.2 + over_freq_medium * 0.4 + over_freq_long * 0.4)
        setup.under_freq = (under_freq_short * 0.2 + under_freq_medium * 0.4 + under_freq_long * 0.4)
        
        # Edge: how far from 50/50?
        setup.freq_edge = abs(setup.over_freq - setup.under_freq)
        
        # Direction: which side is dominant?
        if setup.over_freq > setup.under_freq:
            setup.freq_direction = 1  # Over dominant
        else:
            setup.freq_direction = -1  # Under dominant
        
        # Window agreement: how many windows agree on the direction?
        setup.window_agreement = features.get("ou_window_agreement", 0)
        
        # ─── Step 3: Trend + Frequency Alignment ───
        # CRITICAL: trend direction MUST match frequency direction
        # Uptrend + Over-frequency dominant -> ALIGNED -> strong setup
        # Uptrend + Under-frequency dominant -> CONFLICT -> weak setup
        setup.trend_freq_aligned = (trend_regime == setup.freq_direction)
        
        if not setup.trend_freq_aligned:
            setup.active = False
            setup.reason += f" | MISALIGNED: trend={'UP' if trend_regime==1 else 'DOWN'} but freq={'Over' if setup.freq_direction==1 else 'Under'} dominant"
            self._check_session_break(symbol, setup)
            self._setup_cache[symbol] = setup
            return setup
        
        # ─── Step 4: Setup Quality Score ───
        # Score is a composite of:
        # 1. Trend strength (t-stat magnitude, 0-1 scale)
        # 2. Frequency edge (how far from 50/50, 0-1 scale)
        # 3. Window agreement (0-3, normalized)
        # 4. Alignment bonus (already confirmed, gives +0.1)
        
        # Trend strength score: t-stat 3.0 = 0.5, 5.0 = 0.75, 10+ = 1.0
        trend_score = min(1.0, (setup.trend_strength - TREND_SLOPE_TSTAT_THRESHOLD) / 7.0 + 0.5)
        
        # Frequency edge score: 2% = 0.3, 5% = 0.5, 10%+ = 0.8
        if setup.freq_edge >= MIN_DIGIT_FREQUENCY_EDGE:
            freq_score = min(0.8, 0.3 + (setup.freq_edge - MIN_DIGIT_FREQUENCY_EDGE) * 10)
        else:
            freq_score = setup.freq_edge / MIN_DIGIT_FREQUENCY_EDGE * 0.3
        
        # Window agreement score: 1 window = 0.3, 2 = 0.6, 3 = 0.9
        agreement_score = setup.window_agreement / 3.0 * 0.9
        
        # Alignment bonus
        alignment_bonus = 0.1
        
        # Composite score (weighted average)
        setup.setup_score = (
            trend_score * 0.35 +      # Trend is important but not everything
            freq_score * 0.35 +       # Digit frequency is equally important
            agreement_score * 0.20 +  # Window agreement validates
            alignment_bonus           # Trend-freq alignment bonus
        )
        
        # ─── Step 5: Determine Direction ───
        if trend_regime == 1 and setup.freq_direction == 1:
            setup.direction = CONTRACT_TYPE_OVER
        elif trend_regime == -1 and setup.freq_direction == -1:
            setup.direction = CONTRACT_TYPE_UNDER
        else:
            # Should not reach here (alignment check above), but safety net
            setup.direction = CONTRACT_TYPE_OVER if trend_regime == 1 else CONTRACT_TYPE_UNDER
        
        # ─── Step 6: Is the setup active? ───
        setup.active = setup.setup_score >= MIN_SETUP_SCORE
        
        if setup.active:
            setup.reason += (
                f" | {setup.direction.replace('DIGIT','')} setup "
                f"score={setup.setup_score:.2f} "
                f"(trend={trend_score:.2f} freq={freq_score:.2f} agree={agreement_score:.2f}) "
                f"Over={setup.over_freq:.1%} Under={setup.under_freq:.1%} "
                f"edge={setup.freq_edge:.1%} windows={setup.window_agreement}/3"
            )
            setup.detected_at = time.time()
        else:
            setup.reason += (
                f" | Setup too weak: score={setup.setup_score:.2f} < {MIN_SETUP_SCORE} "
                f"(trend={trend_score:.2f} freq={freq_score:.2f} agree={agreement_score:.2f})"
            )
            self._check_session_break(symbol, setup)
        
        self._setup_cache[symbol] = setup
        return setup
    
    def start_observation(self, symbol: str, direction: int) -> ObservationTracker:
        """Start the observation phase for a market."""
        tracker = ObservationTracker()
        tracker.start(direction)
        self._observation_trackers[symbol] = tracker
        logger.info(f"[{symbol}] OBSERVATION PHASE started: watching digits for {OBSERVATION_PERIOD_SEC}s")
        return tracker
    
    def observe_tick(self, symbol: str, digit: int) -> bool:
        """
        Process a tick during observation phase.
        Returns True if observation is complete.
        """
        tracker = self._observation_trackers.get(symbol)
        if tracker is None:
            return True  # No observation needed
        
        complete = tracker.observe_tick(digit)
        
        if complete and not hasattr(tracker, '_logged_complete'):
            tracker._logged_complete = True
            duration = tracker.get_recommended_duration()
            logger.info(
                f"[{symbol}] OBSERVATION COMPLETE: "
                f"recommended duration={duration}t "
                f"from {len(tracker._flip_durations)} flip observations "
                f"over {tracker._tick_count} ticks"
            )
        
        return complete
    
    def get_observed_duration(self, symbol: str) -> int:
        """Get the duration determined by observation for a market."""
        tracker = self._observation_trackers.get(symbol)
        if tracker and tracker._complete:
            return tracker.get_recommended_duration()
        return 5  # Default
    
    def is_observing(self, symbol: str) -> bool:
        """Check if a market is currently in observation phase."""
        tracker = self._observation_trackers.get(symbol)
        return tracker is not None and not tracker._complete
    
    def clear_observation(self, symbol: str):
        """Clear observation state for a market."""
        self._observation_trackers.pop(symbol, None)
    
    # ─── Market Session Management ───
    
    def get_or_create_session(self, symbol: str) -> MarketSession:
        """Get or create a trading session for a market."""
        if symbol not in self._sessions:
            self._sessions[symbol] = MarketSession(symbol=symbol)
        return self._sessions[symbol]
    
    def record_session_trade(self, symbol: str, won: bool, pnl: float):
        """Record a trade result in the market session."""
        session = self.get_or_create_session(symbol)
        
        if session.session_start == 0:
            session.session_start = time.time()
        
        session.session_trades += 1
        session.session_pnl += pnl
        session.last_trade_time = time.time()
        
        if won:
            session.session_wins += 1
        
        # Check profit target
        if session.session_pnl >= PROFIT_TARGET_PER_MARKET:
            session.profit_target_reached = True
            logger.info(
                f"[{symbol}] PROFIT TARGET REACHED: "
                f"${session.session_pnl:.2f} >= ${PROFIT_TARGET_PER_MARKET:.0f} "
                f"({session.session_trades} trades, {session.session_wins} wins). "
                f"Stopping this market until new setup."
            )
        
        return session
    
    def is_market_tradable(self, symbol: str) -> bool:
        """
        Check if a market is still tradable.
        Not tradable if profit target reached or setup broken.
        """
        session = self._sessions.get(symbol)
        if session is None:
            return True  # No session = never traded = tradable
        
        if session.profit_target_reached:
            # Profit target reached — check if enough time has passed for a new setup
            # Or if a new strong setup has appeared
            setup = self._setup_cache.get(symbol)
            if setup and setup.active and setup.setup_score >= 0.75:
                # Very strong new setup — reset session
                logger.info(
                    f"[{symbol}] New strong setup detected (score={setup.setup_score:.2f}). "
                    f"Resetting session — previous profit target was reached."
                )
                self._sessions[symbol] = MarketSession(symbol=symbol)
                return True
            return False
        
        if session.setup_broken:
            return False
        
        return True
    
    def _check_session_break(self, symbol: str, setup: Setup):
        """Check if the current session's setup has broken."""
        session = self._sessions.get(symbol)
        if session is None:
            return
        
        if session.current_setup and session.current_setup.active:
            if not setup.active:
                session.setup_broken = True
                logger.info(
                    f"[{symbol}] SETUP BROKEN — session ending. "
                    f"PnL=${session.session_pnl:.2f} from {session.session_trades} trades"
                )
    
    def get_session(self, symbol: str) -> Optional[MarketSession]:
        """Get the current session for a market."""
        return self._sessions.get(symbol)
    
    def reset_session(self, symbol: str):
        """Reset a market session (e.g., after profit target + new setup)."""
        self._sessions[symbol] = MarketSession(symbol=symbol)
    
    def get_setup(self, symbol: str) -> Optional[Setup]:
        """Get the cached setup for a market."""
        return self._setup_cache.get(symbol)
