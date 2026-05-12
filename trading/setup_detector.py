"""
Setup Detector — Dynamic Barrier Selection (v9)
=================================================
Evaluates ALL Over/Under barriers and picks the BEST one.

Key Insight: Over 4 / Under 5 are 50/50 contracts. A 5% frequency edge on
Over 4 gives only ~5% EV. But the same 5% edge on Over 8 (10% natural,
~895% payout) gives ~50% EV — 10x more profitable!

The setup detector now:
1. Computes per-digit frequencies across windows
2. For each barrier, calculates observed win probability and EV
3. Picks the barrier with the best risk-adjusted EV
4. Verifies statistical significance (z-score > 2)
5. Trend is a BIAS (boost/penalty), not a strict requirement

Setup Score (v9):
- Frequency edge significance (z-score) — is the deviation real?
- Window agreement — do multiple windows confirm the edge?
- Payout-adjusted EV — how much do we expect to profit?
- Trend alignment — bonus if trend agrees, penalty if it opposes
"""

import math
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
    DYNAMIC_BARRIERS,
    BARRIER_OVER_OPTIONS,
    BARRIER_UNDER_OPTIONS,
    MIN_BARRIER_PROBABILITY,
    MAX_BARRIER_PROBABILITY,
    PAYOUT_HOUSE_MARGIN,
    MIN_EV_FOR_TRADE,
    MIN_FREQ_EDGE_ZSCORE,
    BARRIER_NATURAL_PROB_OVER,
    BARRIER_NATURAL_PROB_UNDER,
    estimate_payout_rate,
    TREND_REQUIRED,
    TREND_CONFIDENCE_BOOST,
    TREND_MISALIGN_PENALTY,
    MIN_DIGIT_FREQUENCY_EDGE_RELATIVE,
)
from utils.logger import setup_logger

logger = setup_logger("trading.setup_detector")


@dataclass
class BarrierEval:
    """Evaluation of a single barrier option."""
    contract_type: str          # "DIGITOVER" or "DIGITUNDER"
    barrier: int                # The digit barrier (e.g., 7 for Over 7)
    natural_prob: float         # Natural/uniform probability (e.g., 0.20 for Over 7)
    observed_prob: float        # Observed win probability from digit frequencies
    freq_edge: float            # observed_prob - natural_prob (absolute edge)
    freq_edge_pct: float        # Edge as percentage (e.g., 5.0 for 5%)
    z_score: float              # Statistical significance of the deviation
    payout_rate: float          # Estimated payout (profit/stake ratio)
    ev: float                   # Expected value per $1 staked
    window_agreement: int       # How many windows confirm this edge (0-3)
    is_significant: bool        # Is the z-score above threshold?


@dataclass
class Setup:
    """Detected market setup — the basis for a trade decision."""
    active: bool = False               # Is there a valid setup right now?
    direction: str = ""                # "DIGITOVER" or "DIGITUNDER"
    barrier: int = 4                   # The BEST barrier to trade (v9: dynamic!)
    trend_regime: int = 0              # 1=uptrend, -1=downtrend, 0=ranging
    trend_strength: float = 0.0        # Min t-stat across 200/500 windows
    over_freq: float = 0.5             # Average Over-frequency across windows (for Over 4)
    under_freq: float = 0.5            # Average Under-frequency across windows (for Under 5)
    freq_edge: float = 0.0             # |over_freq - under_freq| for the CHOSEN barrier
    freq_direction: int = 0            # 1=Over dominant, -1=Under dominant
    window_agreement: int = 0          # How many windows agree on freq direction (0-3)
    trend_freq_aligned: bool = False   # Does trend direction match frequency direction?
    setup_score: float = 0.0           # Composite quality score (0-1)
    reason: str = ""                   # Human-readable explanation
    detected_at: float = 0.0           # When was this setup detected?
    observation_complete: bool = False  # Has the observation phase finished?
    observed_duration: int = 5         # Duration determined by observation (ticks)
    # ─── v9: Dynamic barrier fields ───
    best_barrier_eval: Optional[BarrierEval] = None  # The best barrier evaluation
    natural_prob: float = 0.5          # Natural probability for chosen barrier
    observed_prob: float = 0.5         # Observed probability for chosen barrier
    payout_rate: float = 0.95          # Payout rate for chosen barrier
    z_score: float = 0.0              # Statistical significance
    all_barrier_evals: list = None     # All evaluated barriers (for debugging)


@dataclass
class MarketSession:
    """Tracks a trading session on one market."""
    symbol: str
    session_start: float = 0.0
    session_pnl: float = 0.0
    session_trades: int = 0
    session_wins: int = 0
    profit_target_reached: bool = False
    setup_broken: bool = False
    last_trade_time: float = 0.0
    current_setup: Optional[Setup] = None


class ObservationTracker:
    """
    Tracks digit movement during the observation phase.
    
    When a setup is detected, we WATCH the market for 20-30 seconds.
    During this time, we track:
    - When a non-dominant digit appears, how many ticks until a dominant digit appears?
    - This tells us the OPTIMAL tick duration for the contract.
    """
    
    def __init__(self, observation_sec: float = OBSERVATION_PERIOD_SEC,
                 min_ticks: int = MIN_OBSERVATION_TICKS):
        self._observation_sec = observation_sec
        self._min_ticks = min_ticks
        self._start_time: float = 0.0
        self._tick_count: int = 0
        self._direction: int = 0  # 1=Over dominant, -1=Under dominant
        self._barrier: int = 4    # Which barrier we're observing for
        self._flip_durations: list = []
        self._last_non_dominant_tick: int = 0
        self._in_non_dominant: bool = False
        self._complete: bool = False
    
    def start(self, direction: int, barrier: int = 4):
        """Start observation phase. direction: 1=Over, -1=Under."""
        self._start_time = time.time()
        self._tick_count = 0
        self._direction = direction
        self._barrier = barrier
        self._flip_durations = []
        self._last_non_dominant_tick = 0
        self._in_non_dominant = False
        self._complete = False
    
    def observe_tick(self, digit: int):
        """Process a tick during observation. Returns True if complete."""
        if self._complete:
            return True
        
        self._tick_count += 1
        # Check if this digit is dominant for our chosen barrier
        if self._direction == 1:  # Over
            is_dominant = digit > self._barrier
        else:  # Under
            is_dominant = digit < self._barrier
        
        if not is_dominant:
            if not self._in_non_dominant:
                self._in_non_dominant = True
                self._last_non_dominant_tick = self._tick_count
        else:
            if self._in_non_dominant:
                flip_duration = self._tick_count - self._last_non_dominant_tick
                if flip_duration > 0:
                    self._flip_durations.append(flip_duration)
                self._in_non_dominant = False
        
        elapsed = time.time() - self._start_time
        if elapsed >= self._observation_sec and self._tick_count >= self._min_ticks:
            self._complete = True
        
        return self._complete
    
    def get_recommended_duration(self) -> int:
        """Determine optimal tick duration from observation data."""
        if not self._flip_durations:
            return 5
        
        from collections import Counter
        counts = Counter(self._flip_durations)
        mode_duration = counts.most_common(1)[0][0]
        avg_duration = sum(self._flip_durations) / len(self._flip_durations)
        
        if len(self._flip_durations) >= 5:
            recommended = mode_duration
        else:
            recommended = round(avg_duration)
        
        recommended = max(2, min(10, recommended))
        return recommended
    
    def summary(self) -> dict:
        return {
            "ticks_observed": self._tick_count,
            "flip_count": len(self._flip_durations),
            "flip_durations": self._flip_durations[-10:],
            "recommended_duration": self.get_recommended_duration() if self._flip_durations else None,
            "complete": self._complete,
            "elapsed_sec": round(time.time() - self._start_time, 1) if self._start_time > 0 else 0,
        }


class SetupDetector:
    """
    Detects and scores market setups using DYNAMIC BARRIER SELECTION.
    
    v9 Philosophy:
    - Evaluate ALL Over/Under barriers (not just Over 4 / Under 5)
    - Find the barrier with the best risk-adjusted EV
    - Lower-probability barriers have higher payouts, so small frequency
      deviations are much more valuable
    - Trend is a BIAS (boost/penalty), not a strict requirement
    - Statistical significance is REQUIRED (z-score > 2)
    """
    
    def __init__(self):
        self._observation_trackers: dict[str, ObservationTracker] = {}
        self._sessions: dict[str, MarketSession] = {}
        self._setup_cache: dict[str, Setup] = {}
        
        logger.info("SetupDetector v9 initialized: dynamic barrier selection + EV-based trading")
    
    def evaluate(self, symbol: str, features: dict) -> Setup:
        """
        Evaluate the current market setup with dynamic barrier selection.
        
        This is the MAIN method — called on every tick for each market.
        Returns a Setup object describing the best available trade.
        """
        setup = Setup()
        
        # ─── Step 1: Trend Check (v9: BIAS, not requirement) ───
        trend_regime = features.get("trend_regime", 0)
        tstat_50 = features.get("slope_tstat_50", 0.0)
        tstat_200 = features.get("slope_tstat_200", 0.0)
        tstat_500 = features.get("slope_tstat_500", 0.0)
        
        setup.trend_regime = trend_regime
        
        if trend_regime == 1:
            setup.trend_strength = min(tstat_200, tstat_500)
        elif trend_regime == -1:
            setup.trend_strength = min(abs(tstat_200), abs(tstat_500))
        else:
            setup.trend_strength = 0.0
        
        # ─── Step 2: Evaluate ALL barriers ───
        if DYNAMIC_BARRIERS:
            best_eval = self._evaluate_all_barriers(features, trend_regime)
        else:
            # Fallback: only evaluate Over 4 / Under 5
            best_eval = self._evaluate_fixed_barriers(features, trend_regime)
        
        if best_eval is None:
            setup.active = False
            setup.reason = "No barrier with significant edge found"
            self._check_session_break(symbol, setup)
            self._setup_cache[symbol] = setup
            return setup
        
        # ─── Step 3: Set up the Setup from best barrier ───
        setup.direction = best_eval.contract_type
        setup.barrier = best_eval.barrier
        setup.natural_prob = best_eval.natural_prob
        setup.observed_prob = best_eval.observed_prob
        setup.payout_rate = best_eval.payout_rate
        setup.z_score = best_eval.z_score
        setup.freq_edge = best_eval.freq_edge
        setup.best_barrier_eval = best_eval
        
        # Legacy fields (for Over 4 / Under 5 compatibility)
        setup.over_freq = features.get("over_freq_medium", 0.5)
        setup.under_freq = features.get("under_freq_medium", 0.5)
        setup.freq_direction = 1 if best_eval.contract_type == CONTRACT_TYPE_OVER else -1
        setup.window_agreement = best_eval.window_agreement
        
        # ─── Step 4: Trend alignment check ───
        # v9: Trend is a BIAS, not a requirement
        # Over + uptrend = aligned, Under + downtrend = aligned
        if trend_regime != 0:
            if (trend_regime == 1 and best_eval.contract_type == CONTRACT_TYPE_OVER) or \
               (trend_regime == -1 and best_eval.contract_type == CONTRACT_TYPE_UNDER):
                setup.trend_freq_aligned = True
            else:
                setup.trend_freq_aligned = False
        else:
            # No trend — neutral, no alignment bonus or penalty
            setup.trend_freq_aligned = True  # Neutral = OK
        
        # ─── Step 5: Setup Quality Score (v9: EV-based) ───
        setup.setup_score = self._compute_setup_score(best_eval, setup.trend_freq_aligned, trend_regime)
        
        # ─── Step 6: Is the setup active? ───
        setup.active = setup.setup_score >= MIN_SETUP_SCORE and best_eval.is_significant
        
        # Build reason string
        dir_label = "Over" if best_eval.contract_type == CONTRACT_TYPE_OVER else "Under"
        trend_label = "UP" if trend_regime == 1 else "DOWN" if trend_regime == -1 else "RANGING"
        align_label = "ALIGNED" if setup.trend_freq_aligned else "MISALIGNED"
        
        if setup.active:
            setup.reason = (
                f"{dir_label}{best_eval.barrier}: obs_prob={best_eval.observed_prob:.1%} "
                f"(natural={best_eval.natural_prob:.0%}, edge={best_eval.freq_edge_pct:+.1f}%) "
                f"z={best_eval.z_score:.1f} EV={best_eval.ev:+.1%} "
                f"payout={best_eval.payout_rate:.1%} windows={best_eval.window_agreement}/3 "
                f"trend={trend_label}({align_label}) "
                f"setup_score={setup.setup_score:.2f}"
            )
            setup.detected_at = time.time()
        else:
            setup.reason = (
                f"Best: {dir_label}{best_eval.barrier} obs={best_eval.observed_prob:.1%} "
                f"edge={best_eval.freq_edge_pct:+.1f}% z={best_eval.z_score:.1f} "
                f"EV={best_eval.ev:+.1%} — score={setup.setup_score:.2f} "
                f"{'INSUFFICIENT' if not best_eval.is_significant else 'weak'}"
            )
            self._check_session_break(symbol, setup)
        
        self._setup_cache[symbol] = setup
        return setup
    
    def _evaluate_all_barriers(self, features: dict, trend_regime: int) -> Optional[BarrierEval]:
        """
        Evaluate ALL Over/Under barriers and find the best one.
        
        For each barrier:
        1. Get observed win probability from digit frequencies
        2. Calculate edge (observed - natural)
        3. Calculate z-score (statistical significance)
        4. Estimate payout rate
        5. Calculate EV = observed_prob * (1 + payout_rate) - 1
        6. Check window agreement
        
        Returns the BarrierEval with the best risk-adjusted EV, or None if
        no barrier has a significant edge.
        """
        best_eval: Optional[BarrierEval] = None
        best_score: float = -1.0
        all_evals = []
        
        # ─── Evaluate Over barriers ───
        for barrier in BARRIER_OVER_OPTIONS:
            natural_prob = BARRIER_NATURAL_PROB_OVER.get(barrier, 0.5)
            
            # Skip barriers outside our probability range
            if natural_prob < MIN_BARRIER_PROBABILITY or natural_prob > MAX_BARRIER_PROBABILITY:
                continue
            
            eval_result = self._evaluate_single_barrier(
                contract_type=CONTRACT_TYPE_OVER,
                barrier=barrier,
                natural_prob=natural_prob,
                features=features,
            )
            
            if eval_result is not None:
                all_evals.append(eval_result)
                
                # Score: risk-adjusted EV (EV * significance * window_agreement)
                # Prefer barriers with: high EV, high z-score, multiple window agreement
                if eval_result.is_significant and eval_result.ev > 0:
                    score = eval_result.ev * (1 + eval_result.z_score / 5.0) * (1 + eval_result.window_agreement / 3.0)
                    if score > best_score:
                        best_score = score
                        best_eval = eval_result
        
        # ─── Evaluate Under barriers ───
        for barrier in BARRIER_UNDER_OPTIONS:
            natural_prob = BARRIER_NATURAL_PROB_UNDER.get(barrier, 0.5)
            
            if natural_prob < MIN_BARRIER_PROBABILITY or natural_prob > MAX_BARRIER_PROBABILITY:
                continue
            
            eval_result = self._evaluate_single_barrier(
                contract_type=CONTRACT_TYPE_UNDER,
                barrier=barrier,
                natural_prob=natural_prob,
                features=features,
            )
            
            if eval_result is not None:
                all_evals.append(eval_result)
                
                if eval_result.is_significant and eval_result.ev > 0:
                    score = eval_result.ev * (1 + eval_result.z_score / 5.0) * (1 + eval_result.window_agreement / 3.0)
                    if score > best_score:
                        best_score = score
                        best_eval = eval_result
        
        # Log top 3 barriers (for debugging)
        if all_evals:
            significant = [e for e in all_evals if e.is_significant and e.ev > 0]
            significant.sort(key=lambda e: e.ev, reverse=True)
            if significant:
                top = significant[:3]
                barrier_str = " | ".join(
                    f"{'O' if e.contract_type == CONTRACT_TYPE_OVER else 'U'}{e.barrier} "
                    f"obs={e.observed_prob:.1%} EV={e.ev:+.1%} z={e.z_score:.1f}"
                    for e in top
                )
                logger.debug(f"Top barriers: {barrier_str}")
        
        return best_eval
    
    def _evaluate_single_barrier(self, contract_type: str, barrier: int,
                                  natural_prob: float, features: dict) -> Optional[BarrierEval]:
        """
        Evaluate a single barrier option.
        
        Computes observed win probability, edge, z-score, payout, and EV
        for one specific barrier (e.g., Over 7 or Under 2).
        """
        # Get observed frequency from features
        # The feature_engine computes over{barrier}_freq_{window} and under{barrier}_freq_{window}
        freq_key_prefix = "over" if contract_type == CONTRACT_TYPE_OVER else "under"
        
        observed_probs = []
        for window in ["short", "medium", "trend_long"]:
            key = f"{freq_key_prefix}{barrier}_freq_{window}"
            prob = features.get(key, None)
            if prob is not None:
                observed_probs.append(prob)
        
        if not observed_probs:
            return None
        
        # Weighted average: medium and long windows are more reliable
        if len(observed_probs) == 3:
            observed_prob = observed_probs[0] * 0.2 + observed_probs[1] * 0.4 + observed_probs[2] * 0.4
        elif len(observed_probs) == 2:
            observed_prob = (observed_probs[0] + observed_probs[1]) / 2
        else:
            observed_prob = observed_probs[0]
        
        # Calculate edge
        freq_edge = observed_prob - natural_prob
        freq_edge_pct = freq_edge * 100  # As percentage
        
        # Calculate z-score (statistical significance)
        # SE = sqrt(p * (1-p) / n) where p = natural_prob, n = sample size
        # We use the medium window size as the effective sample size
        n = 200  # medium window size
        se = math.sqrt(natural_prob * (1 - natural_prob) / n) if n > 0 else 1.0
        z_score = freq_edge / se if se > 0 else 0.0
        
        # Is this edge statistically significant?
        # v9: Use BOTH absolute edge AND relative edge thresholds
        # A 3.8% edge on Over 8 (10% natural) = 38% relative edge — very significant!
        # A 3.8% edge on Over 4 (50% natural) = 7.6% relative edge — less significant
        relative_edge = freq_edge / natural_prob if natural_prob > 0 else 0
        is_significant = (
            z_score >= MIN_FREQ_EDGE_ZSCORE and 
            abs(freq_edge) >= MIN_DIGIT_FREQUENCY_EDGE and
            relative_edge >= MIN_DIGIT_FREQUENCY_EDGE_RELATIVE
        )
        
        # Calculate payout rate for this barrier
        payout_rate = estimate_payout_rate(natural_prob)
        
        # Calculate EV = observed_prob * (1 + payout_rate) - 1
        # For a $1 stake: if win, you get $1 + $payout_rate back. If lose, you lose $1.
        ev = observed_prob * (1 + payout_rate) - 1.0
        
        # Window agreement: how many windows show the SAME edge direction?
        window_agreement = 0
        for prob in observed_probs:
            if (prob - natural_prob) * freq_edge > 0:  # Same sign as the average edge
                window_agreement += 1
        
        return BarrierEval(
            contract_type=contract_type,
            barrier=barrier,
            natural_prob=natural_prob,
            observed_prob=observed_prob,
            freq_edge=freq_edge,
            freq_edge_pct=freq_edge_pct,
            z_score=z_score,
            payout_rate=payout_rate,
            ev=ev,
            window_agreement=window_agreement,
            is_significant=is_significant,
        )
    
    def _evaluate_fixed_barriers(self, features: dict, trend_regime: int) -> Optional[BarrierEval]:
        """Fallback: evaluate only Over 4 / Under 5 (legacy mode)."""
        best_eval = None
        best_ev = -1.0
        
        # Over 4
        over4 = self._evaluate_single_barrier(CONTRACT_TYPE_OVER, OVER_BARRIER, 0.50, features)
        if over4 and over4.is_significant and over4.ev > best_ev:
            best_ev = over4.ev
            best_eval = over4
        
        # Under 5
        under5 = self._evaluate_single_barrier(CONTRACT_TYPE_UNDER, UNDER_BARRIER, 0.50, features)
        if under5 and under5.is_significant and under5.ev > best_ev:
            best_ev = under5.ev
            best_eval = under5
        
        return best_eval
    
    def _compute_setup_score(self, barrier_eval: BarrierEval, 
                              trend_aligned: bool, trend_regime: int) -> float:
        """
        Compute setup quality score (0-1).
        
        v9: The score is based on:
        1. EV significance (is the EV high enough to trade?)
        2. Statistical significance (z-score)
        3. Window agreement
        4. Trend alignment bonus/penalty
        
        The score determines IF we should trade this setup.
        Higher = more confidence in the trade.
        """
        # 1. EV score: EV > 10% = 1.0, EV > 5% = 0.7, EV > 2% = 0.4, EV < 0 = 0
        if barrier_eval.ev >= 0.10:
            ev_score = 1.0
        elif barrier_eval.ev >= MIN_EV_FOR_TRADE:
            ev_score = 0.5 + 0.5 * (barrier_eval.ev - MIN_EV_FOR_TRADE) / (0.10 - MIN_EV_FOR_TRADE)
        elif barrier_eval.ev > 0:
            ev_score = 0.2 + 0.3 * barrier_eval.ev / MIN_EV_FOR_TRADE
        else:
            ev_score = 0.0
        
        # 2. Z-score significance: z > 4 = 1.0, z > 2 = 0.7, z > 1 = 0.4, z < 1 = 0.1
        if barrier_eval.z_score >= 4.0:
            z_score_val = 1.0
        elif barrier_eval.z_score >= MIN_FREQ_EDGE_ZSCORE:
            z_score_val = 0.5 + 0.5 * (barrier_eval.z_score - MIN_FREQ_EDGE_ZSCORE) / (4.0 - MIN_FREQ_EDGE_ZSCORE)
        elif barrier_eval.z_score >= 1.0:
            z_score_val = 0.2 + 0.3 * (barrier_eval.z_score - 1.0) / (MIN_FREQ_EDGE_ZSCORE - 1.0)
        else:
            z_score_val = 0.1
        
        # 3. Window agreement: 3/3 = 1.0, 2/3 = 0.7, 1/3 = 0.4, 0/3 = 0.1
        agreement_score = {0: 0.1, 1: 0.4, 2: 0.7, 3: 1.0}.get(barrier_eval.window_agreement, 0.1)
        
        # 4. Trend alignment: bonus or penalty
        if trend_regime == 0:
            trend_factor = 1.0  # No trend — neutral
        elif trend_aligned:
            trend_factor = 1.0 + TREND_CONFIDENCE_BOOST  # Small boost
        else:
            trend_factor = 1.0 - TREND_MISALIGN_PENALTY  # Penalty for misalignment
        
        # Composite score (weighted)
        raw_score = (
            ev_score * 0.40 +           # EV is the most important
            z_score_val * 0.30 +        # Statistical significance
            agreement_score * 0.30       # Window agreement validates
        )
        
        # Apply trend factor
        score = raw_score * trend_factor
        
        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))
        
        return score
    
    def start_observation(self, symbol: str, direction: int, barrier: int = 4) -> ObservationTracker:
        """Start the observation phase for a market."""
        tracker = ObservationTracker()
        tracker.start(direction, barrier)
        self._observation_trackers[symbol] = tracker
        dir_label = "Over" if direction == 1 else "Under"
        logger.info(f"[{symbol}] OBSERVATION PHASE started: {dir_label}{barrier} for {OBSERVATION_PERIOD_SEC}s")
        return tracker
    
    def observe_tick(self, symbol: str, digit: int) -> bool:
        """Process a tick during observation phase. Returns True if complete."""
        tracker = self._observation_trackers.get(symbol)
        if tracker is None:
            return True
        
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
        return 5
    
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
        """Check if a market is still tradable."""
        session = self._sessions.get(symbol)
        if session is None:
            return True
        
        if session.profit_target_reached:
            setup = self._setup_cache.get(symbol)
            if setup and setup.active and setup.setup_score >= 0.75:
                logger.info(
                    f"[{symbol}] New strong setup detected (score={setup.setup_score:.2f}). "
                    f"Resetting session."
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
        return self._sessions.get(symbol)
    
    def reset_session(self, symbol: str):
        self._sessions[symbol] = MarketSession(symbol=symbol)
    
    def get_setup(self, symbol: str) -> Optional[Setup]:
        return self._setup_cache.get(symbol)
