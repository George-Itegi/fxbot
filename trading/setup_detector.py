"""
Setup Detector — Conservative Edge Detection (v10)
====================================================
v9 FAILED: chased extreme barriers, caught noise as "edges."

v10 fixes:
1. BAYESIAN SHRINKAGE: observed prob is blended with natural prob
   - Prevents chasing noise — if n=200, shrink 33% toward natural
   - A 40% observed on Over 8 (10% natural) becomes 30%, not 40%
2. MODERATE BARRIERS ONLY: Over 3-6, Under 4-7
   - These have 30-60% natural prob — manageable variance
   - Even without an edge, you win 30-60% of trades
3. ALL 3 WINDOWS MUST AGREE (was 2/3)
   - Reduces false positives from the multiple testing problem
4. z-score threshold = 3.0 (was 1.3)
   - 3-sigma = 99.7% confidence, not 80%
5. Stricter setup score threshold: 0.70 (was 0.60)
6. Trend misalignment penalty increased to 15% (was 10%)
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
    BAYESIAN_SHRINKAGE_PRIOR,
)
from utils.logger import setup_logger

logger = setup_logger("trading.setup_detector")


@dataclass
class BarrierEval:
    """Evaluation of a single barrier option."""
    contract_type: str          # "DIGITOVER" or "DIGITUNDER"
    barrier: int                # The digit barrier (e.g., 5 for Over 5)
    natural_prob: float         # Natural/uniform probability (e.g., 0.40 for Over 5)
    observed_prob: float        # Raw observed win probability from digit frequencies
    adjusted_prob: float        # v10: Bayesian-shrunk probability (observed blended with natural)
    freq_edge: float            # adjusted_prob - natural_prob (using shrunk prob)
    freq_edge_pct: float        # Edge as percentage
    z_score: float              # Statistical significance of the deviation
    payout_rate: float          # Estimated payout (profit/stake ratio)
    ev: float                   # Expected value per $1 staked (using adjusted_prob)
    window_agreement: int       # How many windows confirm this edge (0-3)
    is_significant: bool        # Is the z-score above threshold?
    sample_size: int            # Effective sample size used for shrinkage


@dataclass
class Setup:
    """Detected market setup — the basis for a trade decision."""
    active: bool = False               # Is there a valid setup right now?
    direction: str = ""                # "DIGITOVER" or "DIGITUNDER"
    barrier: int = 4                   # The BEST barrier to trade
    trend_regime: int = 0              # 1=uptrend, -1=downtrend, 0=ranging
    trend_strength: float = 0.0        # Min t-stat across 200/500 windows
    over_freq: float = 0.5             # Average Over-frequency across windows (for Over 4)
    under_freq: float = 0.5            # Average Under-frequency across windows (for Under 5)
    freq_edge: float = 0.0             # |adjusted_prob - natural_prob| for chosen barrier
    freq_direction: int = 0            # 1=Over dominant, -1=Under dominant
    window_agreement: int = 0          # How many windows agree on freq direction (0-3)
    trend_freq_aligned: bool = False   # Does trend direction match frequency direction?
    setup_score: float = 0.0           # Composite quality score (0-1)
    reason: str = ""                   # Human-readable explanation
    detected_at: float = 0.0           # When was this setup detected?
    observation_complete: bool = True   # v10: Always True (no observation phase)
    observed_duration: int = 5          # v10: Always 5
    # ─── v10: Dynamic barrier fields ───
    best_barrier_eval: Optional[BarrierEval] = None
    natural_prob: float = 0.5          # Natural probability for chosen barrier
    observed_prob: float = 0.5         # Raw observed probability for chosen barrier
    adjusted_prob: float = 0.5         # v10: Bayesian-adjusted probability
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


class SetupDetector:
    """
    Detects and scores market setups — v10 Conservative Edge Detection.
    
    v10 Philosophy:
    - Only test MODERATE barriers (Over 3-6, Under 4-7)
    - Use BAYESIAN SHRINKAGE to prevent chasing noise
    - Require z-score > 3.0 (3-sigma)
    - Require ALL 3 windows to agree
    - Trend is a SOFT BIAS (boost/penalty), not a requirement
    """
    
    def __init__(self):
        self._sessions: dict[str, MarketSession] = {}
        self._setup_cache: dict[str, Setup] = {}
        
        logger.info("SetupDetector v10 initialized: conservative edge detection + Bayesian shrinkage")
    
    def evaluate(self, symbol: str, features: dict) -> Setup:
        """
        Evaluate the current market setup with conservative barrier selection.
        
        This is the MAIN method — called on every tick for each market.
        Returns a Setup object describing the best available trade.
        """
        setup = Setup()
        
        # ─── Step 1: Trend Check (v10: SOFT BIAS, not requirement) ───
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
        
        # ─── Step 2: Evaluate barriers (MODERATE only) ───
        if DYNAMIC_BARRIERS:
            best_eval = self._evaluate_all_barriers(features, trend_regime)
        else:
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
        setup.adjusted_prob = best_eval.adjusted_prob  # v10: Bayesian-adjusted
        setup.payout_rate = best_eval.payout_rate
        setup.z_score = best_eval.z_score
        setup.freq_edge = best_eval.freq_edge
        setup.best_barrier_eval = best_eval
        
        # Legacy fields
        setup.over_freq = features.get("over_freq_medium", 0.5)
        setup.under_freq = features.get("under_freq_medium", 0.5)
        setup.freq_direction = 1 if best_eval.contract_type == CONTRACT_TYPE_OVER else -1
        setup.window_agreement = best_eval.window_agreement
        
        # ─── Step 4: Trend alignment check ───
        if trend_regime != 0:
            if (trend_regime == 1 and best_eval.contract_type == CONTRACT_TYPE_OVER) or \
               (trend_regime == -1 and best_eval.contract_type == CONTRACT_TYPE_UNDER):
                setup.trend_freq_aligned = True
            else:
                setup.trend_freq_aligned = False
        else:
            setup.trend_freq_aligned = True  # Neutral = OK
        
        # ─── Step 5: Setup Quality Score (v10: EV-based with Bayesian adjustment) ───
        setup.setup_score = self._compute_setup_score(best_eval, setup.trend_freq_aligned, trend_regime)
        
        # ─── Step 6: Is the setup active? ───
        # v10: Much stricter — need high z-score, all windows agree, and high setup score
        setup.active = (
            setup.setup_score >= MIN_SETUP_SCORE and 
            best_eval.is_significant and
            best_eval.window_agreement >= DIGIT_FREQ_WINDOW_AGREEMENT  # v10: ALL windows
        )
        
        # Build reason string
        dir_label = "Over" if best_eval.contract_type == CONTRACT_TYPE_OVER else "Under"
        trend_label = "UP" if trend_regime == 1 else "DOWN" if trend_regime == -1 else "RANGING"
        align_label = "ALIGNED" if setup.trend_freq_aligned else "MISALIGNED"
        
        if setup.active:
            setup.reason = (
                f"{dir_label}{best_eval.barrier}: obs={best_eval.observed_prob:.1%} "
                f"adj={best_eval.adjusted_prob:.1%} (natural={best_eval.natural_prob:.0%}, "
                f"edge={best_eval.freq_edge_pct:+.1f}%) "
                f"z={best_eval.z_score:.1f} EV={best_eval.ev:+.1%} "
                f"payout={best_eval.payout_rate:.1%} windows={best_eval.window_agreement}/3 "
                f"trend={trend_label}({align_label}) "
                f"setup_score={setup.setup_score:.2f}"
            )
            setup.detected_at = time.time()
        else:
            setup.reason = (
                f"Best: {dir_label}{best_eval.barrier} obs={best_eval.observed_prob:.1%} "
                f"adj={best_eval.adjusted_prob:.1%} "
                f"edge={best_eval.freq_edge_pct:+.1f}% z={best_eval.z_score:.1f} "
                f"EV={best_eval.ev:+.1%} windows={best_eval.window_agreement}/3 "
                f"— score={setup.setup_score:.2f} "
                f"{'INSUFFICIENT' if not best_eval.is_significant else 'weak'}"
            )
            self._check_session_break(symbol, setup)
        
        self._setup_cache[symbol] = setup
        return setup
    
    def _evaluate_all_barriers(self, features: dict, trend_regime: int) -> Optional[BarrierEval]:
        """
        Evaluate MODERATE Over/Under barriers and find the best one.
        
        v10: Only evaluates Over 3-6 and Under 4-7 (moderate barriers).
        v9 tested ALL 17 barriers which caused massive false positive rate.
        
        Scoring now uses Bayesian-adjusted probability for EV calculation,
        which prevents extreme barriers from dominating via payout amplification.
        """
        best_eval: Optional[BarrierEval] = None
        best_score: float = -1.0
        all_evals = []
        
        # ─── Evaluate Over barriers ───
        for barrier in BARRIER_OVER_OPTIONS:
            natural_prob = BARRIER_NATURAL_PROB_OVER.get(barrier, 0.5)
            
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
                
                # v10: Score based on ADJUSTED EV (Bayesian-shrunk)
                # This prevents extreme barriers from dominating
                if eval_result.is_significant and eval_result.ev > 0:
                    # Score: risk-adjusted EV with window agreement
                    # Prefer barriers with: high adjusted EV, high z-score, all windows agree
                    window_bonus = 1.0 + eval_result.window_agreement / 3.0
                    score = eval_result.ev * window_bonus
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
                    window_bonus = 1.0 + eval_result.window_agreement / 3.0
                    score = eval_result.ev * window_bonus
                    if score > best_score:
                        best_score = score
                        best_eval = eval_result
        
        # Log top barriers (for debugging)
        if all_evals:
            significant = [e for e in all_evals if e.is_significant and e.ev > 0]
            significant.sort(key=lambda e: e.ev, reverse=True)
            if significant:
                top = significant[:3]
                barrier_str = " | ".join(
                    f"{'O' if e.contract_type == CONTRACT_TYPE_OVER else 'U'}{e.barrier} "
                    f"adj={e.adjusted_prob:.1%} EV={e.ev:+.1%} z={e.z_score:.1f}"
                    for e in top
                )
                logger.debug(f"Top barriers: {barrier_str}")
        
        return best_eval
    
    def _evaluate_single_barrier(self, contract_type: str, barrier: int,
                                  natural_prob: float, features: dict) -> Optional[BarrierEval]:
        """
        Evaluate a single barrier option with BAYESIAN SHRINKAGE (v10).
        
        v10 Key Change: observed probability is blended with natural probability
        using Bayesian shrinkage. This prevents the bot from chasing noise.
        
        adjusted_prob = (n * observed + k * natural) / (n + k)
        
        Where n = effective sample size, k = prior strength (BAYESIAN_SHRINKAGE_PRIOR).
        """
        # Get observed frequency from features
        freq_key_prefix = "over" if contract_type == CONTRACT_TYPE_OVER else "under"
        
        observed_probs = []
        sample_sizes = []
        for window in ["short", "medium", "trend_long"]:
            key = f"{freq_key_prefix}{barrier}_freq_{window}"
            prob = features.get(key, None)
            if prob is not None:
                observed_probs.append(prob)
                # Approximate sample size for each window
                if window == "short":
                    sample_sizes.append(50)
                elif window == "medium":
                    sample_sizes.append(200)
                else:
                    sample_sizes.append(500)
        
        if not observed_probs:
            return None
        
        # Weighted average: medium and long windows are more reliable
        if len(observed_probs) == 3:
            observed_prob = observed_probs[0] * 0.2 + observed_probs[1] * 0.4 + observed_probs[2] * 0.4
            effective_n = sample_sizes[0] * 0.2 + sample_sizes[1] * 0.4 + sample_sizes[2] * 0.4
        elif len(observed_probs) == 2:
            observed_prob = (observed_probs[0] + observed_probs[1]) / 2
            effective_n = (sample_sizes[0] + sample_sizes[1]) / 2
        else:
            observed_prob = observed_probs[0]
            effective_n = sample_sizes[0]
        
        # ─── v10: BAYESIAN SHRINKAGE ───
        # Blend observed probability with natural probability
        # adjusted_prob = (n * observed + k * natural) / (n + k)
        k = BAYESIAN_SHRINKAGE_PRIOR  # 100 by default
        adjusted_prob = (effective_n * observed_prob + k * natural_prob) / (effective_n + k)
        
        # Calculate edge using ADJUSTED probability (not raw observed)
        freq_edge = adjusted_prob - natural_prob
        freq_edge_pct = freq_edge * 100
        
        # Calculate z-score using the RAW observed probability
        # (z-score measures how far the raw observation is from natural)
        n = 200  # medium window size for z-score calculation
        se = math.sqrt(natural_prob * (1 - natural_prob) / n) if n > 0 else 1.0
        z_score = (observed_prob - natural_prob) / se if se > 0 else 0.0
        
        # v10: Require ALL 3 windows to agree AND higher thresholds
        relative_edge = freq_edge / natural_prob if natural_prob > 0 else 0
        is_significant = (
            z_score >= MIN_FREQ_EDGE_ZSCORE and 
            abs(freq_edge) >= MIN_DIGIT_FREQUENCY_EDGE and
            relative_edge >= MIN_DIGIT_FREQUENCY_EDGE_RELATIVE
        )
        
        # Calculate payout rate for this barrier
        payout_rate = estimate_payout_rate(natural_prob)
        
        # ─── v10: EV using ADJUSTED probability (not raw) ───
        # This is the KEY fix: EV should use the Bayesian-adjusted probability,
        # not the raw observed probability. This prevents the payout multiplier
        # from amplifying noise on extreme barriers.
        ev = adjusted_prob * (1 + payout_rate) - 1.0
        
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
            adjusted_prob=adjusted_prob,
            freq_edge=freq_edge,
            freq_edge_pct=freq_edge_pct,
            z_score=z_score,
            payout_rate=payout_rate,
            ev=ev,
            window_agreement=window_agreement,
            is_significant=is_significant,
            sample_size=int(effective_n),
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
        
        v10: The score is based on:
        1. EV significance (using Bayesian-adjusted probability)
        2. Statistical significance (z-score)
        3. Window agreement (ALL 3 required for high score)
        4. Trend alignment bonus/penalty
        
        Higher = more confidence in the trade.
        """
        # 1. EV score: EV > 15% = 1.0, EV > 8% = 0.7, EV > 3% = 0.4, EV < 0 = 0
        if barrier_eval.ev >= 0.15:
            ev_score = 1.0
        elif barrier_eval.ev >= MIN_EV_FOR_TRADE:
            ev_score = 0.5 + 0.5 * (barrier_eval.ev - MIN_EV_FOR_TRADE) / (0.15 - MIN_EV_FOR_TRADE)
        elif barrier_eval.ev > 0:
            ev_score = 0.2 + 0.3 * barrier_eval.ev / MIN_EV_FOR_TRADE
        else:
            ev_score = 0.0
        
        # 2. Z-score significance: z > 5 = 1.0, z > 3 = 0.7, z > 2 = 0.4, z < 2 = 0.1
        if barrier_eval.z_score >= 5.0:
            z_score_val = 1.0
        elif barrier_eval.z_score >= MIN_FREQ_EDGE_ZSCORE:
            z_score_val = 0.5 + 0.5 * (barrier_eval.z_score - MIN_FREQ_EDGE_ZSCORE) / (5.0 - MIN_FREQ_EDGE_ZSCORE)
        elif barrier_eval.z_score >= 2.0:
            z_score_val = 0.2 + 0.3 * (barrier_eval.z_score - 2.0) / (MIN_FREQ_EDGE_ZSCORE - 2.0)
        else:
            z_score_val = 0.1
        
        # 3. Window agreement: 3/3 = 1.0, 2/3 = 0.5, 1/3 = 0.2, 0/3 = 0.1
        # v10: Much steeper dropoff — 3/3 is strongly preferred
        agreement_score = {0: 0.1, 1: 0.2, 2: 0.5, 3: 1.0}.get(barrier_eval.window_agreement, 0.1)
        
        # 4. Trend alignment: bonus or penalty
        if trend_regime == 0:
            trend_factor = 1.0  # No trend — neutral
        elif trend_aligned:
            trend_factor = 1.0 + TREND_CONFIDENCE_BOOST  # Small boost
        else:
            trend_factor = 1.0 - TREND_MISALIGN_PENALTY  # v10: Larger penalty (15%)
        
        # Composite score (weighted)
        raw_score = (
            ev_score * 0.35 +           # EV is important
            z_score_val * 0.35 +        # Statistical significance equally important
            agreement_score * 0.30       # Window agreement validates
        )
        
        # Apply trend factor
        score = raw_score * trend_factor
        
        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))
        
        return score
    
    # ─── Observation Phase (v10: DISABLED — fixed 5t duration) ───
    # These methods are kept for API compatibility but are no-ops.
    
    def start_observation(self, symbol: str, direction: int, barrier: int = 4):
        """v10: No-op — observation phase disabled."""
        pass
    
    def observe_tick(self, symbol: str, digit: int) -> bool:
        """v10: Always returns True — observation phase disabled."""
        return True
    
    def get_observed_duration(self, symbol: str) -> int:
        """v10: Always returns 5 — fixed duration."""
        return 5
    
    def is_observing(self, symbol: str) -> bool:
        """v10: Always returns False — no observation phase."""
        return False
    
    def clear_observation(self, symbol: str):
        """v10: No-op — observation phase disabled."""
        pass
    
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
            if setup and setup.active and setup.setup_score >= 0.80:
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
