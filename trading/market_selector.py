"""
Market Selector — Setup-Based Quality Selection (v8)
=====================================================
Picks the BEST market to trade based on SETUP quality, not just model signals.

v8 Changes:
- Setup score is the primary ranking metric
- Market persistence: stay on one market during a setup
- Only switch when: setup breaks, profit target reached, or no trade happening
- Simpler scoring: setup_score * frequency_edge * recent_win_rate
"""

import time
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from config import (META_SELECTOR_EXPLORATION_RATE, META_SELECTOR_MIN_TRADES,
                    META_SELECTOR_SCORE_WINDOW, MARKET_SESSION_MAX_IDLE_SEC,
                    PROFIT_TARGET_PER_MARKET)
from trading.signal_generator import Signal
from utils.logger import setup_logger

logger = setup_logger("trading.market_selector")


@dataclass
class MarketStats:
    symbol: str
    wins: int = 0
    losses: int = 0
    total_trades: int = 0
    recent_results: deque = field(default_factory=lambda: deque(maxlen=META_SELECTOR_SCORE_WINDOW))
    total_pnl: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_trades if self.total_trades > 0 else 0.5

    @property
    def recent_win_rate(self) -> float:
        if not self.recent_results:
            return 0.5
        wins = sum(1 for r in self.recent_results if r)
        return wins / len(self.recent_results)


class MarketSelector:
    """
    Selects the best market to trade based on SETUP quality.
    
    v8 Philosophy:
    - Stay on one market during a good setup (persistence)
    - Only switch when the setup breaks or profit target is reached
    - When choosing a new market, pick the one with the best setup score
    - Quality over quantity — a 0.85 setup on one market beats a 0.60 on another
    """

    def __init__(self, epsilon: float = None):
        # v11: epsilon set to 0 — no random exploration
        self.epsilon = 0.0  # NO random market switching
        self._stats: dict[str, MarketStats] = {}
        self._total_selections = 0
        self._exploration_count = 0
        self._last_selected_market: Optional[str] = None
        self._last_trade_time: float = 0.0
        
        logger.info(f"MarketSelector initialized: setup-based quality selection")

    def select_market(self, workers: dict) -> Optional[str]:
        """
        Select the best market to trade.
        
        Persistence logic:
        - If we recently traded a market and it still has a good setup, stay on it
        - Only switch when the current market's setup breaks or profit target is reached
        """
        candidates = {}
        for symbol, worker in workers.items():
            signal = worker.get_fresh_signal()
            if signal is not None:
                # Check if market is tradable (not at profit target, setup not broken)
                if worker.setup_detector.is_market_tradable(symbol):
                    candidates[symbol] = (worker, signal)

        if not candidates:
            return None

        # ─── PERSISTENCE: Stay on the current market if setup is still good ───
        if self._last_selected_market in candidates:
            worker, signal = candidates[self._last_selected_market]
            
            # Check if enough time has passed without a trade (idle timeout)
            idle_time = time.time() - self._last_trade_time
            session = worker.setup_detector.get_session(self._last_selected_market)
            
            # If the session has trades and hasn't hit profit target, stay
            if session and session.session_trades > 0 and not session.profit_target_reached:
                if idle_time < MARKET_SESSION_MAX_IDLE_SEC:
                    # Stay on this market — its setup is still valid
                    return self._last_selected_market

        if len(candidates) == 1:
            sym = list(candidates.keys())[0]
            worker, signal = candidates[sym]
            self._log_selection(sym, signal, worker, score=0.0, reason="only_candidate")
            return sym

        # ─── SELECT BEST SETUP ───
        self._total_selections += 1
        decayed_epsilon = max(0.02, self.epsilon * (0.995 ** self._total_selections))

        # Exploration: occasionally try a different market
        if random.random() < decayed_epsilon:
            selected = random.choice(list(candidates.keys()))
            self._exploration_count += 1
            worker, signal = candidates[selected]
            self._log_selection(selected, signal, worker, score=0.0, reason="exploring")
            if selected != self._last_selected_market:
                logger.info(f"Market SWITCH: {self._last_selected_market} -> {selected} (exploring)")
            self._last_selected_market = selected
            return selected

        # Exploitation: pick the market with the best setup score
        best_symbol = None
        best_score = -1.0
        best_breakdown = {}

        for symbol, (worker, signal) in candidates.items():
            score, breakdown = self._compute_score(symbol, worker, signal)
            if score > best_score:
                best_score = score
                best_symbol = symbol
                best_breakdown = breakdown

        if best_symbol != self._last_selected_market:
            if self._last_selected_market is not None:
                logger.info(
                    f"Market SWITCH: {self._last_selected_market} -> {best_symbol} "
                    f"(setup_score={best_score:.2f})"
                )
            self._last_selected_market = best_symbol

        worker, signal = candidates[best_symbol]
        self._log_selection(best_symbol, signal, worker, score=best_score,
                           reason="best_setup", breakdown=best_breakdown)

        return best_symbol

    def _compute_score(self, symbol: str, worker, signal: Signal) -> tuple:
        """
        Score a market based on SETUP quality (not model agreement).
        
        v8 scoring:
        1. Setup score (0-1) — the primary quality metric
        2. Frequency edge — how far from 50/50 digit distribution
        3. Recent win rate on this market
        4. ML confirmation (bonus if ML agrees, penalty if it disagrees)
        """
        # 1. Setup score multiplier
        setup_score = signal.setup_score
        if setup_score >= 0.85:
            setup_mult = 2.5
        elif setup_score >= 0.70:
            setup_mult = 1.5 + (setup_score - 0.70) / 0.15 * 1.0
        elif setup_score >= 0.60:
            setup_mult = 1.0 + (setup_score - 0.60) / 0.10 * 0.5
        else:
            setup_mult = 0.5

        # 2. Frequency edge multiplier
        # Higher edge = more confident in direction
        over_freq = worker._current_setup.over_freq if worker._current_setup else 0.5
        under_freq = worker._current_setup.under_freq if worker._current_setup else 0.5
        freq_edge = abs(over_freq - under_freq)
        if freq_edge >= 0.10:
            freq_mult = 2.0
        elif freq_edge >= 0.05:
            freq_mult = 1.0 + freq_edge * 10
        else:
            freq_mult = 0.5 + freq_edge * 10

        # 3. Recent win rate multiplier
        stats = self._stats.get(symbol)
        if stats and stats.total_trades >= 3:
            wr = stats.recent_win_rate
            if wr >= 0.65:
                wr_mult = 1.5
            elif wr >= 0.50:
                wr_mult = 1.0
            else:
                wr_mult = 0.5
        else:
            wr_mult = 1.0

        # 4. ML confirmation bonus
        # If model_agreement is 1.0 (single model agrees), small bonus
        ml_mult = 1.0 if signal.confidence >= 0.55 else 0.8

        # Combined score
        score = setup_mult * freq_mult * wr_mult * ml_mult

        breakdown = {
            "setup": round(setup_mult, 2),
            "freq_edge": round(freq_mult, 2),
            "win_rate": round(wr_mult, 2),
            "ml": round(ml_mult, 2),
            "total": round(score, 2),
        }

        return score, breakdown

    def _log_selection(self, symbol: str, signal: Signal, worker,
                       score: float, reason: str, breakdown: dict = None):
        """Log why a market was selected."""
        setup = worker._current_setup
        setup_info = f"setup={setup.setup_score:.2f}" if setup else "setup=?"

        if breakdown:
            logger.info(
                f"SELECTED {symbol}: score={score:.2f} "
                f"[setup={breakdown['setup']}x freq={breakdown['freq_edge']}x "
                f"wr={breakdown['win_rate']}x ml={breakdown['ml']}x] "
                f"dir={signal.direction.replace('DIGIT','')} "
                f"conf={signal.confidence:.0%} {setup_info} "
                f"({reason})"
            )
        else:
            logger.info(
                f"SELECTED {symbol}: {signal.direction.replace('DIGIT','')} "
                f"conf={signal.confidence:.0%} {setup_info} ({reason})"
            )

    def record_outcome(self, symbol: str, won: bool, pnl: float):
        if symbol not in self._stats:
            self._stats[symbol] = MarketStats(symbol=symbol)

        stats = self._stats[symbol]
        stats.total_trades += 1
        stats.recent_results.append(won)
        stats.total_pnl += pnl
        self._last_trade_time = time.time()

        if won:
            stats.wins += 1
        else:
            stats.losses += 1

    def summary(self) -> dict:
        market_stats = {}
        for symbol, stats in self._stats.items():
            market_stats[symbol] = {
                "win_rate": round(stats.recent_win_rate, 3),
                "total_trades": stats.total_trades,
                "pnl": round(stats.total_pnl, 2),
            }

        return {
            "total_selections": self._total_selections,
            "exploration_count": self._exploration_count,
            "epsilon": round(self.epsilon, 3),
            "last_selected": self._last_selected_market,
            "markets": market_stats,
        }
