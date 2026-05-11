"""
Market Selector — Meta-Selector Bandit
========================================
Picks the best market to trade at any given moment using epsilon-greedy bandit.
"""

import time
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from config import (META_SELECTOR_EXPLORATION_RATE, META_SELECTOR_MIN_TRADES,
                    META_SELECTOR_SCORE_WINDOW)
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

    @property
    def confidence(self) -> float:
        if self.total_trades < META_SELECTOR_MIN_TRADES:
            return self.total_trades / META_SELECTOR_MIN_TRADES * 0.5
        return min(1.0, 0.5 + 0.5 * (META_SELECTOR_MIN_TRADES / max(self.total_trades, 1)))


class MarketSelector:
    """
    Selects the best market to trade using bandit-based scoring.
    """

    def __init__(self, epsilon: float = None):
        self.epsilon = epsilon or META_SELECTOR_EXPLORATION_RATE
        self._stats: dict[str, MarketStats] = {}
        self._total_selections = 0
        self._exploration_count = 0
        self._last_selected_market: Optional[str] = None

        logger.info(f"MarketSelector initialized: epsilon={self.epsilon:.0%}")

    def select_market(self, workers: dict) -> Optional[str]:
        candidates = {}
        for symbol, worker in workers.items():
            signal = worker.get_fresh_signal()
            if signal is not None:
                candidates[symbol] = worker

        if not candidates:
            return None

        if len(candidates) == 1:
            return list(candidates.keys())[0]

        self._total_selections += 1
        decayed_epsilon = max(0.03, self.epsilon * (0.998 ** self._total_selections))

        if random.random() < decayed_epsilon:
            selected = random.choice(list(candidates.keys()))
            self._exploration_count += 1
            logger.debug(f"Market EXPLORING: {selected} (epsilon={decayed_epsilon:.3f})")
            return selected

        best_symbol = None
        best_score = -1.0

        for symbol, worker in candidates.items():
            score = self._compute_score(symbol, worker)
            if score > best_score:
                best_score = score
                best_symbol = symbol

        if best_symbol != self._last_selected_market:
            if self._last_selected_market is not None:
                logger.info(f"Market SWITCH: {self._last_selected_market} -> {best_symbol} (score={best_score:.4f})")
            self._last_selected_market = best_symbol

        return best_symbol

    def _compute_score(self, symbol: str, worker) -> float:
        signal_score = worker.get_signal_score()
        stats = self._stats.get(symbol)
        if stats is None:
            market_factor = 0.5
        else:
            market_factor = stats.recent_win_rate * stats.confidence
        return signal_score * market_factor

    def record_outcome(self, symbol: str, won: bool, pnl: float):
        if symbol not in self._stats:
            self._stats[symbol] = MarketStats(symbol=symbol)

        stats = self._stats[symbol]
        stats.total_trades += 1
        stats.recent_results.append(won)
        stats.total_pnl += pnl

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
                "confidence": round(stats.confidence, 3),
                "pnl": round(stats.total_pnl, 2),
            }

        return {
            "total_selections": self._total_selections,
            "exploration_count": self._exploration_count,
            "epsilon": round(self.epsilon, 3),
            "last_selected": self._last_selected_market,
            "markets": market_stats,
        }
