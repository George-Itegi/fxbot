"""
Market Selector — Aggressive Quality Filter
=============================================
Picks the BEST market to trade by heavily weighting:
1. Model agreement (100% = massive boost, <100% = big penalty)
2. Confidence (higher = much better)
3. Expected value (must be strongly positive)
4. Recent profitability (markets that are winning get priority)
5. Model accuracy (proven models get priority)

The key insight: with 13 markets generating signals, we can AFFORD
to be very picky. Only trade when the signal is STRONG.
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
    Selects the best market to trade using aggressive quality scoring.

    Scoring weights:
    - AGREEMENT BONUS: 100% agreement = 3.0x, 67% = 1.0x, below = 0.3x
    - CONFIDENCE: Scales from 0.5x at 60% to 2.0x at 90%+
    - EV: Scales from 0.5x at 0.01 to 2.0x at 0.50+
    - RECENT WIN RATE: 80%+ wr = 2.0x, 50% = 1.0x, <40% = 0.3x
    - MODEL ACCURACY: Scales from 0.5x at 50% to 1.5x at 65%+

    Exploration drops to near-zero quickly — we want EXPLOITATION
    when we have 13 markets to choose from.
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
                candidates[symbol] = (worker, signal)

        if not candidates:
            return None

        if len(candidates) == 1:
            sym = list(candidates.keys())[0]
            worker, signal = candidates[sym]
            self._log_selection(sym, signal, worker, score=0.0, reason="only_candidate")
            return sym

        self._total_selections += 1
        # Decay exploration fast — with 13 markets, exploitation pays off
        decayed_epsilon = max(0.02, self.epsilon * (0.995 ** self._total_selections))

        if random.random() < decayed_epsilon:
            selected = random.choice(list(candidates.keys()))
            self._exploration_count += 1
            worker, signal = candidates[selected]
            self._log_selection(selected, signal, worker, score=0.0, reason="exploring")
            return selected

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
                logger.info(f"Market SWITCH: {self._last_selected_market} -> {best_symbol} (score={best_score:.2f})")
            self._last_selected_market = best_symbol

        # Log why this market was picked
        worker, signal = candidates[best_symbol]
        self._log_selection(best_symbol, signal, worker, score=best_score,
                           reason="best_score", breakdown=best_breakdown)

        return best_symbol

    def _compute_score(self, symbol: str, worker, signal: Signal) -> tuple:
        """
        Aggressive quality scoring — heavily rewards strong signals.

        Returns (score, breakdown_dict) for logging.
        """
        # ─── 1. AGREEMENT MULTIPLIER (the BIGGEST factor) ───
        # 100% agreement = 3.0x (all 3 models agree — STRONGEST signal)
        # 67% agreement = 1.0x (2 of 3 — acceptable)
        # Below 67% = 0.3x (weak — probably shouldn't trade)
        if signal.model_agreement >= 1.0:
            agreement_mult = 3.0
        elif signal.model_agreement >= 0.67:
            # Scale from 1.0 at 67% to 3.0 at 100%
            agreement_mult = 1.0 + 2.0 * ((signal.model_agreement - 0.67) / 0.33)
        else:
            agreement_mult = 0.3

        # ─── 2. CONFIDENCE MULTIPLIER ───
        # 60% confidence = 0.5x (barely above threshold)
        # 75% confidence = 1.2x (good)
        # 90%+ confidence = 2.0x (very strong)
        if signal.confidence >= 0.90:
            confidence_mult = 2.0
        elif signal.confidence >= 0.60:
            confidence_mult = 0.5 + 1.5 * ((signal.confidence - 0.60) / 0.30)
        else:
            confidence_mult = 0.5

        # ─── 3. EV MULTIPLIER ───
        # EV > 0.50 = 2.0x (huge edge)
        # EV ~ 0.20 = 1.2x (decent edge)
        # EV ~ 0.01 = 0.5x (barely positive)
        if signal.expected_value >= 0.50:
            ev_mult = 2.0
        elif signal.expected_value >= 0.01:
            ev_mult = 0.5 + 1.5 * ((signal.expected_value - 0.01) / 0.49)
        else:
            ev_mult = 0.5

        # ─── 4. RECENT WIN RATE MULTIPLIER ───
        # Markets that are WINNING get priority, losing ones get penalized
        stats = self._stats.get(symbol)
        if stats and stats.total_trades >= 3:
            wr = stats.recent_win_rate
            if wr >= 0.80:
                wr_mult = 2.0
            elif wr >= 0.55:
                wr_mult = 1.0 + 1.0 * ((wr - 0.55) / 0.25)
            elif wr >= 0.40:
                wr_mult = 0.5 + 0.5 * ((wr - 0.40) / 0.15)
            else:
                wr_mult = 0.3  # Losing market — strong penalty
        else:
            wr_mult = 1.0  # No data — neutral

        # ─── 5. MODEL ACCURACY MULTIPLIER ───
        model_acc = worker.model.stats.accuracy if worker.model.stats.total_updates > 0 else 50.0
        if model_acc >= 65.0:
            acc_mult = 1.5
        elif model_acc >= 55.0:
            acc_mult = 1.0 + 0.5 * ((model_acc - 55.0) / 10.0)
        else:
            acc_mult = max(0.5, model_acc / 55.0)

        # ─── COMBINED SCORE ───
        score = agreement_mult * confidence_mult * ev_mult * wr_mult * acc_mult

        breakdown = {
            "agreement": round(agreement_mult, 2),
            "confidence": round(confidence_mult, 2),
            "ev": round(ev_mult, 2),
            "win_rate": round(wr_mult, 2),
            "accuracy": round(acc_mult, 2),
            "total": round(score, 2),
        }

        return score, breakdown

    def _log_selection(self, symbol: str, signal: Signal, worker, 
                       score: float, reason: str, breakdown: dict = None):
        """Log why a market was selected."""
        stats = self._stats.get(symbol)
        wr_str = f"wr={stats.recent_win_rate:.0%}" if stats and stats.total_trades >= 3 else "wr=?"

        if breakdown:
            logger.info(
                f"SELECTED {symbol}: score={score:.2f} "
                f"[agree={breakdown['agreement']}x conf={breakdown['confidence']}x "
                f"ev={breakdown['ev']}x {wr_str}x acc={breakdown['accuracy']}x] "
                f"dir={signal.direction.replace('DIGIT','')} "
                f"prob={signal.confidence:.0%} EV={signal.expected_value:+.3f} "
                f"agree={signal.model_agreement:.0%} "
                f"({reason})"
            )
        else:
            logger.info(
                f"SELECTED {symbol}: {signal.direction.replace('DIGIT','')} "
                f"prob={signal.confidence:.0%} EV={signal.expected_value:+.3f} "
                f"agree={signal.model_agreement:.0%} ({reason})"
            )

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
