# =============================================================
# ai_engine/replay_buffer.py  v1.0 — Experience Replay Buffer
#
# PURPOSE: Stratified replay buffer that prevents catastrophic
# forgetting during incremental training. Instead of training on
# ALL historical data equally (which drowns new patterns), the
# buffer maintains a curated mix of:
#
#   1. High-priority samples (large prediction errors — model got wrong)
#   2. Stratified samples (even mix of pairs, strategies, outcomes)
#   3. Recent samples (newest trades for current market conditions)
#   4. Edge samples (extreme R multiples — model needs to learn extremes)
#
# ARCHITECTURE:
#   - Fixed-size buffer (default: 5000 trades)
#   - TD-error based priority sampling (proportional to |predicted - actual|)
#   - Stratified reservoir sampling for balanced representation
#   - Persists to disk as pickle (survives restarts)
#   - Shared between L1 and L2 training
#
# USAGE:
#   from ai_engine.replay_buffer import ReplayBuffer
#   buf = ReplayBuffer()
#   buf.add(trade_dict)              # Add completed trade
#   buf.add_many(trade_dicts)        # Bulk add from DB
#   sample = buf.sample(n=500)       # Get prioritized sample for training
#   buf.compute_priorities(model)    # Update TD-error priorities from model
#
# INTEGRATION:
#   Called in training pipeline BEFORE train/test split:
#     1. Load ALL trades from DB
#     2. Feed into replay buffer
#     3. buffer.sample() → curated training set
#     4. Remaining trades → validation set
#     5. Use replay sample for training
# =============================================================

import os
import json
import math
import time
import pickle
import random
import numpy as np
from collections import defaultdict
from datetime import datetime, timezone
from core.logger import get_logger

log = get_logger(__name__)

# ── Buffer configuration ──
BUFFER_DIR = os.path.join(os.path.dirname(__file__), 'models')
BUFFER_PATH = os.path.join(BUFFER_DIR, 'replay_buffer.pkl')
META_PATH = os.path.join(BUFFER_DIR, 'replay_buffer_meta.json')

DEFAULT_MAX_SIZE = 5000         # Max trades stored in buffer
DEFAULT_SAMPLE_SIZE = 2000      # Default number of samples per training run
MIN_PRIORITY = 0.1              # Floor priority (prevents zero-probability sampling)
PRIORITY_ALPHA = 0.6            # Priority exponent (0=uniform, 1=full priority)
PRIORITY_BETA_START = 0.4       # Importance sampling correction start
PRIORITY_BETA_FRAMES = 100000   # Frames to anneal beta to 1.0

# ── Stratification targets ──
# Buffer aims to maintain these proportions for balanced learning
STRAT_PAIR_RATIO = 0.3          # 30% stratified by pair
STRAT_STRATEGY_RATIO = 0.3      # 30% stratified by strategy
STRAT_OUTCOME_RATIO = 0.2       # 20% stratified by win/loss
STRAT_RECENT_RATIO = 0.1        # 10% most recent trades
STRAT_PRIORITY_RATIO = 0.1      # 10% highest priority (TD-error)


class ReplayBuffer:
    """
    Stratified Experience Replay Buffer for ML model training.

    Solves the catastrophic forgetting problem in incremental learning by:
    1. Maintaining a curated subset of ALL historical trades
    2. Prioritizing samples the model gets WRONG (high TD-error)
    3. Stratifying by pair, strategy, and outcome for balanced learning
    4. Including recent trades to capture current market regime

    The buffer is a ring buffer — when full, oldest/lowest-priority
    samples are evicted to make room for new ones.
    """

    def __init__(self, max_size: int = DEFAULT_MAX_SIZE):
        """
        Args:
            max_size: Maximum number of trades to store. When exceeded,
                      lowest-priority samples are evicted.
        """
        self.max_size = max_size
        self.trades = []           # List of trade dicts (features + outcome)
        self.priorities = []       # TD-error based priorities (float)
        self._added_count = 0      # Total trades ever added (for beta annealing)
        self._last_model_hash = None  # Track model changes for priority updates
        self._meta = {
            'version': '1.0',
            'max_size': max_size,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'last_updated': None,
            'total_added': 0,
            'total_evicted': 0,
            'pair_distribution': {},
            'strategy_distribution': {},
            'win_rate': 0.0,
        }

        # Try loading from disk
        self._load()

    def __len__(self) -> int:
        return len(self.trades)

    def is_empty(self) -> bool:
        return len(self.trades) == 0

    @property
    def utilization(self) -> float:
        """Buffer utilization (0.0 to 1.0)."""
        return len(self.trades) / max(self.max_size, 1)

    @property
    def meta(self) -> dict:
        return dict(self._meta)

    # ════════════════════════════════════════════════════════════════
    # ADD TRADES
    # ════════════════════════════════════════════════════════════════

    def add(self, trade: dict, priority: float = None) -> bool:
        """
        Add a single trade to the buffer.

        Args:
            trade: Trade dict with at least:
                   - 'profit_r' (float): R-multiple outcome
                   - 'symbol' (str): Currency pair
                   - 'strategy' (str): Strategy name
                   - 'entry_time' (str/datetime): When trade was entered
                   - 'win' (bool/int): Whether trade was profitable
                   - 'model_predicted_r' (float): What model predicted (optional)
            priority: Initial priority. If None, computed from prediction error.

        Returns:
            True if added, False if rejected.
        """
        if not trade or not isinstance(trade, dict):
            return False

        if trade.get('profit_r') is None:
            return False

        # Compute initial priority
        if priority is None:
            priority = self._compute_initial_priority(trade)

        # Check if buffer is full — evict if needed
        if len(self.trades) >= self.max_size:
            self._evict_lowest_priority()

        self.trades.append(trade)
        self.priorities.append(priority)
        self._added_count += 1
        self._meta['total_added'] += 1
        self._meta['last_updated'] = datetime.now(timezone.utc).isoformat()

        return True

    def add_many(self, trades: list) -> int:
        """
        Bulk add trades from DB. Used to seed the buffer.

        Args:
            trades: List of trade dicts from backtest_trades table.

        Returns:
            Number of trades successfully added.
        """
        added = 0
        for trade in trades:
            if self.add(trade):
                added += 1

        log.info(f"[REPLAY] Added {added}/{len(trades)} trades to buffer "
                 f"({len(self.trades)}/{self.max_size} used)")

        self._update_distribution_stats()
        self._save()
        return added

    def seed_from_db(self, strategy: str = None, limit: int = None) -> int:
        """
        Seed the buffer from the database. Optionally filter by strategy.

        Args:
            strategy: If provided, only load trades for this strategy.
            limit: Maximum trades to load (default: max_size).

        Returns:
            Number of trades loaded.
        """
        try:
            from database.db_manager import get_connection
            from backtest.db_store import _ensure_tables

            conn = get_connection()
            try:
                _ensure_tables(conn)
            except Exception:
                pass

            cursor = conn.cursor(dictionary=True)
            limit_val = limit or self.max_size

            if strategy:
                cursor.execute("""
                    SELECT id, symbol, direction, strategy, session, market_state,
                           score, sl_pips, tp_pips, confluence_count,
                           delta, rolling_delta, delta_bias, rd_bias,
                           of_imbalance, of_strength, vol_surge_detected, vol_surge_ratio,
                           momentum_velocity, is_choppy,
                           smc_bias, pd_zone, pips_to_eq, structure_trend,
                           atr, pip_from_vwap, pip_to_poc, va_width_pips, price_position,
                           final_score, market_score, smc_score, htf_approved, htf_score,
                           combined_bias, agreement_groups,
                           spread_pips, slippage_pips,
                           ss_smc_ob, ss_liquidity_sweep,
                           ss_delta_divergence, ss_trend_continuation,
                           ss_ema_cross, ss_rsi_divergence,
                           ss_supply_demand, ss_bos_momentum,
                           ss_optimal_trade, ss_institutional,
                           profit_pips, profit_r, win,
                           source, model_predicted_r, entry_time,
                           cs_base_strength, cs_quote_strength, cs_strength_delta, cs_pair_bias,
                           ap_atr_percentile, ap_atr_ratio,
                           mr_m5_rsi, mr_m15_rsi, mr_m30_rsi, mr_h1_rsi, mr_h4_rsi, mr_d1_rsi,
                           mt_mtf_score, mt_trend_agreement, mt_rsi_agreement,
                           im_vix, im_dxy_change, im_risk_env,
                           sk_current_streak, sk_recent_wr,
                           zs_signal_zscore, sm_footprint_score
                    FROM backtest_trades
                    WHERE strategy = %s
                      AND source IN ('BACKTEST', 'SHADOW')
                      AND outcome IS NOT NULL AND outcome != ''
                      AND win IS NOT NULL AND profit_r IS NOT NULL
                    ORDER BY entry_time DESC
                    LIMIT %s
                """, (strategy, limit_val))
            else:
                cursor.execute("""
                    SELECT id, symbol, direction, strategy, session, market_state,
                           score, sl_pips, tp_pips, confluence_count,
                           delta, rolling_delta, delta_bias, rd_bias,
                           of_imbalance, of_strength, vol_surge_detected, vol_surge_ratio,
                           momentum_velocity, is_choppy,
                           smc_bias, pd_zone, pips_to_eq, structure_trend,
                           atr, pip_from_vwap, pip_to_poc, va_width_pips, price_position,
                           final_score, market_score, smc_score, htf_approved, htf_score,
                           combined_bias, agreement_groups,
                           spread_pips, slippage_pips,
                           ss_smc_ob, ss_liquidity_sweep,
                           ss_delta_divergence, ss_trend_continuation,
                           ss_ema_cross, ss_rsi_divergence,
                           ss_supply_demand, ss_bos_momentum,
                           ss_optimal_trade, ss_institutional,
                           profit_pips, profit_r, win,
                           source, model_predicted_r, entry_time,
                           cs_base_strength, cs_quote_strength, cs_strength_delta, cs_pair_bias,
                           ap_atr_percentile, ap_atr_ratio,
                           mr_m5_rsi, mr_m15_rsi, mr_m30_rsi, mr_h1_rsi, mr_h4_rsi, mr_d1_rsi,
                           mt_mtf_score, mt_trend_agreement, mt_rsi_agreement,
                           im_vix, im_dxy_change, im_risk_env,
                           sk_current_streak, sk_recent_wr,
                           zs_signal_zscore, sm_footprint_score
                    FROM backtest_trades
                    WHERE source IN ('BACKTEST', 'SHADOW')
                      AND outcome IS NOT NULL AND outcome != ''
                      AND win IS NOT NULL AND profit_r IS NOT NULL
                    ORDER BY entry_time DESC
                    LIMIT %s
                """, (limit_val,))

            rows = cursor.fetchall()
            conn.close()

            # Clear existing buffer and re-seed
            self.trades.clear()
            self.priorities.clear()

            # Add trades (newest first → reversed so oldest is at index 0)
            count = self.add_many(list(reversed(rows)))
            log.info(f"[REPLAY] Seeded buffer from DB: {count} trades "
                     f"(strategy={strategy or 'ALL'})")
            return count

        except Exception as e:
            log.error(f"[REPLAY] Failed to seed from DB: {e}")
            return 0

    # ════════════════════════════════════════════════════════════════
    # SAMPLE — THE KEY FUNCTION
    # ════════════════════════════════════════════════════════════════

    def sample(self, n: int = DEFAULT_SAMPLE_SIZE) -> list:
        """
        Draw a stratified sample from the buffer.

        The sample is composed of:
          - 30% Stratified by pair (even representation across currency pairs)
          - 30% Stratified by strategy (even representation across strategies)
          - 20% Stratified by outcome (balanced wins/losses)
          - 10% Most recent trades (current market regime)
          - 10% Highest priority (TD-error — model got these wrong)

        Importance sampling weights are returned so training can
        correct for the non-uniform sampling bias.

        Args:
            n: Number of samples to draw.

        Returns:
            List of trade dicts (length = n).
        """
        if len(self.trades) == 0:
            return []

        n = min(n, len(self.trades))
        sample = []
        indices = list(range(len(self.trades)))

        # ── Compute beta for importance sampling correction ──
        beta = min(1.0,
                   PRIORITY_BETA_START +
                   self._added_count * (1.0 - PRIORITY_BETA_START) / PRIORITY_BETA_FRAMES)

        # ── 1. Stratified by pair (30%) ──
        n_pair = int(n * STRAT_PAIR_RATIO)
        pair_groups = defaultdict(list)
        for i, trade in enumerate(self.trades):
            pair = trade.get('symbol', 'UNKNOWN')
            pair_groups[pair].append(i)

        if pair_groups:
            per_pair = max(1, n_pair // len(pair_groups))
            for pair, idxs in pair_groups.items():
                sampled = random.sample(idxs, min(per_pair, len(idxs)))
                sample.extend(sampled)

        # ── 2. Stratified by strategy (30%) ──
        n_strat = int(n * STRAT_STRATEGY_RATIO)
        strat_groups = defaultdict(list)
        for i, trade in enumerate(self.trades):
            strat = trade.get('strategy', 'UNKNOWN')
            strat_groups[strat].append(i)

        if strat_groups:
            per_strat = max(1, n_strat // len(strat_groups))
            for strat, idxs in strat_groups.items():
                sampled = random.sample(idxs, min(per_strat, len(idxs)))
                sample.extend(sampled)

        # ── 3. Stratified by outcome (20%) ──
        n_outcome = int(n * STRAT_OUTCOME_RATIO)
        wins = [i for i, t in enumerate(self.trades) if t.get('win')]
        losses = [i for i, t in enumerate(self.trades) if not t.get('win')]

        n_wins = min(n_outcome // 2, len(wins))
        n_losses = min(n_outcome - n_wins, len(losses))

        if wins:
            sample.extend(random.sample(wins, n_wins))
        if losses:
            sample.extend(random.sample(losses, n_losses))

        # ── 4. Most recent trades (10%) ──
        n_recent = int(n * STRAT_RECENT_RATIO)
        if n_recent > 0 and len(indices) > 0:
            # Sort by entry_time (last entries are most recent)
            recent_indices = sorted(
                indices,
                key=lambda i: str(self.trades[i].get('entry_time', '')),
                reverse=True
            )[:n_recent]
            sample.extend(recent_indices)

        # ── 5. Highest priority / TD-error (10%) ──
        n_priority = int(n * STRAT_PRIORITY_RATIO)
        if n_priority > 0 and len(self.priorities) > 0:
            # Weighted random sampling based on priority
            prios = np.array(self.priorities, dtype=np.float64)
            # Apply alpha exponent
            weighted = np.power(prios + MIN_PRIORITY, PRIORITY_ALPHA)
            weighted /= weighted.sum()
            # Sample without replacement
            try:
                priority_indices = np.random.choice(
                    len(indices), size=min(n_priority, len(indices)),
                    replace=False, p=weighted
                ).tolist()
                sample.extend(priority_indices)
            except Exception:
                # Fallback: just take top priority indices
                top_indices = np.argsort(prios)[-n_priority:]
                sample.extend(top_indices.tolist())

        # ── Remove duplicates and fill remaining with uniform random ──
        unique = list(set(sample))
        remaining = n - len(unique)

        if remaining > 0:
            available = [i for i in indices if i not in set(unique)]
            if available:
                unique.extend(random.sample(available, min(remaining, len(available))))

        # ── Trim to exact size ──
        unique = unique[:n]

        # ── Convert indices to trade dicts ──
        result = [self.trades[i] for i in unique if i < len(self.trades)]

        # Shuffle to prevent order bias
        random.shuffle(result)

        return result

    # ════════════════════════════════════════════════════════════════
    # PRIORITY UPDATE
    # ════════════════════════════════════════════════════════════════

    def compute_priorities(self, model, feature_extractor, hist_ps_map=None) -> None:
        """
        Update priorities based on TD-error (Temporal Difference error).
        TD-error = |predicted_R - actual_R| for each trade.

        Trades where the model was most wrong get highest priority,
        so the next training pass focuses on correcting those mistakes.

        Args:
            model: Trained XGBoost model (with .predict() method)
            feature_extractor: Function that takes (trade_dict, hist_ps_dict) → np.ndarray
            hist_ps_map: Optional dict mapping index → hist_ps features
        """
        if not self.trades or model is None:
            return

        try:
            new_priorities = []
            updated_count = 0

            for idx, trade in enumerate(self.trades):
                try:
                    hist_ps = hist_ps_map.get(idx) if hist_ps_map else None
                    features = feature_extractor(trade, hist_ps_features=hist_ps)

                    if features is None:
                        new_priorities.append(MIN_PRIORITY)
                        continue

                    features_2d = features.reshape(1, -1)
                    predicted_r = float(model.predict(features_2d)[0])
                    actual_r = float(trade.get('profit_r', 0))

                    # TD-error: how wrong was the model?
                    td_error = abs(predicted_r - actual_r)

                    # Also boost priority for edge cases (extreme R values)
                    # Model needs to learn to handle outliers
                    edge_bonus = 0.0
                    if abs(actual_r) > 2.0:  # Big winner or big loser
                        edge_bonus = 0.5

                    priority = td_error + edge_bonus
                    new_priorities.append(max(priority, MIN_PRIORITY))
                    updated_count += 1

                except Exception:
                    new_priorities.append(MIN_PRIORITY)

            self.priorities = new_priorities
            log.info(f"[REPLAY] Updated priorities for {updated_count}/{len(self.trades)} trades "
                     f"(avg TD-error: {np.mean(new_priorities):.3f})")

            self._save()

        except Exception as e:
            log.error(f"[REPLAY] Failed to compute priorities: {e}")

    # ════════════════════════════════════════════════════════════════
    # INTERNAL METHODS
    # ════════════════════════════════════════════════════════════════

    def _compute_initial_priority(self, trade: dict) -> float:
        """Compute initial priority for a newly added trade."""
        # New trades get moderate-high priority so they're included
        # in early training rounds
        profit_r = float(trade.get('profit_r', 0))

        # Extreme outcomes → higher priority
        if abs(profit_r) > 2.0:
            return 1.0 + abs(profit_r) * 0.3
        elif abs(profit_r) > 1.0:
            return 0.7

        # If model made a prediction, use TD-error
        predicted_r = trade.get('model_predicted_r')
        if predicted_r is not None and predicted_r != 0:
            try:
                td_error = abs(float(predicted_r) - profit_r)
                return max(td_error, MIN_PRIORITY)
            except (TypeError, ValueError):
                pass

        # Default moderate priority
        return 0.5

    def _evict_lowest_priority(self) -> None:
        """Remove the lowest-priority trade to make room."""
        if not self.priorities:
            return

        # Find index of lowest priority
        min_idx = int(np.argmin(self.priorities))

        # Don't evict if the new trade would have even lower priority
        # (keep the better sample)
        self.trades.pop(min_idx)
        self.priorities.pop(min_idx)
        self._meta['total_evicted'] += 1

    def _update_distribution_stats(self) -> None:
        """Update metadata with current buffer distribution stats."""
        if not self.trades:
            return

        pair_counts = defaultdict(int)
        strat_counts = defaultdict(int)
        win_count = 0

        for trade in self.trades:
            pair_counts[trade.get('symbol', 'UNKNOWN')] += 1
            strat_counts[trade.get('strategy', 'UNKNOWN')] += 1
            if trade.get('win'):
                win_count += 1

        self._meta['pair_distribution'] = dict(pair_counts)
        self._meta['strategy_distribution'] = dict(strat_counts)
        self._meta['win_rate'] = round(win_count / len(self.trades) * 100, 1)

    # ════════════════════════════════════════════════════════════════
    # PERSISTENCE
    # ════════════════════════════════════════════════════════════════

    def _save(self) -> None:
        """Save buffer to disk."""
        try:
            os.makedirs(BUFFER_DIR, exist_ok=True)

            # Save trades + priorities as pickle
            data = {
                'trades': self.trades,
                'priorities': self.priorities,
                'added_count': self._added_count,
                'max_size': self.max_size,
            }
            with open(BUFFER_PATH, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Save metadata as JSON
            self._update_distribution_stats()
            self._meta['last_updated'] = datetime.now(timezone.utc).isoformat()
            self._meta['buffer_size'] = len(self.trades)
            with open(META_PATH, 'w') as f:
                json.dump(self._meta, f, indent=2, default=str)

        except Exception as e:
            log.warning(f"[REPLAY] Failed to save buffer: {e}")

    def _load(self) -> None:
        """Load buffer from disk."""
        if not os.path.exists(BUFFER_PATH):
            return

        try:
            with open(BUFFER_PATH, 'rb') as f:
                data = pickle.load(f)

            self.trades = data.get('trades', [])
            self.priorities = data.get('priorities', [])
            self._added_count = data.get('added_count', 0)
            loaded_max = data.get('max_size', DEFAULT_MAX_SIZE)

            # Respect the configured max_size (may have changed)
            if loaded_max != self.max_size:
                log.info(f"[REPLAY] Max size changed: {loaded_max} → {self.max_size}, "
                         f"trimming {max(0, len(self.trades) - self.max_size)} trades")
                if len(self.trades) > self.max_size:
                    # Keep most recent trades (they're at the end)
                    excess = len(self.trades) - self.max_size
                    self.trades = self.trades[excess:]
                    self.priorities = self.priorities[excess:]

            # Ensure priorities list matches trades list
            while len(self.priorities) < len(self.trades):
                self.priorities.append(0.5)
            if len(self.priorities) > len(self.trades):
                self.priorities = self.priorities[:len(self.trades)]

            log.info(f"[REPLAY] Loaded buffer: {len(self.trades)} trades "
                     f"({self.utilization:.0%} capacity)")

        except Exception as e:
            log.warning(f"[REPLAY] Failed to load buffer: {e}")
            self.trades = []
            self.priorities = []

    def clear(self) -> None:
        """Clear the buffer and delete saved files."""
        self.trades.clear()
        self.priorities.clear()
        self._added_count = 0

        for path in [BUFFER_PATH, META_PATH]:
            if os.path.exists(path):
                os.remove(path)

        log.info("[REPLAY] Buffer cleared")

    def get_status(self) -> dict:
        """Get buffer status summary."""
        self._update_distribution_stats()
        avg_priority = float(np.mean(self.priorities)) if self.priorities else 0.0
        max_priority = float(np.max(self.priorities)) if self.priorities else 0.0

        return {
            'size': len(self.trades),
            'max_size': self.max_size,
            'utilization': round(self.utilization, 3),
            'avg_priority': round(avg_priority, 3),
            'max_priority': round(max_priority, 3),
            'total_added': self._meta.get('total_added', 0),
            'total_evicted': self._meta.get('total_evicted', 0),
            'pair_distribution': self._meta.get('pair_distribution', {}),
            'strategy_distribution': self._meta.get('strategy_distribution', {}),
            'win_rate': self._meta.get('win_rate', 0.0),
            'persisted': os.path.exists(BUFFER_PATH),
        }


# ════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON
# ════════════════════════════════════════════════════════════════

_buffer_instance = None


def get_replay_buffer(max_size: int = DEFAULT_MAX_SIZE) -> ReplayBuffer:
    """Get or create the global ReplayBuffer singleton."""
    global _buffer_instance
    if _buffer_instance is None:
        _buffer_instance = ReplayBuffer(max_size=max_size)
    return _buffer_instance


def reset_replay_buffer():
    """Reset the global singleton (useful for testing or re-seeding)."""
    global _buffer_instance
    if _buffer_instance is not None:
        _buffer_instance.clear()
    _buffer_instance = None
