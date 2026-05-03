# =============================================================
# ai_engine/pair_strategy_features.py  v1.0
# Historical Pair-Strategy Performance Features
#
# PURPOSE: Make L1/L2 models pair-aware without creating 216
# separate per-pair models. Instead, enrich the existing 71-feature
# vector with 8 features that capture historical performance of
# each (symbol, strategy) combination.
#
# ARCHITECTURE:
#   For each signal (symbol=X, strategy=Y, time=T):
#     1. Query: all trades WHERE (symbol=X, strategy=Y, entry_time < T)
#     2. Compute dual-window stats:
#        - RECENT window (last 30 days): adaptive to regime changes
#        - ALL-TIME window: stable, high-confidence signal
#     3. Compute exponential decay weighted avg R:
#        - decay=0.95 (half-life ~14 days): recent trades count more
#     4. Compute trend: recent - all-time (positive = improving)
#     5. Pass 8 features into the model alongside existing 71 features
#
# FEATURES (8 new):
#   hist_ps_avg_r_recent     — avg R-multiple for (pair, strat) last 30 days
#   hist_ps_wr_recent        — win rate for (pair, strat) last 30 days
#   hist_ps_trades_recent    — trade count for (pair, strat) last 30 days
#   hist_ps_avg_r_all        — avg R-multiple for (pair, strat) all-time
#   hist_ps_wr_all           — win rate for (pair, strat) all-time
#   hist_ps_trades_all       — trade count for (pair, strat) all-time
#   hist_ps_avg_r_decay      — exponential decay weighted avg R (half-life ~14d)
#   hist_ps_avg_r_trend      — recent avg_r - all_time avg_r (improving?)
#
# LOOK-AHEAD BIAS PROTECTION:
#   CRITICAL: All queries use entry_time < current_signal_time
#   This means during training, each row only sees trades that
#   occurred BEFORE it — no future information leaks.
#
# SELF-IMPROVING FEEDBACK LOOP:
#   Run 1: No history → neutral defaults (0.0, 0.5, 0)
#   Run 2: DB has Run 1 trades → model sees pair-strategy history
#   Run 3: DB has Run 1+2 → more data, model self-corrects
#   Over time: The model learns "TREND_CONT on USDJPY = bad"
#   automatically, without manual blacklists.
#
# USAGE:
#   # During backtest (live inference):
#   provider = PairStrategyFeatureProvider()
#   provider.warmup_from_db()  # Load existing DB trades
#   # During backtest loop, after each trade closes:
#   provider.add_trade(symbol, strategy, entry_time, profit_r, win)
#   # When scoring a signal:
#   features = provider.get_features(symbol, strategy, current_time)
#
#   # During training (from DB):
#   features = compute_hist_ps_features_for_training(rows)
#   # Returns dict: {row_index: {feature_name: value}}
#
# BACKWARD COMPATIBLE:
#   - Works with any backtest window (30-day, 60-day, 90-day)
#   - Neutral defaults when no history exists
#   - Graceful degradation when DB is unavailable
# =============================================================

import math
from datetime import datetime, timedelta
from collections import defaultdict
from core.logger import get_logger

log = get_logger(__name__)

# ── Configuration ──

# Recent window: how many days back for "recent" stats
RECENT_WINDOW_DAYS = 30

# Exponential decay factor (0.95 = half-life ~14 days)
DECAY_FACTOR = 0.95

# Cold-start defaults (used when no history available)
DEFAULT_AVG_R = 0.0        # Neutral — neither good nor bad
DEFAULT_WR = 0.5           # 50/50 — no information
DEFAULT_TRADES = 0         # Zero trades = no information

# Minimum trades before trusting stats (below this, use blended default)
MIN_RELIABLE_TRADES = 5

# Feature names (must match FEATURE_NAMES in ml_gate.py)
HIST_PS_FEATURE_NAMES = [
    'hist_ps_avg_r_recent',
    'hist_ps_wr_recent',
    'hist_ps_trades_recent',
    'hist_ps_avg_r_all',
    'hist_ps_wr_all',
    'hist_ps_trades_all',
    'hist_ps_avg_r_decay',
    'hist_ps_avg_r_trend',
]


def _compute_decay_weighted_avg(trades: list, current_time: datetime,
                                 decay: float = DECAY_FACTOR) -> float:
    """
    Compute exponentially decayed weighted average of R-multiples.

    More recent trades count more. A trade from 14 days ago counts
    ~50% as much as a trade from today (half-life ≈ ln(2)/ln(1/decay)).

    Args:
        trades: List of dicts with 'entry_time' (datetime) and 'profit_r' (float)
        current_time: The time of the signal being evaluated
        decay: Decay factor (0 < decay < 1, closer to 1 = slower decay)

    Returns:
        Weighted average R-multiple, or DEFAULT_AVG_R if no trades
    """
    if not trades:
        return DEFAULT_AVG_R

    numerator = 0.0
    denominator = 0.0

    for t in trades:
        entry_time = t.get('entry_time')
        profit_r = t.get('profit_r', 0.0)

        if entry_time is None:
            continue

        # Handle both datetime objects and strings
        if isinstance(entry_time, str):
            try:
                entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                # Strip timezone for comparison if current_time is naive
                if current_time.tzinfo is None and entry_time.tzinfo:
                    entry_time = entry_time.replace(tzinfo=None)
            except (ValueError, AttributeError):
                continue

        # Days ago (can be 0 for same-day trades)
        days_ago = max(0.0, (current_time - entry_time).total_seconds() / 86400.0)

        # Weight = decay^days_ago
        weight = decay ** days_ago
        numerator += profit_r * weight
        denominator += weight

    if denominator < 1e-9:
        return DEFAULT_AVG_R

    return numerator / denominator


def _compute_stats(trades: list) -> dict:
    """
    Compute basic statistics from a list of trades.

    Args:
        trades: List of dicts with 'profit_r' (float) and 'win' (bool/int)

    Returns:
        dict with avg_r, wr, count
    """
    if not trades:
        return {
            'avg_r': DEFAULT_AVG_R,
            'wr': DEFAULT_WR,
            'count': DEFAULT_TRADES,
        }

    count = len(trades)
    total_r = sum(t.get('profit_r', 0.0) for t in trades)
    wins = sum(1 for t in trades if t.get('win', 0) in (1, True))

    return {
        'avg_r': total_r / count if count > 0 else DEFAULT_AVG_R,
        'wr': wins / count if count > 0 else DEFAULT_WR,
        'count': count,
    }


def _blend_with_default(stats: dict, confidence: float) -> dict:
    """
    Blend computed stats with cold-start defaults based on confidence.

    When we have few trades, we blend toward neutral defaults to avoid
    the model overfitting to noisy small-sample statistics.

    confidence = min(1.0, trades / MIN_RELIABLE_TRADES)
      0 trades  → confidence=0.0 → 100% default
      5 trades  → confidence=1.0 → 100% actual stats

    Args:
        stats: dict with avg_r, wr, count
        confidence: 0.0 (no trust) to 1.0 (full trust)

    Returns:
        Blended stats dict
    """
    return {
        'avg_r': stats['avg_r'] * confidence + DEFAULT_AVG_R * (1 - confidence),
        'wr': stats['wr'] * confidence + DEFAULT_WR * (1 - confidence),
        'count': stats['count'],
    }


class PairStrategyFeatureProvider:
    """
    Live inference provider for historical pair-strategy features.

    Maintains an in-memory cache of historical trades organized by
    (symbol, strategy) for efficient lookups during backtesting.

    Usage:
        provider = PairStrategyFeatureProvider()
        provider.warmup_from_db()  # Pre-load existing DB data
        # ... during backtest loop ...
        features = provider.get_features('EURUSD', 'TREND_CONTINUATION', now)
        # ... after trade closes ...
        provider.add_trade('EURUSD', 'TREND_CONTINUATION', entry_time, profit_r, win)
    """

    def __init__(self):
        # {(symbol, strategy): [(entry_time, profit_r, win), ...]}
        self._history = defaultdict(list)
        self._loaded = False
        self._total_loaded = 0

    def warmup_from_db(self):
        """
        Load all existing trades from the DB into memory.
        Called once at the start of a backtest run.
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
            cursor.execute("""
                SELECT
                    symbol, strategy, entry_time, profit_r, win
                FROM backtest_trades
                WHERE source IN ('BACKTEST', 'SHADOW')
                  AND outcome IS NOT NULL
                  AND outcome != ''
                  AND entry_time IS NOT NULL
                  AND profit_r IS NOT NULL
                ORDER BY entry_time ASC
            """)
            rows = cursor.fetchall()
            conn.close()

            loaded = 0
            for row in rows:
                symbol = str(row.get('symbol', ''))
                strategy = str(row.get('strategy', ''))
                entry_time = row.get('entry_time')
                profit_r = float(row.get('profit_r', 0))
                win = row.get('win', 0) in (1, True, '1')

                if not symbol or not strategy or entry_time is None:
                    continue

                # Normalize entry_time to datetime
                if isinstance(entry_time, str):
                    try:
                        entry_time = datetime.fromisoformat(
                            entry_time.replace('Z', '+00:00'))
                        if entry_time.tzinfo:
                            entry_time = entry_time.replace(tzinfo=None)
                    except (ValueError, AttributeError):
                        continue

                key = (symbol, strategy)
                self._history[key].append({
                    'entry_time': entry_time,
                    'profit_r': profit_r,
                    'win': win,
                })
                loaded += 1

            self._loaded = True
            self._total_loaded = loaded
            log.info(f"[PAIR_STRAT_FEAT] Warmed up: {loaded} historical trades "
                     f"across {len(self._history)} (symbol, strategy) combos")

        except Exception as e:
            log.warning(f"[PAIR_STRAT_FEAT] DB warmup failed (will use empty history): {e}")

    def add_trade(self, symbol: str, strategy: str,
                  entry_time, profit_r: float, win: bool):
        """
        Add a closed trade to the in-memory history.
        Called after a trade closes during backtest.

        This ensures that trades taken in the current backtest run
        become visible to future signals (but never to past ones,
        due to the entry_time filtering in get_features()).
        """
        if isinstance(entry_time, datetime):
            pass
        elif isinstance(entry_time, str):
            try:
                entry_time = datetime.fromisoformat(
                    entry_time.replace('Z', '+00:00'))
                if entry_time.tzinfo:
                    entry_time = entry_time.replace(tzinfo=None)
            except (ValueError, AttributeError):
                return
        else:
            return

        key = (symbol, strategy)
        self._history[key].append({
            'entry_time': entry_time,
            'profit_r': profit_r,
            'win': win,
        })

    def get_features(self, symbol: str, strategy: str,
                     current_time) -> dict:
        """
        Compute 8 historical pair-strategy features for a signal.

        Uses ONLY trades with entry_time < current_time to prevent
        look-ahead bias. This is the CRITICAL property that makes
        the self-improving loop valid.

        Args:
            symbol: e.g. 'EURUSD'
            strategy: e.g. 'TREND_CONTINUATION'
            current_time: The time of the signal being evaluated (datetime)

        Returns:
            dict with 8 feature values keyed by feature name
        """
        key = (symbol, strategy)
        all_trades = self._history.get(key, [])

        # Filter to trades BEFORE current_time (look-ahead bias protection)
        if isinstance(current_time, str):
            try:
                current_time = datetime.fromisoformat(
                    current_time.replace('Z', '+00:00'))
                if current_time.tzinfo:
                    current_time = current_time.replace(tzinfo=None)
            except (ValueError, AttributeError):
                return self._get_defaults()

        past_trades = [t for t in all_trades if t['entry_time'] < current_time]

        if not past_trades:
            return self._get_defaults()

        # ── Recent window (last N days) ──
        recent_cutoff = current_time - timedelta(days=RECENT_WINDOW_DAYS)
        recent_trades = [t for t in past_trades
                         if t['entry_time'] >= recent_cutoff]

        recent_stats = _compute_stats(recent_trades)
        all_stats = _compute_stats(past_trades)

        # ── Blend with defaults based on sample size ──
        recent_confidence = min(1.0, recent_stats['count'] / MIN_RELIABLE_TRADES)
        all_confidence = min(1.0, all_stats['count'] / MIN_RELIABLE_TRADES)

        recent_blended = _blend_with_default(recent_stats, recent_confidence)
        all_blended = _blend_with_default(all_stats, all_confidence)

        # ── Exponential decay weighted avg R ──
        decay_avg_r = _compute_decay_weighted_avg(past_trades, current_time)

        # ── Trend: recent - all-time (positive = improving) ──
        avg_r_trend = recent_blended['avg_r'] - all_blended['avg_r']

        return {
            'hist_ps_avg_r_recent': round(recent_blended['avg_r'], 4),
            'hist_ps_wr_recent': round(recent_blended['wr'], 4),
            'hist_ps_trades_recent': float(recent_stats['count']),
            'hist_ps_avg_r_all': round(all_blended['avg_r'], 4),
            'hist_ps_wr_all': round(all_blended['wr'], 4),
            'hist_ps_trades_all': float(all_stats['count']),
            'hist_ps_avg_r_decay': round(decay_avg_r, 4),
            'hist_ps_avg_r_trend': round(avg_r_trend, 4),
        }

    def get_cache_stats(self) -> dict:
        """Return stats about the in-memory cache for debugging."""
        combos = len(self._history)
        total_trades = sum(len(v) for v in self._history.values())
        top_combos = sorted(
            [(k, len(v)) for k, v in self._history.items()],
            key=lambda x: x[1], reverse=True)[:10]
        return {
            'loaded_from_db': self._total_loaded,
            'combos': combos,
            'total_trades_in_cache': total_trades,
            'top_combos': [(f"{k[0]}/{k[1]}", c) for k, c in top_combos],
        }

    @staticmethod
    def _get_defaults() -> dict:
        """Return cold-start default features."""
        return {
            'hist_ps_avg_r_recent': DEFAULT_AVG_R,
            'hist_ps_wr_recent': DEFAULT_WR,
            'hist_ps_trades_recent': float(DEFAULT_TRADES),
            'hist_ps_avg_r_all': DEFAULT_AVG_R,
            'hist_ps_wr_all': DEFAULT_WR,
            'hist_ps_trades_all': float(DEFAULT_TRADES),
            'hist_ps_avg_r_decay': DEFAULT_AVG_R,
            'hist_ps_avg_r_trend': 0.0,
        }


def compute_hist_ps_features_for_training(rows: list) -> dict:
    """
    Pre-compute historical pair-strategy features for DB training rows.

    This is the training-time counterpart to PairStrategyFeatureProvider.
    It processes a sorted list of DB rows and computes rolling stats
    for each row using ONLY trades that occurred before it.

    CRITICAL: rows MUST be sorted by entry_time ASC for correct
    look-ahead bias protection.

    Args:
        rows: List of DB row dicts, sorted by entry_time ASC.
              Each row must have: symbol, strategy, entry_time, profit_r, win

    Returns:
        dict mapping row_index → {feature_name: value}
        (Use row index because rows may not have unique IDs in this context)
    """
    result = {}

    # Build a running history per (symbol, strategy)
    # As we iterate through rows in time order, each row only sees
    # trades added BEFORE it — no look-ahead bias.
    running_history = defaultdict(list)  # {(symbol, strategy): [trade_dicts]}

    # We need to know each row's entry_time to filter
    for idx, row in enumerate(rows):
        symbol = str(row.get('symbol', ''))
        strategy = str(row.get('strategy', ''))
        entry_time = row.get('entry_time')

        if isinstance(entry_time, str):
            try:
                entry_time = datetime.fromisoformat(
                    entry_time.replace('Z', '+00:00'))
                if entry_time.tzinfo:
                    entry_time = entry_time.replace(tzinfo=None)
            except (ValueError, AttributeError):
                entry_time = None

        if not symbol or not strategy or entry_time is None:
            result[idx] = PairStrategyFeatureProvider._get_defaults()
            continue

        key = (symbol, strategy)
        past_trades = running_history[key]

        if not past_trades:
            result[idx] = PairStrategyFeatureProvider._get_defaults()
        else:
            # ── Recent window ──
            recent_cutoff = entry_time - timedelta(days=RECENT_WINDOW_DAYS)
            recent_trades = [t for t in past_trades
                             if t['entry_time'] >= recent_cutoff]

            recent_stats = _compute_stats(recent_trades)
            all_stats = _compute_stats(past_trades)

            # ── Blend with defaults ──
            recent_confidence = min(1.0, recent_stats['count'] / MIN_RELIABLE_TRADES)
            all_confidence = min(1.0, all_stats['count'] / MIN_RELIABLE_TRADES)

            recent_blended = _blend_with_default(recent_stats, recent_confidence)
            all_blended = _blend_with_default(all_stats, all_confidence)

            # ── Exponential decay ──
            decay_avg_r = _compute_decay_weighted_avg(past_trades, entry_time)

            # ── Trend ──
            avg_r_trend = recent_blended['avg_r'] - all_blended['avg_r']

            result[idx] = {
                'hist_ps_avg_r_recent': round(recent_blended['avg_r'], 4),
                'hist_ps_wr_recent': round(recent_blended['wr'], 4),
                'hist_ps_trades_recent': float(recent_stats['count']),
                'hist_ps_avg_r_all': round(all_blended['avg_r'], 4),
                'hist_ps_wr_all': round(all_blended['wr'], 4),
                'hist_ps_trades_all': float(all_stats['count']),
                'hist_ps_avg_r_decay': round(decay_avg_r, 4),
                'hist_ps_avg_r_trend': round(avg_r_trend, 4),
            }

        # ── Add THIS trade to history for subsequent rows ──
        # This happens AFTER computing features for this row,
        # so this row's outcome doesn't leak into its own features.
        profit_r = float(row.get('profit_r', 0) or 0)
        win = row.get('win', 0) in (1, True, '1')
        running_history[key].append({
            'entry_time': entry_time,
            'profit_r': profit_r,
            'win': win,
        })

    return result


def compute_hist_ps_for_ml_gate_training(rows: list) -> dict:
    """
    Pre-compute hist_ps features for L2 ML Gate training.

    Unlike per-strategy L1 training (which only sees one strategy's rows),
    L2 training sees ALL strategies' rows. So we need to track history
    across all (symbol, strategy) combos.

    Args:
        rows: List of DB row dicts, sorted by entry_time ASC.

    Returns:
        dict mapping row_index → {feature_name: value}
    """
    return compute_hist_ps_features_for_training(rows)
