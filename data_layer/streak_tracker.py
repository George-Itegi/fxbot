# =============================================================
# data_layer/streak_tracker.py
# Streak Detection & Z-Score Calculator
#
# Analyzes recent trade outcomes for streaks (consecutive wins/losses)
# and computes Z-scores to measure how unusual a signal is vs history.
#
# Two main functions:
#   - calculate_streak():      Streak detection from backtest_trades
#   - calculate_signal_zscore(): Z-score analysis of signal scores
# =============================================================

import numpy as np
from core.logger import get_logger

log = get_logger(__name__)

# Safe defaults
_STREAK_DEFAULT = {
    'current_streak': 0,
    'streak_type': 'NONE',
    'recent_wr': 0.5,
    'recent_avg_r': 0.0,
    'is_hot_streak': False,
    'is_cold_streak': False,
    'trades_analyzed': 0,
}

_ZSCORE_DEFAULT = {
    'z_score': 0.0,
    'percentile': 50.0,
    'is_rare': False,
    'mean_score': 0.0,
    'std_score': 1.0,
    'rarity_label': 'COMMON',
}


def calculate_streak(symbol: str = None, strategy: str = None,
                     lookback: int = 10) -> dict:
    """
    Analyze recent trade outcomes for streaks.

    Queries the backtest_trades table for recent completed trades
    (ordered by entry_time DESC) and identifies consecutive wins/losses.

    Args:
        symbol:   Optional filter by symbol (e.g. 'EURJPY').
                  If None, analyzes all symbols.
        strategy: Optional filter by strategy name.
                  If None, includes all strategies.
        lookback: Number of recent trades to analyze.

    Returns:
        dict with:
            - current_streak: int   (positive = wins, negative = losses)
                                 (e.g. +3 = 3 wins in a row)
            - streak_type:   str   ('WIN' | 'LOSS' | 'NONE')
            - recent_wr:     float (win rate of last N trades)
            - recent_avg_r:  float (avg R-multiple of last N trades)
            - is_hot_streak: bool  (3+ consecutive wins)
            - is_cold_streak: bool (3+ consecutive losses)
            - trades_analyzed: int
    """
    result = dict(_STREAK_DEFAULT)

    try:
        from database.db_manager import get_connection

        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        # Build query with optional filters
        conditions = ["outcome IS NOT NULL", "outcome != ''"]
        params = []

        if symbol:
            conditions.append("symbol = %s")
            params.append(symbol.upper())
        if strategy:
            conditions.append("strategy = %s")
            params.append(strategy)

        where_clause = " AND ".join(conditions)
        params_tuple = tuple(params + [lookback])

        cursor.execute(f"""
            SELECT win, profit_r, outcome
            FROM backtest_trades
            WHERE {where_clause}
            ORDER BY entry_time DESC
            LIMIT %s
        """, params_tuple)

        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        if not rows:
            log.debug(f"[STREAK] No trades found for "
                      f"symbol={symbol} strategy={strategy}")
            return result

        result['trades_analyzed'] = len(rows)

        # Calculate streak from the most recent trade
        streak = 0
        for row in rows:
            is_win = bool(row.get('win', 0))
            if streak == 0:
                # First trade: set streak direction
                streak = 1 if is_win else -1
            elif (streak > 0 and is_win) or (streak < 0 and not is_win):
                # Same direction: extend streak
                streak += 1 if is_win else -1
            else:
                # Streak broken
                break

        result['current_streak'] = streak

        if streak > 0:
            result['streak_type'] = 'WIN'
        elif streak < 0:
            result['streak_type'] = 'LOSS'
        else:
            result['streak_type'] = 'NONE'

        # Hot/cold streak detection
        result['is_hot_streak'] = streak >= 3
        result['is_cold_streak'] = streak <= -3

        # Calculate recent win rate and avg R
        wins = sum(1 for r in rows if r.get('win', 0))
        result['recent_wr'] = round(wins / len(rows), 3)

        r_values = [float(r.get('profit_r', 0) or 0) for r in rows]
        valid_r = [r for r in r_values if r != 0 or True]  # Include all
        result['recent_avg_r'] = round(float(np.mean(r_values)), 3) if r_values else 0.0

        log.debug(f"[STREAK] {symbol or 'ALL'}: streak={streak:+d} "
                  f"type={result['streak_type']} "
                  f"wr={result['recent_wr']:.1%} "
                  f"avg_r={result['recent_avg_r']:.2f} "
                  f"trades={len(rows)}")

        return result

    except Exception as e:
        log.error(f"[STREAK] Error calculating streak for "
                  f"symbol={symbol} strategy={strategy}: {e}")
        return result


def calculate_signal_zscore(symbol: str, current_score: int,
                            lookback: int = 100) -> dict:
    """
    Calculate how unusual the current signal score is compared to
    recent historical signal scores for this symbol.

    Uses the backtest_trades table to find historical scores and
    computes the Z-score (how many standard deviations from mean).

    Args:
        symbol:        Forex pair symbol (e.g. 'EURJPY')
        current_score: The current signal score (0-100)
        lookback:      Number of historical trades to compare against

    Returns:
        dict with:
            - z_score:       float (std deviations from mean)
            - percentile:    float (0-100, percentile rank of current score)
            - is_rare:       bool  (|z_score| > 2.0)
            - mean_score:    float (historical mean score)
            - std_score:     float (historical std of scores)
            - rarity_label:  str ('COMMON'|'UNUSUAL'|'RARE'|'EXTREME')
    """
    result = dict(_ZSCORE_DEFAULT)
    result['mean_score'] = float(current_score)

    try:
        from database.db_manager import get_connection

        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        # Query historical scores for this symbol
        cursor.execute("""
            SELECT score
            FROM backtest_trades
            WHERE symbol = %s
              AND score IS NOT NULL
              AND score > 0
            ORDER BY entry_time DESC
            LIMIT %s
        """, (symbol.upper(), lookback))

        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        if not rows or len(rows) < 5:
            log.debug(f"[ZSCORE] Insufficient history for {symbol} "
                      f"({len(rows) if rows else 0} trades)")
            return result

        # Extract historical scores
        hist_scores = np.array([float(r['score']) for r in rows
                                if r.get('score') is not None])

        if len(hist_scores) < 5:
            return result

        mean_s = float(np.mean(hist_scores))
        std_s = float(np.std(hist_scores))

        result['mean_score'] = round(mean_s, 1)
        result['std_score'] = round(std_s, 1)

        if std_s < 0.01:
            # All scores are nearly identical — Z-score is undefined
            result['z_score'] = 0.0
            result['percentile'] = 50.0
            result['rarity_label'] = 'COMMON'
            return result

        # Compute Z-score
        z = (current_score - mean_s) / std_s
        result['z_score'] = round(float(z), 3)

        # Percentile: what % of historical scores is current below?
        below_count = int((hist_scores < current_score).sum())
        percentile = round(below_count / len(hist_scores) * 100, 1)
        result['percentile'] = percentile

        # Rare detection
        abs_z = abs(z)
        result['is_rare'] = abs_z > 2.0

        # Rarity label
        if abs_z > 3.0:
            rarity = 'EXTREME'
        elif abs_z > 2.0:
            rarity = 'RARE'
        elif abs_z > 1.0:
            rarity = 'UNUSUAL'
        else:
            rarity = 'COMMON'

        result['rarity_label'] = rarity

        log.debug(f"[ZSCORE] {symbol}: current={current_score} "
                  f"z={z:.2f} pctl={percentile:.1f}% "
                  f"mean={mean_s:.1f} std={std_s:.1f} "
                  f"label={rarity}")

        return result

    except Exception as e:
        log.error(f"[ZSCORE] Error for {symbol}: {e}")
        return result
