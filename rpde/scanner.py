# =============================================================
# rpde/scanner.py  — Reverse Pattern Discovery Engine: Big Move Scanner
#
# PURPOSE: Scan historical M5 price data to find "golden moments" —
# bars right BEFORE a significant price move happened. This is the
# core of the reverse-engineering approach: instead of predicting
# what will happen, find what DID happen and extract the pattern
# that preceded it.
#
# ARCHITECTURE:
#   For every M5 bar, look FORWARD N bars (default 24 = 2 hours):
#     - If price moved +threshold pips → BUY golden moment
#     - If price moved -threshold pips → SELL golden moment
#     - If price didn't move much → skip (noise)
#
#   For each golden moment, call feature_snapshot.extract_snapshot_at_bar()
#   to capture the 93-feature signature at that point in time.
#
# OUTPUT: Golden moments stored via rpde.database (if available),
#         otherwise returned as list of dicts for in-memory processing.
# =============================================================

import time
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timezone, timedelta

from config.settings import PAIR_WHITELIST
from rpde.config import (
    FORWARD_LOOK_BARS,
    DEFAULT_MIN_MOVE_PIPS,
    PAIR_MOVE_THRESHOLDS,
    SCAN_TIMEFRAME,
    MIN_BAR_SEPARATION,
    DEAD_ZONE_HOURS_GMT,
    FRIDAY_CLOSE_HOUR_GMT,
    PAIR_SESSION_FILTER,
    DEFAULT_ALLOWED_SESSIONS,
    MIN_ATR_PERCENTILE,
    CONFLUENCE_MIN_SCORE,
    CONFLUENCE_THRESHOLD_ORDER_FLOW_DELTA,
    CONFLUENCE_THRESHOLD_ORDER_FLOW_IMBALANCE,
    CONFLUENCE_THRESHOLD_ATR_PERCENTILE,
    CONFLUENCE_THRESHOLD_PD_ZONE,
    CONFLUENCE_TIER1_SESSIONS,
    NEGATIVE_SAMPLE_INTERVAL,
    NEGATIVE_SAMPLE_RATIO,
    NEGATIVE_MAX_SAMPLES_PER_PAIR,
)
from core.pip_utils import get_pip_size
from core.logger import get_logger

log = get_logger(__name__)

# ── Try to import database storage (graceful degradation) ─────
try:
    from rpde.database import store_golden_moment, store_scan_history, update_scan_history
    _DB_AVAILABLE = True
except ImportError:
    log.warning("[RPDE_SCANNER] rpde.database module not found — "
                "results will not be persisted to DB. "
                "Run with database module for full functionality.")
    _DB_AVAILABLE = False

# ── Try to import feature snapshot extraction ────────────────
try:
    from rpde.feature_snapshot import extract_snapshot_at_index
    _SNAPSHOT_AVAILABLE = True
except ImportError:
    log.warning("[RPDE_SCANNER] feature_snapshot module not found — "
                "golden moments will lack 93-feature snapshots.")
    _SNAPSHOT_AVAILABLE = False


# ════════════════════════════════════════════════════════════════
# MT5 TIMEFRAME MAP
# ════════════════════════════════════════════════════════════════

_TF_MAP = {
    "M1": mt5.TIMEFRAME_M1,   "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,  "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}


# ════════════════════════════════════════════════════════════════
# HELPER: Load Historical Candles from MT5
# ════════════════════════════════════════════════════════════════

def _load_m5_candles(pair: str, num_bars: int) -> pd.DataFrame:
    """
    Load M5 historical candles from MT5.

    Args:
        pair: Symbol string (e.g. 'EURJPY')
        num_bars: Number of bars to request

    Returns:
        DataFrame with columns: time, open, high, low, close, volume, spread
        Returns empty DataFrame on failure.
    """
    tf = _TF_MAP.get(SCAN_TIMEFRAME)
    if tf is None:
        log.error(f"[RPDE_SCANNER] Unknown timeframe: {SCAN_TIMEFRAME}")
        return pd.DataFrame()

    rates = mt5.copy_rates_from_pos(pair, tf, 0, num_bars)

    if rates is None or len(rates) == 0:
        error_code, error_msg = mt5.last_error()
        log.error(f"[RPDE_SCANNER] MT5 error for {pair} {SCAN_TIMEFRAME}: "
                  f"[{error_code}] {error_msg}")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread']]
    df = df.rename(columns={'tick_volume': 'volume'})
    df = df.drop_duplicates(subset='time').reset_index(drop=True)

    log.info(f"[RPDE_SCANNER] Loaded {len(df)} M5 bars for {pair} "
             f"({df['time'].iloc[0].strftime('%Y-%m-%d')} to "
             f"{df['time'].iloc[-1].strftime('%Y-%m-%d')})")
    return df


# ════════════════════════════════════════════════════════════════
# CORE: Look Forward to Detect Significant Moves
# ════════════════════════════════════════════════════════════════

def _look_forward(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                  bar_idx: int, look_bars: int, threshold_pips: float,
                  pip_value: float) -> dict:
    """
    Look forward from bar_idx to find if a significant move happens
    within the next `look_bars` bars.

    For BUY: find max(high[bar_idx:bar_idx+look_bars]) - close[bar_idx]
    For SELL: find close[bar_idx] - min(low[bar_idx:bar_idx+look_bars])

    Additionally tracks Maximum Adverse Excursion (MAE) — how far price
    went AGAINST the trade direction before reaching the peak. This is
    critical for stop loss placement intelligence.

    Args:
        closes: numpy array of close prices
        highs: numpy array of high prices
        lows: numpy array of low prices
        bar_idx: index of the bar to look forward from
        look_bars: number of bars to look ahead
        threshold_pips: minimum pips for a significant move
        pip_value: pip size for the pair (e.g. 0.01 for JPY pairs)

    Returns:
        dict with:
            - direction: 'BUY' or 'SELL' or None
            - move_pips: float (best move in pips)
            - peak_price: float (the best price reached)
            - peak_bar_offset: int (bars to reach peak)
            - forward_return: float (move_pips / atr_at_entry, 0.0 if no ATR)
            - mae_pips: float (max adverse excursion in pips BEFORE peak)
            - mae_bars: int (bars until worst adverse point before peak)
            - mae_price: float (worst price before peak)
            - mfe_after_mae: float (favorable move remaining AFTER worst point)
    """
    # Calculate bounds
    end_idx = min(bar_idx + look_bars, len(closes))
    forward_count = end_idx - bar_idx

    if forward_count < 2:
        return {
            'direction': None,
            'move_pips': 0.0,
            'peak_price': closes[bar_idx],
            'peak_bar_offset': 0,
            'forward_return': 0.0,
            'mae_pips': 0.0,
            'mae_bars': 0,
            'mae_price': closes[bar_idx],
            'mfe_after_mae': 0.0,
        }

    entry_close = closes[bar_idx]
    forward_highs = highs[bar_idx:end_idx]
    forward_lows = lows[bar_idx:end_idx]

    # BUY potential: max high forward - entry close
    buy_move_price = float(np.max(forward_highs)) - entry_close
    buy_move_pips = buy_move_price / pip_value if pip_value > 0 else 0.0
    buy_peak_idx = int(np.argmax(forward_highs))

    # SELL potential: entry close - min low forward
    sell_move_price = entry_close - float(np.min(forward_lows))
    sell_move_pips = sell_move_price / pip_value if pip_value > 0 else 0.0
    sell_peak_idx = int(np.argmin(forward_lows))

    # Determine which direction has the stronger signal
    # If both exceed threshold, take the stronger one
    buy_exceeds = buy_move_pips >= threshold_pips
    sell_exceeds = sell_move_pips >= threshold_pips

    if not buy_exceeds and not sell_exceeds:
        return {
            'direction': None,
            'move_pips': 0.0,
            'peak_price': entry_close,
            'peak_bar_offset': 0,
            'forward_return': 0.0,
            'mae_pips': 0.0,
            'mae_bars': 0,
            'mae_price': entry_close,
            'mfe_after_mae': 0.0,
        }

    # Pick the direction with the larger move
    if buy_move_pips >= sell_move_pips:
        direction = 'BUY'
        move_pips = round(buy_move_pips, 1)
        peak_price = round(float(np.max(forward_highs)), 6)
        peak_bar_offset = buy_peak_idx

        # MAE for BUY: worst price = lowest low BEFORE the peak
        # i.e., how far did price drop below entry before rallying?
        pre_peak_lows = forward_lows[:buy_peak_idx + 1]
        if len(pre_peak_lows) > 0:
            worst_low = float(np.min(pre_peak_lows))
            worst_low_idx = int(np.argmin(pre_peak_lows))
            mae_price_pips = (entry_close - worst_low) / pip_value if pip_value > 0 else 0.0
        else:
            worst_low = entry_close
            worst_low_idx = 0
            mae_price_pips = 0.0

        mae_pips = round(max(mae_price_pips, 0.0), 1)
        mae_bars = worst_low_idx
        mae_price = round(worst_low, 6)

        # MFE after MAE: how much favorable move remained after worst point
        # This tells us the "reward" for surviving the adverse excursion
        if worst_low < entry_close:
            mfe_after_mae_price = float(np.max(forward_highs[worst_low_idx:])) - worst_low
            mfe_after_mae = round(mfe_after_mae_price / pip_value if pip_value > 0 else 0.0, 1)
        else:
            mfe_after_mae = move_pips

    else:
        direction = 'SELL'
        move_pips = round(sell_move_pips, 1)
        peak_price = round(float(np.min(forward_lows)), 6)
        peak_bar_offset = sell_peak_idx

        # MAE for SELL: worst price = highest high BEFORE the trough
        # i.e., how far did price rise above entry before dropping?
        pre_peak_highs = forward_highs[:sell_peak_idx + 1]
        if len(pre_peak_highs) > 0:
            worst_high = float(np.max(pre_peak_highs))
            worst_high_idx = int(np.argmax(pre_peak_highs))
            mae_price_pips = (worst_high - entry_close) / pip_value if pip_value > 0 else 0.0
        else:
            worst_high = entry_close
            worst_high_idx = 0
            mae_price_pips = 0.0

        mae_pips = round(max(mae_price_pips, 0.0), 1)
        mae_bars = worst_high_idx
        mae_price = round(worst_high, 6)

        # MFE after MAE: how much favorable move remained after worst point
        if worst_high > entry_close:
            mfe_after_mae_price = worst_high - float(np.min(forward_lows[worst_high_idx:]))
            mfe_after_mae = round(mfe_after_mae_price / pip_value if pip_value > 0 else 0.0, 1)
        else:
            mfe_after_mae = move_pips

    return {
        'direction': direction,
        'move_pips': move_pips,
        'peak_price': peak_price,
        'peak_bar_offset': peak_bar_offset,
        'forward_return': 0.0,  # Will be computed by caller if ATR available
        'mae_pips': mae_pips,
        'mae_bars': mae_bars,
        'mae_price': mae_price,
        'mfe_after_mae': round(mfe_after_mae, 1),
    }


# ════════════════════════════════════════════════════════════════
# REGIME FILTER: Session Classification and Bar Filtering
# ════════════════════════════════════════════════════════════════

# Session classification from GMT hour
_SESSION_MAP = {
    (0, 2): "SYDNEY",           # Sydney open (variable, 0-2 GMT approx)
    (2, 7): "ASIAN_MIDDAY",     # Dead zone — skip
    (7, 9): "LONDON_OPEN",      # London open
    (9, 13): "LONDON_SESSION",  # Full London session
    (13, 16): "NY_LONDON_OVERLAP",  # Overlap
    (16, 20): "NY_AFTERNOON",   # Dead zone — skip
    (20, 22): "NEWYORK_SESSION", # NY session after overlap
    (22, 24): "SYDNEY",         # Late Sydney/early Asian
}


def _session_from_gmt_hour(hour: int, pair: str = None) -> str:
    """Classify session from GMT hour. Returns session name or 'DEAD_ZONE'."""
    for (start, end), name in _SESSION_MAP.items():
        if start <= hour < end:
            return name
    return "DEAD_ZONE"


def _is_bar_allowed(pair: str, bar_time, atr_percentile: float = None) -> tuple:
    """
    Check if a bar passes all regime filters.

    Returns (allowed: bool, reason: str).
    """
    hour_gmt = bar_time.hour

    # 1. Dead zone check (hard skip — all pairs)
    for start, end in DEAD_ZONE_HOURS_GMT:
        if start <= hour_gmt < end:
            return (False, f"dead_zone_{start}-{end}")

    # 2. Friday close check
    if bar_time.weekday() == 4 and hour_gmt >= FRIDAY_CLOSE_HOUR_GMT:
        return (False, "friday_close")

    # 3. Sunday open check (first few hours of Sunday = wide spreads)
    if bar_time.weekday() == 6 and hour_gmt < 2:
        return (False, "sunday_open")

    # 4. Session filter per pair
    session = _session_from_gmt_hour(hour_gmt, pair)
    allowed_sessions = PAIR_SESSION_FILTER.get(pair, DEFAULT_ALLOWED_SESSIONS)
    if session not in allowed_sessions:
        return (False, f"session_{session}")

    # 5. ATR percentile filter (if available)
    if atr_percentile is not None and atr_percentile < MIN_ATR_PERCENTILE:
        return (False, f"low_vol_{atr_percentile:.0f}pct")

    return (True, "ok")


# ════════════════════════════════════════════════════════════════
# MAIN: Scan a Single Pair for Golden Moments
# ════════════════════════════════════════════════════════════════

def scan_pair(pair: str, days: int = 360, scan_id: str = None) -> dict:
    """
    Scan historical data for a single pair to find golden moments.

    Process:
    1. Load M5 historical candles from MT5
    2. For each bar, look forward FORWARD_LOOK_BARS bars
    3. Calculate max favorable excursion (best price) and max adverse excursion
    4. If move >= threshold, capture feature snapshot at that bar
    5. Store golden moments in rpde_pattern_scans table (if DB available)

    Args:
        pair: Symbol string (e.g. 'EURJPY')
        days: Number of days of history to scan
        scan_id: Optional scan ID (auto-generated if None)

    Returns:
        dict with:
            - pair: str
            - bars_scanned: int
            - golden_moments: int
            - buy_moments: int
            - sell_moments: int
            - avg_move_pips: float
            - scan_id: str
            - moments: list of golden moment dicts (if DB not available)
    """
    t0 = time.time()

    # Generate scan_id if not provided
    if scan_id is None:
        scan_id = f"scan_{pair}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Determine pip value and threshold
    pip_value = get_pip_size(pair)
    threshold_pips = PAIR_MOVE_THRESHOLDS.get(pair, DEFAULT_MIN_MOVE_PIPS)

    log.info(f"[RPDE_SCANNER] ══════════════════════════════════════")
    log.info(f"[RPDE_SCANNER] Scanning {pair}: {days} days, "
             f"threshold={threshold_pips} pips, look={FORWARD_LOOK_BARS} bars")
    log.info(f"[RPDE_SCANNER] Pip size: {pip_value}, Scan ID: {scan_id}")

    # Load M5 data: ~288 bars/day * days + buffer
    num_bars = int(days * 288) + 200
    df = _load_m5_candles(pair, num_bars)

    if df.empty or len(df) < FORWARD_LOOK_BARS + 50:
        log.error(f"[RPDE_SCANNER] Insufficient data for {pair}: "
                  f"{len(df)} bars (need at least {FORWARD_LOOK_BARS + 50})")
        return {
            'pair': pair,
            'bars_scanned': 0,
            'golden_moments': 0,
            'buy_moments': 0,
            'sell_moments': 0,
            'avg_move_pips': 0.0,
            'scan_id': scan_id,
            'duration_seconds': round(time.time() - t0, 1),
            'error': 'Insufficient historical data',
        }

    # Convert to numpy arrays for speed
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values

    # Compute ATR(14) for forward_return calculation
    df = _compute_atr_series(df)

    # Pre-compute indicators for feature extraction
    snapshot_ok = _SNAPSHOT_AVAILABLE
    if snapshot_ok:
        try:
            from rpde.feature_snapshot import _add_indicators
            if 'rsi' not in df.columns:
                t_pre = time.time()
                df = _add_indicators(df)
                log.info(f"[RPDE_SCANNER] Pre-computed indicators in "
                         f"{time.time() - t_pre:.1f}s ({len(df)} bars)")
        except Exception as ex:
            log.warning(f"[RPDE_SCANNER] Failed to pre-compute indicators: {ex}")
            snapshot_ok = False

    # Scan every bar (except the last FORWARD_LOOK_BARS where we can't look ahead)
    # Also skip the first 50 bars for indicator warmup
    warmup = 50
    scan_end = len(df) - FORWARD_LOOK_BARS

    buy_count = 0
    sell_count = 0
    total_move_pips = 0.0
    golden_moments = []
    bars_filtered = 0
    bars_low_confluence = 0
    last_moment_bar = -MIN_BAR_SEPARATION - 1  # Enforce separation
    non_golden_bar_count = 0
    negative_samples = []  # collected during scan for DB storage

    log.info(f"[RPDE_SCANNER] Scanning bars {warmup} to {scan_end - 1} "
             f"({scan_end - warmup} bars to evaluate)...")

    for i in range(warmup, scan_end):
        # Progress logging every 1000 bars
        if (i - warmup) % 1000 == 0 and (i - warmup) > 0:
            pct = (i - warmup) / (scan_end - warmup) * 100
            log.info(f"[RPDE_SCANNER] {pair} progress: {pct:.0f}% "
                     f"(bar {i}/{scan_end}, {buy_count + sell_count} moments found)")

        # Enforce minimum separation between golden moments
        if (i - last_moment_bar) < MIN_BAR_SEPARATION:
            continue

        # v5.1: Regime filter — skip bars in dead zones, wrong sessions, low vol
        bar_time = df['time'].iloc[i]
        atr_pct = df['atr_percentile'].iloc[i] if 'atr_percentile' in df.columns else None
        allowed, filter_reason = _is_bar_allowed(pair, bar_time, atr_pct)
        if not allowed:
            bars_filtered += 1
            continue

        # Look forward for significant move
        result = _look_forward(
            closes, highs, lows, i,
            FORWARD_LOOK_BARS, threshold_pips, pip_value
        )

        if result['direction'] is None:
            # Non-golden bar that passed regime filter — candidate for negative sampling
            non_golden_bar_count += 1
            if non_golden_bar_count % NEGATIVE_SAMPLE_INTERVAL == 0 and snapshot_ok:
                try:
                    # Compute actual best forward move (even though below threshold)
                    neg_end_idx = min(i + FORWARD_LOOK_BARS, len(closes))
                    neg_forward_highs = highs[i:neg_end_idx]
                    neg_forward_lows = lows[i:neg_end_idx]
                    neg_entry_close = closes[i]
                    neg_buy_move = float(np.max(neg_forward_highs)) - neg_entry_close
                    neg_sell_move = neg_entry_close - float(np.min(neg_forward_lows))
                    neg_best_move_pips = max(neg_buy_move, neg_sell_move) / pip_value if pip_value > 0 else 0.0

                    # Compute actual forward return (best_move / ATR)
                    neg_atr_val = df['atr'].iloc[i] if 'atr' in df.columns else 0.0
                    neg_forward_return = 0.0
                    if neg_atr_val > 0 and pip_value > 0:
                        neg_atr_pips = neg_atr_val / pip_value
                        neg_forward_return = round(neg_best_move_pips / neg_atr_pips, 2) if neg_atr_pips > 0 else 0.0

                    # Extract feature snapshot at this bar
                    neg_snapshot = extract_snapshot_at_index(pair, i, df)
                    if neg_snapshot:
                        neg_sample = {
                            'scan_id': scan_id,
                            'pair': pair,
                            'bar_time': df['time'].iloc[i],
                            'bar_index': i,
                            'direction': 'NONE',
                            'entry_price': round(float(closes[i]), 6),
                            'move_pips': round(neg_best_move_pips, 1),
                            'peak_price': round(float(np.max(neg_forward_highs)) if neg_buy_move >= neg_sell_move else float(np.min(neg_forward_lows)), 6),
                            'peak_bar_offset': 0,
                            'forward_return': neg_forward_return,
                            'mae_pips': 0.0,
                            'mae_bars': 0,
                            'mae_price': closes[i],
                            'mfe_after_mae': 0.0,
                            'atr': round(float(neg_atr_val), 6),
                            'spread': float(df['spread'].iloc[i]) if 'spread' in df.columns else 0.0,
                            'volume': 0,
                            'pip_value': pip_value,
                            'threshold_pips': threshold_pips,
                            'feature_snapshot': neg_snapshot,
                            'session': neg_snapshot.get('_meta', {}).get('session', ''),
                            'market_state': neg_snapshot.get('_meta', {}).get('market_state', ''),
                        }
                        negative_samples.append(neg_sample)
                except Exception:
                    pass
            continue

        direction = result['direction']
        move_pips = result['move_pips']

        # Compute forward return (move_pips / ATR)
        atr_val = df['atr'].iloc[i] if 'atr' in df.columns else 0.0
        if atr_val > 0 and pip_value > 0:
            atr_pips = atr_val / pip_value
            forward_return = round(move_pips / atr_pips, 2) if atr_pips > 0 else 0.0
        else:
            forward_return = 0.0

        # Build the golden moment record
        moment = {
            'scan_id': scan_id,
            'pair': pair,
            'bar_time': df['time'].iloc[i],
            'bar_index': i,
            'direction': direction,
            'entry_price': round(float(closes[i]), 6),
            'move_pips': move_pips,
            'peak_price': result['peak_price'],
            'peak_bar_offset': result['peak_bar_offset'],
            'forward_return': forward_return,
            'mae_pips': result['mae_pips'],
            'mae_bars': result['mae_bars'],
            'mae_price': result['mae_price'],
            'mfe_after_mae': result['mfe_after_mae'],
            'atr': round(float(atr_val), 6),
            'spread': float(df['spread'].iloc[i]) if 'spread' in df.columns else 0.0,
            'volume': int(df['volume'].iloc[i]) if 'volume' in df.columns else 0,
            'pip_value': pip_value,
            'threshold_pips': threshold_pips,
        }

        # Extract 93-feature snapshot at this bar
        if snapshot_ok:
            try:
                snapshot = extract_snapshot_at_index(pair, i, df)
                if snapshot:
                    moment['feature_snapshot'] = snapshot
                    moment['session'] = snapshot.get('_meta', {}).get('session', '')
                    moment['market_state'] = snapshot.get('_meta', {}).get('market_state', '')
            except Exception as ex:
                log.debug(f"[RPDE_SCANNER] Snapshot failed at bar {i}: {ex}")

        # v5.1: Institutional confluence scoring — filter low-confluence moments
        confluence = _score_confluence(moment)
        moment['confluence_score'] = confluence
        if confluence < CONFLUENCE_MIN_SCORE:
            bars_low_confluence += 1
            continue

        golden_moments.append(moment)
        last_moment_bar = i

        if direction == 'BUY':
            buy_count += 1
        else:
            sell_count += 1
        total_move_pips += move_pips

    # ── Compute stats ──
    total_moments = buy_count + sell_count
    avg_move = round(total_move_pips / total_moments, 1) if total_moments > 0 else 0.0
    bars_scanned = scan_end - warmup

    # ── Store negative samples to database ──
    if _DB_AVAILABLE and negative_samples:
        # Cap at NEGATIVE_SAMPLE_RATIO * golden_moments_count
        max_negatives = min(
            total_moments * NEGATIVE_SAMPLE_RATIO if total_moments > 0 else NEGATIVE_MAX_SAMPLES_PER_PAIR,
            NEGATIVE_MAX_SAMPLES_PER_PAIR,
            len(negative_samples)
        )
        if max_negatives > 0:
            if max_negatives < len(negative_samples):
                import random
                random.shuffle(negative_samples)
                negative_samples = negative_samples[:max_negatives]

            neg_stored = 0
            for neg in negative_samples:
                try:
                    store_golden_moment(neg)  # reuse same store function with direction='NONE'
                    neg_stored += 1
                except Exception:
                    pass
            log.info(f"[RPDE_SCANNER] Stored {neg_stored} negative samples for {pair}")

    # ── Store results to database ──
    if _DB_AVAILABLE:
        try:
            # Store scan history record
            store_scan_history(
                scan_id=scan_id,
                pairs_scanned=1,
                days=days,
                bars_scanned=bars_scanned,
                golden_moments=total_moments,
                buy_moments=buy_count,
                sell_moments=sell_count,
                avg_move_pips=avg_move,
            )

            # Store each golden moment
            for moment in golden_moments:
                try:
                    store_golden_moment(moment)
                except Exception as ex:
                    log.debug(f"[RPDE_SCANNER] Failed to store moment: {ex}")

            log.info(f"[RPDE_SCANNER] Stored {total_moments} golden moments "
                     f"to database for {pair}")

        except Exception as ex:
            log.warning(f"[RPDE_SCANNER] Database storage failed: {ex}")
            log.warning("[RPDE_SCANNER] Golden moments retained in-memory only")

    duration = round(time.time() - t0, 1)

    log.info(f"[RPDE_SCANNER] ──── {pair} scan complete ────")
    log.info(f"[RPDE_SCANNER]   Bars scanned:    {bars_scanned:,}")
    log.info(f"[RPDE_SCANNER]   Bars filtered (regime):  {bars_filtered}")
    log.info(f"[RPDE_SCANNER]   Non-golden bars (passed filter): {non_golden_bar_count:,}")
    log.info(f"[RPDE_SCANNER]   Negative samples collected:  {len(negative_samples)}")
    log.info(f"[RPDE_SCANNER]   Bars filtered (confluence): {bars_low_confluence}")
    log.info(f"[RPDE_SCANNER]   Golden moments:  {total_moments}")
    log.info(f"[RPDE_SCANNER]   BUY moments:     {buy_count}")
    log.info(f"[RPDE_SCANNER]   SELL moments:    {sell_count}")
    log.info(f"[RPDE_SCANNER]   Avg move:        {avg_move} pips")
    log.info(f"[RPDE_SCANNER]   Duration:        {duration}s")

    return {
        'pair': pair,
        'bars_scanned': bars_scanned,
        'golden_moments': total_moments,
        'buy_moments': buy_count,
        'sell_moments': sell_count,
        'avg_move_pips': avg_move,
        'scan_id': scan_id,
        'duration_seconds': duration,
        'bars_filtered': bars_filtered,
        'bars_low_confluence': bars_low_confluence,
        'negative_samples_collected': len(negative_samples),
        'non_golden_bars': non_golden_bar_count,
        'moments': golden_moments if not _DB_AVAILABLE else [],
    }


# ════════════════════════════════════════════════════════════════
# BATCH: Scan All Pairs
# ════════════════════════════════════════════════════════════════

def scan_all_pairs(days: int = 360) -> dict:
    """
    Scan all pairs from config.settings.PAIR_WHITELIST for golden moments.

    Args:
        days: Number of days of history to scan per pair

    Returns:
        dict with:
            - scan_id: str (batch scan ID)
            - total_pairs: int
            - total_moments: int
            - per_pair: {pair: result_dict}
            - duration_seconds: int
            - errors: list of pairs that failed
    """
    t0 = time.time()
    batch_scan_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    log.info(f"[RPDE_SCANNER] ╔══════════════════════════════════════════╗")
    log.info(f"[RPDE_SCANNER] ║  RPDE BATCH SCAN STARTED                ║")
    log.info(f"[RPDE_SCANNER] ║  Pairs: {len(PAIR_WHITELIST):<33}║")
    log.info(f"[RPDE_SCANNER] ║  Days:  {days:<33}║")
    log.info(f"[RPDE_SCANNER] ║  ID:    {batch_scan_id:<33}║")
    log.info(f"[RPDE_SCANNER] ╚══════════════════════════════════════════╝")

    # Verify MT5 connection
    if not mt5.initialize():
        error_code, error_msg = mt5.last_error()
        log.error(f"[RPDE_SCANNER] MT5 initialization failed: "
                  f"[{error_code}] {error_msg}")
        return {
            'scan_id': batch_scan_id,
            'total_pairs': 0,
            'total_moments': 0,
            'per_pair': {},
            'duration_seconds': round(time.time() - t0, 1),
            'errors': ['MT5 initialization failed'],
        }

    # Check terminal data availability
    terminal_info = mt5.terminal_info()
    if terminal_info is None:
        log.warning("[RPDE_SCANNER] MT5 terminal_info returned None — "
                    "broker may be disconnected")

    per_pair = {}
    total_moments = 0
    errors = []

    for idx, pair in enumerate(PAIR_WHITELIST):
        try:
            log.info(f"[RPDE_SCANNER] ── [{idx + 1}/{len(PAIR_WHITELIST)}] {pair} ──")
            result = scan_pair(pair, days=days, scan_id=f"{batch_scan_id}_{pair}")
            per_pair[pair] = result
            total_moments += result.get('golden_moments', 0)

        except Exception as ex:
            log.error(f"[RPDE_SCANNER] Failed to scan {pair}: {ex}")
            errors.append({'pair': pair, 'error': str(ex)})
            per_pair[pair] = {
                'pair': pair,
                'bars_scanned': 0,
                'golden_moments': 0,
                'buy_moments': 0,
                'sell_moments': 0,
                'avg_move_pips': 0.0,
                'scan_id': batch_scan_id,
                'error': str(ex),
            }

    duration = round(time.time() - t0, 1)

    # ── Summary ──
    successful_pairs = [p for p, r in per_pair.items()
                        if r.get('golden_moments', 0) > 0]

    log.info(f"[RPDE_SCANNER] ╔══════════════════════════════════════════╗")
    log.info(f"[RPDE_SCANNER] ║  RPDE BATCH SCAN COMPLETE               ║")
    log.info(f"[RPDE_SCANNER] ║  Pairs scanned:   {len(PAIR_WHITELIST):<26}║")
    log.info(f"[RPDE_SCANNER] ║  Pairs with hits: {len(successful_pairs):<26}║")
    log.info(f"[RPDE_SCANNER] ║  Total moments:   {total_moments:<26,}║")
    log.info(f"[RPDE_SCANNER] ║  Errors:          {len(errors):<26}║")
    log.info(f"[RPDE_SCANNER] ║  Duration:        {duration:<22.1f}s║")
    log.info(f"[RPDE_SCANNER] ╚══════════════════════════════════════════╝")

    if successful_pairs:
        log.info(f"[RPDE_SCANNER] Top pairs by moments:")
        sorted_pairs = sorted(
            [(p, r) for p, r in per_pair.items() if r.get('golden_moments', 0) > 0],
            key=lambda x: x[1].get('golden_moments', 0),
            reverse=True
        )
        for pair, result in sorted_pairs:
            log.info(f"[RPDE_SCANNER]   {pair:<10} "
                     f"{result.get('golden_moments', 0):>5} moments "
                     f"(BUY:{result.get('buy_moments', 0)} "
                     f"SELL:{result.get('sell_moments', 0)} "
                     f"avg:{result.get('avg_move_pips', 0)}pips)")

    return {
        'scan_id': batch_scan_id,
        'total_pairs': len(PAIR_WHITELIST),
        'total_moments': total_moments,
        'per_pair': per_pair,
        'duration_seconds': duration,
        'errors': errors,
    }


# ════════════════════════════════════════════════════════════════
# HELPER: Compute ATR Series on DataFrame
# ════════════════════════════════════════════════════════════════

def _score_confluence(moment: dict) -> int:
    """
    Score a golden moment's institutional confluence (0-6).
    
    Each criterion is worth 1 point:
    +1  Strong order flow (delta > threshold)
    +1  Order flow imbalance (imbalance > threshold)
    +1  Clear market structure (trend direction exists)
    +1  Near supply/demand zone (pd_zone < threshold)
    +1  Above-average volatility (atr_percentile > threshold)
    +1  Tier 1 session (London Open, L-NY Overlap, NY Open)
    
    Returns:
        int: Confluence score (0-6)
    """
    score = 0
    features = moment.get('feature_snapshot', {})
    if not features:
        return 0
    
    meta = features.get('_meta', {})
    
    # 1. Order flow delta
    delta = features.get('of_delta')
    if delta is not None and abs(delta) > CONFLUENCE_THRESHOLD_ORDER_FLOW_DELTA:
        score += 1
    
    # 2. Order flow imbalance
    imbalance = features.get('of_imbalance')
    if imbalance is not None and abs(imbalance) > CONFLUENCE_THRESHOLD_ORDER_FLOW_IMBALANCE:
        score += 1
    
    # 3. Market structure trend
    structure_trend = features.get('smc_structure_trend')
    if structure_trend is not None and structure_trend != 0:
        score += 1
    
    # 4. Near PD zone
    pd_zone = features.get('smc_pd_zone')
    if pd_zone is not None and abs(pd_zone) < CONFLUENCE_THRESHOLD_PD_ZONE:
        score += 1
    
    # 5. ATR percentile
    atr_pct = features.get('ap_atr_percentile')
    if atr_pct is not None and atr_pct > CONFLUENCE_THRESHOLD_ATR_PERCENTILE:
        score += 1
    
    # 6. Tier 1 session
    session = meta.get('session', '')
    if session in CONFLUENCE_TIER1_SESSIONS:
        score += 1
    
    return score


def _compute_atr_series(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Compute ATR(14) and add as 'atr' column to DataFrame.
    Mirrors the ATR calculation used in backtest/data_loader.py.
    """
    h = df['high']
    l = df['low']
    c = df['close']

    hl = h - l
    hc = (h - c.shift()).abs()
    lc = (l - c.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['atr'] = tr.ewm(com=period - 1, min_periods=period).mean()

    return df


# ════════════════════════════════════════════════════════════════
# ENTRY POINT (standalone execution)
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # Allow running: python -m rpde.scanner [PAIR] [DAYS]
    pair_arg = sys.argv[1] if len(sys.argv) > 1 else None
    days_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 360

    if pair_arg:
        # Scan single pair
        result = scan_pair(pair_arg.upper(), days=days_arg)
        print(f"\n{'='*50}")
        print(f"  SCAN RESULTS: {result['pair']}")
        print(f"{'='*50}")
        print(f"  Bars scanned:    {result['bars_scanned']:,}")
        print(f"  Golden moments:  {result['golden_moments']}")
        print(f"  BUY moments:     {result['buy_moments']}")
        print(f"  SELL moments:    {result['sell_moments']}")
        print(f"  Avg move:        {result['avg_move_pips']} pips")
        print(f"  Duration:        {result['duration_seconds']}s")
        print(f"{'='*50}")
    else:
        # Scan all pairs
        result = scan_all_pairs(days=days_arg)
        print(f"\n{'='*50}")
        print(f"  BATCH SCAN RESULTS")
        print(f"{'='*50}")
        print(f"  Total pairs:     {result['total_pairs']}")
        print(f"  Total moments:   {result['total_moments']:,}")
        print(f"  Duration:        {result['duration_seconds']}s")
        if result['errors']:
            print(f"  Errors:          {len(result['errors'])}")
            for err in result['errors']:
                print(f"    - {err['pair']}: {err['error']}")
        print(f"{'='*50}")
