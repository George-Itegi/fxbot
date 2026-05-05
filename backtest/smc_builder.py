# =============================================================
# backtest/smc_builder.py
# Builds SMC (Smart Money Concepts) report from historical candles.
# Detects: structure (HH/HL/LH/LL), BOS/CHOCH, Order Blocks,
#          FVGs, sweeps, premium/discount, HTF alignment.
# Simplified version of the live smc_scanner for backtesting.
# =============================================================

import numpy as np
import pandas as pd
from core.logger import get_logger

log = get_logger(__name__)


def build_smc_report(df_h1: pd.DataFrame,
                     df_h4: pd.DataFrame,
                     current_time=None) -> dict:
    """
    Build a simplified SMC report from historical H1 and H4 candles.
    Matches the schema expected by strategies.
    """
    if df_h1 is None or df_h4 is None or len(df_h1) < 50 or len(df_h4) < 20:
        return _empty_smc()

    structure = _detect_structure(df_h1)

    # Order blocks — find recent bullish/bearish OBs on H1
    nearest_ob = _find_nearest_ob(df_h1, current_price=float(df_h1.iloc[-1]['close']))

    # Sweeps — find recent liquidity sweeps
    last_sweep = _find_last_sweep(df_h1)

    # FVGs
    nearest_fvg = _find_nearest_fvg(df_h1, float(df_h1.iloc[-1]['close']))
    quality_fvgs = _find_quality_fvgs(df_h1, float(df_h1.iloc[-1]['close']))

    # Premium/Discount
    pd_info = _calc_premium_discount(df_h1)

    # HTF Alignment (H4)
    htf = _calc_htf_alignment(df_h4)

    # SMC score
    smc_score = _calc_smc_score(structure, nearest_ob, last_sweep,
                                 quality_fvgs, htf)

    # SMC bias
    smc_bias = structure.get('trend', 'NEUTRAL')

    # Liquidity pool
    nearest_pool = _find_nearest_pool(df_h1, float(df_h1.iloc[-1]['close']))

    return {
        'structure': structure,
        'nearest_ob': nearest_ob,
        'last_sweep': last_sweep,
        'nearest_fvg': nearest_fvg,
        'quality_fvgs': quality_fvgs,
        'nearest_pool': nearest_pool,
        'premium_discount': pd_info,
        'htf_alignment': htf,
        'smc_score': smc_score,
        'smc_bias': smc_bias,
    }


def _detect_structure(df_h1: pd.DataFrame) -> dict:
    """
    Detect market structure: Higher Highs, Higher Lows, etc.
    Uses last 50 H1 candles.
    """
    recent = df_h1.tail(50)

    highs = recent['high'].values
    lows  = recent['low'].values
    closes = recent['close'].values
    n = len(highs)

    if n < 10:
        return {'trend': 'NEUTRAL', 'hh_count': 0, 'hl_count': 0,
                'lh_count': 0, 'll_count': 0, 'bos': None, 'choch': None}

    # Count swing points (local highs/lows over 5-bar window)
    lookback = 5
    swing_highs = []
    swing_lows = []

    for i in range(lookback, n - lookback):
        if highs[i] == max(highs[i-lookback:i+lookback+1]):
            swing_highs.append(highs[i])
        if lows[i] == min(lows[i-lookback:i+lookback+1]):
            swing_lows.append(lows[i])

    # Count HH/HL/LH/LL
    hh = hl = lh = ll = 0
    for i in range(1, len(swing_highs)):
        if swing_highs[i] > swing_highs[i-1]:
            hh += 1
        else:
            lh += 1

    for i in range(1, len(swing_lows)):
        if swing_lows[i] > swing_lows[i-1]:
            hl += 1
        else:
            ll += 1

    # Determine trend
    if hh >= 2 and hl >= 2:
        trend = 'BULLISH'
    elif lh >= 2 and ll >= 2:
        trend = 'BEARISH'
    elif hh > lh:
        trend = 'BULLISH'
    elif lh > hh:
        trend = 'BEARISH'
    else:
        trend = 'RANGING'

    # BOS (Break of Structure) — check if recent price broke recent swing
    bos = None
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        last_swing_high = swing_highs[-1]
        last_swing_low = swing_lows[-1]
        prev_swing_high = swing_highs[-2] if len(swing_highs) >= 2 else None
        prev_swing_low = swing_lows[-2] if len(swing_lows) >= 2 else None

        current_price = closes[-1]

        if trend == 'BULLISH' and prev_swing_high:
            if current_price > prev_swing_high:
                bos = {'type': 'BULLISH_BOS', 'level': round(prev_swing_high, 5)}
        elif trend == 'BEARISH' and prev_swing_low:
            if current_price < prev_swing_low:
                bos = {'type': 'BEARISH_BOS', 'level': round(prev_swing_low, 5)}

    # CHOCH (Change of Character)
    choch = None
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        if trend == 'BEARISH' and hh > 0:
            choch = {'type': 'BULLISH_CHOCH', 'level': round(swing_highs[-1], 5)}
        elif trend == 'BULLISH' and ll > 0:
            choch = {'type': 'BEARISH_CHOCH', 'level': round(swing_lows[-1], 5)}

    return {
        'trend': trend,
        'hh_count': hh,
        'hl_count': hl,
        'lh_count': lh,
        'll_count': ll,
        'bos': bos,
        'choch': choch,
    }


def _find_nearest_ob(df_h1: pd.DataFrame, current_price: float) -> dict:
    """Find the nearest unmitigated order block."""
    recent = df_h1.tail(50)
    pip_size = _guess_pip_size(current_price)

    for i in range(len(recent) - 1, max(0, len(recent) - 50), -1):
        bar = recent.iloc[i]
        body = bar['close'] - bar['open']
        bar_range = bar['high'] - bar['low']

        if bar_range == 0:
            continue

        # Bullish OB: big green candle, followed by bearish move down
        if body > bar_range * 0.5:  # Strong bullish candle
            ob_top = bar['close']
            ob_bottom = bar['open']
            ob_mid = (ob_top + ob_bottom) / 2

            # Check if price has returned to this zone (mitigated)
            if current_price >= ob_bottom - pip_size * 2:
                distance = abs(current_price - ob_mid) / pip_size
                if distance <= 80:  # Within 80 pips
                    return {
                        'type': 'BULLISH_OB',
                        'top': round(ob_top, 5),
                        'bottom': round(ob_bottom, 5),
                        'mid': round(ob_mid, 5),
                        'pips_away': round(distance, 1),
                        'mitigated': False,
                    }

        # Bearish OB: big red candle, followed by bullish move up
        elif body < -bar_range * 0.5:
            ob_top = bar['open']
            ob_bottom = bar['close']
            ob_mid = (ob_top + ob_bottom) / 2

            if current_price <= ob_top + pip_size * 2:
                distance = abs(current_price - ob_mid) / pip_size
                if distance <= 80:
                    return {
                        'type': 'BEARISH_OB',
                        'top': round(ob_top, 5),
                        'bottom': round(ob_bottom, 5),
                        'mid': round(ob_mid, 5),
                        'pips_away': round(distance, 1),
                        'mitigated': False,
                    }

    return {}


def _find_last_sweep(df_h1: pd.DataFrame) -> dict:
    """
    Find the most recent liquidity sweep using proper swing point detection.
    A sweep occurs when price pierces a swing high/low then reverses.

    v2: Added bars_since_sweep (recency) and follow_through confirmation.
    Only returns sweeps where the next H1 candle continued the reversal.
    """
    recent = df_h1.tail(50)
    if len(recent) < 20:
        return {}

    pip_size = _guess_pip_size(float(recent.iloc[-1]['close']))
    highs = recent['high'].values
    lows = recent['low'].values
    closes = recent['close'].values
    n = len(highs)
    last_bar_idx = n - 1

    # Detect proper swing points (local extrema with lookback=5)
    swing_length = 5
    swing_highs = []  # (index, price)
    swing_lows = []

    for i in range(swing_length, n - swing_length):
        if highs[i] == max(highs[i - swing_length:i + swing_length + 1]):
            swing_highs.append((i, highs[i]))
        if lows[i] == min(lows[i - swing_length:i + swing_length + 1]):
            swing_lows.append((i, lows[i]))

    if not swing_highs and not swing_lows:
        return {}

    best_sweep = None
    sweep_min_reversal = 3.0  # Minimum 3 pips reversal
    max_bars_since = 3  # Sweep must be within 3 H1 bars (hours) to be actionable

    # Check for HIGH sweeps (price pierced above swing high, then reversed)
    for si, (sidx, sprice) in enumerate(swing_highs):
        if si >= len(swing_highs) - 1:
            break
        # Look at bars AFTER this swing high until the next swing high
        next_si = si + 1
        end_idx = swing_highs[next_si][0] if next_si < len(swing_highs) else n
        for j in range(sidx + 1, min(end_idx, n)):
            if highs[j] > sprice:
                # Pierced above swing high — check for reversal
                reversal_pips = (highs[j] - closes[j]) / pip_size
                if reversal_pips >= sweep_min_reversal and closes[j] < sprice:
                    # Recency check: how many bars since this sweep?
                    bars_since = last_bar_idx - j
                    if bars_since > max_bars_since:
                        break  # Too old, skip this swing high entirely

                    # Follow-through check: next bar must continue bearish
                    follow_through = False
                    if j + 1 < n:
                        next_close = closes[j + 1]
                        # For bearish sweep: next close should be below sweep close
                        if next_close < closes[j]:
                            follow_through = True
                    else:
                        # Last bar in window — accept without follow-through
                        follow_through = True

                    if follow_through and (best_sweep is None or j > best_sweep[5]):
                        best_sweep = (sidx, sprice, 'HIGH_SWEEP', 'BEARISH',
                                      reversal_pips, j, bars_since, follow_through)
                    break
            elif closes[j] > sprice * 1.003:
                # Price moved above and stayed — not a sweep, break
                break

    # Check for LOW sweeps (price pierced below swing low, then reversed)
    for si, (sidx, sprice) in enumerate(swing_lows):
        if si >= len(swing_lows) - 1:
            break
        next_si = si + 1
        end_idx = swing_lows[next_si][0] if next_si < len(swing_lows) else n
        for j in range(sidx + 1, min(end_idx, n)):
            if lows[j] < sprice:
                reversal_pips = (closes[j] - lows[j]) / pip_size
                if reversal_pips >= sweep_min_reversal and closes[j] > sprice:
                    # Recency check
                    bars_since = last_bar_idx - j
                    if bars_since > max_bars_since:
                        break

                    # Follow-through check: next bar must continue bullish
                    follow_through = False
                    if j + 1 < n:
                        next_close = closes[j + 1]
                        # For bullish sweep: next close should be above sweep close
                        if next_close > closes[j]:
                            follow_through = True
                    else:
                        follow_through = True

                    if follow_through and (best_sweep is None or j > best_sweep[5]):
                        best_sweep = (sidx, sprice, 'LOW_SWEEP', 'BULLISH',
                                      reversal_pips, j, bars_since, follow_through)
                    break
            elif closes[j] < sprice * 0.997:
                break

    if best_sweep:
        sidx, sprice, stype, bias, rev_pips, sweep_bar, bars_since, ft = best_sweep
        return {
            'type': stype,
            'swept_level': round(sprice, 5),
            'bias': bias,
            'reversal_pips': round(rev_pips, 1),
            'time': str(recent.iloc[sweep_bar].get('time', '')),
            'bars_since_sweep': bars_since,
            'follow_through': ft,
        }

    return {}


def _find_nearest_fvg(df_h1, current_price):
    """Find nearest Fair Value Gap."""
    pip_size = _guess_pip_size(current_price)
    recent = df_h1.tail(30)

    best_fvg = None
    best_dist = 999

    for i in range(2, len(recent)):
        bar1 = recent.iloc[i-2]
        bar2 = recent.iloc[i-1]
        bar3 = recent.iloc[i]

        # Bullish FVG: gap between bar1 high and bar3 low
        if bar3['low'] > bar1['high']:
            gap = bar3['low'] - bar1['high']
            gap_pips = gap / pip_size
            if gap_pips >= 2.5:  # Minimum 2.5 pips (match strategy)
                mid = (bar1['high'] + bar3['low']) / 2
                distance = abs(current_price - mid) / pip_size
                if distance <= 100 and distance < best_dist:
                    # Check fill: scan ALL subsequent bars (not just bar2)
                    filled = _check_fvg_filled(recent, i, 'BULLISH', bar1['high'])
                    quality = min(100, int(gap_pips * 2) + (20 if distance < 20 else 10 if distance < 50 else 0) + (30 if not filled else 0))
                    best_fvg = {
                        'type': 'BULLISH_FVG',
                        'bottom': round(bar1['high'], 5),
                        'top': round(bar3['low'], 5),
                        'mid': round(mid, 5),
                        'gap_pips': round(gap_pips, 1),
                        'filled': filled,
                        'quality_score': quality,
                    }
                    best_dist = distance

        # Bearish FVG: gap between bar3 high and bar1 low
        elif bar1['low'] > bar3['high']:
            gap = bar1['low'] - bar3['high']
            gap_pips = gap / pip_size
            if gap_pips >= 2.5:
                mid = (bar3['high'] + bar1['low']) / 2
                distance = abs(current_price - mid) / pip_size
                if distance <= 100 and distance < best_dist:
                    filled = _check_fvg_filled(recent, i, 'BEARISH', bar1['low'])
                    quality = min(100, int(gap_pips * 2) + (20 if distance < 20 else 10 if distance < 50 else 0) + (30 if not filled else 0))
                    best_fvg = {
                        'type': 'BEARISH_FVG',
                        'bottom': round(bar3['high'], 5),
                        'top': round(bar1['low'], 5),
                        'mid': round(mid, 5),
                        'gap_pips': round(gap_pips, 1),
                        'filled': filled,
                        'quality_score': quality,
                    }
                    best_dist = distance

    return best_fvg if best_fvg else {}


def _find_quality_fvgs(df_h1, current_price):
    """Find all quality FVGs (gap >= 2.5 pips, unfilled, within 100 pips).
    Quality scoring now matches live system: size + freshness + fill status."""
    pip_size = _guess_pip_size(current_price)
    recent = df_h1.tail(50)
    fvgs = []

    for i in range(2, len(recent)):
        bar1 = recent.iloc[i-2]
        bar2 = recent.iloc[i-1]
        bar3 = recent.iloc[i]

        if bar3['low'] > bar1['high']:
            gap = bar3['low'] - bar1['high']
            gap_pips = gap / pip_size
            if gap_pips >= 2.5:  # Lowered from 5 to match strategy MIN_FVG_SIZE_PIPS
                mid = (bar1['high'] + bar3['low']) / 2
                dist = abs(current_price - mid) / pip_size
                if dist <= 100:  # Widened from 80
                    # Multi-factor quality score (matches live system)
                    filled = _check_fvg_filled(recent, i, 'BULLISH', bar1['high'])
                    quality = int(gap_pips * 2)  # Size component
                    # Freshness bonus — closer FVGs are more relevant
                    if dist < 20:
                        quality += 20
                    elif dist < 50:
                        quality += 10
                    # Not filled bonus — unfilled gaps are more likely to fill
                    if not filled:
                        quality += 30
                    quality = min(100, quality)

                    fvgs.append({
                        'type': 'BULLISH_FVG',
                        'bottom': round(bar1['high'], 5),
                        'top': round(bar3['low'], 5),
                        'mid': round(mid, 5),
                        'gap_pips': round(gap_pips, 1),
                        'quality_score': quality,
                        'filled': filled,
                    })
        elif bar1['low'] > bar3['high']:
            gap = bar1['low'] - bar3['high']
            gap_pips = gap / pip_size
            if gap_pips >= 2.5:
                mid = (bar3['high'] + bar1['low']) / 2
                dist = abs(current_price - mid) / pip_size
                if dist <= 100:
                    filled = _check_fvg_filled(recent, i, 'BEARISH', bar1['low'])
                    quality = int(gap_pips * 2)
                    if dist < 20:
                        quality += 20
                    elif dist < 50:
                        quality += 10
                    if not filled:
                        quality += 30
                    quality = min(100, quality)

                    fvgs.append({
                        'type': 'BEARISH_FVG',
                        'bottom': round(bar3['high'], 5),
                        'top': round(bar1['low'], 5),
                        'mid': round(mid, 5),
                        'gap_pips': round(gap_pips, 1),
                        'quality_score': quality,
                        'filled': filled,
                    })

    return fvgs


def _check_fvg_filled(recent, fvg_idx, fvg_type, level):
    """
    Check if an FVG has been filled by scanning ALL subsequent bars.
    Matches live system behavior (not just the middle bar).
    """
    for j in range(fvg_idx + 1, len(recent)):
        bar = recent.iloc[j]
        if fvg_type == 'BULLISH':
            # Bullish FVG filled when price drops into the gap
            if bar['low'] <= level:
                return True
        elif fvg_type == 'BEARISH':
            # Bearish FVG filled when price rises into the gap
            if bar['high'] >= level:
                return True
    return False


def _find_nearest_pool(df_h1, current_price):
    """Find nearest liquidity pool (cluster of equal highs/lows)."""
    recent = df_h1.tail(50)
    pip_size = _guess_pip_size(current_price)

    # Find clusters of similar highs (within 3 pips)
    highs = recent['high'].values
    for i in range(len(highs) - 1, max(0, len(highs) - 20), -1):
        level = highs[i]
        touches = sum(1 for h in highs if abs(h - level) / pip_size < 3)
        if touches >= 3:
            distance = abs(current_price - level) / pip_size
            return {
                'type': 'HIGH_POOL',
                'level': round(level, 5),
                'touches': touches,
                'pips_away': round(distance, 1),
                'swept': current_price > level,
            }

    # Check lows
    lows = recent['low'].values
    for i in range(len(lows) - 1, max(0, len(lows) - 20), -1):
        level = lows[i]
        touches = sum(1 for l in lows if abs(l - level) / pip_size < 3)
        if touches >= 3:
            distance = abs(current_price - level) / pip_size
            return {
                'type': 'LOW_POOL',
                'level': round(level, 5),
                'touches': touches,
                'pips_away': round(distance, 1),
                'swept': current_price < level,
            }

    return {}


def _calc_premium_discount(df_h1: pd.DataFrame) -> dict:
    """Calculate premium/discount zone."""
    recent = df_h1.tail(100)
    if len(recent) < 20:
        return {'zone': 'NEUTRAL', 'position_pct': 50.0, 'bias': 'NEUTRAL',
                'pips_to_eq': 0.0}

    range_high = recent['high'].max()
    range_low  = recent['low'].min()
    price_range = range_high - range_low

    if price_range == 0:
        return {'zone': 'NEUTRAL', 'position_pct': 50.0, 'bias': 'NEUTRAL',
                'pips_to_eq': 0.0}

    current = float(recent.iloc[-1]['close'])
    position_pct = ((current - range_low) / price_range) * 100

    if position_pct > 75:
        zone = 'EXTREME_PREMIUM'
        bias = 'SELL'
    elif position_pct > 55:
        zone = 'PREMIUM'
        bias = 'SELL'
    elif position_pct < 25:
        zone = 'EXTREME_DISCOUNT'
        bias = 'BUY'
    elif position_pct < 45:
        zone = 'DISCOUNT'
        bias = 'BUY'
    else:
        zone = 'NEUTRAL'
        bias = 'NEUTRAL'

    equilibrium = (range_high + range_low) / 2
    pip_size = _guess_pip_size(current)
    pips_to_eq = (equilibrium - current) / pip_size

    return {
        'zone': zone,
        'position_pct': round(position_pct, 1),
        'bias': bias,
        'pips_to_eq': round(pips_to_eq, 1),
        'range_high': round(range_high, 5),
        'range_low': round(range_low, 5),
    }


def _calc_htf_alignment(df_h4: pd.DataFrame) -> dict:
    """Calculate H4 alignment for bias confirmation."""
    if df_h4 is None or len(df_h4) < 10:
        return {'approved': True, 'h4_bias': 'NEUTRAL', 'score': 50}

    last = df_h4.iloc[-1]
    h4_bull = last['ema_9'] > last['ema_21'] > last['ema_50']
    h4_bear = last['ema_9'] < last['ema_21'] < last['ema_50']
    h4_st_bull = int(last.get('supertrend_dir', 0)) == 1
    h4_st_bear = int(last.get('supertrend_dir', 0)) == -1

    if h4_bull and h4_st_bull:
        return {'approved': True, 'h4_bias': 'BULLISH', 'score': 80}
    elif h4_bear and h4_st_bear:
        return {'approved': True, 'h4_bias': 'BEARISH', 'score': 80}
    elif h4_bull or h4_st_bull:
        return {'approved': True, 'h4_bias': 'BULLISH', 'score': 60}
    elif h4_bear or h4_st_bear:
        return {'approved': True, 'h4_bias': 'BEARISH', 'score': 60}
    else:
        return {'approved': False, 'h4_bias': 'NEUTRAL', 'score': 30}


def _calc_smc_score(structure, nearest_ob, last_sweep,
                    quality_fvgs, htf) -> int:
    """Calculate SMC score (0-100)."""
    score = 0

    # Structure quality (30 pts)
    trend = structure.get('trend', 'NEUTRAL')
    if trend in ('BULLISH', 'BEARISH'):
        score += 30
    elif trend == 'RANGING':
        score += 15

    # OB nearby (25 pts)
    if nearest_ob:
        score += 25

    # Sweep aligned (20 pts)
    if last_sweep:
        score += 20

    # FVG quality (25 pts)
    unfilled = [f for f in quality_fvgs if not f.get('filled', False)]
    if unfilled:
        best = max(unfilled, key=lambda f: f.get('quality_score', 0))
        score += min(25, best.get('quality_score', 0))

    # HTF alignment bonus
    if htf.get('approved'):
        score += 10

    return min(100, score)


def _guess_pip_size(price: float) -> float:
    """Guess pip size from price level."""
    if price > 1000:   # Gold
        return 0.1
    elif price > 500:  # Indices
        return 1.0
    elif price > 50:   # JPY pairs
        return 0.01
    else:             # Standard forex
        return 0.0001


def _empty_smc() -> dict:
    """Return empty SMC report."""
    return {
        'structure': {'trend': 'NEUTRAL', 'hh_count': 0, 'hl_count': 0,
                      'lh_count': 0, 'll_count': 0, 'bos': None, 'choch': None},
        'nearest_ob': {},
        'last_sweep': {},
        'nearest_fvg': {},
        'quality_fvgs': [],
        'nearest_pool': {},
        'premium_discount': {'zone': 'NEUTRAL', 'position_pct': 50.0,
                             'bias': 'NEUTRAL', 'pips_to_eq': 0.0},
        'htf_alignment': {'approved': True, 'h4_bias': 'NEUTRAL', 'score': 50},
        'smc_score': 0,
        'smc_bias': 'NEUTRAL',
    }
