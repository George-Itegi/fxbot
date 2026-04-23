# =============================================================
# backtest/market_builder.py
# Builds market_report from historical candle data.
# Replaces the live market_scanner for backtesting.
# Calculates: VWAP, volume profile, trade_score, market_state.
# =============================================================

import numpy as np
import pandas as pd
from core.logger import get_logger

log = get_logger(__name__)


def build_market_report(df_m15: pd.DataFrame,
                        flow_data: dict,
                        smc_report: dict,
                        symbol: str) -> dict:
    """
    Build a market_report dict from historical M15 candles + simulated flow.
    Matches the schema expected by strategies and strategy_engine.
    """
    if df_m15 is None or len(df_m15) < 30:
        return _empty_market(symbol)

    # ── Session tag ──
    current_time = df_m15.iloc[-1]['time']
    session = _tag_session(current_time)

    # --- Combined bias (from flow data) ---
    full_bias = flow_data.get('delta', {}).get('bias', 'NEUTRAL')
    roll_bias = flow_data.get('rolling_delta', {}).get('bias', 'NEUTRAL')

    # Use rolling delta as primary bias (matches live behavior)
    if roll_bias != 'NEUTRAL':
        combined_bias = roll_bias
    else:
        combined_bias = full_bias

    # --- VWAP context ---
    vwap = _calc_vwap(df_m15)
    vwap_bias = vwap.get('bias', 'NEUTRAL')

    # Adjust combined bias with VWAP
    if vwap_bias in ('STRONG_BULL', 'BULL') and combined_bias == 'NEUTRAL':
        combined_bias = 'BULLISH'
    elif vwap_bias in ('STRONG_BEAR', 'BEAR') and combined_bias == 'NEUTRAL':
        combined_bias = 'BEARISH'

    # Volume profile (simplified)
    profile = _calc_volume_profile(df_m15)

    # Market state
    trade_score_data = _calc_tradeability_score(
        flow_data, vwap, profile, combined_bias)
    trade_score = trade_score_data['score']
    reasons = trade_score_data['reasons']
    market_state = _detect_market_state(
        trade_score, combined_bias, vwap, profile,
        flow_data.get('momentum', {}),
        flow_data.get('volume_surge', {}))

    # Bias votes
    bias_votes = 0
    if full_bias == 'BULLISH': bias_votes += 1
    elif full_bias == 'BEARISH': bias_votes -= 1
    if roll_bias == 'BULLISH': bias_votes += 1
    elif roll_bias == 'BEARISH': bias_votes -= 1
    if profile.get('bias') == 'BULLISH': bias_votes += 1
    elif profile.get('bias') == 'BEARISH': bias_votes -= 1
    if vwap_bias in ('STRONG_BULL', 'BULL'): bias_votes += 1
    elif vwap_bias in ('STRONG_BEAR', 'BEAR'): bias_votes -= 1

    report = {
        'symbol': symbol,
        'delta': flow_data.get('delta', {}),
        'rolling_delta': flow_data.get('rolling_delta', {}),
        'order_flow_imbalance': flow_data.get('order_flow_imbalance', {}),
        'volume_surge': flow_data.get('volume_surge', {}),
        'momentum': flow_data.get('momentum', {}),
        'profile': profile,
        'vwap': vwap,
        'combined_bias': combined_bias,
        'bias_votes': bias_votes,
        'trade_score': trade_score,
        'score_reasons': reasons,
        'market_state': market_state,
        'session': session,
    }

    return report


def _calc_vwap(df_m15: pd.DataFrame) -> dict:
    """Calculate VWAP from intraday M15 candles."""
    current = df_m15.iloc[-1]
    current_price = float(current['close'])
    pip_size = _guess_pip_size(current_price)

    # Reset VWAP at start of each day
    today = current['time'].date()
    today_bars = df_m15[df_m15['time'].dt.date == today]

    if len(today_bars) < 2:
        return _empty_vwap(current_price, pip_size)

    typical_price = (today_bars['high'] + today_bars['low'] +
                     today_bars['close']) / 3
    cum_tp_vol = (typical_price * today_bars['tick_volume']).cumsum()
    cum_vol = today_bars['tick_volume'].cumsum()
    vwap_val = cum_tp_vol.iloc[-1] / cum_vol.iloc[-1] if cum_vol.iloc[-1] > 0 else current_price

    # Upper/Lower bands (1 and 2 std dev)
    tp_series = (today_bars['high'] + today_bars['low'] +
                 today_bars['close']) / 3
    vwap_series = cum_tp_vol / cum_vol

    # Simplified bands: use rolling std of typical price deviations from VWAP
    deviations = tp_series - vwap_series
    if len(deviations) > 20:
        std_dev = deviations.rolling(20).std().iloc[-1]
        if np.isnan(std_dev):
            std_dev = pip_size * 5
    else:
        std_dev = pip_size * 5

    upper1 = vwap_val + std_dev
    lower1 = vwap_val - std_dev
    upper2 = vwap_val + 2 * std_dev
    lower2 = vwap_val - 2 * std_dev

    pip_from_vwap = (current_price - vwap_val) / pip_size

    # Position relative to VWAP
    if pip_from_vwap > 15:
        position = 'FAR_ABOVE'
        bias = 'STRONG_BULL'
    elif pip_from_vwap > 5:
        position = 'ABOVE'
        bias = 'BULL'
    elif pip_from_vwap < -15:
        position = 'FAR_BELOW'
        bias = 'STRONG_BEAR'
    elif pip_from_vwap < -5:
        position = 'BELOW'
        bias = 'BEAR'
    else:
        position = 'AT_VWAP'
        bias = 'NEUTRAL'

    return {
        'vwap': round(vwap_val, 5),
        'upper_band_1': round(upper1, 5),
        'lower_band_1': round(lower1, 5),
        'upper_band_2': round(upper2, 5),
        'lower_band_2': round(lower2, 5),
        'pip_from_vwap': round(pip_from_vwap, 1),
        'position': position,
        'bias': bias,
        'note': f"{'Overbought' if pip_from_vwap > 10 else 'Oversold' if pip_from_vwap < -10 else 'Fair value'}",
    }


def _calc_volume_profile(df_m15: pd.DataFrame) -> dict:
    """Calculate simplified volume profile from M15 candles."""
    current = df_m15.iloc[-1]
    current_price = float(current['close'])
    pip_size = _guess_pip_size(current_price)

    recent = df_m15.tail(200)

    # POC: price level with highest volume
    typical_prices = (recent['high'] + recent['low'] + recent['close']) / 3
    # Bin prices into 5-pip buckets
    price_range = recent['high'].max() - recent['low'].min()
    if price_range == 0:
        return _empty_profile(current_price, pip_size)

    bin_size_pips = 5
    bin_size = bin_size_pips * pip_size
    n_bins = max(1, int(price_range / bin_size))

    volume_by_bin = {}
    for _, row in recent.iterrows():
        tp = (row['high'] + row['low'] + row['close']) / 3
        bin_idx = int((tp - recent['low'].min()) / bin_size)
        if bin_idx not in volume_by_bin:
            volume_by_bin[bin_idx] = 0
        volume_by_bin[bin_idx] += row['tick_volume']

    if not volume_by_bin:
        return _empty_profile(current_price, pip_size)

    # POC = bin with highest volume
    poc_bin = max(volume_by_bin, key=volume_by_bin.get)
    poc_price = recent['low'].min() + (poc_bin + 0.5) * bin_size

    # Value Area (70% of volume)
    total_vol = sum(volume_by_bin.values())
    sorted_bins = sorted(volume_by_bin.items(), key=lambda x: x[1], reverse=True)
    va_vol = 0
    va_bins = []
    for bin_idx, vol in sorted_bins:
        va_vol += vol
        va_bins.append(bin_idx)
        if va_vol >= total_vol * 0.7:
            break

    vah_price = recent['low'].min() + (max(va_bins) + 1) * bin_size
    val_price = recent['low'].min() + min(va_bins) * bin_size

    pip_to_poc = (current_price - poc_price) / pip_size

    # Price position
    if current_price > vah_price:
        price_position = 'ABOVE_VAH'
    elif current_price < val_price:
        price_position = 'BELOW_VAL'
    else:
        price_position = 'INSIDE_VA'

    return {
        'current_price': round(current_price, 5),
        'poc': round(poc_price, 5),
        'vah': round(vah_price, 5),
        'val': round(val_price, 5),
        'va_width_pips': round((vah_price - val_price) / pip_size, 1),
        'pip_to_poc': round(pip_to_poc, 1),
        'price_position': price_position,
        'bias': 'BULLISH' if pip_to_poc > 5 else 'BEARISH' if pip_to_poc < -5 else 'NEUTRAL',
        'hvn_list': [],
        'lvn_list': [],
        'note': f"{'Above value' if price_position == 'ABOVE_VAH' else 'Inside value' if price_position == 'INSIDE_VA' else 'Below value'}",
    }


def _calc_tradeability_score(flow_data, vwap, profile, combined_bias):
    """Calculate trade score (0-100) from available data."""
    score = 0
    reasons = []

    delta = flow_data.get('delta', {})
    roll_delta = flow_data.get('rolling_delta', {})
    imb = flow_data.get('order_flow_imbalance', {})
    surge = flow_data.get('volume_surge', {})
    mom = flow_data.get('momentum', {})

    # Delta aligned (15 pts)
    if delta.get('bias') == roll_delta.get('bias') and delta.get('bias') != 'NEUTRAL':
        score += 15
        reasons.append(f"Delta aligned ({delta.get('bias')})")

    # Delta strength (10 pts)
    if roll_delta.get('strength') == 'STRONG':
        score += 10
        reasons.append("Strong rolling delta")
    elif roll_delta.get('strength') == 'MODERATE':
        score += 5
        reasons.append("Moderate rolling delta")

    # OF imbalance (15 pts)
    imb_str = imb.get('strength', 'NONE')
    if imb_str in ('EXTREME', 'STRONG'):
        score += 15
        reasons.append(f"Strong OF imbalance ({imb.get('imbalance', 0):+.2f})")
    elif imb_str == 'MODERATE':
        score += 7
        reasons.append(f"Moderate OF imbalance ({imb.get('imbalance', 0):+.2f})")

    # Volume surge (10 pts)
    if surge.get('surge_detected', False):
        score += 10
        reasons.append(f"Volume surge ({surge.get('surge_ratio')}x)")

    # Momentum (8 pts)
    if mom.get('is_scalpable', False):
        score += 8
        reasons.append("Momentum active")
    elif mom.get('is_choppy', True):
        score -= 5
        reasons.append("Choppy market")

    # VWAP distance (15 pts)
    pip_vwap = abs(vwap.get('pip_from_vwap', 999))
    if pip_vwap <= 10:
        score += 15
        reasons.append("Price close to VWAP")
    elif pip_vwap <= 20:
        score += 10
        reasons.append("Price near VWAP")
    elif pip_vwap <= 35:
        score += 5

    # POC distance (10 pts)
    pip_poc = abs(profile.get('pip_to_poc', 999))
    if pip_poc <= 10:
        score += 10
        reasons.append("Price at POC")
    elif pip_poc <= 30:
        score += 7

    # Delta confirms bias (5 pts)
    d_bias = delta.get('bias', 'NEUTRAL')
    if combined_bias == 'BULLISH' and d_bias == 'BULLISH':
        score += 5
        reasons.append("Delta confirms bullish")
    elif combined_bias == 'BEARISH' and d_bias == 'BEARISH':
        score += 5
        reasons.append("Delta confirms bearish")

    return {'score': max(0, min(100, score)), 'reasons': reasons}


def _detect_market_state(score, combined_bias, vwap, profile,
                          momentum, volume_surge):
    """Classify market into institutional state."""
    pip_vwap = abs(vwap.get('pip_from_vwap', 0))
    pos = profile.get('price_position', '')
    votes = 0
    if combined_bias == 'BULLISH': votes = 1
    elif combined_bias == 'BEARISH': votes = -1

    # TRENDING_EXTENDED
    if votes != 0 and pip_vwap > 25:
        return "TRENDING_EXTENDED"

    # BREAKOUT
    if pos in ('ABOVE_VAH', 'BELOW_VAL'):
        return "BREAKOUT_ACCEPTED" if votes != 0 else "BREAKOUT_REJECTED"

    # TRENDING_STRONG
    if votes != 0 and pip_vwap <= 20 and momentum.get('is_scalpable', False):
        return "TRENDING_STRONG"

    # REVERSAL_RISK
    if pip_vwap > 30:
        return "REVERSAL_RISK"

    # CHOPPY
    if momentum.get('is_choppy', False) and not volume_surge.get('surge_detected', False):
        return "BALANCED"

    return "BALANCED"


def _tag_session(dt) -> str:
    """Tag each candle with the trading session (UTC-based)."""
    try:
        hour = dt.hour
        if 21 <= hour < 24: return "SYDNEY"
        if 0  <= hour <  7:  return "TOKYO"
        if 7  <= hour <  8:  return "LONDON_OPEN"
        if 8  <= hour < 12:  return "LONDON_SESSION"
        if 12 <= hour < 16:  return "NY_LONDON_OVERLAP"
        if 16 <= hour < 21:  return "NY_AFTERNOON"
        return "SYDNEY"
    except Exception:
        return "UNKNOWN"


def _guess_pip_size(price: float) -> float:
    if price > 1000: return 0.1   # Gold
    elif price > 500: return 1.0   # Indices
    elif price > 50: return 0.01    # JPY pairs
    else: return 0.0001           # Standard forex


def _empty_vwap(price, pip_size):
    return {'vwap': price, 'upper_band_1': price, 'lower_band_1': price,
            'pip_from_vwap': 0.0, 'position': 'AT_VWAP', 'bias': 'NEUTRAL',
            'note': ''}


def _empty_profile(price, pip_size):
    return {'current_price': price, 'poc': price, 'vah': price, 'val': price,
            'va_width_pips': 0, 'pip_to_poc': 0, 'price_position': 'INSIDE_VA',
            'bias': 'NEUTRAL', 'hvn_list': [], 'lvn_list': [], 'note': ''}


def _empty_market(symbol):
    return {'symbol': symbol, 'combined_bias': 'NEUTRAL', 'bias_votes': 0,
            'trade_score': 0, 'market_state': 'BALANCED',
            'delta': {}, 'rolling_delta': {},
            'order_flow_imbalance': {}, 'volume_surge': {},
            'momentum': {}, 'profile': _empty_profile(0, 0.0001),
            'vwap': _empty_vwap(0, 0.0001), 'score_reasons': []}
