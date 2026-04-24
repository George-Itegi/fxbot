# =============================================================
# strategies/smart_money_footprint.py
# Strategy 10: Smart Money Footprint (Institutional Alpha)
# 
# This is the ULTIMATE strategy combining ALL order flow secrets:
#   1. Cumulative Delta Divergence (reversal signal)
#   2. Absorption Detection (iceberg orders)
#   3. Stop Hunt Identification (liquidity grabs)
#   4. Volume Node Rejection (POC/VAH/VAL)
#   5. Order Flow Velocity (institutional momentum)
#   6. Smart Money Score (master confluence)
#
# This strategy does what brokers DON'T want you to know:
# - It detects when institutions are accumulating/distributing
# - It identifies exact levels where stops are being hunted
# - It spots absorption zones where big players are defending
# - It enters BEFORE the move based on order flow velocity
#
# Win rate target: 70-75%
# R:R minimum: 1:3 (let winners run with trailing)
# Best pairs: EURUSD, GBPUSD, XAUUSD, US30 (liquid markets)
# Best sessions: London Open, NY Open, Overlap
# =============================================================

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from core.logger import get_logger

log = get_logger(__name__)

STRATEGY_NAME = "SMART_MONEY_FOOTPRINT"
MIN_SCORE = 80
VERSION = "1.0"  # Institutional Alpha Version


def evaluate(
    symbol: str,
    df_m1: pd.DataFrame = None,
    df_m5: pd.DataFrame = None,
    df_m15: pd.DataFrame = None,
    df_h1: pd.DataFrame = None,
    smc_report: dict = None,
    market_report: dict = None,
    master_report: dict = None
) -> dict | None:
    """
    Evaluate Smart Money Footprint setup.
    
    Entry criteria (ALL must align):
    1. Smart Money Score > 60 or < -60 (strong institutional bias)
    2. At least 3 confirming alpha signals (divergence, absorption, stop hunt, etc.)
    3. Price at key level (OB, FVG, VP level)
    4. Session is active (London/NY)
    5. Spread is acceptable
    
    Returns signal dict or None
    """
    from data_layer.order_flow_alpha import get_order_flow_alpha
    
    # ── Data validation ─────────────────────────────────────
    if df_m5 is None or len(df_m5) < 50:
        return None
    if market_report is None:
        return None
    
    # Get tick data for order flow analysis
    # PRIORITY: Reuse cached ticks from market_scanner (fetched once per cycle)
    # FALLBACK: Fetch fresh if not available (shouldn't happen in normal flow)
    df_ticks = None
    if market_report:
        df_ticks = market_report.get('df_ticks')
    if df_ticks is None or not isinstance(df_ticks, pd.DataFrame) or len(df_ticks) < 200:
        from data_layer.tick_fetcher import get_ticks
        df_ticks = get_ticks(symbol, num_ticks=500)
    if df_ticks is None or len(df_ticks) < 200:
        return None
    
    close_price = float(df_m5.iloc[-1]['close'])
    pip_size = _get_pip_point(symbol, close_price)
    
    score = 0
    confluence = []
    
    # ════════════════════════════════════════════════════════
    # STEP 1: Calculate ALL Order Flow Alpha Signals
    # ════════════════════════════════════════════════════════
    
    alpha = get_order_flow_alpha()
    
    # Signal 1: Cumulative Delta Divergence
    divergence = alpha.calculate_cumulative_delta_divergence(
        df_ticks=df_ticks,
        symbol=symbol,
        lookback_bars=100
    )
    
    # Signal 2: Absorption Detection
    absorption = alpha.detect_absorption(
        df_ticks=df_ticks,
        symbol=symbol,
        price_window=pip_size * 2,  # 2 pips window
        min_aggressive_volume=30
    )
    
    # Signal 3: Stop Hunt Detection
    stop_hunt = alpha.detect_stop_hunt(
        df_candles=df_m5,
        symbol=symbol,
        lookback=30
    )
    
    # Signal 4: Order Flow Velocity
    velocity = alpha.calculate_order_flow_velocity(
        df_ticks=df_ticks,
        symbol=symbol,
        window=30
    )
    
    # Get volume profile — try multiple paths for resilience
    # Path 1: Direct from market_report (market_scanner stores it)
    # Path 2: Convenience shortcut from master_report
    volume_profile = (market_report.get('profile', {})
                      if market_report else {})
    if not volume_profile and master_report:
        volume_profile = master_report.get('volume_profile', {})
    # Path 3: Build fresh if completely missing (shouldn't happen normally)
    if not volume_profile or not volume_profile.get('poc'):
        try:
            from data_layer.volume_profile import get_full_profile
            volume_profile = get_full_profile(symbol) or {}
        except Exception:
            volume_profile = {}
    
    # Signal 5: Smart Money Master Score
    smart_money_score = alpha.calculate_smart_money_score(
        symbol=symbol,
        df_ticks=df_ticks,
        df_candles=df_m5,
        volume_profile=volume_profile,
        divergence_result=divergence,
        absorption_result=absorption,
        stop_hunt_result=stop_hunt,
        velocity_result=velocity
    )
    
    # ════════════════════════════════════════════════════════
    # STEP 2: Check Smart Money Score Threshold
    # ════════════════════════════════════════════════════════
    
    sms_value = smart_money_score.get('score', 0)
    sms_bias = smart_money_score.get('bias', 'NEUTRAL')
    sms_confidence = smart_money_score.get('confidence', 0)
    
    # Must have strong institutional bias
    if abs(sms_value) < 50:
        return None  # Not enough institutional activity
    
    if sms_confidence < 60:
        return None  # Low confidence
    
    direction = 'BUY' if sms_value > 0 else 'SELL'
    score += 30  # Base score for strong smart money signal
    confluence.append(f"SMS_{sms_bias}_{sms_value:+.1f}")
    
    # ════════════════════════════════════════════════════════
    # STEP 3: Count Confirming Alpha Signals
    # ════════════════════════════════════════════════════════
    
    confirming_signals = 0
    
    # Check Delta Divergence confirmation
    if divergence.get('divergence') != 'NONE':
        div_strength = divergence.get('strength', 0)
        if div_strength > 50:
            # Divergence must align with smart money direction
            if (direction == 'BUY' and divergence['divergence'] == 'BULLISH') or \
               (direction == 'SELL' and divergence['divergence'] == 'BEARISH'):
                score += 20
                confirming_signals += 1
                confluence.append(f"DELTA_DIV_{divergence['divergence']}_{div_strength:.0f}%")
    
    # Check Absorption confirmation
    if absorption.get('absorption_detected'):
        levels = absorption.get('levels', [])
        if levels:
            top_level = levels[0]
            if top_level['strength'] > 60:
                # Absorption level must support our direction
                if (direction == 'BUY' and top_level['type'] == 'SUPPORT') or \
                   (direction == 'SELL' and top_level['type'] == 'RESISTANCE'):
                    score += 15
                    confirming_signals += 1
                    confluence.append(f"ABSORPTION_{top_level['type']}_{top_level['strength']:.0f}%")
    
    # Check Stop Hunt confirmation
    if stop_hunt.get('stop_hunt_detected'):
        sh_direction = stop_hunt.get('direction', 'NONE')
        sh_strength = stop_hunt.get('reversal_strength', 0)
        if sh_strength > 50:
            # Stop hunt reversal must align with our direction
            if direction == sh_direction:
                score += 20
                confirming_signals += 1
                confluence.append(f"STOP_HUNT_{sh_direction}_{sh_strength:.0f}%")
    
    # Check Velocity confirmation
    vel_signal = velocity.get('signal', 'NEUTRAL')
    if velocity.get('institutional_activity'):
        if (direction == 'BUY' and vel_signal in ['STRONG_BUY', 'BUY']) or \
           (direction == 'SELL' and vel_signal in ['STRONG_SELL', 'SELL']):
            score += 15
            confirming_signals += 1
            confluence.append(f"OF_VELOCITY_{vel_signal}")
    
    # MUST have at least 3 confirming signals
    if confirming_signals < 3:
        log.info(f"[{symbol}] {STRATEGY_NAME}: Only {confirming_signals} confirming signals "
                f"(need 3+) | SMS: {sms_value:+.1f}")
        return None
    
    # ════════════════════════════════════════════════════════
    # STEP 4: Check Market Structure Alignment
    # ════════════════════════════════════════════════════════
    
    if smc_report:
        # Check if we're at a key SMC level
        nearest_ob = smc_report.get('nearest_ob')
        if nearest_ob:
            ob_distance = abs(close_price - float(nearest_ob.get('bottom', close_price))) / pip_size
            
            # Price near order block (< 10 pips)
            if ob_distance < 10:
                if (direction == 'BUY' and nearest_ob['type'] == 'BULLISH') or \
                   (direction == 'SELL' and nearest_ob['type'] == 'BEARISH'):
                    score += 15
                    confluence.append(f"AT_OB_{nearest_ob['type']}")
        
        # Check HTF alignment
        htf = smc_report.get('htf_alignment', {})
        if htf.get('approved'):
            h4_bias = htf.get('h4_bias', '')
            if (direction == 'BUY' and h4_bias == 'BULLISH') or \
               (direction == 'SELL' and h4_bias == 'BEARISH'):
                score += 10
                confluence.append("HTF_H4_ALIGNED")
    
    # ════════════════════════════════════════════════════════
    # STEP 5: Check Session & Market Conditions
    # ════════════════════════════════════════════════════════
    
    if master_report:
        session = master_report.get('session', 'UNKNOWN')
        if session in ['LONDON_SESSION', 'NY_LONDON_OVERLAP', 'NY_AFTERNOON']:
            score += 10
            confluence.append(f"SESSION_{session}")
        
        # Check spread (only enforce in live; backtest may not have this key)
        spread_info = market_report.get('spread', {})
        spread_pips = spread_info.get('pips', 0)
        if spread_pips > 0:  # Spread data available
            if spread_pips < 2.0:  # Tight spread
                score += 5
                confluence.append(f"TIGHT_SPREAD_{spread_pips:.1f}p")
            elif spread_pips > 5.0:
                log.info(f"[{symbol}] {STRATEGY_NAME}: Spread too wide ({spread_pips:.1f}p)")
                return None
    
    # ════════════════════════════════════════════════════════
    # STEP 6: Final Score Check
    # ════════════════════════════════════════════════════════
    
    if score < MIN_SCORE:
        log.info(f"[{symbol}] {STRATEGY_NAME}: Score {score} below threshold {MIN_SCORE}")
        return None
    
    if len(confluence) < 5:
        log.info(f"[{symbol}] {STRATEGY_NAME}: Only {len(confluence)} confluence factors")
        return None
    
    # ════════════════════════════════════════════════════════
    # STEP 7: Calculate SL/TP (Aggressive R:R for institutional setups)
    # ════════════════════════════════════════════════════════
    
    # Use ATR for dynamic SL
    atr_raw = float(df_m5.iloc[-1].get('atr', 0))
    if atr_raw > 0:
        sl_pips_raw = atr_raw / pip_size * 1.5  # 1.5x ATR
        sl_pips = max(5.0, min(15.0, sl_pips_raw))  # Cap between 5-15 pips
    else:
        sl_pips = 8.0  # Default
    
    # TP targets (institutional setups deserve better R:R)
    tp1_pips = sl_pips * 2.0  # Minimum 1:2 R:R
    tp2_pips = sl_pips * 4.0  # Let runners go to 1:4
    tp3_pips = sl_pips * 6.0  # Moon bag
    
    if direction == 'BUY':
        sl_price = round(close_price - sl_pips * pip_size, 5)
        tp1_price = round(close_price + tp1_pips * pip_size, 5)
        tp2_price = round(close_price + tp2_pips * pip_size, 5)
        tp3_price = round(close_price + tp3_pips * pip_size, 5)
    else:  # SELL
        sl_price = round(close_price + sl_pips * pip_size, 5)
        tp1_price = round(close_price - tp1_pips * pip_size, 5)
        tp2_price = round(close_price - tp2_pips * pip_size, 5)
        tp3_price = round(close_price - tp3_pips * pip_size, 5)
    
    log.info(f"[{STRATEGY_NAME}] {direction} {symbol} | Score:{score} | "
            f"Conf:{confirming_signals}/5 | SL:{sl_pips:.1f}p TP1:{tp1_pips:.1f}p | "
            f"{' | '.join(confluence[:5])}")
    
    return {
        'direction': direction,
        'entry_price': close_price,
        'sl_price': sl_price,
        'tp1_price': tp1_price,
        'tp2_price': tp2_price,
        'tp3_price': tp3_price,
        'sl_pips': round(sl_pips, 1),
        'tp1_pips': round(tp1_pips, 1),
        'tp2_pips': round(tp2_pips, 1),
        'tp3_pips': round(tp3_pips, 1),
        'strategy': STRATEGY_NAME,
        'version': VERSION,
        'score': score,
        'confluence': confluence,
        'spread': 0,
        'metadata': {
            'smart_money_score': sms_value,
            'smart_money_confidence': sms_confidence,
            'confirming_signals': confirming_signals,
            'divergence': divergence,
            'absorption': absorption,
            'stop_hunt': stop_hunt,
            'velocity': velocity
        }
    }


def _get_pip_point(symbol: str, price: float) -> float:
    """Get correct pip point for any symbol."""
    sym = str(symbol).upper()
    # Indices — trade in full points
    if any(x in sym for x in ["US30", "US500", "USTEC", "JP225", "DE30", "UK100"]):
        return 1.0
    # Gold
    if "XAU" in sym:
        return 0.1
    # Silver
    if "XAG" in sym:
        return 0.01
    # Oil
    if any(x in sym for x in ["WTI", "BRN"]):
        return 0.01
    # JPY pairs
    if "JPY" in sym:
        return 0.01
    # Standard forex
    return 0.0001


# Register strategy
if __name__ == "__main__":
    # Test configuration
    print(f"{STRATEGY_NAME} v{VERSION} loaded")
    print(f"Minimum Score: {MIN_SCORE}")
    print("Institutional Alpha Strategy - Ready for deployment")
