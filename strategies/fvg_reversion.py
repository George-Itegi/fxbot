# =============================================================
# strategies/fvg_reversion.py
# Strategy: Fair Value Gap Reversion
# When price moves too fast in one direction, it leaves an
# imbalance (gap between candle 1 wick and candle 3 wick).
# Price is magnetically drawn back to fill these gaps.
# Statistically 60-70% of quality FVGs get filled on majors.
#
# BUY  when: price retraces INTO a bullish FVG (demand imbalance)
# SELL when: price retraces INTO a bearish FVG (supply imbalance)
#
# Entry: at FVG deep edge (bottom for BUY, top for SELL)
# SL:   beyond FVG edge using ATR-based buffer (not fixed pips)
# TP:   opposite edge of FVG (gap fill target)
#
# Win rate target: 60-68%
# Best session: LONDON_SESSION, NY_LONDON_OVERLAP
# Best state:  BALANCED, TRENDING_STRONG, BREAKOUT_ACCEPTED
# =============================================================

import pandas as pd
import numpy as np
from core.logger import get_logger

log = get_logger(__name__)

STRATEGY_NAME = "FVG_REVERSION"
MIN_SCORE     = 68
VERSION       = "1.1"

# --- FVG Reversion Parameters ---
MIN_FVG_QUALITY   = 35      # Lowered from 60 — quality = gap_pips*3, so 35 = ~12pips FVG
MAX_FVG_DISTANCE  = 100     # Widened from 80 — more FVGs reachable
MIN_FVG_SIZE_PIPS = 2.5     # Lowered from 3.0 — more FVGs qualify
PARTIAL_FILL_TP   = 0.7     # Close 50% at 70% of gap fill (book profit early)
SL_BUFFER_PIPS    = 2.0     # Extra pips beyond FVG edge for SL (minimum)
SL_ATR_MULTIPLIER = 0.5      # Use 50% of ATR as SL buffer (whichever is larger)


def _get_pip_size(symbol: str, price: float) -> float:
    from core.pip_utils import get_pip_size as _gps
    return _gps(symbol, price)


def _price_in_fvg(price: float, fvg: dict, buffer_pips: float = 0, symbol: str = '') -> bool:
    """
    Check if current price is inside or near the FVG zone.
    With buffer_pips, we also trigger if price is approaching the zone.
    """
    if fvg['type'] == 'BULLISH_FVG':
        # Bullish FVG: bottom (prev high) to top (next candle low)
        zone_bottom = fvg['bottom']
        zone_top    = fvg['top']
    else:  # BEARISH_FVG
        # Bearish FVG: bottom (next candle high) to top (prev low)
        zone_bottom = fvg['bottom']
        zone_top    = fvg['top']

    pip_size = _get_pip_size(symbol, price)
    adjusted_bottom = zone_bottom - buffer_pips * pip_size
    adjusted_top    = zone_top + buffer_pips * pip_size

    return adjusted_bottom <= price <= adjusted_top


def _fvg_direction(fvg: dict) -> str:
    """Return BUY for bullish FVG, SELL for bearish FVG."""
    return "BUY" if fvg['type'] == 'BULLISH_FVG' else "SELL"


def _calc_risk_reward(fvg: dict, price: float, direction: str,
                      pip_size: float, atr_pips: float = 10.0) -> dict:
    """
    Calculate SL and TP based on FVG zone.
    SL: beyond the opposite edge of the FVG (thesis = gap fills)
    TP1: at FVG fill target (opposite edge)
    TP2: beyond FVG (let winner run to ATR extension)

    SL uses the LARGER of fixed buffer or ATR-based buffer to prevent
    tiny 2-pip stops on deep entries.
    """
    # Use the larger of fixed buffer or ATR-based buffer
    atr_buffer = max(SL_BUFFER_PIPS, atr_pips * SL_ATR_MULTIPLIER)

    if direction == "BUY":
        # BUY bullish FVG: price pulls back into demand zone
        # Entry is at bottom (deep edge), SL below the far side
        sl_price = round(fvg['bottom'] - atr_buffer * pip_size, 5)
        tp1_price = round(fvg['top'], 5)   # Fill target = top of gap
        sl_pips = round((price - sl_price) / pip_size, 1)
        tp1_pips = round((tp1_price - price) / pip_size, 1)
        # TP2 = extend 1.5x beyond gap fill
        gap_size = fvg['top'] - fvg['bottom']
        tp2_price = round(tp1_price + gap_size * 0.5, 5)
        tp2_pips = round((tp2_price - price) / pip_size, 1)
    else:
        # SELL bearish FVG: price pulls back into supply zone
        # Entry is at top (deep edge), SL above the far side
        sl_price = round(fvg['top'] + atr_buffer * pip_size, 5)
        tp1_price = round(fvg['bottom'], 5)  # Fill target = bottom of gap
        sl_pips = round((sl_price - price) / pip_size, 1)
        tp1_pips = round((price - tp1_price) / pip_size, 1)
        # TP2 = extend 1.5x beyond gap fill
        gap_size = fvg['top'] - fvg['bottom']
        tp2_price = round(tp1_price - gap_size * 0.5, 5)
        tp2_pips = round((tp2_price - price) / pip_size, 1)

    return {
        'sl_price':  sl_price,
        'tp1_price': tp1_price,
        'tp2_price': tp2_price,
        'sl_pips':   sl_pips,
        'tp1_pips':  tp1_pips,
        'tp2_pips':  tp2_pips,
    }


def evaluate(symbol: str,
             df_m1: pd.DataFrame = None,
             df_m5: pd.DataFrame = None,
             df_m15: pd.DataFrame = None,
             df_h1: pd.DataFrame = None,
             smc_report: dict = None,
             market_report: dict = None,
             df_h4: pd.DataFrame = None,
             master_report: dict = None) -> dict | None:
    """
    Fires when price retraces into a quality FVG zone.

    BUY  signal: price pulls back INTO a bullish FVG (demand imbalance)
    SELL signal: price pulls back INTO a bearish FVG (supply imbalance)

    Entry criteria:
      1. Quality FVG detected (score >= 60, gap >= 3 pips, unfilled)
      2. Price is inside or entering the FVG zone
      3. Order flow imbalance CONFIRMS FVG direction
      4. StochRSI supports reversal at zone
      5. Premium/Discount zone aligns with direction
      6. M5 candle shows rejection (wick into zone, body away)
    """
    # ── Data validation ─────────────────────────────────────
    if df_m15 is None or df_h1 is None:
        return None
    if len(df_m15) < 50 or len(df_h1) < 50:
        return None

    close_price = float(df_m15.iloc[-1]['close'])
    pip_size    = _get_pip_size(symbol, close_price)
    atr_pips    = float(df_m15.iloc[-1].get('atr', 0)) / pip_size

    if atr_pips < 2.0:
        return None

    # ── Get FVG data from SMC report ────────────────────────
    if not smc_report:
        return None

    quality_fvgs = smc_report.get('quality_fvgs', [])
    nearest_fvg  = smc_report.get('nearest_fvg', None)

    if not quality_fvgs and not nearest_fvg:
        return None

    # ── Find the best FVG that price is touching/entering ──
    best_fvg = None
    best_score = 0

    candidates = []
    if quality_fvgs:
        candidates = [f for f in quality_fvgs if not f.get('filled', False)]
    elif nearest_fvg and not nearest_fvg.get('filled', False):
        candidates = [nearest_fvg]

    for fvg in candidates:
        # FVG must be large enough
        if fvg.get('gap_pips', 0) < MIN_FVG_SIZE_PIPS:
            continue

        # FVG must be high quality
        quality = fvg.get('quality_score', 0)
        if quality < MIN_FVG_QUALITY:
            continue

        # FVG must be within reachable distance
        distance = abs(close_price - fvg['mid']) / pip_size
        if distance > MAX_FVG_DISTANCE:
            continue

        # Price must be near or inside the FVG zone
        # Use a buffer of 5 pips to catch entries as price approaches
        if not _price_in_fvg(close_price, fvg, buffer_pips=5.0, symbol=symbol):
            continue

        # Score this candidate
        candidate_score = quality
        # Prefer closer FVGs
        if distance < 10:
            candidate_score += 15
        elif distance < 30:
            candidate_score += 5
        # Prefer larger gaps (stronger imbalance)
        if fvg.get('gap_pips', 0) >= 8:
            candidate_score += 10

        if candidate_score > best_score:
            best_score = candidate_score
            best_fvg = fvg

    if best_fvg is None:
        return None

    direction = _fvg_direction(best_fvg)
    score = 0
    confluence = []

    # ── Score: FVG Quality ─────────────────────────────────
    fvg_quality = best_fvg.get('quality_score', 0)
    gap_pips = best_fvg.get('gap_pips', 0)
    confluence.append(f"FVG_QUALITY_{fvg_quality}")

    if fvg_quality >= 80:
        score += 25
        confluence.append("FVG_EXCELLENT")
    elif fvg_quality >= 65:
        score += 18
        confluence.append("FVG_HIGH")
    else:
        score += 10
        confluence.append("FVG_AVERAGE")

    # ── Score: Gap Size ────────────────────────────────────
    if gap_pips >= 10:
        score += 10
        confluence.append(f"LARGE_GAP_{gap_pips}p")
    elif gap_pips >= 5:
        score += 5
        confluence.append(f"MEDIUM_GAP_{gap_pips}p")

    # ── Score: Order Flow Confirmation (MANDATORY) ─────────
    # FVG direction must match OF imbalance — this is the
    # institutional confirmation that the gap WILL fill.
    if market_report:
        of_imbalance  = market_report.get('order_flow_imbalance', {})
        volume_surge  = market_report.get('volume_surge', {})
        imb           = of_imbalance.get('imbalance', 0)
        imb_strength  = of_imbalance.get('strength', 'NONE')

        if direction == "BUY":
            if imb > 0.15 and imb_strength in ('STRONG', 'EXTREME'):
                score += 15
                confluence.append(f"OF_BULL_{imb:+.2f}_{imb_strength}")
            elif imb > 0.1:
                score += 8
                confluence.append(f"OF_BULL_{imb:+.2f}")
            else:
                # No OF confirmation — skip (retail only fills)
                confluence.append("OF_NO_CONFIRM_SKIP")
                return None
        elif direction == "SELL":
            if imb < -0.15 and imb_strength in ('STRONG', 'EXTREME'):
                score += 15
                confluence.append(f"OF_BEAR_{imb:+.2f}_{imb_strength}")
            elif imb < -0.1:
                score += 8
                confluence.append(f"OF_BEAR_{imb:+.2f}")
            else:
                confluence.append("OF_NO_CONFIRM_SKIP")
                return None

        # Volume surge bonus
        if volume_surge.get('surge_detected', False):
            score += 8
            confluence.append(f"VOL_SURGE_{volume_surge.get('surge_ratio', 0)}x")

    # ── Score: Premium/Discount Zone ───────────────────────
    if smc_report:
        pd_info = smc_report.get('premium_discount', {})
        pd_zone = pd_info.get('zone', '')

        if direction == "SELL" and "PREMIUM" in pd_zone:
            score += 10
            confluence.append("PREMIUM_ZONE_SELL")
        elif direction == "BUY" and "DISCOUNT" in pd_zone:
            score += 10
            confluence.append("DISCOUNT_ZONE_BUY")

    # ── Score: StochRSI at FVG Zone ────────────────────────
    if df_m15 is not None and len(df_m15) >= 3:
        stoch_k = float(df_m15.iloc[-1].get('stoch_rsi_k', 50))
        prev_k  = float(df_m15.iloc[-2].get('stoch_rsi_k', 50))

        if direction == "SELL" and stoch_k > 65:
            score += 10
            confluence.append("STOCHRSI_OVERBOUGHT")
            if prev_k > stoch_k:
                score += 5
                confluence.append("STOCHRSI_TURNING_DOWN")
        elif direction == "BUY" and stoch_k < 35:
            score += 10
            confluence.append("STOCHRSI_OVERSOLD")
            if prev_k < stoch_k:
                score += 5
                confluence.append("STOCHRSI_TURNING_UP")

    # ── Score: M5 Candle Rejection at FVG ──────────────────
    # Look for wick into FVG with body rejecting = strongest entry
    if df_m5 is not None and len(df_m5) >= 2:
        m5_last = df_m5.iloc[-1]
        m5_body = m5_last['close'] - m5_last['open']
        m5_wick_up = m5_last['high'] - max(m5_last['open'], m5_last['close'])
        m5_wick_dn = min(m5_last['open'], m5_last['close']) - m5_last['low']
        m5_range = m5_last['high'] - m5_last['low']

        if direction == "BUY":
            # Bullish: look for lower wick dipping into FVG zone
            wick_into_fvg = m5_last['low'] <= best_fvg['top']
            if wick_into_fvg and m5_body > 0:
                # Hammer/rejection candle at FVG
                if m5_wick_dn > abs(m5_body) * 0.5:
                    score += 10
                    confluence.append("M5_HAMMER_AT_FVG")
                else:
                    score += 5
                    confluence.append("M5_REJECTION_AT_FVG")
        elif direction == "SELL":
            # Bearish: look for upper wick poking into FVG zone
            wick_into_fvg = m5_last['high'] >= best_fvg['bottom']
            if wick_into_fvg and m5_body < 0:
                # Shooting star/rejection candle at FVG
                if m5_wick_up > abs(m5_body) * 0.5:
                    score += 10
                    confluence.append("M5_STAR_AT_FVG")
                else:
                    score += 5
                    confluence.append("M5_REJECTION_AT_FVG")

    # ── Score: Market Structure Alignment ──────────────────
    # FVG reversion works best when structure supports the fill
    if smc_report:
        structure = smc_report.get('structure', {})
        trend = structure.get('trend', 'RANGING')

        # Slight bonus if trend aligns (not required — FVGs work in ranging)
        if direction == "BUY" and trend == 'BULLISH':
            score += 5
            confluence.append("STRUCTURE_BULL_ALIGN")
        elif direction == "SELL" and trend == 'BEARISH':
            score += 5
            confluence.append("STRUCTURE_BEAR_ALIGN")

        # Bonus: FVG is near an Order Block = institutional zone
        nearest_ob = smc_report.get('nearest_ob', None)
        if nearest_ob:
            ob_mid = (nearest_ob.get('top', 0) + nearest_ob.get('bottom', 0)) / 2
            fvg_mid = best_fvg['mid']
            ob_fvg_dist = abs(ob_mid - fvg_mid) / pip_size
            if ob_fvg_dist < 20:
                # OB and FVG confluence = very strong zone
                ob_type = nearest_ob.get('type', '')
                if direction == "BUY" and "BULLISH" in ob_type:
                    score += 10
                    confluence.append(f"OB+FVG_CONFLUENCE_{ob_fvg_dist:.0f}p")
                elif direction == "SELL" and "BEARISH" in ob_type:
                    score += 10
                    confluence.append(f"OB+FVG_CONFLUENCE_{ob_fvg_dist:.0f}p")

    # ── Score: Momentum at FVG (price slowing into zone) ──
    if master_report:
        momentum = master_report.get('momentum', {})
        vel_dir = momentum.get('velocity_direction', 'FLAT')
        velocity = momentum.get('velocity_pips_min', 0)

        # Price decelerating into FVG = better entry
        if direction == "BUY" and vel_dir == "DOWN" and velocity < 1.0:
            score += 5
            confluence.append("PRICE_DECELERATING_INTO_FVG")
        elif direction == "SELL" and vel_dir == "UP" and velocity < 1.0:
            score += 5
            confluence.append("PRICE_DECELERATING_INTO_FVG")

    # ── Score threshold ─────────────────────────────────────
    if len(confluence) < 4:
        return None
    if score < MIN_SCORE:
        return None

    # Entry at FVG edge for best R:R (not candle close)
    if direction == "BUY":
        entry_price = round(best_fvg['bottom'], 5)  # Enter at demand zone bottom
    else:
        entry_price = round(best_fvg['top'], 5)     # Enter at supply zone top

    risk = _calc_risk_reward(best_fvg, entry_price, direction, pip_size, atr_pips)

    # Verify minimum R:R of 1.5:1
    if risk['sl_pips'] > 0:
        rr_ratio = risk['tp1_pips'] / risk['sl_pips']
        if rr_ratio < 1.5:
            # Gap too small relative to SL — skip
            confluence.append(f"RR_TOO_LOW_{rr_ratio:.1f}:1")
            return None

    log.info(f"[{STRATEGY_NAME}] {direction} {symbol}"
             f" Score:{score} | FVG:{best_fvg['type']} gap={gap_pips}p"
             f" quality={fvg_quality} entry={entry_price:.5f}"
             f" | {', '.join(confluence)}")

    return {
        "direction":   direction,
        "entry_price": entry_price,
        "sl_price":    risk['sl_price'],
        "tp1_price":   risk['tp1_price'],
        "tp2_price":   risk['tp2_price'],
        "sl_pips":     risk['sl_pips'],
        "tp1_pips":    risk['tp1_pips'],
        "tp2_pips":    risk['tp2_pips'],
        "strategy":    STRATEGY_NAME,
        "version":     VERSION,
        "score":       score,
        "confluence":  confluence,
        "fvg":         best_fvg,
        "spread":      0,
    }
