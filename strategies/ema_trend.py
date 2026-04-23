# =============================================================
# strategies/ema_trend.py
# Strategy 1: Multi-Timeframe EMA Trend + Confluence Entry
# UPGRADED v3.0: HYBRID — Intraday + Scalping
# Timeframes: H4 (trend) → H1 (structure) → M15 (bias) →
#             M5 (structure confirmation) → M1 (precise entry)
# NEW: Volume surge, order flow imbalance, momentum velocity
#      as mandatory entry confirmation gates.
# Best state : TRENDING_STRONG or BREAKOUT_ACCEPTED
# Best session: LONDON_KILLZONE, NY_LONDON_OVERLAP
# =============================================================

import pandas as pd
from core.logger import get_logger

log = get_logger(__name__)

STRATEGY_NAME = "EMA_TREND_MTF"
MIN_SCORE     = 75   # v4.3 STRICT — only trade the best setups (was 65)
VERSION       = "4.0"  # EMA21 pullback entry — not M1 candle close


def evaluate(symbol: str,
             df_m1: pd.DataFrame,
             df_m5: pd.DataFrame,
             df_m15: pd.DataFrame,
             df_h1:  pd.DataFrame,
             df_h4:  pd.DataFrame,
             smc_report:  dict = None,
             master_report: dict = None) -> dict | None:
    """
    Evaluates EMA trend setup across H4, H1, M15, M5, M1.
    Returns signal dict or None.

    HYBRID MODE:
      - M15 determines bias direction (same as before)
      - M5 provides structure confirmation (NEW)
      - M1 provides precise entry with volume surge + imbalance + velocity (NEW)
    
    Signal keys:
        direction, entry_price, sl_price, tp1_price, tp2_price,
        sl_pips, tp1_pips, tp2_pips, strategy, version,
        score, confluence, regime
    """
    if df_m15 is None or df_h1 is None or df_h4 is None:
        return None
    if len(df_m15) < 10 or len(df_h1) < 10 or len(df_h4) < 10:
        return None

    h4  = df_h4.iloc[-1]
    h1  = df_h1.iloc[-1]
    m15 = df_m15.iloc[-1]

    # Pip size detection — matches order_manager._get_pip_point exactly
    sym = str(symbol).upper()
    close_price = float(m15['close'])
    if any(x in sym for x in ["US30", "US500", "USTEC", "JP225", "DE30", "UK100"]):
        pip_size = 1.0
    elif "XAU" in sym:
        pip_size = 0.1
    elif "XAG" in sym:
        pip_size = 0.01
    elif any(x in sym for x in ["WTI", "BRN"]):
        pip_size = 0.01
    elif close_price > 50:
        pip_size = 0.01  # JPY pairs
    else:
        pip_size = 0.0001  # Standard forex
    atr_pips    = float(m15['atr']) / pip_size

    if atr_pips < 3.0:
        return None  # Market too quiet for a meaningful trade

    # ── EXTERNAL CONTEXT ──────────────────────────────────
    pd_zone   = ''
    htf_ok    = True
    smc_bias  = 'NEUTRAL'
    if smc_report:
        pd_zone  = smc_report.get('premium_discount', {}).get('zone', '')
        htf_ok   = smc_report.get('htf_alignment', {}).get('approved', True)
        smc_bias = smc_report.get('smc_bias', 'NEUTRAL')

    # ── NEW: Scalping data from master_report ─────────────
    scalping_signal = master_report.get('scalping_signal', {}) if master_report else {}
    order_flow_imb  = master_report.get('order_flow_imbalance', {}) if master_report else {}
    volume_surge    = master_report.get('volume_surge', {}) if master_report else {}
    momentum        = master_report.get('momentum', {}) if master_report else {}

    # ── NEW: M5 Structure Confirmation ────────────────────
    m5_confirmed = False
    m5_bias = 'NEUTRAL'
    if df_m5 is not None and len(df_m5) >= 10:
        m5 = df_m5.iloc[-1]
        m5_bull = (m5['ema_9'] > m5['ema_21'])
        m5_bear = (m5['ema_9'] < m5['ema_21'])
        m5_st_bull = (int(m5.get('supertrend_dir', 0)) == 1)
        m5_st_bear = (int(m5.get('supertrend_dir', 0)) == -1)
        if m5_bull and m5_st_bull:
            m5_confirmed = True
            m5_bias = 'BULLISH'
        elif m5_bear and m5_st_bear:
            m5_confirmed = True
            m5_bias = 'BEARISH'

    # ── NEW: M1 Entry Trigger ─────────────────────────────
    m1_trigger_aligned = False
    m1_volume_confirms = False
    m1_stochrsi_signal = 'NONE'
    if df_m1 is not None and len(df_m1) >= 20:
        m1 = df_m1.iloc[-1]
        m1_prev = df_m1.iloc[-2]

        # M1 StochRSI timing
        stoch_k = float(m1.get('stoch_rsi_k', 50))
        stoch_d = float(m1.get('stoch_rsi_d', 50))
        prev_k = float(m1_prev.get('stoch_rsi_k', 50))
        prev_d = float(m1_prev.get('stoch_rsi_d', 50))

        if prev_k <= prev_d and stoch_k > stoch_d and stoch_k < 40:
            m1_stochrsi_signal = 'BUY_CROSS'
        elif prev_k >= prev_d and stoch_k < stoch_d and stoch_k > 60:
            m1_stochrsi_signal = 'SELL_CROSS'
        elif stoch_k < 20:
            m1_stochrsi_signal = 'OVERSOLD'
        elif stoch_k > 80:
            m1_stochrsi_signal = 'OVERBOUGHT'

        # M1 volume confirmation
        vol_ma = float(m1.get('vol_ma20', 0))
        if vol_ma > 0 and m1['tick_volume'] >= vol_ma * 1.5:
            m1_volume_confirms = True

    # ── BUY SETUP ─────────────────────────────────────────
    # H4 trend → H1 structure → M15 pullback → M5 confirm → M1 trigger
    h4_bull = (h4['ema_9'] > h4['ema_21'] > h4['ema_50'])
    h1_bull = (h1['ema_9'] > h1['ema_21'])
    st_bull = (int(h1.get('supertrend_dir', 0)) == 1)
    m15_pull= (float(m15.get('stoch_rsi_k', 50)) < 35)   # Slightly relaxed — entry gate is now EMA21 proximity
    m15_zone= (h1['ema_21'] * 0.9990 <= m15['close'] <= h1['ema_21'] * 1.0020)  # 20 pip window

    if h4_bull and h1_bull:
        # MANDATORY: rolling delta must agree before scoring anything
        rolling_delta = master_report.get('rolling_delta', {}) if master_report else {}
        if not rolling_delta:
            rolling_delta = market_report.get('rolling_delta', {}) if market_report else {}
        delta_bias = rolling_delta.get('bias', 'NEUTRAL') if isinstance(rolling_delta, dict) else 'NEUTRAL'
        if delta_bias != 'BULLISH':
            return None  # Lagging indicators say bull but flow disagrees — skip

        score      = 0
        confluence = []

        # --- Layer 1: H4/H1 Macro (up to 45 pts) ---
        score += 25; confluence.append("H4_H1_BULL_STACK")
        if st_bull:
            score += 20; confluence.append("SUPERTREND_BULL")

        # --- Layer 2: M15 Bias (up to 30 pts) ---
        if m15_pull:
            score += 20; confluence.append("STOCHRSI_OVERSOLD")
        if m15_zone:
            score += 10; confluence.append("PRICE_AT_EMA21")
        if m15['macd_hist'] > 0:
            score += 10; confluence.append("MACD_BULL")
        if m15['tick_volume'] > m15['vol_ma20'] * 1.2:
            score += 10; confluence.append("VOLUME_CONFIRM")

        # --- Layer 3: M5 Structure Confirmation (up to 15 pts) NEW ---
        if m5_confirmed and m5_bias == 'BULLISH':
            score += 15; confluence.append("M5_STRUCTURE_CONFIRMED")
        elif m5_confirmed:
            score += 5; confluence.append("M5_PARTIAL_CONFIRM")

        # --- Layer 4: M1 Entry Trigger (up to 20 pts) NEW ---
        if m1_stochrsi_signal == 'BUY_CROSS':
            score += 10; confluence.append("M1_STOCHRSI_BUY_CROSS")
        if m1_volume_confirms:
            score += 10; confluence.append("M1_VOLUME_SPIKE")

        # --- Layer 5: Scalping Gates (max 8 pts — deduplicated) NEW ---
        # NOTE: OF, volume, momentum are already scored in market_scanner.
        # Only add a small bonus here for alignment, don't double-count.
        scalp_bonus = 0
        if order_flow_imb.get('imbalance', 0) > 0.2:
            scalp_bonus += 3; confluence.append("OF_IMBALANCE_BUY")
        if volume_surge.get('surge_detected', False):
            scalp_bonus += 3; confluence.append("VOLUME_SURGE")
        if momentum.get('is_scalpable', False):
            scalp_bonus += 3; confluence.append("MOMENTUM_STRONG")
        # Cap scalping bonus to prevent score inflation
        if scalp_bonus > 8:
            scalp_bonus = 8
            confluence.append("SCALP_CAP")
        score += scalp_bonus

        # --- Penalties ---
        if smc_bias == 'BULLISH':
            score += 10; confluence.append("SMC_ALIGNED")
        if 'EXTREME_PREMIUM' in pd_zone:
            score -= 15; confluence.append("PD_PREMIUM_PENALTY")
        if not htf_ok:
            score -= 20; confluence.append("HTF_REJECTED")

        # --- NEW: Choppy market penalty ---
        if momentum.get('is_choppy', False):
            score -= 10; confluence.append("CHOPPY_MARKET_PENALTY")

        # v4.3 STRICT: Require minimum confluence count
        if len(confluence) < 5:
            return None  # Not enough factors aligned

        if score >= MIN_SCORE:
            # ── ENTRY AT EMA21 PULLBACK LEVEL (not M1 candle close) ──
            #
            # The EMA21 on H1 is the institutional mean for intraday trend.
            # In a bullish trend, price pulls back TO this level then bounces.
            # Entering at the EMA21 level (not wherever M1 happened to close)
            # gives a far better R:R because SL is below EMA50 and entry is
            # at the actual institutional support zone.
            #
            # Logic:
            #   1. Ideal entry = H1 EMA21 (the pullback target)
            #   2. If price is already ABOVE EMA21 (bounced), use M1 close
            #      but only if within 5 pips of EMA21 (still near the zone)
            #   3. If price is BELOW EMA21 (hasn't touched yet), skip —
            #      wait for the touch (don't chase)

            h1_ema21     = float(h1['ema_21'])
            h1_ema50     = float(h1['ema_50'])
            current_m1   = float(df_m1.iloc[-1]['close']) if df_m1 is not None and len(df_m1) > 0 else close_price
            pips_above_ema21 = (current_m1 - h1_ema21) / pip_size

            if pips_above_ema21 < 0:
                # Price still below EMA21 — pullback not complete, skip
                log.debug(f"[{STRATEGY_NAME}] BUY {symbol} skipped — "
                          f"price {pips_above_ema21:.1f}p below EMA21, wait for touch")
                return None
            elif pips_above_ema21 <= 8:
                # Price just touched/bounced from EMA21 — ideal zone
                # Enter at the EMA21 level for best R:R
                entry = round(h1_ema21, 5)
                confluence.append("ENTRY_AT_EMA21_ZONE")
            else:
                # Price bounced hard and is now more than 8 pips above EMA21
                # Entry here means chasing — skip this candle
                log.debug(f"[{STRATEGY_NAME}] BUY {symbol} skipped — "
                          f"price {pips_above_ema21:.1f}p above EMA21, too late to enter")
                return None

            # SL below EMA50 with small ATR buffer — institutional support
            sl_price  = round(h1_ema50 - atr_pips * 0.3 * pip_size, 5)
            sl_pips   = round((entry - sl_price) / pip_size, 1)

            # Validate SL makes sense (must be below entry)
            if sl_pips <= 0:
                log.debug(f"[{STRATEGY_NAME}] BUY {symbol} — invalid SL (entry={entry}, sl={sl_price})")
                return None

            tp1_pips  = round(sl_pips * 2.0, 1)
            tp2_pips  = round(sl_pips * 3.5, 1)
            tp1_price = round(entry + tp1_pips * pip_size, 5)
            tp2_price = round(entry + tp2_pips * pip_size, 5)

            log.info(f"[{STRATEGY_NAME} v{VERSION}] BUY {symbol}"
                     f" entry={entry} (EMA21={float(h1['ema_21']):.5f})"
                     f" SL={sl_pips}p TP1={tp1_pips}p"
                     f" Score:{score} | {', '.join(confluence)}")
            return {
                "direction":   "BUY",
                "entry_price": entry,
                "sl_price":    sl_price,
                "tp1_price":   tp1_price,
                "tp2_price":   tp2_price,
                "sl_pips":     sl_pips,
                "tp1_pips":    tp1_pips,
                "tp2_pips":    tp2_pips,
                "strategy":    STRATEGY_NAME,
                "version":     VERSION,
                "score":       score,
                "confluence":  confluence,
                # --- NEW metadata for analysis ---
                "m5_confirmed":   m5_confirmed,
                "m1_trigger":     m1_stochrsi_signal,
                "scalping_status": scalping_signal.get('status', 'N/A'),
            }

    # ── SELL SETUP ────────────────────────────────────────
    h4_bear = (h4['ema_9'] < h4['ema_21'] < h4['ema_50'])
    h1_bear = (h1['ema_9'] < h1['ema_21'])
    st_bear = (int(h1.get('supertrend_dir', 0)) == -1)
    m15_pull= (float(m15.get('stoch_rsi_k', 50)) > 65)   # Slightly relaxed
    m15_zone= (h1['ema_21'] * 0.9980 <= m15['close'] <= h1['ema_21'] * 1.0010)  # 20 pip window

    if h4_bear and h1_bear:
        # MANDATORY: rolling delta must agree before scoring anything
        rolling_delta = master_report.get('rolling_delta', {}) if master_report else {}
        if not rolling_delta:
            rolling_delta = market_report.get('rolling_delta', {}) if market_report else {}
        delta_bias = rolling_delta.get('bias', 'NEUTRAL') if isinstance(rolling_delta, dict) else 'NEUTRAL'
        if delta_bias != 'BEARISH':
            return None  # Lagging indicators say bear but flow disagrees — skip

        score      = 0
        confluence = []

        # --- Layer 1: H4/H1 Macro (up to 45 pts) ---
        score += 25; confluence.append("H4_H1_BEAR_STACK")
        if st_bear:
            score += 20; confluence.append("SUPERTREND_BEAR")

        # --- Layer 2: M15 Bias (up to 30 pts) ---
        if m15_pull:
            score += 20; confluence.append("STOCHRSI_OVERBOUGHT")
        if m15_zone:
            score += 10; confluence.append("PRICE_AT_EMA21")
        if m15['macd_hist'] < 0:
            score += 10; confluence.append("MACD_BEAR")
        if m15['tick_volume'] > m15['vol_ma20'] * 1.2:
            score += 10; confluence.append("VOLUME_CONFIRM")

        # --- Layer 3: M5 Structure Confirmation (up to 15 pts) NEW ---
        if m5_confirmed and m5_bias == 'BEARISH':
            score += 15; confluence.append("M5_STRUCTURE_CONFIRMED")
        elif m5_confirmed:
            score += 5; confluence.append("M5_PARTIAL_CONFIRM")

        # --- Layer 4: M1 Entry Trigger (up to 20 pts) NEW ---
        if m1_stochrsi_signal == 'SELL_CROSS':
            score += 10; confluence.append("M1_STOCHRSI_SELL_CROSS")
        if m1_volume_confirms:
            score += 10; confluence.append("M1_VOLUME_SPIKE")

        # --- Layer 5: Scalping Gates (max 8 pts — deduplicated) NEW ---
        scalp_bonus = 0
        if order_flow_imb.get('imbalance', 0) < -0.2:
            scalp_bonus += 3; confluence.append("OF_IMBALANCE_SELL")
        if volume_surge.get('surge_detected', False):
            scalp_bonus += 3; confluence.append("VOLUME_SURGE")
        if momentum.get('is_scalpable', False):
            scalp_bonus += 3; confluence.append("MOMENTUM_STRONG")
        if scalp_bonus > 8:
            scalp_bonus = 8
            confluence.append("SCALP_CAP")
        score += scalp_bonus

        # --- Penalties ---
        if smc_bias == 'BEARISH':
            score += 10; confluence.append("SMC_ALIGNED")
        if 'EXTREME_DISCOUNT' in pd_zone:
            score -= 15; confluence.append("PD_DISCOUNT_PENALTY")
        if not htf_ok:
            score -= 20; confluence.append("HTF_REJECTED")

        # --- NEW: Choppy market penalty ---
        if momentum.get('is_choppy', False):
            score -= 10; confluence.append("CHOPPY_MARKET_PENALTY")

        # v4.3 STRICT: Require minimum confluence count
        if len(confluence) < 5:
            return None  # Not enough factors aligned

        if score >= MIN_SCORE:
            # ── ENTRY AT EMA21 PULLBACK LEVEL — SELL VERSION ──
            #
            # In a bearish trend, price pulls back UP to EMA21 then reverses.
            # Enter at the EMA21 level (resistance), SL above EMA50 (above
            # the structure that should hold if trend is valid).

            h1_ema21     = float(h1['ema_21'])
            h1_ema50     = float(h1['ema_50'])
            current_m1   = float(df_m1.iloc[-1]['close']) if df_m1 is not None and len(df_m1) > 0 else close_price
            pips_below_ema21 = (h1_ema21 - current_m1) / pip_size

            if pips_below_ema21 < 0:
                # Price still above EMA21 — pullback not complete, skip
                log.debug(f"[{STRATEGY_NAME}] SELL {symbol} skipped — "
                          f"price {abs(pips_below_ema21):.1f}p above EMA21, wait for touch")
                return None
            elif pips_below_ema21 <= 8:
                # Price just touched/rejected from EMA21 — ideal zone
                entry = round(h1_ema21, 5)
                confluence.append("ENTRY_AT_EMA21_ZONE")
            else:
                # Price already more than 8 pips below EMA21 — chasing
                log.debug(f"[{STRATEGY_NAME}] SELL {symbol} skipped — "
                          f"price {pips_below_ema21:.1f}p below EMA21, too late to enter")
                return None

            # SL above EMA50 with small ATR buffer — institutional resistance
            sl_price  = round(h1_ema50 + atr_pips * 0.3 * pip_size, 5)
            sl_pips   = round((sl_price - entry) / pip_size, 1)

            if sl_pips <= 0:
                log.debug(f"[{STRATEGY_NAME}] SELL {symbol} — invalid SL (entry={entry}, sl={sl_price})")
                return None

            tp1_pips  = round(sl_pips * 2.0, 1)
            tp2_pips  = round(sl_pips * 3.5, 1)
            tp1_price = round(entry - tp1_pips * pip_size, 5)
            tp2_price = round(entry - tp2_pips * pip_size, 5)

            log.info(f"[{STRATEGY_NAME} v{VERSION}] SELL {symbol}"
                     f" entry={entry} (EMA21={float(h1['ema_21']):.5f})"
                     f" SL={sl_pips}p TP1={tp1_pips}p"
                     f" Score:{score} | {', '.join(confluence)}")
            return {
                "direction":   "SELL",
                "entry_price": entry,
                "sl_price":    sl_price,
                "tp1_price":   tp1_price,
                "tp2_price":   tp2_price,
                "sl_pips":     sl_pips,
                "tp1_pips":    tp1_pips,
                "tp2_pips":    tp2_pips,
                "strategy":    STRATEGY_NAME,
                "version":     VERSION,
                "score":       score,
                "confluence":  confluence,
                # --- NEW metadata for analysis ---
                "m5_confirmed":   m5_confirmed,
                "m1_trigger":     m1_stochrsi_signal,
                "scalping_status": scalping_signal.get('status', 'N/A'),
            }

    return None
