# =============================================================
# strategies/ema_trend.py
# Strategy 1: Multi-Timeframe EMA Trend + Confluence Entry
# UPGRADED: + Supertrend, StochRSI, HTF alignment, P/D filter
# Best state : TRENDING_STRONG or BREAKOUT_ACCEPTED
# Best session: LONDON_KILLZONE, NY_LONDON_OVERLAP
# Timeframes  : H4 bias → H1 confirm → M15 entry timing
# =============================================================

import pandas as pd
from core.logger import get_logger

log = get_logger(__name__)

STRATEGY_NAME = "EMA_TREND_MTF"
MIN_SCORE     = 65   # Raised from 55 — only high quality setups
VERSION       = "2.0"


def evaluate(symbol: str,
             df_m15: pd.DataFrame,
             df_h1:  pd.DataFrame,
             df_h4:  pd.DataFrame,
             smc_report:  dict = None,
             master_report: dict = None) -> dict | None:
    """
    Evaluates EMA trend setup across H4, H1, M15.
    Returns signal dict or None.

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

    # Pip size detection (JPY and Gold differ)
    close_price = float(m15['close'])
    pip_size    = 0.01 if close_price > 50 else 0.0001
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

    score      = 0
    confluence = []

    # ── BUY SETUP ─────────────────────────────────────────
    # Condition: H4 bull stack + H1 bull + M15 pullback confirmed
    h4_bull = (h4['ema_9'] > h4['ema_21'] > h4['ema_50'])
    h1_bull = (h1['ema_9'] > h1['ema_21'])
    st_bull = (int(h1.get('supertrend_dir', 0)) == 1)
    m15_pull= (float(m15.get('stoch_rsi_k', 50)) < 30)
    m15_zone= (h1['ema_21'] * 0.9998 <= m15['close'] <= h1['ema_21'] * 1.0010)

    if h4_bull and h1_bull:
        score += 25; confluence.append("H4_H1_BULL_STACK")
        if st_bull:
            score += 20; confluence.append("SUPERTREND_BULL")
        if m15_pull:
            score += 20; confluence.append("STOCHRSI_OVERSOLD")
        if m15_zone:
            score += 15; confluence.append("PRICE_AT_EMA21")
        if m15['macd_hist'] > 0:
            score += 10; confluence.append("MACD_BULL")
        if m15['tick_volume'] > m15['vol_ma20'] * 1.2:
            score += 10; confluence.append("VOLUME_CONFIRM")
        if smc_bias == 'BULLISH':
            score += 10; confluence.append("SMC_ALIGNED")
        # Penalize if in extreme premium
        if 'EXTREME_PREMIUM' in pd_zone:
            score -= 15; confluence.append("PD_PREMIUM_PENALTY")
        if not htf_ok:
            score -= 20; confluence.append("HTF_REJECTED")

        if score >= MIN_SCORE:
            sl_price  = round(float(h1['ema_50']) - atr_pips * 0.3 * pip_size, 5)
            tp1_price = round(close_price + atr_pips * 1.5 * pip_size, 5)
            tp2_price = round(close_price + atr_pips * 2.5 * pip_size, 5)
            sl_pips   = round((close_price - sl_price) / pip_size, 1)
            tp1_pips  = round((tp1_price - close_price) / pip_size, 1)
            tp2_pips  = round((tp2_price - close_price) / pip_size, 1)
            log.info(f"[{STRATEGY_NAME}] BUY {symbol} Score:{score} | "
                     f"{', '.join(confluence)}")
            return {
                "direction":   "BUY",
                "entry_price": close_price,
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
            }

    # ── SELL SETUP ────────────────────────────────────────
    score      = 0
    confluence = []

    h4_bear = (h4['ema_9'] < h4['ema_21'] < h4['ema_50'])
    h1_bear = (h1['ema_9'] < h1['ema_21'])
    st_bear = (int(h1.get('supertrend_dir', 0)) == -1)
    m15_pull= (float(m15.get('stoch_rsi_k', 50)) > 70)
    m15_zone= (h1['ema_21'] * 0.9990 <= m15['close'] <= h1['ema_21'] * 1.0002)

    if h4_bear and h1_bear:
        score += 25; confluence.append("H4_H1_BEAR_STACK")
        if st_bear:
            score += 20; confluence.append("SUPERTREND_BEAR")
        if m15_pull:
            score += 20; confluence.append("STOCHRSI_OVERBOUGHT")
        if m15_zone:
            score += 15; confluence.append("PRICE_AT_EMA21")
        if m15['macd_hist'] < 0:
            score += 10; confluence.append("MACD_BEAR")
        if m15['tick_volume'] > m15['vol_ma20'] * 1.2:
            score += 10; confluence.append("VOLUME_CONFIRM")
        if smc_bias == 'BEARISH':
            score += 10; confluence.append("SMC_ALIGNED")
        if 'EXTREME_DISCOUNT' in pd_zone:
            score -= 15; confluence.append("PD_DISCOUNT_PENALTY")
        if not htf_ok:
            score -= 20; confluence.append("HTF_REJECTED")

        if score >= MIN_SCORE:
            sl_price  = round(float(h1['ema_50']) + atr_pips * 0.3 * pip_size, 5)
            tp1_price = round(close_price - atr_pips * 1.5 * pip_size, 5)
            tp2_price = round(close_price - atr_pips * 2.5 * pip_size, 5)
            sl_pips   = round((sl_price - close_price) / pip_size, 1)
            tp1_pips  = round((close_price - tp1_price) / pip_size, 1)
            tp2_pips  = round((close_price - tp2_price) / pip_size, 1)
            log.info(f"[{STRATEGY_NAME}] SELL {symbol} Score:{score} | "
                     f"{', '.join(confluence)}")
            return {
                "direction":   "SELL",
                "entry_price": close_price,
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
            }

    return None
