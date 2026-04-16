# =============================================================
# strategies/liquidity_sweep_entry.py
# Strategy 3: Liquidity Sweep + BOS Entry
# Fires after stop hunts — the actual institutional entry trigger.
# Best state : BREAKOUT_ACCEPTED after sweep
# Best session: London open 08:00 UTC, NY open 13:00 UTC
# =============================================================

import pandas as pd
from core.logger import get_logger

log = get_logger(__name__)

STRATEGY_NAME = "LIQUIDITY_SWEEP_ENTRY"
MIN_SCORE     = 65
VERSION       = "1.0"


def evaluate(symbol: str,
             df_m15:  pd.DataFrame,
             df_h1:   pd.DataFrame,
             smc_report:    dict = None,
             market_report: dict = None) -> dict | None:
    """
    Fires when price sweeps a liquidity pool then reverses
    with BOS confirmation on M15.
    """
    if df_m15 is None or df_h1 is None or smc_report is None:
        return None

    m15 = df_m15.iloc[-1]
    h1  = df_h1.iloc[-1]

    close_price = float(m15['close'])
    pip_size    = 0.01 if close_price > 50 else 0.0001
    atr_pips    = float(m15['atr']) / pip_size

    if atr_pips < 3.0:
        return None

    # Get sweep data
    last_sweep    = smc_report.get('last_sweep')
    recent_sweeps = smc_report.get('recent_sweeps', [])
    structure     = smc_report.get('structure', {})
    bos           = structure.get('bos')
    smc_bias      = smc_report.get('smc_bias', 'NEUTRAL')
    htf_ok        = smc_report.get('htf_alignment', {}).get('approved', True)
    pd_zone       = smc_report.get('premium_discount', {}).get('zone', '')

    # Must have a recent sweep
    if not last_sweep:
        return None

    sweep_bias  = last_sweep.get('bias', 'NEUTRAL')
    reversal_p  = float(last_sweep.get('reversal_pips', 0))
    swept_level = float(last_sweep.get('swept_level', 0))

    # Reversal must be meaningful
    if reversal_p < 3.0:
        return None

    # Get delta from market report
    rolling_delta = market_report.get('rolling_delta', {}) if market_report else {}
    delta_bias    = rolling_delta.get('bias', 'NEUTRAL')
    delta_strength= rolling_delta.get('strength', 'WEAK')

    score      = 0
    confluence = []

    # ── BULLISH SWEEP ENTRY (BUY) ──────────────────────────
    # Sell stops swept below → price reverses up → BUY
    if sweep_bias == 'BULLISH':
        # Price must be above the swept level now (reversal confirmed)
        if close_price <= swept_level:
            return None

        score += 25; confluence.append("BULLISH_SWEEP_CONFIRMED")

        # Reversal strength
        if reversal_p >= 10:
            score += 15; confluence.append(f"STRONG_REVERSAL_{reversal_p:.0f}p")
        elif reversal_p >= 5:
            score += 8;  confluence.append(f"REVERSAL_{reversal_p:.0f}p")

        # BOS confirms structure shift
        if bos and 'BULLISH' in bos.get('type', ''):
            score += 20; confluence.append("BOS_BULL_CONFIRMED")

        # Supertrend turning or already bull
        if int(h1.get('supertrend_dir', 0)) == 1:
            score += 10; confluence.append("SUPERTREND_BULL")

        # StochRSI recovering from oversold
        stoch_k = float(m15.get('stoch_rsi_k', 50))
        if 20 <= stoch_k <= 45:
            score += 15; confluence.append("STOCHRSI_RECOVERING")

        # Delta confirming
        if delta_bias == 'BULLISH':
            score += 10; confluence.append("DELTA_BULL")
            if delta_strength in ('STRONG', 'MODERATE'):
                score += 5; confluence.append("DELTA_STRONG")

        # Volume spike on reversal
        if m15['tick_volume'] > m15['vol_ma20'] * 1.5:
            score += 10; confluence.append("VOLUME_SPIKE")

        # HTF + SMC alignment
        if htf_ok and smc_bias == 'BULLISH':
            score += 10; confluence.append("HTF_SMC_BULL")

        if 'EXTREME_PREMIUM' in pd_zone:
            score -= 15; confluence.append("PD_PREMIUM_PENALTY")

        if score >= MIN_SCORE:
            sl_price  = round(swept_level - atr_pips * 0.3 * pip_size, 5)
            tp1_price = round(close_price + atr_pips * 1.5 * pip_size, 5)
            tp2_price = round(close_price + atr_pips * 3.0 * pip_size, 5)
            sl_pips   = round((close_price - sl_price) / pip_size, 1)
            tp1_pips  = round((tp1_price - close_price) / pip_size, 1)
            tp2_pips  = round((tp2_price - close_price) / pip_size, 1)
            log.info(f"[{STRATEGY_NAME}] BUY {symbol} after sweep"
                     f" Score:{score} | {', '.join(confluence)}")
            return {
                "direction":    "BUY",
                "entry_price":  close_price,
                "sl_price":     sl_price,
                "tp1_price":    tp1_price,
                "tp2_price":    tp2_price,
                "sl_pips":      sl_pips,
                "tp1_pips":     tp1_pips,
                "tp2_pips":     tp2_pips,
                "strategy":     STRATEGY_NAME,
                "version":      VERSION,
                "score":        score,
                "confluence":   confluence,
                "swept_level":  swept_level,
            }

    # ── BEARISH SWEEP ENTRY (SELL) ─────────────────────────
    # Buy stops swept above → price reverses down → SELL
    score      = 0
    confluence = []

    if sweep_bias == 'BEARISH':
        if close_price >= swept_level:
            return None

        score += 25; confluence.append("BEARISH_SWEEP_CONFIRMED")

        if reversal_p >= 10:
            score += 15; confluence.append(f"STRONG_REVERSAL_{reversal_p:.0f}p")
        elif reversal_p >= 5:
            score += 8;  confluence.append(f"REVERSAL_{reversal_p:.0f}p")

        if bos and 'BEARISH' in bos.get('type', ''):
            score += 20; confluence.append("BOS_BEAR_CONFIRMED")

        if int(h1.get('supertrend_dir', 0)) == -1:
            score += 10; confluence.append("SUPERTREND_BEAR")

        stoch_k = float(m15.get('stoch_rsi_k', 50))
        if 55 <= stoch_k <= 80:
            score += 15; confluence.append("STOCHRSI_RECOVERING_DOWN")

        if delta_bias == 'BEARISH':
            score += 10; confluence.append("DELTA_BEAR")
            if delta_strength in ('STRONG', 'MODERATE'):
                score += 5; confluence.append("DELTA_STRONG")

        if m15['tick_volume'] > m15['vol_ma20'] * 1.5:
            score += 10; confluence.append("VOLUME_SPIKE")

        if htf_ok and smc_bias == 'BEARISH':
            score += 10; confluence.append("HTF_SMC_BEAR")

        if 'EXTREME_DISCOUNT' in pd_zone:
            score -= 15; confluence.append("PD_DISCOUNT_PENALTY")

        if score >= MIN_SCORE:
            sl_price  = round(swept_level + atr_pips * 0.3 * pip_size, 5)
            tp1_price = round(close_price - atr_pips * 1.5 * pip_size, 5)
            tp2_price = round(close_price - atr_pips * 3.0 * pip_size, 5)
            sl_pips   = round((sl_price - close_price) / pip_size, 1)
            tp1_pips  = round((close_price - tp1_price) / pip_size, 1)
            tp2_pips  = round((close_price - tp2_price) / pip_size, 1)
            log.info(f"[{STRATEGY_NAME}] SELL {symbol} after sweep"
                     f" Score:{score} | {', '.join(confluence)}")
            return {
                "direction":    "SELL",
                "entry_price":  close_price,
                "sl_price":     sl_price,
                "tp1_price":    tp1_price,
                "tp2_price":    tp2_price,
                "sl_pips":      sl_pips,
                "tp1_pips":     tp1_pips,
                "tp2_pips":     tp2_pips,
                "strategy":     STRATEGY_NAME,
                "version":      VERSION,
                "score":        score,
                "confluence":   confluence,
                "swept_level":  swept_level,
            }

    return None
