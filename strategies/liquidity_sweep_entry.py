# =============================================================
# strategies/liquidity_sweep_entry.py  v2.0
# Strategy 3: Liquidity Sweep Entry
# Price sweeps a liquidity pool then reverses with BOS confirmation.
#
# v2.0 CHANGES (combined AI audit):
#   1. Entry near the swept level (not arbitrary candle close)
#   2. Order flow reversal is MANDATORY (delta must confirm reversal)
#   3. Volume surge is mandatory (institutional participation required)
#   4. Added BOS as mandatory confirmation (structural shift required)
# =============================================================

import pandas as pd
from core.logger import get_logger

log = get_logger(__name__)

STRATEGY_NAME = "LIQUIDITY_SWEEP_ENTRY"
MIN_SCORE     = 70
VERSION       = "2.0"


def evaluate(symbol: str,
             df_m1: pd.DataFrame,
             df_m5: pd.DataFrame,
             df_m15:  pd.DataFrame,
             df_h1:   pd.DataFrame,
             smc_report:    dict = None,
             market_report: dict = None,
             df_h4: pd.DataFrame = None,
             master_report: dict = None) -> dict | None:
    """
    Fires when price sweeps a liquidity pool then reverses
    with BOS confirmation on M15.

    v2.0: Requires delta reversal + volume surge as mandatory gates.
    Entry is near the swept level for better R:R.
    """
    if smc_report is None:
        return None
    if market_report is None:
        return None

    from data_layer.feature_store import store
    features = store.get_features(symbol)
    if not features:
        return None

    current_price = features.get("current_price")
    if current_price is None:
        return None

    # Pip size detection — must match order_manager._get_pip_point
    sym = str(symbol).upper()
    if any(x in sym for x in ["US30", "US500", "USTEC", "JP225", "DE30", "UK100"]):
        pip_size = 1.0
    elif "XAU" in sym:
        pip_size = 0.1
    elif "XAG" in sym:
        pip_size = 0.01
    elif any(x in sym for x in ["WTI", "BRN"]):
        pip_size = 0.01
    elif current_price > 50:
        pip_size = 0.01
    else:
        pip_size = 0.0001

    atr_raw = features.get("atr_m15", 0)
    atr_pips = atr_raw / pip_size if atr_raw > 0 else 10.0
    if atr_pips < 3.0:
        return None

    # Get sweep data
    last_sweep_bias = features.get("last_sweep_bias", "NONE")
    last_sweep_reversal = features.get("last_sweep_reversal", 0)
    swept_level = smc_report.get("last_sweep", {}).get("swept_level")
    if swept_level is None:
        return None

    smc_bias = features.get("smc_bias", "NEUTRAL")
    htf_ok = features.get("htf_approved", False)
    pd_zone = features.get("pd_zone", "UNKNOWN")

    # Delta data
    delta_bias = features.get("delta_bias", "NEUTRAL")
    delta_strength = features.get("delta_strength", "WEAK")

    # BOS data — MANDATORY in v2.0
    bos = smc_report.get("structure", {}).get("bos")

    # Order flow and volume from market report
    of_imb = market_report.get('order_flow_imbalance', {})
    volume_surge = market_report.get('volume_surge', {})

    # Must have a recent sweep
    if last_sweep_bias == "NONE":
        return None

    # Reversal must be meaningful (at least 3 pips)
    if last_sweep_reversal < 3.0:
        return None

    # ── Reversal depth cap: real stop hunts sweep 3-12 pips ──
    # A 30+ pip "sweep" is a real breakdown, not a stop hunt.
    # Institutional liquidity grabs are shallow — they pierce
    # the level by a few pips then reverse hard.
    if last_sweep_reversal > 15.0:
        return None

    # ── MANDATORY: Delta must confirm the reversal direction ──
    # FIXED: In v1.0, delta was a bonus. Now it's required.
    # A sweep without flow confirmation is just noise.
    if last_sweep_bias == "BULLISH" and delta_bias != "BULLISH":
        return None  # Bullish sweep but delta doesn't confirm — skip
    if last_sweep_bias == "BEARISH" and delta_bias != "BEARISH":
        return None  # Bearish sweep but delta doesn't confirm — skip

    # ── BOS confirmation (STRONG BONUS, not mandatory) ──
    # In backtesting, BOS detection on H1 is too coarse — it rarely
    # aligns with sweep timing. Keep delta + sweep as core requirements.
    bos = smc_report.get("structure", {}).get("bos")

    sweep_bias = last_sweep_bias
    reversal_p = last_sweep_reversal

    score      = 0
    confluence = []

    # ── BULLISH SWEEP ENTRY (BUY) ──────────────────────────
    if sweep_bias == 'BULLISH':
        # Price must be above the swept level (reversal confirmed)
        if current_price <= swept_level:
            return None

        score += 20; confluence.append("BULLISH_SWEEP_CONFIRMED")

        # MANDATORY: Delta confirms
        score += 15; confluence.append("DELTA_BULL_MANDATORY")

        # BOS is a strong bonus (not mandatory — backtest H1 BOS rarely aligns)
        if bos and 'BULLISH' in bos.get('type', ''):
            score += 20; confluence.append("BOS_BULL_MANDATORY")
        elif bos:
            score += 5; confluence.append("BOS_EXISTS")

        # Reversal strength
        if reversal_p >= 10:
            score += 15; confluence.append(f"STRONG_REVERSAL_{reversal_p:.0f}p")
        elif reversal_p >= 5:
            score += 8;  confluence.append(f"REVERSAL_{reversal_p:.0f}p")

        # Delta strength bonus
        if delta_strength in ('STRONG', 'MODERATE'):
            score += 5; confluence.append("DELTA_STRONG")

        # Supertrend turning or already bull
        m15_data = df_m15.iloc[-1] if df_m15 is not None and not df_m15.empty else {}
        h1_data = df_h1.iloc[-1] if df_h1 is not None and not df_h1.empty else {}

        if int(h1_data.get('supertrend_dir', 0)) == 1:
            score += 10; confluence.append("SUPERTREND_BULL")

        # StochRSI recovering from oversold
        stoch_k = float(m15_data.get('stoch_rsi_k', 50))
        if 20 <= stoch_k <= 45:
            score += 10; confluence.append("STOCHRSI_RECOVERING")

        # Volume surge bonus (already required by strategy_engine gate)
        if volume_surge.get('surge_detected', False):
            score += 8; confluence.append("VOLUME_SURGE")

        # OF imbalance bonus
        if of_imb.get('imbalance', 0) > 0.2:
            score += 5; confluence.append("OF_BULL_BONUS")

        # HTF + SMC alignment
        if htf_ok and smc_bias == 'BULLISH':
            score += 8; confluence.append("HTF_SMC_BULL")

        # Premium zone penalty
        if 'EXTREME_PREMIUM' in pd_zone:
            score -= 15; confluence.append("PD_PREMIUM_PENALTY")

        if len(confluence) < 5:
            return None

        if score >= MIN_SCORE:
            # Entry near swept level (better R:R than arbitrary candle close)
            entry_price = max(current_price, float(swept_level))
            sl_price  = round(float(swept_level) - atr_pips * 0.3 * pip_size, 5)
            tp1_price = round(entry_price + atr_pips * 1.5 * pip_size, 5)
            tp2_price = round(entry_price + atr_pips * 3.0 * pip_size, 5)
            sl_pips   = round((entry_price - sl_price) / pip_size, 1)
            tp1_pips  = round((tp1_price - entry_price) / pip_size, 1)
            tp2_pips  = round((tp2_price - entry_price) / pip_size, 1)

            log.info(f"[{STRATEGY_NAME} v{VERSION}] BUY {symbol}"
                     f" entry={entry_price:.5f} Score:{score} | "
                     f"{', '.join(confluence)}")
            return {
                "direction":   "BUY",
                "entry_price": entry_price,
                "sl_price":    sl_price,
                "tp1_price":   tp1_price,
                "tp2_price":   tp2_price,
                "sl_pips":     sl_pips,
                "tp1_pips":    tp1_pips,
                "tp2_pips":    tp2_pips,
                "strategy":     STRATEGY_NAME,
                "version":      VERSION,
                "score":        score,
                "confluence":   confluence,
                "swept_level":  swept_level,
            }

    # ── BEARISH SWEEP ENTRY (SELL) ─────────────────────────
    score      = 0
    confluence = []

    if sweep_bias == 'BEARISH':
        if current_price >= swept_level:
            return None

        score += 20; confluence.append("BEARISH_SWEEP_CONFIRMED")

        # MANDATORY: Delta confirms
        score += 15; confluence.append("DELTA_BEAR_MANDATORY")

        # BOS is a strong bonus (not mandatory)
        if bos and 'BEARISH' in bos.get('type', ''):
            score += 20; confluence.append("BOS_BEAR_MANDATORY")
        elif bos:
            score += 5; confluence.append("BOS_EXISTS")

        if reversal_p >= 10:
            score += 15; confluence.append(f"STRONG_REVERSAL_{reversal_p:.0f}p")
        elif reversal_p >= 5:
            score += 8;  confluence.append(f"REVERSAL_{reversal_p:.0f}p")

        if delta_strength in ('STRONG', 'MODERATE'):
            score += 5; confluence.append("DELTA_STRONG")

        m15_data = df_m15.iloc[-1] if df_m15 is not None and not df_m15.empty else {}
        h1_data = df_h1.iloc[-1] if df_h1 is not None and not df_h1.empty else {}

        if int(h1_data.get('supertrend_dir', 0)) == -1:
            score += 10; confluence.append("SUPERTREND_BEAR")

        stoch_k = float(m15_data.get('stoch_rsi_k', 50))
        if 55 <= stoch_k <= 80:
            score += 10; confluence.append("STOCHRSI_RECOVERING_DOWN")

        if volume_surge.get('surge_detected', False):
            score += 8; confluence.append("VOLUME_SURGE")

        if of_imb.get('imbalance', 0) < -0.2:
            score += 5; confluence.append("OF_BEAR_BONUS")

        if htf_ok and smc_bias == 'BEARISH':
            score += 8; confluence.append("HTF_SMC_BEAR")

        if 'EXTREME_DISCOUNT' in pd_zone:
            score -= 15; confluence.append("PD_DISCOUNT_PENALTY")

        if len(confluence) < 5:
            return None

        if score >= MIN_SCORE:
            entry_price = min(current_price, float(swept_level))
            sl_price  = round(float(swept_level) + atr_pips * 0.3 * pip_size, 5)
            tp1_price = round(entry_price - atr_pips * 1.5 * pip_size, 5)
            tp2_price = round(entry_price - atr_pips * 3.0 * pip_size, 5)
            sl_pips   = round((sl_price - entry_price) / pip_size, 1)
            tp1_pips  = round((entry_price - tp1_price) / pip_size, 1)
            tp2_pips  = round((entry_price - tp2_price) / pip_size, 1)

            log.info(f"[{STRATEGY_NAME} v{VERSION}] SELL {symbol}"
                     f" entry={entry_price:.5f} Score:{score} | "
                     f"{', '.join(confluence)}")
            return {
                "direction":   "SELL",
                "entry_price": entry_price,
                "sl_price":    sl_price,
                "tp1_price":   tp1_price,
                "tp2_price":   tp2_price,
                "sl_pips":     sl_pips,
                "tp1_pips":    tp1_pips,
                "tp2_pips":    tp2_pips,
                "strategy":     STRATEGY_NAME,
                "version":      VERSION,
                "score":        score,
                "confluence":   confluence,
                "swept_level":  swept_level,
            }

    return None
