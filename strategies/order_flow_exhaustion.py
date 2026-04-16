import pandas as pd
from core.logger import get_logger
from data_layer.feature_store import store

log = get_logger(__name__)

STRATEGY_NAME = "ORDER_FLOW_EXHAUSTION"
MIN_SCORE     = 65
VERSION       = "1.0"

def evaluate(symbol: str,
             df_m1: pd.DataFrame = None,
             df_m5: pd.DataFrame = None,
             df_m15:  pd.DataFrame = None,
             df_h1:   pd.DataFrame = None,
             smc_report:    dict = None,
             market_report: dict = None,
             df_h4: pd.DataFrame = None,
             master_report: dict = None) -> dict | None:
    """
    Fires when price makes a new swing level but order flow delta
    is exhausted (divergence).
    """
    # Use the Feature Store for consistent data
    features = store.get_features(symbol)
    if not features: return None

    current_price = features.get("current_price")
    if current_price is None: return None

    delta_rolling = features.get("delta_rolling", 0)
    delta_bias    = features.get("delta_bias", "NEUTRAL")
    
    # Need ATR for SL/TP
    # Ensure df_m15 is not empty before accessing iloc[-1]
    if df_m15.empty:
        return None
    atr_value = df_m15.iloc[-1].get("atr")
    if atr_value is None:
        return None
    pip_size = 0.01 if current_price > 50 else 0.0001
    atr_pips = float(atr_value) / pip_size

    # Get SMC context
    last_sweep = smc_report.get("last_sweep")
    pd_zone    = features.get("pd_zone", "UNKNOWN")
    
    score = 0
    confluence = []

    # Ensure last_sweep is not None before proceeding with sweep-related logic
    if last_sweep is None: return None
    reversal_pips = last_sweep.get("reversal_pips")
    if reversal_pips is None: return None

    # ── BEARISH EXHAUSTION (SELL) ──────────────────────────
    # Price sweeps a high (buyside liquidity), but delta is negative (aggressive selling)
    if last_sweep.get("type") == "BUYSIDE_LIQUIDITY":
        if delta_bias == "BEARISH" and delta_rolling < -50:
            score += 40; confluence.append("BEARISH_DELTA_DIVERGENCE")
            
            if "PREMIUM" in pd_zone:
                score += 20; confluence.append("PREMIUM_ZONE_CONFLUENCE")
            
            if reversal_pips > 2:
                score += 15; confluence.append("REVERSAL_CONFIRMED")

            if score >= MIN_SCORE:
                return {
                    "direction":    "SELL",
                    "entry_price":  current_price,
                    "sl_price":     round(current_price + atr_pips * 1.0 * pip_size, 5),
                    "tp1_price":    round(current_price - atr_pips * 1.5 * pip_size, 5),
                    "tp2_price":    round(current_price - atr_pips * 3.0 * pip_size, 5),
                    "sl_pips":      round(atr_pips * 1.0, 1),
                    "tp1_pips":     round(atr_pips * 1.5, 1),
                    "tp2_pips":     round(atr_pips * 3.0, 1),
                    "strategy":     STRATEGY_NAME,
                    "version":      VERSION,
                    "score":        score,
                    "confluence":   confluence
                }

    # ── BULLISH EXHAUSTION (BUY) ───────────────────────────
    # Price sweeps a low (sellside liquidity), but delta is positive (aggressive buying)
    if last_sweep.get("type") == "SELLSIDE_LIQUIDITY":
        if delta_bias == "BULLISH" and delta_rolling > 50:
            score += 40; confluence.append("BULLISH_DELTA_DIVERGENCE")
            
            if "DISCOUNT" in pd_zone:
                score += 20; confluence.append("DISCOUNT_ZONE_CONFLUENCE")
            
            if reversal_pips > 2:
                score += 15; confluence.append("REVERSAL_CONFIRMED")

            if score >= MIN_SCORE:
                return {
                    "direction":    "BUY",
                    "entry_price":  current_price,
                    "sl_price":     round(current_price - atr_pips * 1.0 * pip_size, 5),
                    "tp1_price":    round(current_price + atr_pips * 1.5 * pip_size, 5),
                    "tp2_price":    round(current_price + atr_pips * 3.0 * pip_size, 5),
                    "sl_pips":      round(atr_pips * 1.0, 1),
                    "tp1_pips":     round(atr_pips * 1.5, 1),
                    "tp2_pips":     round(atr_pips * 3.0, 1),
                    "strategy":     STRATEGY_NAME,
                    "version":      VERSION,
                    "score":        score,
                    "confluence":   confluence
                }

    return None
