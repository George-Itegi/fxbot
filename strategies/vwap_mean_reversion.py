# strategies/vwap_mean_reversion.py v1.1
import pandas as pd
from core.logger import get_logger

log = get_logger(__name__)
STRATEGY_NAME = "VWAP_MEAN_REVERSION"
MIN_SCORE = 60   # Lowered — intraday VWAP setups score 55-70 realistically
VERSION = "1.1"
MIN_RR = 1.5

def evaluate(symbol, df_m1=None, df_m5=None, df_m15=None, df_h1=None,
             market_report=None,
             smc_report=None, external_data=None, master_report=None,
             df_h4=None):
    if df_m15 is None or df_h1 is None or market_report is None:
        return None
    m15 = df_m15.iloc[-1]
    h1 = df_h1.iloc[-1]
    close_price = float(m15['close'])
    pip_size = 0.01 if close_price > 50 else 0.0001
    atr_pips = float(m15['atr']) / pip_size
    if atr_pips < 2.0:
        return None
    adx = float(m15.get('adx', 30))
    # Only block extreme trending (ADX>45), not normal trending (ADX 25-35)
    # ADX>30 was blocking this strategy 80% of the time in London/NY
    if adx > 45:
        return None
    vwap_data = market_report.get('vwap', {})
    vwap = float(vwap_data.get('vwap', 0))
    vwap_pos = vwap_data.get('position', '')
    pip_from_v = float(vwap_data.get('pip_from_vwap', 0))
    if vwap == 0:
        return None
    prof = market_report.get('profile', {})
    poc = float(prof.get('poc', 0))
    vah = float(prof.get('vah', 0))
    val = float(prof.get('val', 0))
    va_pos = prof.get('price_position', '')
    pd_zone = ''
    htf_ok = True
    if smc_report:
        pd_zone = smc_report.get('premium_discount', {}).get('zone', '')
        htf_ok = smc_report.get('htf_alignment', {}).get('approved', True)
    vix = float(external_data.get('vix', 20)) if external_data else 20.0
    fg_score = float(external_data.get('fear_greed', 50)) if external_data else 50.0
    if vix > 25:
        return None
    master_bias = master_report.get('combined_bias', '') if master_report else ''

    # BUY SETUP
    if 'BELOW_VWAP' in vwap_pos or va_pos == 'BELOW_VAL':
        if master_bias == 'BEARISH':
            log.info(f"[{STRATEGY_NAME}] BUY blocked — master BEARISH")
            return None
        if 'PREMIUM' in pd_zone and 'DISCOUNT' not in pd_zone:
            return None
        score = 0
        confluence = []
        score += 20; confluence.append("PRICE_BELOW_VWAP")
        if 5 <= abs(pip_from_v) <= 30:
            score += 15; confluence.append(f"VWAP_DIST_{abs(pip_from_v):.0f}p")
        if poc > close_price:
            poc_dist = (poc - close_price) / pip_size
            if poc_dist <= 40:
                score += 15; confluence.append(f"POC_ABOVE_{poc_dist:.0f}p")
        if int(h1.get('supertrend_dir', 0)) == 1:
            score += 15; confluence.append("SUPERTREND_BULL")
        stoch_k = float(m15.get('stoch_rsi_k', 50))
        stoch_d = float(m15.get('stoch_rsi_d', 50))
        if stoch_k < 30 and stoch_k > stoch_d:
            score += 20; confluence.append("STOCHRSI_CROSS_UP")
        elif stoch_k < 40:
            score += 10; confluence.append("STOCHRSI_OVERSOLD")
        if fg_score <= 30:
            score += 10; confluence.append(f"FEAR_{fg_score:.0f}")
        if 'BAND' in vwap_pos:
            score += 10; confluence.append("AT_VWAP_BAND")
        if not htf_ok:
            score -= 15; confluence.append("HTF_REJECTED")
        if len(confluence) < 5: return None
        if score >= MIN_SCORE:
            sl_price = round(close_price - atr_pips * 1.2 * pip_size, 5)
            tp1_price = round(vwap, 5)
            tp2_price = round(poc if poc > close_price else vah, 5)
            sl_pips = round((close_price - sl_price) / pip_size, 1)
            tp1_pips = round((tp1_price - close_price) / pip_size, 1)
            tp2_pips = round((tp2_price - close_price) / pip_size, 1)
            if sl_pips <= 0 or tp1_pips < sl_pips * MIN_RR:
                log.info(f"[{STRATEGY_NAME}] BUY rejected R:R TP:{tp1_pips}/SL:{sl_pips}")
                return None
            log.info(f"[{STRATEGY_NAME}] BUY {symbol} Score:{score} | {', '.join(confluence)}")
            return {"direction": "BUY", "entry_price": close_price,
                    "sl_price": sl_price, "tp1_price": tp1_price, "tp2_price": tp2_price,
                    "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips,
                    "strategy": STRATEGY_NAME, "version": VERSION,
                    "score": score, "confluence": confluence}

    # SELL SETUP
    if 'ABOVE_VWAP' in vwap_pos or va_pos == 'ABOVE_VAH':
        if master_bias == 'BULLISH':
            log.info(f"[{STRATEGY_NAME}] SELL blocked — master BULLISH")
            return None
        if 'DISCOUNT' in pd_zone and 'PREMIUM' not in pd_zone:
            return None
        score = 0
        confluence = []
        score += 20; confluence.append("PRICE_ABOVE_VWAP")
        if 5 <= abs(pip_from_v) <= 30:
            score += 15; confluence.append(f"VWAP_DIST_{abs(pip_from_v):.0f}p")
        if poc < close_price:
            poc_dist = (close_price - poc) / pip_size
            if poc_dist <= 40:
                score += 15; confluence.append(f"POC_BELOW_{poc_dist:.0f}p")
        if int(h1.get('supertrend_dir', 0)) == -1:
            score += 15; confluence.append("SUPERTREND_BEAR")
        stoch_k = float(m15.get('stoch_rsi_k', 50))
        stoch_d = float(m15.get('stoch_rsi_d', 50))
        if stoch_k > 70 and stoch_k < stoch_d:
            score += 20; confluence.append("STOCHRSI_CROSS_DOWN")
        elif stoch_k > 60:
            score += 10; confluence.append("STOCHRSI_OVERBOUGHT")
        if fg_score >= 70:
            score += 10; confluence.append(f"GREED_{fg_score:.0f}")
        if 'BAND' in vwap_pos:
            score += 10; confluence.append("AT_VWAP_BAND")
        if not htf_ok:
            score -= 15; confluence.append("HTF_REJECTED")
        if len(confluence) < 5: return None
        if score >= MIN_SCORE:
            sl_price = round(close_price + atr_pips * 1.2 * pip_size, 5)
            tp1_price = round(vwap, 5)
            tp2_price = round(poc if poc < close_price else val, 5)
            sl_pips = round((sl_price - close_price) / pip_size, 1)
            tp1_pips = round((close_price - tp1_price) / pip_size, 1)
            tp2_pips = round((close_price - tp2_price) / pip_size, 1)
            if sl_pips <= 0 or tp1_pips < sl_pips * MIN_RR:
                log.info(f"[{STRATEGY_NAME}] SELL rejected R:R TP:{tp1_pips}/SL:{sl_pips}")
                return None
            log.info(f"[{STRATEGY_NAME}] SELL {symbol} Score:{score} | {', '.join(confluence)}")
            return {"direction": "SELL", "entry_price": close_price,
                    "sl_price": sl_price, "tp1_price": tp1_price, "tp2_price": tp2_price,
                    "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips,
                    "strategy": STRATEGY_NAME, "version": VERSION,
                    "score": score, "confluence": confluence}

    return None