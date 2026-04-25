# =============================================================
# strategies/ema_cross_momentum.py  v1.0
# Strategy 12: EMA Crossover Momentum (TREND_FOLLOWING group)
#
# Purpose: Second TREND_FOLLOWING strategy to remove single point
# of failure. Unlike TREND_CONTINUATION which waits for pullback
# to EMA21, this strategy enters on the CROSSOVER itself — it
# catches trend STARTS that TC misses when price never pulls back.
#
# Entry logic:
#   1. H4 EMA9/21 crossover (trend shift detection)
#   2. H1 RSI momentum (above 55 BUY / below 45 SELL)
#   3. M15 ADX > 20 (trend strength filter)
#   4. Rolling delta must confirm direction
#   5. Supertrend on H1 alignment
#
# Win rate target: 45-55%
# Best session: LONDON_SESSION, NY_LONDON_OVERLAP, LONDON_OPEN
# Best state:  TRENDING_STRONG, BREAKOUT_ACCEPTED
# =============================================================

import pandas as pd
from core.logger import get_logger

log = get_logger(__name__)

STRATEGY_NAME = "EMA_CROSS_MOMENTUM"
MIN_SCORE     = 70
VERSION       = "1.0"

# --- Parameters ---
RSI_MOMENTUM_BUY  = 55    # H1 RSI above this = bullish momentum
RSI_MOMENTUM_SELL = 45    # H1 RSI below this = bearish momentum
ADX_MIN           = 20    # M15 ADX must exceed this (trend strength)
CROSSBAR_WINDOW   = 5     # Bars to look back for recent crossover


def _get_pip_size(symbol: str, price: float) -> float:
    from core.pip_utils import get_pip_size as _gps
    return _gps(symbol, price)


def _detect_recent_cross(df: pd.DataFrame, direction: str,
                         window: int = CROSSBAR_WINDOW) -> dict:
    """
    Detect if EMA9 crossed over/under EMA21 within the last N bars.
    Returns dict with crossover info or None.
    """
    if df is None or len(df) < window + 2:
        return {"crossed": False, "bars_ago": 999, "strength": 0}

    crossed = False
    bars_ago = window + 1

    for i in range(len(df) - window - 1, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i - 1]

        curr_ema9  = float(curr.get('ema_9', 0))
        curr_ema21 = float(curr.get('ema_21', 0))
        prev_ema9  = float(prev.get('ema_9', 0))
        prev_ema21 = float(prev.get('ema_21', 0))

        idx_from_end = len(df) - 1 - i
        if idx_from_end > window:
            break

        if direction == "BUY":
            # Bullish cross: EMA9 crosses ABOVE EMA21
            if prev_ema9 <= prev_ema21 and curr_ema9 > curr_ema21:
                crossed = True
                bars_ago = idx_from_end
                break
        elif direction == "SELL":
            # Bearish cross: EMA9 crosses BELOW EMA21
            if prev_ema9 >= prev_ema21 and curr_ema9 < curr_ema21:
                crossed = True
                bars_ago = idx_from_end
                break

    # Scoring: more recent crossover = stronger signal
    if not crossed:
        return {"crossed": False, "bars_ago": 999, "strength": 0}

    if bars_ago <= 1:
        strength = 30
    elif bars_ago <= 3:
        strength = 20
    else:
        strength = 10

    return {"crossed": True, "bars_ago": bars_ago, "strength": strength}


def _check_h4_alignment(df_h4: pd.DataFrame, direction: str) -> dict:
    """Check H4 EMA alignment for higher-timeframe confirmation."""
    if df_h4 is None or len(df_h4) < 20:
        return {"aligned": False, "score": 0}

    last = df_h4.iloc[-1]
    ema9  = float(last.get('ema_9', 0))
    ema21 = float(last.get('ema_21', 0))
    ema50 = float(last.get('ema_50', 0))
    st_dir = int(last.get('supertrend_dir', 0))
    close = float(last['close'])

    score = 0
    if direction == "BUY":
        if ema9 > ema21 > ema50 and close > ema9:
            score += 25
        elif ema9 > ema21 and close > ema21:
            score += 15
        elif close > ema21:
            score += 5
        if st_dir == 1:
            score += 10
    elif direction == "SELL":
        if ema9 < ema21 < ema50 and close < ema9:
            score += 25
        elif ema9 < ema21 and close < ema21:
            score += 15
        elif close < ema21:
            score += 5
        if st_dir == -1:
            score += 10

    return {"aligned": score >= 15, "score": score}


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
    EMA Crossover Momentum Strategy:
    Enters on fresh EMA9/21 crossover with momentum confirmation.

    Key difference from TREND_CONTINUATION:
      - TC waits for pullback to EMA21 (misses strong moves)
      - This enters on the crossover itself (catches trend starts)
    """
    if df_m15 is None or df_h1 is None or df_h4 is None:
        return None
    if len(df_m15) < 30 or len(df_h1) < 20 or len(df_h4) < 20:
        return None
    if market_report is None:
        return None

    close_price = float(df_m15.iloc[-1]['close'])
    pip_size = _get_pip_size(symbol, close_price)
    atr_pips = float(df_m15.iloc[-1].get('atr', 0)) / pip_size

    if atr_pips < 2.0:
        return None

    score = 0
    confluence = []

    # ── Step 1: Detect H4 EMA crossover (mandatory) ──────
    h4_bull_cross = _detect_recent_cross(df_h4, "BUY")
    h4_bear_cross = _detect_recent_cross(df_h4, "SELL")

    if h4_bull_cross['crossed'] and h4_bull_cross['strength'] >= h4_bear_cross['strength']:
        direction = "BUY"
        score += h4_bull_cross['strength']
        confluence.append(f"H4_BULL_CROSS({h4_bull_cross['bars_ago']}bars)")
    elif h4_bear_cross['crossed'] and h4_bear_cross['strength'] >= h4_bull_cross['strength']:
        direction = "SELL"
        score += h4_bear_cross['strength']
        confluence.append(f"H4_BEAR_CROSS({h4_bear_cross['bars_ago']}bars)")
    else:
        # No recent crossover — check if H4 is strongly aligned
        # (can enter if trend is fresh, even if cross was slightly older)
        h4_bull = _check_h4_alignment(df_h4, "BUY")
        h4_bear = _check_h4_alignment(df_h4, "SELL")
        if h4_bull['score'] > h4_bear['score'] and h4_bull['score'] >= 25:
            direction = "BUY"
            score += 15
            confluence.append("H4_STRONG_BULL_ALIGN")
        elif h4_bear['score'] > h4_bull['score'] and h4_bear['score'] >= 25:
            direction = "SELL"
            score += 15
            confluence.append("H4_STRONG_BEAR_ALIGN")
        else:
            return None

    # ── Step 2: H1 RSI momentum (mandatory) ──────────────
    h1 = df_h1.iloc[-1]
    h1_rsi = float(h1.get('rsi', 50))

    if direction == "BUY":
        if h1_rsi > RSI_MOMENTUM_BUY:
            score += 15
            confluence.append(f"H1_RSI_BULL_{h1_rsi:.1f}")
        elif h1_rsi > 50:
            score += 8
            confluence.append(f"H1_RSI_NEUTRAL_BULL_{h1_rsi:.1f}")
        else:
            return None  # RSI doesn't confirm bullish momentum
    elif direction == "SELL":
        if h1_rsi < RSI_MOMENTUM_SELL:
            score += 15
            confluence.append(f"H1_RSI_BEAR_{h1_rsi:.1f}")
        elif h1_rsi < 50:
            score += 8
            confluence.append(f"H1_RSI_NEUTRAL_BEAR_{h1_rsi:.1f}")
        else:
            return None  # RSI doesn't confirm bearish momentum

    # ── Step 3: M15 ADX trend strength (mandatory) ───────
    m15 = df_m15.iloc[-1]
    adx = float(m15.get('adx', 0))

    if adx < ADX_MIN:
        return None  # Not trending enough
    elif adx >= 35:
        score += 15
        confluence.append(f"ADX_STRONG_{adx:.1f}")
    elif adx >= 25:
        score += 10
        confluence.append(f"ADX_MODERATE_{adx:.1f}")
    else:
        score += 5
        confluence.append(f"ADX_WEAK_{adx:.1f}")

    # ── Step 4: Rolling delta confirmation (mandatory) ───
    rolling_delta = market_report.get('rolling_delta', {})
    delta_bias = rolling_delta.get('bias', 'NEUTRAL')
    of_imb = market_report.get('order_flow_imbalance', {})
    imb = of_imb.get('imbalance', 0)
    of_strength = of_imb.get('strength', 'NONE')

    delta_confirms = False
    if direction == "BUY":
        if delta_bias == "BULLISH":
            delta_confirms = True
            score += 12
            confluence.append("DELTA_BULL")
        elif imb > 0.1 and of_strength in ('STRONG', 'MODERATE', 'EXTREME'):
            delta_confirms = True
            score += 10
            confluence.append(f"OF_BULL_{imb:+.2f}")
    elif direction == "SELL":
        if delta_bias == "BEARISH":
            delta_confirms = True
            score += 12
            confluence.append("DELTA_BEAR")
        elif imb < -0.1 and of_strength in ('STRONG', 'MODERATE', 'EXTREME'):
            delta_confirms = True
            score += 10
            confluence.append(f"OF_BEAR_{imb:+.2f}")

    if not delta_confirms:
        return None  # No flow confirmation

    # ── Step 5: H1 Supertrend alignment (bonus) ──────────
    h1_st = int(h1.get('supertrend_dir', 0))
    if (direction == "BUY" and h1_st == 1) or (direction == "SELL" and h1_st == -1):
        score += 8
        confluence.append("H1_SUPERTREND_ALIGNED")

    # ── Step 6: H4 alignment (bonus) ─────────────────────
    h4_align = _check_h4_alignment(df_h4, direction)
    if h4_align['aligned'] and h4_align['score'] > 0:
        score += h4_align['score']
        if h4_align['score'] >= 25:
            confluence.append(f"H4_TREND_ALIGN({h4_align['score']}pts)")

    # ── Step 7: M5 candle confirmation (bonus) ──────────
    if df_m5 is not None and len(df_m5) >= 3:
        m5_last = df_m5.iloc[-1]
        m5_body = m5_last['close'] - m5_last['open']
        if (direction == "BUY" and m5_body > 0) or \
           (direction == "SELL" and m5_body < 0):
            score += 5
            confluence.append("M5_DIRECTIONAL")

    # ── Step 8: Volume surge (bonus) ────────────────────
    if market_report:
        surge = market_report.get('volume_surge', {})
        if surge.get('surge_detected', False):
            score += 5
            confluence.append("VOLUME_SURGE")

    # ── Step 9: SMC/HTF confirmation (bonus) ────────────
    if smc_report:
        htf = smc_report.get('htf_alignment', {})
        if htf.get('approved', False):
            h4_bias = htf.get('h4_bias', '')
            if (direction == "BUY" and h4_bias == "BULLISH") or \
               (direction == "SELL" and h4_bias == "BEARISH"):
                score += 8
                confluence.append("HTF_ALIGNED")

    # ── Choppy market penalty ────────────────────────────
    if master_report:
        momentum = master_report.get('momentum', {})
        if momentum.get('is_choppy', False):
            score -= 15
            confluence.append("CHOPPY_PENALTY")

    # ── Fibonacci confluence bonus ──────────────────────
    try:
        from backtest.fib_builder import build_fib_report, check_fib_confluence
        fib_report = build_fib_report(df_h1=df_h1, df_h4=df_h4, current_price=close_price)
        fib_check = check_fib_confluence(close_price, direction, fib_report, pip_size)
        if fib_check['fib_bonus'] > 0:
            score += fib_check['fib_bonus']
            confluence.extend(fib_check['confluence'])
    except Exception:
        pass

    if len(confluence) < 5:
        return None

    # ── Score threshold ──────────────────────────────────
    if score < MIN_SCORE:
        return None

    # ── Calculate SL/TP ──────────────────────────────────
    entry = close_price

    sl_pips = round(atr_pips * 1.5, 1)
    sl_pips = max(sl_pips, 3.0)

    tp1_pips = round(sl_pips * 2.5, 1)
    tp2_pips = round(sl_pips * 4.0, 1)

    if direction == "BUY":
        sl_price  = round(entry - sl_pips * pip_size, 5)
        tp1_price = round(entry + tp1_pips * pip_size, 5)
        tp2_price = round(entry + tp2_pips * pip_size, 5)
    else:
        sl_price  = round(entry + sl_pips * pip_size, 5)
        tp1_price = round(entry - tp1_pips * pip_size, 5)
        tp2_price = round(entry - tp2_pips * pip_size, 5)

    log.info(f"[{STRATEGY_NAME} v{VERSION}] {direction} {symbol}"
             f" entry={entry:.5f} Score:{score} | "
             f"{', '.join(confluence)}")

    return {
        "direction":   direction,
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
        "spread":      0,
    }
