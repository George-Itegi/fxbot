# =============================================================
# strategies/trend_continuation.py  v2.0
# Strategy 9: Trend Continuation (Ride the London/NY Expansion)
#
# v2.0 CHANGES (combined AI audit):
#   1. M15 pullback to EMA is MANDATORY (not soft fallback)
#   2. Order flow must confirm direction (mandatory)
#   3. Entry at the pullback EMA level (not arbitrary candle close)
#   4. Volume surge or strong OF imbalance required
# =============================================================

import pandas as pd
from core.logger import get_logger

log = get_logger(__name__)

STRATEGY_NAME = "TREND_CONTINUATION"
MIN_SCORE     = 80
VERSION       = "2.1"


def _get_pip_size(symbol: str, price: float) -> float:
    from core.pip_utils import get_pip_size as _gps
    return _gps(symbol, price)


def _detect_rejection_candle(df_m5: pd.DataFrame, direction: str) -> str:
    if df_m5 is None or len(df_m5) < 3:
        return "NONE"

    last = df_m5.iloc[-1]
    body = abs(last['close'] - last['open'])
    full_range = last['high'] - last['low']

    if full_range <= 0:
        return "NONE"

    upper_wick = last['high'] - max(last['close'], last['open'])
    lower_wick = min(last['close'], last['open']) - last['low']
    body_ratio = body / full_range

    if lower_wick > body * 2.0 and upper_wick < body * 0.5 and body_ratio < 0.4:
        if last['close'] > last['open']:
            return "REJECTION_BULL"

    if upper_wick > body * 2.0 and lower_wick < body * 0.5 and body_ratio < 0.4:
        if last['close'] < last['open']:
            return "REJECTION_BEAR"

    if df_m5 is not None and len(df_m5) >= 2:
        prev = df_m5.iloc[-2]
        prev_body = prev['close'] - prev['open']
        curr_body = last['close'] - last['open']
        if prev_body < 0 and curr_body > 0 and curr_body > abs(prev_body) * 1.2:
            return "REJECTION_BULL"
        if prev_body > 0 and curr_body < 0 and abs(curr_body) > prev_body * 1.2:
            return "REJECTION_BEAR"

    return "NONE"


def _is_pullback_to_ema(df_m15: pd.DataFrame, direction: str, symbol: str = '') -> dict:
    if df_m15 is None or len(df_m15) < 20:
        return {"is_pullback": False, "level": 0, "ema_type": "", "dist_pips": 999}

    last = df_m15.iloc[-1]
    ema21 = float(last.get('ema_21', 0))
    ema50 = float(last.get('ema_50', 0))
    close = float(last['close'])
    low = float(last['low'])
    high = float(last['high'])
    pip_size = _get_pip_size(symbol, close)

    tolerance = 2.0  # pips

    if direction == "BUY":
        dist_ema21 = (close - ema21) / pip_size
        dist_ema50 = (close - ema50) / pip_size

        if abs(dist_ema21) <= tolerance:
            return {"is_pullback": True, "level": ema21, "ema_type": "EMA21",
                    "dist_pips": dist_ema21}
        if abs(dist_ema50) <= tolerance * 1.5:
            return {"is_pullback": True, "level": ema50, "ema_type": "EMA50",
                    "dist_pips": dist_ema50}
        if low <= ema21 * 1.001 and close > ema21:
            return {"is_pullback": True, "level": ema21, "ema_type": "EMA21_TOUCH",
                    "dist_pips": dist_ema21}
        if low <= ema50 * 1.001 and close > ema50:
            return {"is_pullback": True, "level": ema50, "ema_type": "EMA50_TOUCH",
                    "dist_pips": dist_ema50}

    elif direction == "SELL":
        dist_ema21 = (ema21 - close) / pip_size
        dist_ema50 = (ema50 - close) / pip_size

        if abs(dist_ema21) <= tolerance:
            return {"is_pullback": True, "level": ema21, "ema_type": "EMA21",
                    "dist_pips": dist_ema21}
        if abs(dist_ema50) <= tolerance * 1.5:
            return {"is_pullback": True, "level": ema50, "ema_type": "EMA50",
                    "dist_pips": dist_ema50}
        if high >= ema21 * 0.999 and close < ema21:
            return {"is_pullback": True, "level": ema21, "ema_type": "EMA21_TOUCH",
                    "dist_pips": dist_ema21}
        if high >= ema50 * 0.999 and close < ema50:
            return {"is_pullback": True, "level": ema50, "ema_type": "EMA50_TOUCH",
                    "dist_pips": dist_ema50}

    return {"is_pullback": False, "level": 0, "ema_type": "", "dist_pips": 999}


def _check_h4_trend(df_h4: pd.DataFrame, direction: str) -> dict:
    if df_h4 is None or len(df_h4) < 20:
        return {"aligned": False, "strength": 0}

    last = df_h4.iloc[-1]
    ema9 = float(last.get('ema_9', 0))
    ema21 = float(last.get('ema_21', 0))
    ema50 = float(last.get('ema_50', 0))
    st_dir = int(last.get('supertrend_dir', 0))
    close = float(last['close'])

    score = 0
    if direction == "BUY":
        if ema9 > ema21 > ema50 and close > ema9:
            score += 30
        elif ema9 > ema21 and close > ema21:
            score += 20
        elif close > ema21:
            score += 10
        if st_dir == 1:
            score += 15
    elif direction == "SELL":
        if ema9 < ema21 < ema50 and close < ema9:
            score += 30
        elif ema9 < ema21 and close < ema21:
            score += 20
        elif close < ema21:
            score += 10
        if st_dir == -1:
            score += 15

    return {"aligned": score >= 20, "strength": score}


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
    Trend Continuation Strategy:
    Enters on pullbacks within a strong H4 trend.

    v2.0: Pullback to EMA is mandatory, OF must confirm, entry at EMA level.
    """
    if df_m15 is None or df_h1 is None or df_h4 is None:
        return None
    if len(df_m15) < 30 or len(df_h1) < 20 or len(df_h4) < 20:
        return None
    if df_m5 is None or len(df_m5) < 20:
        return None
    if market_report is None:
        return None

    close_price = float(df_m15.iloc[-1]['close'])
    pip_size = _get_pip_size(symbol, close_price)
    atr_pips = float(df_m15.iloc[-1].get('atr', 0)) / pip_size

    if atr_pips < 2.0:
        return None

    # ── v2.1: HARD gate — only trade in trending markets ──────
    # TREND_CONTINUATION was generating 73 trades at +0.055R avg.
    # Root cause: firing in ranging/choppy markets where trends don't
    # continue. This gate restricts entries to confirmed trends only.
    market_state = (master_report or {}).get('market_state', 'BALANCED')
    allowed_states = ('TRENDING_STRONG', 'TRENDING_EXTENDED', 'BREAKOUT_ACCEPTED')
    if market_state not in allowed_states:
        return None

    # ── v2.1: HARD gate — reject choppy markets ──────────────
    # Previously choppy was just a -15 penalty (could still fire at score 87+).
    # Now it's a hard block — no trend continuation in choppy conditions.
    momentum_data = (master_report or {}).get('momentum', {})
    if momentum_data.get('is_choppy', False):
        return None

    score = 0
    confluence = []

    # ── Step 1: Determine H4 trend direction ────────────────
    h4_bull = _check_h4_trend(df_h4, "BUY")
    h4_bear = _check_h4_trend(df_h4, "SELL")

    if h4_bull['strength'] > h4_bear['strength'] and h4_bull['aligned']:
        direction = "BUY"
        h4_trend_score = h4_bull['strength']
        score += h4_bull['strength']
        confluence.append(f"H4_BULL_TREND({h4_bull['strength']}pts)")
    elif h4_bear['strength'] > h4_bull['strength'] and h4_bear['aligned']:
        direction = "SELL"
        h4_trend_score = h4_bear['strength']
        score += h4_bear['strength']
        confluence.append(f"H4_BEAR_TREND({h4_bear['strength']}pts)")
    else:
        return None  # No clear H4 trend

    # ── Step 2: H1 structure confirmation ───────────────────
    h1 = df_h1.iloc[-1]
    h1_ema_bull = float(h1.get('ema_9', 0)) > float(h1.get('ema_21', 0))
    h1_ema_bear = float(h1.get('ema_9', 0)) < float(h1.get('ema_21', 0))
    h1_st = int(h1.get('supertrend_dir', 0))

    if direction == "BUY" and h1_ema_bull:
        h1_ema_aligned = True
        score += 15; confluence.append("H1_STRUCTURE_BULL")
    elif direction == "SELL" and h1_ema_bear:
        h1_ema_aligned = True
        score += 15; confluence.append("H1_STRUCTURE_BEAR")
    else:
        h1_ema_aligned = False
        return None

    if (direction == "BUY" and h1_st == 1) or (direction == "SELL" and h1_st == -1):
        score += 5; confluence.append("H1_SUPERTREND_ALIGNED")

    # ── Step 3: MANDATORY pullback to key EMA ───────────────
    pullback = _is_pullback_to_ema(df_m15, direction, symbol)
    if not pullback['is_pullback']:
        return None  # v2.0: Pullback is MANDATORY — no soft fallback

    score += 20; confluence.append(f"M15_PULLBACK_{pullback['ema_type']}")

    # ── Step 4: MANDATORY order flow confirmation ─────────────
    of_imb = market_report.get('order_flow_imbalance', {})
    imb = of_imb.get('imbalance', 0)
    of_strength = of_imb.get('strength', 'NONE')

    # Delta from market report
    rolling_delta = market_report.get('rolling_delta', {})
    delta_bias = rolling_delta.get('bias', 'NEUTRAL')

    # FIXED: OF or delta must confirm direction
    of_confirms = False
    if direction == "BUY":
        if (imb > 0.15 and of_strength in ('STRONG', 'MODERATE', 'EXTREME')):
            of_confirms = True
            score += 12; confluence.append(f"OF_BULL_{imb:+.2f}")
        elif delta_bias == "BULLISH":
            of_confirms = True
            score += 10; confluence.append("DELTA_BULL_MANDATORY")
    elif direction == "SELL":
        if (imb < -0.15 and of_strength in ('STRONG', 'MODERATE', 'EXTREME')):
            of_confirms = True
            score += 12; confluence.append(f"OF_BEAR_{imb:+.2f}")
        elif delta_bias == "BEARISH":
            of_confirms = True
            score += 10; confluence.append("DELTA_BEAR_MANDATORY")

    if not of_confirms:
        return None  # No flow confirmation — skip

    # ── Step 5: M5 rejection candle (bonus) ────────────────
    rejection = _detect_rejection_candle(df_m5, direction)
    if (direction == "BUY" and rejection == "REJECTION_BULL") or \
       (direction == "SELL" and rejection == "REJECTION_BEAR"):
        score += 15; confluence.append("M5_REJECTION_CANDLE")
    else:
        if df_m5 is not None and len(df_m5) >= 3:
            m5_last = df_m5.iloc[-1]
            m5_body = m5_last['close'] - m5_last['open']
            if (direction == "BUY" and m5_body > 0) or \
               (direction == "SELL" and m5_body < 0):
                score += 5; confluence.append("M5_DIRECTIONAL")

    # ── Step 6: Momentum velocity (bonus) ──────────────────
    velocity = 0
    vel_dir = 'FLAT'
    is_scalpable = False
    if master_report:
        momentum = master_report.get('momentum', {})
        velocity = momentum.get('velocity_pips_min', 0)
        vel_dir = momentum.get('velocity_direction', 'FLAT')
        is_scalpable = momentum.get('is_scalpable', False)

        if is_scalpable:
            if (direction == "BUY" and vel_dir == "UP") or \
               (direction == "SELL" and vel_dir == "DOWN"):
                score += 10; confluence.append(f"VELOCITY_{velocity}p/m")
            else:
                score += 3; confluence.append("VELOCITY_EXISTS")

    # ── Step 7: Volume surge bonus ────────────────────────
    if market_report:
        surge = market_report.get('volume_surge', {})
        if surge.get('surge_detected', False):
            score += 5; confluence.append("VOLUME_SURGE")

    # ── Step 8: SMC/HTF confirmation (bonus) ────────────────
    if smc_report:
        htf = smc_report.get('htf_alignment', {})
        if htf.get('approved', False):
            h4_bias = htf.get('h4_bias', '')
            if (direction == "BUY" and h4_bias == "BULLISH") or \
               (direction == "SELL" and h4_bias == "BEARISH"):
                score += 8; confluence.append("HTF_ALIGNED")

    # ── Choppy market check (hard gate above already blocks) ─
    # No penalty needed — blocked at the top of evaluate()
    # confluence stays cleaner without CHOPPY_PENALTY noise

    # ── Fibonacci confluence bonus ──────────────────────────
    try:
        from backtest.fib_builder import build_fib_report, check_fib_confluence
        fib_report = build_fib_report(df_h1=df_h1, df_h4=df_h4, current_price=close_price)
        fib_check = check_fib_confluence(close_price, direction, fib_report, pip_size)
        if fib_check['fib_bonus'] > 0:
            score += fib_check['fib_bonus']
            confluence.extend(fib_check['confluence'])
    except Exception:
        pass

    if len(confluence) < 6:
        return None

    # ── Score threshold ─────────────────────────────────────
    if score < MIN_SCORE:
        return None

    # ── Entry at pullback EMA level ────────────────────────
    entry = pullback['level']  # Enter at the EMA where pullback happened
    # Sanity: if EMA level is too far from price, use current price
    entry_dist = abs(close_price - entry) / pip_size
    if entry_dist > 5.0:
        entry = close_price  # Fallback to current price if too far

    # ── Calculate SL/TP ─────────────────────────────────────
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
        "_trend_cont_features": {
            'h4_trend_score': h4_trend_score,
            'pullback_ema_type': pullback['ema_type'],
            'pullback_dist_pips': pullback['dist_pips'],
            'h1_ema_aligned': 1 if h1_ema_aligned else 0,
            'h1_supertrend_dir': h1_st,
            'of_imbalance': imb,
            'of_strength': of_strength,
            'delta_confirms': 1 if of_confirms else 0,
            'rejection_type': rejection,
            'velocity_pips': velocity,
            'velocity_dir': vel_dir,
            'is_scalpable': 1 if is_scalpable else 0,
            'market_state': market_state,
            'is_choppy': 1 if momentum_data.get('is_choppy', False) else 0,
            'atr_pips': atr_pips,
        },
    }
