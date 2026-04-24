# =============================================================
# ai_engine/xgboost_classifier.py  v3.1
# PURPOSE: Predict win/loss probability for any trade signal.
# Uses 60 features extracted from market conditions at signal time.
#
# v3.1 CHANGES (from v3.0 42 features):
#   - EXPANDED: 42 features -> 60 features
#   - Added 11 raw DB columns not previously used:
#       symbol, strategy, strategy_group, rd_bias, of_strength,
#       vol_surge_ratio, momentum_direction, smc_bias,
#       structure_trend, slippage_pips, bias_confidence
#   - Added 7 engineered features:
#       hour_of_day, day_of_week, atr_normalized, vwap_atr_ratio,
#       sl_atr_ratio, score_per_confluence, sum_ss_scores
#   - All 60 features available in backtest_trades DB table
#   - NULL-safe: existing DB rows without new columns default to 0
#
# v3.0 CHANGES:
#   - EXPANDED: 21 features -> 42 features
#   - Added 10 strategy scores (ss_smc_ob, ss_trend_cont, etc.)
#   - Added market_state, conviction, agreement_groups
#   - Added of_imbalance, vol_surge, momentum_velocity, is_choppy, atr
#   - Added rr_ratio (tp/sl — key for expected value)
#   - Added confluence_count, market_score, spread_pips
#   - Added combined_bias encoding
# =============================================================

import numpy as np
import os
from datetime import datetime
from core.logger import get_logger

log = get_logger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__),
                          'models', 'xgb_model.pkl')
MIN_TRADES_TO_TRAIN = 50   # Minimum trades before training


# ── Feature names (must match extract_features + train_from_backtest exactly) ──
FEATURE_NAMES = [
    # ── Signal quality (6) ──
    'score',                  # composite signal score
    'sl_pips',                # stop loss distance
    'tp_pips',                # take profit distance
    'rr_ratio',               # tp_pips / sl_pips (risk-reward ratio)
    'direction',              # BUY=1.0, SELL=-1.0
    'confluence_count',       # number of confluence factors

    # ── Session (1) ──
    'session',                # ordinal encoding (0-4)

    # ── Order flow (5) ──
    'delta',                  # tick delta
    'rolling_delta',          # smoothed delta
    'delta_bias',             # -1.0, 0.0, 1.0
    'rd_bias',                # rolling delta bias (-1, 0, 1)
    'of_imbalance',           # order flow imbalance value

    # ── Volume & momentum (5) ──
    'vol_surge_detected',     # 0.0 or 1.0
    'vol_surge_ratio',        # volume surge ratio value
    'momentum_velocity',      # pips per minute
    'momentum_direction',     # FLAT=0, BULLISH=1, BEARISH=-1
    'is_choppy',              # 0.0 or 1.0

    # ── ATR & timing (3) ──
    'atr',                    # ATR value at entry
    'hour_of_day',            # 0-23 (hour of entry)
    'day_of_week',            # 0=Mon .. 6=Sun

    # ── VWAP & Volume Profile (4) ──
    'pip_from_vwap',          # pips from VWAP
    'vwap_position',          # ABOVE=1.0, BELOW=-1.0
    'pip_to_poc',             # pips to POC
    'va_width_pips',          # value area width

    # ── SMC structure (6) ──
    'pd_zone',                # -2 to +2 (EXTREME_PREMIUM->EXTREME_DISCOUNT)
    'pips_to_eq',             # pips to equilibrium
    'htf_approved',           # 0.0 or 1.0
    'htf_score',              # HTF alignment score
    'price_position',         # ABOVE_VAH=1, BELOW_VAL=-1, INSIDE=0
    'smc_bias',               # BULLISH=1, NEUTRAL=0, BEARISH=-1

    # ── Market structure (1) ──
    'structure_trend',        # RANGING=0, BULLISH=1, BEARISH=-1

    # ── Market scores (3) ──
    'final_score',            # master final score
    'smc_score',              # SMC composite score
    'market_score',           # market composite score

    # ── Market state (4, encoded) ──
    'market_state',           # ordinal encoding
    'combined_bias',          # -1.0 (BEARISH), 0.0 (NEUTRAL), 1.0 (BULLISH)
    'conviction',             # 0=LOW, 1=MEDIUM, 2=HIGH
    'bias_confidence',        # 0=LOW, 1=MODERATE, 2=HIGH

    # ── Trade context (4) ──
    'agreement_groups',       # number of agreeing strategy groups
    'spread_pips',            # spread at entry
    'slippage_pips',          # slippage at entry (0 if unknown)
    'of_strength',            # order flow strength (NONE=0..STRONG=3)

    # ── Identity (3) ──
    'symbol',                 # ordinal encoding of pair (0-16)
    'strategy',               # ordinal encoding of strategy (0-9)
    'strategy_group',         # ordinal encoding of group (0-5)

    # ── Engineered ratios (4) ──
    'atr_normalized',         # atr / va_width_pips (normalized ATR)
    'vwap_atr_ratio',         # pip_from_vwap / atr (VWAP dist in ATR units)
    'sl_atr_ratio',           # sl_pips / atr (SL in ATR units)
    'score_per_confluence',   # score / max(confluence_count, 1)

    # ── Strategy scores (10) — individual strategy quality at signal time ──
    'ss_smc_ob',              # SMC_OB_REVERSAL
    'ss_liquidity_sweep',     # LIQUIDITY_SWEEP_ENTRY
    'ss_vwap_reversion',      # VWAP_MEAN_REVERSION
    'ss_delta_divergence',    # DELTA_DIVERGENCE
    'ss_trend_continuation',  # TREND_CONTINUATION
    'ss_fvg_reversion',       # FVG_REVERSION
    'ss_ema_cross',           # EMA_CROSS_MOMENTUM
    'ss_rsi_divergence',      # RSI_DIVERGENCE_SMC
    'ss_breakout_momentum',   # BREAKOUT_MOMENTUM
    'ss_structure_align',     # STRUCTURE_ALIGNMENT

    # ── Aggregated (1) ──
    'sum_ss_scores',          # sum of all 10 strategy scores
]

N_FEATURES = len(FEATURE_NAMES)  # 60


# ── Encoding maps (shared between training and prediction) ──
SESSION_MAP = {
    'NY_LONDON_OVERLAP': 4,  # Distribution — highest liquidity
    'LONDON_SESSION':    3,  # Expansion — strong moves
    'LONDON_OPEN':       3,  # Manipulation — high opportunity
    'NY_AFTERNOON':      2,  # Late distribution
    'TOKYO':             1,  # Accumulation
    'SYDNEY':            0,  # Price discovery
}

PD_ZONE_MAP = {
    'EXTREME_PREMIUM': -2, 'PREMIUM': -1,
    'NEUTRAL': 0, 'DISCOUNT': 1, 'EXTREME_DISCOUNT': 2,
}

MARKET_STATE_MAP = {
    'BREAKOUT_REJECTED':  0,
    'BALANCED':           1,
    'REVERSAL_RISK':      2,
    'BREAKOUT_ACCEPTED':  3,
    'TRENDING_STRONG':    4,
    'TRENDING_EXTENDED':  5,
}

CONVICTION_MAP = {
    'LOW':    0.0,
    'MEDIUM': 1.0,
    'HIGH':   2.0,
}

BIAS_CONFIDENCE_MAP = {
    'LOW':      0.0,
    'MODERATE': 1.0,
    'HIGH':     2.0,
}

OF_STRENGTH_MAP = {
    'NONE':     0.0,
    'WEAK':     1.0,
    'MODERATE': 2.0,
    'STRONG':   3.0,
}

MOMENTUM_DIR_MAP = {
    'FLAT':    0.0,
    'BULLISH': 1.0,
    'BEARISH': -1.0,
}

STRUCTURE_TREND_MAP = {
    'RANGING':  0.0,
    'BULLISH':  1.0,
    'BEARISH': -1.0,
}

# 17 pairs in the watchlist
SYMBOL_MAP = {
    'EURUSD': 0,  'GBPUSD': 1,  'USDJPY': 2,  'AUDUSD': 3,
    'USDCAD': 4,  'NZDUSD': 5,  'EURJPY': 6,  'GBPJPY': 7,
    'AUDJPY': 8,  'EURGBP': 9,  'EURAUD': 10, 'GBPAUD': 11,
    'EURCAD': 12, 'GBPCAD': 13, 'AUDCAD': 14, 'NZDJPY': 15,
    'XAUUSD': 16,
}

# 10 active strategies
STRATEGY_MAP = {
    'SMC_OB_REVERSAL':       0,
    'LIQUIDITY_SWEEP_ENTRY': 1,
    'VWAP_MEAN_REVERSION':   2,
    'DELTA_DIVERGENCE':      3,
    'TREND_CONTINUATION':    4,
    'FVG_REVERSION':         5,
    'EMA_CROSS_MOMENTUM':    6,
    'RSI_DIVERGENCE_SMC':    7,
    'BREAKOUT_MOMENTUM':     8,
    'STRUCTURE_ALIGNMENT':   9,
}

# 5 strategy groups + OTHER
STRATEGY_GROUP_MAP = {
    'SMC_STRUCTURE':    0,
    'TREND_FOLLOWING':  1,
    'MEAN_REVERSION':   2,
    'ORDER_FLOW':       3,
    'OSCILLATOR':       4,
    'OTHER':            5,
}


def _encode_bias(raw: str) -> float:
    """Encode BULLISH/BEARISH/NEUTRAL string to float."""
    upper = str(raw).upper()
    if 'BULL' in upper:
        return 1.0
    elif 'BEAR' in upper:
        return -1.0
    return 0.0


def extract_features(signal: dict,
                     market_report: dict,
                     smc_report: dict,
                     strategy_scores: dict = None) -> np.ndarray | None:
    """
    Extract 60 numerical features from trade context.
    These are the inputs XGBoost learns from.
    Output shape: (1, 60)

    Args:
        signal:          trade signal dict (score, direction, session, sl_pips, tp_pips, etc.)
        market_report:   market analysis dict (delta, vwap, profile, final_score, etc.)
        smc_report:      SMC analysis dict (premium_discount, htf_alignment, structure)
        strategy_scores: dict of {strategy_name: score} for all 10 strategies (optional)
    """
    try:
        m = market_report or {}
        s = smc_report    or {}

        d   = m.get('delta', {})
        rd  = m.get('rolling_delta', {})
        of  = m.get('order_flow_imbalance', {})
        vs  = m.get('volume_surge', {})
        mom = m.get('momentum', {})
        vwap= m.get('vwap', {})
        prof= m.get('profile', {})
        pd_z= s.get('premium_discount', {})
        htf = s.get('htf_alignment', {})
        struct = s.get('structure', {})

        # ── Encode direction ──
        dir_enc = 1.0 if signal.get('direction') == 'BUY' else -1.0

        # ── Encode session ──
        sess_enc = float(SESSION_MAP.get(
            signal.get('session', 'UNKNOWN'), 1))

        # ── Encode premium/discount ──
        pd_enc = float(PD_ZONE_MAP.get(pd_z.get('zone', 'NEUTRAL'), 0))

        # ── Encode combined bias ──
        bias_enc = _encode_bias(m.get('combined_bias', 'NEUTRAL'))

        # ── Encode delta biases ──
        delta_bias_enc = _encode_bias(d.get('bias', 'NEUTRAL'))
        rd_bias_enc    = _encode_bias(rd.get('bias', 'NEUTRAL'))

        # ── Encode market state ──
        state_raw = str(m.get('market_state', 'BALANCED')).upper()
        state_enc = float(MARKET_STATE_MAP.get(state_raw, 1))

        # ── Encode conviction ──
        conv_raw = str(signal.get('conviction', 'MEDIUM')).upper()
        conv_enc = CONVICTION_MAP.get(conv_raw, 1.0)

        # ── Encode bias_confidence ──
        bc_raw = str(m.get('bias_confidence', 'MODERATE')).upper()
        bc_enc = BIAS_CONFIDENCE_MAP.get(bc_raw, 1.0)

        # ── Encode vwap position ──
        vwap_pos = 1.0 if 'ABOVE' in vwap.get('position', '') else -1.0

        # ── Encode price position ──
        pp = str(prof.get('price_position', 'INSIDE_VA')).upper()
        pp_enc = 1.0 if pp == 'ABOVE_VAH' else -1.0 if pp == 'BELOW_VAL' else 0.0

        # ── Encode of_strength ──
        of_str_raw = str(of.get('strength', 'NONE')).upper()
        of_str_enc = OF_STRENGTH_MAP.get(of_str_raw, 0.0)

        # ── Encode momentum_direction ──
        mom_dir_raw = str(mom.get('velocity_direction', 'FLAT')).upper()
        mom_dir_enc = MOMENTUM_DIR_MAP.get(mom_dir_raw, 0.0)

        # ── Encode smc_bias ──
        smc_bias_enc = _encode_bias(s.get('smc_bias', 'NEUTRAL'))

        # ── Encode structure_trend ──
        struct_raw = str(struct.get('trend', 'RANGING')).upper()
        struct_enc = STRUCTURE_TREND_MAP.get(struct_raw, 0.0)

        # ── SL/TP ──
        sl = float(signal.get('sl_pips', 10))
        tp = float(signal.get('tp_pips', 15) or signal.get('tp1_pips', 15))
        rr_ratio = tp / sl if sl > 0 else 1.5

        # ── ATR ──
        atr_val = float(m.get('atr', 0))

        # ── Temporal features ──
        now = datetime.utcnow()
        hour_of_day = float(signal.get('hour', now.hour))
        day_of_week = float(signal.get('day_of_week', now.weekday()))

        # ── Identity features ──
        symbol_enc = float(SYMBOL_MAP.get(
            signal.get('symbol', ''), 0))
        strategy_enc = float(STRATEGY_MAP.get(
            signal.get('strategy', ''), 0))
        group_raw = str(signal.get('group', 'OTHER')).upper()
        group_enc = float(STRATEGY_GROUP_MAP.get(group_raw, 5))

        # ── Strategy scores (default 0 if not provided) ──
        ss = strategy_scores or {}
        ss_defaults = {
            'SMC_OB_REVERSAL': 0, 'LIQUIDITY_SWEEP_ENTRY': 0,
            'VWAP_MEAN_REVERSION': 0, 'DELTA_DIVERGENCE': 0,
            'TREND_CONTINUATION': 0, 'FVG_REVERSION': 0,
            'EMA_CROSS_MOMENTUM': 0, 'RSI_DIVERGENCE_SMC': 0,
            'BREAKOUT_MOMENTUM': 0, 'STRUCTURE_ALIGNMENT': 0,
        }
        ss_smc_ob           = float(ss.get('SMC_OB_REVERSAL', 0) or ss_defaults['SMC_OB_REVERSAL'])
        ss_liquidity_sweep  = float(ss.get('LIQUIDITY_SWEEP_ENTRY', 0) or ss_defaults['LIQUIDITY_SWEEP_ENTRY'])
        ss_vwap_reversion   = float(ss.get('VWAP_MEAN_REVERSION', 0) or ss_defaults['VWAP_MEAN_REVERSION'])
        ss_delta_divergence = float(ss.get('DELTA_DIVERGENCE', 0) or ss_defaults['DELTA_DIVERGENCE'])
        ss_trend_cont       = float(ss.get('TREND_CONTINUATION', 0) or ss_defaults['TREND_CONTINUATION'])
        ss_fvg_rev          = float(ss.get('FVG_REVERSION', 0) or ss_defaults['FVG_REVERSION'])
        ss_ema_cross        = float(ss.get('EMA_CROSS_MOMENTUM', 0) or ss_defaults['EMA_CROSS_MOMENTUM'])
        ss_rsi_div          = float(ss.get('RSI_DIVERGENCE_SMC', 0) or ss_defaults['RSI_DIVERGENCE_SMC'])
        ss_breakout         = float(ss.get('BREAKOUT_MOMENTUM', 0) or ss_defaults['BREAKOUT_MOMENTUM'])
        ss_struct           = float(ss.get('STRUCTURE_ALIGNMENT', 0) or ss_defaults['STRUCTURE_ALIGNMENT'])
        sum_ss = (ss_smc_ob + ss_liquidity_sweep + ss_vwap_reversion +
                  ss_delta_divergence + ss_trend_cont + ss_fvg_rev +
                  ss_ema_cross + ss_rsi_div + ss_breakout + ss_struct)

        # ── Engineered ratios ──
        va_width = float(prof.get('va_width_pips', 50))
        atr_normalized = atr_val / max(va_width, 1.0)
        pip_from_vwap_val = float(vwap.get('pip_from_vwap', 0))
        vwap_atr_ratio = pip_from_vwap_val / max(atr_val, 0.001)
        sl_atr_ratio = sl / max(atr_val, 0.001)
        conf_count = float(len(signal.get('confluence', [])))
        score_val = float(signal.get('score', 0))
        score_per_conf = score_val / max(conf_count, 1.0)

        features = [
            # ── Signal quality (6) ──
            score_val,
            sl,
            tp,
            rr_ratio,
            dir_enc,
            conf_count,

            # ── Session (1) ──
            sess_enc,

            # ── Order flow (5) ──
            float(d.get('delta', 0)),
            float(rd.get('delta', 0)),
            delta_bias_enc,
            rd_bias_enc,
            float(of.get('imbalance', 0)),

            # ── Volume & momentum (5) ──
            1.0 if vs.get('surge_detected') else 0.0,
            float(vs.get('surge_ratio', 1.0)),
            float(mom.get('velocity_pips_min', 0)),
            mom_dir_enc,
            1.0 if mom.get('is_choppy') else 0.0,

            # ── ATR & timing (3) ──
            atr_val,
            hour_of_day,
            day_of_week,

            # ── VWAP & Volume Profile (4) ──
            pip_from_vwap_val,
            vwap_pos,
            float(prof.get('pip_to_poc', 50)),
            va_width,

            # ── SMC structure (6) ──
            pd_enc,
            float(pd_z.get('pips_to_eq', 0)),
            1.0 if htf.get('approved') else -1.0,
            float(htf.get('score', 50)),
            pp_enc,
            smc_bias_enc,

            # ── Market structure (1) ──
            struct_enc,

            # ── Market scores (3) ──
            float(m.get('final_score', 50)),
            float(m.get('smc_score', 50)),
            float(m.get('market_score', 50)),

            # ── Market state (4, encoded) ──
            state_enc,
            bias_enc,
            conv_enc,
            bc_enc,

            # ── Trade context (4) ──
            float(signal.get('agreement_groups', 1)),
            float(signal.get('spread_pips', 1.5)),
            float(signal.get('slippage_pips', 0)),
            of_str_enc,

            # ── Identity (3) ──
            symbol_enc,
            strategy_enc,
            group_enc,

            # ── Engineered ratios (4) ──
            atr_normalized,
            vwap_atr_ratio,
            sl_atr_ratio,
            score_per_conf,

            # ── Strategy scores (10) ──
            ss_smc_ob,
            ss_liquidity_sweep,
            ss_vwap_reversion,
            ss_delta_divergence,
            ss_trend_cont,
            ss_fvg_rev,
            ss_ema_cross,
            ss_rsi_div,
            ss_breakout,
            ss_struct,

            # ── Aggregated (1) ──
            sum_ss,
        ]

        if len(features) != N_FEATURES:
            log.error(f"[XGB] Feature count mismatch: {len(features)} vs {N_FEATURES}")
            return None

        return np.array(features, dtype=np.float32).reshape(1, -1)

    except Exception as ex:
        log.error(f"[XGB] Feature extraction failed: {ex}")
        return None


def predict_win_probability(features: np.ndarray) -> float:
    """
    Use trained XGBoost model to predict win probability.
    Returns float 0.0-1.0 (0=likely loss, 1=likely win).
    Returns 0.5 (neutral) if model not trained yet.
    """
    if not os.path.exists(MODEL_PATH):
        return 0.5
    try:
        import joblib
        model = joblib.load(MODEL_PATH)
        prob  = model.predict_proba(features)[0][1]
        return round(float(prob), 4)
    except Exception as e:
        log.error(f"[XGB] Prediction failed: {e}")
        return 0.5


def is_model_trained() -> bool:
    """Check if a trained XGBoost model exists on disk."""
    return os.path.exists(MODEL_PATH)


def get_model_info() -> dict:
    """Get information about the trained model (size, age, etc.)."""
    info = {
        'trained': False,
        'path': MODEL_PATH,
        'size_kb': 0,
        'n_features': N_FEATURES,
    }
    if os.path.exists(MODEL_PATH):
        info['trained'] = True
        info['size_kb'] = round(os.path.getsize(MODEL_PATH) / 1024, 1)
        import time
        info['age_hours'] = round(
            (time.time() - os.path.getmtime(MODEL_PATH)) / 3600, 1)
    return info


# ── BACKTEST TRAINING (60 features from backtest_trades) ─────

def train_from_backtest() -> dict:
    """
    Train XGBoost on backtest_trades table (rich 78-column data).
    Uses 60 features matching extract_features() exactly.

    Handles NULL strategy scores gracefully (existing DB rows
    before ss_* columns were added will default to 0).

    Returns dict with training results.
    """
    try:
        import xgboost as xgb
        import joblib
        from database.db_manager import get_connection

        conn   = get_connection()

        # Ensure backtest tables exist + auto-migrate new columns
        try:
            from backtest.db_store import _ensure_tables
            _ensure_tables(conn)
        except Exception:
            pass

        cursor = conn.cursor(dictionary=True)

        # Fetch all backtest trades with 60 features + label
        cursor.execute("""
            SELECT
                score, sl_pips, tp_pips, direction, session,
                confluence_count,
                delta, rolling_delta, delta_bias, rd_bias,
                of_imbalance,
                vol_surge_detected, vol_surge_ratio,
                momentum_velocity, momentum_direction, is_choppy,
                atr, entry_time,
                pip_from_vwap, pip_to_poc, va_width_pips,
                pd_zone, pips_to_eq, htf_approved, htf_score,
                price_position,
                smc_bias,
                structure_trend,
                final_score, smc_score, market_score,
                market_state, combined_bias, conviction,
                bias_confidence,
                agreement_groups, spread_pips, slippage_pips,
                of_strength,
                symbol, strategy, strategy_group,
                win
            FROM backtest_trades
            WHERE source = 'BACKTEST'
              AND outcome IS NOT NULL
              AND outcome != ''
        """)
        rows = cursor.fetchall()
        conn.close()

        if len(rows) < MIN_TRADES_TO_TRAIN:
            log.info(f"[XGB] Only {len(rows)} backtest trades — need "
                     f"{MIN_TRADES_TO_TRAIN} to train")
            return {
                'status': 'skipped',
                'reason': f'Only {len(rows)} trades (need {MIN_TRADES_TO_TRAIN})',
                'rows': len(rows),
            }

        # Build feature matrix — MUST match extract_features() exactly
        X = []
        y = []

        for row in rows:
            try:
                sl = float(row.get('sl_pips', 10) or 10)
                tp = float(row.get('tp_pips', 15) or 15)
                rr_ratio = tp / sl if sl > 0 else 1.5

                atr_val = float(row.get('atr', 0) or 0)
                va_width = float(row.get('va_width_pips', 50) or 50)
                pip_from_vwap_val = float(row.get('pip_from_vwap', 0) or 0)
                conf_count = float(row.get('confluence_count', 0) or 0)
                score_val = float(row.get('score', 0) or 0)

                # Temporal features from entry_time
                entry_time = row.get('entry_time')
                if entry_time:
                    if hasattr(entry_time, 'hour'):
                        hour_of_day = float(entry_time.hour)
                        day_of_week = float(entry_time.weekday())
                    else:
                        from datetime import datetime as dt
                        try:
                            et = dt.strptime(str(entry_time), '%Y-%m-%d %H:%M:%S')
                            hour_of_day = float(et.hour)
                            day_of_week = float(et.weekday())
                        except Exception:
                            hour_of_day = 12.0
                            day_of_week = 2.0
                else:
                    hour_of_day = 12.0
                    day_of_week = 2.0

                # Strategy scores (NULL defaults to 0)
                ss_smc_ob           = float(row.get('ss_smc_ob', 0) or 0)
                ss_liquidity_sweep  = float(row.get('ss_liquidity_sweep', 0) or 0)
                ss_vwap_reversion   = float(row.get('ss_vwap_reversion', 0) or 0)
                ss_delta_divergence = float(row.get('ss_delta_divergence', 0) or 0)
                ss_trend_cont       = float(row.get('ss_trend_continuation', 0) or 0)
                ss_fvg_rev          = float(row.get('ss_fvg_reversion', 0) or 0)
                ss_ema_cross        = float(row.get('ss_ema_cross', 0) or 0)
                ss_rsi_div          = float(row.get('ss_rsi_divergence', 0) or 0)
                ss_breakout         = float(row.get('ss_breakout_momentum', 0) or 0)
                ss_struct           = float(row.get('ss_structure_align', 0) or 0)
                sum_ss = (ss_smc_ob + ss_liquidity_sweep + ss_vwap_reversion +
                          ss_delta_divergence + ss_trend_cont + ss_fvg_rev +
                          ss_ema_cross + ss_rsi_div + ss_breakout + ss_struct)

                features = [
                    # ── Signal quality (6) ──
                    score_val,
                    sl,
                    tp,
                    rr_ratio,
                    1.0 if str(row.get('direction', '')).strip() == 'BUY' else -1.0,
                    conf_count,

                    # ── Session (1) ──
                    float(SESSION_MAP.get(row.get('session', ''), 1)),

                    # ── Order flow (5) ──
                    float(row.get('delta', 0) or 0),
                    float(row.get('rolling_delta', 0) or 0),
                    _encode_bias(row.get('delta_bias', 'NEUTRAL')),
                    _encode_bias(row.get('rd_bias', 'NEUTRAL')),
                    float(row.get('of_imbalance', 0) or 0),

                    # ── Volume & momentum (5) ──
                    1.0 if row.get('vol_surge_detected') else 0.0,
                    float(row.get('vol_surge_ratio', 1.0) or 1.0),
                    float(row.get('momentum_velocity', 0) or 0),
                    MOMENTUM_DIR_MAP.get(
                        str(row.get('momentum_direction', 'FLAT')).strip().upper(), 0.0),
                    1.0 if row.get('is_choppy') else 0.0,

                    # ── ATR & timing (3) ──
                    atr_val,
                    hour_of_day,
                    day_of_week,

                    # ── VWAP & Volume Profile (4) ──
                    pip_from_vwap_val,
                    1.0 if 'ABOVE' in str(row.get('pip_from_vwap', '')) else -1.0,
                    float(row.get('pip_to_poc', 50) or 50),
                    va_width,

                    # ── SMC structure (6) ──
                    float(PD_ZONE_MAP.get(row.get('pd_zone', 'NEUTRAL'), 0)),
                    float(row.get('pips_to_eq', 0) or 0),
                    1.0 if row.get('htf_approved') else -1.0,
                    float(row.get('htf_score', 50) or 50),
                    1.0 if str(row.get('price_position', '')).strip() == 'ABOVE_VAH' else
                    -1.0 if 'BELOW' in str(row.get('price_position', '')) else 0.0,
                    _encode_bias(row.get('smc_bias', 'NEUTRAL')),

                    # ── Market structure (1) ──
                    STRUCTURE_TREND_MAP.get(
                        str(row.get('structure_trend', 'RANGING')).strip().upper(), 0.0),

                    # ── Market scores (3) ──
                    float(row.get('final_score', 50) or 50),
                    float(row.get('smc_score', 50) or 50),
                    float(row.get('market_score', 50) or 50),

                    # ── Market state (4, encoded) ──
                    float(MARKET_STATE_MAP.get(
                        str(row.get('market_state', 'BALANCED')).strip().upper(), 1)),
                    1.0 if 'BULL' in str(row.get('combined_bias', '')).upper() else
                    -1.0 if 'BEAR' in str(row.get('combined_bias', '')).upper() else 0.0,
                    float(CONVICTION_MAP.get(
                        str(row.get('conviction', 'MEDIUM')).strip().upper(), 1.0)),
                    float(BIAS_CONFIDENCE_MAP.get(
                        str(row.get('bias_confidence', 'MODERATE')).strip().upper(), 1.0)),

                    # ── Trade context (4) ──
                    float(row.get('agreement_groups', 1) or 1),
                    float(row.get('spread_pips', 1.5) or 1.5),
                    float(row.get('slippage_pips', 0) or 0),
                    float(OF_STRENGTH_MAP.get(
                        str(row.get('of_strength', 'NONE')).strip().upper(), 0.0)),

                    # ── Identity (3) ──
                    float(SYMBOL_MAP.get(
                        str(row.get('symbol', '')).strip().upper(), 0)),
                    float(STRATEGY_MAP.get(
                        str(row.get('strategy', '')).strip(), 0)),
                    float(STRATEGY_GROUP_MAP.get(
                        str(row.get('strategy_group', 'OTHER')).strip().upper(), 5)),

                    # ── Engineered ratios (4) ──
                    atr_val / max(va_width, 1.0),
                    pip_from_vwap_val / max(atr_val, 0.001),
                    sl / max(atr_val, 0.001),
                    score_val / max(conf_count, 1.0),

                    # ── Strategy scores (10) ──
                    ss_smc_ob,
                    ss_liquidity_sweep,
                    ss_vwap_reversion,
                    ss_delta_divergence,
                    ss_trend_cont,
                    ss_fvg_rev,
                    ss_ema_cross,
                    ss_rsi_div,
                    ss_breakout,
                    ss_struct,

                    # ── Aggregated (1) ──
                    sum_ss,
                ]

                if len(features) != N_FEATURES:
                    log.warning(f"[XGB] Row feature count {len(features)} != {N_FEATURES}, skipping")
                    continue

                label = 1 if row.get('win') else 0
                X.append(features)
                y.append(label)
            except Exception:
                continue

        if len(X) < MIN_TRADES_TO_TRAIN:
            return {
                'status': 'skipped',
                'reason': f'Only {len(X)} valid rows after encoding',
                'rows': len(X),
            }

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)

        # Class balance check
        win_count = int(y.sum())
        loss_count = len(y) - win_count
        log.info(f"[XGB] Training data: {len(y)} trades "
                 f"({win_count} wins / {loss_count} losses "
                 f"= {win_count/len(y)*100:.1f}% WR)")

        # Scale weight if imbalanced
        scale_pos = loss_count / win_count if win_count > 0 and win_count < loss_count else 1.0

        model = xgb.XGBClassifier(
            n_estimators=150, max_depth=5,
            learning_rate=0.1, use_label_encoder=False,
            eval_metric='logloss', random_state=42,
            scale_pos_weight=scale_pos,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
        )

        # Train with validation split for accuracy reporting
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)

        # Calculate accuracy
        train_acc = round(model.score(X_train, y_train) * 100, 1)
        val_acc = round(model.score(X_val, y_val) * 100, 1)

        # Feature importance
        importance = model.feature_importances_
        top_features = sorted(zip(FEATURE_NAMES, importance),
                              key=lambda x: x[1], reverse=True)[:10]

        # Retrain on full data for production
        model.fit(X, y, verbose=False)

        # Save model
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        log.info(f"[XGB] Model trained on {len(y)} backtest trades "
                 f"({N_FEATURES} features, train_acc={train_acc}%, "
                 f"val_acc={val_acc}%)")

        return {
            'status': 'trained',
            'source': 'backtest_trades',
            'total_trades': len(y),
            'wins': win_count,
            'losses': loss_count,
            'win_rate': round(win_count / len(y) * 100, 1),
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'n_features': N_FEATURES,
            'top_features': [(f, round(i, 4)) for f, i in top_features],
            'model_size_kb': round(os.path.getsize(MODEL_PATH) / 1024, 1),
        }

    except ImportError as e:
        log.error(f"[XGB] Missing dependency: {e}")
        return {'status': 'error', 'reason': str(e)}
    except Exception as e:
        log.error(f"[XGB] Backtest training failed: {e}")
        return {'status': 'error', 'reason': str(e)}


# ── LIVE TRAINING (7 features from live trades table) ───────

def train_from_live() -> dict:
    """
    Train XGBoost on live trades table (simpler features).
    Fallback when no backtest data exists.

    Returns dict with training results.
    """
    try:
        import xgboost as xgb
        import joblib
        from database.db_manager import get_connection

        conn   = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT ai_score, sl_pips, tp_pips, rsi_at_entry,
                   atr_at_entry, spread_at_entry, session,
                   market_regime, outcome
            FROM trades
            WHERE outcome IS NOT NULL
            AND outcome != ''
        """)
        rows = cursor.fetchall()
        conn.close()

        if len(rows) < MIN_TRADES_TO_TRAIN:
            log.info(f"[XGB] Only {len(rows)} live trades — need "
                     f"{MIN_TRADES_TO_TRAIN} to train")
            return {
                'status': 'skipped',
                'reason': f'Only {len(rows)} live trades (need {MIN_TRADES_TO_TRAIN})',
                'rows': len(rows),
            }

        # Build feature matrix from live DB data
        X = []
        y = []
        for row in rows:
            ai_score, sl, tp, rsi, atr, spread, session, regime, outcome = row
            sess_map = {'LONDON_SESSION': 3, 'NY_LONDON_OVERLAP': 4,
                        'NY_AFTERNOON': 2, 'TOKYO': 1, 'SYDNEY': 0}
            features = [
                float(ai_score or 50),
                float(sl or 10),
                float(tp or 15),
                float(rsi or 50),
                float(atr or 0.001),
                float(spread or 1),
                float(sess_map.get(session or '', 1)),
            ]
            label = 1 if outcome in ('WIN_TP1', 'WIN_TP2') else 0
            X.append(features)
            y.append(label)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)

        win_count = int(y.sum())

        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4,
            learning_rate=0.1, use_label_encoder=False,
            eval_metric='logloss', random_state=42)
        model.fit(X, y)

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        train_acc = round(model.score(X, y) * 100, 1)
        log.info(f"[XGB] Live model trained on {len(rows)} trades "
                 f"(acc={train_acc}%)")

        return {
            'status': 'trained',
            'source': 'live_trades',
            'total_trades': len(rows),
            'wins': win_count,
            'losses': len(rows) - win_count,
            'win_rate': round(win_count / len(rows) * 100, 1),
            'train_accuracy': train_acc,
        }

    except ImportError as e:
        return {'status': 'error', 'reason': str(e)}
    except Exception as e:
        log.error(f"[XGB] Live training failed: {e}")
        return {'status': 'error', 'reason': str(e)}


def train_model() -> dict:
    """
    Train XGBoost — tries backtest_trades first (60 features),
    falls back to live trades (simpler 7 features).

    Returns dict with training results.
    """
    log.info("[XGB] Starting model training...")

    # Try backtest data first (richer features)
    result = train_from_backtest()
    if result['status'] == 'trained':
        return result

    log.info(f"[XGB] Backtest training {result['status']}: "
             f"{result.get('reason', '')} — trying live trades...")

    # Fallback to live trades
    result = train_from_live()
    return result


def score_signal(signal: dict, market_report: dict,
                 smc_report: dict,
                 strategy_scores: dict = None) -> dict:
    """
    Main function — extract features and predict win probability.
    Returns dict with probability and recommendation.

    Args:
        signal:          trade signal dict
        market_report:   market analysis dict
        smc_report:      SMC analysis dict
        strategy_scores: optional dict of {strategy_name: score}
    """
    features = extract_features(signal, market_report,
                                 smc_report, strategy_scores)
    if features is None:
        return {'probability': 0.5, 'recommendation': 'NEUTRAL',
                'trained': False}

    prob = predict_win_probability(features)
    trained = os.path.exists(MODEL_PATH)

    if prob >= 0.70:
        rec = 'STRONG_TAKE'
    elif prob >= 0.60:
        rec = 'TAKE'
    elif prob >= 0.45:
        rec = 'CAUTION'
    else:
        rec = 'SKIP'

    return {
        'probability':    prob,
        'recommendation': rec,
        'trained':        trained,
    }
