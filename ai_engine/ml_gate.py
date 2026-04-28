# =============================================================
# ai_engine/ml_gate.py  v3.3 — Strategy-Informed ML Gate (Regression)
#
# PURPOSE: Replace ALL 7 hardcoded gates with a single ML model
# that learns which combinations of strategy scores + market features
# predict profitable trades — now using REGRESSION (predict R-multiple)
# instead of classification (predict win/loss).
#
# ARCHITECTURE:
#   Layer 1 — Feature Engineering (66 features):
#     - 8  market quality features (score, delta, OF, volume)
#     - 4  VWAP features
#     - 8  SMC features (BOS, OBs, FVGs, sweeps, P/D zone)
#     - 3  HTF alignment features
#     - 5  trade parameter features (SL, TP, R:R)
#     - 10 strategy scores (the KEY addition)
#     - 5  consensus features (groups, counts, direction split)
#     - 5  session/time features
#     - 6  volatility/state features
#     - 3  symbol type features
#     - 3  self-improvement features (recent win rates)
#     - 3  Fibonacci confluence features
#     - 1  price context (spread)
#     - 3  self-calibration features (v3.3: source, predicted_r, error)
#     = 66 total features
#
#   Layer 2 — ML Model (XGBoost Regressor):
#     Input:  66-feature vector
#     Output: Predicted R-multiple (continuous float) per trade
#     Trained on: backtest_trades + shadow_trades with profit_r as target
#     v3.1: Switched from XGBClassifier (binary) to XGBRegressor
#     v3.3: SHADOW trades now included (full TradeTracker simulation)
#
#   Layer 3 — Execution Filter (R-based thresholds):
#     >= R:0.5 → TAKE trade (good edge expected)
#     0.0-0.5   → CAUTION (marginal edge → SHADOW simulate)
#     < 0.0     → SKIP (negative or zero expected R)
#
#   Layer 4 — Self-Improvement Loop:
#     Every 50 closed trades → retrain on ALL historical data
#     Log MAE, R², correlation, top features, quintile calibration
#
# WHY REGRESSION over classification:
#   Classification treats ALL wins equally (R:0.05 = R:5.0 → both = 1).
#   Regression learns to distinguish high-quality trades (R:2.0+) from
#   marginal scalps (R:0.05). This enables:
#     1. Better trade selection (only take trades with high expected R)
#     2. Kelly-style position sizing by confidence level
#     3. True edge measurement (expected R per signal)
#
# BACKWARD COMPATIBLE:
#   - Works with existing backtest_trades DB table (no schema change)
#   - Falls back to rule-based gates if no model exists
#   - Seeds from backtest DB on first run
# =============================================================

import os
import json
import time
import numpy as np
from datetime import datetime, timezone
from core.logger import get_logger

log = get_logger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__),
                          'models', 'ml_gate_v3r.pkl')
META_PATH = os.path.join(os.path.dirname(__file__),
                         'models', 'ml_gate_v3r_meta.json')

MIN_TRADES_TO_TRAIN = 80   # Minimum trades before training
RETRAIN_EVERY_N_TRADES = 50  # Auto-retrain frequency

# ── Regression-specific constants ──
R_CLIP_MIN = -2.0    # Clip R-multiple outliers below this
R_CLIP_MAX = 5.0     # Clip R-multiple outliers above this
TAKE_THRESHOLD = 0.3  # Minimum predicted R to TAKE a trade (lowered from 0.5)
CAUTION_THRESHOLD = 0.0  # Below this → SKIP


# ════════════════════════════════════════════════════════════════
# FEATURE NAMES (63 total) — must match extract_features exactly
# ════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    # ── Group 1: Market quality (7) ─────────────────────────
    'fq_final_score',        # Master combined score (0-100)
    'fq_market_score',       # Market report score (0-100)
    'fq_smc_score',          # SMC report score (0-100)
    'fq_combined_bias',      # Encoded: BULLISH=1, BEARISH=-1, NEUTRAL=0, CONFLICTED=-2
    'fq_bias_confidence',    # Encoded: HIGH=2, MODERATE=1, LOW=0
    'fq_htf_approved',       # 1 or -1
    'fq_htf_score',          # HTF alignment score (0-100)

    # ── Group 2: Order flow (6) ────────────────────────────
    'of_delta',              # Cumulative delta value
    'of_rolling_delta',      # Rolling delta value
    'of_delta_bias',         # Encoded: BULLISH=1, BEARISH=-1, NEUTRAL=0
    'of_rd_bias',            # Rolling delta bias encoded
    'of_imbalance',          # Order flow imbalance ratio
    'of_imb_strength',       # Encoded: EXTREME=3, STRONG=2, MODERATE=1, WEAK/NONE=0

    # ── Group 3: VWAP (4) ──────────────────────────────────
    'vw_pip_from_vwap',      # Distance to VWAP in pips
    'vw_position',           # Encoded: ABOVE=1, BELOW=-1, AT=0
    'vw_pip_to_poc',         # Distance to POC in pips
    'vw_price_position',     # Encoded: ABOVE_VAH=2, ABOVE_VA=1, INSIDE_VA=0, BELOW_VA=-1, BELOW_VAL=-2

    # ── Group 4: SMC structure (8) ─────────────────────────
    'smc_structure_trend',   # Encoded: BULLISH=2, RANGING=0, BEARISH=-2
    'smc_has_bos',           # 1 if BOS detected, 0 if not
    'smc_bos_direction',     # Encoded: BULL=1, BEAR=-1, NONE=0
    'smc_pd_zone',           # Encoded: EXTREME_PREMIUM=-2, PREMIUM=-1, NEUTRAL=0, DISCOUNT=1, EXTREME_DISCOUNT=2
    'smc_pips_to_eq',        # Pips to equilibrium
    'smc_smc_bias',          # Encoded: BULLISH=1, BEARISH=-1, NEUTRAL=0
    'smc_has_sweep',         # 1 if recent sweep detected
    'smc_sweep_aligned',     # 1 if sweep aligns with combined_bias

    # ── Group 5: Trade parameters (5) ──────────────────────
    'tp_score',              # Strategy score (0-100)
    'tp_sl_pips',            # Stop loss in pips
    'tp_tp_pips',            # Take profit in pips
    'tp_rr_ratio',           # TP/SL ratio
    'tp_direction',          # 1 for BUY, -1 for SELL

    # ── Group 6: Strategy scores (10) — THE KEY ADDITION ───
    'ss_smc_ob',             # SMC_OB_REVERSAL score (0 if no signal)
    'ss_liquidity_sweep',    # LIQUIDITY_SWEEP_ENTRY score
    'ss_vwap_reversion',     # VWAP_MEAN_REVERSION score
    'ss_delta_divergence',   # DELTA_DIVERGENCE score
    'ss_trend_continuation', # TREND_CONTINUATION score
    'ss_fvg_reversion',      # FVG_REVERSION score
    'ss_ema_cross',          # EMA_CROSS_MOMENTUM score
    'ss_rsi_divergence',     # RSI_DIVERGENCE_SMC score
    'ss_breakout_momentum',  # BREAKOUT_MOMENTUM score
    'ss_structure_align',    # STRUCTURE_ALIGNMENT score

    # ── Group 7: Consensus features (3) ────────────────────
    'cs_total_signals',      # How many strategies fired (0-10)
    'cs_groups_agreeing',    # How many DIFFERENT strategy groups agree (0-5)
    'cs_direction_clear',    # 1 if direction is clear, 0 if mixed signals

    # ── Group 8: Session/Time (5) ──────────────────────────
    'st_session',            # Encoded: NY_LONDON=4, LONDON=3, NY=2, TOKYO=1, SYDNEY=0
    'st_is_london_open',     # 1 or 0
    'st_is_overlap',         # 1 or 0
    'st_is_ny_afternoon',    # 1 or 0
    'st_vol_surge',          # 1 if volume surge active

    # ── Group 9: Volatility/State (5) ─────────────────────
    'vs_atr',                # ATR value (normalized later)
    'vs_market_state',       # Encoded: TRENDING_STRONG=3, BREAKOUT_ACCEPTED=2, TRENDING_EXTENDED=1,
                             #   BALANCED=0, REVERSAL_RISK=-1, BREAKOUT_REJECTED=-2
    'vs_surge_ratio',        # Volume surge ratio (1.0 if no surge)
    'vs_momentum_velocity',  # Price velocity in pips/min
    'vs_choppy',             # 1 if choppy market, 0 otherwise

    # ── Group 10: Symbol type (3) ──────────────────────────
    'sym_is_jpy',            # 1 if JPY pair
    'sym_is_commodity',      # 1 if XAU/XAG
    'sym_is_index',          # 1 if US30/US500/etc

    # ── Group 11: Self-improvement (3) ─────────────────────
    'si_recent_wr',          # Recent 50-trade win rate (0.0-1.0), 0.5 if N/A
    'si_recent_avg_r',       # Recent 50-trade avg R-multiple, 0.0 if N/A
    'si_strategy_wr',        # This strategy's historical WR, 0.5 if N/A

    # ── Group 12: Price context (1) ─────────────────────────
    'fx_spread_pips',        # Spread at entry

    # ── Group 13: Fibonacci confluence (3) ──────────────────
    'fib_confluence_score',  # Fibonacci confluence score (0-20)
    'fib_in_golden_zone',    # 1 if price is in 0.618-0.786 golden zone
    'fib_bias_aligned',      # 1 if Fib bias aligns with trade direction
]

# Should be 63 features
assert len(FEATURE_NAMES) == 63, f"Expected 63 features, got {len(FEATURE_NAMES)}"


# ════════════════════════════════════════════════════════════════
# ENCODING MAPS
# ════════════════════════════════════════════════════════════════

_SESSION_MAP = {
    'NY_LONDON_OVERLAP': 4,
    'LONDON_SESSION': 3,
    'LONDON_OPEN': 3,
    'NY_SESSION': 2,
    'NY_AFTERNOON': 2,
    'TOKYO': 1,
    'SYDNEY': 0,
    'UNKNOWN': 1,
}

_BIAS_MAP = {'BULLISH': 1.0, 'BEARISH': -1.0, 'NEUTRAL': 0.0, 'CONFLICTED': -2.0}
_CONFIDENCE_MAP = {'HIGH': 2.0, 'MODERATE': 1.0, 'LOW': 0.0}

_OF_STRENGTH_MAP = {
    'EXTREME': 3.0, 'STRONG': 2.0, 'MODERATE': 1.0, 'WEAK': 0.0, 'NONE': 0.0,
}

_PD_ZONE_MAP = {
    'EXTREME_PREMIUM': -2.0, 'PREMIUM': -1.0, 'NEUTRAL': 0.0,
    'DISCOUNT': 1.0, 'EXTREME_DISCOUNT': 2.0,
}

_PRICE_POS_MAP = {
    'ABOVE_VAH': 2.0, 'ABOVE_VA': 1.0, 'INSIDE_VA': 0.0,
    'BELOW_VA': -1.0, 'BELOW_VAL': -2.0,
}

_STATE_MAP = {
    'TRENDING_STRONG': 3.0, 'BREAKOUT_ACCEPTED': 2.0, 'TRENDING_EXTENDED': 1.0,
    'BALANCED': 0.0, 'REVERSAL_RISK': -1.0, 'BREAKOUT_REJECTED': -2.0,
    'RANGING': 0.0,
}

_TREND_MAP = {'BULLISH': 2.0, 'RANGING': 0.0, 'BEARISH': -2.0, 'NEUTRAL': 0.0}


def _safe_calc_prediction_error(actual_r: float, predicted_r) -> float:
    """Calculate prediction error (actual - predicted). Returns 0.0 if no prediction."""
    if predicted_r is None or predicted_r == '' or predicted_r == 0:
        return 0.0
    try:
        pred = float(predicted_r)
        return actual_r - pred
    except (TypeError, ValueError):
        return 0.0


# ════════════════════════════════════════════════════════════════
# LAYER 1: FEATURE EXTRACTION
# ════════════════════════════════════════════════════════════════

def extract_features(signal: dict,
                     master_report: dict,
                     market_report: dict,
                     smc_report: dict,
                     flow_data: dict,
                     all_strategy_scores: dict = None,
                     symbol: str = '',
                     spread_pips: float = 0.0,
                     self_improvement: dict = None) -> np.ndarray:
    """
    Extract 66 numerical features from the full market context.

    Args:
        signal:              The best strategy signal dict (with score, direction, SL/TP, etc.)
        master_report:       Combined master report
        market_report:       Market analysis report
        smc_report:          SMC structure report
        flow_data:           Order flow data (delta, imbalance, surge, momentum)
        all_strategy_scores: Dict of {strategy_name: score_or_None} for ALL 10 strategies
                             This is THE KEY — strategy scores become features, not decisions
        symbol:              Symbol name (for symbol type features)
        spread_pips:         Spread at entry
        self_improvement:    Dict with 'recent_wr', 'recent_avg_r', 'strategy_wr'

    Returns:
        np.ndarray of shape (1, 66) or None on error
    """
    try:
        mr = market_report or {}
        sr = smc_report or {}
        fd = flow_data or {}

        # ── Shorthand accessors ──
        delta_d = fd.get('delta', {})
        rd_d = fd.get('rolling_delta', {})
        of_d = fd.get('order_flow_imbalance', {})
        vs_d = fd.get('volume_surge', {})
        mom_d = fd.get('momentum', {})
        pd_d = sr.get('premium_discount', {})
        htf_d = sr.get('htf_alignment', {})
        struct_d = sr.get('structure', {})
        vwap_d = mr.get('vwap', {})
        prof_d = mr.get('profile', {})
        sweep_d = sr.get('last_sweep', {})

        # ── Strategy scores (default 0 = no signal) ──
        ss = all_strategy_scores or {}
        ss_smc_ob = float(ss.get('SMC_OB_REVERSAL', 0) or 0)
        ss_liq_sweep = float(ss.get('LIQUIDITY_SWEEP_ENTRY', 0) or 0)
        ss_vwap = float(ss.get('VWAP_MEAN_REVERSION', 0) or 0)
        ss_delta = float(ss.get('DELTA_DIVERGENCE', 0) or 0)
        ss_trend = float(ss.get('TREND_CONTINUATION', 0) or 0)
        ss_fvg = float(ss.get('FVG_REVERSION', 0) or 0)
        ss_ema = float(ss.get('EMA_CROSS_MOMENTUM', 0) or 0)
        ss_rsi = float(ss.get('RSI_DIVERGENCE_SMC', 0) or 0)
        ss_breakout = float(ss.get('BREAKOUT_MOMENTUM', 0) or 0)
        ss_struct = float(ss.get('STRUCTURE_ALIGNMENT', 0) or 0)

        # ── Fibonacci confluence ──
        fib_data = signal.get('fib_data', {})
        if not fib_data:
            try:
                direction = str(signal.get('direction', ''))
                entry_price = float(signal.get('entry_price', 0))
                if entry_price > 0:
                    from backtest.fib_builder import build_fib_report, check_fib_confluence
                    fib_report = build_fib_report(current_price=entry_price)
                    fib_check = check_fib_confluence(entry_price, direction, fib_report)
                    fib_data = fib_check
                    signal['fib_data'] = fib_data  # Cache for reuse
            except Exception:
                fib_data = {}

        # ── Consensus features ──
        non_zero = {k: v for k, v in ss.items() if v and v > 0}
        cs_total = len(non_zero)
        # Group agreement: count unique strategy groups that fired
        from strategies.strategy_engine import STRATEGY_GROUPS
        fired_groups = set()
        for sn in non_zero:
            for group_name, members in STRATEGY_GROUPS.items():
                if sn in members:
                    fired_groups.add(group_name)
        cs_groups = len(fired_groups)
        # Direction clarity: 1 if all non-zero scores agree on direction
        signal_dir = str(signal.get('direction', ''))
        cs_clear = 1.0  # default: assume clear (single best signal chosen)

        # ── Session features ──
        session = master_report.get('session', 'UNKNOWN')
        sess_enc = _SESSION_MAP.get(session, 1)
        is_london_open = 1.0 if session == 'LONDON_OPEN' else 0.0
        is_overlap = 1.0 if session == 'NY_LONDON_OVERLAP' else 0.0
        is_ny_afternoon = 1.0 if session == 'NY_AFTERNOON' else 0.0
        vol_surge = 1.0 if vs_d.get('surge_detected') else 0.0

        # ── Symbol type features ──
        sym_upper = symbol.upper()
        is_jpy = 1.0 if 'JPY' in sym_upper else 0.0
        is_commodity = 1.0 if ('XAU' in sym_upper or 'XAG' in sym_upper) else 0.0
        is_index = 1.0 if any(x in sym_upper for x in ['US30', 'US500', 'USTEC', 'JP225', 'DE30', 'UK100']) else 0.0

        # ── Self-improvement features ──
        si = self_improvement or {}
        recent_wr = float(si.get('recent_wr', 0.5))
        recent_avg_r = float(si.get('recent_avg_r', 0.0))
        strategy_wr = float(si.get('strategy_wr', 0.5))

        # ── Build the 63-feature vector ──
        features = [
            # Market quality (7)
            float(master_report.get('final_score', 0)),
            float(mr.get('trade_score', 0)),
            float(sr.get('smc_score', 0)),
            _BIAS_MAP.get(str(master_report.get('combined_bias', 'NEUTRAL')), 0.0),
            _CONFIDENCE_MAP.get(str(master_report.get('bias_confidence', 'MODERATE')), 1.0),
            1.0 if htf_d.get('approved') else -1.0,
            float(htf_d.get('score', 50)),

            # Order flow (6)
            float(delta_d.get('delta', 0)),
            float(rd_d.get('delta', 0)),
            _BIAS_MAP.get(str(delta_d.get('bias', 'NEUTRAL')), 0.0),
            _BIAS_MAP.get(str(rd_d.get('bias', 'NEUTRAL')), 0.0),
            float(of_d.get('imbalance', 0)),
            _OF_STRENGTH_MAP.get(str(of_d.get('strength', 'NONE')), 0.0),

            # VWAP (4)
            float(vwap_d.get('pip_from_vwap', 0)),
            1.0 if 'ABOVE' in str(vwap_d.get('position', '')) else
            -1.0 if 'BELOW' in str(vwap_d.get('position', '')) else 0.0,
            float(prof_d.get('pip_to_poc', 50)),
            _PRICE_POS_MAP.get(str(prof_d.get('price_position', 'INSIDE_VA')), 0.0),

            # SMC structure (8)
            _TREND_MAP.get(str(struct_d.get('trend', 'RANGING')), 0.0),
            1.0 if struct_d.get('bos') else 0.0,
            1.0 if struct_d.get('bos') and 'BULL' in str(struct_d.get('bos', {}).get('type', '')) else
            -1.0 if struct_d.get('bos') and 'BEAR' in str(struct_d.get('bos', {}).get('type', '')) else 0.0,
            _PD_ZONE_MAP.get(str(pd_d.get('zone', 'NEUTRAL')), 0.0),
            float(pd_d.get('pips_to_eq', 0)),
            _BIAS_MAP.get(str(sr.get('smc_bias', 'NEUTRAL')), 0.0),
            1.0 if sweep_d and sweep_d.get('swept_level') else 0.0,
            1.0 if master_report.get('sweep_aligned') else 0.0,

            # Trade parameters (5)
            float(signal.get('score', 0)),
            float(signal.get('sl_pips', 10)),
            float(signal.get('tp1_pips', 15) or signal.get('tp_pips', 15)),
            (float(signal.get('tp1_pips', 15) or signal.get('tp_pips', 15)) /
             max(float(signal.get('sl_pips', 10)), 0.1)),
            1.0 if str(signal.get('direction', '')) == 'BUY' else -1.0,

            # Strategy scores (10) — THE KEY ADDITION
            ss_smc_ob,
            ss_liq_sweep,
            ss_vwap,
            ss_delta,
            ss_trend,
            ss_fvg,
            ss_ema,
            ss_rsi,
            ss_breakout,
            ss_struct,

            # Consensus (3)
            cs_total,
            cs_groups,
            cs_clear,

            # Session (5)
            sess_enc,
            is_london_open,
            is_overlap,
            is_ny_afternoon,
            vol_surge,

            # Volatility/State (5)
            float(mr.get('atr', 0)),
            _STATE_MAP.get(str(master_report.get('market_state', 'BALANCED')), 0.0),
            float(vs_d.get('surge_ratio', 1.0)),
            float(mom_d.get('velocity_pips_min', 0)),
            1.0 if mom_d.get('is_choppy') else 0.0,

            # Symbol type (3)
            is_jpy,
            is_commodity,
            is_index,

            # Self-improvement (3)
            recent_wr,
            recent_avg_r,
            strategy_wr,

            # Price context (1)
            float(spread_pips),

            # Fibonacci confluence (3)
            float(fib_data.get('confluence_score', 0)),
            1.0 if fib_data.get('in_golden_zone', False) else 0.0,
            1.0 if fib_data.get('fib_bias_aligned', False) else 0.0,

            # Self-calibration (3) — v3.3: always 0.0 for live scoring
            # (model knows this is a live signal, not historical)
            0.0,  # source: N/A for live (always 0.0)
            0.0,  # model_predicted_r: N/A for live (always 0.0)
            0.0,  # prediction_error: N/A for live (always 0.0)
        ]

        return np.array(features, dtype=np.float32).reshape(1, -1)

    except Exception as ex:
        log.error(f"[ML_GATE] Feature extraction failed: {ex}")
        return None


def extract_features_from_db(row: dict, all_strategy_scores: dict = None) -> np.ndarray:
    """
    Extract 66 features from a backtest_trades DB row.
    Used for training from stored historical data.
    """
    try:
        ss = all_strategy_scores or {}

        features = [
            # Market quality (7)
            float(row.get('final_score', 0) or 0),
            float(row.get('market_score', 0) or 0),
            float(row.get('smc_score', 0) or 0),
            _BIAS_MAP.get(str(row.get('combined_bias', 'NEUTRAL')), 0.0),
            1.0,  # bias_confidence — not in DB, default MODERATE
            1.0 if row.get('htf_approved') else -1.0,
            float(row.get('htf_score', 50) or 50),

            # Order flow (6)
            float(row.get('delta', 0) or 0),
            float(row.get('rolling_delta', 0) or 0),
            float(row.get('delta_bias', 0) or 0),
            float(row.get('rd_bias', 0) or 0),
            float(row.get('of_imbalance', 0) or 0),
            _OF_STRENGTH_MAP.get(str(row.get('of_strength', 'NONE')), 0.0),

            # VWAP (4)
            float(row.get('pip_from_vwap', 0) or 0),
            1.0 if 'ABOVE' in str(row.get('price_position', '')) else
            -1.0 if 'BELOW' in str(row.get('price_position', '')) else 0.0,
            float(row.get('pip_to_poc', 50) or 50),
            _PRICE_POS_MAP.get(str(row.get('price_position', 'INSIDE_VA')), 0.0),

            # SMC structure (8)
            _TREND_MAP.get(str(row.get('structure_trend', 'RANGING')), 0.0),
            0.0,  # has_bos — not directly in DB
            0.0,  # bos_direction — not directly in DB
            _PD_ZONE_MAP.get(str(row.get('pd_zone', 'NEUTRAL')), 0.0),
            float(row.get('pips_to_eq', 0) or 0),
            _BIAS_MAP.get(str(row.get('smc_bias', 'NEUTRAL')), 0.0),
            0.0,  # has_sweep — not directly in DB
            0.0,  # sweep_aligned — not directly in DB

            # Trade parameters (5)
            float(row.get('score', 0) or 0),
            float(row.get('sl_pips', 10) or 10),
            float(row.get('tp_pips', 15) or 15),
            (float(row.get('tp_pips', 15) or 15) /
             max(float(row.get('sl_pips', 10) or 10), 0.1)),
            1.0 if str(row.get('direction', '')) == 'BUY' else -1.0,

            # Strategy scores (10) — directly from DB columns
            float(row.get('ss_smc_ob', 0) or 0),
            float(row.get('ss_liquidity_sweep', 0) or 0),
            float(row.get('ss_vwap_reversion', 0) or 0),
            float(row.get('ss_delta_divergence', 0) or 0),
            float(row.get('ss_trend_continuation', 0) or 0),
            float(row.get('ss_fvg_reversion', 0) or 0),
            float(row.get('ss_ema_cross', 0) or 0),
            float(row.get('ss_rsi_divergence', 0) or 0),
            float(row.get('ss_breakout_momentum', 0) or 0),
            float(row.get('ss_structure_align', 0) or 0),

            # Consensus (3) — not fully in DB, use defaults
            float(row.get('agreement_groups', 1) or 1),
            float(row.get('agreement_groups', 1) or 1),  # groups ≈ agreement_groups
            1.0,  # direction clear — default

            # Session (5)
            _SESSION_MAP.get(str(row.get('session', '')), 1),
            0.0, 0.0, 0.0,  # is_london_open, is_overlap, is_ny_afternoon — not in DB
            1.0 if row.get('vol_surge_detected') else 0.0,

            # Volatility/State (5)
            float(row.get('atr', 0) or 0),
            _STATE_MAP.get(str(row.get('market_state', 'BALANCED')), 0.0),
            float(row.get('vol_surge_ratio', 1.0) or 1.0),
            float(row.get('momentum_velocity', 0) or 0),
            1.0 if row.get('is_choppy') else 0.0,

            # Symbol type (3) — computed from symbol name
            1.0 if 'JPY' in str(row.get('symbol', '')).upper() else 0.0,
            1.0 if any(x in str(row.get('symbol', '')).upper()
                       for x in ['XAU', 'XAG']) else 0.0,
            1.0 if any(x in str(row.get('symbol', '')).upper()
                       for x in ['US30', 'US500', 'USTEC', 'JP225', 'DE30', 'UK100']) else 0.0,

            # Self-improvement (3) — defaults for DB training
            0.5,  # recent_wr
            0.0,  # recent_avg_r
            0.5,  # strategy_wr

            # Price context (1)
            float(row.get('spread_pips', 0) or 0),

            # Fibonacci confluence (3) — from DB columns
            float(row.get('fib_confluence_score', 0) or 0),
            1.0 if row.get('fib_in_golden_zone') else 0.0,
            1.0 if row.get('fib_bias_aligned') else 0.0,

            # Self-calibration (3) — v3.3: model learns from its own past decisions
            # source: 1.0 if SHADOW (model was unsure), 0.0 if BACKTEST (model was confident)
            1.0 if str(row.get('source', 'BACKTEST')) == 'SHADOW' else 0.0,
            # model_predicted_r: what the model predicted for this trade (NULL for older data)
            float(row.get('model_predicted_r', 0) or 0),
            # prediction_error: actual R minus predicted R (0.0 if no prediction)
            _safe_calc_prediction_error(
                float(row.get('profit_r', 0) or 0),
                row.get('model_predicted_r')
            ),
        ]

        return np.array(features, dtype=np.float32)

    except Exception as ex:
        log.error(f"[ML_GATE] DB feature extraction failed: {ex}")
        return None


# ════════════════════════════════════════════════════════════════
# LAYER 2: MODEL — PREDICT (Regression: returns R-multiple)
# ════════════════════════════════════════════════════════════════

def is_model_trained() -> bool:
    """Check if a trained ML gate model exists."""
    return os.path.exists(MODEL_PATH)


def get_model_info() -> dict:
    """Get model metadata (age, training stats, feature importance)."""
    info = {'trained': False, 'path': MODEL_PATH}
    if os.path.exists(MODEL_PATH):
        info['trained'] = True
        info['size_kb'] = round(os.path.getsize(MODEL_PATH) / 1024, 1)
        info['age_hours'] = round(
            (time.time() - os.path.getmtime(MODEL_PATH)) / 3600, 1)
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, 'r') as f:
                meta = json.load(f)
            info.update(meta)
        except Exception:
            pass
    return info


def predict(features: np.ndarray) -> float:
    """
    Predict expected R-multiple using trained regression model.
    Returns 0.0 (neutral) if no model exists.
    """
    if not os.path.exists(MODEL_PATH):
        return 0.0
    try:
        import joblib
        model = joblib.load(MODEL_PATH)
        predicted_r = float(model.predict(features)[0])
        # Clip to reasonable range
        predicted_r = max(R_CLIP_MIN, min(R_CLIP_MAX, predicted_r))
        return round(predicted_r, 4)
    except Exception as e:
        log.error(f"[ML_GATE] Prediction failed: {e}")
        return 0.0


def score_signal(signal: dict,
                 master_report: dict,
                 market_report: dict,
                 smc_report: dict,
                 flow_data: dict,
                 all_strategy_scores: dict = None,
                 symbol: str = '',
                 spread_pips: float = 0.0) -> dict:
    """
    Main scoring function — extract features + predict expected R-multiple.

    R-based thresholds:
      >= 0.5 → TAKE (good edge — expected return is at least 50% of risk)
      0.0-0.5 → CAUTION (marginal edge — might take with reduced size)
      < 0.0   → SKIP (negative or zero expected R)
    """
    features = extract_features(
        signal, master_report, market_report, smc_report,
        flow_data, all_strategy_scores, symbol, spread_pips)

    if features is None:
        return {
            'predicted_r': 0.0,
            'recommendation': 'NEUTRAL',
            'trained': False,
        }

    predicted_r = predict(features)
    trained = os.path.exists(MODEL_PATH)

    if predicted_r >= TAKE_THRESHOLD:
        rec = 'TAKE'
    elif predicted_r >= CAUTION_THRESHOLD:
        rec = 'CAUTION'
    else:
        rec = 'SKIP'

    return {
        'predicted_r': predicted_r,
        'recommendation': rec,
        'trained': trained,
        'features': features.tolist() if trained else None,
    }


# ════════════════════════════════════════════════════════════════
# LAYER 4: SELF-IMPROVEMENT — TRAIN (Regression Model)
# ════════════════════════════════════════════════════════════════

def _json_default(obj):
    """Fallback serializer for numpy types that json can't handle."""
    if hasattr(obj, 'item'):  # numpy scalars
        return obj.item()
    if hasattr(obj, '__float__'):
        return float(obj)
    return str(obj)


def train_model(source: str = 'auto') -> dict:
    """
    Train the ML gate REGRESSION model from database.

    Target: profit_r (R-multiple of each trade) — continuous float.
    This allows the model to distinguish between a barely-profitable
    scalp (R:+0.05) and a strong trend ride (R:+3.0), unlike binary
    classification which treats both identically as "win = 1".

    Args:
        source: 'backtest' | 'live' | 'auto'

    Returns dict with training results and regression metrics.
    """
    try:
        import xgboost as xgb
        import joblib
        from database.db_manager import get_connection

        conn = get_connection()

        # Ensure tables exist
        try:
            from backtest.db_store import _ensure_tables
            _ensure_tables(conn)
        except Exception:
            pass

        cursor = conn.cursor(dictionary=True)

        # ── Fetch training data ──
        # v3.3: Include SHADOW trades — they now use the full TradeTracker
        # so their outcomes are real. The 'source' feature tells the model
        # the category (EXECUTED vs SHADOW) so it can self-calibrate.
        if source in ('backtest', 'auto'):
            cursor.execute("""
                SELECT
                    symbol, direction, strategy, session, market_state,
                    score, sl_pips, tp_pips, confluence_count,
                    delta, rolling_delta, delta_bias, rd_bias,
                    of_imbalance, of_strength, vol_surge_detected, vol_surge_ratio,
                    momentum_velocity, is_choppy,
                    smc_bias, pd_zone, pips_to_eq, structure_trend,
                    atr, pip_from_vwap, pip_to_poc, va_width_pips, price_position,
                    final_score, market_score, smc_score, htf_approved, htf_score,
                    combined_bias, agreement_groups,
                    spread_pips, slippage_pips,
                    ss_smc_ob, ss_liquidity_sweep, ss_vwap_reversion,
                    ss_delta_divergence, ss_trend_continuation,
                    ss_fvg_reversion, ss_ema_cross, ss_rsi_divergence,
                    ss_breakout_momentum, ss_structure_align,
                    profit_pips, profit_r, win,
                    source, model_predicted_r
                FROM backtest_trades
                WHERE source IN ('BACKTEST', 'SHADOW')
                  AND outcome IS NOT NULL
                  AND outcome != ''
                  AND win IS NOT NULL
                  AND profit_r IS NOT NULL
                ORDER BY entry_time ASC
            """)
            rows = cursor.fetchall()

        if source == 'auto' and len(rows) < MIN_TRADES_TO_TRAIN:
            log.info(f"[ML_GATE] Only {len(rows)} backtest trades — trying live...")
            cursor.execute("""
                SELECT
                    symbol, direction, strategy, session, market_state,
                    ai_score as score, sl_pips, tp_pips,
                    rsi_at_entry, atr_at_entry, spread_at_entry,
                    market_regime, outcome
                FROM trades
                WHERE outcome IS NOT NULL AND outcome != ''
            """)
            rows = cursor.fetchall()

        conn.close()

        if len(rows) < MIN_TRADES_TO_TRAIN:
            return {
                'status': 'skipped',
                'reason': f'Only {len(rows)} trades (need {MIN_TRADES_TO_TRAIN})',
                'rows': len(rows),
            }

        # ── Build feature matrix + R-multiple target ──
        X = []
        y = []  # R-multiple (continuous target)
        backtest_count = 0
        shadow_count = 0

        for row in rows:
            try:
                features = extract_features_from_db(row)
                if features is None:
                    continue
                r_multiple = float(row.get('profit_r', 0) or 0)
                X.append(features)
                y.append(r_multiple)
                if str(row.get('source', 'BACKTEST')) == 'SHADOW':
                    shadow_count += 1
                else:
                    backtest_count += 1
            except Exception:
                continue

        if len(X) < MIN_TRADES_TO_TRAIN:
            return {
                'status': 'skipped',
                'reason': f'Only {len(X)} valid rows after encoding',
                'rows': len(X),
            }

        X = np.array(X, dtype=np.float32)
        y_raw = np.array(y, dtype=np.float32)

        # ── Clip outliers to prevent extreme R values from dominating loss ──
        y = np.clip(y_raw, R_CLIP_MIN, R_CLIP_MAX)

        # ── Target statistics ──
        win_count = int((y_raw > 0).sum())
        loss_count = int((y_raw <= 0).sum())
        wr = win_count / len(y) * 100
        mean_r = float(np.mean(y_raw))
        median_r = float(np.median(y_raw))
        std_r = float(np.std(y_raw))
        positive_mean = float(np.mean(y_raw[y_raw > 0])) if win_count > 0 else 0.0
        negative_mean = float(np.mean(y_raw[y_raw <= 0])) if loss_count > 0 else 0.0

        log.info(f"[ML_GATE] Regression training: {len(y)} trades "
                 f"({backtest_count} real + {shadow_count} shadow) "
                 f"({win_count}W / {loss_count}L = {wr:.1f}% WR, "
                 f"mean_R={mean_r:.3f}, median_R={median_r:.3f})")

        # ── Train/val split (no stratify — regression target is continuous) ──
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # ── XGBoost Regressor ──
        # Overfitting control: with 63 features and ~250-500 samples,
        # we use conservative depth + strong regularization.
        # As data grows past 1000 trades, these can be relaxed.
        n_trades = len(y)
        if n_trades < 500:
            # Small data — very conservative
            depth, eta, child_w = 3, 0.03, 10
            alpha, lam = 1.0, 5.0
            cols = 0.5
        elif n_trades < 1000:
            # Medium data — moderate
            depth, eta, child_w = 4, 0.04, 7
            alpha, lam = 0.5, 3.0
            cols = 0.6
        else:
            # Large data — standard
            depth, eta, child_w = 5, 0.05, 5
            alpha, lam = 0.1, 1.0
            cols = 0.7

        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=depth,
            learning_rate=eta,
            objective='reg:squarederror',
            random_state=42,
            min_child_weight=child_w,
            subsample=0.8,
            colsample_bytree=cols,
            reg_alpha=alpha,
            reg_lambda=lam,
            early_stopping_rounds=50,
        )

        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)

        # ── Regression metrics ──
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)

        train_mae = float(np.mean(np.abs(train_preds - y_train)))
        val_mae = float(np.mean(np.abs(val_preds - y_val)))
        val_rmse = float(np.sqrt(np.mean((val_preds - y_val) ** 2)))

        # R² score (how much variance the model explains)
        ss_res = np.sum((y_val - val_preds) ** 2)
        ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
        val_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Pearson correlation (directional agreement)
        if len(y_val) > 2 and np.std(y_val) > 0 and np.std(val_preds) > 0:
            val_corr = float(np.corrcoef(y_val, val_preds)[0, 1])
        else:
            val_corr = 0.0

        # ── Feature importance (top 15) ──
        importance = model.feature_importances_
        top_features = sorted(zip(FEATURE_NAMES, importance),
                              key=lambda x: x[1], reverse=True)[:15]

        # ── Quintile calibration ──
        calibration = _check_regression_calibration(val_preds, y_val)

        # ── Retrain on full data for production ──
        # Use same adaptive params as training model
        best_n = getattr(model, 'best_iteration', 500)
        model_final = xgb.XGBRegressor(
            n_estimators=best_n,
            max_depth=depth,
            learning_rate=eta,
            objective='reg:squarederror',
            random_state=42,
            min_child_weight=child_w,
            subsample=0.8,
            colsample_bytree=cols,
            reg_alpha=alpha,
            reg_lambda=lam,
        )
        model_final.fit(X, y, verbose=False)

        # ── Save model + metadata ──
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model_final, MODEL_PATH)

        meta = {
            'status': 'trained',
            'version': '3.1-regression',
            'model_type': 'XGBRegressor',
            'target': 'profit_r (R-multiple, continuous)',
            'n_features': len(FEATURE_NAMES),
            'total_trades': int(len(y)),
            'wins': int(win_count),
            'losses': int(loss_count),
            'win_rate': float(round(wr, 1)),
            # R-multiple distribution
            'mean_r': round(mean_r, 3),
            'median_r': round(median_r, 3),
            'std_r': round(std_r, 3),
            'positive_mean_r': round(positive_mean, 3),
            'negative_mean_r': round(negative_mean, 3),
            'r_clip_range': [R_CLIP_MIN, R_CLIP_MAX],
            # Regression metrics
            'train_mae': round(train_mae, 4),
            'val_mae': round(val_mae, 4),
            'val_rmse': round(val_rmse, 4),
            'val_r2': round(val_r2, 4),
            'val_correlation': round(val_corr, 4),
            # Training params
            'best_iteration': int(best_n),
            'take_threshold': TAKE_THRESHOLD,
            'caution_threshold': CAUTION_THRESHOLD,
            # Feature importance
            'top_features': [(f, float(round(i, 4))) for f, i in top_features],
            'calibration': calibration,
            'model_size_kb': float(round(os.path.getsize(MODEL_PATH) / 1024, 1)),
            'trained_at': datetime.now(timezone.utc).isoformat(),
        }
        with open(META_PATH, 'w') as f:
            json.dump(meta, f, indent=2, default=_json_default)

        log.info(f"[ML_GATE] Regression model trained: {len(y)} trades "
                 f"(MAE={val_mae:.3f}, R²={val_r2:.3f}, corr={val_corr:.3f}) "
                 f"top_feature={top_features[0][0]} ({top_features[0][1]:.3f})")

        return meta

    except ImportError as e:
        return {'status': 'error', 'reason': f'Missing dependency: {e}'}
    except Exception as e:
        log.error(f"[ML_GATE] Training failed: {e}")
        return {'status': 'error', 'reason': str(e)}


def _check_regression_calibration(preds: np.ndarray, actuals: np.ndarray,
                                 n_buckets: int = 5) -> list:
    """
    Check regression calibration by splitting predictions into quintiles.
    If the model is well-calibrated, predicted R should match actual R
    within each quintile — the bottom quintile should have the worst
    actual R and the top quintile should have the best.

    Returns list of {quintile, predicted_avg, actual_avg, actual_median, count}.
    """
    try:
        calibration = []
        sorted_idx = np.argsort(preds)
        chunk_size = max(1, len(sorted_idx) // n_buckets)

        for i in range(n_buckets):
            start = i * chunk_size
            end = start + chunk_size if i < n_buckets - 1 else len(sorted_idx)
            chunk_idx = sorted_idx[start:end]

            if len(chunk_idx) < 2:
                continue

            pred_mean = round(float(preds[chunk_idx].mean()), 3)
            actual_mean = round(float(actuals[chunk_idx].mean()), 3)
            actual_median = round(float(np.median(actuals[chunk_idx])), 3)
            actual_std = round(float(np.std(actuals[chunk_idx])), 3)
            win_rate = round(float((actuals[chunk_idx] > 0).sum() / len(chunk_idx) * 100), 1)

            calibration.append({
                f'quintile_{i+1}': {
                    'predicted_r': pred_mean,
                    'actual_mean_r': actual_mean,
                    'actual_median_r': actual_median,
                    'actual_std_r': actual_std,
                    'win_rate_pct': win_rate,
                    'count': int(len(chunk_idx)),
                }
            })
        return calibration
    except Exception:
        return []


# ════════════════════════════════════════════════════════════════
# CONVENIENCE: Collect all strategy scores for a bar
# ════════════════════════════════════════════════════════════════

def collect_all_strategy_scores(symbol: str,
                                df_m1, df_m5, df_m15, df_h1, df_h4,
                                smc_report: dict,
                                market_report: dict,
                                market_state: str,
                                session: str,
                                master_report: dict = None) -> dict:
    """
    Run ALL 10 strategies and return their scores (even if they don't pass gates).
    This is what makes the system "strategy-informed" — we collect scores from
    every strategy, not just the ones that pass.

    Returns dict: {strategy_name: score_or_None}
    """
    from strategies.strategy_engine import _run_one_strategy

    scores = {}
    strategies = [
        'SMC_OB_REVERSAL', 'LIQUIDITY_SWEEP_ENTRY', 'VWAP_MEAN_REVERSION',
        'DELTA_DIVERGENCE', 'TREND_CONTINUATION', 'FVG_REVERSION',
        'EMA_CROSS_MOMENTUM', 'RSI_DIVERGENCE_SMC', 'BREAKOUT_MOMENTUM',
        'STRUCTURE_ALIGNMENT',
    ]

    for name in strategies:
        try:
            # Run WITH relaxed=True for strategies that need it (DELTA_DIVERGENCE,
            # RSI_DIVERGENCE_SMC, BREAKOUT_MOMENTUM) so they bypass hard state/session
            # gates. This ensures we collect scores and shadow data from ALL strategies,
            # not just the ones whose state gates happen to match the current bar.
            signal = _run_one_strategy(
                name, symbol,
                df_m1, df_m5, df_m15, df_h1, df_h4,
                smc_report, market_report,
                market_state, session,
                master_report=master_report,
                relaxed=True)

            if signal is not None:
                scores[name] = signal.get('score', 0)
            else:
                scores[name] = 0
        except Exception:
            scores[name] = 0

    return scores
