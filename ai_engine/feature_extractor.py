# =============================================================
# ai_engine/feature_extractor.py
# PURPOSE: Central feature engineering for Strategy-Informed ML.
#
# ARCHITECTURE — Strategy-Informed Approach 3:
#   Layer 1: Market features (delta, VWAP, OB, sweeps, etc.)
#   Layer 2: Strategy scores as features (not decisions)
#   Layer 3: Context features (session, volatility, symbol type)
#   Total: 60 features → XGBoost predicts WIN probability
#
# The key insight: every strategy's score becomes an input.
# XGBoost learns WHICH COMBINATION of strategy scores wins.
# No more hardcoded rules — the model discovers the patterns.
# =============================================================

import numpy as np
from core.logger import get_logger

log = get_logger(__name__)

FEATURE_COUNT = 60

FEATURE_NAMES = [
    # ── Market quality (8) ──────────────────────────────────
    'final_score',          # 0-100 master score
    'market_score',         # 0-100 market component
    'smc_score',            # 0-100 SMC component
    'delta_value',          # raw delta (buy-sell ticks)
    'rolling_delta',        # short-term delta
    'of_imbalance',         # -1 to +1 order flow imbalance
    'of_strength',          # 0-4 (NONE/WEAK/MOD/STRONG/EXTREME)
    'volume_surge_ratio',   # surge multiplier (1.0 = no surge)

    # ── VWAP context (4) ────────────────────────────────────
    'vwap_distance_pips',   # pips from VWAP (signed)
    'vwap_position',        # -2 to +2 (far below to far above)
    'poc_distance_pips',    # pips from POC (signed)
    'va_width_pips',        # value area width

    # ── SMC structure (8) ───────────────────────────────────
    'has_bos',              # 1=break of structure confirmed
    'has_choch',            # 1=change of character (reversal)
    'ob_distance_pips',     # pips to nearest order block
    'ob_type',              # 1=bullish OB, -1=bearish OB, 0=none
    'fvg_quality',          # 0-100 quality of nearest FVG
    'fvg_distance_pips',    # pips to nearest FVG
    'sweep_reversal_pips',  # pips reversed after sweep
    'pd_zone',              # -2 to +2 (extreme premium to extreme discount)

    # ── HTF alignment (3) ───────────────────────────────────
    'htf_approved',         # 1=approved, -1=rejected
    'htf_score',            # 0-100
    'h4_ema_aligned',       # 1=aligned with direction, -1=against

    # ── Trade parameters (5) ────────────────────────────────
    'sl_pips',              # stop loss in pips
    'tp1_pips',             # take profit 1 in pips
    'rr_ratio',             # tp/sl ratio
    'direction',            # 1=BUY, -1=SELL
    'confluence_count',     # number of confluence factors

    # ── Strategy scores — THE KEY INNOVATION (12) ───────────
    # Each strategy returns its own score (0-100) or 0 if it
    # did not fire. XGBoost learns which combination wins.
    'score_ema_trend',
    'score_smc_ob',
    'score_liquidity_sweep',
    'score_vwap_reversion',
    'score_order_flow_exhaustion',
    'score_m1_scalp',
    'score_orb',
    'score_delta_divergence',
    'score_trend_continuation',
    'score_smart_money_footprint',
    'score_fvg_reversion',
    'score_rsi_divergence_smc',

    # ── Consensus features (4) ──────────────────────────────
    'groups_agreeing',      # how many different groups agreed
    'strategies_fired',     # total strategies that fired
    'buy_signals_count',    # BUY signal count
    'sell_signals_count',   # SELL signal count

    # ── Session and time context (5) ────────────────────────
    'session',              # 0-4 encoded session quality
    'hour_utc',             # 0-23 hour of day
    'is_london',            # 1=London session
    'is_ny',                # 1=NY session
    'is_overlap',           # 1=London-NY overlap

    # ── Volatility and market state (5) ─────────────────────
    'atr_pips',             # current ATR in pips
    'market_state',         # 0-5 encoded market state
    'is_trending',          # 1=trending market
    'is_breakout',          # 1=breakout condition
    'momentum_scalpable',   # 1=momentum active

    # ── Symbol type (3) ─────────────────────────────────────
    'is_forex_major',       # 1=major pair
    'is_gold',              # 1=XAUUSD
    'is_index',             # 1=US30/US500/USTEC etc.

    # ── Self-improvement features (3) ────────────────────────
    'symbol_recent_wr',     # recent win rate on this symbol (0-1)
    'strategy_recent_wr',   # recent win rate of this strategy (0-1)
    'session_recent_wr',    # recent win rate in this session (0-1)
]

assert len(FEATURE_NAMES) == FEATURE_COUNT, \
    f"Feature count mismatch: {len(FEATURE_NAMES)} != {FEATURE_COUNT}"


# ── Encoding maps (shared) ───────────────────────────────────
SESSION_MAP = {
    'NY_LONDON_OVERLAP': 4,
    'LONDON_SESSION':    3,
    'LONDON_OPEN':       3,
    'NY_AFTERNOON':      2,
    'TOKYO':             1,
    'SYDNEY':            0,
}

OF_STRENGTH_MAP = {
    'EXTREME':  4,
    'STRONG':   3,
    'MODERATE': 2,
    'WEAK':     1,
    'NONE':     0,
}

PD_MAP = {
    'EXTREME_PREMIUM':  -2,
    'PREMIUM':          -1,
    'NEUTRAL':           0,
    'DISCOUNT':          1,
    'EXTREME_DISCOUNT':  2,
}

MARKET_STATE_MAP = {
    'TRENDING_STRONG':    5,
    'BREAKOUT_ACCEPTED':  4,
    'TRENDING_EXTENDED':  3,
    'BALANCED':           2,
    'BREAKOUT_REJECTED':  1,
    'REVERSAL_RISK':      0,
}

STRATEGY_SCORE_KEYS = [
    ('EMA_TREND_MTF',         'score_ema_trend'),
    ('SMC_OB_REVERSAL',       'score_smc_ob'),
    ('LIQUIDITY_SWEEP_ENTRY', 'score_liquidity_sweep'),
    ('VWAP_MEAN_REVERSION',   'score_vwap_reversion'),
    ('ORDER_FLOW_EXHAUSTION', 'score_order_flow_exhaustion'),
    ('M1_MOMENTUM_SCALP',     'score_m1_scalp'),
    ('OPENING_RANGE_BREAKOUT','score_orb'),
    ('DELTA_DIVERGENCE',      'score_delta_divergence'),
    ('TREND_CONTINUATION',    'score_trend_continuation'),
    ('SMART_MONEY_FOOTPRINT', 'score_smart_money_footprint'),
    ('FVG_REVERSION',         'score_fvg_reversion'),
    ('RSI_DIVERGENCE_SMC',    'score_rsi_divergence_smc'),
]


def extract_features(signal: dict,
                     master_report: dict,
                     market_report: dict,
                     smc_report: dict,
                     all_signals: list = None,
                     symbol: str = None,
                     performance_cache: dict = None) -> np.ndarray | None:
    """
    Extract 60 features from a signal and its market context.
    This is the core of Strategy-Informed ML.

    Args:
        signal         : The best signal dict from strategy_engine
        master_report  : Full master_report from master_scanner
        market_report  : Market-level analysis
        smc_report     : SMC structure analysis
        all_signals    : ALL signals from this scan cycle (for consensus features)
        symbol         : Symbol name for type detection
        performance_cache: Recent win rates by symbol/strategy/session

    Returns np.ndarray of shape (1, 60) or None if extraction fails.
    """
    try:
        m  = market_report or {}
        s  = smc_report    or {}
        mr = master_report or {}

        d    = m.get('delta', {})
        rd   = m.get('rolling_delta', {})
        of   = m.get('order_flow_imbalance',
                     mr.get('order_flow_imbalance', {}))
        vwap = m.get('vwap', {})
        prof = m.get('profile', {})
        pd_z = s.get('premium_discount', {})
        htf  = s.get('htf_alignment', {})
        mom  = m.get('momentum',
                     mr.get('momentum', {}))
        surge= m.get('volume_surge',
                     mr.get('volume_surge', {}))

        direction  = 1.0 if signal.get('direction') == 'BUY' else -1.0
        sl_pips    = float(signal.get('sl_pips', 10) or 10)
        tp1_pips   = float(signal.get('tp1_pips', 20) or 20)
        rr_ratio   = round(tp1_pips / sl_pips, 2) if sl_pips > 0 else 0.0
        session    = mr.get('session', m.get('session', 'UNKNOWN'))
        hour_utc   = 0
        try:
            from datetime import datetime, timezone
            hour_utc = datetime.now(timezone.utc).hour
        except Exception:
            pass

        # Market state
        market_state_str = mr.get('market_state',
                                   m.get('market_state', 'BALANCED'))

        # OB distance
        ob_dist  = 999.0
        ob_type  = 0.0
        nearest_ob = s.get('nearest_ob', {})
        if nearest_ob:
            try:
                close = float(m.get('profile', {}).get(
                    'current_price', 0) or 0)
                ob_mid = ((float(nearest_ob.get('top', 0)) +
                           float(nearest_ob.get('bottom', 0))) / 2)
                pip_size = _pip_size(symbol or '', close)
                ob_dist = abs(close - ob_mid) / pip_size if pip_size > 0 else 999
                ob_type = 1.0 if 'BULLISH' in str(
                    nearest_ob.get('type', '')) else -1.0
            except Exception:
                pass

        # FVG quality and distance
        fvg_quality = 0.0
        fvg_dist    = 999.0
        nearest_fvg = s.get('nearest_fvg', {})
        if nearest_fvg:
            try:
                fvg_quality = float(nearest_fvg.get('quality_score', 0) or 0)
                close = float(prof.get('current_price', 0) or 0)
                pip_size = _pip_size(symbol or '', close)
                fvg_mid = float(nearest_fvg.get('mid', close))
                fvg_dist = abs(close - fvg_mid) / pip_size if pip_size > 0 else 999
            except Exception:
                pass

        # Strategy scores — collect from all_signals if available
        strat_scores = {k: 0.0 for _, k in STRATEGY_SCORE_KEYS}
        if all_signals:
            for sig in all_signals:
                strat_name = sig.get('strategy', '')
                strat_score = float(sig.get('score', 0) or 0)
                for name, key in STRATEGY_SCORE_KEYS:
                    if strat_name == name:
                        strat_scores[key] = strat_score
        # Always include the winning signal's score
        sig_strat = signal.get('strategy', '')
        sig_score = float(signal.get('score', 0) or 0)
        for name, key in STRATEGY_SCORE_KEYS:
            if sig_strat == name:
                strat_scores[key] = sig_score

        # Consensus features
        buy_sigs  = [s for s in (all_signals or []) if s.get('direction') == 'BUY']
        sell_sigs = [s for s in (all_signals or []) if s.get('direction') == 'SELL']
        from strategies.strategy_engine import _get_strategy_group, STRATEGY_GROUPS
        direction_sigs = buy_sigs if signal.get('direction') == 'BUY' else sell_sigs
        groups_agreeing = len(set(
            _get_strategy_group(s.get('strategy', ''))
            for s in direction_sigs))

        # Symbol type detection
        sym_upper = (symbol or '').upper()
        is_gold   = 1.0 if 'XAU' in sym_upper else 0.0
        is_index  = 1.0 if any(x in sym_upper for x in
                               ['US30','US500','USTEC','JP225','UK100','DE30']) else 0.0
        majors    = ['EURUSD','GBPUSD','USDJPY','USDCHF','USDCAD','AUDUSD','NZDUSD']
        is_major  = 1.0 if sym_upper in majors else 0.0

        # BOS / CHoCH
        structure = s.get('structure', {})
        has_bos   = 1.0 if structure.get('bos') else 0.0
        has_choch = 1.0 if structure.get('choch') else 0.0

        # H4 EMA alignment
        h4_aligned = 0.0
        if master_report:
            h4_bias = htf.get('h4_bias', '')
            if direction > 0 and h4_bias == 'BULLISH':  h4_aligned = 1.0
            elif direction < 0 and h4_bias == 'BEARISH': h4_aligned = 1.0
            else:                                         h4_aligned = -1.0

        # Sweep reversal
        sweep_rev = 0.0
        last_sweep = s.get('last_sweep', {})
        if last_sweep:
            sweep_rev = float(last_sweep.get('reversal_pips', 0) or 0)

        # Recent win rates (from performance cache)
        pc = performance_cache or {}
        symbol_wr   = float(pc.get(f'symbol_{sym_upper}', 0.5))
        strategy_wr = float(pc.get(f'strategy_{sig_strat}', 0.5))
        session_wr  = float(pc.get(f'session_{session}', 0.5))

        # ATR from market report
        atr_pips = float(m.get('atr_pips', 0) or
                         prof.get('va_width_pips', 20) or 20)

        features = [
            # Market quality (8)
            float(mr.get('final_score', 0) or 0) / 100,
            float(mr.get('market_score', 0) or 0) / 100,
            float(mr.get('smc_score', 0) or 0) / 100,
            float(d.get('delta', 0) or 0) / 10000,
            float(rd.get('delta', 0) or 0) / 5000,
            float(of.get('imbalance', 0) or 0),
            float(OF_STRENGTH_MAP.get(of.get('strength', 'NONE'), 0)) / 4,
            min(float(surge.get('surge_ratio', 1.0) or 1.0) / 5, 1.0),

            # VWAP context (4)
            float(vwap.get('pip_from_vwap', 0) or 0) / 50,
            float({'FAR_ABOVE': 2,'ABOVE': 1,'AT_VWAP': 0,
                   'BELOW': -1,'FAR_BELOW': -2}.get(
                   vwap.get('position','AT_VWAP'), 0)) / 2,
            float(prof.get('pip_to_poc', 0) or 0) / 50,
            float(prof.get('va_width_pips', 20) or 20) / 100,

            # SMC structure (8)
            has_bos,
            has_choch,
            min(ob_dist / 50, 1.0),
            (ob_type + 1) / 2,
            fvg_quality / 100,
            min(fvg_dist / 100, 1.0),
            min(sweep_rev / 20, 1.0),
            (float(PD_MAP.get(pd_z.get('zone', 'NEUTRAL'), 0)) + 2) / 4,

            # HTF alignment (3)
            1.0 if htf.get('approved') else 0.0,
            float(htf.get('score', 50) or 50) / 100,
            (h4_aligned + 1) / 2,

            # Trade parameters (5)
            min(sl_pips / 30, 1.0),
            min(tp1_pips / 60, 1.0),
            min(rr_ratio / 5, 1.0),
            (direction + 1) / 2,
            min(float(len(signal.get('confluence', []))) / 10, 1.0),

            # Strategy scores (12) — all normalised 0-1
            strat_scores['score_ema_trend'] / 100,
            strat_scores['score_smc_ob'] / 100,
            strat_scores['score_liquidity_sweep'] / 100,
            strat_scores['score_vwap_reversion'] / 100,
            strat_scores['score_order_flow_exhaustion'] / 100,
            strat_scores['score_m1_scalp'] / 100,
            strat_scores['score_orb'] / 100,
            strat_scores['score_delta_divergence'] / 100,
            strat_scores['score_trend_continuation'] / 100,
            strat_scores['score_smart_money_footprint'] / 100,
            strat_scores['score_fvg_reversion'] / 100,
            strat_scores['score_rsi_divergence_smc'] / 100,

            # Consensus features (4)
            min(groups_agreeing / 5, 1.0),
            min(len(all_signals or []) / 10, 1.0),
            min(len(buy_sigs) / 10, 1.0),
            min(len(sell_sigs) / 10, 1.0),

            # Session and time context (5)
            float(SESSION_MAP.get(session, 1)) / 4,
            float(hour_utc) / 23,
            1.0 if 'LONDON' in session else 0.0,
            1.0 if 'NY' in session else 0.0,
            1.0 if session == 'NY_LONDON_OVERLAP' else 0.0,

            # Volatility and market state (5)
            min(atr_pips / 50, 1.0),
            float(MARKET_STATE_MAP.get(market_state_str, 2)) / 5,
            1.0 if 'TRENDING' in market_state_str else 0.0,
            1.0 if 'BREAKOUT' in market_state_str else 0.0,
            1.0 if mom.get('is_scalpable') else 0.0,

            # Symbol type (3)
            is_major,
            is_gold,
            is_index,

            # Self-improvement (3)
            symbol_wr,
            strategy_wr,
            session_wr,
        ]

        arr = np.array(features, dtype=np.float32).reshape(1, -1)
        assert arr.shape == (1, FEATURE_COUNT), \
            f"Feature shape {arr.shape} != (1,{FEATURE_COUNT})"
        return arr

    except Exception as e:
        log.error(f"[FEATURES] Extraction failed: {e}")
        return None


def _pip_size(symbol: str, price: float) -> float:
    """Determine pip size from symbol name."""
    s = symbol.upper()
    if any(x in s for x in ['US30','US500','USTEC','JP225','DE30','UK100']):
        return 1.0
    if 'XAU' in s: return 0.1
    if 'XAG' in s: return 0.01
    if 'JPY' in s: return 0.01
    return 0.0001
