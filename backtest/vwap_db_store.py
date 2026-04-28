# =============================================================
# backtest/vwap_db_store.py  v1.0 — Dedicated VWAP L1 Tables
#
# PURPOSE: Separate database tables for VWAP_MEAN_REVERSION Layer 1
# model training. Keeps VWAP data isolated from the main backtest_trades
# table and stores VWAP-SPECIFIC features that the generic table lacks
# (ADX, VIX, Fear/Greed, StochRSI K/D, POC distance, Supertrend dir).
#
# WHY SEPARATE TABLES:
#   1. backtest_trades has ~85 columns, many irrelevant for VWAP
#   2. VWAP-specific features (ADX, VIX, StochRSI) are NOT in the
#      generic table — they're lost during DB storage
#   3. L1 model needs clean, strategy-specific feature columns
#   4. Walk-forward validation needs per-row timestamps + run_ids
#   5. Isolation: VWAP experiments don't pollute main training data
#
# TABLES:
#   vwap_signals — EVERY VWAP signal generated (including gate-blocked)
#     Stores the raw signal + VWAP features BEFORE trade execution.
#     Used to analyze "what would have happened if gates hadn't blocked"
#     and as the primary training set for the L1 model.
#
#   vwap_trades — ONLY VWAP trades that were actually executed/shadowed
#     Stores trade outcomes + features for P&L validation.
#     Links to vwap_signals via signal_id.
#
# USAGE:
#   from backtest.vwap_db_store import store_vwap_signal, store_vwap_trade
#   store_vwap_signal(signal, vwap_features, master_report, ...)
#   store_vwap_trade(trade, signal_id, vwap_features, ...)
# =============================================================

import datetime
import json
from core.logger import get_logger

log = get_logger(__name__)

_vwap_tables_ensured = False


def _get_or_create_conn():
    """Get a MySQL connection, ensuring VWAP tables exist on first use."""
    global _vwap_tables_ensured
    from database.db_manager import get_connection
    conn = get_connection()
    if not _vwap_tables_ensured:
        try:
            _ensure_vwap_tables(conn)
            _vwap_tables_ensured = True
        except Exception as e:
            log.warning(f"[VWAP_DB] Table creation check: {e}")
    # Consume any leftover unread results from pooled connection
    try:
        while conn.unread_result:
            conn.consume_results()
    except Exception:
        pass
    return conn


def _ensure_vwap_tables(conn):
    """Create VWAP-specific tables if they don't exist."""
    c = conn.cursor(dictionary=True)

    # ═══════════════════════════════════════════════════════════════
    # TABLE 1: vwap_signals
    # Every VWAP signal generated (with or without trade execution).
    # This is the PRIMARY training data for the L1 strategy model.
    # ═══════════════════════════════════════════════════════════════
    c.execute("""
        CREATE TABLE IF NOT EXISTS vwap_signals (
            id                  INT AUTO_INCREMENT PRIMARY KEY,
            run_id              VARCHAR(50),
            signal_time         DATETIME,
            symbol              VARCHAR(20),
            direction           VARCHAR(10),

            -- Trade parameters
            entry_price         DOUBLE,
            sl_price            DOUBLE,
            tp1_price           DOUBLE,
            tp2_price           DOUBLE,
            sl_pips             DOUBLE,
            tp1_pips            DOUBLE,
            tp2_pips            DOUBLE,
            rr_ratio            DOUBLE,

            -- Signal quality
            score               INT,
            confluence_count    INT,
            confluence          TEXT,

            -- ═══ VWAP-SPECIFIC FEATURES (NOT in backtest_trades) ═══
            -- These are the gate conditions converted to features.
            -- The L1 model learns their importance instead of using
            -- hard-coded thresholds.

            -- Volatility features
            atr_pips            DOUBLE,        -- ATR in pips (gate: < 2.0 blocked)
            adx                 DOUBLE,        -- ADX value (gate: > 45 blocked)

            -- External sentiment
            vix                 DOUBLE,        -- VIX index (gate: > 25 blocked)
            fear_greed_score    DOUBLE,        -- Fear & Greed 0-100

            -- Stochastic RSI
            stoch_rsi_k         DOUBLE,        -- StochRSI K line
            stoch_rsi_d         DOUBLE,        -- StochRSI D line

            -- VWAP / Volume Profile context
            pip_from_vwap       DOUBLE,        -- Distance from VWAP in pips
            vwap_position       VARCHAR(30),   -- BELOW/ABOVE/INSIDE_VA etc.
            va_position         VARCHAR(30),   -- BELOW_VAL/ABOVE_VAH/INSIDE_VA
            pd_zone             VARCHAR(30),   -- PREMIUM/DISCOUNT/NEUTRAL

            -- HTF alignment
            htf_approved        TINYINT,       -- HTF alignment approved (gate)

            -- Master bias
            master_bias         VARCHAR(15),   -- BULLISH/BEARISH/NEUTRAL (gate)

            -- Supertrend direction
            supertrend_dir      INT,           -- -1/0/1 (H1 supertrend)

            -- POC distance
            poc_distance_pips   DOUBLE,        -- Distance to POC in pips
            poc_above           TINYINT,       -- POC above price (BUY signal)
            poc_below           TINYINT,       -- POC below price (SELL signal)
            vah_above           TINYINT,       # VAH above price
            val_below           TINYINT,       -- VAL below price

            -- ═══ BASE FEATURES (shared with ML Gate v3.3) ═══
            -- These overlap with backtest_trades but are duplicated
            -- here for self-contained training queries.

            session             VARCHAR(30),
            market_state        VARCHAR(30),
            combined_bias       VARCHAR(15),
            final_score         INT,
            market_score        INT,
            smc_score           INT,

            -- Order flow
            delta               DOUBLE,
            rolling_delta       DOUBLE,
            delta_bias          DOUBLE,
            of_imbalance        DOUBLE,
            vol_surge           TINYINT,

            -- SMC structure
            smc_bias            VARCHAR(15),
            pips_to_eq          DOUBLE,
            structure_trend     VARCHAR(15),

            -- Price features
            atr                 DOUBLE,        -- ATR in price (not pips)
            price_position      VARCHAR(20),

            -- ═══ GATE TRACKING ═══
            -- Which gates would have blocked this signal?
            -- (All converted from hard gates to soft features in no-gate mode)
            hard_gates_blocked  VARCHAR(200),  -- JSON list of gate names that WOULD block

            -- ═══ OUTCOME (filled after trade execution) ═══
            was_traded          TINYINT DEFAULT 0,  -- 1 = executed as real/shadow
            trade_ticket        INT DEFAULT NULL,   -- Link to vwap_trades.ticket
            skip_reason         VARCHAR(100),        -- Why not traded

            -- ═══ SELF-CALIBRATION (for retraining) ═══
            model_predicted_r   DOUBLE DEFAULT NULL,
            strategy_model_verdict  VARCHAR(10) DEFAULT NULL,
            strategy_model_predicted_r DOUBLE DEFAULT NULL,

            INDEX idx_symbol (symbol),
            INDEX idx_direction (direction),
            INDEX idx_run_id (run_id),
            INDEX idx_signal_time (signal_time),
            INDEX idx_was_traded (was_traded),
            INDEX idx_market_state (market_state),
            INDEX idx_session (session),
            INDEX idx_adx (adx),
            INDEX idx_vix (vix),
            INDEX idx_stoch_rsi_k (stoch_rsi_k),
            INDEX idx_pip_from_vwap (pip_from_vwap)
        )
    """)

    # ═══════════════════════════════════════════════════════════════
    # TABLE 2: vwap_trades
    # Only VWAP trades that were actually executed (real or shadow).
    # Stores trade outcomes for P&L validation.
    # ═══════════════════════════════════════════════════════════════
    c.execute("""
        CREATE TABLE IF NOT EXISTS vwap_trades (
            id                  INT AUTO_INCREMENT PRIMARY KEY,
            run_id              VARCHAR(50),
            signal_id           INT,           -- Link to vwap_signals.id
            ticket              INT,
            symbol              VARCHAR(20),
            direction           VARCHAR(10),
            source              VARCHAR(20) DEFAULT 'BACKTEST',  -- BACKTEST or SHADOW

            -- Timing
            entry_time          DATETIME,
            exit_time           DATETIME,
            duration_minutes    INT,
            session             VARCHAR(30),

            -- Prices
            entry_price         DOUBLE,
            exit_price          DOUBLE,
            sl_price            DOUBLE,
            tp_price            DOUBLE,
            original_sl         DOUBLE,
            original_tp         DOUBLE,
            sl_pips             DOUBLE,
            tp_pips             DOUBLE,

            -- Results
            profit_pips         DOUBLE,
            profit_r            DOUBLE,
            profit_usd          DOUBLE,
            outcome             VARCHAR(30),
            exit_reason         VARCHAR(20),
            win                 TINYINT,

            -- Position sizing
            lot_size            DOUBLE,
            risk_percent        DOUBLE,

            -- Signal quality at entry
            score               INT,
            confluence_count    INT,
            confluence          TEXT,

            -- ═══ VWAP-SPECIFIC FEATURES (snapshot at entry) ═══
            atr_pips            DOUBLE,
            adx                 DOUBLE,
            vix                 DOUBLE,
            fear_greed_score    DOUBLE,
            stoch_rsi_k         DOUBLE,
            stoch_rsi_d         DOUBLE,
            pip_from_vwap       DOUBLE,
            vwap_position       VARCHAR(30),
            pd_zone             VARCHAR(30),
            htf_approved        TINYINT,
            master_bias         VARCHAR(15),
            supertrend_dir      INT,
            poc_distance_pips   DOUBLE,

            -- ═══ BASE FEATURES ═══
            market_state        VARCHAR(30),
            combined_bias       VARCHAR(15),
            final_score         INT,
            delta               DOUBLE,
            rolling_delta       DOUBLE,
            delta_bias          DOUBLE,
            of_imbalance        DOUBLE,
            vol_surge           TINYINT,
            smc_bias            VARCHAR(15),
            atr                 DOUBLE,
            price_position      VARCHAR(20),

            -- Trade management
            partial_tp_triggered   TINYINT,
            partial_tp_pips        DOUBLE,
            trail_activated        TINYINT,
            tp_extended            TINYINT,
            highest_profit_pips    DOUBLE,

            -- ═══ SELF-CALIBRATION ═══
            model_predicted_r   DOUBLE DEFAULT NULL,
            strategy_model_verdict  VARCHAR(10) DEFAULT NULL,
            strategy_model_predicted_r DOUBLE DEFAULT NULL,

            INDEX idx_symbol (symbol),
            INDEX idx_direction (direction),
            INDEX idx_source (source),
            INDEX idx_run_id (run_id),
            INDEX idx_entry_time (entry_time),
            INDEX idx_outcome (outcome),
            INDEX idx_win (win),
            INDEX idx_signal_id (signal_id)
        )
    """)

    c.close()


def _safe_float(val, default=0.0) -> float:
    """Safely convert to float."""
    try:
        return round(float(val), 6)
    except (TypeError, ValueError):
        return default


def _safe_int(val, default=0) -> int:
    """Safely convert to int."""
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _determine_blocked_gates(vwap_features: dict) -> list:
    """
    Determine which original hard gates would have blocked this signal.
    Used for analysis: "the model learned to reject what gate X would reject".
    Returns list of gate names that WOULD block in the original strategy.
    """
    blocked = []

    # Original hard gates from vwap_mean_reversion.py v1.1:
    if vwap_features.get('atr_pips', 99) < 2.0:
        blocked.append('LOW_ATR')
    if vwap_features.get('adx', 0) > 45:
        blocked.append('HIGH_ADX')
    if vwap_features.get('vix', 0) > 25:
        blocked.append('HIGH_VIX')
    if vwap_features.get('master_bias') == 'BEARISH':
        # Would only block BUY — simplified here as general gate
        blocked.append('MASTER_BIAS')
    if 'PREMIUM' in vwap_features.get('pd_zone', '') and 'DISCOUNT' not in vwap_features.get('pd_zone', ''):
        blocked.append('PREMIUM_ZONE')

    return blocked


def store_vwap_signal(
        signal: dict,
        vwap_features: dict,
        master_report: dict = None,
        market_report: dict = None,
        smc_report: dict = None,
        flow_data: dict = None,
        run_id: str = 'default',
        signal_time: datetime.datetime = None):
    """
    Store a VWAP signal into the vwap_signals table.

    This is called for EVERY VWAP signal, regardless of whether it gets
    traded. The signal_id is returned so we can link the trade outcome later.

    Args:
        signal: The VWAP signal dict from evaluate() or evaluate_no_gates()
        vwap_features: The _vwap_features dict from the signal (ADX, VIX, etc.)
        master_report: Market master report at signal time
        market_report: Market report at signal time
        smc_report: SMC report at signal time
        flow_data: Order flow data at signal time
        run_id: Run identifier
        signal_time: When the signal was generated

    Returns:
        signal_id (int) or None on error
    """
    try:
        conn = _get_or_create_conn()
        c = conn.cursor(dictionary=True)

        mr = market_report or {}
        sr = smc_report or {}
        fd = flow_data or {}
        vf = vwap_features or signal.get('_vwap_features', {})

        # Extract nested data
        delta_d = fd.get('delta', {})
        rd_d = fd.get('rolling_delta', {})
        of_d = fd.get('order_flow_imbalance', {})
        vs_d = fd.get('volume_surge', {})
        pd_d = sr.get('premium_discount', {})
        htf_d = sr.get('htf_alignment', {})
        struct_d = sr.get('structure', {})
        vwap_d = mr.get('vwap', {})
        prof_d = mr.get('profile', {})

        # Bias encoding
        bias_map = {'BULLISH': 1.0, 'BEARISH': -1.0, 'NEUTRAL': 0.0}
        delta_bias_val = bias_map.get(str(delta_d.get('bias', 'NEUTRAL')), 0.0)

        # R:R ratio
        sl_p = _safe_float(signal.get('sl_pips', 0))
        tp_p = _safe_float(signal.get('tp1_pips', 0))
        rr = round(tp_p / sl_p, 2) if sl_p > 0 else 0.0

        # Confluence
        confluence = signal.get('confluence', [])
        confluence_str = json.dumps(confluence) if confluence else '[]'

        # Signal time
        if signal_time is None:
            signal_time = datetime.datetime.now(datetime.timezone.utc)

        # Determine which gates would have blocked this signal
        blocked_gates = _determine_blocked_gates(vf)
        blocked_gates_str = json.dumps(blocked_gates)

        # Dedup check
        time_str = signal_time.strftime('%Y-%m-%d %H:%M:%S')
        symbol = signal.get('symbol', signal.get('strategy', 'VWAP_MEAN_REVERSION'))

        c.execute("""
            SELECT id FROM vwap_signals
            WHERE symbol = %s AND direction = %s AND signal_time = %s AND run_id = %s
            LIMIT 1
        """, (symbol, signal.get('direction', ''), time_str, run_id))
        if c.fetchone():
            c.close()
            conn.close()
            return None

        c.execute("""
            INSERT INTO vwap_signals (
                run_id, signal_time, symbol, direction,
                entry_price, sl_price, tp1_price, tp2_price,
                sl_pips, tp1_pips, tp2_pips, rr_ratio,
                score, confluence_count, confluence,
                atr_pips, adx, vix, fear_greed_score,
                stoch_rsi_k, stoch_rsi_d,
                pip_from_vwap, vwap_position, va_position, pd_zone,
                htf_approved, master_bias, supertrend_dir,
                poc_distance_pips, poc_above, poc_below, vah_above, val_below,
                session, market_state, combined_bias,
                final_score, market_score, smc_score,
                delta, rolling_delta, delta_bias, of_imbalance, vol_surge,
                smc_bias, pips_to_eq, structure_trend,
                atr, price_position,
                hard_gates_blocked
            ) VALUES (
                %s,%s,%s,%s,
                %s,%s,%s,%s,
                %s,%s,%s,%s,
                %s,%s,%s,
                %s,%s,%s,%s,
                %s,%s,
                %s,%s,%s,%s,
                %s,%s,%s,
                %s,%s,%s,%s,%s,
                %s,%s,%s,
                %s,%s,%s,
                %s,%s,%s,%s,%s,
                %s,%s,%s,
                %s,%s,
                %s
            )
        """, (
            run_id,
            time_str,
            symbol,
            signal.get('direction', ''),
            # Prices
            _safe_float(signal.get('entry_price')),
            _safe_float(signal.get('sl_price')),
            _safe_float(signal.get('tp1_price')),
            _safe_float(signal.get('tp2_price')),
            # Pips
            sl_p, tp_p,
            _safe_float(signal.get('tp2_pips')),
            rr,
            # Score
            signal.get('score', 0),
            len(confluence) if confluence else 0,
            confluence_str,
            # VWAP-specific features
            _safe_float(vf.get('atr_pips')),
            _safe_float(vf.get('adx')),
            _safe_float(vf.get('vix')),
            _safe_float(vf.get('fg_score')),
            _safe_float(vf.get('stoch_k')),
            _safe_float(vf.get('stoch_d')),
            _safe_float(vf.get('pip_from_vwap')),
            str(vf.get('vwap_pos', '')),
            str(vf.get('va_pos', '')),
            str(vf.get('pd_zone', '')),
            1 if vf.get('htf_ok', True) else 0,
            str(vf.get('master_bias', '')),
            _safe_int(vf.get('supertrend_dir', 0)),
            _safe_float(vf.get('poc_dist', 0)),
            1 if vf.get('poc_above', 0) else 0,
            1 if vf.get('poc_below', 0) else 0,
            1 if vf.get('vah_above', 0) else 0,
            1 if vf.get('val_below', 0) else 0,
            # Base features
            (master_report or {}).get('session', 'UNKNOWN'),
            (master_report or {}).get('market_state', 'BALANCED'),
            (master_report or {}).get('combined_bias', 'NEUTRAL'),
            (master_report or {}).get('final_score', 0),
            (master_report or {}).get('market_score', 0),
            (master_report or {}).get('smc_score', 0),
            # Order flow
            _safe_float(delta_d.get('delta', 0)),
            _safe_float(rd_d.get('delta', 0)),
            delta_bias_val,
            _safe_float(of_d.get('imbalance', 0)),
            1 if vs_d.get('surge_detected') else 0,
            # SMC
            sr.get('smc_bias', 'NEUTRAL'),
            _safe_float(pd_d.get('pips_to_eq', 0)),
            struct_d.get('trend', 'RANGING'),
            # Price
            _safe_float(mr.get('atr') or 0),
            str(prof_d.get('price_position', 'INSIDE_VA')),
            # Gate tracking
            blocked_gates_str,
        ))

        conn.commit()
        signal_id = c.lastrowid
        c.close()
        conn.close()

        return signal_id

    except Exception as e:
        log.warning(f"[VWAP_DB] store_vwap_signal error: {e}")
        return None


def store_vwap_trade(
        trade,
        signal_id: int = None,
        vwap_features: dict = None,
        master_report: dict = None,
        market_report: dict = None,
        smc_report: dict = None,
        flow_data: dict = None,
        run_id: str = 'default',
        spread_pips: float = 0.0,
        slippage_pips: float = 0.0,
        source: str = 'BACKTEST',
        model_predicted_r: float = None,
        strategy_model_verdict: str = None,
        strategy_model_predicted_r: float = None):
    """
    Store a VWAP trade outcome into the vwap_trades table.

    Args:
        trade: BacktestTrade object
        signal_id: Link to vwap_signals.id
        vwap_features: VWAP-specific features from _vwap_features dict
        master_report: Market report at entry
        market_report: Market report at entry
        smc_report: SMC report at entry
        flow_data: Order flow data at entry
        run_id: Run identifier
        source: 'BACKTEST' (real trade) or 'SHADOW' (simulated)
        model_predicted_r: L2 model prediction
        strategy_model_verdict: L1 model verdict
        strategy_model_predicted_r: L1 model predicted R
    """
    try:
        conn = _get_or_create_conn()
        c = conn.cursor(dictionary=True)

        mr = market_report or {}
        sr = smc_report or {}
        fd = flow_data or {}
        vf = vwap_features or {}

        delta_d = fd.get('delta', {})
        rd_d = fd.get('rolling_delta', {})
        of_d = fd.get('order_flow_imbalance', {})
        vs_d = fd.get('volume_surge', {})
        pd_d = sr.get('premium_discount', {})
        struct_d = sr.get('structure', {})
        prof_d = mr.get('profile', {})

        bias_map = {'BULLISH': 1.0, 'BEARISH': -1.0, 'NEUTRAL': 0.0}
        delta_bias_val = bias_map.get(str(delta_d.get('bias', 'NEUTRAL')), 0.0)

        is_win = 1 if trade.profit_pips > 0 else 0

        duration_min = 0
        if trade.entry_time and trade.exit_time:
            duration_min = int((trade.exit_time - trade.entry_time).total_seconds() / 60)

        entry_time_str = trade.entry_time.strftime('%Y-%m-%d %H:%M:%S') if trade.entry_time else None
        exit_time_str = trade.exit_time.strftime('%Y-%m-%d %H:%M:%S') if trade.exit_time else None

        confluence = trade.confluence or []
        confluence_str = json.dumps(confluence) if confluence else '[]'

        # Dedup
        c.execute("""
            SELECT id FROM vwap_trades
            WHERE symbol = %s AND direction = %s AND entry_time = %s AND run_id = %s
            LIMIT 1
        """, (trade.symbol, trade.direction, entry_time_str, run_id))
        if c.fetchone():
            c.close()
            conn.close()
            return

        c.execute("""
            INSERT INTO vwap_trades (
                run_id, signal_id, ticket, symbol, direction, source,
                entry_time, exit_time, duration_minutes, session,
                entry_price, exit_price, sl_price, tp_price,
                original_sl, original_tp, sl_pips, tp_pips,
                profit_pips, profit_r, profit_usd, outcome, exit_reason, win,
                lot_size, risk_percent,
                score, confluence_count, confluence,
                atr_pips, adx, vix, fear_greed_score,
                stoch_rsi_k, stoch_rsi_d,
                pip_from_vwap, vwap_position, pd_zone,
                htf_approved, master_bias, supertrend_dir,
                poc_distance_pips,
                market_state, combined_bias, final_score,
                delta, rolling_delta, delta_bias, of_imbalance, vol_surge,
                smc_bias, atr, price_position,
                partial_tp_triggered, partial_tp_pips,
                trail_activated, tp_extended, highest_profit_pips,
                model_predicted_r,
                strategy_model_verdict, strategy_model_predicted_r
            ) VALUES (
                %s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,
                %s,%s,%s,%s,
                %s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,
                %s,%s,
                %s,%s,%s,
                %s,%s,%s,%s,
                %s,%s,%s,%s,
                %s,%s,%s,%s,
                %s,%s,%s,
                %s,%s,%s,%s,%s,
                %s,%s,%s,
                %s,%s,%s,%s,%s,
                %s,
                %s,%s
            )
        """, (
            run_id, signal_id, trade.ticket, trade.symbol, trade.direction, source,
            entry_time_str, exit_time_str, duration_min, trade.session,
            _safe_float(trade.entry_price),
            _safe_float(trade.exit_price),
            _safe_float(trade.sl_price),
            _safe_float(trade.tp_price),
            _safe_float(trade.original_sl_price),
            _safe_float(trade.original_tp_price),
            _safe_float(trade.sl_pips),
            _safe_float(trade.tp_pips),
            _safe_float(trade.profit_pips),
            _safe_float(trade.profit_r),
            _safe_float(trade.profit_usd),
            trade.outcome, trade.exit_reason, is_win,
            _safe_float(trade.lot_size),
            _safe_float(trade.risk_percent),
            trade.score,
            len(confluence) if confluence else 0,
            confluence_str,
            # VWAP-specific features
            _safe_float(vf.get('atr_pips')),
            _safe_float(vf.get('adx')),
            _safe_float(vf.get('vix')),
            _safe_float(vf.get('fg_score')),
            _safe_float(vf.get('stoch_k')),
            _safe_float(vf.get('stoch_d')),
            _safe_float(vf.get('pip_from_vwap')),
            str(vf.get('vwap_pos', '')),
            str(vf.get('pd_zone', '')),
            1 if vf.get('htf_ok', True) else 0,
            str(vf.get('master_bias', '')),
            _safe_int(vf.get('supertrend_dir', 0)),
            _safe_float(vf.get('poc_dist', 0)),
            # Base features
            trade.market_state,
            (master_report or {}).get('combined_bias', 'NEUTRAL'),
            (master_report or {}).get('final_score', 0),
            _safe_float(delta_d.get('delta', 0)),
            _safe_float(rd_d.get('delta', 0)),
            delta_bias_val,
            _safe_float(of_d.get('imbalance', 0)),
            1 if vs_d.get('surge_detected') else 0,
            sr.get('smc_bias', 'NEUTRAL'),
            _safe_float(mr.get('atr') or 0),
            str(prof_d.get('price_position', 'INSIDE_VA')),
            # Trade management
            1 if trade.partial_tp_triggered else 0,
            _safe_float(trade.partial_tp_pips),
            1 if trade.trail_activated else 0,
            1 if trade.tp_extended else 0,
            _safe_float(trade.highest_profit_pips),
            # Self-calibration
            model_predicted_r,
            strategy_model_verdict,
            strategy_model_predicted_r,
        ))

        conn.commit()
        c.close()
        conn.close()

    except Exception as e:
        log.warning(f"[VWAP_DB] store_vwap_trade error: {e}")


def link_vwap_signal_to_trade(
        signal_time: datetime.datetime,
        symbol: str,
        direction: str,
        run_id: str = 'default'):
    """
    Mark a vwap_signal as 'was_traded' after its trade is executed.
    Called during DB storage phase to link signal → trade.
    """
    try:
        conn = _get_or_create_conn()
        c = conn.cursor(dictionary=True)
        time_str = signal_time.strftime('%Y-%m-%d %H:%M:%S')

        c.execute("""
            UPDATE vwap_signals
            SET was_traded = 1
            WHERE symbol = %s AND direction = %s AND signal_time = %s AND run_id = %s
              AND was_traded = 0
        """, (symbol, direction, time_str, run_id))

        conn.commit()
        affected = c.rowcount
        c.close()
        conn.close()
        return affected > 0

    except Exception as e:
        log.warning(f"[VWAP_DB] link_signal error: {e}")
        return False


def get_vwap_training_data() -> list:
    """
    Fetch all VWAP trades from vwap_trades for L1 model training.
    Includes VWAP-specific features + trade outcomes.
    Returns list of dicts.
    """
    try:
        conn = _get_or_create_conn()
        c = conn.cursor(dictionary=True)

        c.execute("""
            SELECT
                id, symbol, direction, source,
                entry_time, exit_time, duration_minutes, session,
                sl_pips, tp_pips, score, confluence_count,
                -- VWAP-specific features (THE KEY ADDITION)
                atr_pips, adx, vix, fear_greed_score,
                stoch_rsi_k, stoch_rsi_d,
                pip_from_vwap, vwap_position, pd_zone,
                htf_approved, master_bias, supertrend_dir,
                poc_distance_pips,
                -- Base features
                market_state, combined_bias, final_score,
                delta, rolling_delta, delta_bias, of_imbalance, vol_surge,
                smc_bias, atr, price_position,
                -- Outcome
                profit_pips, profit_r, profit_usd, outcome, win,
                -- Trade management
                partial_tp_triggered, trail_activated, tp_extended,
                highest_profit_pips,
                -- Self-calibration
                model_predicted_r,
                strategy_model_verdict, strategy_model_predicted_r
            FROM vwap_trades
            WHERE source IN ('BACKTEST', 'SHADOW')
              AND outcome IS NOT NULL
              AND profit_r IS NOT NULL
            ORDER BY entry_time ASC
        """)

        rows = c.fetchall()
        c.close()
        conn.close()
        return rows

    except Exception as e:
        log.error(f"[VWAP_DB] get_vwap_training_data error: {e}")
        return []


def get_vwap_signal_stats() -> dict:
    """
    Get stats about VWAP signals and trades.
    Useful for monitoring data collection progress.
    """
    try:
        conn = _get_or_create_conn()
        c = conn.cursor(dictionary=True)

        # Total signals
        c.execute("SELECT COUNT(*) as cnt FROM vwap_signals")
        total_signals = c.fetchone()['cnt']

        # Signals traded vs not traded
        c.execute("SELECT COUNT(*) as cnt FROM vwap_signals WHERE was_traded = 1")
        traded = c.fetchone()['cnt']
        c.execute("SELECT COUNT(*) as cnt FROM vwap_signals WHERE was_traded = 0")
        not_traded = c.fetchone()['cnt']

        # Trade outcomes
        c.execute("SELECT COUNT(*) as cnt FROM vwap_trades WHERE source='BACKTEST'")
        real_trades = c.fetchone()['cnt']
        c.execute("SELECT COUNT(*) as cnt FROM vwap_trades WHERE source='SHADOW'")
        shadow_trades = c.fetchone()['cnt']
        c.execute("SELECT COUNT(*) as cnt FROM vwap_trades WHERE win = 1")
        wins = c.fetchone()['cnt']

        # Win rates
        c.execute("""
            SELECT source, COUNT(*) as trades, SUM(win) as wins,
                   ROUND(AVG(profit_r), 3) as avg_r,
                   ROUND(SUM(profit_pips), 1) as total_pips
            FROM vwap_trades
            WHERE outcome IS NOT NULL
            GROUP BY source
        """)
        by_source = c.fetchall()

        # Per-symbol stats
        c.execute("""
            SELECT symbol, COUNT(*) as trades, SUM(win) as wins,
                   ROUND(AVG(profit_r), 3) as avg_r
            FROM vwap_trades
            WHERE outcome IS NOT NULL
            GROUP BY symbol
            ORDER BY trades DESC
        """)
        by_symbol = c.fetchall()

        # Gate analysis: how many signals blocked by each gate?
        c.execute("""
            SELECT hard_gates_blocked, COUNT(*) as cnt
            FROM vwap_signals
            GROUP BY hard_gates_blocked
            ORDER BY cnt DESC
            LIMIT 20
        """)
        gate_stats = c.fetchall()

        # VWAP feature distributions
        c.execute("""
            SELECT
                ROUND(AVG(adx), 1) as avg_adx,
                ROUND(AVG(vix), 1) as avg_vix,
                ROUND(AVG(atr_pips), 1) as avg_atr_pips,
                ROUND(AVG(stoch_rsi_k), 1) as avg_stoch_k,
                ROUND(AVG(pip_from_vwap), 1) as avg_vwap_dist,
                ROUND(AVG(poc_distance_pips), 1) as avg_poc_dist
            FROM vwap_signals
        """)
        feature_stats = c.fetchone()

        # Sessions
        c.execute("""
            SELECT session, COUNT(*) as signals,
                   SUM(was_traded) as traded_count,
                   SUM(CASE WHEN was_traded=1 AND
                       (SELECT win FROM vwap_trades vt
                        WHERE vt.symbol=vs.symbol
                          AND vt.entry_time=vs.signal_time LIMIT 1) = 1
                       THEN 1 ELSE 0 END) as win_count
            FROM vwap_signals vs
            GROUP BY session
            ORDER BY signals DESC
        """)
        by_session = c.fetchall()

        c.close()
        conn.close()

        total_trades = real_trades + shadow_trades
        return {
            'total_signals': total_signals,
            'signals_traded': traded,
            'signals_not_traded': not_traded,
            'trade_conversion_pct': round(traded / total_signals * 100, 1) if total_signals > 0 else 0,
            'real_trades': real_trades,
            'shadow_trades': shadow_trades,
            'total_trades': total_trades,
            'total_wins': wins,
            'overall_wr': round(wins / total_trades * 100, 1) if total_trades > 0 else 0,
            'by_source': by_source,
            'by_symbol': by_symbol,
            'by_session': by_session,
            'gate_analysis': gate_stats,
            'feature_stats': feature_stats,
        }

    except Exception as e:
        log.error(f"[VWAP_DB] get_vwap_signal_stats error: {e}")
        return {'total_signals': 0, 'error': str(e)}


def get_vwap_walkforward_data(
        window_size: int = 300,
        step_size: int = 100,
        min_test_trades: int = 30) -> list:
    """
    Generate walk-forward validation windows from VWAP trade data.

    Each window is a dict:
      {
        'window_id': 1,
        'train_start': '2024-01-01',
        'train_end': '2024-06-01',
        'test_start': '2024-06-01',
        'test_end': '2024-08-01',
        'train_data': [rows...],
        'test_data': [rows...],
        'train_count': 300,
        'test_count': 50,
      }

    Args:
        window_size: Number of trades in training window
        step_size: Number of trades to advance per step
        min_test_trades: Minimum test trades for a valid window
    """
    try:
        all_data = get_vwap_training_data()
        if len(all_data) < window_size + min_test_trades:
            return []

        windows = []
        window_id = 1
        start = 0

        while start + window_size + min_test_trades <= len(all_data):
            train_data = all_data[start:start + window_size]
            test_start_idx = start + window_size

            # Find the test end (next step_size trades or until end)
            test_end_idx = min(test_start_idx + step_size, len(all_data))
            test_data = all_data[test_start_idx:test_end_idx]

            if len(test_data) < min_test_trades:
                break

            windows.append({
                'window_id': window_id,
                'train_start': train_data[0].get('entry_time', ''),
                'train_end': train_data[-1].get('entry_time', ''),
                'test_start': test_data[0].get('entry_time', ''),
                'test_end': test_data[-1].get('entry_time', ''),
                'train_data': train_data,
                'test_data': test_data,
                'train_count': len(train_data),
                'test_count': len(test_data),
            })

            window_id += 1
            start += step_size

        return windows

    except Exception as e:
        log.error(f"[VWAP_DB] get_vwap_walkforward_data error: {e}")
        return []
