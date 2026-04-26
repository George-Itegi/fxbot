# =============================================================
# backtest/db_store.py  v1.0
# Stores backtest trades and signals into MySQL for model training.
#
# WHY: The XGBoost and LSTM models need 200+ trades with rich
# feature data to learn what makes a winning trade. Backtesting
# generates this data quickly. Every trade + every blocked signal
# gets stored with 21+ features for ML training.
#
# Tables:
#   backtest_trades   — every executed trade with full feature set
#   backtest_signals  — every signal (traded OR blocked) with features
#
# Features stored (matching XGBoost extract_features):
#   score, sl_pips, tp1_pips, tp2_pips, direction, session,
#   delta, rolling_delta, delta_bias, rd_bias, vwap_pip_from,
#   vwap_position, pip_to_poc, price_position, va_width_pips,
#   pd_zone, pips_to_eq, htf_approved, htf_score,
#   market_score, smc_score, combined_bias, market_state,
#   atr, volatility, spread, order_flow_imbalance, etc.
# =============================================================

import datetime
import json
from core.logger import get_logger

log = get_logger(__name__)


def _ensure_tables(conn):
    """Create backtest tables if they don't exist."""
    c = conn.cursor(dictionary=True)

    c.execute("""
        CREATE TABLE IF NOT EXISTS backtest_trades (
            id                  INT AUTO_INCREMENT PRIMARY KEY,
            run_id              VARCHAR(50),
            ticket              INT,
            symbol              VARCHAR(20),
            direction           VARCHAR(10),
            strategy            VARCHAR(50),
            strategy_group      VARCHAR(30),

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
            conviction          VARCHAR(10),
            agreement_groups    INT,

            -- Signal quality
            score               INT,
            confluence_count    INT,
            confluence          TEXT,

            -- Market state at entry
            market_state        VARCHAR(30),
            combined_bias       VARCHAR(15),
            bias_confidence     VARCHAR(15),
            final_score         INT,
            market_score        INT,
            smc_score           INT,
            htf_approved        TINYINT,

            -- Order flow features (for ML)
            delta               DOUBLE,
            rolling_delta       DOUBLE,
            delta_bias          DOUBLE,
            rd_bias             DOUBLE,
            of_imbalance        DOUBLE,
            of_strength         VARCHAR(20),
            vol_surge_detected  TINYINT,
            vol_surge_ratio     DOUBLE,

            -- Momentum features
            momentum_velocity   DOUBLE,
            momentum_direction  VARCHAR(10),
            is_choppy           TINYINT,

            -- SMC features
            smc_bias            VARCHAR(15),
            pd_zone             VARCHAR(30),
            pips_to_eq          DOUBLE,
            structure_trend     VARCHAR(15),
            htf_score           INT DEFAULT 50,

            -- Price features
            atr                 DOUBLE,
            pip_from_vwap       DOUBLE,
            pip_to_poc          DOUBLE,
            va_width_pips       DOUBLE,
            price_position      VARCHAR(20) DEFAULT 'INSIDE_VA',

            -- Spread info
            spread_pips         DOUBLE,
            slippage_pips       DOUBLE,

            -- Partial TP
            partial_tp_triggered   TINYINT,
            partial_tp_pips        DOUBLE,
            partial_tp_usd         DOUBLE,
            trail_activated        TINYINT,
            tp_extended            TINYINT,
            highest_profit_pips    DOUBLE,

            -- Strategy scores (all 10, for ML Gate v3.0 training)
            ss_smc_ob           INT DEFAULT 0,
            ss_liquidity_sweep  INT DEFAULT 0,
            ss_vwap_reversion   INT DEFAULT 0,
            ss_delta_divergence INT DEFAULT 0,
            ss_trend_continuation INT DEFAULT 0,
            ss_fvg_reversion    INT DEFAULT 0,
            ss_ema_cross        INT DEFAULT 0,
            ss_rsi_divergence   INT DEFAULT 0,
            ss_breakout_momentum INT DEFAULT 0,
            ss_structure_align  INT DEFAULT 0,

            -- Fibonacci confluence (3 features for ML Gate v3.1)
            fib_confluence_score DOUBLE DEFAULT 0,
            fib_in_golden_zone   TINYINT DEFAULT 0,
            fib_bias_aligned     TINYINT DEFAULT 0,

            -- Flag: is this a backtest trade?
            source              VARCHAR(20) DEFAULT 'BACKTEST',

            INDEX idx_symbol (symbol),
            INDEX idx_strategy (strategy),
            INDEX idx_outcome (outcome),
            INDEX idx_session (session),
            INDEX idx_market_state (market_state),
            INDEX idx_source (source),
            INDEX idx_run_id (run_id),
            INDEX idx_win (win)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS backtest_signals (
            id                  INT AUTO_INCREMENT PRIMARY KEY,
            run_id              VARCHAR(50),
            timestamp           DATETIME,
            symbol              VARCHAR(20),
            direction           VARCHAR(10),
            strategy            VARCHAR(50),
            strategy_group      VARCHAR(30),

            -- Signal quality
            score               INT,
            confluence_count    INT,
            confluence          TEXT,              -- Full confluence list (comma-separated)

            -- Was this traded?
            was_traded          TINYINT,
            was_executed        TINYINT,
            trade_ticket        INT DEFAULT NULL,   -- Links to backtest_trades.ticket
            skip_reason         VARCHAR(100),

            -- Market context
            session             VARCHAR(30),
            market_state        VARCHAR(30),
            combined_bias       VARCHAR(15),
            final_score         INT,

            -- Order flow
            delta               DOUBLE,
            rolling_delta       DOUBLE,
            of_imbalance        DOUBLE,
            vol_surge           TINYINT,

            -- SMC
            smc_bias            VARCHAR(15),
            pd_zone             VARCHAR(30),
            structure_trend     VARCHAR(15),

            -- If traded, the outcome
            outcome             VARCHAR(30),
            profit_r            DOUBLE,

            INDEX idx_symbol (symbol),
            INDEX idx_strategy (strategy),
            INDEX idx_was_traded (was_traded),
            INDEX idx_was_executed (was_executed),
            INDEX idx_run_id (run_id),
            INDEX idx_trade_ticket (trade_ticket)
        )
    """)

    # Note: backtest_signals table no longer populated (blocked signals removed)
    # Only backtest_trades is active for ML training
    _auto_migrate_trades(c, conn)
    c.close()


def _auto_migrate_trades(cursor, conn):
    """Add missing columns to backtest_trades if they don't exist.
    Note: backtest_signals migrations removed (table no longer populated)."""
    migrations = [
        ('backtest_trades', 'htf_score',      'INT DEFAULT 50'),
        ('backtest_trades', 'price_position', "VARCHAR(20) DEFAULT 'INSIDE_VA'"),
        # ── ML Gate v3.0: 10 strategy score columns ──
        ('backtest_trades', 'ss_smc_ob',           'DOUBLE DEFAULT 0'),
        ('backtest_trades', 'ss_liquidity_sweep',  'DOUBLE DEFAULT 0'),
        ('backtest_trades', 'ss_vwap_reversion',   'DOUBLE DEFAULT 0'),
        ('backtest_trades', 'ss_delta_divergence', 'DOUBLE DEFAULT 0'),
        ('backtest_trades', 'ss_trend_continuation','DOUBLE DEFAULT 0'),
        ('backtest_trades', 'ss_fvg_reversion',    'DOUBLE DEFAULT 0'),
        ('backtest_trades', 'ss_ema_cross',        'DOUBLE DEFAULT 0'),
        ('backtest_trades', 'ss_rsi_divergence',   'DOUBLE DEFAULT 0'),
        ('backtest_trades', 'ss_breakout_momentum','DOUBLE DEFAULT 0'),
        ('backtest_trades', 'ss_structure_align',  'DOUBLE DEFAULT 0'),
        ('backtest_trades', 'fib_confluence_score', 'DOUBLE DEFAULT 0'),
        ('backtest_trades', 'fib_in_golden_zone',   'TINYINT DEFAULT 0'),
        ('backtest_trades', 'fib_bias_aligned',     'TINYINT DEFAULT 0'),
    ]
    for table, col, col_def in migrations:
        try:
            cursor.execute(f"SELECT {col} FROM {table} LIMIT 1")
            cursor.fetchall()  # consume result set to avoid "Unread result found"
        except Exception:
            try:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_def}")
                conn.commit()
                log.info(f"[DB_STORE] Auto-migrated: added {table}.{col}")
            except Exception as e:
                log.warning(f"[DB_STORE] Migration failed for {table}.{col}: {e}")


_tables_ensured = False


def _get_or_create_conn():
    """Get a MySQL connection, ensuring tables exist on first use only."""
    global _tables_ensured
    from database.db_manager import get_connection
    conn = get_connection()
    # Only check/create tables ONCE per process — not on every connection
    if not _tables_ensured:
        try:
            _ensure_tables(conn)
            _tables_ensured = True
        except Exception as e:
            log.warning(f"[DB_STORE] Table creation check: {e}")
    # Consume any leftover unread results from pooled connection
    try:
        while conn.unread_result:
            conn.consume_results()
    except Exception:
        pass
    return conn


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


def store_trade(trade, master_report: dict = None,
                market_report: dict = None, smc_report: dict = None,
                flow_data: dict = None, run_id: str = 'default',
                spread_pips: float = 0.0, slippage_pips: float = 0.0,
                strategy_scores: dict = None, source: str = 'BACKTEST'):
    """
    Store a completed backtest trade into MySQL.
    Includes ALL features needed for ML model training.

    strategy_scores: dict of {strategy_name: score} for ALL 10 strategies.
                     This is the KEY data for ML Gate v3.0 training.
    """
    try:
        conn = _get_or_create_conn()
        c = conn.cursor(dictionary=True)

        # Market features
        mr = market_report or {}
        sr = smc_report or {}
        fd = flow_data or {}

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

        # Encode bias values
        bias_map = {'BULLISH': 1.0, 'BEARISH': -1.0, 'NEUTRAL': 0.0}
        delta_bias_val = bias_map.get(str(delta_d.get('bias', 'NEUTRAL')), 0.0)
        rd_bias_val = bias_map.get(str(rd_d.get('bias', 'NEUTRAL')), 0.0)

        # Encode strategy scores (KEY for ML Gate v3.0)
        ss = strategy_scores or {}
        ss_smc_ob           = _safe_float(ss.get('SMC_OB_REVERSAL', 0))
        ss_liquidity_sweep  = _safe_float(ss.get('LIQUIDITY_SWEEP_ENTRY', 0))
        ss_vwap_reversion   = _safe_float(ss.get('VWAP_MEAN_REVERSION', 0))
        ss_delta_divergence = _safe_float(ss.get('DELTA_DIVERGENCE', 0))
        ss_trend_continuation = _safe_float(ss.get('TREND_CONTINUATION', 0))
        ss_fvg_reversion    = _safe_float(ss.get('FVG_REVERSION', 0))
        ss_ema_cross        = _safe_float(ss.get('EMA_CROSS_MOMENTUM', 0))
        ss_rsi_divergence   = _safe_float(ss.get('RSI_DIVERGENCE_SMC', 0))
        ss_breakout_momentum = _safe_float(ss.get('BREAKOUT_MOMENTUM', 0))
        ss_structure_align  = _safe_float(ss.get('STRUCTURE_ALIGNMENT', 0))

        # Fibonacci confluence (from strategy_scores._fib_data)
        # check_fib_confluence returns: {fib_bonus, in_golden_zone, fib_bias_aligned, ...}
        fib_data = (strategy_scores or {}).get('_fib_data', {})
        fib_confluence_score = _safe_float(fib_data.get('fib_bonus', 0))
        fib_in_golden_zone = 1 if fib_data.get('in_golden_zone', False) else 0
        fib_bias_aligned = 1 if fib_data.get('fib_bias_aligned', False) else 0

        # Outcome: was this a win?
        is_win = 1 if trade.profit_pips > 0 else 0

        # Duration in minutes
        duration_min = 0
        if trade.entry_time and trade.exit_time:
            duration_min = int((trade.exit_time - trade.entry_time).total_seconds() / 60)

        # Convert times for MySQL
        entry_time_str = trade.entry_time.strftime('%Y-%m-%d %H:%M:%S') if trade.entry_time else None
        exit_time_str = trade.exit_time.strftime('%Y-%m-%d %H:%M:%S') if trade.exit_time else None

        # Strategy group
        strategy_group = 'UNKNOWN'
        try:
            from strategies.strategy_engine import _get_strategy_group
            strategy_group = _get_strategy_group(trade.strategy)
        except Exception:
            pass

        # ── Dedup: skip if this exact trade already exists ──
        c.execute("""
            SELECT id FROM backtest_trades
            WHERE symbol = %s AND strategy = %s AND direction = %s
              AND entry_time = %s AND run_id = %s
            LIMIT 1
        """, (trade.symbol, trade.strategy, trade.direction,
               entry_time_str, run_id))
        if c.fetchone():
            c.close()
            conn.close()
            log.debug(f"[DB_STORE] Skipping duplicate trade: {trade.symbol} {trade.strategy} {entry_time_str}")
            return

        # 81 columns = 80 %s + 1 literal 'BACKTEST'
        c.execute("""
            INSERT INTO backtest_trades (
                run_id, ticket, symbol, direction, strategy, strategy_group,
                entry_time, exit_time, duration_minutes, session,
                entry_price, exit_price, sl_price, tp_price,
                original_sl, original_tp, sl_pips, tp_pips,
                profit_pips, profit_r, profit_usd, outcome, exit_reason, win,
                lot_size, risk_percent, conviction, agreement_groups,
                score, confluence_count, confluence,
                market_state, combined_bias, bias_confidence,
                final_score, market_score, smc_score, htf_approved,
                delta, rolling_delta, delta_bias, rd_bias,
                of_imbalance, of_strength, vol_surge_detected, vol_surge_ratio,
                momentum_velocity, momentum_direction, is_choppy,
                smc_bias, pd_zone, pips_to_eq, structure_trend, htf_score,
                atr, pip_from_vwap, pip_to_poc, va_width_pips, price_position,
                spread_pips, slippage_pips,
                partial_tp_triggered, partial_tp_pips, partial_tp_usd,
                trail_activated, tp_extended, highest_profit_pips,
                ss_smc_ob, ss_liquidity_sweep, ss_vwap_reversion,
                ss_delta_divergence, ss_trend_continuation,
                ss_fvg_reversion, ss_ema_cross, ss_rsi_divergence,
                ss_breakout_momentum, ss_structure_align,
                fib_confluence_score, fib_in_golden_zone, fib_bias_aligned,
                source
            ) VALUES (
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s
                , %s
            )
        """, (
            run_id, trade.ticket, trade.symbol, trade.direction,
            trade.strategy, strategy_group,
            entry_time_str, exit_time_str, duration_min,
            trade.session,
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
            trade.conviction,
            trade.agreement_groups,
            trade.score,
            len(trade.confluence) if trade.confluence else 0,
            json.dumps(trade.confluence) if trade.confluence else '[]',
            trade.market_state,
            (master_report or {}).get('combined_bias', 'NEUTRAL'),
            (master_report or {}).get('bias_confidence', 'MODERATE'),
            (master_report or {}).get('final_score', 0),
            (master_report or {}).get('market_score', 0),
            (master_report or {}).get('smc_score', 0),
            1 if (master_report or {}).get('htf_approved') else 0,
            _safe_float(delta_d.get('delta', 0)),
            _safe_float(rd_d.get('delta', 0)),
            delta_bias_val, rd_bias_val,
            _safe_float(of_d.get('imbalance', 0)),
            str(of_d.get('strength', 'NONE')),
            1 if vs_d.get('surge_detected') else 0,
            _safe_float(vs_d.get('surge_ratio', 1.0)),
            _safe_float(mom_d.get('velocity_pips_min', 0)),
            str(mom_d.get('velocity_direction', 'FLAT')),
            1 if mom_d.get('is_choppy') else 0,
            sr.get('smc_bias', 'NEUTRAL'),
            str(pd_d.get('zone', 'NEUTRAL')),
            _safe_float(pd_d.get('pips_to_eq', 0)),
            struct_d.get('trend', 'RANGING'),
            _safe_int(htf_d.get('score', 50)),
            _safe_float((mr.get('atr') or 0)),
            _safe_float(vwap_d.get('pip_from_vwap', 0)),
            _safe_float(prof_d.get('pip_to_poc', 0)),
            _safe_float(prof_d.get('va_width_pips', 0)),
            str(prof_d.get('price_position', 'INSIDE_VA')),
            _safe_float(spread_pips),
            _safe_float(slippage_pips),
            1 if trade.partial_tp_triggered else 0,
            _safe_float(trade.partial_tp_pips),
            _safe_float(trade.partial_tp_usd),
            1 if trade.trail_activated else 0,
            1 if trade.tp_extended else 0,
            _safe_float(trade.highest_profit_pips),
            # ── Strategy scores (10 features for ML Gate v3.0) ──
            ss_smc_ob, ss_liquidity_sweep, ss_vwap_reversion,
            ss_delta_divergence, ss_trend_continuation,
            ss_fvg_reversion, ss_ema_cross, ss_rsi_divergence,
            ss_breakout_momentum, ss_structure_align,
            # ── Fibonacci confluence (3 features for ML Gate v3.1) ──
            fib_confluence_score, fib_in_golden_zone, fib_bias_aligned,
            # ── Source: BACKTEST (real) or SHADOW (simulated) ──
            source,
        ))

        conn.commit()
        c.close()
        conn.close()

    except Exception as e:
        log.warning(f"[DB_STORE] store_trade error: {e}")


# ── BLOCKED SIGNAL STORAGE REMOVED (v1.1) ──────────────────
# Blocked signals were never used in ML training (only executed
# trades with outcomes are used). Storing them caused DB pool
# exhaustion with 17 concurrent pairs. Removed to eliminate the
# #1 source of connection pool drain.
# ── SIGNAL EXECUTED / OUTCOME UPDATES REMOVED ──────────────
# These updated the backtest_signals table which is no longer
# populated. Dead code — removed along with signal storage.


def get_training_data(min_trades: int = 50) -> list:
    """
    Fetch all backtest trades in ML-ready format.
    Returns list of dicts with features + label (win/loss).
    """
    try:
        conn = _get_or_create_conn()
        c = conn.cursor(dictionary=True)

        c.execute("""
            SELECT
                symbol, direction, strategy, session, market_state,
                score, sl_pips, tp_pips, confluence_count,
                delta, rolling_delta, delta_bias, rd_bias,
                of_imbalance, vol_surge_detected,
                momentum_velocity, momentum_direction, is_choppy,
                smc_bias, pd_zone, pips_to_eq, structure_trend,
                atr, pip_from_vwap, pip_to_poc, va_width_pips,
                final_score, market_score, smc_score, htf_approved,
                combined_bias, conviction, agreement_groups,
                profit_pips, profit_r, profit_usd, outcome, win
            FROM backtest_trades
            ORDER BY entry_time ASC
        """)

        rows = c.fetchall()
        c.close()
        conn.close()

        return rows

    except Exception as e:
        log.error(f"[DB_STORE] get_training_data error: {e}")
        return []


def get_stats() -> dict:
    """Get quick stats about stored backtest data (real + shadow)."""
    try:
        conn = _get_or_create_conn()
        c = conn.cursor(dictionary=True)

        c.execute("SELECT COUNT(*) as total FROM backtest_trades WHERE source='BACKTEST'")
        total = c.fetchone()['total']

        c.execute("SELECT COUNT(*) as total FROM backtest_trades WHERE source='BACKTEST' AND win=1")
        wins = c.fetchone()['total']

        c.execute("SELECT COUNT(*) as total FROM backtest_trades WHERE source='SHADOW'")
        shadow_total = c.fetchone()['total']

        c.execute("SELECT COUNT(*) as total FROM backtest_trades WHERE source='SHADOW' AND win=1")
        shadow_wins = c.fetchone()['total']

        c.execute("SELECT COUNT(*) as total FROM backtest_signals WHERE was_executed=0")
        blocked = c.fetchone()['total']

        c.execute("SELECT COUNT(*) as total FROM backtest_signals WHERE was_executed=1")
        executed_signals = c.fetchone()['total']

        c.execute("""
            SELECT strategy, COUNT(*) as trades,
                   SUM(win) as wins,
                   ROUND(AVG(profit_r),3) as avg_r,
                   ROUND(AVG(score),1) as avg_score
            FROM backtest_trades
            WHERE source IN ('BACKTEST', 'SHADOW')
            GROUP BY strategy
        """)
        strat_rows = c.fetchall()

        c.execute("""
            SELECT session, COUNT(*) as trades,
                   SUM(win) as wins,
                   ROUND(AVG(profit_r),3) as avg_r
            FROM backtest_trades
            WHERE source='BACKTEST'
            GROUP BY session
        """)
        session_rows = c.fetchall()

        c.execute("""
            SELECT market_state, COUNT(*) as trades,
                   SUM(win) as wins,
                   ROUND(AVG(profit_r),3) as avg_r
            FROM backtest_trades
            WHERE source='BACKTEST'
            GROUP BY market_state
        """)
        state_rows = c.fetchall()

        c.execute("""
            SELECT skip_reason, COUNT(*) as cnt
            FROM backtest_signals
            WHERE was_executed = 0
            GROUP BY skip_reason
            ORDER BY cnt DESC
            LIMIT 10
        """)
        skip_rows = c.fetchall()

        c.close()
        conn.close()

        return {
            'total_trades': total,
            'total_wins': wins,
            'shadow_trades': shadow_total,
            'shadow_wins': shadow_wins,
            'total_blocked_signals': blocked,
            'total_executed_signals': executed_signals,
            'win_rate': round(wins / total * 100, 1) if total > 0 else 0,
            'by_strategy': strat_rows,
            'by_session': session_rows,
            'by_market_state': state_rows,
            'skip_reasons': skip_rows,
        }

    except Exception as e:
        log.error(f"[DB_STORE] get_stats error: {e}")
        return {'total_trades': 0}
