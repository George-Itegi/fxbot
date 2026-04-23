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

            -- Price features
            atr                 DOUBLE,
            pip_from_vwap       DOUBLE,
            pip_to_poc          DOUBLE,
            va_width_pips       DOUBLE,

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

            -- Was this traded?
            was_traded          TINYINT,
            was_executed        TINYINT,
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
            INDEX idx_run_id (run_id)
        )
    """)

    c.close()


def _get_or_create_conn():
    """Get a MySQL connection, creating tables on first use."""
    from database.db_manager import get_connection
    conn = get_connection()
    try:
        _ensure_tables(conn)
    except Exception as e:
        log.warning(f"[DB_STORE] Table creation check: {e}")
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
                spread_pips: float = 0.0, slippage_pips: float = 0.0):
    """
    Store a completed backtest trade into MySQL.
    Includes ALL features needed for ML model training.
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

        # 66 columns = 65 %s + 1 literal 'BACKTEST'
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
                smc_bias, pd_zone, pips_to_eq, structure_trend,
                atr, pip_from_vwap, pip_to_poc, va_width_pips,
                spread_pips, slippage_pips,
                partial_tp_triggered, partial_tp_pips, partial_tp_usd,
                trail_activated, tp_extended, highest_profit_pips,
                source
            ) VALUES (
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,
                'BACKTEST'
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
            _safe_float((mr.get('atr') or 0)),  # atr from market_report
            _safe_float(vwap_d.get('pip_from_vwap', 0)),
            _safe_float(prof_d.get('pip_to_poc', 0)),
            _safe_float(prof_d.get('va_width_pips', 0)),
            _safe_float(spread_pips),
            _safe_float(slippage_pips),
            1 if trade.partial_tp_triggered else 0,
            _safe_float(trade.partial_tp_pips),
            _safe_float(trade.partial_tp_usd),
            1 if trade.trail_activated else 0,
            1 if trade.tp_extended else 0,
            _safe_float(trade.highest_profit_pips),
        ))

        conn.commit()
        c.close()
        conn.close()

    except Exception as e:
        log.warning(f"[DB_STORE] store_trade error: {e}")


def store_blocked_signal(symbol: str, direction: str, strategy: str,
                         score: int, confluence: list,
                         master_report: dict, market_report: dict,
                         smc_report: dict, flow_data: dict,
                         was_traded: bool, skip_reason: str,
                         run_id: str = 'default'):
    """
    Store a signal that was generated but blocked (consensus/gates/score).
    These are CRITICAL for ML — they tell the model what NOT to trade.
    """
    try:
        conn = _get_or_create_conn()
        c = conn.cursor(dictionary=True)

        mr = market_report or {}
        sr = smc_report or {}
        fd = flow_data or {}

        delta_d = fd.get('delta', {})
        rd_d = fd.get('rolling_delta', {})
        of_d = fd.get('order_flow_imbalance', {})

        strategy_group = 'UNKNOWN'
        try:
            from strategies.strategy_engine import _get_strategy_group
            strategy_group = _get_strategy_group(strategy)
        except Exception:
            pass

        timestamp_str = (master_report or {}).get('timestamp', '')

        c.execute("""
            INSERT INTO backtest_signals (
                run_id, timestamp, symbol, direction, strategy, strategy_group,
                score, confluence_count,
                was_traded, was_executed, skip_reason,
                session, market_state, combined_bias, final_score,
                delta, rolling_delta, of_imbalance, vol_surge,
                smc_bias, pd_zone, structure_trend
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            run_id, timestamp_str, symbol, direction, strategy, strategy_group,
            score, len(confluence) if confluence else 0,
            1 if was_traded else 0, 0, skip_reason[:100],
            (master_report or {}).get('session', 'UNKNOWN'),
            (master_report or {}).get('market_state', 'BALANCED'),
            (master_report or {}).get('combined_bias', 'NEUTRAL'),
            (master_report or {}).get('final_score', 0),
            _safe_float(delta_d.get('delta', 0)),
            _safe_float(rd_d.get('delta', 0)),
            _safe_float(of_d.get('imbalance', 0)),
            1 if (fd.get('volume_surge', {})).get('surge_detected') else 0,
            sr.get('smc_bias', 'NEUTRAL'),
            str((sr.get('premium_discount', {})).get('zone', 'NEUTRAL')),
            (sr.get('structure', {})).get('trend', 'RANGING'),
        ))

        conn.commit()
        c.close()
        conn.close()

    except Exception as e:
        log.warning(f"[DB_STORE] store_blocked_signal error: {e}")


def store_traded_signal(trade, run_id: str = 'default'):
    """Update a blocked signal to mark it as executed after trade opens."""
    try:
        conn = _get_or_create_conn()
        c = conn.cursor(dictionary=True)

        # Mark the closest matching signal as executed
        c.execute("""
            UPDATE backtest_signals
            SET was_executed = 1, was_traded = 1,
                skip_reason = 'EXECUTED'
            WHERE symbol = %s
              AND direction = %s
              AND strategy = %s
              AND run_id = %s
              AND was_executed = 0
            ORDER BY id DESC LIMIT 1
        """, (trade.symbol, trade.direction, trade.strategy, run_id))

        conn.commit()
        c.close()
        conn.close()
    except Exception as e:
        log.debug(f"[DB_STORE] store_traded_signal error: {e}")


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
    """Get quick stats about stored backtest data."""
    try:
        conn = _get_or_create_conn()
        c = conn.cursor(dictionary=True)

        c.execute("SELECT COUNT(*) as total FROM backtest_trades WHERE source='BACKTEST'")
        total = c.fetchone()['total']

        c.execute("SELECT COUNT(*) as total FROM backtest_trades WHERE source='BACKTEST' AND win=1")
        wins = c.fetchone()['total']

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
            WHERE source='BACKTEST'
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
