# =============================================================
# backtest/db_store.py  v1.2
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

            -- Strategy scores (for ML Gate v3.0 training)
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
            ss_supply_demand    INT DEFAULT 0,
            ss_bos_momentum     INT DEFAULT 0,
            ss_optimal_trade    INT DEFAULT 0,
            ss_institutional    INT DEFAULT 0,

            -- Fibonacci confluence (3 features for ML Gate v3.1)
            fib_confluence_score DOUBLE DEFAULT 0,
            fib_in_golden_zone   TINYINT DEFAULT 0,
            fib_bias_aligned     TINYINT DEFAULT 0,

            -- Flag: is this a backtest trade?
            source              VARCHAR(20) DEFAULT 'BACKTEST',

            -- ML Gate model prediction (for self-calibration)
            model_predicted_r   DOUBLE DEFAULT NULL,

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

    # ── VWAP-specific features table (1:1 with backtest_trades) ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS backtest_vwap_features (
            id              INT AUTO_INCREMENT PRIMARY KEY,
            trade_id        INT NOT NULL,
            atr_pips        DOUBLE,
            adx             DOUBLE,
            vix             DOUBLE,
            fg_score        DOUBLE,
            vwap_pos        VARCHAR(20),
            va_pos          VARCHAR(20),
            pd_zone         VARCHAR(20),
            htf_ok          INT,
            master_bias     VARCHAR(10),
            stoch_k         DOUBLE,
            stoch_d         DOUBLE,
            poc_dist        DOUBLE,
            supertrend_dir  INT,
            poc_above       INT,
            val_below       INT,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (trade_id) REFERENCES backtest_trades(id)
                ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    # ── Breakout-specific features table (1:1 with backtest_trades) ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS backtest_breakout_features (
            id                  INT AUTO_INCREMENT PRIMARY KEY,
            trade_id            INT NOT NULL,
            consol_type         VARCHAR(20),
            range_pips          DOUBLE,
            adx                 DOUBLE,
            atr_pips            DOUBLE,
            atr_ratio           DOUBLE,
            retest              INT,
            dist_to_level       DOUBLE,
            delta_confirms      INT,
            of_imbalance        DOUBLE,
            of_strength         VARCHAR(20),
            vol_surge           INT,
            vol_surge_ratio     DOUBLE,
            h4_trend_aligned    INT,
            h4_supertrend       INT,
            m5_momentum         INT,
            bos_aligned         INT,
            is_choppy           INT,
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (trade_id) REFERENCES backtest_trades(id)
                ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    # ── SMC OB-specific features table (1:1 with backtest_trades) ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS backtest_smc_ob_features (
            id                  INT AUTO_INCREMENT PRIMARY KEY,
            trade_id            INT NOT NULL,
            ob_type             VARCHAR(20),
            ob_dist_pips        DOUBLE,
            price_at_ob         INT,
            trend               VARCHAR(15),
            delta_bias          VARCHAR(15),
            delta_strength      VARCHAR(15),
            of_imbalance        DOUBLE,
            of_strength         VARCHAR(20),
            stoch_rsi_k         DOUBLE,
            supertrend_dir_h1   INT,
            htf_ok              INT,
            smc_bias            VARCHAR(15),
            pd_zone             VARCHAR(20),
            atr_pips            DOUBLE,
            has_bos             INT,
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (trade_id) REFERENCES backtest_trades(id)
                ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    # ── Liquidity Sweep-specific features table (1:1 with backtest_trades) ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS backtest_liq_sweep_features (
            id                  INT AUTO_INCREMENT PRIMARY KEY,
            trade_id            INT NOT NULL,
            sweep_bias          VARCHAR(15),
            reversal_pips       DOUBLE,
            swept_level_dist    DOUBLE,
            delta_bias          VARCHAR(15),
            delta_strength      VARCHAR(15),
            has_bos             INT,
            bos_type            VARCHAR(20),
            stoch_rsi_k         DOUBLE,
            supertrend_dir_h1   INT,
            htf_ok              INT,
            smc_bias            VARCHAR(15),
            pd_zone             VARCHAR(20),
            vol_surge           INT,
            of_imbalance        DOUBLE,
            atr_pips            DOUBLE,
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (trade_id) REFERENCES backtest_trades(id)
                ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    # ── Delta Divergence-specific features table (1:1 with backtest_trades) ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS backtest_delta_div_features (
            id                  INT AUTO_INCREMENT PRIMARY KEY,
            trade_id            INT NOT NULL,
            div_type            VARCHAR(15),
            div_strength        VARCHAR(15),
            swing_range_pips    DOUBLE,
            delta_value         DOUBLE,
            delta_bias          VARCHAR(15),
            of_imbalance        DOUBLE,
            of_strength         VARCHAR(20),
            vol_surge           INT,
            surge_ratio         DOUBLE,
            surge_absorption    INT,
            stoch_rsi_k         DOUBLE,
            stoch_rsi_turning   INT,
            pd_zone             VARCHAR(20),
            m5_body_ratio       DOUBLE,
            atr_pips            DOUBLE,
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (trade_id) REFERENCES backtest_trades(id)
                ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    # ── Trend Continuation-specific features table (1:1 with backtest_trades) ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS backtest_trend_cont_features (
            id                  INT AUTO_INCREMENT PRIMARY KEY,
            trade_id            INT NOT NULL,
            h4_trend_score      INT,
            pullback_ema_type   VARCHAR(20),
            pullback_dist_pips  DOUBLE,
            h1_ema_aligned      INT,
            h1_supertrend_dir   INT,
            of_imbalance        DOUBLE,
            of_strength         VARCHAR(20),
            delta_confirms      INT,
            rejection_type      VARCHAR(20),
            velocity_pips       DOUBLE,
            velocity_dir        VARCHAR(10),
            is_scalpable        INT,
            market_state        VARCHAR(30),
            is_choppy           INT,
            atr_pips            DOUBLE,
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (trade_id) REFERENCES backtest_trades(id)
                ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    # ── FVG Reversion-specific features table (1:1 with backtest_trades) ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS backtest_fvg_features (
            id                  INT AUTO_INCREMENT PRIMARY KEY,
            trade_id            INT NOT NULL,
            fvg_type            VARCHAR(15),
            fvg_quality_score   INT,
            fvg_gap_pips        DOUBLE,
            fvg_distance_pips   DOUBLE,
            of_imbalance        DOUBLE,
            of_strength         VARCHAR(20),
            vol_surge           INT,
            vol_surge_ratio     DOUBLE,
            stoch_rsi_k         DOUBLE,
            stoch_rsi_turning   INT,
            m5_wick_rejection   INT,
            ob_fvg_confluence   INT,
            ob_fvg_distance     DOUBLE,
            pd_zone             VARCHAR(20),
            atr_pips            DOUBLE,
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (trade_id) REFERENCES backtest_trades(id)
                ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    # ── EMA Cross Momentum-specific features table (1:1 with backtest_trades) ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS backtest_ema_cross_features (
            id                  INT AUTO_INCREMENT PRIMARY KEY,
            trade_id            INT NOT NULL,
            h4_cross_bars_ago   INT,
            h4_cross_strength   INT,
            h4_alignment_score  INT,
            h1_rsi              DOUBLE,
            m15_adx             DOUBLE,
            delta_bias          VARCHAR(15),
            of_imbalance        DOUBLE,
            of_strength         VARCHAR(20),
            h1_supertrend_dir   INT,
            h4_supertrend_dir   INT,
            h4_ema_spread_9_21  DOUBLE,
            is_choppy           INT,
            vol_surge           INT,
            atr_pips            DOUBLE,
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (trade_id) REFERENCES backtest_trades(id)
                ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    # ── RSI Divergence SMC-specific features table (1:1 with backtest_trades) ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS backtest_rsi_div_features (
            id                  INT AUTO_INCREMENT PRIMARY KEY,
            trade_id            INT NOT NULL,
            div_type            VARCHAR(15),
            div_strength        VARCHAR(15),
            rsi_diff            DOUBLE,
            curr_rsi            DOUBLE,
            prev_rsi            DOUBLE,
            price_range_pips    DOUBLE,
            smc_confirmed       INT,
            smc_bias            VARCHAR(15),
            ob_distance_pips    DOUBLE,
            fvg_distance_pips   DOUBLE,
            delta_bias          VARCHAR(15),
            of_imbalance        DOUBLE,
            stoch_rsi_k         DOUBLE,
            pd_zone             VARCHAR(20),
            is_choppy           INT,
            atr_pips            DOUBLE,
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (trade_id) REFERENCES backtest_trades(id)
                ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    # ── Structure Alignment-specific features table (1:1 with backtest_trades) ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS backtest_structure_features (
            id                  INT AUTO_INCREMENT PRIMARY KEY,
            trade_id            INT NOT NULL,
            bos_direction       VARCHAR(15),
            bos_count           INT,
            h1_ema_aligned      INT,
            h1_full_ema_aligned INT,
            h1_supertrend_dir   INT,
            delta_value         DOUBLE,
            delta_bias          VARCHAR(15),
            of_imbalance        DOUBLE,
            of_strength         VARCHAR(20),
            has_opposing_fvg    INT,
            pd_zone             VARCHAR(20),
            h4_trend_aligned    INT,
            vol_surge           INT,
            is_choppy           INT,
            atr_pips            DOUBLE,
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (trade_id) REFERENCES backtest_trades(id)
                ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    # ── Supply/Demand Zone-specific features table (1:1 with backtest_trades) ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS backtest_sd_zone_features (
            id                  INT AUTO_INCREMENT PRIMARY KEY,
            trade_id            INT NOT NULL,
            zone_type           VARCHAR(20),
            zone_range_pips     DOUBLE,
            price_at_zone       INT,
            displacement_pips   DOUBLE,
            age_bars            INT,
            trend               VARCHAR(15),
            delta_bias          VARCHAR(15),
            delta_strength      VARCHAR(15),
            of_imbalance        DOUBLE,
            of_strength         VARCHAR(20),
            stoch_rsi_k         DOUBLE,
            supertrend_dir_h1   INT,
            supertrend_dir_h4   INT,
            htf_ok              INT,
            smc_bias            VARCHAR(15),
            pd_zone             VARCHAR(20),
            vol_surge           INT,
            has_bos             INT,
            atr_pips            DOUBLE,
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (trade_id) REFERENCES backtest_trades(id)
                ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    # ── Break of Structure Momentum-specific features table (1:1 with backtest_trades) ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS backtest_bos_momentum_features (
            id                  INT AUTO_INCREMENT PRIMARY KEY,
            trade_id            INT NOT NULL,
            bos_type            VARCHAR(20),
            broken_level        DOUBLE,
            displacement_pips   DOUBLE,
            bars_since_bos      INT,
            pullback_depth_pips DOUBLE,
            rejection_strength  VARCHAR(20),
            trend               VARCHAR(15),
            delta_bias          VARCHAR(15),
            delta_strength      VARCHAR(15),
            of_imbalance        DOUBLE,
            of_strength         VARCHAR(20),
            stoch_rsi_k         DOUBLE,
            supertrend_dir_h1   INT,
            supertrend_dir_h4   INT,
            htf_ok              INT,
            smc_bias            VARCHAR(15),
            pd_zone             VARCHAR(20),
            vol_surge           INT,
            has_smc_bos         INT,
            atr_pips            DOUBLE,
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (trade_id) REFERENCES backtest_trades(id)
                ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    # ── Optimal Trade Entry (Fibonacci)-specific features table (1:1 with backtest_trades) ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS backtest_ote_fib_features (
            id                  INT AUTO_INCREMENT PRIMARY KEY,
            trade_id            INT NOT NULL,
            fib_direction       VARCHAR(10),
            ote_zone_low        DOUBLE,
            ote_zone_high       DOUBLE,
            in_gz               INT,
            fib_bias            VARCHAR(15),
            fib_confluence_score INT,
            delta_bias          VARCHAR(15),
            delta_strength      VARCHAR(15),
            supertrend_dir_h1   INT,
            supertrend_dir_h4   INT,
            stoch_rsi_k         DOUBLE,
            of_imbalance        DOUBLE,
            of_strength         VARCHAR(20),
            of_direction        VARCHAR(15),
            vol_surge           INT,
            htf_ok              INT,
            smc_bias            VARCHAR(15),
            pd_zone             VARCHAR(20),
            has_bull_bos        INT,
            has_bear_bos        INT,
            displacement_pips   DOUBLE,
            has_displacement    INT,
            swing_high          DOUBLE,
            swing_low           DOUBLE,
            atr_pips            DOUBLE,
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (trade_id) REFERENCES backtest_trades(id)
                ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    # ── Institutional Candles-specific features table (1:1 with backtest_trades) ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS backtest_inst_candles_features (
            id                  INT AUTO_INCREMENT PRIMARY KEY,
            trade_id            INT NOT NULL,
            pattern_type        VARCHAR(30),
            direction           VARCHAR(10),
            body_pips           DOUBLE,
            wick_ratio          DOUBLE,
            quality             VARCHAR(15),
            context_types       VARCHAR(200),
            context_count       INT,
            context_score       INT,
            delta_bias          VARCHAR(15),
            delta_strength      VARCHAR(15),
            spread_pips         DOUBLE,
            of_imbalance        DOUBLE,
            of_strength         VARCHAR(20),
            stoch_rsi_k         DOUBLE,
            supertrend_dir_h1   INT,
            supertrend_dir_h4   INT,
            htf_ok              INT,
            smc_bias            VARCHAR(15),
            pd_zone             VARCHAR(20),
            vol_surge           INT,
            has_bos             INT,
            fib_bonus           INT,
            atr_pips            DOUBLE,
            sl_pips             DOUBLE,
            trend               VARCHAR(15),
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (trade_id) REFERENCES backtest_trades(id)
                ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
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
        ('backtest_trades', 'ss_supply_demand',     'DOUBLE DEFAULT 0'),
        ('backtest_trades', 'ss_bos_momentum',      'DOUBLE DEFAULT 0'),
        ('backtest_trades', 'ss_optimal_trade',     'DOUBLE DEFAULT 0'),
        ('backtest_trades', 'ss_institutional',     'DOUBLE DEFAULT 0'),
        ('backtest_trades', 'fib_confluence_score', 'DOUBLE DEFAULT 0'),
        ('backtest_trades', 'fib_in_golden_zone',   'TINYINT DEFAULT 0'),
        ('backtest_trades', 'fib_bias_aligned',     'TINYINT DEFAULT 0'),
        # ── Shadow trade system ──
        ('backtest_trades', 'source',               "VARCHAR(20) DEFAULT 'BACKTEST'"),
        # ── ML Gate self-calibration ──
        ('backtest_trades', 'model_predicted_r',    'DOUBLE DEFAULT NULL'),
        # ── Layer 1 Strategy Model ──
        ('backtest_trades', 'strategy_model_verdict',      "VARCHAR(10) DEFAULT NULL"),
        ('backtest_trades', 'strategy_model_predicted_r',  'DOUBLE DEFAULT NULL'),
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


def store_vwap_features(cursor, trade_id: int, vwap_features: dict):
    """Store VWAP-specific strategy features for a trade.

    Args:
        cursor: Database cursor
        trade_id: The backtest_trades.id this row links to
        vwap_features: dict from signal._vwap_features
    """
    if not vwap_features or not trade_id:
        return
    try:
        cursor.execute("""
            INSERT INTO backtest_vwap_features
                (trade_id, atr_pips, adx, vix, fg_score, vwap_pos, va_pos,
                 pd_zone, htf_ok, master_bias, stoch_k, stoch_d,
                 poc_dist, supertrend_dir, poc_above, val_below)
            VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            trade_id,
            vwap_features.get('atr_pips'),
            vwap_features.get('adx'),
            vwap_features.get('vix'),
            vwap_features.get('fg_score'),
            vwap_features.get('vwap_pos'),
            vwap_features.get('va_pos'),
            vwap_features.get('pd_zone'),
            1 if vwap_features.get('htf_ok') else 0,
            vwap_features.get('master_bias'),
            vwap_features.get('stoch_k'),
            vwap_features.get('stoch_d'),
            vwap_features.get('poc_dist'),
            vwap_features.get('supertrend_dir'),
            1 if vwap_features.get('poc_above') else 0,
            1 if vwap_features.get('val_below') else 0,
        ))
        log.info(f"[DB_STORE] Stored VWAP features for trade {trade_id}: "
                 f"atr={vwap_features.get('atr_pips',0):.1f} "
                 f"adx={vwap_features.get('adx',0):.1f} "
                 f"vwap_pos={vwap_features.get('vwap_pos','')} "
                 f"htf_ok={vwap_features.get('htf_ok')}")
    except Exception as e:
        log.warning(f"[DB_STORE] Failed to store VWAP features for trade {trade_id}: {e}")


def store_breakout_features(cursor, trade_id: int, breakout_features: dict):
    """Store Breakout-specific strategy features for a trade.

    Args:
        cursor: Database cursor
        trade_id: The backtest_trades.id this row links to
        breakout_features: dict from signal._breakout_features
    """
    if not breakout_features or not trade_id:
        return
    try:
        cursor.execute("""
            INSERT INTO backtest_breakout_features
                (trade_id, consol_type, range_pips, adx, atr_pips, atr_ratio,
                 retest, dist_to_level, delta_confirms, of_imbalance, of_strength,
                 vol_surge, vol_surge_ratio, h4_trend_aligned, h4_supertrend,
                 m5_momentum, bos_aligned, is_choppy)
            VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            trade_id,
            breakout_features.get('consol_type'),
            breakout_features.get('range_pips'),
            breakout_features.get('adx'),
            breakout_features.get('atr_pips'),
            breakout_features.get('atr_ratio'),
            breakout_features.get('retest'),
            breakout_features.get('dist_to_level'),
            breakout_features.get('delta_confirms'),
            breakout_features.get('of_imbalance'),
            breakout_features.get('of_strength'),
            breakout_features.get('vol_surge'),
            breakout_features.get('vol_surge_ratio'),
            breakout_features.get('h4_trend_aligned'),
            breakout_features.get('h4_supertrend'),
            breakout_features.get('m5_momentum'),
            breakout_features.get('bos_aligned'),
            breakout_features.get('is_choppy'),
        ))
        log.info(f"[DB_STORE] Stored Breakout features for trade {trade_id}: "
                 f"consol={breakout_features.get('consol_type','')} "
                 f"range={breakout_features.get('range_pips',0):.1f}p "
                 f"retest={breakout_features.get('retest')}")
    except Exception as e:
        log.warning(f"[DB_STORE] Failed to store Breakout features for trade {trade_id}: {e}")


def store_smc_ob_features(cursor, trade_id: int, smc_ob_features: dict):
    """Store SMC OB Reversal-specific strategy features for a trade."""
    if not smc_ob_features or not trade_id:
        return
    try:
        cursor.execute("""
            INSERT INTO backtest_smc_ob_features
                (trade_id, ob_type, ob_dist_pips, price_at_ob, trend,
                 delta_bias, delta_strength, of_imbalance, of_strength,
                 stoch_rsi_k, supertrend_dir_h1, htf_ok, smc_bias,
                 pd_zone, atr_pips, has_bos)
            VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            trade_id,
            smc_ob_features.get('ob_type'),
            smc_ob_features.get('ob_dist_pips'),
            smc_ob_features.get('price_at_ob'),
            smc_ob_features.get('trend'),
            smc_ob_features.get('delta_bias'),
            smc_ob_features.get('delta_strength'),
            smc_ob_features.get('of_imbalance'),
            smc_ob_features.get('of_strength'),
            smc_ob_features.get('stoch_rsi_k'),
            smc_ob_features.get('supertrend_dir_h1'),
            smc_ob_features.get('htf_ok'),
            smc_ob_features.get('smc_bias'),
            smc_ob_features.get('pd_zone'),
            smc_ob_features.get('atr_pips'),
            smc_ob_features.get('has_bos'),
        ))
        log.info(f"[DB_STORE] Stored SMC OB features for trade {trade_id}: "
                 f"ob_type={smc_ob_features.get('ob_type','')} "
                 f"trend={smc_ob_features.get('trend','')}")
    except Exception as e:
        log.warning(f"[DB_STORE] Failed to store SMC OB features for trade {trade_id}: {e}")


def store_liq_sweep_features(cursor, trade_id: int, liq_sweep_features: dict):
    """Store Liquidity Sweep-specific strategy features for a trade."""
    if not liq_sweep_features or not trade_id:
        return
    try:
        cursor.execute("""
            INSERT INTO backtest_liq_sweep_features
                (trade_id, sweep_bias, reversal_pips, swept_level_dist,
                 delta_bias, delta_strength, has_bos, bos_type,
                 stoch_rsi_k, supertrend_dir_h1, htf_ok, smc_bias,
                 pd_zone, vol_surge, of_imbalance, atr_pips)
            VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            trade_id,
            liq_sweep_features.get('sweep_bias'),
            liq_sweep_features.get('reversal_pips'),
            liq_sweep_features.get('swept_level_dist'),
            liq_sweep_features.get('delta_bias'),
            liq_sweep_features.get('delta_strength'),
            liq_sweep_features.get('has_bos'),
            liq_sweep_features.get('bos_type'),
            liq_sweep_features.get('stoch_rsi_k'),
            liq_sweep_features.get('supertrend_dir_h1'),
            liq_sweep_features.get('htf_ok'),
            liq_sweep_features.get('smc_bias'),
            liq_sweep_features.get('pd_zone'),
            liq_sweep_features.get('vol_surge'),
            liq_sweep_features.get('of_imbalance'),
            liq_sweep_features.get('atr_pips'),
        ))
        log.info(f"[DB_STORE] Stored Liquidity Sweep features for trade {trade_id}: "
                 f"sweep_bias={liq_sweep_features.get('sweep_bias','')}")
    except Exception as e:
        log.warning(f"[DB_STORE] Failed to store Liquidity Sweep features for trade {trade_id}: {e}")


def store_delta_div_features(cursor, trade_id: int, delta_div_features: dict):
    """Store Delta Divergence-specific strategy features for a trade."""
    if not delta_div_features or not trade_id:
        return
    try:
        cursor.execute("""
            INSERT INTO backtest_delta_div_features
                (trade_id, div_type, div_strength, swing_range_pips,
                 delta_value, delta_bias, of_imbalance, of_strength,
                 vol_surge, surge_ratio, surge_absorption,
                 stoch_rsi_k, stoch_rsi_turning, pd_zone,
                 m5_body_ratio, atr_pips)
            VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            trade_id,
            delta_div_features.get('div_type'),
            delta_div_features.get('div_strength'),
            delta_div_features.get('swing_range_pips'),
            delta_div_features.get('delta_value'),
            delta_div_features.get('delta_bias'),
            delta_div_features.get('of_imbalance'),
            delta_div_features.get('of_strength'),
            delta_div_features.get('vol_surge'),
            delta_div_features.get('surge_ratio'),
            delta_div_features.get('surge_absorption'),
            delta_div_features.get('stoch_rsi_k'),
            delta_div_features.get('stoch_rsi_turning'),
            delta_div_features.get('pd_zone'),
            delta_div_features.get('m5_body_ratio'),
            delta_div_features.get('atr_pips'),
        ))
        log.info(f"[DB_STORE] Stored Delta Divergence features for trade {trade_id}: "
                 f"div_type={delta_div_features.get('div_type','')}")
    except Exception as e:
        log.warning(f"[DB_STORE] Failed to store Delta Divergence features for trade {trade_id}: {e}")


def store_trend_cont_features(cursor, trade_id: int, trend_cont_features: dict):
    """Store Trend Continuation-specific strategy features for a trade."""
    if not trend_cont_features or not trade_id:
        return
    try:
        cursor.execute("""
            INSERT INTO backtest_trend_cont_features
                (trade_id, h4_trend_score, pullback_ema_type, pullback_dist_pips,
                 h1_ema_aligned, h1_supertrend_dir, of_imbalance, of_strength,
                 delta_confirms, rejection_type, velocity_pips, velocity_dir,
                 is_scalpable, market_state, is_choppy, atr_pips)
            VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            trade_id,
            trend_cont_features.get('h4_trend_score'),
            trend_cont_features.get('pullback_ema_type'),
            trend_cont_features.get('pullback_dist_pips'),
            trend_cont_features.get('h1_ema_aligned'),
            trend_cont_features.get('h1_supertrend_dir'),
            trend_cont_features.get('of_imbalance'),
            trend_cont_features.get('of_strength'),
            trend_cont_features.get('delta_confirms'),
            trend_cont_features.get('rejection_type'),
            trend_cont_features.get('velocity_pips'),
            trend_cont_features.get('velocity_dir'),
            trend_cont_features.get('is_scalpable'),
            trend_cont_features.get('market_state'),
            trend_cont_features.get('is_choppy'),
            trend_cont_features.get('atr_pips'),
        ))
        log.info(f"[DB_STORE] Stored Trend Continuation features for trade {trade_id}: "
                 f"h4_score={trend_cont_features.get('h4_trend_score',0)}")
    except Exception as e:
        log.warning(f"[DB_STORE] Failed to store Trend Continuation features for trade {trade_id}: {e}")


def store_fvg_features(cursor, trade_id: int, fvg_features: dict):
    """Store FVG Reversion-specific strategy features for a trade."""
    if not fvg_features or not trade_id:
        return
    try:
        cursor.execute("""
            INSERT INTO backtest_fvg_features
                (trade_id, fvg_type, fvg_quality_score, fvg_gap_pips,
                 fvg_distance_pips, of_imbalance, of_strength, vol_surge,
                 vol_surge_ratio, stoch_rsi_k, stoch_rsi_turning,
                 m5_wick_rejection, ob_fvg_confluence, ob_fvg_distance,
                 pd_zone, atr_pips)
            VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            trade_id,
            fvg_features.get('fvg_type'),
            fvg_features.get('fvg_quality_score'),
            fvg_features.get('fvg_gap_pips'),
            fvg_features.get('fvg_distance_pips'),
            fvg_features.get('of_imbalance'),
            fvg_features.get('of_strength'),
            fvg_features.get('vol_surge'),
            fvg_features.get('vol_surge_ratio'),
            fvg_features.get('stoch_rsi_k'),
            fvg_features.get('stoch_rsi_turning'),
            fvg_features.get('m5_wick_rejection'),
            fvg_features.get('ob_fvg_confluence'),
            fvg_features.get('ob_fvg_distance'),
            fvg_features.get('pd_zone'),
            fvg_features.get('atr_pips'),
        ))
        log.info(f"[DB_STORE] Stored FVG features for trade {trade_id}: "
                 f"fvg_type={fvg_features.get('fvg_type','')}")
    except Exception as e:
        log.warning(f"[DB_STORE] Failed to store FVG features for trade {trade_id}: {e}")


def store_ema_cross_features(cursor, trade_id: int, ema_cross_features: dict):
    """Store EMA Cross Momentum-specific strategy features for a trade."""
    if not ema_cross_features or not trade_id:
        return
    try:
        cursor.execute("""
            INSERT INTO backtest_ema_cross_features
                (trade_id, h4_cross_bars_ago, h4_cross_strength, h4_alignment_score,
                 h1_rsi, m15_adx, delta_bias, of_imbalance, of_strength,
                 h1_supertrend_dir, h4_supertrend_dir, h4_ema_spread_9_21,
                 is_choppy, vol_surge, atr_pips)
            VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            trade_id,
            ema_cross_features.get('h4_cross_bars_ago'),
            ema_cross_features.get('h4_cross_strength'),
            ema_cross_features.get('h4_alignment_score'),
            ema_cross_features.get('h1_rsi'),
            ema_cross_features.get('m15_adx'),
            ema_cross_features.get('delta_bias'),
            ema_cross_features.get('of_imbalance'),
            ema_cross_features.get('of_strength'),
            ema_cross_features.get('h1_supertrend_dir'),
            ema_cross_features.get('h4_supertrend_dir'),
            ema_cross_features.get('h4_ema_spread_9_21'),
            ema_cross_features.get('is_choppy'),
            ema_cross_features.get('vol_surge'),
            ema_cross_features.get('atr_pips'),
        ))
        log.info(f"[DB_STORE] Stored EMA Cross features for trade {trade_id}: "
                 f"h4_cross_bars_ago={ema_cross_features.get('h4_cross_bars_ago',0)}")
    except Exception as e:
        log.warning(f"[DB_STORE] Failed to store EMA Cross features for trade {trade_id}: {e}")


def store_rsi_div_features(cursor, trade_id: int, rsi_div_features: dict):
    """Store RSI Divergence SMC-specific strategy features for a trade."""
    if not rsi_div_features or not trade_id:
        return
    try:
        cursor.execute("""
            INSERT INTO backtest_rsi_div_features
                (trade_id, div_type, div_strength, rsi_diff, curr_rsi,
                 prev_rsi, price_range_pips, smc_confirmed, smc_bias,
                 ob_distance_pips, fvg_distance_pips, delta_bias,
                 of_imbalance, stoch_rsi_k, pd_zone, is_choppy, atr_pips)
            VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            trade_id,
            rsi_div_features.get('div_type'),
            rsi_div_features.get('div_strength'),
            rsi_div_features.get('rsi_diff'),
            rsi_div_features.get('curr_rsi'),
            rsi_div_features.get('prev_rsi'),
            rsi_div_features.get('price_range_pips'),
            rsi_div_features.get('smc_confirmed'),
            rsi_div_features.get('smc_bias'),
            rsi_div_features.get('ob_distance_pips'),
            rsi_div_features.get('fvg_distance_pips'),
            rsi_div_features.get('delta_bias'),
            rsi_div_features.get('of_imbalance'),
            rsi_div_features.get('stoch_rsi_k'),
            rsi_div_features.get('pd_zone'),
            rsi_div_features.get('is_choppy'),
            rsi_div_features.get('atr_pips'),
        ))
        log.info(f"[DB_STORE] Stored RSI Divergence features for trade {trade_id}: "
                 f"div_type={rsi_div_features.get('div_type','')}")
    except Exception as e:
        log.warning(f"[DB_STORE] Failed to store RSI Divergence features for trade {trade_id}: {e}")


def store_structure_features(cursor, trade_id: int, structure_features: dict):
    """Store Structure Alignment-specific strategy features for a trade."""
    if not structure_features or not trade_id:
        return
    try:
        cursor.execute("""
            INSERT INTO backtest_structure_features
                (trade_id, bos_direction, bos_count, h1_ema_aligned,
                 h1_full_ema_aligned, h1_supertrend_dir, delta_value,
                 delta_bias, of_imbalance, of_strength, has_opposing_fvg,
                 pd_zone, h4_trend_aligned, vol_surge, is_choppy, atr_pips)
            VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            trade_id,
            structure_features.get('bos_direction'),
            structure_features.get('bos_count'),
            structure_features.get('h1_ema_aligned'),
            structure_features.get('h1_full_ema_aligned'),
            structure_features.get('h1_supertrend_dir'),
            structure_features.get('delta_value'),
            structure_features.get('delta_bias'),
            structure_features.get('of_imbalance'),
            structure_features.get('of_strength'),
            structure_features.get('has_opposing_fvg'),
            structure_features.get('pd_zone'),
            structure_features.get('h4_trend_aligned'),
            structure_features.get('vol_surge'),
            structure_features.get('is_choppy'),
            structure_features.get('atr_pips'),
        ))
        log.info(f"[DB_STORE] Stored Structure Alignment features for trade {trade_id}: "
                 f"bos_dir={structure_features.get('bos_direction','')}")
    except Exception as e:
        log.warning(f"[DB_STORE] Failed to store Structure Alignment features for trade {trade_id}: {e}")


def store_sd_zone_features(cursor, trade_id: int, sd_zone_features: dict):
    """Store Supply/Demand Zone-specific strategy features for a trade."""
    if not sd_zone_features or not trade_id:
        return
    try:
        cursor.execute("""
            INSERT INTO backtest_sd_zone_features
                (trade_id, zone_type, zone_range_pips, price_at_zone,
                 displacement_pips, age_bars, trend,
                 delta_bias, delta_strength, of_imbalance, of_strength,
                 stoch_rsi_k, supertrend_dir_h1, supertrend_dir_h4,
                 htf_ok, smc_bias, pd_zone, vol_surge, has_bos, atr_pips)
            VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            trade_id,
            sd_zone_features.get('zone_type'),
            sd_zone_features.get('zone_range_pips'),
            sd_zone_features.get('price_at_zone'),
            sd_zone_features.get('displacement_pips'),
            sd_zone_features.get('age_bars'),
            sd_zone_features.get('trend'),
            sd_zone_features.get('delta_bias'),
            sd_zone_features.get('delta_strength'),
            sd_zone_features.get('of_imbalance'),
            sd_zone_features.get('of_strength'),
            sd_zone_features.get('stoch_rsi_k'),
            sd_zone_features.get('supertrend_dir_h1'),
            sd_zone_features.get('supertrend_dir_h4'),
            sd_zone_features.get('htf_ok'),
            sd_zone_features.get('smc_bias'),
            sd_zone_features.get('pd_zone'),
            sd_zone_features.get('vol_surge'),
            sd_zone_features.get('has_bos'),
            sd_zone_features.get('atr_pips'),
        ))
        log.info(f"[DB_STORE] Stored S&D Zone features for trade {trade_id}: "
                 f"zone_type={sd_zone_features.get('zone_type','')}")
    except Exception as e:
        log.warning(f"[DB_STORE] Failed to store S&D Zone features for trade {trade_id}: {e}")


def store_bos_momentum_features(cursor, trade_id: int, bos_momentum_features: dict):
    """Store Break of Structure Momentum-specific strategy features for a trade."""
    if not bos_momentum_features or not trade_id:
        return
    try:
        cursor.execute("""
            INSERT INTO backtest_bos_momentum_features
                (trade_id, bos_type, broken_level, displacement_pips,
                 bars_since_bos, pullback_depth_pips, rejection_strength,
                 trend, delta_bias, delta_strength, of_imbalance, of_strength,
                 stoch_rsi_k, supertrend_dir_h1, supertrend_dir_h4,
                 htf_ok, smc_bias, pd_zone, vol_surge, has_smc_bos, atr_pips)
            VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            trade_id,
            bos_momentum_features.get('bos_type'),
            bos_momentum_features.get('broken_level'),
            bos_momentum_features.get('displacement_pips'),
            bos_momentum_features.get('bars_since_bos'),
            bos_momentum_features.get('pullback_depth_pips'),
            bos_momentum_features.get('rejection_strength'),
            bos_momentum_features.get('trend'),
            bos_momentum_features.get('delta_bias'),
            bos_momentum_features.get('delta_strength'),
            bos_momentum_features.get('of_imbalance'),
            bos_momentum_features.get('of_strength'),
            bos_momentum_features.get('stoch_rsi_k'),
            bos_momentum_features.get('supertrend_dir_h1'),
            bos_momentum_features.get('supertrend_dir_h4'),
            bos_momentum_features.get('htf_ok'),
            bos_momentum_features.get('smc_bias'),
            bos_momentum_features.get('pd_zone'),
            bos_momentum_features.get('vol_surge'),
            bos_momentum_features.get('has_smc_bos'),
            bos_momentum_features.get('atr_pips'),
        ))
        log.info(f"[DB_STORE] Stored BOS Momentum features for trade {trade_id}: "
                 f"bos_type={bos_momentum_features.get('bos_type','')}")
    except Exception as e:
        log.warning(f"[DB_STORE] Failed to store BOS Momentum features for trade {trade_id}: {e}")


def store_ote_fib_features(cursor, trade_id: int, ote_fib_features: dict):
    """Store Optimal Trade Entry (Fibonacci)-specific strategy features for a trade."""
    if not ote_fib_features or not trade_id:
        return
    try:
        cursor.execute("""
            INSERT INTO backtest_ote_fib_features
                (trade_id, fib_direction, ote_zone_low, ote_zone_high,
                 in_gz, fib_bias, fib_confluence_score,
                 delta_bias, delta_strength, supertrend_dir_h1, supertrend_dir_h4,
                 stoch_rsi_k, of_imbalance, of_strength, of_direction,
                 vol_surge, htf_ok, smc_bias, pd_zone,
                 has_bull_bos, has_bear_bos, displacement_pips, has_displacement,
                 swing_high, swing_low, atr_pips)
            VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            trade_id,
            ote_fib_features.get('fib_direction'),
            ote_fib_features.get('ote_zone_low'),
            ote_fib_features.get('ote_zone_high'),
            ote_fib_features.get('in_gz'),
            ote_fib_features.get('fib_bias'),
            ote_fib_features.get('fib_confluence_score'),
            ote_fib_features.get('delta_bias'),
            ote_fib_features.get('delta_strength'),
            ote_fib_features.get('supertrend_dir_h1'),
            ote_fib_features.get('supertrend_dir_h4'),
            ote_fib_features.get('stoch_rsi_k'),
            ote_fib_features.get('of_imbalance'),
            ote_fib_features.get('of_strength'),
            ote_fib_features.get('of_direction'),
            ote_fib_features.get('vol_surge'),
            ote_fib_features.get('htf_ok'),
            ote_fib_features.get('smc_bias'),
            ote_fib_features.get('pd_zone'),
            ote_fib_features.get('has_bull_bos'),
            ote_fib_features.get('has_bear_bos'),
            ote_fib_features.get('displacement_pips'),
            ote_fib_features.get('has_displacement'),
            ote_fib_features.get('swing_high'),
            ote_fib_features.get('swing_low'),
            ote_fib_features.get('atr_pips'),
        ))
        log.info(f"[DB_STORE] Stored OTE Fib features for trade {trade_id}: "
                 f"fib_dir={ote_fib_features.get('fib_direction','')}")
    except Exception as e:
        log.warning(f"[DB_STORE] Failed to store OTE Fib features for trade {trade_id}: {e}")


def store_inst_candles_features(cursor, trade_id: int, inst_candles_features: dict):
    """Store Institutional Candles-specific strategy features for a trade."""
    if not inst_candles_features or not trade_id:
        return
    try:
        cursor.execute("""
            INSERT INTO backtest_inst_candles_features
                (trade_id, pattern_type, direction, body_pips, wick_ratio,
                 quality, context_types, context_count, context_score,
                 delta_bias, delta_strength, spread_pips,
                 of_imbalance, of_strength, stoch_rsi_k,
                 supertrend_dir_h1, supertrend_dir_h4,
                 htf_ok, smc_bias, pd_zone, vol_surge,
                 has_bos, fib_bonus, atr_pips, sl_pips, trend)
            VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            trade_id,
            inst_candles_features.get('pattern_type'),
            inst_candles_features.get('direction'),
            inst_candles_features.get('body_pips'),
            inst_candles_features.get('wick_ratio'),
            inst_candles_features.get('quality'),
            inst_candles_features.get('context_types'),
            inst_candles_features.get('context_count'),
            inst_candles_features.get('context_score'),
            inst_candles_features.get('delta_bias'),
            inst_candles_features.get('delta_strength'),
            inst_candles_features.get('spread_pips'),
            inst_candles_features.get('of_imbalance'),
            inst_candles_features.get('of_strength'),
            inst_candles_features.get('stoch_rsi_k'),
            inst_candles_features.get('supertrend_dir_h1'),
            inst_candles_features.get('supertrend_dir_h4'),
            inst_candles_features.get('htf_ok'),
            inst_candles_features.get('smc_bias'),
            inst_candles_features.get('pd_zone'),
            inst_candles_features.get('vol_surge'),
            inst_candles_features.get('has_bos'),
            inst_candles_features.get('fib_bonus'),
            inst_candles_features.get('atr_pips'),
            inst_candles_features.get('sl_pips'),
            inst_candles_features.get('trend'),
        ))
        log.info(f"[DB_STORE] Stored Institutional Candles features for trade {trade_id}: "
                 f"pattern={inst_candles_features.get('pattern_type','')}")
    except Exception as e:
        log.warning(f"[DB_STORE] Failed to store Institutional Candles features for trade {trade_id}: {e}")


def store_trade(trade, master_report: dict = None,
                market_report: dict = None, smc_report: dict = None,
                flow_data: dict = None, run_id: str = 'default',
                spread_pips: float = 0.0, slippage_pips: float = 0.0,
                strategy_scores: dict = None, source: str = 'BACKTEST',
                model_predicted_r: float = None,
                strategy_model_verdict: str = None,
                strategy_model_predicted_r: float = None,
                vwap_features: dict = None,
                breakout_features: dict = None,
                smc_ob_features: dict = None,
                liq_sweep_features: dict = None,
                delta_div_features: dict = None,
                trend_cont_features: dict = None,
                fvg_features: dict = None,
                ema_cross_features: dict = None,
                rsi_div_features: dict = None,
                structure_features: dict = None,
                sd_zone_features: dict = None,
                bos_momentum_features: dict = None,
                ote_fib_features: dict = None,
                inst_candles_features: dict = None):
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
        ss_supply_demand    = _safe_float(ss.get('SUPPLY_DEMAND_ZONE_ENTRY', 0))
        ss_bos_momentum     = _safe_float(ss.get('BREAK_OF_STRUCTURE_MOMENTUM', 0))
        ss_optimal_trade    = _safe_float(ss.get('OPTIMAL_TRADE_ENTRY_FIB', 0))
        ss_institutional    = _safe_float(ss.get('INSTITUTIONAL_CANDLES', 0))

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
        dup_row = c.fetchone()
        if dup_row:
            existing_id = dup_row['id']
            # Even if trade exists, backfill strategy-specific features if missing
            if vwap_features and trade.strategy == 'VWAP_MEAN_REVERSION':
                try:
                    c.execute("""
                        SELECT id FROM backtest_vwap_features
                        WHERE trade_id = %s LIMIT 1
                    """, (existing_id,))
                    if not c.fetchone():
                        store_vwap_features(c, existing_id, vwap_features)
                        conn.commit()
                        log.info(f"[DB_STORE] Backfilled VWAP features for existing trade {existing_id}")
                except Exception as e:
                    log.warning(f"[DB_STORE] VWAP backfill error: {e}")
            if breakout_features and trade.strategy == 'BREAKOUT_MOMENTUM':
                try:
                    c.execute("""
                        SELECT id FROM backtest_breakout_features
                        WHERE trade_id = %s LIMIT 1
                    """, (existing_id,))
                    if not c.fetchone():
                        store_breakout_features(c, existing_id, breakout_features)
                        conn.commit()
                        log.info(f"[DB_STORE] Backfilled breakout features for existing trade {existing_id}")
                except Exception as e:
                    log.warning(f"[DB_STORE] Breakout backfill error: {e}")
            if smc_ob_features and trade.strategy == 'SMC_OB_REVERSAL':
                try:
                    c.execute("SELECT id FROM backtest_smc_ob_features WHERE trade_id = %s LIMIT 1", (existing_id,))
                    if not c.fetchone():
                        store_smc_ob_features(c, existing_id, smc_ob_features)
                        conn.commit()
                        log.info(f"[DB_STORE] Backfilled SMC OB features for existing trade {existing_id}")
                except Exception as e:
                    log.warning(f"[DB_STORE] SMC OB backfill error: {e}")
            if liq_sweep_features and trade.strategy == 'LIQUIDITY_SWEEP_ENTRY':
                try:
                    c.execute("SELECT id FROM backtest_liq_sweep_features WHERE trade_id = %s LIMIT 1", (existing_id,))
                    if not c.fetchone():
                        store_liq_sweep_features(c, existing_id, liq_sweep_features)
                        conn.commit()
                        log.info(f"[DB_STORE] Backfilled Liquidity Sweep features for existing trade {existing_id}")
                except Exception as e:
                    log.warning(f"[DB_STORE] Liquidity Sweep backfill error: {e}")
            if delta_div_features and trade.strategy == 'DELTA_DIVERGENCE':
                try:
                    c.execute("SELECT id FROM backtest_delta_div_features WHERE trade_id = %s LIMIT 1", (existing_id,))
                    if not c.fetchone():
                        store_delta_div_features(c, existing_id, delta_div_features)
                        conn.commit()
                        log.info(f"[DB_STORE] Backfilled Delta Divergence features for existing trade {existing_id}")
                except Exception as e:
                    log.warning(f"[DB_STORE] Delta Divergence backfill error: {e}")
            if trend_cont_features and trade.strategy == 'TREND_CONTINUATION':
                try:
                    c.execute("SELECT id FROM backtest_trend_cont_features WHERE trade_id = %s LIMIT 1", (existing_id,))
                    if not c.fetchone():
                        store_trend_cont_features(c, existing_id, trend_cont_features)
                        conn.commit()
                        log.info(f"[DB_STORE] Backfilled Trend Continuation features for existing trade {existing_id}")
                except Exception as e:
                    log.warning(f"[DB_STORE] Trend Continuation backfill error: {e}")
            if fvg_features and trade.strategy == 'FVG_REVERSION':
                try:
                    c.execute("SELECT id FROM backtest_fvg_features WHERE trade_id = %s LIMIT 1", (existing_id,))
                    if not c.fetchone():
                        store_fvg_features(c, existing_id, fvg_features)
                        conn.commit()
                        log.info(f"[DB_STORE] Backfilled FVG features for existing trade {existing_id}")
                except Exception as e:
                    log.warning(f"[DB_STORE] FVG backfill error: {e}")
            if ema_cross_features and trade.strategy == 'EMA_CROSS_MOMENTUM':
                try:
                    c.execute("SELECT id FROM backtest_ema_cross_features WHERE trade_id = %s LIMIT 1", (existing_id,))
                    if not c.fetchone():
                        store_ema_cross_features(c, existing_id, ema_cross_features)
                        conn.commit()
                        log.info(f"[DB_STORE] Backfilled EMA Cross features for existing trade {existing_id}")
                except Exception as e:
                    log.warning(f"[DB_STORE] EMA Cross backfill error: {e}")
            if rsi_div_features and trade.strategy == 'RSI_DIVERGENCE_SMC':
                try:
                    c.execute("SELECT id FROM backtest_rsi_div_features WHERE trade_id = %s LIMIT 1", (existing_id,))
                    if not c.fetchone():
                        store_rsi_div_features(c, existing_id, rsi_div_features)
                        conn.commit()
                        log.info(f"[DB_STORE] Backfilled RSI Divergence features for existing trade {existing_id}")
                except Exception as e:
                    log.warning(f"[DB_STORE] RSI Divergence backfill error: {e}")
            if structure_features and trade.strategy == 'STRUCTURE_ALIGNMENT':
                try:
                    c.execute("SELECT id FROM backtest_structure_features WHERE trade_id = %s LIMIT 1", (existing_id,))
                    if not c.fetchone():
                        store_structure_features(c, existing_id, structure_features)
                        conn.commit()
                        log.info(f"[DB_STORE] Backfilled Structure Alignment features for existing trade {existing_id}")
                except Exception as e:
                    log.warning(f"[DB_STORE] Structure Alignment backfill error: {e}")
            if sd_zone_features and trade.strategy == 'SUPPLY_DEMAND_ZONE_ENTRY':
                try:
                    c.execute("SELECT id FROM backtest_sd_zone_features WHERE trade_id = %s LIMIT 1", (existing_id,))
                    if not c.fetchone():
                        store_sd_zone_features(c, existing_id, sd_zone_features)
                        conn.commit()
                        log.info(f"[DB_STORE] Backfilled S&D Zone features for existing trade {existing_id}")
                except Exception as e:
                    log.warning(f"[DB_STORE] S&D Zone backfill error: {e}")
            if bos_momentum_features and trade.strategy == 'BREAK_OF_STRUCTURE_MOMENTUM':
                try:
                    c.execute("SELECT id FROM backtest_bos_momentum_features WHERE trade_id = %s LIMIT 1", (existing_id,))
                    if not c.fetchone():
                        store_bos_momentum_features(c, existing_id, bos_momentum_features)
                        conn.commit()
                        log.info(f"[DB_STORE] Backfilled BOS Momentum features for existing trade {existing_id}")
                except Exception as e:
                    log.warning(f"[DB_STORE] BOS Momentum backfill error: {e}")
            if ote_fib_features and trade.strategy == 'OPTIMAL_TRADE_ENTRY_FIB':
                try:
                    c.execute("SELECT id FROM backtest_ote_fib_features WHERE trade_id = %s LIMIT 1", (existing_id,))
                    if not c.fetchone():
                        store_ote_fib_features(c, existing_id, ote_fib_features)
                        conn.commit()
                        log.info(f"[DB_STORE] Backfilled OTE Fib features for existing trade {existing_id}")
                except Exception as e:
                    log.warning(f"[DB_STORE] OTE Fib backfill error: {e}")
            if inst_candles_features and trade.strategy == 'INSTITUTIONAL_CANDLES':
                try:
                    c.execute("SELECT id FROM backtest_inst_candles_features WHERE trade_id = %s LIMIT 1", (existing_id,))
                    if not c.fetchone():
                        store_inst_candles_features(c, existing_id, inst_candles_features)
                        conn.commit()
                        log.info(f"[DB_STORE] Backfilled Institutional Candles features for existing trade {existing_id}")
                except Exception as e:
                    log.warning(f"[DB_STORE] Institutional Candles backfill error: {e}")
            c.close()
            conn.close()
            log.debug(f"[DB_STORE] Skipping duplicate trade: {trade.symbol} {trade.strategy} {entry_time_str}")
            return

        # 88 columns
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
                ss_supply_demand, ss_bos_momentum,
                ss_optimal_trade, ss_institutional,
                fib_confluence_score, fib_in_golden_zone, fib_bias_aligned,
                source,
                model_predicted_r,
                strategy_model_verdict,
                strategy_model_predicted_r
            ) VALUES (
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s, %s, %s, %s, %s, %s, %s, %s
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
            # ── Strategy scores (features for ML Gate v3.0) ──
            ss_smc_ob, ss_liquidity_sweep, ss_vwap_reversion,
            ss_delta_divergence, ss_trend_continuation,
            ss_fvg_reversion, ss_ema_cross, ss_rsi_divergence,
            ss_breakout_momentum, ss_structure_align,
            ss_supply_demand, ss_bos_momentum,
            ss_optimal_trade, ss_institutional,
            # ── Fibonacci confluence (3 features for ML Gate v3.1) ──
            fib_confluence_score, fib_in_golden_zone, fib_bias_aligned,
            # ── Source: BACKTEST (real) or SHADOW (simulated) ──
            source,
            # ── ML Gate model prediction (self-calibration) ──
            model_predicted_r,
            # ── Layer 1 Strategy Model ──
            strategy_model_verdict,
            strategy_model_predicted_r,
        ))

        conn.commit()
        trade_id = c.lastrowid

        # ── Store strategy-specific features if present ──
        if vwap_features and trade.strategy == 'VWAP_MEAN_REVERSION':
            store_vwap_features(c, trade_id, vwap_features)
            conn.commit()
        if breakout_features and trade.strategy == 'BREAKOUT_MOMENTUM':
            store_breakout_features(c, trade_id, breakout_features)
            conn.commit()
        if smc_ob_features and trade.strategy == 'SMC_OB_REVERSAL':
            store_smc_ob_features(c, trade_id, smc_ob_features)
            conn.commit()
        if liq_sweep_features and trade.strategy == 'LIQUIDITY_SWEEP_ENTRY':
            store_liq_sweep_features(c, trade_id, liq_sweep_features)
            conn.commit()
        if delta_div_features and trade.strategy == 'DELTA_DIVERGENCE':
            store_delta_div_features(c, trade_id, delta_div_features)
            conn.commit()
        if trend_cont_features and trade.strategy == 'TREND_CONTINUATION':
            store_trend_cont_features(c, trade_id, trend_cont_features)
            conn.commit()
        if fvg_features and trade.strategy == 'FVG_REVERSION':
            store_fvg_features(c, trade_id, fvg_features)
            conn.commit()
        if ema_cross_features and trade.strategy == 'EMA_CROSS_MOMENTUM':
            store_ema_cross_features(c, trade_id, ema_cross_features)
            conn.commit()
        if rsi_div_features and trade.strategy == 'RSI_DIVERGENCE_SMC':
            store_rsi_div_features(c, trade_id, rsi_div_features)
            conn.commit()
        if structure_features and trade.strategy == 'STRUCTURE_ALIGNMENT':
            store_structure_features(c, trade_id, structure_features)
            conn.commit()
        if sd_zone_features and trade.strategy == 'SUPPLY_DEMAND_ZONE_ENTRY':
            store_sd_zone_features(c, trade_id, sd_zone_features)
            conn.commit()
        if bos_momentum_features and trade.strategy == 'BREAK_OF_STRUCTURE_MOMENTUM':
            store_bos_momentum_features(c, trade_id, bos_momentum_features)
            conn.commit()
        if ote_fib_features and trade.strategy == 'OPTIMAL_TRADE_ENTRY_FIB':
            store_ote_fib_features(c, trade_id, ote_fib_features)
            conn.commit()
        if inst_candles_features and trade.strategy == 'INSTITUTIONAL_CANDLES':
            store_inst_candles_features(c, trade_id, inst_candles_features)
            conn.commit()

        c.close()
        conn.close()
        return trade_id

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
