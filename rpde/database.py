# =============================================================
# rpde/database.py  v5.0
# RPDE Database Module — Reverse Pattern Discovery Engine
#
# Manages SEPARATE database tables for the RPDE system.
# Does NOT touch existing v4.2 tables (trades, backtest_trades, etc.).
#
# Tables:
#   rpde_scan_history        — Tracks when scans were run
#   rpde_pattern_scans       — Raw golden moment data (big moves found)
#   rpde_pattern_library     — Discovered and validated patterns
#   rpde_pattern_trades      — Trades taken by RPDE in live/paper trading
#   rpde_pattern_stats       — Rolling performance statistics per pattern
#   rpde_tft_training_log    — TFT model training history and metrics
#   rpde_tft_models          — TFT model registry with metadata
#   rpde_rl_training_log     — RL agent training history and metrics (Phase 3)
#   rpde_rl_models           — RL agent model registry (Phase 3)
#   rpde_learning_log        — Continuous learning retrain audit trail (Phase 3)
#   rpde_safety_events       — Safety guard check events log (Phase 3)
#   rpde_trade_experiences   — Full trade experience records for RL (Phase 3)
# =============================================================

import json
import datetime
from core.logger import get_logger

log = get_logger(__name__)

# ── Module-level flag: tables created once per process ───────
_tables_initialized = False


def _get_conn():
    """Get a pooled MySQL connection, ensuring RPDE tables exist on first use."""
    from database.db_manager import get_connection
    conn = get_connection()
    # Consume any leftover unread results from pooled connection
    try:
        while conn.unread_result:
            conn.consume_results()
    except Exception:
        pass
    return conn


# ═══════════════════════════════════════════════════════════════
#  TABLE CREATION
# ═══════════════════════════════════════════════════════════════

def init_rpde_tables():
    """Create all RPDE tables. Call once on first use (idempotent)."""
    global _tables_initialized
    if _tables_initialized:
        return

    conn = _get_conn()
    c = conn.cursor(dictionary=True)

    try:
        # ── 1. rpde_scan_history ────────────────────────────
        c.execute("""
            CREATE TABLE IF NOT EXISTS rpde_scan_history (
                id                  INT AUTO_INCREMENT PRIMARY KEY,
                scan_id             VARCHAR(50) UNIQUE,
                scan_time           DATETIME,
                pairs_scanned       INT,
                total_bars          INT,
                golden_moments_found INT,
                duration_seconds    INT,
                status              VARCHAR(20),
                notes               TEXT,
                INDEX idx_status (status),
                INDEX idx_scan_time (scan_time)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        # ── 2. rpde_pattern_scans ───────────────────────────
        c.execute("""
            CREATE TABLE IF NOT EXISTS rpde_pattern_scans (
                id                  INT AUTO_INCREMENT PRIMARY KEY,
                scan_id             VARCHAR(50),
                pair                VARCHAR(20),
                direction           VARCHAR(10),
                bar_timestamp       DATETIME,
                entry_price         DOUBLE,
                peak_price          DOUBLE,
                move_pips           DOUBLE,
                move_duration_bars  INT,
                forward_return      DOUBLE,
                is_win              TINYINT,
                session             VARCHAR(30),
                market_state        VARCHAR(30),
                atr_at_entry        DOUBLE,
                spread_at_entry     DOUBLE,
                features_json       TEXT,
                created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_pair (pair),
                INDEX idx_scan_id (scan_id),
                INDEX idx_direction (direction),
                INDEX idx_is_win (is_win),
                INDEX idx_pair_direction (pair, direction),
                INDEX idx_bar_timestamp (bar_timestamp)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        # ── 3. rpde_pattern_library ─────────────────────────
        c.execute("""
            CREATE TABLE IF NOT EXISTS rpde_pattern_library (
                id                      INT AUTO_INCREMENT PRIMARY KEY,
                pattern_id              VARCHAR(50) UNIQUE,
                pair                    VARCHAR(20),
                direction               VARCHAR(10),
                cluster_id              INT,
                tier                    VARCHAR(20),
                occurrences             INT,
                wins                    INT,
                losses                  INT,
                win_rate                DOUBLE,
                avg_profit_pips         DOUBLE,
                avg_loss_pips           DOUBLE,
                profit_factor           DOUBLE,
                avg_expected_r          DOUBLE,
                max_drawdown_pips       DOUBLE,
                max_consecutive_losses  INT,
                sharpe_ratio            DOUBLE,
                backtest_start          DATETIME,
                backtest_end            DATETIME,
                backtest_days           INT,
                currency_tag            VARCHAR(30),
                currency_boost_pairs    TEXT,
                cluster_center_json     TEXT,
                feature_ranges_json     TEXT,
                top_features_json       TEXT,
                model_path              VARCHAR(200),
                is_active               TINYINT DEFAULT 1,
                hibernating_since       DATETIME,
                last_validated          DATETIME,
                created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_pair (pair),
                INDEX idx_tier (tier),
                INDEX idx_is_active (is_active),
                INDEX idx_pair_tier (pair, tier),
                INDEX idx_pair_active (pair, is_active)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        # ── 4. rpde_pattern_trades ──────────────────────────
        c.execute("""
            CREATE TABLE IF NOT EXISTS rpde_pattern_trades (
                id                          INT AUTO_INCREMENT PRIMARY KEY,
                ticket                      INT,
                pattern_id                  VARCHAR(50),
                pair                        VARCHAR(20),
                direction                   VARCHAR(10),
                entry_time                  DATETIME,
                exit_time                   DATETIME,
                entry_price                 DOUBLE,
                exit_price                  DOUBLE,
                sl_price                    DOUBLE,
                tp_price                    DOUBLE,
                lot_size                    DOUBLE,
                profit_pips                 DOUBLE,
                profit_r                    DOUBLE,
                profit_usd                  DOUBLE,
                outcome                     VARCHAR(30),
                model_confidence            DOUBLE,
                model_predicted_r           DOUBLE,
                gate_confidence             DOUBLE,
                pattern_win_rate_at_entry   DOUBLE,
                pattern_tier_at_entry       VARCHAR(20),
                session                     VARCHAR(30),
                source                      VARCHAR(20),
                created_at                  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_pair (pair),
                INDEX idx_pattern_id (pattern_id),
                INDEX idx_outcome (outcome),
                INDEX idx_source (source),
                INDEX idx_pair_outcome (pair, outcome),
                INDEX idx_entry_time (entry_time)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        # ── 5. rpde_pattern_stats ───────────────────────────
        c.execute("""
            CREATE TABLE IF NOT EXISTS rpde_pattern_stats (
                id                  INT AUTO_INCREMENT PRIMARY KEY,
                pattern_id          VARCHAR(50) UNIQUE,
                last_30_trades      INT,
                last_30_wins        INT,
                last_30_win_rate    DOUBLE,
                last_30_avg_r       DOUBLE,
                last_100_trades     INT,
                last_100_wins       INT,
                last_100_win_rate   DOUBLE,
                last_100_avg_r      DOUBLE,
                all_time_trades     INT,
                all_time_wins       INT,
                all_time_win_rate   DOUBLE,
                all_time_avg_r      DOUBLE,
                is_decaying         TINYINT DEFAULT 0,
                updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_pattern_id (pattern_id),
                INDEX idx_is_decaying (is_decaying)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        # ── 6. rpde_tft_training_log ──────────────────────
        c.execute("""
            CREATE TABLE IF NOT EXISTS rpde_tft_training_log (
                id                  INT AUTO_INCREMENT PRIMARY KEY,
                pair                VARCHAR(20),
                training_id         VARCHAR(50) UNIQUE,
                started_at          DATETIME,
                completed_at        DATETIME,
                duration_seconds    INT,
                status              VARCHAR(20),
                n_samples           INT,
                train_samples       INT,
                val_samples         INT,
                epochs_trained      INT,
                best_epoch          INT,
                best_val_loss       DOUBLE,
                final_val_loss      DOUBLE,
                train_loss          DOUBLE,
                val_pattern_mae     DOUBLE,
                val_momentum_mae    DOUBLE,
                val_reversal_mae    DOUBLE,
                device              VARCHAR(20),
                n_parameters        INT,
                config_json         TEXT,
                history_json        TEXT,
                notes               TEXT,
                INDEX idx_pair (pair),
                INDEX idx_training_id (training_id),
                INDEX idx_status (status),
                INDEX idx_started_at (started_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        # ── 7. rpde_tft_models ─────────────────────────────
        c.execute("""
            CREATE TABLE IF NOT EXISTS rpde_tft_models (
                id                  INT AUTO_INCREMENT PRIMARY KEY,
                pair                VARCHAR(20) UNIQUE,
                model_path           VARCHAR(300),
                meta_path            VARCHAR(300),
                model_size_kb        INT,
                trained_at          DATETIME,
                last_retrained_at    DATETIME,
                training_samples    INT,
                val_corr             DOUBLE,
                val_r2               DOUBLE,
                val_pattern_mae      DOUBLE,
                val_momentum_mae     DOUBLE,
                val_reversal_mae     DOUBLE,
                best_val_loss        DOUBLE,
                epochs_trained      INT,
                device              VARCHAR(20),
                n_parameters        INT,
                is_active           TINYINT DEFAULT 1,
                config_snapshot_json TEXT,
                INDEX idx_pair (pair),
                INDEX idx_is_active (is_active),
                INDEX idx_trained_at (trained_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        # ── 8. rpde_rl_training_log (Phase 3) ───────────────
        c.execute("""
            CREATE TABLE IF NOT EXISTS rpde_rl_training_log (
                id                  INT AUTO_INCREMENT PRIMARY KEY,
                pair                VARCHAR(20),
                training_id         VARCHAR(50) UNIQUE,
                started_at          DATETIME,
                completed_at        DATETIME,
                duration_seconds    INT,
                status              VARCHAR(20),
                episodes_trained    INT,
                total_steps         INT,
                avg_reward          DOUBLE,
                best_reward         DOUBLE,
                policy_loss         DOUBLE,
                value_loss          DOUBLE,
                entropy             DOUBLE,
                device              VARCHAR(20),
                n_parameters        INT,
                config_json         TEXT,
                history_json        TEXT,
                notes               TEXT,
                INDEX idx_pair (pair),
                INDEX idx_training_id (training_id),
                INDEX idx_status (status),
                INDEX idx_started_at (started_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        # ── 9. rpde_rl_models (Phase 3) ─────────────────────
        c.execute("""
            CREATE TABLE IF NOT EXISTS rpde_rl_models (
                id                  INT AUTO_INCREMENT PRIMARY KEY,
                pair                VARCHAR(20) UNIQUE,
                model_path           VARCHAR(300),
                meta_path            VARCHAR(300),
                model_size_kb        INT,
                trained_at          DATETIME,
                last_retrained_at    DATETIME,
                episodes_trained    INT,
                total_steps         INT,
                avg_reward          DOUBLE,
                best_reward         DOUBLE,
                final_policy_loss   DOUBLE,
                final_value_loss    DOUBLE,
                device              VARCHAR(20),
                n_parameters        INT,
                is_active           TINYINT DEFAULT 1,
                config_snapshot_json TEXT,
                INDEX idx_pair (pair),
                INDEX idx_is_active (is_active),
                INDEX idx_trained_at (trained_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        # ── 10. rpde_learning_log (Phase 3) ────────────────
        c.execute("""
            CREATE TABLE IF NOT EXISTS rpde_learning_log (
                id                  INT AUTO_INCREMENT PRIMARY KEY,
                pair                VARCHAR(20),
                component           VARCHAR(30),
                action              VARCHAR(30),
                triggered_at        DATETIME,
                completed_at        DATETIME,
                duration_seconds    INT,
                status              VARCHAR(20),
                details_json        TEXT,
                notes               TEXT,
                INDEX idx_pair (pair),
                INDEX idx_component (component),
                INDEX idx_action (action),
                INDEX idx_triggered_at (triggered_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        # ── 11. rpde_safety_events (Phase 3) ────────────────
        c.execute("""
            CREATE TABLE IF NOT EXISTS rpde_safety_events (
                id                  INT AUTO_INCREMENT PRIMARY KEY,
                pair                VARCHAR(20),
                guard_name          VARCHAR(50),
                severity            VARCHAR(10),
                action_taken        VARCHAR(20),
                trade_approved      TINYINT,
                reason              TEXT,
                trade_request_json  TEXT,
                account_state_json  TEXT,
                market_state_json   TEXT,
                created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_pair (pair),
                INDEX idx_guard_name (guard_name),
                INDEX idx_severity (severity),
                INDEX idx_action_taken (action_taken),
                INDEX idx_created_at (created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        # ── 12. rpde_trade_experiences (Phase 3) ────────────
        c.execute("""
            CREATE TABLE IF NOT EXISTS rpde_trade_experiences (
                id                      INT AUTO_INCREMENT PRIMARY KEY,
                trade_id                INT,
                pair                    VARCHAR(20),
                direction               VARCHAR(10),
                entry_time              DATETIME,
                exit_time               DATETIME,
                entry_price             DOUBLE,
                exit_price              DOUBLE,
                profit_pips             DOUBLE,
                profit_r                DOUBLE,
                profit_usd              DOUBLE,
                outcome                 VARCHAR(30),
                fusion_confidence       DOUBLE,
                fusion_expected_r       DOUBLE,
                signal_agreement        VARCHAR(20),
                reversal_warning        TINYINT,
                rl_action               INT,
                rl_action_name          VARCHAR(30),
                rl_predicted_value      DOUBLE,
                session                 VARCHAR(30),
                spread_at_entry         DOUBLE,
                atr_at_entry            DOUBLE,
                mae_r                   DOUBLE,
                mfe_r                   DOUBLE,
                hold_time_hours         DOUBLE,
                created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_pair (pair),
                INDEX idx_outcome (outcome),
                INDEX idx_entry_time (entry_time),
                INDEX idx_pair_entry (pair, entry_time)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        _tables_initialized = True
        log.info("[RPDE_DB] All 12 RPDE tables initialized successfully.")

    except Exception as e:
        log.error(f"[RPDE_DB] Failed to create RPDE tables: {e}")
        raise
    finally:
        try:
            c.close()
            conn.close()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

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


def _safe_datetime(val):
    """Safely convert to datetime string for MySQL."""
    if val is None:
        return None
    if isinstance(val, datetime.datetime):
        return val.strftime('%Y-%m-%d %H:%M:%S')
    if isinstance(val, str):
        return val
    return str(val)


def _safe_json(val) -> str:
    """Safely serialize a value to JSON string."""
    if val is None:
        return None
    if isinstance(val, str):
        return val  # assume already serialized
    try:
        return json.dumps(val, default=str, separators=(',', ':'))
    except (TypeError, ValueError) as e:
        log.warning(f"[RPDE_DB] JSON serialization failed: {e}")
        return '{}'


def _parse_json(val):
    """Safely parse a JSON string back to Python object."""
    if val is None:
        return None
    if isinstance(val, (dict, list)):
        return val
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return None


def _close(cursor, conn):
    """Safely close cursor and connection back to pool."""
    try:
        cursor.close()
    except Exception:
        pass
    try:
        conn.close()
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════
#  SCAN HISTORY
# ═══════════════════════════════════════════════════════════════

def store_scan_history(scan_id: str, **kwargs):
    """Insert a scan history record.

    Args:
        scan_id: Unique identifier for the scan run
        **kwargs: Optional fields — scan_time, pairs_scanned, total_bars,
                  golden_moments_found, duration_seconds, status, notes
    """
    init_rpde_tables()
    conn = _get_conn()
    c = conn.cursor(dictionary=True)

    try:
        scan_time = kwargs.get('scan_time') or datetime.datetime.now()
        c.execute("""
            INSERT INTO rpde_scan_history (
                scan_id, scan_time, pairs_scanned, total_bars,
                golden_moments_found, duration_seconds, status, notes
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            scan_id,
            _safe_datetime(scan_time),
            _safe_int(kwargs.get('pairs_scanned', 0)),
            _safe_int(kwargs.get('total_bars', 0)),
            _safe_int(kwargs.get('golden_moments_found', 0)),
            _safe_int(kwargs.get('duration_seconds', 0)),
            kwargs.get('status', 'RUNNING'),
            kwargs.get('notes'),
        ))
        log.info(f"[RPDE_DB] Scan history stored: {scan_id} "
                 f"status={kwargs.get('status', 'RUNNING')}")
    except Exception as e:
        log.error(f"[RPDE_DB] store_scan_history failed for {scan_id}: {e}")
    finally:
        _close(c, conn)


def update_scan_history(scan_id: str, status: str, **kwargs):
    """Update an existing scan history record.

    Args:
        scan_id: Unique identifier for the scan run
        status: New status — RUNNING, COMPLETED, FAILED
        **kwargs: Optional fields — pairs_scanned, total_bars,
                  golden_moments_found, duration_seconds, notes
    """
    init_rpde_tables()
    conn = _get_conn()
    c = conn.cursor(dictionary=True)

    try:
        # Build dynamic SET clause from provided kwargs
        set_parts = ["status = %s"]
        params = [status]

        for field in ('pairs_scanned', 'total_bars', 'golden_moments_found',
                       'duration_seconds', 'notes'):
            if field in kwargs:
                set_parts.append(f"{field} = %s")
                val = kwargs[field]
                params.append(_safe_int(val) if field != 'notes' else val)

        params.append(scan_id)

        query = f"""
            UPDATE rpde_scan_history
            SET {', '.join(set_parts)}
            WHERE scan_id = %s
        """
        c.execute(query, params)
        log.info(f"[RPDE_DB] Scan history updated: {scan_id} → {status}")
    except Exception as e:
        log.error(f"[RPDE_DB] update_scan_history failed for {scan_id}: {e}")
    finally:
        _close(c, conn)


# ═══════════════════════════════════════════════════════════════
#  GOLDEN MOMENTS (rpde_pattern_scans)
# ═══════════════════════════════════════════════════════════════

def store_golden_moment(moment: dict):
    """Store a golden moment (big move) with its feature snapshot.

    Args:
        moment: dict with keys from the scanner:
            scan_id, pair, bar_time, direction, entry_price, move_pips,
            peak_price, peak_bar_offset, forward_return, atr, spread,
            volume, pip_value, threshold_pips, bar_index,
            plus feature_snapshot dict if available.
    """
    init_rpde_tables()
    conn = _get_conn()
    c = conn.cursor(dictionary=True)

    try:
        c.execute("""
            INSERT INTO rpde_pattern_scans (
                scan_id, pair, direction, bar_timestamp, entry_price,
                peak_price, move_pips, move_duration_bars, forward_return,
                is_win, session, market_state, atr_at_entry, spread_at_entry,
                features_json
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            moment.get('scan_id', ''),
            moment.get('pair', ''),
            moment.get('direction', ''),
            _safe_datetime(moment.get('bar_time')),
            _safe_float(moment.get('entry_price')),
            _safe_float(moment.get('peak_price')),
            _safe_float(moment.get('move_pips')),
            _safe_int(moment.get('peak_bar_offset', 0)),
            _safe_float(moment.get('forward_return')),
            1 if moment.get('forward_return', 0) > 0 else 0,
            moment.get('session', ''),
            moment.get('market_state', ''),
            _safe_float(moment.get('atr')),
            _safe_float(moment.get('spread')),
            _safe_json(moment.get('feature_snapshot', moment)),
        ))
    except Exception as e:
        log.error(f"[RPDE_DB] store_golden_moment failed for "
                  f"{moment.get('pair', '?')}: {e}")
    finally:
        _close(c, conn)


def load_golden_moments(pair=None, min_pips=0, scan_id=None):
    """Load golden moments from the database.

    Args:
        pair: Filter by currency pair (None = all pairs)
        min_pips: Minimum move_pips threshold
        scan_id: Filter by specific scan (None = all scans)

    Returns:
        List of dicts, each representing a golden moment row.
        features_json is automatically parsed into a dict.
    """
    init_rpde_tables()
    conn = _get_conn()
    c = conn.cursor(dictionary=True)

    try:
        query = "SELECT * FROM rpde_pattern_scans WHERE 1=1"
        params = []

        if pair:
            query += " AND pair = %s"
            params.append(pair)
        if min_pips > 0:
            query += " AND move_pips >= %s"
            params.append(float(min_pips))
        if scan_id:
            query += " AND scan_id = %s"
            params.append(scan_id)

        query += " ORDER BY bar_timestamp DESC"

        c.execute(query, params)
        rows = c.fetchall()

        # Parse features_json into actual dicts
        for row in rows:
            row['features'] = _parse_json(row.get('features_json'))

        log.info(f"[RPDE_DB] Loaded {len(rows)} golden moments "
                 f"(pair={pair or 'ALL'}, min_pips={min_pips})")
        return rows

    except Exception as e:
        log.error(f"[RPDE_DB] load_golden_moments failed: {e}")
        return []
    finally:
        _close(c, conn)


# ═══════════════════════════════════════════════════════════════
#  PATTERN LIBRARY (rpde_pattern_library)
# ═══════════════════════════════════════════════════════════════

def store_pattern(pattern_dict: dict):
    """Insert or update a pattern in the library.

    Uses ON DUPLICATE KEY UPDATE so re-running validation overwrites
    the previous stats without duplicating rows.

    Args:
        pattern_dict: Dict with all pattern fields. Required: pattern_id.
            All other fields use defaults if not provided.
    """
    init_rpde_tables()
    conn = _get_conn()
    c = conn.cursor(dictionary=True)

    try:
        pattern_id = pattern_dict.get('pattern_id')
        if not pattern_id:
            log.error("[RPDE_DB] store_pattern called without pattern_id")
            return

        # Serialize JSON fields
        currency_boost = _safe_json(pattern_dict.get('currency_boost_pairs'))
        cluster_center = _safe_json(pattern_dict.get('cluster_center_json'))
        feature_ranges = _safe_json(pattern_dict.get('feature_ranges_json'))
        top_features = _safe_json(pattern_dict.get('top_features_json'))

        c.execute("""
            INSERT INTO rpde_pattern_library (
                pattern_id, pair, direction, cluster_id, tier,
                occurrences, wins, losses, win_rate,
                avg_profit_pips, avg_loss_pips, profit_factor,
                avg_expected_r, max_drawdown_pips, max_consecutive_losses,
                sharpe_ratio, backtest_start, backtest_end, backtest_days,
                currency_tag, currency_boost_pairs, cluster_center_json,
                feature_ranges_json, top_features_json, model_path,
                is_active, hibernating_since, last_validated
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON DUPLICATE KEY UPDATE
                pair                    = VALUES(pair),
                direction               = VALUES(direction),
                cluster_id              = VALUES(cluster_id),
                tier                    = VALUES(tier),
                occurrences             = VALUES(occurrences),
                wins                    = VALUES(wins),
                losses                  = VALUES(losses),
                win_rate                = VALUES(win_rate),
                avg_profit_pips         = VALUES(avg_profit_pips),
                avg_loss_pips           = VALUES(avg_loss_pips),
                profit_factor           = VALUES(profit_factor),
                avg_expected_r          = VALUES(avg_expected_r),
                max_drawdown_pips       = VALUES(max_drawdown_pips),
                max_consecutive_losses  = VALUES(max_consecutive_losses),
                sharpe_ratio            = VALUES(sharpe_ratio),
                backtest_start          = VALUES(backtest_start),
                backtest_end            = VALUES(backtest_end),
                backtest_days           = VALUES(backtest_days),
                currency_tag            = VALUES(currency_tag),
                currency_boost_pairs    = VALUES(currency_boost_pairs),
                cluster_center_json     = VALUES(cluster_center_json),
                feature_ranges_json     = VALUES(feature_ranges_json),
                top_features_json       = VALUES(top_features_json),
                model_path              = VALUES(model_path),
                is_active               = VALUES(is_active),
                hibernating_since       = VALUES(hibernating_since),
                last_validated          = VALUES(last_validated)
        """, (
            pattern_id,
            pattern_dict.get('pair'),
            pattern_dict.get('direction'),
            _safe_int(pattern_dict.get('cluster_id')),
            pattern_dict.get('tier', 'PROBATIONARY'),
            _safe_int(pattern_dict.get('occurrences', 0)),
            _safe_int(pattern_dict.get('wins', 0)),
            _safe_int(pattern_dict.get('losses', 0)),
            _safe_float(pattern_dict.get('win_rate')),
            _safe_float(pattern_dict.get('avg_profit_pips')),
            _safe_float(pattern_dict.get('avg_loss_pips')),
            _safe_float(pattern_dict.get('profit_factor')),
            _safe_float(pattern_dict.get('avg_expected_r')),
            _safe_float(pattern_dict.get('max_drawdown_pips')),
            _safe_int(pattern_dict.get('max_consecutive_losses', 0)),
            _safe_float(pattern_dict.get('sharpe_ratio')),
            _safe_datetime(pattern_dict.get('backtest_start')),
            _safe_datetime(pattern_dict.get('backtest_end')),
            _safe_int(pattern_dict.get('backtest_days', 0)),
            pattern_dict.get('currency_tag', 'PAIR_ONLY'),
            currency_boost,
            cluster_center,
            feature_ranges,
            top_features,
            pattern_dict.get('model_path'),
            1 if pattern_dict.get('is_active', True) else 0,
            _safe_datetime(pattern_dict.get('hibernating_since')),
            _safe_datetime(pattern_dict.get('last_validated')),
        ))
        log.info(f"[RPDE_DB] Pattern stored: {pattern_id} "
                 f"tier={pattern_dict.get('tier', 'PROBATIONARY')} "
                 f"wr={_safe_float(pattern_dict.get('win_rate', 0)):.1%}")
    except Exception as e:
        log.error(f"[RPDE_DB] store_pattern failed for {pattern_dict.get('pattern_id')}: {e}")
    finally:
        _close(c, conn)


def load_pattern_library(pair=None, active_only=True, min_tier=None):
    """Load patterns from the library.

    Args:
        pair: Filter by currency pair (None = all pairs)
        active_only: If True, only return is_active=1 patterns
        min_tier: Minimum tier to include. Tiers ordered:
            GOD_TIER > STRONG > VALID > PROBATIONARY
            If set, only return patterns at this tier or above.

    Returns:
        List of dicts. JSON text fields are parsed into Python objects.
    """
    init_rpde_tables()
    conn = _get_conn()
    c = conn.cursor(dictionary=True)

    try:
        query = "SELECT * FROM rpde_pattern_library WHERE 1=1"
        params = []

        if pair:
            query += " AND pair = %s"
            params.append(pair)
        if active_only:
            query += " AND is_active = 1"
        if min_tier:
            tier_order = {
                'PROBATIONARY': 0,
                'VALID': 1,
                'STRONG': 2,
                'GOD_TIER': 3,
            }
            min_level = tier_order.get(min_tier, 0)
            query += " AND CASE tier"
            for tier_name, level in tier_order.items():
                query += f" WHEN '{tier_name}' THEN {level}"
            query += f" ELSE 0 END >= %s"
            params.append(min_level)

        query += " ORDER BY pair, tier DESC, win_rate DESC"

        c.execute(query, params)
        rows = c.fetchall()

        # Parse JSON fields
        for row in rows:
            row['currency_boost_pairs'] = _parse_json(row.get('currency_boost_pairs'))
            row['cluster_center'] = _parse_json(row.get('cluster_center_json'))
            row['feature_ranges'] = _parse_json(row.get('feature_ranges_json'))
            row['top_features'] = _parse_json(row.get('top_features_json'))

        log.info(f"[RPDE_DB] Loaded {len(rows)} patterns "
                 f"(pair={pair or 'ALL'}, active={active_only})")
        return rows

    except Exception as e:
        log.error(f"[RPDE_DB] load_pattern_library failed: {e}")
        return []
    finally:
        _close(c, conn)


# ═══════════════════════════════════════════════════════════════
#  PATTERN TRADES (rpde_pattern_trades)
# ═══════════════════════════════════════════════════════════════

def store_pattern_trade(trade_dict: dict):
    """Record a pattern-based trade (live, paper, or backtest).

    Args:
        trade_dict: Dict with trade fields. Required: pair, direction, entry_time.
            ticket, pattern_id, exit_time, entry_price, exit_price,
            sl_price, tp_price, lot_size, profit_pips, profit_r,
            profit_usd, outcome, model_confidence, model_predicted_r,
            gate_confidence, pattern_win_rate_at_entry,
            pattern_tier_at_entry, session, source are all optional.
    """
    init_rpde_tables()
    conn = _get_conn()
    c = conn.cursor(dictionary=True)

    try:
        c.execute("""
            INSERT INTO rpde_pattern_trades (
                ticket, pattern_id, pair, direction, entry_time, exit_time,
                entry_price, exit_price, sl_price, tp_price, lot_size,
                profit_pips, profit_r, profit_usd, outcome,
                model_confidence, model_predicted_r, gate_confidence,
                pattern_win_rate_at_entry, pattern_tier_at_entry,
                session, source
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
        """, (
            trade_dict.get('ticket'),
            trade_dict.get('pattern_id'),
            trade_dict.get('pair'),
            trade_dict.get('direction'),
            _safe_datetime(trade_dict.get('entry_time')),
            _safe_datetime(trade_dict.get('exit_time')),
            _safe_float(trade_dict.get('entry_price')),
            _safe_float(trade_dict.get('exit_price')),
            _safe_float(trade_dict.get('sl_price')),
            _safe_float(trade_dict.get('tp_price')),
            _safe_float(trade_dict.get('lot_size')),
            _safe_float(trade_dict.get('profit_pips')),
            _safe_float(trade_dict.get('profit_r')),
            _safe_float(trade_dict.get('profit_usd')),
            trade_dict.get('outcome'),
            _safe_float(trade_dict.get('model_confidence')),
            _safe_float(trade_dict.get('model_predicted_r')),
            _safe_float(trade_dict.get('gate_confidence')),
            _safe_float(trade_dict.get('pattern_win_rate_at_entry')),
            trade_dict.get('pattern_tier_at_entry'),
            trade_dict.get('session'),
            trade_dict.get('source', 'LIVE'),
        ))
        log.info(f"[RPDE_DB] Pattern trade stored: "
                 f"{trade_dict.get('pair')} {trade_dict.get('direction')} "
                 f"outcome={trade_dict.get('outcome')} "
                 f"R={_safe_float(trade_dict.get('profit_r', 0)):.2f}")
    except Exception as e:
        log.error(f"[RPDE_DB] store_pattern_trade failed for "
                  f"{trade_dict.get('pair')}: {e}")
    finally:
        _close(c, conn)


def load_pattern_trades(pair=None, pattern_id=None, source=None):
    """Load pattern-based trades from the database.

    Args:
        pair: Filter by currency pair (None = all)
        pattern_id: Filter by pattern_id (None = all)
        source: Filter by source — LIVE, PAPER, BACKTEST (None = all)

    Returns:
        List of dicts, each representing a trade row.
    """
    init_rpde_tables()
    conn = _get_conn()
    c = conn.cursor(dictionary=True)

    try:
        query = "SELECT * FROM rpde_pattern_trades WHERE 1=1"
        params = []

        if pair:
            query += " AND pair = %s"
            params.append(pair)
        if pattern_id:
            query += " AND pattern_id = %s"
            params.append(pattern_id)
        if source:
            query += " AND source = %s"
            params.append(source)

        query += " ORDER BY entry_time DESC"

        c.execute(query, params)
        rows = c.fetchall()

        log.info(f"[RPDE_DB] Loaded {len(rows)} pattern trades "
                 f"(pair={pair or 'ALL'}, pattern={pattern_id or 'ALL'})")
        return rows

    except Exception as e:
        log.error(f"[RPDE_DB] load_pattern_trades failed: {e}")
        return []
    finally:
        _close(c, conn)


# ═══════════════════════════════════════════════════════════════
#  PATTERN STATS (rpde_pattern_stats)
# ═══════════════════════════════════════════════════════════════

def update_pattern_stats(pattern_id: str):
    """Recalculate rolling statistics for a specific pattern.

    Computes:
      - Last 30 trades: count, wins, win rate, avg R
      - Last 100 trades: count, wins, win rate, avg R
      - All-time: count, wins, win rate, avg R
      - is_decaying: 1 if last_30_win_rate < 70% of all_time_win_rate

    Uses ON DUPLICATE KEY UPDATE for upsert behavior.

    Args:
        pattern_id: The pattern to recalculate stats for.
    """
    init_rpde_tables()
    conn = _get_conn()
    c = conn.cursor(dictionary=True)

    try:
        # ── Compute all-time stats ──────────────────────────
        c.execute("""
            SELECT
                COUNT(*)                    AS total_trades,
                SUM(CASE WHEN profit_r > 0 THEN 1 ELSE 0 END) AS total_wins,
                AVG(profit_r)               AS avg_r
            FROM rpde_pattern_trades
            WHERE pattern_id = %s
              AND profit_r IS NOT NULL
        """, (pattern_id,))
        at = c.fetchone()
        all_time_trades = _safe_int(at['total_trades']) if at else 0
        all_time_wins = _safe_int(at['total_wins']) if at else 0
        all_time_win_rate = (all_time_wins / all_time_trades) if all_time_trades > 0 else 0.0
        all_time_avg_r = _safe_float(at['avg_r']) if at and at['avg_r'] is not None else 0.0

        # ── Compute last 100 trades ─────────────────────────
        c.execute("""
            SELECT
                COUNT(*)                    AS cnt,
                SUM(CASE WHEN profit_r > 0 THEN 1 ELSE 0 END) AS wins,
                AVG(profit_r)               AS avg_r
            FROM (
                SELECT profit_r
                FROM rpde_pattern_trades
                WHERE pattern_id = %s
                  AND profit_r IS NOT NULL
                ORDER BY entry_time DESC
                LIMIT 100
            ) sub
        """, (pattern_id,))
        r100 = c.fetchone()
        last_100_trades = _safe_int(r100['cnt']) if r100 else 0
        last_100_wins = _safe_int(r100['wins']) if r100 else 0
        last_100_win_rate = (last_100_wins / last_100_trades) if last_100_trades > 0 else 0.0
        last_100_avg_r = _safe_float(r100['avg_r']) if r100 and r100['avg_r'] is not None else 0.0

        # ── Compute last 30 trades ──────────────────────────
        c.execute("""
            SELECT
                COUNT(*)                    AS cnt,
                SUM(CASE WHEN profit_r > 0 THEN 1 ELSE 0 END) AS wins,
                AVG(profit_r)               AS avg_r
            FROM (
                SELECT profit_r
                FROM rpde_pattern_trades
                WHERE pattern_id = %s
                  AND profit_r IS NOT NULL
                ORDER BY entry_time DESC
                LIMIT 30
            ) sub
        """, (pattern_id,))
        r30 = c.fetchone()
        last_30_trades = _safe_int(r30['cnt']) if r30 else 0
        last_30_wins = _safe_int(r30['wins']) if r30 else 0
        last_30_win_rate = (last_30_wins / last_30_trades) if last_30_trades > 0 else 0.0
        last_30_avg_r = _safe_float(r30['avg_r']) if r30 and r30['avg_r'] is not None else 0.0

        # ── Decay detection ─────────────────────────────────
        # Pattern is decaying if recent win rate < 70% of all-time
        is_decaying = 0
        if all_time_win_rate > 0 and all_time_trades >= 10:
            decay_threshold = all_time_win_rate * 0.70
            if last_30_win_rate < decay_threshold and last_30_trades >= 10:
                is_decaying = 1

        # ── Upsert stats ────────────────────────────────────
        c.execute("""
            INSERT INTO rpde_pattern_stats (
                pattern_id,
                last_30_trades, last_30_wins, last_30_win_rate, last_30_avg_r,
                last_100_trades, last_100_wins, last_100_win_rate, last_100_avg_r,
                all_time_trades, all_time_wins, all_time_win_rate, all_time_avg_r,
                is_decaying
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON DUPLICATE KEY UPDATE
                last_30_trades      = VALUES(last_30_trades),
                last_30_wins        = VALUES(last_30_wins),
                last_30_win_rate    = VALUES(last_30_win_rate),
                last_30_avg_r       = VALUES(last_30_avg_r),
                last_100_trades     = VALUES(last_100_trades),
                last_100_wins       = VALUES(last_100_wins),
                last_100_win_rate   = VALUES(last_100_win_rate),
                last_100_avg_r      = VALUES(last_100_avg_r),
                all_time_trades     = VALUES(all_time_trades),
                all_time_wins       = VALUES(all_time_wins),
                all_time_win_rate   = VALUES(all_time_win_rate),
                all_time_avg_r      = VALUES(all_time_avg_r),
                is_decaying         = VALUES(is_decaying)
        """, (
            pattern_id,
            last_30_trades, last_30_wins,
            round(last_30_win_rate, 6), round(last_30_avg_r, 6),
            last_100_trades, last_100_wins,
            round(last_100_win_rate, 6), round(last_100_avg_r, 6),
            all_time_trades, all_time_wins,
            round(all_time_win_rate, 6), round(all_time_avg_r, 6),
            is_decaying,
        ))

        log.info(f"[RPDE_DB] Stats updated for {pattern_id}: "
                 f"AT={all_time_trades}tr WR={all_time_win_rate:.1%} "
                 f"L30={last_30_win_rate:.1%} L100={last_100_win_rate:.1%} "
                 f"decaying={is_decaying}")

    except Exception as e:
        log.error(f"[RPDE_DB] update_pattern_stats failed for {pattern_id}: {e}")
    finally:
        _close(c, conn)


# ═══════════════════════════════════════════════════════════════
#  PERFORMANCE REPORT
# ═══════════════════════════════════════════════════════════════

def get_pattern_performance_report():
    """Get summary statistics across all patterns.

    Returns a dict with:
      - total_patterns: Total patterns in library
      - active_patterns: Currently active patterns
      - by_tier: Dict of tier -> count
      - by_pair: Dict of pair -> list of pattern summaries
      - by_direction: Dict of BUY/SELL -> count
      - decaying_patterns: List of pattern_ids that are decaying
      - top_patterns: Top 10 patterns by win_rate (min 10 trades)
      - avg_win_rate: Average win rate across all patterns
      - avg_profit_factor: Average profit factor across all patterns
      - total_trades: Total pattern-based trades executed
      - total_profit_r: Total R-multiple from all pattern trades
      - overall_win_rate: Win rate across all pattern trades
    """
    init_rpde_tables()
    conn = _get_conn()
    c = conn.cursor(dictionary=True)

    try:
        report = {}

        # ── Pattern library overview ────────────────────────
        c.execute("""
            SELECT
                COUNT(*)                AS total_patterns,
                SUM(is_active = 1)      AS active_patterns,
                AVG(win_rate)           AS avg_win_rate,
                AVG(profit_factor)      AS avg_profit_factor,
                SUM(occurrences)        AS total_occurrences
            FROM rpde_pattern_library
        """)
        overview = c.fetchone()
        report['total_patterns'] = _safe_int(overview['total_patterns'])
        report['active_patterns'] = _safe_int(overview['active_patterns'])
        report['avg_win_rate'] = _safe_float(overview['avg_win_rate'])
        report['avg_profit_factor'] = _safe_float(overview['avg_profit_factor'])
        report['total_occurrences'] = _safe_int(overview['total_occurrences'])

        # ── By tier ─────────────────────────────────────────
        c.execute("""
            SELECT tier, COUNT(*) AS cnt
            FROM rpde_pattern_library
            GROUP BY tier
            ORDER BY cnt DESC
        """)
        report['by_tier'] = {row['tier']: row['cnt'] for row in c.fetchall()}

        # ── By pair ─────────────────────────────────────────
        c.execute("""
            SELECT
                pair,
                COUNT(*)                    AS pattern_count,
                AVG(win_rate)               AS avg_win_rate,
                AVG(profit_factor)          AS avg_profit_factor,
                SUM(occurrences)            AS total_occurrences,
                SUM(CASE WHEN is_active = 1 THEN 1 ELSE 0 END) AS active_count
            FROM rpde_pattern_library
            GROUP BY pair
            ORDER BY avg_win_rate DESC
        """)
        report['by_pair'] = c.fetchall()

        # ── By direction ────────────────────────────────────
        c.execute("""
            SELECT direction, COUNT(*) AS cnt
            FROM rpde_pattern_library
            GROUP BY direction
        """)
        report['by_direction'] = {row['direction']: row['cnt'] for row in c.fetchall()}

        # ── Decaying patterns ───────────────────────────────
        c.execute("""
            SELECT s.pattern_id, s.all_time_win_rate, s.last_30_win_rate,
                   p.pair, p.tier
            FROM rpde_pattern_stats s
            JOIN rpde_pattern_library p ON s.pattern_id = p.pattern_id
            WHERE s.is_decaying = 1
              AND p.is_active = 1
            ORDER BY s.last_30_win_rate ASC
        """)
        report['decaying_patterns'] = c.fetchall()

        # ── Top patterns (min 10 trades in library) ─────────
        c.execute("""
            SELECT
                pattern_id, pair, direction, tier,
                occurrences, win_rate, profit_factor, avg_expected_r,
                avg_profit_pips, max_drawdown_pips
            FROM rpde_pattern_library
            WHERE occurrences >= 10
              AND is_active = 1
            ORDER BY profit_factor DESC
            LIMIT 10
        """)
        report['top_patterns'] = c.fetchall()

        # ── Trade statistics ────────────────────────────────
        c.execute("""
            SELECT
                COUNT(*)                                    AS total_trades,
                SUM(CASE WHEN profit_r > 0 THEN 1 ELSE 0 END) AS total_wins,
                SUM(profit_r)                               AS total_profit_r,
                SUM(profit_pips)                            AS total_profit_pips,
                AVG(profit_r)                               AS avg_profit_r,
                AVG(profit_pips)                            AS avg_profit_pips
            FROM rpde_pattern_trades
            WHERE profit_r IS NOT NULL
        """)
        trade_stats = c.fetchone()
        total_trades = _safe_int(trade_stats['total_trades'])
        total_wins = _safe_int(trade_stats['total_wins'])
        report['total_trades'] = total_trades
        report['total_wins'] = total_wins
        report['overall_win_rate'] = (total_wins / total_trades) if total_trades > 0 else 0.0
        report['total_profit_r'] = _safe_float(trade_stats['total_profit_r'])
        report['total_profit_pips'] = _safe_float(trade_stats['total_profit_pips'])
        report['avg_profit_r'] = _safe_float(trade_stats['avg_profit_r'])
        report['avg_profit_pips'] = _safe_float(trade_stats['avg_profit_pips'])

        # ── Trade stats by source ───────────────────────────
        c.execute("""
            SELECT
                source,
                COUNT(*)                                    AS trades,
                SUM(CASE WHEN profit_r > 0 THEN 1 ELSE 0 END) AS wins,
                SUM(profit_r)                               AS total_r,
                AVG(profit_r)                               AS avg_r
            FROM rpde_pattern_trades
            WHERE profit_r IS NOT NULL
            GROUP BY source
        """)
        by_source_rows = c.fetchall()
        report['by_source'] = []
        for row in by_source_rows:
            t = _safe_int(row['trades'])
            w = _safe_int(row['wins'])
            report['by_source'].append({
                'source': row['source'],
                'trades': t,
                'wins': w,
                'win_rate': (w / t) if t > 0 else 0.0,
                'total_r': _safe_float(row['total_r']),
                'avg_r': _safe_float(row['avg_r']),
            })

        # ── Trade stats by pair ─────────────────────────────
        c.execute("""
            SELECT
                pair,
                COUNT(*)                                    AS trades,
                SUM(CASE WHEN profit_r > 0 THEN 1 ELSE 0 END) AS wins,
                SUM(profit_r)                               AS total_r,
                AVG(profit_r)                               AS avg_r
            FROM rpde_pattern_trades
            WHERE profit_r IS NOT NULL
            GROUP BY pair
            ORDER BY total_r DESC
        """)
        by_pair_trades_rows = c.fetchall()
        report['trades_by_pair'] = []
        for row in by_pair_trades_rows:
            t = _safe_int(row['trades'])
            w = _safe_int(row['wins'])
            report['trades_by_pair'].append({
                'pair': row['pair'],
                'trades': t,
                'wins': w,
                'win_rate': (w / t) if t > 0 else 0.0,
                'total_r': _safe_float(row['total_r']),
                'avg_r': _safe_float(row['avg_r']),
            })

        # ── Hibernating patterns ────────────────────────────
        c.execute("""
            SELECT pattern_id, pair, tier, hibernating_since,
                   win_rate, occurrences
            FROM rpde_pattern_library
            WHERE is_active = 0
              AND hibernating_since IS NOT NULL
            ORDER BY hibernating_since DESC
        """)
        report['hibernating_patterns'] = c.fetchall()

        log.info(f"[RPDE_DB] Performance report generated: "
                 f"{report['total_patterns']} patterns, "
                 f"{report['active_patterns']} active, "
                 f"{report['total_trades']} trades, "
                 f"WR={report['overall_win_rate']:.1%}")

        return report

    except Exception as e:
        log.error(f"[RPDE_DB] get_pattern_performance_report failed: {e}")
        return {
            'total_patterns': 0, 'active_patterns': 0,
            'by_tier': {}, 'by_pair': [], 'by_direction': {},
            'decaying_patterns': [], 'top_patterns': [],
            'total_trades': 0, 'overall_win_rate': 0.0,
            'total_profit_r': 0.0, 'by_source': [],
            'trades_by_pair': [], 'hibernating_patterns': [],
            'avg_win_rate': 0.0, 'avg_profit_factor': 0.0,
            'total_occurrences': 0, 'total_wins': 0,
            'total_profit_pips': 0.0, 'avg_profit_r': 0.0,
            'avg_profit_pips': 0.0,
        }
    finally:
        _close(c, conn)
