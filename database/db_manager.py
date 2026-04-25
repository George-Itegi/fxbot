# =============================================================
# database/db_manager.py
# The system's memory. Every signal, trade, and data point
# is recorded here for AI analysis and performance review.
#
# v4.1: Converted from SQLite to MySQL (mysql-connector-python).
# Uses connection pooling with automatic reconnect.
# All ON CONFLICT → ON DUPLICATE KEY UPDATE for MySQL syntax.
# =============================================================

import os
import mysql.connector
from mysql.connector import Error, pooling
from datetime import datetime

# ── MySQL Connection ──────────────────────────────────────────
# Reads from environment variables or .env file.
# Falls back to sensible defaults for local development.
DB_HOST     = os.getenv('DB_HOST', 'localhost')
DB_PORT     = int(os.getenv('DB_PORT', '3306'))
DB_USER     = os.getenv('DB_USER', 'root')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')
DB_NAME     = os.getenv('DB_NAME', 'apex_trader')

# Connection pool (shared across all threads)
_pool = None


def _get_pool():
    """Create or return the MySQL connection pool."""
    global _pool
    if _pool is None:
        try:
            _pool = pooling.MySQLConnectionPool(
                pool_name="apex_pool",
                pool_size=50,
                pool_reset_session=True,
                host=DB_HOST,
                port=DB_PORT,
                user=DB_USER,
                password=DB_PASSWORD,
                database=DB_NAME,
                autocommit=True,
            )
        except Error as e:
            print(f"[DATABASE] ❌ Cannot create MySQL pool: {e}")
            raise
    return _pool


def get_connection():
    """Get a connection from the pool with retry on exhaustion."""
    import time
    pool = _get_pool()
    for attempt in range(3):
        try:
            return pool.get_connection()
        except Exception:
            time.sleep(0.1 * (attempt + 1))
    # Last attempt — let it raise
    return pool.get_connection()


def _row_to_dict(cursor, row):
    """Convert a DB row to a dict (simulates sqlite3.Row)."""
    if row is None:
        return None
    return {cursor.description[i][0]: row[i] for i in range(len(cursor.description))}


def init_db():
    """Create all tables on first run (MySQL syntax)."""
    conn = get_connection()
    c = conn.cursor(dictionary=True)

    # --- TRADES TABLE ---
    c.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id                INT AUTO_INCREMENT PRIMARY KEY,
            ticket            INT,
            timestamp_open    DATETIME,
            timestamp_close   DATETIME,
            symbol            VARCHAR(20),
            direction         VARCHAR(10),
            strategy          VARCHAR(50),
            session           VARCHAR(30),
            timeframe         VARCHAR(10),
            entry_price       DOUBLE,
            exit_price        DOUBLE,
            sl_price          DOUBLE,
            tp_price          DOUBLE,
            lot_size          DOUBLE,
            profit_loss       DOUBLE,
            outcome           VARCHAR(20),
            ai_score          DOUBLE,
            confluence_count  INT,
            rsi_at_entry      DOUBLE,
            atr_at_entry      DOUBLE,
            spread_at_entry   DOUBLE,
            market_regime     VARCHAR(30),
            notes             TEXT,
            INDEX idx_ticket (ticket),
            INDEX idx_symbol (symbol),
            INDEX idx_outcome (outcome)
        )
    """)

    # --- SIGNALS TABLE (every evaluated signal, traded or not) ---
    c.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id               INT AUTO_INCREMENT PRIMARY KEY,
            timestamp        DATETIME,
            symbol           VARCHAR(20),
            direction        VARCHAR(10),
            strategy         VARCHAR(50),
            ai_score         DOUBLE,
            confluence_count INT,
            was_traded       TINYINT,
            skip_reason      TEXT,
            session          VARCHAR(30),
            market_regime    VARCHAR(30),
            INDEX idx_symbol (symbol),
            INDEX idx_strategy (strategy)
        )
    """)

    # --- MARKET SNAPSHOT TABLE (external data logged each cycle) ---
    c.execute("""
        CREATE TABLE IF NOT EXISTS market_snapshots (
            id              INT AUTO_INCREMENT PRIMARY KEY,
            timestamp       DATETIME,
            fear_greed      DOUBLE,
            vix             DOUBLE,
            dxy             DOUBLE,
            gold_price      DOUBLE,
            oil_price       DOUBLE,
            sp500           DOUBLE,
            bond_yield_10y  DOUBLE,
            cot_net_pos     TEXT,
            news_sentiment  DOUBLE
        )
    """)

    # --- STRATEGY PERFORMANCE TABLE (updated after each trade closes) ---
    c.execute("""
        CREATE TABLE IF NOT EXISTS strategy_performance (
            id           INT AUTO_INCREMENT PRIMARY KEY,
            strategy     VARCHAR(50) UNIQUE,
            total_trades INT DEFAULT 0,
            wins         INT DEFAULT 0,
            losses       INT DEFAULT 0,
            breakevens   INT DEFAULT 0,
            total_pnl    DOUBLE DEFAULT 0.0,
            win_rate     DOUBLE DEFAULT 0.0,
            avg_rr       DOUBLE DEFAULT 0.0,
            last_updated DATETIME
        )
    """)

    c.close()
    conn.close()
    print("[DATABASE] ✅ All tables initialized (MySQL).")


def log_trade(data: dict):
    """Record a new trade when it opens."""
    try:
        conn = get_connection()
        c = conn.cursor(dictionary=True)
        c.execute("""
            INSERT INTO trades (
                ticket, timestamp_open, symbol, direction, strategy, session,
                timeframe, entry_price, sl_price, tp_price, lot_size,
                ai_score, confluence_count, rsi_at_entry, atr_at_entry,
                spread_at_entry, market_regime, notes
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            data.get('ticket'), datetime.now().isoformat(),
            data['symbol'], data['direction'], data['strategy'],
            data.get('session'), data.get('timeframe'), data['entry_price'],
            data['sl_price'], data['tp_price'], data['lot_size'],
            data.get('ai_score'), data.get('confluence_count'),
            data.get('rsi'), data.get('atr'), data.get('spread'),
            data.get('market_regime'), data.get('notes')
        ))
        c.close()
        conn.close()
    except Exception as e:
        from core.logger import get_logger
        get_logger("DB_LOG").error(f"[DB] log_trade failed for {data.get('symbol')}: {e}")


def close_trade(ticket: int, exit_price: float,
                profit_loss: float, outcome: str):
    """
    Update trade record when it closes.
    Called after SL hit, TP hit, or manual close.
    outcome: 'WIN_TP' | 'WIN_TP2' | 'LOSS' | 'LOSS_SL' | 'BREAKEVEN' | 'MANUAL' | 'MANUAL_LOSS'
    
    v4.2 FIX: Logs rowcount to detect silent UPDATE failures.
    Also handles LOSS_SL outcome type.
    Returns True if update succeeded, False if no row matched.
    """
    from core.logger import get_logger
    _log = get_logger("DB_CLOSE")
    
    conn = get_connection()
    c = conn.cursor(dictionary=True)
    
    try:
        c.execute("""
            UPDATE trades
            SET timestamp_close = %s,
                exit_price      = %s,
                profit_loss     = %s,
                outcome         = %s
            WHERE ticket = %s
              AND timestamp_close IS NULL
        """, (
            datetime.now().isoformat(),
            exit_price,
            round(profit_loss, 2),
            outcome,
            ticket
        ))
        
        if c.rowcount == 0:
            # v4.2: No row matched — either ticket doesn't exist or already closed
            _log.warning(f"[DB] close_trade: No row matched for ticket #{ticket}. "
                         f"Already closed or ticket not found.")
            c.close()
            conn.close()
            return False
        
        _log.info(f"[DB] ✅ Updated trade #{ticket}: {outcome} "
                  f"P&L:{profit_loss:.2f} exit:{exit_price}")

        # Also update strategy_performance table
        c.execute("SELECT strategy FROM trades WHERE ticket = %s", (ticket,))
        row = c.fetchone()
        if row:
            strategy = row['strategy']
            won = 1 if 'WIN' in outcome else 0
            lost = 1 if 'LOSS' in outcome else 0
            be = 1 if outcome == 'BREAKEVEN' else 0
            now = datetime.now().isoformat()

            # MySQL: INSERT ... ON DUPLICATE KEY UPDATE
            c.execute("""
                INSERT INTO strategy_performance
                    (strategy, total_trades, wins, losses, breakevens,
                     total_pnl, win_rate, last_updated)
                VALUES (%s, 1, %s, %s, %s, %s, 0, %s)
                ON DUPLICATE KEY UPDATE
                    total_trades = total_trades + 1,
                    wins         = wins + VALUES(wins),
                    losses       = losses + VALUES(losses),
                    breakevens   = breakevens + VALUES(breakevens),
                    total_pnl    = round(total_pnl + VALUES(total_pnl), 2),
                    win_rate     = round(100.0*(wins+VALUES(wins)) /
                                   (total_trades+1), 1),
                    last_updated = VALUES(last_updated)
            """, (
                strategy, won, lost, be,
                round(profit_loss, 2), now,
            ))
    except Exception as e:
        _log.error(f"[DB] close_trade error for #{ticket}: {e}")
    finally:
        try:
            c.close()
            conn.close()
        except Exception:
            pass
    
    return True


def log_signal(data: dict):
    try:
        conn = get_connection()
        c = conn.cursor(dictionary=True)
        c.execute("""
            INSERT INTO signals (
                timestamp, symbol, direction, strategy, ai_score,
                confluence_count, was_traded, skip_reason, session, market_regime
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            datetime.now().isoformat(), data['symbol'], data.get('direction'),
            data.get('strategy'), data.get('ai_score'), data.get('confluence_count'),
            int(data.get('was_traded', False)), data.get('skip_reason'),
            data.get('session'), data.get('market_regime')
        ))
        c.close()
        conn.close()
    except Exception:
        pass  # Don't crash scan if DB is busy


def log_market_snapshot(data: dict):
    try:
        conn = get_connection()
        c = conn.cursor(dictionary=True)
        c.execute("""
            INSERT INTO market_snapshots (
                timestamp, fear_greed, vix, dxy, gold_price,
                oil_price, sp500, bond_yield_10y, cot_net_pos, news_sentiment
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            datetime.now().isoformat(),
            data.get('fear_greed'), data.get('vix'), data.get('dxy'),
            data.get('gold_price'), data.get('oil_price'), data.get('sp500'),
            data.get('bond_yield_10y'), str(data.get('cot_net_pos', {})),
            data.get('news_sentiment')
        ))
        c.close()
        conn.close()
    except Exception:
        pass
