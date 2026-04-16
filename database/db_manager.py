# =============================================================
# database/db_manager.py
# The system's memory. Every signal, trade, and data point
# is recorded here for AI analysis and performance review.
# =============================================================

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), 'apex_trader.db')


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all tables on first run."""
    conn = get_connection()
    c = conn.cursor()

    # --- TRADES TABLE ---
    c.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket            INTEGER,
            timestamp_open    TEXT,
            timestamp_close   TEXT,
            symbol            TEXT,
            direction         TEXT,
            strategy          TEXT,
            session           TEXT,
            timeframe         TEXT,
            entry_price       REAL,
            exit_price        REAL,
            sl_price          REAL,
            tp_price          REAL,
            lot_size          REAL,
            profit_loss       REAL,
            outcome           TEXT,
            ai_score          REAL,
            confluence_count  INTEGER,
            rsi_at_entry      REAL,
            atr_at_entry      REAL,
            spread_at_entry   REAL,
            market_regime     TEXT,
            notes             TEXT
        )
    """)

    # --- SIGNALS TABLE (every evaluated signal, traded or not) ---
    c.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT,
            symbol           TEXT,
            direction        TEXT,
            strategy         TEXT,
            ai_score         REAL,
            confluence_count INTEGER,
            was_traded       INTEGER,
            skip_reason      TEXT,
            session          TEXT,
            market_regime    TEXT
        )
    """)

    # --- MARKET SNAPSHOT TABLE (external data logged each cycle) ---
    c.execute("""
        CREATE TABLE IF NOT EXISTS market_snapshots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT,
            fear_greed      REAL,
            vix             REAL,
            dxy             REAL,
            gold_price      REAL,
            oil_price       REAL,
            sp500           REAL,
            bond_yield_10y  REAL,
            cot_net_pos     TEXT,
            news_sentiment  REAL
        )
    """)

    # --- STRATEGY PERFORMANCE TABLE (updated after each trade closes) ---
    c.execute("""
        CREATE TABLE IF NOT EXISTS strategy_performance (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy     TEXT UNIQUE,
            total_trades INTEGER DEFAULT 0,
            wins         INTEGER DEFAULT 0,
            losses       INTEGER DEFAULT 0,
            breakevens   INTEGER DEFAULT 0,
            total_pnl    REAL DEFAULT 0.0,
            win_rate     REAL DEFAULT 0.0,
            avg_rr       REAL DEFAULT 0.0,
            last_updated TEXT
        )
    """)

    conn.commit()
    conn.close()
    print("[DATABASE] ✅ All tables initialized.")

def log_trade(data: dict):
    """Record a new trade when it opens."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO trades (
            ticket, timestamp_open, symbol, direction, strategy, session,
            timeframe, entry_price, sl_price, tp_price, lot_size,
            ai_score, confluence_count, rsi_at_entry, atr_at_entry,
            spread_at_entry, market_regime, notes
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        data.get('ticket'), datetime.now().isoformat(),
        data['symbol'], data['direction'], data['strategy'],
        data.get('session'), data.get('timeframe'), data['entry_price'],
        data['sl_price'], data['tp_price'], data['lot_size'],
        data.get('ai_score'), data.get('confluence_count'),
        data.get('rsi'), data.get('atr'), data.get('spread'),
        data.get('market_regime'), data.get('notes')
    ))
    conn.commit()
    conn.close()


def close_trade(ticket: int, exit_price: float,
                profit_loss: float, outcome: str):
    """
    Update trade record when it closes.
    Called after SL hit, TP hit, or manual close.
    outcome: 'WIN_TP' | 'WIN_TP2' | 'LOSS' | 'BREAKEVEN' | 'MANUAL'
    """
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        UPDATE trades
        SET timestamp_close = ?,
            exit_price      = ?,
            profit_loss     = ?,
            outcome         = ?
        WHERE ticket = ?
    """, (
        datetime.now().isoformat(),
        exit_price,
        round(profit_loss, 2),
        outcome,
        ticket
    ))

    # Also update strategy_performance table
    if c.rowcount > 0:
        c.execute("SELECT strategy FROM trades WHERE ticket = ?", (ticket,))
        row = c.fetchone()
        if row:
            strategy = row['strategy']
            won = 1 if 'WIN' in outcome else 0
            lost = 1 if outcome == 'LOSS' else 0
            be = 1 if outcome == 'BREAKEVEN' else 0
            c.execute("""
                INSERT INTO strategy_performance
                    (strategy, total_trades, wins, losses, breakevens,
                     total_pnl, win_rate, last_updated)
                VALUES (?, 1, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(strategy) DO UPDATE SET
                    total_trades = total_trades + 1,
                    wins         = wins + ?,
                    losses       = losses + ?,
                    breakevens   = breakevens + ?,
                    total_pnl    = round(total_pnl + ?, 2),
                    win_rate     = round(100.0*(wins+?) /
                                   (total_trades+1), 1),
                    last_updated = ?
            """, (
                strategy, won, lost, be,
                round(profit_loss, 2), datetime.now().isoformat(),
                won, lost, be,
                round(profit_loss, 2), won,
                datetime.now().isoformat()
            ))

    conn.commit()
    conn.close()


def log_signal(data: dict):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO signals (
            timestamp, symbol, direction, strategy, ai_score,
            confluence_count, was_traded, skip_reason, session, market_regime
        ) VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.now().isoformat(), data['symbol'], data.get('direction'),
        data.get('strategy'), data.get('ai_score'), data.get('confluence_count'),
        int(data.get('was_traded', False)), data.get('skip_reason'),
        data.get('session'), data.get('market_regime')
    ))
    conn.commit()
    conn.close()


def log_market_snapshot(data: dict):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO market_snapshots (
            timestamp, fear_greed, vix, dxy, gold_price,
            oil_price, sp500, bond_yield_10y, cot_net_pos, news_sentiment
        ) VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.now().isoformat(),
        data.get('fear_greed'), data.get('vix'), data.get('dxy'),
        data.get('gold_price'), data.get('oil_price'), data.get('sp500'),
        data.get('bond_yield_10y'), str(data.get('cot_net_pos', {})),
        data.get('news_sentiment')
    ))
    conn.commit()
    conn.close()
