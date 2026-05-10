"""
Performance Metrics Tracker
============================
Tracks P&L, win rate, drawdowns, Sharpe ratio, and all key trading metrics.
Persists to SQLite for historical analysis.
"""

import sqlite3
import time
from dataclasses import dataclass, field
from typing import Optional

from config import DB_PATH
from utils.logger import setup_logger

logger = setup_logger("utils.metrics")


@dataclass
class TradeRecord:
    """Single trade record."""
    trade_id: int
    timestamp: float
    symbol: str
    direction: str        # DIGITOVER or DIGITUNDER
    barrier: int
    confidence: float
    expected_value: float
    stake: float
    payout: float         # Actual payout received (0 if lost)
    won: bool
    balance_after: float
    notes: str = ""


class PerformanceTracker:
    """
    Tracks all trading performance metrics.
    Thread-safe via SQLite WAL mode.
    """
    
    def __init__(self, db_path: str = str(DB_PATH)):
        self.db_path = db_path
        self._init_db()
        
        # Runtime state
        self.trade_count = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.peak_balance = 0.0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0
        self.consecutive_wins_max = 0
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self._last_day = time.strftime("%Y-%m-%d")
        
        # Load from DB
        self._load_state()
    
    def _init_db(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    barrier INTEGER NOT NULL,
                    confidence REAL,
                    expected_value REAL,
                    stake REAL NOT NULL,
                    payout REAL DEFAULT 0,
                    won INTEGER NOT NULL,
                    balance_after REAL,
                    notes TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_summary (
                    date TEXT PRIMARY KEY,
                    trades INTEGER,
                    wins INTEGER,
                    losses INTEGER,
                    pnl REAL,
                    max_drawdown REAL,
                    bankroll REAL
                )
            """)
            conn.commit()
        logger.info(f"Database initialized: {self.db_path}")
    
    def _load_state(self):
        """Load cumulative state from database."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(*), SUM(CASE WHEN won=1 THEN 1 ELSE 0 END), "
                               "SUM(CASE WHEN won=0 THEN 1 ELSE 0 END), "
                               "SUM(CASE WHEN won=1 THEN payout ELSE -stake END) "
                               "FROM trades").fetchone()
            
            if row and row[0] > 0:
                self.trade_count = row[0]
                self.wins = row[1] or 0
                self.losses = row[2] or 0
                self.total_pnl = row[3] or 0.0
        
        logger.info(f"Loaded state: {self.trade_count} trades, "
                     f"{self.wins}W/{self.losses}L, P&L: ${self.total_pnl:.2f}")
    
    def record_trade(self, trade: TradeRecord):
        """Record a completed trade."""
        # Update runtime counters
        self.trade_count += 1
        self.daily_trades += 1
        
        if trade.won:
            self.wins += 1
            self.consecutive_losses = 0
            self.consecutive_wins += 1
            self.consecutive_wins_max = max(self.consecutive_wins_max, self.consecutive_wins)
            self.daily_pnl += trade.payout - trade.stake
            self.total_pnl += trade.payout - trade.stake
        else:
            self.losses += 1
            self.consecutive_wins = 0
            self.consecutive_losses += 1
            self.max_consecutive_losses = max(
                self.max_consecutive_losses, self.consecutive_losses
            )
            self.daily_pnl -= trade.stake
            self.total_pnl -= trade.stake
        
        # Update peak and drawdown
        self.peak_balance = max(self.peak_balance, trade.balance_after)
        if self.peak_balance > 0:
            self.current_drawdown = (self.peak_balance - trade.balance_after) / self.peak_balance
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        # Persist to DB
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO trades (timestamp, symbol, direction, barrier, confidence,
                    expected_value, stake, payout, won, balance_after, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (trade.timestamp, trade.symbol, trade.direction, trade.barrier,
                  trade.confidence, trade.expected_value, trade.stake, trade.payout,
                  int(trade.won), trade.balance_after, trade.notes))
            conn.commit()
        
        # Check for day rollover
        today = time.strftime("%Y-%m-%d")
        if today != self._last_day:
            self._save_daily_summary(self._last_day)
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self._last_day = today
        
        logger.info(
            f"Trade #{trade.trade_id}: {trade.direction} {trade.symbol} "
            f"barrier={trade.barrier} | {'WIN' if trade.won else 'LOSS'} | "
            f"stake=${trade.stake:.2f} | balance=${trade.balance_after:.2f}"
        )
    
    def _save_daily_summary(self, date: str):
        """Save daily summary to DB."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO daily_summary (date, trades, wins, losses, 
                    pnl, max_drawdown, bankroll)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (date, self.daily_trades, self.wins, self.losses,
                  self.daily_pnl, self.max_drawdown, self.peak_balance))
            conn.commit()
    
    @property
    def win_rate(self) -> float:
        if self.trade_count == 0:
            return 0.0
        return self.wins / self.trade_count
    
    @property
    def profit_factor(self) -> float:
        """Ratio of gross profit to gross loss."""
        if self.trade_count == 0:
            return 0.0
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT SUM(CASE WHEN won=1 THEN payout ELSE 0 END), "
                "SUM(CASE WHEN won=0 THEN stake ELSE 0 END) FROM trades"
            ).fetchone()
            gross_profit = row[0] or 0
            gross_loss = row[1] or 0
            return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    @property
    def avg_win(self) -> float:
        if self.wins == 0:
            return 0.0
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT AVG(payout) FROM trades WHERE won=1"
            ).fetchone()
            return row[0] or 0.0
    
    @property
    def avg_loss(self) -> float:
        if self.losses == 0:
            return 0.0
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT AVG(stake) FROM trades WHERE won=0"
            ).fetchone()
            return row[0] or 0.0
    
    def get_recent_trades(self, limit: int = 20) -> list:
        """Get most recent N trades."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY trade_id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(row) for row in rows]
    
    def summary(self) -> dict:
        """Get full performance summary."""
        return {
            "total_trades": self.trade_count,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate * 100, 1),
            "total_pnl": round(self.total_pnl, 2),
            "max_drawdown": round(self.max_drawdown * 100, 1),
            "profit_factor": round(self.profit_factor, 2),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "max_consecutive_losses": self.max_consecutive_losses,
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_trades": self.daily_trades,
        }
