"""
Performance Metrics Tracker
============================
Tracks P&L, win rate, drawdowns, Sharpe ratio, and all key trading metrics.
Persists to MySQL (XAMPP apex_trader database).
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from utils.db import execute_query
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
    Uses MySQL via connection pool for thread-safe persistence.
    """
    
    def __init__(self):
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
        execute_query("""
            CREATE TABLE IF NOT EXISTS deriv_trades (
                trade_id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp DOUBLE NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                direction VARCHAR(20) NOT NULL,
                barrier INT NOT NULL,
                confidence DOUBLE,
                expected_value DOUBLE,
                stake DOUBLE NOT NULL,
                payout DOUBLE DEFAULT 0,
                won TINYINT NOT NULL,
                balance_after DOUBLE,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_symbol (symbol),
                INDEX idx_direction (direction),
                INDEX idx_won (won),
                INDEX idx_timestamp (timestamp)
            )
        """)
        execute_query("""
            CREATE TABLE IF NOT EXISTS deriv_daily_summary (
                date VARCHAR(10) PRIMARY KEY,
                trades INT,
                wins INT,
                losses INT,
                pnl DOUBLE,
                max_drawdown DOUBLE,
                bankroll DOUBLE,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
        """)
        logger.info("MySQL tables initialized (deriv_trades, deriv_daily_summary)")
    
    def _load_state(self):
        """Load cumulative state from database."""
        try:
            row = execute_query(
                "SELECT COUNT(*) AS cnt, "
                "COALESCE(SUM(CASE WHEN won=1 THEN 1 ELSE 0 END), 0) AS wins, "
                "COALESCE(SUM(CASE WHEN won=0 THEN 1 ELSE 0 END), 0) AS losses, "
                "COALESCE(SUM(CASE WHEN won=1 THEN payout ELSE -stake END), 0) AS pnl "
                "FROM deriv_trades",
                fetch="one"
            )
            
            if row and row["cnt"] > 0:
                self.trade_count = row["cnt"]
                self.wins = row["wins"] or 0
                self.losses = row["losses"] or 0
                self.total_pnl = row["pnl"] or 0.0
            
            logger.info(f"Loaded state: {self.trade_count} trades, "
                         f"{self.wins}W/{self.losses}L, P&L: ${self.total_pnl:.2f}")
        except Exception as e:
            logger.warning(f"Could not load state from DB (tables may be empty): {e}")
    
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
        
        # Persist to MySQL
        execute_query(
            """INSERT INTO deriv_trades 
               (timestamp, symbol, direction, barrier, confidence,
                expected_value, stake, payout, won, balance_after, notes)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (trade.timestamp, trade.symbol, trade.direction, trade.barrier,
             trade.confidence, trade.expected_value, trade.stake, trade.payout,
             int(trade.won), trade.balance_after, trade.notes)
        )
        
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
        """Save daily summary to DB (upsert)."""
        execute_query(
            """INSERT INTO deriv_daily_summary 
               (date, trades, wins, losses, pnl, max_drawdown, bankroll)
               VALUES (%s, %s, %s, %s, %s, %s, %s)
               ON DUPLICATE KEY UPDATE
               trades=VALUES(trades), wins=VALUES(wins), losses=VALUES(losses),
               pnl=VALUES(pnl), max_drawdown=VALUES(max_drawdown), bankroll=VALUES(bankroll)""",
            (date, self.daily_trades, self.wins, self.losses,
             self.daily_pnl, self.max_drawdown, self.peak_balance)
        )
    
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
        row = execute_query(
            "SELECT COALESCE(SUM(CASE WHEN won=1 THEN payout ELSE 0 END), 0) AS gross_profit, "
            "COALESCE(SUM(CASE WHEN won=0 THEN stake ELSE 0 END), 0) AS gross_loss "
            "FROM deriv_trades",
            fetch="one"
        )
        if not row:
            return 0.0
        gross_profit = row["gross_profit"] or 0
        gross_loss = row["gross_loss"] or 0
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    @property
    def avg_win(self) -> float:
        if self.wins == 0:
            return 0.0
        row = execute_query(
            "SELECT AVG(payout) AS avg_val FROM deriv_trades WHERE won=1",
            fetch="one"
        )
        return (row["avg_val"] or 0.0) if row else 0.0
    
    @property
    def avg_loss(self) -> float:
        if self.losses == 0:
            return 0.0
        row = execute_query(
            "SELECT AVG(stake) AS avg_val FROM deriv_trades WHERE won=0",
            fetch="one"
        )
        return (row["avg_val"] or 0.0) if row else 0.0
    
    def get_recent_trades(self, limit: int = 20) -> list:
        """Get most recent N trades."""
        rows = execute_query(
            "SELECT * FROM deriv_trades ORDER BY trade_id DESC LIMIT %s",
            (limit,),
            fetch="all"
        )
        return rows or []
    
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
