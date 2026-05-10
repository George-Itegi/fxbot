"""
Trade Logger
=============
Structured trade logging for analysis and debugging.
Logs every trade with full context to a separate file.
"""

import json
import time
from pathlib import Path
from typing import Optional

from config import LOG_DIR
from trading.signal_generator import Signal
from trading.execution_engine import OrderResult
from utils.logger import setup_logger

logger = setup_logger("trading.trade_logger")


class TradeLogger:
    """
    Logs every trade signal, risk decision, and execution result
    to a structured JSONL file for post-analysis.
    """
    
    def __init__(self, log_dir: str = str(LOG_DIR)):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create daily log file
        today = time.strftime("%Y%m%d")
        self.log_file = self.log_dir / f"trades_{today}.jsonl"
        self._trade_count = 0
    
    def log_signal(self, signal: Signal):
        """Log a generated signal."""
        entry = {
            "type": "signal",
            "timestamp": signal.timestamp,
            "direction": signal.direction,
            "barrier": signal.barrier,
            "confidence": signal.confidence,
            "expected_value": signal.expected_value,
            "stake": signal.stake,
            "reason": signal.reason,
        }
        self._write(entry)
    
    def log_risk_decision(self, signal: Signal, decision):
        """Log risk check result."""
        entry = {
            "type": "risk_decision",
            "timestamp": time.time(),
            "approved": decision.approved,
            "reason": decision.reason,
            "adjusted_stake": decision.adjusted_stake,
            "checks": decision.checks,
            "signal_direction": signal.direction,
            "signal_barrier": signal.barrier,
        }
        self._write(entry)
    
    def log_execution(self, signal: Signal, result: OrderResult):
        """Log order execution result."""
        entry = {
            "type": "execution",
            "timestamp": result.timestamp,
            "success": result.success,
            "is_paper": result.is_paper,
            "direction": result.direction,
            "barrier": result.barrier,
            "stake": result.stake,
            "payout": result.payout,
            "buy_price": result.buy_price,
            "contract_id": result.contract_id,
            "error": result.error,
            "paper_outcome": result.paper_outcome,
            "signal_confidence": signal.confidence,
            "signal_ev": signal.expected_value,
        }
        self._write(entry)
    
    def log_outcome(self, signal: Signal, won: bool, payout: float,
                    bankroll_after: float):
        """Log final trade outcome with P&L."""
        pnl = payout - signal.stake if won else -signal.stake
        entry = {
            "type": "outcome",
            "timestamp": time.time(),
            "direction": signal.direction,
            "barrier": signal.barrier,
            "won": won,
            "stake": signal.stake,
            "payout": payout,
            "pnl": pnl,
            "bankroll_after": bankroll_after,
            "signal_confidence": signal.confidence,
        }
        self._write(entry)
    
    def log_model_update(self, features: dict, outcome: int, prediction_before: float):
        """Log model learning event."""
        entry = {
            "type": "model_update",
            "timestamp": time.time(),
            "outcome": outcome,
            "prediction_before": prediction_before,
            "correct": (prediction_before > 0.5) == bool(outcome),
        }
        self._write(entry)
    
    def _write(self, entry: dict):
        """Append entry to JSONL file."""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            self._trade_count += 1
        except Exception as e:
            logger.error(f"Failed to write trade log: {e}")
    
    @property
    def trade_count(self) -> int:
        return self._trade_count
