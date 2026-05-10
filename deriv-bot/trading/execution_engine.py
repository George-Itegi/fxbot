"""
Execution Engine
=================
Handles order placement via Deriv API.
Supports both PAPER and LIVE trading modes.

In PAPER mode: logs what would happen without placing real orders.
In LIVE mode: places real orders via Deriv WebSocket.
"""

import time
from dataclasses import dataclass
from typing import Optional, Callable, Awaitable

from config import TRADING_MODE, DEFAULT_SYMBOL, CONTRACT_DURATION
from trading.signal_generator import Signal
from trading.risk_manager import RiskDecision
from utils.logger import setup_logger

logger = setup_logger("trading.execution_engine")


@dataclass
class OrderResult:
    """Result of an order execution."""
    success: bool
    order_id: Optional[str] = None
    contract_id: Optional[int] = None
    buy_price: float = 0.0
    payout: float = 0.0
    direction: str = ""
    barrier: int = 0
    stake: float = 0.0
    timestamp: float = 0.0
    is_paper: bool = False
    paper_outcome: Optional[bool] = None  # For paper trading
    error: str = ""


class ExecutionEngine:
    """
    Places trades via Deriv API or simulates in paper mode.
    
    Flow:
    1. Receive approved signal + risk decision
    2. If PAPER: simulate trade, determine random outcome
    3. If LIVE: get proposal → buy contract → track result
    4. Return OrderResult
    """
    
    def __init__(self, deriv_ws=None):
        self.ws = deriv_ws
        self.mode = TRADING_MODE
        self.pending_contracts: dict[int, Signal] = {}  # contract_id → signal
        self._total_orders = 0
        self._total_paper_orders = 0
        
        logger.info(f"ExecutionEngine initialized: mode={self.mode}")
        
        if self.mode == "live":
            logger.warning("⚠️  LIVE TRADING MODE — REAL MONEY AT RISK")
        else:
            logger.info("📝 PAPER TRADING MODE — no real orders placed")
    
    async def execute(self, signal: Signal, risk_decision: RiskDecision) -> OrderResult:
        """
        Execute a trade.
        
        Args:
            signal: The trade signal to execute
            risk_decision: Approved risk decision with adjusted stake
        
        Returns:
            OrderResult with trade details
        """
        if not risk_decision.approved:
            return OrderResult(success=False, error="Risk check failed")
        
        stake = risk_decision.adjusted_stake
        
        if self.mode == "paper":
            return await self._paper_trade(signal, stake)
        elif self.mode == "live":
            return await self._live_trade(signal, stake)
        else:
            return OrderResult(success=False, error=f"Unknown mode: {self.mode}")
    
    async def _paper_trade(self, signal: Signal, stake: float) -> OrderResult:
        """
        Simulate a trade without placing real orders.
        Uses the model's confidence as a rough probability of winning.
        """
        import random
        
        self._total_paper_orders += 1
        
        # Simulate outcome based on confidence (rough approximation)
        # In reality, the actual digit distribution determines this
        won = random.random() < signal.confidence
        
        # Simulate payout (typically 70-90% on Deriv)
        simulated_payout = stake * random.uniform(0.70, 0.90)
        
        result = OrderResult(
            success=True,
            order_id=f"PAPER-{self._total_paper_orders:06d}",
            contract_id=None,
            buy_price=stake,
            payout=simulated_payout if won else 0,
            direction=signal.direction,
            barrier=signal.barrier,
            stake=stake,
            timestamp=time.time(),
            is_paper=True,
            paper_outcome=won,
        )
        
        status = "WIN" if won else "LOSS"
        logger.info(
            f"📝 PAPER [{status}]: {signal.direction} barrier={signal.barrier} "
            f"stake=${stake:.2f} conf={signal.confidence:.2%}"
        )
        
        return result
    
    async def _live_trade(self, signal: Signal, stake: float) -> OrderResult:
        """
        Place a real order via Deriv API.
        1. Get proposal (check payout)
        2. Buy contract
        3. Store signal for outcome tracking
        """
        if not self.ws or not self.ws.is_connected:
            logger.error("WebSocket not connected — cannot place live trade")
            return OrderResult(success=False, error="WebSocket not connected")
        
        self._total_orders += 1
        
        # Step 1: Get proposal
        proposal = await self.ws.get_proposal(
            symbol=DEFAULT_SYMBOL,
            contract_type=signal.direction,
            barrier=signal.barrier,
            stake=stake,
            duration=CONTRACT_DURATION,
        )
        
        if not proposal:
            return OrderResult(success=False, error="Failed to get proposal")
        
        actual_payout = proposal["payout"]
        actual_stake = proposal["stake"]
        
        # Step 2: Buy contract
        buy_result = await self.ws.buy_contract(
            proposal_id=proposal["id"],
            stake=actual_stake,
        )
        
        if not buy_result:
            return OrderResult(success=False, error="Failed to buy contract")
        
        contract_id = buy_result["contract_id"]
        
        # Store signal for outcome tracking
        self.pending_contracts[contract_id] = signal
        
        result = OrderResult(
            success=True,
            order_id=buy_result.get("longcode", ""),
            contract_id=contract_id,
            buy_price=actual_stake,
            payout=actual_payout,
            direction=signal.direction,
            barrier=signal.barrier,
            stake=actual_stake,
            timestamp=time.time(),
            is_paper=False,
        )
        
        logger.info(
            f"💰 LIVE: {signal.direction} barrier={signal.barrier} "
            f"stake=${actual_stake:.2f} payout=${actual_payout:.2f} "
            f"contract={contract_id}"
        )
        
        return result
    
    def handle_contract_result(self, contract_data: dict) -> Optional[tuple]:
        """
        Process a completed contract notification.
        
        Args:
            contract_data: Contract result from Deriv WebSocket
        
        Returns:
            Tuple of (Signal, won_bool, payout) or None
        """
        contract_id = contract_data.get("contract_id")
        if contract_id not in self.pending_contracts:
            return None
        
        signal = self.pending_contracts.pop(contract_id)
        
        won = contract_data.get("profit", 0) > 0
        payout = abs(contract_data.get("profit", 0))
        buy_price = contract_data.get("buy_price", signal.stake)
        
        status = "WIN" if won else "LOSS"
        logger.info(
            f"{'🟢' if won else '🔴'} CONTRACT {status}: "
            f"{signal.direction} barrier={signal.barrier} "
            f"profit={'+' if won else '-'}${payout:.2f}"
        )
        
        return (signal, won, payout + buy_price if won else 0)
    
    def set_ws(self, deriv_ws):
        """Set WebSocket connection (for deferred initialization)."""
        self.ws = deriv_ws
    
    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "live_orders": self._total_orders,
            "paper_orders": self._total_paper_orders,
            "pending_contracts": len(self.pending_contracts),
        }
