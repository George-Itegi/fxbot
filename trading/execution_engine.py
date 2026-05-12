"""
Execution Engine (v4 — Live Only)
===================================
Handles order placement via Deriv API.
NO paper/simulation mode — all trades go through the real Deriv API
using your demo account balance.

Flow:
1. Receive approved signal + risk decision
2. Get proposal from Deriv (payout + stake details)
3. Buy contract via API
4. Contract settles via WebSocket callback → model learns from result
"""

import time
import asyncio
from dataclasses import dataclass
from typing import Optional

from config import DEFAULT_SYMBOL
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
    error: str = ""


class ExecutionEngine:
    """
    Places LIVE trades via Deriv API on your demo account.
    
    Flow:
    1. Receive approved signal + risk decision
    2. Get proposal (check payout is still valid)
    3. Buy contract
    4. Store signal for outcome tracking (settled via WS callback)
    5. Return OrderResult
    
    Contract results come back asynchronously via the
    `on_trade_result` WebSocket callback, which calls
    `handle_contract_result()` to match the signal and process P&L.
    """
    
    def __init__(self, deriv_ws=None, symbol: str = None, duration_unit: str = "t"):
        self.ws = deriv_ws
        self.symbol = symbol or DEFAULT_SYMBOL
        self.duration_unit = duration_unit
        self.pending_contracts: dict[int, Signal] = {}  # contract_id → signal
        self._total_orders = 0
        
        logger.info(
            f"ExecutionEngine initialized: LIVE mode, "
            f"symbol={self.symbol}, duration_unit={self.duration_unit}"
        )
    
    async def execute(self, signal: Signal, risk_decision: RiskDecision) -> OrderResult:
        """
        Execute a LIVE trade via Deriv API.
        
        Args:
            signal: The trade signal to execute
            risk_decision: Approved risk decision with adjusted stake
        
        Returns:
            OrderResult with trade details (contract is OPEN, not settled yet)
        """
        if not risk_decision.approved:
            return OrderResult(success=False, error="Risk check failed")
        
        if not self.ws or not self.ws.is_connected:
            logger.error("WebSocket not connected — cannot place trade")
            return OrderResult(success=False, error="WebSocket not connected")
        
        stake = risk_decision.adjusted_stake
        self._total_orders += 1
        
        # Step 1: Get proposal (validates payout, stake, duration)
        proposal = await self.ws.get_proposal(
            symbol=self.symbol,
            contract_type=signal.direction,
            barrier=signal.barrier,
            stake=stake,
            duration=signal.contract_duration,
            duration_unit=self.duration_unit,
        )
        
        if not proposal:
            return OrderResult(success=False, error="Failed to get proposal from Deriv")
        
        actual_payout = proposal["payout"]
        actual_stake = proposal["stake"]
        
        # Step 2: Buy the contract
        buy_result = await self.ws.buy_contract(
            proposal_id=proposal["id"],
            stake=actual_stake,
        )
        
        if not buy_result:
            return OrderResult(success=False, error="Failed to buy contract on Deriv")
        
        contract_id = buy_result["contract_id"]
        
        # Store signal for outcome tracking (will be matched when WS sends result)
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
        )
        
        logger.info(
            f"TRADE PLACED: {signal.direction} barrier={signal.barrier} "
            f"stake=${actual_stake:.2f} payout=${actual_payout:.2f} "
            f"dur={signal.contract_duration}{self.duration_unit} "
            f"contract={contract_id}"
        )
        
        return result
    
    def handle_contract_result(self, contract_data: dict) -> Optional[tuple]:
        """
        Process a completed contract notification from WebSocket.
        
        Called when Deriv sends a proposal_open_contract with is_sold=true.
        Matches the contract_id to the stored signal and returns the result.
        
        Args:
            contract_data: Contract result from Deriv WebSocket
        
        Returns:
            Tuple of (Signal, won_bool, total_payout) or None if not our contract
        """
        contract_id = contract_data.get("contract_id")
        if contract_id not in self.pending_contracts:
            return None
        
        signal = self.pending_contracts.pop(contract_id)
        
        won = contract_data.get("profit", 0) > 0
        profit = abs(contract_data.get("profit", 0))
        buy_price = contract_data.get("buy_price", signal.stake)
        
        # Total payout = buy_price + profit (for wins), 0 (for losses)
        total_payout = buy_price + profit if won else 0
        
        status = "WIN" if won else "LOSS"
        settle_digit = contract_data.get("contract_type", "")
        
        logger.info(
            f"{'GREEN' if won else 'RED'} CONTRACT {status}: "
            f"{signal.direction} barrier={signal.barrier} "
            f"profit={'+' if won else '-'}${profit:.2f} "
            f"dur={signal.contract_duration}{self.duration_unit}"
        )
        
        return (signal, won, total_payout)
    
    def set_ws(self, deriv_ws):
        """Set WebSocket connection (for deferred initialization)."""
        self.ws = deriv_ws
    
    def summary(self) -> dict:
        return {
            "mode": "live",
            "total_orders": self._total_orders,
            "pending_contracts": len(self.pending_contracts),
        }
