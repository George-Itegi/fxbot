"""
Deriv WebSocket Connection
============================
Persistent WebSocket connection to Deriv API with:
- Automatic reconnection with exponential backoff
- Message routing (ticks, proposals, contracts)
- Heartbeat keepalive
- Demo account support (token-based)
"""

import asyncio
import json
import time
import uuid
from typing import Callable, Optional, Awaitable

import websockets

from config import DERIV_WS_URL, DERIV_API_TOKEN
from utils.logger import setup_logger

logger = setup_logger("data.deriv_ws")


class DerivWS:
    """
    Async WebSocket client for Deriv API.
    
    Usage:
        ws = DerivWS()
        ws.on_tick = my_tick_handler
        ws.on_trade_result = my_trade_handler
        await ws.connect()
        await ws.subscribe_ticks("R_100")
        # ... later
        await ws.disconnect()
    """
    
    def __init__(self):
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 20
        self._reconnect_delay_base = 1.0  # seconds
        self._heartbeat_interval = 30  # seconds
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._recv_task: Optional[asyncio.Task] = None
        self._pending_requests: dict = {}  # req_id -> asyncio.Future
        
        # Subscriptions we want to restore after reconnect
        self._subscriptions: dict = {}
        
        # ─── Callbacks ───
        # Set these before connecting
        self.on_tick: Optional[Callable] = None           # (symbol, tick_data)
        self.on_trade_result: Optional[Callable] = None   # (contract_data)
        self.on_proposal: Optional[Callable] = None       # (proposal_data)
        self.on_candles: Optional[Callable] = None        # (symbol, candles_data)
        self.on_error: Optional[Callable] = None          # (error_msg)
        self.on_disconnect: Optional[Callable] = None     # ()
        self.on_reconnect: Optional[Callable] = None      # ()
    
    async def connect(self):
        """Connect to Deriv WebSocket with reconnection."""
        while self._reconnect_attempts < self._max_reconnect_attempts:
            try:
                logger.info(f"Connecting to Deriv WS (attempt {self._reconnect_attempts + 1})...")
                self._ws = await websockets.connect(
                    DERIV_WS_URL,
                    ping_interval=None,  # We handle keepalive ourselves
                    ping_timeout=None,
                    close_timeout=5,
                )
                self._connected = True
                self._reconnect_attempts = 0
                
                # Authorize if token available
                if DERIV_API_TOKEN:
                    await self._authorize()
                
                # Start heartbeat and message receiver
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                self._recv_task = asyncio.create_task(self._recv_loop())
                
                # Restore subscriptions
                await self._restore_subscriptions()
                
                logger.info("Connected to Deriv WebSocket")
                return True
                
            except Exception as e:
                self._reconnect_attempts += 1
                delay = min(
                    self._reconnect_delay_base * (2 ** self._reconnect_attempts),
                    60  # Cap at 60 seconds
                )
                logger.warning(f"Connection failed: {e}. Retrying in {delay:.1f}s...")
                self._connected = False
                await asyncio.sleep(delay)
        
        logger.error("Max reconnection attempts reached")
        if self.on_error:
            self.on_error("Failed to connect after max attempts")
        return False
    
    async def disconnect(self):
        """Clean disconnect."""
        self._connected = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._recv_task:
            self._recv_task.cancel()
        if self._ws:
            await self._ws.close()
        logger.info("Disconnected from Deriv WebSocket")
    
    async def _authorize(self):
        """Authorize with API token."""
        response = await self._send({"authorize": DERIV_API_TOKEN})
        if response.get("error"):
            logger.error(f"Authorization failed: {response['error']['message']}")
        else:
            auth = response.get("authorize", {})
            logger.info(
                f"Authorized: {auth.get('fullname', 'Unknown')} | "
                f"Balance: ${auth.get('balance', 0):.2f} {auth.get('currency', 'USD')} | "
                f"Account: {auth.get('loginid', 'Unknown')}"
            )
    
    async def subscribe_ticks(self, symbol: str):
        """Subscribe to real-time tick stream for a symbol."""
        self._subscriptions["ticks"] = {"ticks": symbol, "subscribe": 1}
        await self._send({"ticks": symbol, "subscribe": 1})
        logger.info(f"Subscribed to ticks: {symbol}")
    
    async def unsubscribe_ticks(self, symbol: str):
        """Unsubscribe from tick stream."""
        await self._send({"ticks": symbol, "subscribe": 0})
        self._subscriptions.pop("ticks", None)
        logger.info(f"Unsubscribed from ticks: {symbol}")
    
    async def get_tick_history(self, symbol: str, count: int = 1000) -> list:
        """
        Fetch historical tick data for warmup/analysis.
        Returns list of tick dicts with 'epoch', 'quote', 'symbol'.
        """
        response = await self._send({
            "ticks_history": symbol,
            "count": count,
            "end": "latest",
            "style": "ticks",
            "subscribe": 0,
        })
        
        if response.get("error"):
            logger.error(f"Tick history error: {response['error']['message']}")
            return []
        
        history = response.get("history", {})
        prices = history.get("prices", [])
        times = history.get("times", [])
        
        ticks = []
        for i, (t, p) in enumerate(zip(times, prices)):
            ticks.append({
                "epoch": t,
                "quote": float(p),
                "symbol": symbol,
                "tick_index": i,
            })
        
        logger.info(f"Fetched {len(ticks)} historical ticks for {symbol}")
        return ticks
    
    async def get_candle_history(self, symbol: str, granularity: int = 60, 
                                  count: int = 500) -> list:
        """
        Fetch historical candle data.
        granularity: seconds per candle (60=1min, 300=5min, etc.)
        """
        response = await self._send({
            "ticks_history": symbol,
            "count": count,
            "end": "latest",
            "style": "candles",
            "granularity": granularity,
            "subscribe": 0,
        })
        
        if response.get("error"):
            logger.error(f"Candle history error: {response['error']['message']}")
            return []
        
        candles = response.get("candles", [])
        logger.info(f"Fetched {len(candles)} candles for {symbol}")
        return candles
    
    async def get_proposal(self, symbol: str, contract_type: str, barrier: int,
                           stake: float, duration: int = 5) -> Optional[dict]:
        """
        Get contract proposal (payout details) before buying.
        Always check payout before placing a trade.
        """
        response = await self._send({
            "proposal": 1,
            "amount": stake,
            "basis": "stake",
            "contract_type": contract_type,
            "currency": "USD",
            "symbol": symbol,
            "duration": duration,
            "duration_unit": "t",
            "barrier": str(barrier),
        })
        
        if response.get("error"):
            logger.warning(f"Proposal error: {response['error']['message']}")
            return None
        
        proposal = response.get("proposal", {})
        return {
            "id": proposal.get("id"),
            "payout": float(proposal.get("payout", 0)),
            "stake": float(proposal.get("ask_price", stake)),
            "spot": float(proposal.get("spot_value", 0)),
            "spot_time": proposal.get("spot_time"),
        }
    
    async def buy_contract(self, proposal_id: str, stake: float = None) -> Optional[dict]:
        """
        Buy a contract using a previously obtained proposal ID.
        """
        buy_params = {"buy": proposal_id, "price": stake}
        response = await self._send(buy_params)
        
        if response.get("error"):
            logger.error(f"Buy error: {response['error']['message']}")
            return None
        
        buy = response.get("buy", {})
        contract_id = buy.get("contract_id")
        
        # Subscribe to contract updates
        await self._send({
            "proposal_open_contract": 1,
            "contract_id": contract_id,
            "subscribe": 1,
        })
        
        logger.info(f"Contract bought: ID={contract_id}, "
                     f"buy_price=${buy.get('buy_price', 0):.2f}")
        
        return {
            "contract_id": contract_id,
            "buy_price": float(buy.get("buy_price", 0)),
            "payout": float(buy.get("payout", 0)),
            "longcode": buy.get("longcode", ""),
            "start_time": buy.get("start_time"),
        }
    
    async def subscribe_to_balance(self):
        """Subscribe to real-time balance updates."""
        self._subscriptions["balance"] = {"balance": 1, "subscribe": 1}
        await self._send({"balance": 1, "subscribe": 1})
    
    async def get_account_summary(self) -> Optional[dict]:
        """Get current account balance and details."""
        response = await self._send({"statement": 1, "description": 1, "limit": 1})
        if response.get("error"):
            logger.error(f"Account summary error: {response['error']['message']}")
            return None
        return response
    
    # ─── Internal Methods ───
    
    async def _send(self, data: dict) -> dict:
        """Send request and wait for response."""
        if not self._ws or not self._connected:
            raise ConnectionError("WebSocket not connected")
        
        req_id = data.get("req_id") or str(uuid.uuid4())[:8]
        data["req_id"] = req_id
        
        future = asyncio.get_event_loop().create_future()
        self._pending_requests[req_id] = future
        
        try:
            await self._ws.send(json.dumps(data))
            # Wait for response with timeout
            result = await asyncio.wait_for(future, timeout=10.0)
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Request timed out: {data}")
            return {"error": {"message": "Request timeout"}}
        finally:
            self._pending_requests.pop(req_id, None)
    
    async def _recv_loop(self):
        """Main message receiver loop."""
        try:
            async for raw_message in self._ws:
                try:
                    data = json.loads(raw_message)
                    await self._route_message(data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON: {raw_message}")
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self._connected = False
            if self.on_disconnect:
                self.on_disconnect()
            # Attempt reconnect
            asyncio.create_task(self._auto_reconnect())
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Recv loop error: {e}")
    
    async def _route_message(self, data: dict):
        """Route incoming messages to appropriate handlers."""
        msg_type = data.get("msg_type", "")
        
        # Handle pending request responses
        req_id = data.get("req_id")
        if req_id and req_id in self._pending_requests:
            future = self._pending_requests[req_id]
            if not future.done():
                future.set_result(data)
        
        # Route by message type
        if msg_type == "tick":
            tick = data.get("tick", {})
            symbol = tick.get("symbol", "")
            if self.on_tick:
                if asyncio.iscoroutinefunction(self.on_tick):
                    await self.on_tick(symbol, tick)
                else:
                    self.on_tick(symbol, tick)
        
        elif msg_type == "proposal_open_contract":
            contract = data.get("proposal_open_contract", {})
            if contract.get("is_sold") or contract.get("status") == "sold":
                if self.on_trade_result:
                    if asyncio.iscoroutinefunction(self.on_trade_result):
                        await self.on_trade_result(contract)
                    else:
                        self.on_trade_result(contract)
        
        elif msg_type == "proposal":
            if self.on_proposal:
                if asyncio.iscoroutinefunction(self.on_proposal):
                    await self.on_proposal(data.get("proposal", {}))
                else:
                    self.on_proposal(data.get("proposal", {}))
        
        elif msg_type == "balance":
            balance = data.get("balance", {})
            logger.debug(f"Balance update: ${balance.get('balance', 0):.2f}")
        
        elif msg_type == "error":
            error = data.get("error", {})
            logger.error(f"API error: {error.get('message', 'Unknown')} "
                         f"(code: {error.get('code', 'N/A')})")
            if self.on_error:
                self.on_error(error.get("message", "Unknown API error"))
    
    async def _heartbeat_loop(self):
        """Send periodic ping to keep connection alive."""
        try:
            while self._connected:
                await asyncio.sleep(self._heartbeat_interval)
                if self._ws and self._connected:
                    # Deriv doesn't use standard WS ping, but we need activity
                    # Send a lightweight request as keepalive
                    await self._ws.send(json.dumps({"ping": 1}))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"Heartbeat error: {e}")
    
    async def _auto_reconnect(self):
        """Automatically reconnect and restore state."""
        if await self.connect():
            logger.info("Reconnected successfully")
            if self.on_reconnect:
                self.on_reconnect()
        else:
            logger.error("Auto-reconnect failed")
    
    async def _restore_subscriptions(self):
        """Re-subscribe to all active subscriptions after reconnect."""
        for name, sub in self._subscriptions.items():
            try:
                await self._send(sub)
                logger.info(f"Restored subscription: {name}")
            except Exception as e:
                logger.warning(f"Failed to restore {name}: {e}")
    
    @property
    def is_connected(self) -> bool:
        return self._connected
