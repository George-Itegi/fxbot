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
import random
import time
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
        
        # ─── Symbol metadata (decimal places per symbol) ───
        # Auto-populated from Deriv API. Maps symbol -> decimal_places
        self.symbol_decimals: dict[str, int] = {}
        
        # ─── Callbacks ───
        # Set these before connecting
        self.on_tick: Optional[Callable] = None           # (symbol, tick_data)
        self.on_trade_result: Optional[Callable] = None   # (contract_data)
        self.on_proposal: Optional[Callable] = None       # (proposal_data)
        self.on_candles: Optional[Callable] = None        # (symbol, candles_data)
        self.on_balance: Optional[Callable] = None        # (balance_data)
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
                
                # CRITICAL: Start recv loop BEFORE auth so responses get read
                self._recv_task = asyncio.create_task(self._recv_loop())
                
                # Authorize if token available
                if DERIV_API_TOKEN:
                    await self._authorize()
                
                # Start heartbeat
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                
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
        if not DERIV_API_TOKEN:
            logger.info("No API token set — running in anonymous demo mode")
            return
        response = await self._send({"authorize": DERIV_API_TOKEN}, timeout=30)
        if response.get("error"):
            logger.warning(f"Authorization failed: {response['error']['message']} — continuing in demo mode")
        else:
            auth = response.get("authorize", {})
            logger.info(
                f"Authorized: {auth.get('fullname', 'Unknown')} | "
                f"Balance: ${auth.get('balance', 0):.2f} {auth.get('currency', 'USD')} | "
                f"Account: {auth.get('loginid', 'Unknown')}"
            )
    
    async def subscribe_ticks(self, symbol: str, retries: int = 2):
        """Subscribe to real-time tick stream for a symbol with retry logic."""
        sub_key = f"ticks_{symbol}"
        self._subscriptions[sub_key] = {"ticks": symbol, "subscribe": 1}
        
        for attempt in range(1, retries + 2):  # 1 initial + retries
            response = await self._send(
                {"ticks": symbol, "subscribe": 1},
                timeout=20.0  # Longer timeout for subscriptions
            )
            if response.get("error"):
                if attempt <= retries:
                    delay = attempt * 3  # 3s, 6s backoff
                    logger.warning(
                        f"Ticks subscription FAILED for {symbol} (attempt {attempt}/{retries+1}): "
                        f"{response['error']}. Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Ticks subscription FAILED for {symbol} after {retries+1} attempts: "
                        f"{response['error']}"
                    )
            else:
                logger.info(f"Subscribed to ticks: {symbol}")
                return True
        return False
    
    async def unsubscribe_ticks(self, symbol: str):
        """Unsubscribe from tick stream for a specific symbol."""
        sub_key = f"ticks_{symbol}"
        await self._send({"forget_all": "ticks"})
        self._subscriptions.pop(sub_key, None)
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
        })
        
        if response.get("error"):
            logger.error(f"Candle history error: {response['error']['message']}")
            return []
        
        candles = response.get("candles", [])
        logger.info(f"Fetched {len(candles)} candles for {symbol}")
        return candles
    
    async def get_proposal(self, symbol: str, contract_type: str, barrier: int,
                           stake: float, duration: int = 5,
                           duration_unit: str = "t") -> Optional[dict]:
        """
        Get contract proposal (payout details) before buying.
        Always check payout before placing a trade.
        
        Args:
            duration_unit: "t" for ticks, "s" for seconds (1HZ symbols use "s")
        """
        response = await self._send({
            "proposal": 1,
            "amount": stake,
            "basis": "stake",
            "contract_type": contract_type,
            "currency": "USD",
            "symbol": symbol,
            "duration": duration,
            "duration_unit": duration_unit,
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
    
    async def get_active_symbols(self) -> dict[str, int]:
        """
        Fetch active symbols and their decimal places from Deriv API.
        
        Deriv's active_symbols response may include:
        - `decimal_places`: Direct field (sometimes missing for synthetics)
        - `pip`: Pip size (e.g., 0.01 = 2 dp, 0.001 = 3 dp)
        - `display_decimal_places`: Display precision
        
        We check all three fields and also use `pip` as a reliable fallback
        for synthetic indices where `decimal_places` is often absent.
        
        Returns:
            Dict mapping symbol -> decimal_places (e.g., {"R_100": 2, "R_75": 4})
        """
        response = await self._send({
            "active_symbols": "brief",
            "product_type": "basic",
        })
        
        if response.get("error"):
            logger.error(f"Active symbols error: {response['error']['message']}")
            return {}
        
        symbols_data = response.get("active_symbols", [])
        decimal_map = {}
        
        # Debug: log what fields the first symbol has
        if symbols_data:
            first_sym = symbols_data[0]
            available_fields = list(first_sym.keys())
            logger.debug(f"Active symbol fields sample: {available_fields}")
        
        for sym in symbols_data:
            symbol = sym.get("symbol", "")
            if not symbol:
                continue
            
            dp = None
            
            # Method 1: Direct decimal_places field
            dp_raw = sym.get("decimal_places")
            if dp_raw is not None:
                dp = int(dp_raw)
            
            # Method 2: Compute from pip size (e.g., pip=0.01 → 2 dp)
            if dp is None:
                pip = sym.get("pip")
                if pip is not None:
                    pip_val = float(pip)
                    if pip_val > 0:
                        import math
                        dp = max(0, round(-math.log10(pip_val)))
            
            # Method 3: display_decimal_places field
            if dp is None:
                dp_raw = sym.get("display_decimal_places")
                if dp_raw is not None:
                    dp = int(dp_raw)
            
            if dp is not None:
                decimal_map[symbol] = dp
        
        # Update internal cache
        self.symbol_decimals.update(decimal_map)
        
        # Log the ones we care about
        for sym_name in ["R_10", "R_25", "R_50", "R_75", "R_100",
                          "1HZ10V", "1HZ25V", "1HZ50V", "1HZ75V", "1HZ100V"]:
            if sym_name in decimal_map:
                logger.info(f"  {sym_name}: {decimal_map[sym_name]} decimal places (from API)")
        
        # Debug: if R_100 wasn't found, log what synthetic symbols we DID find
        if "R_100" not in decimal_map:
            synthetics = [s for s in decimal_map if s.startswith("R_") or s.startswith("1HZ")]
            if synthetics:
                logger.debug(f"Synthetic symbols found: {synthetics}")
            else:
                # Try to find what the API calls R_100
                r100_candidates = [s for s in symbols_data if "100" in str(s.get("symbol", "")) or "100" in str(s.get("display_name", ""))]
                if r100_candidates:
                    for c in r100_candidates[:3]:
                        logger.debug(f"R_100 candidate: symbol={c.get('symbol')}, name={c.get('display_name')}, pip={c.get('pip')}")
        
        logger.info(f"Fetched decimal places for {len(decimal_map)} symbols")
        return decimal_map
    
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
    
    async def _send(self, data: dict, timeout: float = 10.0) -> dict:
        """Send request and wait for response. Always uses req_id for reliable matching."""
        if not self._ws or not self._connected:
            raise ConnectionError("WebSocket not connected")
        
        # Always assign a req_id — Deriv echoes it back in the response
        req_id = data.get("req_id") or random.randint(100000, 999999)
        data["req_id"] = req_id
        
        future = asyncio.get_event_loop().create_future()
        self._pending_requests[req_id] = future
        
        try:
            await self._ws.send(json.dumps(data))
            # Wait for response with timeout
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Request timed out: {data}")
            return {"error": {"message": "Request timeout"}}
        finally:
            self._pending_requests.pop(req_id, None)
    
    async def _recv_loop(self):
        """Main message receiver loop."""
        msg_count = 0
        try:
            async for raw_message in self._ws:
                try:
                    data = json.loads(raw_message)
                    msg_count += 1
                    
                    # Log first 10 messages to diagnose routing
                    if msg_count <= 10:
                        msg_type = data.get("msg_type", "NO_TYPE")
                        req_id = data.get("req_id", "")
                        keys = list(data.keys())
                        logger.info(f"MSG #{msg_count}: type={msg_type}, req_id={req_id}, keys={keys}")
                    
                    await self._route_message(data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON: {raw_message}")
                except Exception as e:
                    # CRITICAL: Do NOT let one bad message kill the loop
                    logger.error(f"Error routing message: {e}", exc_info=True)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self._connected = False
            if self.on_disconnect:
                self.on_disconnect()
            asyncio.create_task(self._auto_reconnect())
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Recv loop error: {e}")
    
    async def _route_message(self, data: dict):
        """Route incoming messages to appropriate handlers."""
        msg_type = data.get("msg_type", "")
        
        # Handle pending request responses (always matched by req_id now)
        req_id = data.get("req_id")
        if req_id and req_id in self._pending_requests:
            future = self._pending_requests.pop(req_id)
            if not future.done():
                future.set_result(data)
        # Fallback: try to match responses that Deriv didn't echo req_id for
        # (shouldn't happen anymore, but kept as safety net)
        elif msg_type in ("history", "candles", "authorize", "active_symbols") and not req_id:
            for key in list(self._pending_requests.keys()):
                if isinstance(key, str) and key.startswith("_direct_"):
                    future = self._pending_requests.pop(key)
                    if not future.done():
                        future.set_result(data)
                    break
        
        # Route by message type
        if msg_type == "tick":
            tick = data.get("tick", {})
            symbol = tick.get("symbol", "")
            
            # ─── Auto-detect decimal places from pip_size ───
            # Deriv includes pip_size in every tick response.
            # This is the authoritative source for how many decimal places the price has.
            pip_size = tick.get("pip_size")
            if pip_size is not None and symbol:
                detected_dp = int(pip_size) if isinstance(pip_size, (int, float)) else None
                if detected_dp is not None:
                    if symbol not in self.symbol_decimals or self.symbol_decimals[symbol] != detected_dp:
                        logger.info(f"{symbol} pip_size={detected_dp} decimal places detected from tick")
                        self.symbol_decimals[symbol] = detected_dp
            
            # Include decimal_places in tick data for downstream consumers
            tick["decimal_places"] = self.symbol_decimals.get(symbol, 2)
            
            if self.on_tick:
                try:
                    if asyncio.iscoroutinefunction(self.on_tick):
                        await self.on_tick(symbol, tick)
                    else:
                        self.on_tick(symbol, tick)
                except Exception as e:
                    logger.error(f"on_tick callback error: {e}", exc_info=True)
        
        elif msg_type == "ticks":
            # Subscription confirmation response
            if data.get("error"):
                logger.error(f"Ticks subscription error: {data['error']}")
            else:
                logger.info(f"Ticks subscription confirmed for {data.get('echo_req', {}).get('ticks', '?')}")
        
        elif msg_type == "proposal_open_contract":
            contract = data.get("proposal_open_contract", {})
            if contract.get("is_sold") or contract.get("status") == "sold":
                if self.on_trade_result:
                    try:
                        if asyncio.iscoroutinefunction(self.on_trade_result):
                            await self.on_trade_result(contract)
                        else:
                            self.on_trade_result(contract)
                    except Exception as e:
                        logger.error(f"on_trade_result error: {e}", exc_info=True)
        
        elif msg_type == "proposal":
            if self.on_proposal:
                if asyncio.iscoroutinefunction(self.on_proposal):
                    await self.on_proposal(data.get("proposal", {}))
                else:
                    self.on_proposal(data.get("proposal", {}))
        
        elif msg_type == "balance":
            balance = data.get("balance", {})
            balance_amount = float(balance.get("balance", 0))
            currency = balance.get("currency", "USD")
            logger.info(f"Balance update: ${balance_amount:.2f} {currency}")
            if self.on_balance:
                try:
                    if asyncio.iscoroutinefunction(self.on_balance):
                        await self.on_balance(balance)
                    else:
                        self.on_balance(balance)
                except Exception as e:
                    logger.error(f"on_balance callback error: {e}")
        
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
                await self._send(sub, timeout=20.0)
                logger.info(f"Restored subscription: {name}")
                await asyncio.sleep(1)  # Delay between restores to avoid throttling
            except Exception as e:
                logger.warning(f"Failed to restore {name}: {e}")
    
    @property
    def is_connected(self) -> bool:
        return self._connected
