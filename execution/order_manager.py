import MetaTrader5 as mt5
from core.logger import get_logger
from config.settings import (
    MAGIC_NUMBER, PROFIT_GUARD_TRIGGER_PIPS, TRAILING_STOP_PIPS,
    DYNAMIC_TP_MULTIPLIER, PARTIAL_TP_ENABLED, PARTIAL_TP_RATIO,
    PARTIAL_TP_AT_R_MULTIPLE, MAX_SPREAD,
)
from database.db_manager import log_trade, close_trade

log = get_logger(__name__)

def _get_pip_point(symbol: str, sym_info) -> float:
    """Correct pip size for every symbol.
    
    IMPORTANT: For indices (JP225, DE30, UK100, US30, US500, USTEC),
    we use point size directly since these trade in integer points.
    For JPY pairs, 1 pip = 0.01. For standard forex, 1 pip = 0.0001.
    Gold = 0.1 (ounces), Silver = 0.01.
    """
    sym = str(symbol).upper()
    # Indices — trade in full points (1 point = 1 pip)
    if any(x in sym for x in ["US30", "US500", "USTEC", "JP225", "DE30", "UK100"]):
        return 1.0
    # Gold
    if "XAU" in sym:
        return 0.1
    # Silver
    if "XAG" in sym:
        return 0.01
    # Oil (WTI, Brent)
    if any(x in sym for x in ["WTI", "BRN"]):
        return 0.01
    # JPY pairs
    if sym_info.digits <= 3:
        return 0.01
    # Standard forex
    return 0.0001


def _get_pip_point_for_pricing(symbol: str) -> float:
    """Get pip size for pricing calculations (used by strategies).
    Must match _get_pip_point exactly.
    """
    sym = str(symbol).upper()
    if any(x in sym for x in ["US30", "US500", "USTEC", "JP225", "DE30", "UK100"]):
        return 1.0
    if "XAU" in sym:
        return 0.1
    if "XAG" in sym:
        return 0.01
    if any(x in sym for x in ["WTI", "BRN"]):
        return 0.01
    return 0.01 if any(x in sym for x in ["JPY", "USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY"]) else 0.0001


def get_atr_for_symbol(symbol: str, timeframe='M5', count=50) -> float:
    """Fetch ATR value for a symbol. Used for dynamic SL/TP/trailing."""
    import pandas as pd
    try:
        import talib
        use_talib = True
    except ImportError:
        use_talib = False
    
    tf_map = {'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5,
              'M15': mt5.TIMEFRAME_M15, 'H1': mt5.TIMEFRAME_H1}
    mt5_tf = tf_map.get(timeframe, mt5.TIMEFRAME_M5)
    
    rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count + 1)
    if rates is None or len(rates) < count:
        return 0.0
    
    df = pd.DataFrame(rates)
    df.columns = [c if isinstance(c, str) else c.decode() for c in df.columns]
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    pip_point = _get_pip_point(symbol, mt5.symbol_info(symbol))
    
    if use_talib:
        atr_raw = talib.ATR(high, low, close, timeperiod=min(14, count - 1))
        atr_raw = atr_raw[~pd.isna(atr_raw)]
        if len(atr_raw) > 0:
            return float(atr_raw[-1]) / pip_point
    
    # Fallback: manual ATR calculation
    tr = []
    for i in range(1, len(df)):
        h, l, c_prev = high[i], low[i], close[i-1]
        tr.append(max(h - l, abs(h - c_prev), abs(l - c_prev)))
    if not tr:
        return 0.0
    return sum(tr[-14:]) / min(14, len(tr)) / pip_point

def place_order(symbol: str, direction: str, lot_size: float,
                sl_pips: float, tp_pips: float,
                strategy: str, ai_score: float,
                session: str, market_regime: str,
                rsi: float = None, atr: float = None,
                spread: float = None) -> bool:
    """Place a market order with validated SL/TP.
    
    SAFETY: SL is ALWAYS required. Order will be rejected if:
    - sl_pips <= 0 or tp_pips <= 0
    - sl_pips > MAX_SL_PIPS (prevents runaway risk)
    - R:R < MIN_RISK_REWARD_RATIO
    """
    # ── HARD SAFETY: Reject orders without valid SL/TP ──
    if sl_pips <= 0 or tp_pips <= 0:
        log.error(f"[EXEC] REJECTED {symbol}: sl_pips={sl_pips} tp_pips={tp_pips} — "
                  f"BOTH must be > 0. No naked positions allowed.")
        return False
    
    # ── MAX SL CAP: Never risk more than 25 pips on any single trade ──
    MAX_SL_PIPS = 25.0
    if sl_pips > MAX_SL_PIPS:
        log.warning(f"[EXEC] {symbol} SL capped from {sl_pips:.1f}p to {MAX_SL_PIPS}p")
        sl_pips = MAX_SL_PIPS
    
    # ── MIN SL FLOOR: Never set SL tighter than 3 pips (prevents noise stopouts) ──
    MIN_SL_PIPS = 3.0
    if sl_pips < MIN_SL_PIPS:
        log.warning(f"[EXEC] {symbol} SL raised from {sl_pips:.1f}p to {MIN_SL_PIPS}p")
        sl_pips = MIN_SL_PIPS
    tick     = mt5.symbol_info_tick(symbol)
    sym_info = mt5.symbol_info(symbol)
    if tick is None or sym_info is None:
        log.error(f"[EXEC] Cannot get tick/info for {symbol}")
        return False

    point = _get_pip_point(symbol, sym_info)

    # ── v4.4: SPREAD RE-CHECK RIGHT BEFORE EXECUTION ──
    # Spread can widen between can_trade() check (50ms ago) and actual
    # order placement. Reject if spread has blown out at the last moment.
    spread_points = tick.ask - tick.bid
    spread_pips = spread_points / point
    max_sp = MAX_SPREAD.get(symbol, MAX_SPREAD.get("DEFAULT", 4.0))
    if spread_pips > max_sp:
        log.warning(f"[EXEC] REJECTED {symbol}: spread re-check {spread_pips:.1f}p "
                    f"> max {max_sp}p — spread widened between check and execution")
        return False

    if direction == "BUY":
        price    = tick.ask
        order_t  = mt5.ORDER_TYPE_BUY
        sl_price = round(price - sl_pips * point, sym_info.digits)
        tp_price = round(price + tp_pips * point, sym_info.digits)
    else:
        price    = tick.bid
        order_t  = mt5.ORDER_TYPE_SELL
        sl_price = round(price + sl_pips * point, sym_info.digits)
        tp_price = round(price - tp_pips * point, sym_info.digits)

    # Validate before sending
    if direction == "BUY":
        if sl_price >= price or tp_price <= price:
            log.error(f"[EXEC] Bad BUY levels: Price:{price} SL:{sl_price} TP:{tp_price}")
            return False
    else:
        if sl_price <= price or tp_price >= price:
            log.error(f"[EXEC] Bad SELL levels: Price:{price} SL:{sl_price} TP:{tp_price}")
            return False

    fill = mt5.ORDER_FILLING_IOC
    if sym_info.filling_mode & 1:
        fill = mt5.ORDER_FILLING_FOK

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       lot_size,
        "type":         order_t,
        "price":        price,
        "sl":           sl_price,
        "tp":           tp_price,
        "deviation":    20,
        "magic":        MAGIC_NUMBER,
        "comment":      f"APEX|{strategy[:10]}|{ai_score:.0f}",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": fill,
    }

    # Log with human-readable SL/TP in points for indices
    log.info(f"[EXEC] {direction} {symbol} Price:{price} "
             f"SL:{sl_price}({sl_pips}p) TP:{tp_price}({tp_pips}p) "
             f"Point:{point} Lot:{lot_size}")

    result = mt5.order_send(request)
    if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
        # v4.5 FIX: Get the REAL position ticket using 3 methods:
        #
        # MT5 ticket system:
        #   result.order = Order ticket (request ID — NOT the position)
        #   result.deal  = Deal ticket (execution record)
        #   deal.position_id = Position ticket (what MT5 tracks, what sync uses)
        #   pos.ticket  = Position ticket (same as deal.position_id)
        #
        # sync_closed_trades() matches by deal.position_id against DB ticket column.
        # So we MUST store deal.position_id in the DB.
        #
        # METHOD 1 (best): Look up the deal by result.deal → get deal.position_id
        # METHOD 2 (fallback): mt5.positions_get() → find our position
        # METHOD 3 (last resort): Use result.order (old behavior, has fallback matching)

        position_id = result.order  # METHOD 3: default fallback
        method = "order_ticket"

        # METHOD 1: Use the deal to get position_id directly
        # Try BOTH: by deal ticket AND by order ticket (different MT5 builds)
        if method == "order_ticket":
            try:
                deal_info = mt5.history_deals_get(result.deal, result.deal)
                if deal_info and len(deal_info) > 0:
                    d = deal_info[0]
                    pos_from_deal = getattr(d, 'position_id', 0)
                    if pos_from_deal and pos_from_deal > 0:
                        position_id = pos_from_deal
                        method = "deal.position_id"
            except Exception as e:
                log.warning(f"[EXEC] Method 1a (deal lookup) failed: {e}")

        # METHOD 1b: Look up by order ticket in deals history
        if method == "order_ticket" and result.order > 0:
            try:
                from datetime import timezone, timedelta
                import datetime as dt
                to_dt = dt.datetime.now(timezone.utc)
                from_dt = to_dt - timedelta(seconds=60)
                order_deals = mt5.history_deals_get(from_dt, to_dt)
                if order_deals:
                    for d in order_deals:
                        d_order = getattr(d, 'order', 0)
                        if d_order == result.order:
                            pos_from_deal = getattr(d, 'position_id', 0)
                            if pos_from_deal and pos_from_deal > 0:
                                position_id = pos_from_deal
                                method = "order_deal_lookup"
                                break
            except Exception as e:
                log.warning(f"[EXEC] Method 1b (order deal lookup) failed: {e}")

        # METHOD 2: Find our position from live positions list
        if method == "order_ticket":
            try:
                open_positions = mt5.positions_get(symbol=symbol)
                if open_positions:
                    for p in sorted(open_positions, key=lambda x: x.time, reverse=True):
                        if p.magic == MAGIC_NUMBER:
                            position_id = p.ticket
                            method = "positions_get"
                            break
            except Exception as e:
                log.warning(f"[EXEC] Method 2 (positions_get) failed: {e}")

        log.info(f"[EXEC] 🎫 Opened {symbol}: order={result.order} "
                 f"deal={result.deal} → pos_id={position_id} (via {method})")
        log.info(f"[EXEC] ✅ {direction} {symbol} | "
                 f"Position:{position_id} SL:{sl_pips}p TP:{tp_pips}p")
        log_trade({
            "ticket": position_id, "symbol": symbol,
            "direction": direction, "strategy": strategy,
            "session": session, "market_regime": market_regime,
            "entry_price": price, "sl_price": sl_price,
            "tp_price": tp_price, "lot_size": lot_size,
            "ai_score": ai_score, "rsi": rsi, "atr": atr, "spread": spread,
        })
        return True

    log.error(f"[EXEC] ❌ Failed retcode:{result.retcode if result else 'None'}"
              f" {mt5.last_error()}")
    return False

# v4.2: Cache ATR values to avoid fetching every 1 second (expensive)
_atr_cache: dict = {}  # {symbol: (atr_pips, timestamp)}
_ATR_CACHE_SECONDS = 30  # Refresh ATR every 30 seconds

# v4.4: Partial TP state tracker per position
# {ticket: {"original_sl_pips": float, "partial_done": bool}}
_partial_tp_state: dict = {}


def _get_cached_atr(symbol: str) -> float:
    """Get ATR with 30-second cache to avoid expensive MT5 calls every loop."""
    import time
    now = time.time()
    cached = _atr_cache.get(symbol)
    if cached and (now - cached[1]) < _ATR_CACHE_SECONDS:
        return cached[0]
    # Fresh fetch
    atr = get_atr_for_symbol(symbol, 'M5', 50)
    _atr_cache[symbol] = (atr, now)
    return atr


def manage_positions():
    """
    Every 1s (background thread): manage all bot positions.
    1. Force close if TP or SL hit (fallback safety)
    2. Break-even move: move SL to entry after trigger profit
    3. ATR-adaptive trailing stop — adjusts SL based on current volatility
    4. Dynamic TP extension based on ATR when 75% to target
    
    v4.3 FIXES:
    - Trailing now uses ATR * 1.0 (was 1.5) for tighter trails
    - Trailing activates right after break-even (profit_pips > trail_distance)
    - Added step-by-step diagnostic logging for every phase
    - Logs broker stops_level to catch SL modify failures
    """
    positions = mt5.positions_get()
    if not positions:
        return

    for pos in positions:
        if pos.magic != MAGIC_NUMBER:
            continue
        sym_info = mt5.symbol_info(pos.symbol)
        if sym_info is None:
            continue

        point = _get_pip_point(pos.symbol, sym_info)
        price = pos.price_current

        # ── Fallback close at TP or SL ────────────────────
        hit_tp = False
        hit_sl = False

        if pos.type == mt5.ORDER_TYPE_BUY:
            if pos.tp > 0 and price >= pos.tp:
                hit_tp = True
            if pos.sl > 0 and price <= pos.sl:
                hit_sl = True
        else:
            if pos.tp > 0 and price <= pos.tp:
                hit_tp = True
            if pos.sl > 0 and price >= pos.sl:
                hit_sl = True

        if hit_tp:
            log.info(f"[EXEC] 🎯 TP HIT {pos.symbol} #{pos.ticket} "
                     f"Price:{price} TP:{pos.tp}")
            _close_position(pos, "TP_HIT", sym_info)
            continue

        if hit_sl:
            log.info(f"[EXEC] 🛑 SL HIT {pos.symbol} #{pos.ticket} "
                     f"Price:{price} SL:{pos.sl}")
            _close_position(pos, "SL_HIT", sym_info)
            continue

        # ── Clean up stale partial TP state for closed positions ──
        # If position volume dropped to 0 or near 0, remove state
        if pos.volume <= sym_info.volume_min:
            _partial_tp_state.pop(pos.ticket, None)
            continue

        # ── Calculate profit in pips ──────────────────────
        profit_pips = 0
        if pos.type == mt5.ORDER_TYPE_BUY:
            profit_pips = (price - pos.price_open) / point
        else:
            profit_pips = (pos.price_open - price) / point

        # ── Get ATR (cached to avoid expensive fetch every second) ──
        atr_pips = _get_cached_atr(pos.symbol)

        # ── Broker minimum stop level (for logging) ──
        min_stop_level = sym_info.trade_stops_level * point if sym_info.trade_stops_level > 0 else point

        # ════════════════════════════════════════════════════════
        # STEP 0: PARTIAL TP — Close 50% at 1R (v4.4)
        # This is the BIGGEST profitability improvement.
        # At 1R profit: close 50%, move SL to breakeven, trail rest.
        # Even if the remaining 50% gets stopped at BE, you still
        # pocketed the first 50% at 1R. Net result: risk-free trade.
        # ════════════════════════════════════════════════════════
        if PARTIAL_TP_ENABLED:
            ticket = pos.ticket
            state = _partial_tp_state.get(ticket)

            if state is None:
                # First time seeing this position — record original SL distance
                original_sl_dist = abs(pos.price_open - pos.sl) / point
                if original_sl_dist > 1.0:
                    # SL hasn't been moved yet — store original distance
                    _partial_tp_state[ticket] = {
                        "original_sl_pips": original_sl_dist,
                        "partial_done": False,
                    }
                    state = _partial_tp_state[ticket]

            if state and not state.get("partial_done", False):
                orig_sl = state["original_sl_pips"]
                trigger_pips = orig_sl * PARTIAL_TP_AT_R_MULTIPLE

                if trigger_pips > 0 and profit_pips >= trigger_pips:
                    log.info(f"[EXEC] 🎯 PARTIAL TP TRIGGER {pos.symbol} #{pos.ticket} "
                             f"profit={profit_pips:.1f}p >= 1R={trigger_pips:.1f}p "
                             f"(SL was {orig_sl:.1f}p)")

                    # Close 50% of position
                    closed = _partial_close_position(
                        pos, sym_info, point,
                        f"1R_{profit_pips:.0f}p")

                    if closed:
                        # Move SL to breakeven immediately
                        if pos.type == mt5.ORDER_TYPE_BUY:
                            be_sl = round(pos.price_open + point * 0.5, sym_info.digits)
                        else:
                            be_sl = round(pos.price_open - point * 0.5, sym_info.digits)
                        _modify_sl(pos, be_sl, sym_info, "PARTIAL_BE")

                        # Mark as done — trailing takes over for remainder
                        _partial_tp_state[ticket]["partial_done"] = True
                        log.info(f"[EXEC] 📊 {pos.symbol} #{pos.ticket} "
                                 f"PARTIAL TP DONE — 50% banked at 1R, SL at BE, "
                                 f"remainder trailing with ATR")

                        # Skip rest of management this cycle —
                        # position state changed, re-evaluate next second
                        continue

        # ── STEP 1: Break-Even Move ───────────────────────
        # Move SL to entry + tiny buffer after PROFIT_GUARD_TRIGGER_PIPS profit.
        # This is INDEPENDENT of ATR — uses config value directly.
        be_trigger = PROFIT_GUARD_TRIGGER_PIPS  # e.g. 5 pips

        if profit_pips >= be_trigger:
            if pos.type == mt5.ORDER_TYPE_BUY:
                be_sl = round(pos.price_open + point * 0.5, sym_info.digits)
                if pos.sl < pos.price_open:
                    _modify_sl(pos, be_sl, sym_info,
                               f"BREAKEVEN(+{profit_pips:.1f}p)")
                else:
                    log.debug(f"[EXEC] {pos.symbol} #{pos.ticket} BE skip: "
                              f"SL already at {pos.sl} (>= entry {pos.price_open})")
            else:
                be_sl = round(pos.price_open - point * 0.5, sym_info.digits)
                if pos.sl > pos.price_open:
                    _modify_sl(pos, be_sl, sym_info,
                               f"BREAKEVEN(+{profit_pips:.1f}p)")
                else:
                    log.debug(f"[EXEC] {pos.symbol} #{pos.ticket} BE skip: "
                              f"SL already at {pos.sl} (<= entry {pos.price_open})")

        # ── STEP 2: ATR-Adaptive Trailing Stop ────────────
        # v4.3: Trail distance = ATR * 1.0 (tighter, was 1.5)
        # Trailing activates right after break-even (SL at/beyond entry).
        # Requires profit_pips > trail_distance to start trailing.
        if atr_pips > 0:
            trail_distance = atr_pips * 1.0
        else:
            trail_distance = TRAILING_STOP_PIPS

        # Only trail if SL is already at or beyond entry (break-even done)
        if pos.type == mt5.ORDER_TYPE_BUY:
            sl_at_or_above_entry = pos.sl >= pos.price_open
            if not sl_at_or_above_entry:
                log.debug(f"[EXEC] {pos.symbol} #{pos.ticket} Trail waiting: "
                          f"SL={pos.sl} < entry={pos.price_open} (BE not done yet)")
            elif profit_pips <= trail_distance:
                log.debug(f"[EXEC] {pos.symbol} #{pos.ticket} Trail waiting: "
                          f"profit={profit_pips:.1f}p <= trail_dist={trail_distance:.1f}p")
            else:
                new_sl = round(price - trail_distance * point, sym_info.digits)
                if new_sl > pos.sl:
                    _modify_sl(pos, new_sl, sym_info,
                               f"TRAIL(+{profit_pips:.1f}p, trail={trail_distance:.1f}p)")
                    log.info(f"[EXEC] 📈 {pos.symbol} #{pos.ticket} "
                             f"TRAILING SL: {pos.sl:.5f} -> {new_sl:.5f} "
                             f"(+{profit_pips:.1f}p, trail_dist={trail_distance:.1f}p, ATR={atr_pips:.1f}p)")
        else:
            sl_at_or_below_entry = pos.sl <= pos.price_open
            if not sl_at_or_below_entry:
                log.debug(f"[EXEC] {pos.symbol} #{pos.ticket} Trail waiting: "
                          f"SL={pos.sl} > entry={pos.price_open} (BE not done yet)")
            elif profit_pips <= trail_distance:
                log.debug(f"[EXEC] {pos.symbol} #{pos.ticket} Trail waiting: "
                          f"profit={profit_pips:.1f}p <= trail_dist={trail_distance:.1f}p")
            else:
                new_sl = round(price + trail_distance * point, sym_info.digits)
                if new_sl < pos.sl:
                    _modify_sl(pos, new_sl, sym_info,
                               f"TRAIL(+{profit_pips:.1f}p, trail={trail_distance:.1f}p)")
                    log.info(f"[EXEC] 📈 {pos.symbol} #{pos.ticket} "
                             f"TRAILING SL: {pos.sl:.5f} -> {new_sl:.5f} "
                             f"(+{profit_pips:.1f}p, trail_dist={trail_distance:.1f}p, ATR={atr_pips:.1f}p)")

        # ── STEP 3: Dynamic TP Extension ───────────────────
        # v4.3: Extend TP when position is 60% to target (was 75%)
        if pos.tp > 0:
            current_tp_pips = abs(pos.tp - pos.price_open) / point
            if profit_pips > current_tp_pips * 0.60 and current_tp_pips > 0:
                extension = max(atr_pips * 2.0, trail_distance * 1.5) if atr_pips > 0 else trail_distance * 1.5
                if pos.type == mt5.ORDER_TYPE_BUY:
                    new_tp = round(price + extension * point, sym_info.digits)
                    if new_tp > pos.tp:
                        _modify_tp(pos, new_tp, sym_info,
                                   f"EXTEND({extension:.1f}p)")
                        log.info(f"[EXEC] 🎯 {pos.symbol} #{pos.ticket} "
                                 f"TP EXTENDED: {pos.tp:.5f} -> {new_tp:.5f} "
                                 f"(+{profit_pips:.1f}p, ext={extension:.1f}p)")
                else:
                    new_tp = round(price - extension * point, sym_info.digits)
                    if new_tp < pos.tp:
                        _modify_tp(pos, new_tp, sym_info,
                                   f"EXTEND({extension:.1f}p)")
                        log.info(f"[EXEC] 🎯 {pos.symbol} #{pos.ticket} "
                                 f"TP EXTENDED: {pos.tp:.5f} -> {new_tp:.5f} "
                                 f"(+{profit_pips:.1f}p, ext={extension:.1f}p)")

def _partial_close_position(pos, sym_info, point: float, reason: str) -> bool:
    """Close a portion of the position (e.g., 50% at 1R profit).
    
    v4.4: Partial TP — close PARTIAL_TP_RATIO of volume,
    move SL to breakeven, let remainder trail with ATR.
    Returns True if partial close succeeded.
    """
    close_volume = pos.volume * PARTIAL_TP_RATIO
    # Round to broker's volume step
    close_volume = round(close_volume / sym_info.volume_step) * sym_info.volume_step

    # Safety: can't close less than broker minimum
    if close_volume < sym_info.volume_min:
        log.debug(f"[EXEC] Partial close skipped: {close_volume:.2f} lots "
                  f"below broker min {sym_info.volume_min} for {pos.symbol} #{pos.ticket}")
        return False

    tick = mt5.symbol_info_tick(pos.symbol)
    if tick is None:
        return False

    order_type = (mt5.ORDER_TYPE_SELL
                  if pos.type == mt5.ORDER_TYPE_BUY
                  else mt5.ORDER_TYPE_BUY)
    price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask

    fill = mt5.ORDER_FILLING_IOC
    if sym_info.filling_mode & 1:
        fill = mt5.ORDER_FILLING_FOK

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       pos.symbol,
        "volume":       close_volume,
        "type":         order_type,
        "position":     pos.ticket,
        "price":        price,
        "deviation":    20,
        "magic":        MAGIC_NUMBER,
        "comment":      f"PTP|{reason}",  # Partial TP comment
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": fill,
    }

    result = mt5.order_send(request)
    if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
        log.info(f"[EXEC] ✅ PARTIAL CLOSE {pos.symbol} #{pos.ticket} "
                 f"{close_volume:.2f}/{pos.volume:.2f} lots @ {price} "
                 f"({reason}) — banked ~{PARTIAL_TP_RATIO*100:.0f}% at 1R")
        return True

    retcode = result.retcode if result else "None"
    log.warning(f"[EXEC] ⚠️ Partial close FAILED {pos.symbol} #{pos.ticket} "
                f"code:{retcode} {mt5.last_error()}")
    return False


def _close_position(pos, reason: str, sym_info=None):
    """Close a position with market order.
    
    v4.2 FIX: Always records the close in the database, even if the
    actual profit differs from pos.profit (use deal profit instead).
    Also handles the case where MT5 already closed the position.
    """
    if sym_info is None:
        sym_info = mt5.symbol_info(pos.symbol)
    tick = mt5.symbol_info_tick(pos.symbol)
    if tick is None or sym_info is None:
        # v4.2: Even if we can't get tick, try to record the close
        # using the last known profit from the position
        log.error(f"[EXEC] Cannot get tick/info for {pos.symbol} "
                  f"#{pos.ticket} — attempting DB close anyway")
        try:
            outcome = "WIN" if pos.profit > 0 else "LOSS"
            close_trade(
                ticket      = pos.ticket,
                exit_price  = pos.price_current,
                profit_loss = pos.profit,
                outcome     = outcome,
            )
        except Exception as e:
            log.error(f"[EXEC] Emergency DB close failed: {e}")
        return False

    order_type = (mt5.ORDER_TYPE_SELL
                  if pos.type == mt5.ORDER_TYPE_BUY
                  else mt5.ORDER_TYPE_BUY)
    price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask

    fill = mt5.ORDER_FILLING_IOC
    if sym_info.filling_mode & 1:
        fill = mt5.ORDER_FILLING_FOK

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       pos.symbol,
        "volume":       pos.volume,
        "type":         order_type,
        "position":     pos.ticket,
        "price":        price,
        "deviation":    20,
        "magic":        MAGIC_NUMBER,
        "comment":      f"CLOSE|{reason}",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": fill,
    }

    result = mt5.order_send(request)
    if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
        # v4.2 FIX: Use result.deal to get actual profit from the deal
        # pos.profit can be stale — the deal object has the real final P&L
        actual_profit = pos.profit
        if result.deal > 0:
            deal_info = mt5.history_deals_get(result.deal, result.deal)
            if deal_info and len(deal_info) > 0:
                actual_profit = deal_info[0].profit
        
        log.info(f"[EXEC] ✅ Closed {pos.symbol} #{pos.ticket} "
                 f"{reason} P&L:{actual_profit:.2f}")

        # Determine outcome for database
        if reason == "TP_HIT":
            outcome = "WIN_TP"
        elif reason == "SL_HIT":
            outcome = "LOSS"
        elif reason == "MANUAL":
            outcome = "MANUAL" if actual_profit >= 0 else "MANUAL_LOSS"
        else:
            outcome = "WIN" if actual_profit > 0 else "LOSS"

        # Record close in database
        try:
            close_trade(
                ticket      = pos.ticket,
                exit_price  = price,
                profit_loss = actual_profit,
                outcome     = outcome,
            )
        except Exception as e:
            log.error(f"[EXEC] DB close_trade failed: {e}")
        return True

    code = result.retcode if result else "NONE"
    log.error(f"[EXEC] ❌ Close failed {pos.symbol} #{pos.ticket} "
              f"code:{code} {mt5.last_error()}")
    return False

def _modify_sl(pos, new_sl: float, sym_info, label: str):
    """Modify stop loss with detailed error logging."""
    new_sl = round(new_sl, sym_info.digits)
    request = {
        "action":   mt5.TRADE_ACTION_SLTP,
        "symbol":   pos.symbol,
        "position": pos.ticket,
        "sl":       new_sl,
        "tp":       pos.tp,
    }
    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        log.info(f"[EXEC] 🛡️ {label} SL:{new_sl} "
                 f"{pos.symbol} #{pos.ticket}")
    else:
        retcode = result.retcode if result else "None"
        err = mt5.last_error()
        log.warning(f"[EXEC] ⚠️ SL modify FAILED {label}: retcode={retcode} "
                    f"error={err} | {pos.symbol} #{pos.ticket} "
                    f"SL:{pos.sl}->{new_sl} stops_level={sym_info.trade_stops_level}")

def _modify_tp(pos, new_tp: float, sym_info, label: str):
    """Modify take profit with detailed error logging."""
    new_tp = round(new_tp, sym_info.digits)
    request = {
        "action":   mt5.TRADE_ACTION_SLTP,
        "symbol":   pos.symbol,
        "position": pos.ticket,
        "sl":       pos.sl,
        "tp":       new_tp,
    }
    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        log.info(f"[EXEC] 📈 {label} TP:{new_tp} "
                 f"{pos.symbol} #{pos.ticket}")
    else:
        retcode = result.retcode if result else "None"
        err = mt5.last_error()
        log.warning(f"[EXEC] ⚠️ TP modify FAILED {label}: retcode={retcode} "
                    f"error={err} | {pos.symbol} #{pos.ticket} "
                    f"TP:{pos.tp}->{new_tp}")

def sync_closed_trades():
    """
    Sync trades closed by MT5 server (SL/TP hit on broker side,
    or manually closed from MT5 terminal) to our database.
    Call this every cycle alongside manage_positions().

    v4.2 FIXES:
    - Scans last 7 days (not just today) to catch missed closes after restart
    - Handles deals where magic number might not match (manual close in MT5)
    - Tries BOTH deal.position_id AND deal.order_id for DB matching
    - Logs every step for debugging
    - Handles connection errors gracefully without crashing
    """
    from datetime import timezone, timedelta
    import datetime as dt

    # v4.2 FIX: Look back 7 days, not just today.
    # Trades closed after bot restart or across midnight were missed.
    to_dt   = dt.datetime.now(timezone.utc)
    from_dt = to_dt - timedelta(days=7)

    # Get ALL closed deals in the last 7 days from MT5 history
    deals = mt5.history_deals_get(from_dt, to_dt)
    if not deals:
        return

    from database.db_manager import get_connection, close_trade
    
    try:
        conn = get_connection()
        c = conn.cursor(dictionary=True)
    except Exception as e:
        log.error(f"[SYNC] Cannot get DB connection: {e}")
        return

    synced = 0
    for deal in deals:
        # Only closing deals (DEAL_ENTRY_OUT = 1, DEAL_ENTRY_INOUT = 2)
        if deal.entry not in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT):
            continue

        # Match by position_id (most reliable)
        position_id = deal.position_id
        order_id = deal.order  # Also try order ticket as fallback
        if position_id == 0 and order_id == 0:
            continue

        # v4.5: Log every closing deal for diagnosis
        # NOTE: TradeDeal has .ticket (not .deal) for deal ticket ID on some builds
        deal_ticket = getattr(deal, 'ticket', getattr(deal, 'deal', 0))
        log.info(f"[SYNC] Closing deal: {deal.symbol} deal_id={deal_ticket} "
                 f"pos_id={position_id} order_id={order_id} "
                 f"profit={deal.profit:.2f} comment={deal.comment}")

        # Check if already recorded as closed in our DB
        try:
            # v4.5: Try position_id first (primary match)
            c.execute("""
                SELECT id, timestamp_close, profit_loss, symbol, outcome
                FROM trades
                WHERE ticket = %s AND timestamp_close IS NULL
            """, (position_id,))
            row = c.fetchone()

            # Fallback: try order_id if position_id didn't match
            matched_ticket = position_id
            if not row and order_id > 0 and order_id != position_id:
                c.execute("""
                    SELECT id, timestamp_close, profit_loss, symbol, outcome
                    FROM trades
                    WHERE ticket = %s AND timestamp_close IS NULL
                """, (order_id,))
                row = c.fetchone()
                if row:
                    matched_ticket = order_id
                    log.info(f"[SYNC] Matched by order_id fallback: #{order_id}")

            # v4.5: If still no match, log ALL open tickets for this symbol
            if not row:
                c.execute("""
                    SELECT ticket, symbol, direction, timestamp_open
                    FROM trades
                    WHERE symbol = %s AND timestamp_close IS NULL
                """, (deal.symbol,))
                open_rows = c.fetchall()
                open_tickets = [r['ticket'] for r in open_rows] if open_rows else []
                log.debug(f"[SYNC] No DB match for {deal.symbol} pos_id={position_id} "
                          f"order_id={order_id} — open DB tickets: {open_tickets}")
                continue
        except Exception as e:
            log.error(f"[SYNC] DB query error: {e}")
            continue

        # Skip if already synced
        if row['timestamp_close'] is not None:
            continue  # Already synced

        # v4.2 FIX: Determine outcome with more detail
        comment = str(deal.comment).upper() if deal.comment else ""
        if deal.profit > 0:
            if "TP" in comment or "TAKE" in comment:
                outcome = "WIN_TP"
            else:
                outcome = "WIN"
        elif deal.profit < 0:
            if "SL" in comment or "STOP" in comment:
                outcome = "LOSS_SL"
            else:
                outcome = "LOSS"
        else:
            outcome = "BREAKEVEN"

        # Detect manual close (comment won't contain CLOSE| or APEX)
        if "CLOSE" not in comment and "APEX" not in comment:
            outcome = "MANUAL" if deal.profit >= 0 else "MANUAL_LOSS"

        try:
            close_trade(
                ticket      = matched_ticket,
                exit_price  = deal.price,
                profit_loss = deal.profit,
                outcome     = outcome,
            )
            synced += 1
            log.info(f"[SYNC] ✅ Recorded close: {row['symbol']} "
                     f"#{matched_ticket} {outcome} P&L:{deal.profit:.2f}")
        except Exception as e:
            log.error(f"[SYNC] Failed to record close #{position_id}: {e}")

    c.close()
    conn.close()
    
    if synced > 0:
        log.info(f"[SYNC] Synced {synced} closed trade(s) to database")

    # Clean up partial TP state for fully closed positions
    open_tickets = set()
    live_positions = mt5.positions_get()
    if live_positions:
        open_tickets = {p.ticket for p in live_positions if p.magic == MAGIC_NUMBER}
    stale = [t for t in _partial_tp_state if t not in open_tickets]
    for t in stale:
        _partial_tp_state.pop(t, None)
