import MetaTrader5 as mt5
from core.logger import get_logger
from config.settings import MAGIC_NUMBER, PROFIT_GUARD_TRIGGER_PIPS, TRAILING_STOP_PIPS, DYNAMIC_TP_MULTIPLIER
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
    """Place a market order with validated SL/TP."""
    tick     = mt5.symbol_info_tick(symbol)
    sym_info = mt5.symbol_info(symbol)
    if tick is None or sym_info is None:
        log.error(f"[EXEC] Cannot get tick/info for {symbol}")
        return False

    point = _get_pip_point(symbol, sym_info)

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
        # v4.3 FIX: Store POSITION ID (not ORDER ticket) in database.
        # In MT5, result.order = the order ticket (sent request)
        #             result.position = the position ID (what MT5 tracks)
        # These are DIFFERENT numbers! sync_closed_trades() matches
        # by deal.position_id and _close_position() uses pos.ticket —
        # both are position IDs. So the DB must store position ID.
        position_id = result.position if result.position and result.position > 0 else result.order
        log.info(f"[EXEC] 🎫 Opened {symbol}: order_ticket={result.order} "
                 f"position_id={result.position} deal_id={result.deal} "
                 f"→ DB ticket={position_id}")
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

        # Check if already recorded as closed in our DB
        try:
            # v4.3: Try BOTH position_id AND order_id for matching.
            # Old trades used result.order as ticket, new trades use result.position.
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
        except Exception as e:
            log.error(f"[SYNC] DB query error: {e}")
            continue

        # Skip if already closed or trade not in our DB at all
        if not row:
            continue
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
