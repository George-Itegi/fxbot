import MetaTrader5 as mt5
from core.logger import get_logger
from config.settings import MAGIC_NUMBER, PROFIT_GUARD_TRIGGER_PIPS, TRAILING_STOP_PIPS, DYNAMIC_TP_MULTIPLIER
from database.db_manager import log_trade, close_trade

log = get_logger(__name__)

def _get_pip_point(symbol: str, sym_info) -> float:
    """Correct pip size for every symbol."""
    sym = symbol.upper()
    if "XAU" in sym:
        return 0.1
    if "XAG" in sym:
        return 0.01
    if any(x in sym for x in ["US30", "US500", "USTEC"]):
        return 1.0
    if sym_info.digits <= 3:
        return 0.01    # JPY pairs
    return 0.0001      # Standard forex

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

    log.info(f"[EXEC] {direction} {symbol} Price:{price} "
             f"SL:{sl_price}({sl_pips}p) TP:{tp_price}({tp_pips}p)")

    result = mt5.order_send(request)
    if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
        log.info(f"[EXEC] ✅ {direction} {symbol} | "
                 f"Ticket:{result.order} SL:{sl_pips}p TP:{tp_pips}p")
        log_trade({
            "ticket": result.order, "symbol": symbol,
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

def manage_positions():
    """
    Every 30s: manage all bot positions.
    1. Force close if TP or SL hit (fallback safety)
    2. Implement dynamic profit protection (trailing stop & dynamic TP)
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

        # ── Dynamic Profit Protection (Trailing Stop & Dynamic TP) ──
        profit_pips = 0
        if pos.type == mt5.ORDER_TYPE_BUY:
            profit_pips = (price - pos.price_open) / point
        else:
            profit_pips = (pos.price_open - price) / point

        if profit_pips >= PROFIT_GUARD_TRIGGER_PIPS:
            # Calculate new trailing stop loss
            new_sl = 0.0
            if pos.type == mt5.ORDER_TYPE_BUY:
                new_sl = round(price - TRAILING_STOP_PIPS * point, sym_info.digits)
                # Ensure new SL is above original SL and entry price
                if new_sl > pos.sl and new_sl > pos.price_open:
                    _modify_sl(pos, new_sl, sym_info, "TRAILING_SL")
            else: # SELL
                new_sl = round(price + TRAILING_STOP_PIPS * point, sym_info.digits)
                # Ensure new SL is below original SL and entry price
                if new_sl < pos.sl and new_sl < pos.price_open:
                    _modify_sl(pos, new_sl, sym_info, "TRAILING_SL")

            # Dynamically adjust TP (optional, but good for extending winners)
            # This logic can be more sophisticated, e.g., based on ATR, market structure
            # For now, we'll extend it by a multiplier if it's getting close
            current_tp_pips = abs(pos.tp - pos.price_open) / point
            if profit_pips > current_tp_pips * 0.75: # If 75% to original TP
                new_tp = 0.0
                if pos.type == mt5.ORDER_TYPE_BUY:
                    new_tp = round(pos.price_open + current_tp_pips * DYNAMIC_TP_MULTIPLIER * point, sym_info.digits)
                    if new_tp > pos.tp: # Only extend if new TP is further
                        _modify_tp(pos, new_tp, sym_info, "DYNAMIC_TP")
                else: # SELL
                    new_tp = round(pos.price_open - current_tp_pips * DYNAMIC_TP_MULTIPLIER * point, sym_info.digits)
                    if new_tp < pos.tp: # Only extend if new TP is further
                        _modify_tp(pos, new_tp, sym_info, "DYNAMIC_TP")

def _close_position(pos, reason: str, sym_info=None):
    """Close a position with market order."""
    if sym_info is None:
        sym_info = mt5.symbol_info(pos.symbol)
    tick = mt5.symbol_info_tick(pos.symbol)
    if tick is None or sym_info is None:
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
        log.info(f"[EXEC] ✅ Closed {pos.symbol} #{pos.ticket} "
                 f"{reason} P&L:{pos.profit:.2f}")

        # Determine outcome for database
        if reason == "TP_HIT":
            outcome = "WIN_TP"
        elif reason == "SL_HIT":
            outcome = "LOSS"
        elif reason == "MANUAL":
            outcome = "MANUAL" if pos.profit >= 0 else "MANUAL_LOSS"
        else:
            outcome = "WIN" if pos.profit > 0 else "LOSS"

        # Record close in database
        try:
            close_trade(
                ticket      = pos.ticket,
                exit_price  = price,
                profit_loss = pos.profit,
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
    """Modify stop loss."""
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
        log.warning(f"[EXEC] SL modify failed: {mt5.last_error()}")

def _modify_tp(pos, new_tp: float, sym_info, label: str):
    """Modify take profit."""
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
        log.warning(f"[EXEC] TP modify failed: {mt5.last_error()}")

def sync_closed_trades():
    """
    Sync trades closed by MT5 server (SL/TP hit on broker side,
    or manually closed from MT5 terminal) to our database.
    Call this every cycle alongside manage_positions().
    """
    from datetime import timezone
    import datetime as dt

    today = dt.datetime.now(timezone.utc).date()
    from_dt = dt.datetime(
        today.year, today.month, today.day, tzinfo=timezone.utc)
    to_dt = dt.datetime.now(timezone.utc)

    # Get all closed deals today from MT5 history
    deals = mt5.history_deals_get(from_dt, to_dt)
    if not deals:
        return

    from database.db_manager import get_connection, close_trade
    conn = get_connection()

    for deal in deals:
        # Only our bot's trades, only closing deals (entry=1 means close)
        if deal.magic != MAGIC_NUMBER:
            continue
        if deal.entry != mt5.DEAL_ENTRY_OUT:
            continue

        # Check if already recorded as closed
        c = conn.cursor()
        c.execute("""
            SELECT id, timestamp_close, profit_loss
            FROM trades
            WHERE ticket = ?
        """, (deal.position_id,))
        row = c.fetchone()

        if row and row['timestamp_close'] is None:
            # Trade exists but not yet closed in our DB — update it
            if deal.profit > 0:
                outcome = "WIN_TP"
            elif deal.profit < 0:
                outcome = "LOSS"
            else:
                outcome = "BREAKEVEN"

            # Check if it was a manual close (comment won't have APEX)
            if "CLOSE" not in str(deal.comment) and \
               "APEX" not in str(deal.comment):
                outcome = "MANUAL" if deal.profit >= 0 else "MANUAL_LOSS"

            try:
                close_trade(
                    ticket      = deal.position_id,
                    exit_price  = deal.price,
                    profit_loss = deal.profit,
                    outcome     = outcome,
                )
                log.info(f"[SYNC] Recorded close: #{deal.position_id}"
                         f" {outcome} P&L:{deal.profit:.2f}")
            except Exception as e:
                log.error(f"[SYNC] Failed to record close: {e}")

    conn.close()
