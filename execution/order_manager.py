# =============================================================
# execution/order_manager.py
# Handles all trade placement and position management.
# Smart entries, breakeven, trailing stops, partial closes.
# =============================================================

import MetaTrader5 as mt5
from core.logger import get_logger
from config.settings import MAGIC_NUMBER
from database.db_manager import log_trade

log = get_logger(__name__)


def place_order(symbol: str, direction: str, lot_size: float,
                sl_pips: float, tp_pips: float,
                strategy: str, ai_score: float,
                session: str, market_regime: str,
                rsi: float = None, atr: float = None,
                spread: float = None) -> bool:
    """Place a market order with full context logging."""
    tick     = mt5.symbol_info_tick(symbol)
    sym_info = mt5.symbol_info(symbol)
    if tick is None or sym_info is None:
        log.error(f"[EXEC] Cannot get tick/info for {symbol}")
        return False

    # Institutional Pip calculation
    digits = sym_info.digits
    if digits == 3 or digits == 5:
        point = sym_info.point * 10
    else:
        point = sym_info.point
    
    # Gold (XAUUSD) specific: 1 pip = 0.1 move
    if "XAU" in symbol:
        point = 0.1

    if direction == "BUY":
        price    = tick.ask
        order_t  = mt5.ORDER_TYPE_BUY
        sl_price = price - sl_pips * point
        tp_price = price + tp_pips * point
    else:
        price    = tick.bid
        order_t  = mt5.ORDER_TYPE_SELL
        sl_price = price + sl_pips * point
        tp_price = price - tp_pips * point

    fill = mt5.ORDER_FILLING_IOC
    if sym_info.filling_mode & 1:
        fill = mt5.ORDER_FILLING_FOK

    request = {
        "action":      mt5.TRADE_ACTION_DEAL,
        "symbol":      symbol,
        "volume":      lot_size,
        "type":        order_t,
        "price":       price,
        "sl":          round(sl_price, sym_info.digits),
        "tp":          round(tp_price, sym_info.digits),
        "deviation":   20,
        "magic":       MAGIC_NUMBER,
        "comment":     f"APEX|{strategy[:10]}|{ai_score:.0f}",
        "type_time":   mt5.ORDER_TIME_GTC,
        "type_filling":fill,
    }

    result = mt5.order_send(request)
    if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
        log.info(f"[EXEC] ✅ {direction} {symbol} | Lot:{lot_size} | "
                 f"SL:{sl_pips}p | TP:{tp_pips}p | Score:{ai_score:.0f}")
        log_trade({
            "ticket": result.order, "symbol": symbol,
            "direction": direction, "strategy": strategy,
            "session": session, "market_regime": market_regime,
            "entry_price": price, "sl_price": sl_price,
            "tp_price": tp_price, "lot_size": lot_size,
            "ai_score": ai_score, "rsi": rsi, "atr": atr, "spread": spread,
        })
        return True

    log.error(f"[EXEC] ❌ Order failed: {result.retcode if result else 'None'} "
              f"— {mt5.last_error()}")
    return False

def manage_positions():
    """
    Called every cycle. Manages all open bot positions:
    - Move to breakeven at 3 pips profit
    - Trail stop using 1.5x ATR above/below price
    - FALLBACK: Manually close if SL or TP is reached (safety net)
    """
    positions = mt5.positions_get()
    if not positions:
        # log.debug("[EXEC] No open positions to manage")
        return

    for pos in positions:
        if pos.magic != MAGIC_NUMBER:
            continue
            
        sym_info = mt5.symbol_info(pos.symbol)
        if sym_info is None:
            continue

        # Institutional Pip calculation
        digits = sym_info.digits
        if digits == 3 or digits == 5:
            point = sym_info.point * 10
        else:
            point = sym_info.point
            
        if "XAU" in pos.symbol:
            point = 0.1

        price  = pos.price_current
        
        # ── 1. SL/TP Fallback Check ───────────────────────
        # Close if price hit or exceeded SL/TP (if set)
        hit_sl = False
        hit_tp = False
        
        if pos.type == mt5.ORDER_TYPE_BUY:
            if pos.sl > 0 and price <= pos.sl: hit_sl = True
            if pos.tp > 0 and price >= pos.tp: hit_tp = True
        else:
            if pos.sl > 0 and price >= pos.sl: hit_sl = True
            if pos.tp > 0 and price <= pos.tp: hit_tp = True

        if hit_sl or hit_tp:
            reason = "FALLBACK_SL" if hit_sl else "FALLBACK_TP"
            log.info(f"[EXEC] 🚨 {reason} hit on {pos.symbol}! Close Price:{price} Target:{pos.sl if hit_sl else pos.tp}")
            _close_position(pos, reason, sym_info)
            continue

        # ── 2. Breakeven Management ──────────────────────
        profit_pips = (
            (price - pos.price_open) / point
            if pos.type == mt5.ORDER_TYPE_BUY
            else (pos.price_open - price) / point
        )

        # Move to breakeven at 3 pips profit
        if profit_pips >= 3.0:
            be_sl = pos.price_open
            if ((pos.type == mt5.ORDER_TYPE_BUY  and pos.sl < be_sl) or
                (pos.type == mt5.ORDER_TYPE_SELL and pos.sl > be_sl)):
                _modify_sl(pos, be_sl, sym_info, "BREAKEVEN")


def _close_position(pos, reason: str, sym_info=None):
    """Close an open position completely with dynamic filling mode."""
    if sym_info is None:
        sym_info = mt5.symbol_info(pos.symbol)
    
    tick = mt5.symbol_info_tick(pos.symbol)
    if tick is None or sym_info is None:
        return False
        
    order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    price      = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask

    # Dynamic filling mode
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
        log.info(f"[EXEC] ✅ Closed {pos.symbol} ticket {pos.ticket} ({reason})")
        return True
    
    err = mt5.last_error()
    ret = result.retcode if result else "NO_RESULT"
    log.error(f"[EXEC] ❌ Failed to close {pos.symbol} (Ticket:{pos.ticket}): Code:{ret} Error:{err}")
    return False


def _modify_sl(pos, new_sl: float, sym_info, label: str):
    """Send SL modification request to MT5."""
    new_sl = round(new_sl, sym_info.digits)
    request = {
        "action":      mt5.TRADE_ACTION_SLTP,
        "symbol":      pos.symbol,
        "position":    pos.ticket,
        "sl":          new_sl,
        "tp":          pos.tp,
    }
    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        log.info(f"[EXEC] 🛡️ {label} set on {pos.symbol} ticket {pos.ticket}")
    else:
        log.warning(f"[EXEC] SL modify failed: {mt5.last_error()}")
