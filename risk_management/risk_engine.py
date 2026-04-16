# =============================================================
# risk_management/risk_engine.py
# Protects the account at all times.
# FIXES: cooldown per symbol, JPY pip calc, R:R check
# =============================================================

import MetaTrader5 as mt5
from datetime import datetime, timezone, timedelta
from core.logger import get_logger
from config.settings import (
    RISK_PERCENT_PER_TRADE, MAX_OPEN_TRADES,
    MAX_DAILY_LOSS_PERCENT, MAGIC_NUMBER, MAX_SPREAD,
    SYMBOL_COOLDOWN_MINUTES, MIN_RISK_REWARD_RATIO,
    MAX_CONSECUTIVE_LOSSES, CONSECUTIVE_LOSS_PAUSE_MINUTES
)

log = get_logger(__name__)

# In-memory cooldown tracker: {symbol: last_trade_datetime}
_last_trade_time: dict = {}

# Consecutive loss tracker
_consecutive_losses: int = 0
_consecutive_loss_pause_until: datetime = None


def calculate_lot_size(symbol: str, sl_pips: float) -> float:
    """
    Risk-based position sizing with correct JPY/Gold pip values.
    Never risk more than RISK_PERCENT_PER_TRADE of balance per trade.
    """
    account  = mt5.account_info()
    sym_info = mt5.symbol_info(symbol)
    if account is None or sym_info is None:
        return 0.01

    risk_amount = account.balance * (RISK_PERCENT_PER_TRADE / 100)

    # Correct pip size per symbol type
    # Indices: 1 pip = 1 point
    # JPY pairs: 1 pip = 0.01
    # Gold XAUUSD: 1 pip = 0.1
    # Oil: 1 pip = 0.01
    # Standard forex: 1 pip = 0.0001
    sym = str(symbol).upper()
    close_price = sym_info.bid
    if any(x in sym for x in ["US30", "US500", "USTEC", "JP225", "DE30", "UK100"]):
        pip_size = 1.0
    elif "XAU" in sym:
        pip_size = 0.1
    elif "XAG" in sym:
        pip_size = 0.01
    elif any(x in sym for x in ["WTI", "BRN"]):
        pip_size = 0.01
    elif close_price > 50 or sym_info.digits <= 3:  # JPY pairs
        pip_size = 0.01
    else:  # Standard forex
        pip_size = 0.0001

    # Pip value in account currency
    pip_value_per_lot = sym_info.trade_tick_value * (pip_size / sym_info.trade_tick_size)

    if pip_value_per_lot <= 0 or sl_pips <= 0:
        log.warning(f"[RISK] {symbol} pip_value={pip_value_per_lot} sl={sl_pips} — using min lot")
        return sym_info.volume_min

    lot = risk_amount / (sl_pips * pip_value_per_lot)
    lot = max(sym_info.volume_min, min(lot, sym_info.volume_max))
    lot = round(lot / sym_info.volume_step) * sym_info.volume_step

    log.info(f"[RISK] {symbol} Lot:{lot:.2f} | Risk:${risk_amount:.2f}"
             f" | SL:{sl_pips}p | PipVal:${pip_value_per_lot:.2f}")
    return round(lot, 2)

def is_symbol_on_cooldown(symbol: str) -> bool:
    """
    FIX #1: Prevent same symbol from being traded repeatedly.
    Enforces minimum SYMBOL_COOLDOWN_MINUTES between trades.
    """
    last = _last_trade_time.get(symbol)
    if last is None:
        return False
    elapsed = (datetime.now(timezone.utc) - last).total_seconds() / 60
    if elapsed < SYMBOL_COOLDOWN_MINUTES:
        remaining = int(SYMBOL_COOLDOWN_MINUTES - elapsed)
        log.info(f"[RISK] {symbol} on cooldown — {remaining} mins remaining")
        return True
    return False


def register_trade(symbol: str, won: bool = None):
    """Call this after a trade is placed to start the cooldown.
    If won is provided, updates consecutive loss counter."""
    _last_trade_time[symbol] = datetime.now(timezone.utc)
    log.info(f"[RISK] {symbol} cooldown started — next trade in "
             f"{SYMBOL_COOLDOWN_MINUTES} mins")


def update_consecutive_losses(won: bool):
    """Update consecutive loss counter. Call after each trade closes."""
    global _consecutive_losses, _consecutive_loss_pause_until
    if won:
        _consecutive_losses = 0
    else:
        _consecutive_losses += 1
        if _consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            _consecutive_loss_pause_until = datetime.now(timezone.utc) + \
                timedelta(minutes=CONSECUTIVE_LOSS_PAUSE_MINUTES)
            log.warning(f"[RISK] 🛑 Consecutive loss limit: "
                        f"{_consecutive_losses} losses — pausing "
                        f"{CONSECUTIVE_LOSS_PAUSE_MINUTES} mins")


def is_consecutive_loss_paused() -> bool:
    """Check if trading is paused due to consecutive losses."""
    global _consecutive_loss_pause_until
    if _consecutive_loss_pause_until is None:
        return False
    if datetime.now(timezone.utc) < _consecutive_loss_pause_until:
        remaining = int((_consecutive_loss_pause_until - datetime.now(timezone.utc)).total_seconds() / 60)
        log.info(f"[RISK] Paused — consecutive losses. {remaining} mins remaining")
        return True
    _consecutive_loss_pause_until = None
    return False


def check_risk_reward(sl_pips: float, tp_pips: float) -> bool:
    """
    FIX #2: Enforce minimum risk/reward ratio.
    Rejects trades where TP is smaller than SL * MIN_RISK_REWARD_RATIO.
    """
    if sl_pips <= 0 or tp_pips <= 0:
        return False
    rr = tp_pips / sl_pips
    if rr < MIN_RISK_REWARD_RATIO:
        log.warning(f"[RISK] ❌ R:R too low: {rr:.2f} "
                    f"(TP:{tp_pips}p / SL:{sl_pips}p) "
                    f"— need {MIN_RISK_REWARD_RATIO}:1")
        return False
    return True


def is_daily_loss_limit_hit() -> bool:
    """Halt trading if daily loss exceeds the configured threshold."""
    account = mt5.account_info()
    if account is None:
        return False
    today = datetime.now(timezone.utc).date()
    history = mt5.history_deals_get(
        datetime(today.year, today.month, today.day, tzinfo=timezone.utc),
        datetime.now(timezone.utc)
    )
    if history is None:
        return False
    daily_pnl = sum(d.profit for d in history if d.magic == MAGIC_NUMBER)
    max_loss  = account.balance * (MAX_DAILY_LOSS_PERCENT / 100)
    if daily_pnl < -max_loss:
        log.warning(f"[RISK] 🛑 Daily loss limit hit! P&L: ${daily_pnl:.2f}")
        return True
    return False


def count_open_positions() -> int:
    """Count open positions held by this bot."""
    positions = mt5.positions_get()
    if positions is None:
        return 0
    return sum(1 for p in positions if p.magic == MAGIC_NUMBER)


def count_symbol_positions(symbol: str) -> int:
    """Count open positions for one specific symbol."""
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return 0
    return sum(1 for p in positions if p.magic == MAGIC_NUMBER)


def is_spread_acceptable(symbol: str) -> bool:
    """Check if current spread is within allowed limit."""
    tick     = mt5.symbol_info_tick(symbol)
    sym_info = mt5.symbol_info(symbol)
    if tick is None or sym_info is None:
        return False

    # Use same pip calculation as calculate_lot_size
    sym = str(symbol).upper()
    if any(x in sym for x in ["US30", "US500", "USTEC", "JP225", "DE30", "UK100"]):
        pip_size = 1.0
    elif "XAU" in sym:
        pip_size = 0.1
    elif "XAG" in sym:
        pip_size = 0.01
    elif any(x in sym for x in ["WTI", "BRN"]):
        pip_size = 0.01
    elif sym_info.digits <= 3:
        pip_size = 0.01
    else:
        pip_size = 0.0001

    spread = (tick.ask - tick.bid) / pip_size
    max_sp = MAX_SPREAD.get(symbol, MAX_SPREAD["DEFAULT"])
    if spread > max_sp:
        log.info(f"[RISK] ❌ {symbol} spread {spread:.1f}p > max {max_sp}p")
        return False
    return True


def is_news_blackout(high_impact_events: list,
                     window_minutes: int = 30) -> bool:
    """Block trading around high-impact news events."""
    now = datetime.now(timezone.utc)
    for event in (high_impact_events or []):
        try:
            event_time = datetime.fromisoformat(event.get('date', ''))
            if event_time.tzinfo is None:
                event_time = event_time.replace(tzinfo=timezone.utc)
            diff_mins = abs((event_time - now).total_seconds() / 60)
            if diff_mins <= window_minutes:
                log.warning(f"[RISK] 📰 News blackout: "
                            f"{event.get('title')} in {diff_mins:.0f} mins")
                return True
        except Exception:
            continue
    return False


def can_trade(symbol: str,
              high_impact_events: list = None,
              direction: str = None) -> tuple[bool, str]:
    """
    Master pre-trade gate. Returns (True,'ok') or (False,'reason').
    Checks in order: consecutive loss pause → daily limit → max positions
    → symbol positions → cooldown → spread → news → correlation.
    """
    # Consecutive loss protection
    if is_consecutive_loss_paused():
        return False, "consecutive_loss_pause"

    if is_daily_loss_limit_hit():
        return False, "daily_loss_limit"

    if count_open_positions() >= MAX_OPEN_TRADES:
        return False, "max_positions_reached"

    # FIX #1: No duplicate trades on same symbol
    if count_symbol_positions(symbol) > 0:
        return False, f"already_in_{symbol}"

    if is_symbol_on_cooldown(symbol):
        return False, "symbol_cooldown"

    if not is_spread_acceptable(symbol):
        return False, "spread_too_wide"

    if high_impact_events and is_news_blackout(high_impact_events):
        return False, "news_blackout"

    # ── Correlation Risk Check ──
    if direction:
        try:
            import MetaTrader5 as mt5
            from risk_management.correlation_manager import check_correlation_risk
            positions = mt5.positions_get()
            if positions:
                allowed, reason = check_correlation_risk(symbol, direction, positions)
                if not allowed:
                    return False, reason
        except Exception as e:
            log.warning(f"[RISK] Correlation check failed: {e}")

    return True, "ok"
