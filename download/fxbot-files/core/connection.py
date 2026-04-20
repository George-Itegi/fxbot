# =============================================================
# core/connection.py
# Handles all MetaTrader 5 connection logic.
# Every other module imports from here — single source of truth.
# =============================================================

import MetaTrader5 as mt5
from dotenv import load_dotenv
import os

load_dotenv()

def connect() -> bool:
    """Initialize and authenticate with MT5 terminal."""
    if not mt5.initialize():
        print(f"[CONNECTION] ❌ Failed to initialize MT5: {mt5.last_error()}")
        return False

    login_id = int(os.getenv("MT5_LOGIN"))
    password  = os.getenv("MT5_PASSWORD")
    server    = os.getenv("MT5_SERVER")

    if not mt5.login(login_id, password=password, server=server):
        print(f"[CONNECTION] ❌ Login failed: {mt5.last_error()}")
        return False

    info = mt5.account_info()
    print(f"[CONNECTION] ✅ Connected | Account: {info.login} | "
          f"Balance: {info.balance} {info.currency} | "
          f"Leverage: 1:{info.leverage}")
    return True


def disconnect():
    """Cleanly shut down the MT5 connection."""
    mt5.shutdown()
    print("[CONNECTION] 🔌 MT5 disconnected.")


def is_algo_trading_enabled() -> bool:
    """Check that AlgoTrading is enabled in the MT5 terminal."""
    terminal = mt5.terminal_info()
    if not terminal.trade_allowed:
        print("[CONNECTION] 🛑 AlgoTrading is DISABLED in MT5 terminal!")
        return False
    return True


def get_account_info() -> dict:
    """Return key account stats as a clean dictionary."""
    info = mt5.account_info()
    if info is None:
        return {}
    return {
        "login":    info.login,
        "balance":  info.balance,
        "equity":   info.equity,
        "margin":   info.margin,
        "currency": info.currency,
        "leverage": info.leverage,
        "profit":   info.profit,
    }
