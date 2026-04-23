# =============================================================
# core/logger.py  v2.0
# Clean trading-focused logs.
# Console: only what matters for trading decisions.
# File: full debug log saved daily.
# =============================================================

import logging
import os
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# ── Modules to silence on console (still saved to file) ──────
CONSOLE_SILENT_MODULES = {
    "execution.order_manager",   # hides [SYNC] noise
    "data_layer.tick_aggregator",
    "data_layer.tick_fetcher",
    "data_layer.delta_calculator",
    "data_layer.feature_store",
    "data_layer.fractal_alignment",
    "data_layer.market_scanner",
    "data_layer.smc.smc_scanner",
    "data_layer.smc.market_structure",
    "data_layer.smc.order_blocks",
    "data_layer.smc.liquidity_pools",
    "data_layer.smc.liquidity_sweeps",
    "data_layer.smc.fair_value_gaps",
    "data_layer.smc.premium_discount",
    "data_layer.smc.htf_alignment",
    "data_layer.volume_profile",
    "data_layer.vwap_calculator",
    "data_layer.momentum_velocity",
    "data_layer.tick_volume_surge",
    "data_layer.order_flow_imbalance",
    "ai_engine.model_trainer",
    "ai_engine.phase_manager",
    "risk_management.risk_engine",
    "strategies.strategy_registry",
}

class TradingConsoleFilter(logging.Filter):
    """
    Only show on console what a trader cares about:
    - Scan cycle headers
    - Signal found / blocked / placed
    - Trade opened or closed
    - Errors and warnings
    Suppresses all data-layer and sync noise.
    """
    SILENT = CONSOLE_SILENT_MODULES

    def filter(self, record: logging.LogRecord) -> bool:
        name = record.name

        # Always show errors and warnings from any module
        if record.levelno >= logging.WARNING:
            return True

        # Silence noisy data/sync modules at INFO level
        for silent in self.SILENT:
            if name == silent or name.startswith(silent + "."):
                return False

        return True


class TradingFormatter(logging.Formatter):
    """
    Clean, readable format for console.
    Adds visual separators for important events.
    """
    ICONS = {
        "SIGNAL":  "📶",
        "TRADE":   "💰",
        "WIN":     "✅",
        "LOSS":    "❌",
        "BIAS":    "🧭",
        "BLOCK":   "🚫",
        "SCAN":    "🔍",
        "WARN":    "⚠️",
        "ERROR":   "🔴",
    }

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        t   = datetime.now().strftime('%H:%M:%S')
        lvl = record.levelname

        # Errors/warnings stand out
        if record.levelno >= logging.ERROR:
            return f"\n{'─'*60}\n🔴 [{t}] ERROR in {record.name}:\n   {msg}\n{'─'*60}"
        if record.levelno >= logging.WARNING:
            return f"⚠️  [{t}] {msg}"

        return f"   [{t}] {msg}"


def get_logger(name: str) -> logging.Logger:
    """
    Returns a named logger.
    Console: clean trading signals only.
    File: full debug everything.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # ── Console handler — trading-focused, filtered ───────────
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.addFilter(TradingConsoleFilter())
    ch.setFormatter(TradingFormatter())
    logger.addHandler(ch)

    # ── File handler — full debug, saved daily ────────────────
    today = datetime.now().strftime('%Y-%m-%d')
    fh = logging.FileHandler(
        os.path.join(LOG_DIR, f'{today}.log'), encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'))
    logger.addHandler(fh)

    return logger
