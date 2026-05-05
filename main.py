# =============================================================
# main.py — APEX TRADER v4.2
# The master orchestrator. Run this file to start the bot.
# Connects ALL layers: Data → SMC → External → Strategies
#                    → AI Models → Risk → Execution
#
# v4.2 CHANGES:
#   - Fixed numpy.float64 .upper() error in parallel scan
#   - Added str() type safety at ALL .upper() call sites
#   - Added full traceback logging for parallel scan errors
#   - Sanitized all bias/direction fields from dict.get() calls
#   - Feature store now enforces explicit type casting
#   - Strategy engine sanitizes signal direction before use
#
# v4.1 CHANGES:
#   - MySQL database support (replaces SQLite)
#   - Fixed fractal alignment blocking all trades (relaxed gating)
#   - Fixed pip calculation for indices/metals (JP225, XAUUSD, etc.)
#   - Fixed tick aggregator delta (was always 0)
#   - Fixed dashboard scanner crash
#   - Bias cross-validation: strategy direction vs master combined_bias
#   - Re-entry logic: resume if setup still valid after TP
#   - Correlation risk check: prevent currency over-exposure
#   - ATR-adaptive trailing stop + ATR-based M1 Scalp SL
#   - Score deduplication in market_scanner
# =============================================================

import time
import traceback
import MetaTrader5 as mt5
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import os
import sys
import argparse
import numpy as np

from core.connection import connect, disconnect, is_algo_trading_enabled
from core.logger import get_logger
from database.db_manager import init_db, log_trade, log_signal
from config.settings import (
    WATCHLIST, MAGIC_NUMBER,
    ALLOW_REENTRY, REENTRY_COOLDOWN_MINUTES, REENTRY_MIN_SCORE_INCREASE,
    MIN_CONFLUENCE_COUNT,
    SESSION_WHITELIST, PAIR_BLACKLIST,
    SCAN_MODE, SCAN_PAIR_INTERVAL, SCAN_VERBOSE,
)

load_dotenv()
log = get_logger("APEX_TRADER")

# ── Timing constants ──────────────────────────────────────────
SCAN_INTERVAL          = 30     # Seconds between scan cycles
EXTERNAL_REFRESH_SECS  = 3600   # External data refresh (1 hour)
MODEL_RETRAIN_TRADES   = 50     # Retrain XGBoost every N trades
TRADE_COUNT_SINCE_TRAIN= 0      # Counter for retraining trigger

# ── Runtime flags (set by CLI args) ───────────────────────────
_relaxed       = False
_no_post_gates = False
_no_limit      = False


def run():
    """Main entry point — starts the full bot loop."""
    global _consecutive_losses, _relaxed, _no_post_gates, _no_limit

    log.info("=" * 60)
    log.info("  APEX TRADER — INSTITUTIONAL GRADE BOT")
    log.info("  Version 4.2 | numpy.float64 Fix + Type Safety + Traceback Logging")
    log.info("=" * 60)

    # Parse CLI arguments
    args = _parse_args()
    _relaxed       = args.relaxed
    _no_post_gates = args.no_post_gates
    _no_limit      = args.no_limit

    mode_label = []
    if _relaxed:
        mode_label.append("RELAXED")
    if _no_post_gates:
        mode_label.append("NO_POST_GATES")
    if _no_limit:
        mode_label.append("NO_LIMIT")
    if mode_label:
        log.info(f"[STARTUP] Mode flags: {' | '.join(mode_label)}")

    # ── Startup checks ────────────────────────────────────────
    if not connect():
        log.error("Cannot connect to MT5. Exiting.")
        return
    if not is_algo_trading_enabled():
        log.error("Enable AlgoTrading in MT5 terminal first.")
        disconnect()
        return

    init_db()
    log.info("[STARTUP] Database initialized")
    log.info(f"[STARTUP] Watchlist ({len(WATCHLIST)}): {', '.join(WATCHLIST)}")
    log.info(f"[STARTUP] Scan mode: {SCAN_MODE.upper()}"
             + (f" | interval: {SCAN_PAIR_INTERVAL}s between pairs" if SCAN_MODE == "sequential" else " | parallel")
             + (f" | verbose: ON" if SCAN_VERBOSE else ""))

    # ── Initialize Tick Aggregator ───────────────────────────
    from data_layer.tick_aggregator import init_aggregator
    init_aggregator(WATCHLIST)
    log.info("[STARTUP] Tick Aggregator initialized (continuous streaming)")

    # ── Import all modules ────────────────────────────────────
    from data_layer.master_scanner import master_scan
    from strategies.strategy_engine import run_strategies
    from strategies.strategy_registry import update_performance, get_summary
    from risk_management.risk_engine import (
        can_trade, calculate_lot_size, count_open_positions,
        is_daily_loss_limit_hit, register_trade, check_risk_reward,
        update_consecutive_losses, calculate_dynamic_risk_pct
    )
    from execution.order_manager import place_order, manage_positions
    from execution.order_manager import sync_closed_trades
    from ai_engine.phase_manager import check_all_promotions

    # ── High-frequency position management thread ─────────────
    import threading
    def position_manager_worker():
        log.info("[STARTUP] Position Manager thread started (1s interval)")
        last_heartbeat = 0
        while True:
            try:
                manage_positions()
                sync_closed_trades()

                # Heartbeat every 5 minutes
                if time.time() - last_heartbeat > 300:
                    log.info("[THREAD] Position Manager Heartbeat: Active")
                    last_heartbeat = time.time()
            except Exception as e:
                log.error(f"[THREAD] Position manager error: {e}")
            time.sleep(1)

    pm_thread = threading.Thread(target=position_manager_worker, daemon=True)
    pm_thread.start()

    # ── State variables ───────────────────────────────────────
    trade_count        = 0
    global TRADE_COUNT_SINCE_TRAIN

    log.info("[STARTUP] All modules loaded. Starting main loop...\n")

    try:
        while True:
            now     = time.time()
            cycle_t = datetime.now(timezone.utc).strftime("%H:%M:%S")

            # ── Check daily loss limit ────────────────────────
            if is_daily_loss_limit_hit():
                log.warning("[CYCLE] Daily loss limit hit — sleeping 1 hour")
                time.sleep(3600)
                continue

            from data_layer.market_regime import get_session
            session = get_session()

            # ── v2.0: Hard session gate — skip entire cycle if blocked ──
            if session not in SESSION_WHITELIST:
                log.debug(f"[CYCLE] Session {session} not in whitelist — sleeping {SCAN_INTERVAL}s")
                time.sleep(SCAN_INTERVAL)
                continue

            log.info(f"")
            log.info(f"{'━'*60}")
            log.info(f"🔍  SCAN CYCLE  {cycle_t} UTC  |  {session}")
            log.info(f"{'━'*60}")

            # ── Scan each symbol ───────────────────────────────────────
            if SCAN_MODE == "sequential":
                # Sequential: one pair at a time, with delay between each
                for i, symbol in enumerate(WATCHLIST):
                    try:
                        if SCAN_VERBOSE:
                            log.info(f"  [{i+1}/{len(WATCHLIST)}] Scanning {symbol}...")

                        result = _scan_and_trade(
                            symbol, session,
                            master_scan, run_strategies,
                            can_trade, calculate_lot_size,
                            place_order, log_signal)

                        if result:
                            trade_count += 1
                            TRADE_COUNT_SINCE_TRAIN += 1

                    except Exception as e:
                        log.error(f"[CYCLE] Error scanning {symbol}: {e}\n{traceback.format_exc()}")

                    # Delay between pairs (except after the last one)
                    if i < len(WATCHLIST) - 1:
                        time.sleep(SCAN_PAIR_INTERVAL)
            else:
                # Parallel: all pairs at once (original behavior)
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
                    futures = [executor.submit(_scan_and_trade,
                                               symbol,
                                               session,
                                               master_scan, run_strategies,
                                               can_trade, calculate_lot_size,
                                               place_order, log_signal)
                               for symbol in WATCHLIST]
                    for future in futures:
                        try:
                            result = future.result()
                            if result:
                                trade_count += 1
                                TRADE_COUNT_SINCE_TRAIN += 1
                        except Exception as e:
                            log.error(f"[CYCLE] Error during parallel scan: {e}\n{traceback.format_exc()}")

            # ── Retrain models if enough new trades ───────────
            if TRADE_COUNT_SINCE_TRAIN >= MODEL_RETRAIN_TRADES:
                log.info("[CYCLE] Triggering model retraining...")
                try:
                    from ai_engine.model_trainer import train_all_models
                    train_all_models()
                    check_all_promotions()
                    log.info(get_summary())
                    TRADE_COUNT_SINCE_TRAIN = 0
                except Exception as e:
                    log.error(f"[CYCLE] Retraining failed: {e}")

            log.info(f"🏁  Cycle done — trades today: {trade_count} | next scan in {SCAN_INTERVAL}s\n")
            time.sleep(SCAN_INTERVAL)

    except KeyboardInterrupt:
        log.info("\n[APEX] Stopped by user.")
    except Exception as e:
        log.error(f"[APEX] Fatal error: {e}")
    finally:
        disconnect()
        log.info("[APEX] Shutdown complete.")


def _parse_args():
    """Parse command-line arguments for live trading."""
    parser = argparse.ArgumentParser(description="Apex Trader v4.2 — Live Trading Bot")
    parser.add_argument("--relaxed", action="store_true",
                        help="Relaxed mode: lower confluence, skip post-gates for some strategies")
    parser.add_argument("--no-post-gates", action="store_true",
                        help="Skip bias/conflict/confluence post-gates (let ML handle filtering)")
    parser.add_argument("--no-limit", action="store_true",
                        help="Remove max open position limits")
    parser.add_argument("--use-strategy-models", action="store_true",
                        help="Use L1 strategy XGBoost models")
    parser.add_argument("--use-model", action="store_true",
                        help="Use L2 ML Gate model")
    parser.add_argument("--store-db", action="store_true",
                        help="Store trades in MySQL database")
    # Backwards compat: silently accept these flags (they're already defaults in live)
    parser.add_argument("--clear-data", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _check_reentry(symbol: str, direction: str, score: int) -> tuple:
    """
    Check if this is a re-entry and if it should be allowed.
    A re-entry is when the same symbol+direction is traded again
    shortly after a previous TP exit.

    Returns: (allowed: bool, is_reentry: bool, reason: str)
    """
    global _recent_exits

    if not ALLOW_REENTRY:
        return True, False, "ok"

    exit_info = _recent_exits.get(symbol)
    if exit_info is None:
        return True, False, "ok"  # No recent exit, not a re-entry

    exit_time = exit_info.get("exit_time")
    exit_direction = exit_info.get("direction")
    exit_score = exit_info.get("score", 0)

    # Check if direction matches
    if exit_direction != direction:
        # Different direction — not a re-entry, clear old record
        _recent_exits.pop(symbol, None)
        return True, False, "ok"

    # Check cooldown
    elapsed_minutes = (datetime.now(timezone.utc) - exit_time).total_seconds() / 60
    if elapsed_minutes < REENTRY_COOLDOWN_MINUTES:
        remaining = int(REENTRY_COOLDOWN_MINUTES - elapsed_minutes)
        return False, True, f"reentry_cooldown_{remaining}m"

    # Check if score is higher than before (must be stronger signal)
    if score < exit_score + REENTRY_MIN_SCORE_INCREASE:
        return False, True, f"reentry_score_too_low ({score}<{exit_score + REENTRY_MIN_SCORE_INCREASE})"

    # Re-entry allowed — clear the record
    _recent_exits.pop(symbol, None)
    return True, True, "reentry_allowed"


def _record_exit(symbol: str, direction: str, score: int, was_win: bool):
    """Record a trade exit for re-entry tracking."""
    global _recent_exits
    if was_win:
        _recent_exits[symbol] = {
            "exit_time": datetime.now(timezone.utc),
            "direction": direction,
            "score": score,
        }
    else:
        _recent_exits.pop(symbol, None)  # Don't allow re-entry after losses


def _scan_and_trade(symbol: str,
                    session: str,
                    master_scan, run_strategies,
                    can_trade, calculate_lot_size,
                    place_order, log_signal) -> bool:
    """
    Full pipeline for one symbol in one scan cycle.
    Returns True if a trade was placed.

    v4.0: Added bias cross-validation, correlation check, re-entry logic.
    v4.3: Added verbose scanning logs for live monitoring.
    """
    from risk_management.risk_engine import register_trade, check_risk_reward

    # ── Pre-trade risk checks ─────────────────────────────
    # Initial check without direction (for cooldown, daily limit, etc.)
    tradeable, reason = can_trade(symbol, [])
    if not tradeable:
        if SCAN_VERBOSE:
            log.info(f"      └─ ❌ Risk blocked — {reason}")
        return False

    # ── Run master scanner (pass session) ─────────────────
    master = master_scan(symbol, session=session)
    if master is None:
        if SCAN_VERBOSE:
            log.info(f"      └─ ❌ Master scan failed (no data)")
        return False

    final_score = master.get("final_score", 0)
    bias        = master.get("combined_bias", "NEUTRAL")
    state       = master.get("market_state", "UNKNOWN")
    action      = master.get("recommendation", {}).get("action", "SKIP")
    fractal     = master.get("fractal_alignment", {})
    fractal_rec = fractal.get("recommendation", "N/A")
    fractal_aligned = fractal.get("aligned", False)
    m5_confirmed = fractal.get("m5_structure", {}).get("confirmed", False)
    m1_aligned   = fractal.get("trigger", {}).get("trigger_aligned", False)

    bias_icon = "📈" if bias == "BULLISH" else "📉" if bias == "BEARISH" else "↔️"
    log.info(f"🧭  {symbol:<10} Score:{final_score:>3}/100  {bias:<9}{bias_icon}  {state}")

    # ── Verbose: show key metrics ─────────────────────────
    if SCAN_VERBOSE:
        of_imb = master.get("order_flow_imbalance", {})
        imb_val = of_imb.get("imbalance", 0)
        imb_str = of_imb.get("strength", "NONE")
        surge   = master.get("volume_surge", {})
        surge_on = surge.get("surge_detected", False)
        mom     = master.get("momentum", {})
        choppy  = mom.get("is_choppy", True)
        inst_ok = imb_str in ("STRONG", "EXTREME") or surge_on

        inst_label = "✅" if inst_ok else "⚠️ (soft)"
        choppy_label = "🔴 CHOPPY" if choppy else "✅"
        log.info(f"      ├─ OF imbalance: {imb_val:+.2f} ({imb_str}) | "
                 f"Surge: {'YES' if surge_on else 'no'} | "
                 f"Institutional: {inst_label} | Choppy: {choppy_label}")
        log.info(f"      ├─ Fractal: {'ALIGNED' if fractal_aligned else 'not aligned'}"
                 + (f" ({factors_agreed} factors)" if fractal_aligned else ""))

    # ── Fractal Alignment Gate — RELAXED for intraday ─────────
    setup_quality  = fractal.get('setup_quality', 0)
    factors_agreed = fractal.get('factors_agreed', 0)

    if not fractal_aligned:
        # Allow if score is reasonable and bias is clear
        if final_score >= 55 and bias not in ("CONFLICTED", "NEUTRAL"):
            if SCAN_VERBOSE:
                log.info(f"      ├─ Fractal gate BYPASSED (score={final_score}, bias={bias})")
        elif state in ("TRENDING_STRONG", "BREAKOUT_ACCEPTED") and final_score >= 50:
            if SCAN_VERBOSE:
                log.info(f"      ├─ Fractal gate BYPASSED (trending: {state})")
        else:
            if SCAN_VERBOSE:
                log.info(f"      └─ ❌ Fractal gate BLOCKED (score={final_score}, no alignment)")
            return False

    # ── Check scalping signal — skip if market is too choppy ──
    scalping = master.get("scalping_signal", {})
    if scalping.get("status") == "CHOPPY_SKIP":
        if SCAN_VERBOSE:
            log.info(f"      └─ ❌ Choppy market — SKIP")
        return False

    # ── Order Flow Direction Gate ─────────────────────────────
    of_imb = master.get("order_flow_imbalance", {})
    imb_value = of_imb.get("imbalance", 0)
    imb_strength = of_imb.get("strength", "NONE")
    if imb_strength in ("EXTREME", "STRONG"):
        if bias == "BULLISH" and imb_value < -0.3:
            if SCAN_VERBOSE:
                log.info(f"      └─ ❌ Order flow opposes BUY (imb={imb_value:+.2f})")
            return False
        elif bias == "BEARISH" and imb_value > 0.3:
            if SCAN_VERBOSE:
                log.info(f"      └─ ❌ Order flow opposes SELL (imb={imb_value:+.2f})")
            return False

    # ── Run strategy engine ───────────────────────────────
    if SCAN_VERBOSE:
        log.info(f"      ├─ Running strategies...")

    signal = run_strategies(symbol, master)

    if signal is None:
        if SCAN_VERBOSE:
            log.info(f"      └─ 📭 No strategy fired (action={action})")
        log_signal({
            'symbol': symbol, 'direction': None, 'strategy': 'NONE',
            'ai_score': final_score, 'was_traded': False,
            'skip_reason': f"No strategy fired | {action}",
            'session': session, 'market_regime': state,
        })
        return False

    strategy_name = signal.get('strategy', 'UNKNOWN')
    direction     = str(signal.get('direction', ''))
    score         = signal.get('score', 0)
    confluence    = signal.get('confluence', [])

    # FIXED: Use MIN_CONFLUENCE_COUNT from settings (not hardcoded)
    if len(confluence) < MIN_CONFLUENCE_COUNT:
        if SCAN_VERBOSE:
            log.info(f"      └─ ❌ Confluence too low: {len(confluence)}/{MIN_CONFLUENCE_COUNT} "
                     f"[{strategy_name}]")
        return False

    dir_icon = "📈" if direction == "BUY" else "📉"
    log.info(f"")
    log.info(f"📶  SIGNAL  {symbol}  {dir_icon} {direction}  [{strategy_name}]"
             f"  score={score}  conf={len(confluence)}"
             f"  SL={signal.get('sl_pips')}p  TP={signal.get('tp1_pips')}p")

    # ── Post-gates can be bypassed with --no-post-gates ───────
    if not _no_post_gates:
        # ══════════════════════════════════════════════════════════
        # BIAS CROSS-VALIDATION
        # ══════════════════════════════════════════════════════════
        direction = direction.upper() if isinstance(direction, str) else str(direction)

        if bias not in ("NEUTRAL", "CONFLICTED"):
            if bias == "BULLISH" and direction == "SELL":
                log.warning(f"🚫  {symbol}: BIAS CONFLICT — master={bias} signal={direction} [{strategy_name}] BLOCKED")
                log_signal({'symbol': symbol, 'direction': direction, 'strategy': strategy_name,
                            'ai_score': score, 'was_traded': False,
                            'skip_reason': f"BIAS_CONFLICT({bias} vs {direction})",
                            'session': session, 'market_regime': state})
                return False
            elif bias == "BEARISH" and direction == "BUY":
                log.warning(f"🚫  {symbol}: BIAS CONFLICT — master={bias} signal={direction} [{strategy_name}] BLOCKED")
                log_signal({'symbol': symbol, 'direction': direction, 'strategy': strategy_name,
                            'ai_score': score, 'was_traded': False,
                            'skip_reason': f"BIAS_CONFLICT({bias} vs {direction})",
                            'session': session, 'market_regime': state})
                return False
            else:
                if SCAN_VERBOSE:
                    log.info(f"      ├─ ✅ Bias aligned — {bias} matches {direction}")

        # ══════════════════════════════════════════════════════════
        # CORRELATION RISK CHECK (with direction)
        # ══════════════════════════════════════════════════════════
        tradeable, reason = can_trade(symbol, [], direction=direction)
        if not tradeable:
            log.info(f"🚫  {symbol}: Risk gate blocked after signal — {reason}")
            return False

        # ══════════════════════════════════════════════════════════
        # RE-ENTRY CHECK
        # ══════════════════════════════════════════════════════════
        reentry_ok, is_reentry, reentry_reason = _check_reentry(
            symbol, direction, score)
        if not reentry_ok:
            log.info(f"🚫  {symbol}: Re-entry blocked — {reentry_reason}")
            return False
        if is_reentry:
            log.info(f"♻️   {symbol}: Re-entry allowed — {reentry_reason}")
    else:
        if SCAN_VERBOSE:
            log.info(f"      ├─ Post-gates SKIPPED (--no-post-gates)")
        direction = direction.upper() if isinstance(direction, str) else str(direction)

    # ── AI model scoring ──────────────────────────────────
    from ai_engine.model_trainer import get_ai_score
    from data_layer.price_feed import get_candles
    df_m15  = get_candles(symbol, 'M15', 100)
    ai_result = get_ai_score(
        signal,
        master.get('market_report', {}),
        master.get('smc_report', {}),
        df_candles=df_m15)

    ai_score   = ai_result.get('ai_score', 50)
    ai_rec     = ai_result.get('recommendation', 'NEUTRAL')
    ai_trained = ai_result.get('xgb_trained', False)

    if SCAN_VERBOSE:
        log.info(f"      ├─ AI Gate: score={ai_score} | rec={ai_rec}"
                 + (" | model=NOT_TRAINED" if not ai_trained else ""))

    if ai_trained and ai_rec == 'SKIP':
        log.info(f"🚫  {symbol}: AI model rejected signal — {ai_result.get('note')}")
        log_signal({'symbol': symbol, 'direction': direction, 'strategy': strategy_name,
                    'ai_score': ai_score, 'was_traded': False,
                    'skip_reason': f"AI:{ai_rec}", 'session': session, 'market_regime': state})
        return False

    # ── Calculate position size ───────────────────────────
    sl_pips  = signal.get('sl_pips', 10)
    tp1_pips = signal.get('tp1_pips', 15)

    # Enforce minimum R:R
    if not check_risk_reward(sl_pips, tp1_pips):
        log.warning(f"🚫  {symbol}: R:R too low — TP:{tp1_pips}p / SL:{sl_pips}p")
        return False

    # v4.4: Dynamic position sizing based on conviction
    conviction_score = float(score + ai_score) / 2
    agreement_groups = signal.get('agreement_groups', 2)
    lot_size = calculate_lot_size(
        symbol, sl_pips,
        conviction_score=conviction_score,
        agreement_groups=agreement_groups)
    if lot_size <= 0:
        log.warning(f"🚫  {symbol}: Lot size calculation failed")
        return False

    # ── Place order ───────────────────────────────────────
    spread = signal.get('spread', 0)
    placed = place_order(
        symbol        = symbol,
        direction     = direction,
        lot_size      = lot_size,
        sl_pips       = sl_pips,
        tp_pips       = signal.get('tp1_pips', 15),
        strategy      = strategy_name,
        ai_score      = float(score + ai_score) / 2,
        session       = session,
        market_regime = state,
        rsi           = None,
        atr           = None,
        spread        = spread,
    )

    if placed:
        register_trade(symbol)
        dir_icon = "📈" if direction == "BUY" else "📉"
        log.info(f"💰  ORDER PLACED  {symbol}  {dir_icon} {direction}"
                 f"  SL:{sl_pips}p  TP:{signal.get('tp1_pips')}p"
                 f"  Lot:{lot_size}  [{strategy_name}]")
        log_signal({
            'symbol': symbol, 'direction': direction,
            'strategy': strategy_name,
            'ai_score': float(score + ai_score) / 2,
            'was_traded': True, 'skip_reason': None,
            'session': session, 'market_regime': state,
        })
    else:
        log.warning(f"⚠️  {symbol}: Order rejected by MT5 — check terminal")

    return placed


if __name__ == "__main__":
    run()
