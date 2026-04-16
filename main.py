# =============================================================
# main.py — APEX TRADER v4.1
# The master orchestrator. Run this file to start the bot.
# Connects ALL layers: Data → SMC → External → Strategies
#                    → AI Models → Risk → Execution
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
import MetaTrader5 as mt5
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import os

from core.connection import connect, disconnect, is_algo_trading_enabled
from core.logger import get_logger
from database.db_manager import init_db, log_trade, log_signal
from config.settings import (
    WATCHLIST, MAGIC_NUMBER,
    ALLOW_REENTRY, REENTRY_COOLDOWN_MINUTES, REENTRY_MIN_SCORE_INCREASE
)

load_dotenv()
log = get_logger("APEX_TRADER")

# ── Timing constants ──────────────────────────────────────────
SCAN_INTERVAL          = 30     # Seconds between scan cycles
EXTERNAL_REFRESH_SECS  = 3600   # External data refresh (1 hour)
MODEL_RETRAIN_TRADES   = 50     # Retrain XGBoost every N trades
TRADE_COUNT_SINCE_TRAIN= 0      # Counter for retraining trigger

# ── Re-entry tracking ────────────────────────────────────────
# {symbol: {"exit_time": datetime, "direction": str, "score": int}}
_recent_exits: dict = {}

# ── Consecutive loss tracking ────────────────────────────────
_consecutive_losses = 0


def run():
    """Main entry point — starts the full bot loop."""
    global _consecutive_losses

    log.info("=" * 60)
    log.info("  APEX TRADER — INSTITUTIONAL GRADE BOT")
    log.info("  Version 4.1 | Signal Quality + Risk + MySQL + Pip Fixes")
    log.info("=" * 60)

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
    log.info(f"[STARTUP] Watchlist: {', '.join(WATCHLIST)}")

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
        update_consecutive_losses
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

            log.info(f"\n{'='*52}")
            log.info(f"  SCAN | {cycle_t} | {session}")
            log.info(f"{'='*52}")

            # ── Scan each symbol in parallel ──────────────────────────────
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
                        log.error(f"[CYCLE] Error during parallel scan: {e}")

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

            log.info(f"[CYCLE] Complete. Trades today: {trade_count}"
                     f" | Next in {SCAN_INTERVAL}s\n")
            time.sleep(SCAN_INTERVAL)

    except KeyboardInterrupt:
        log.info("\n[APEX] Stopped by user.")
    except Exception as e:
        log.error(f"[APEX] Fatal error: {e}")
    finally:
        disconnect()
        log.info("[APEX] Shutdown complete.")


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
    """
    from risk_management.risk_engine import register_trade, check_risk_reward

    # ── Pre-trade risk checks ─────────────────────────────
    # Initial check without direction (for cooldown, daily limit, etc.)
    tradeable, reason = can_trade(symbol, [])
    if not tradeable:
        log.info(f"  {symbol}: Risk blocked — {reason}")
        return False

    # ── Run master scanner (pass session) ─────────────────
    master = master_scan(symbol, session=session)
    if master is None:
        log.info(f"  {symbol}: Master scan failed")
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

    log.info(f"  {symbol}: Score={final_score} | {bias}"
             f" | {state} | -> {action} | Fractal: {fractal_rec}")

    # ── Fractal Alignment Gate (v4.1 RELAXED) ──
    # setup_quality is 0-3 from fractal_alignment: in_zone + M5_confirmed + M1_aligned
    setup_quality = fractal.get('setup_quality', 0)
    factors_agreed = fractal.get('factors_agreed', 0)
    
    if not fractal_aligned:
        # Bypass conditions (from most to least strict):
        # 1. High score + trending + any setup confirmation
        if final_score >= 70 and state in ("TRENDING_STRONG", "BREAKOUT_ACCEPTED") and setup_quality >= 2:
            log.info(f"  {symbol}: M1 bypassed — high-confidence setup "
                     f"(score={final_score}, {state}, setup_q={setup_quality})")
        # 2. Good score + M5 confirmed + directional bias
        elif final_score >= 60 and m5_confirmed and bias not in ("CONFLICTED", "NEUTRAL"):
            log.info(f"  {symbol}: M1 bypassed — strong setup "
                     f"(score={final_score}, bias={bias}, M5 confirmed)")
        # 3. Very high score alone (strong institutional signal)
        elif final_score >= 80 and bias not in ("CONFLICTED", "NEUTRAL"):
            log.info(f"  {symbol}: M1 bypassed — very high score "
                     f"(score={final_score}, bias={bias})")
        # 4. In setup zone + M5 confirmed (enough structure)
        elif setup_quality >= 2 and bias not in ("CONFLICTED", "NEUTRAL"):
            log.info(f"  {symbol}: M1 bypassed — in setup zone with M5 confirm "
                     f"(setup_q={setup_quality}, bias={bias})")
        else:
            log.info(f"  {symbol}: Skipping due to no fractal alignment. "
                     f"(score={final_score} q={setup_quality} factors={factors_agreed})")
            return False

    # ── Check scalping signal — skip if market is too choppy ──
    scalping = master.get("scalping_signal", {})
    if scalping.get("status") == "CHOPPY_SKIP":
        log.info(f"  {symbol}: Skipping — choppy market "
                 f"(velocity: {scalping.get('velocity_pips_min', 0)} pips/min)")
        return False

    # ── Order Flow Direction Gate ─────────────────────────────
    of_imb = master.get("order_flow_imbalance", {})
    imb_value = of_imb.get("imbalance", 0)
    imb_strength = of_imb.get("strength", "NONE")
    if imb_strength in ("EXTREME", "STRONG"):
        if bias == "BULLISH" and imb_value < -0.3:
            log.info(f"  {symbol}: Skipping — order flow opposes BUY "
                     f"(imbalance={imb_value:+.2f})")
            return False
        elif bias == "BEARISH" and imb_value > 0.3:
            log.info(f"  {symbol}: Skipping — order flow opposes SELL "
                     f"(imbalance={imb_value:+.2f})")
            return False

    # ── Run strategy engine ───────────────────────────────
    signal = run_strategies(symbol, master)

    if signal is None:
        log_signal({
            'symbol':       symbol,
            'direction':    None,
            'strategy':     'NONE',
            'ai_score':     final_score,
            'was_traded':   False,
            'skip_reason':  f"No strategy fired | {action}",
            'session':      session,
            'market_regime':state,
        })
        return False

    strategy_name = signal.get('strategy', 'UNKNOWN')
    direction     = signal.get('direction', '')
    score         = signal.get('score', 0)
    log.info(f"  {symbol}: SIGNAL {direction} from {strategy_name}"
             f" score={score}")

    # ══════════════════════════════════════════════════════════
    # NEW v4.0: BIAS CROSS-VALIDATION
    # Strategy direction must align with master combined_bias.
    # If master says BULLISH but strategy says SELL → BLOCK.
    # Exception: CONFLICTED bias → allow strategy direction.
    # ══════════════════════════════════════════════════════════
    if bias not in ("NEUTRAL", "CONFLICTED"):
        if bias == "BULLISH" and direction == "SELL":
            log.warning(f"  {symbol}: ❌ BIAS CONFLICT — Master says {bias} "
                        f"but {strategy_name} says {direction}. BLOCKED.")
            log_signal({
                'symbol': symbol, 'direction': direction,
                'strategy': strategy_name, 'ai_score': score,
                'was_traded': False, 'skip_reason': f"BIAS_CONFLICT(master={bias},signal={direction})",
                'session': session, 'market_regime': state,
            })
            return False
        elif bias == "BEARISH" and direction == "BUY":
            log.warning(f"  {symbol}: ❌ BIAS CONFLICT — Master says {bias} "
                        f"but {strategy_name} says {direction}. BLOCKED.")
            log_signal({
                'symbol': symbol, 'direction': direction,
                'strategy': strategy_name, 'ai_score': score,
                'was_traded': False, 'skip_reason': f"BIAS_CONFLICT(master={bias},signal={direction})",
                'session': session, 'market_regime': state,
            })
            return False
        else:
            log.info(f"  {symbol}: ✅ Bias validated — {bias} aligned with {direction}")

    # ══════════════════════════════════════════════════════════
    # NEW v4.0: CORRELATION RISK CHECK (with direction)
    # ══════════════════════════════════════════════════════════
    tradeable, reason = can_trade(symbol, [], direction=direction)
    if not tradeable:
        log.info(f"  {symbol}: Risk blocked after signal — {reason}")
        return False

    # ══════════════════════════════════════════════════════════
    # NEW v4.0: RE-ENTRY CHECK
    # ══════════════════════════════════════════════════════════
    reentry_ok, is_reentry, reentry_reason = _check_reentry(
        symbol, direction, score)
    if not reentry_ok:
        log.info(f"  {symbol}: Re-entry blocked — {reentry_reason}")
        return False
    if is_reentry:
        log.info(f"  {symbol}: Re-entry allowed — {reentry_reason}")

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

    if ai_trained and ai_rec == 'SKIP':
        log.info(f"  {symbol}: AI rejected — {ai_result.get('note')}")
        log_signal({
            'symbol': symbol, 'direction': direction,
            'strategy': strategy_name, 'ai_score': ai_score,
            'was_traded': False, 'skip_reason': f"AI:{ai_rec}",
            'session': session, 'market_regime': state,
        })
        return False

    # ── Calculate position size ───────────────────────────
    sl_pips  = signal.get('sl_pips', 10)
    tp1_pips = signal.get('tp1_pips', 15)

    # Enforce minimum R:R
    if not check_risk_reward(sl_pips, tp1_pips):
        log.warning(f"  {symbol}: Rejected — R:R too low "
                    f"(TP:{tp1_pips}p / SL:{sl_pips}p)")
        return False

    lot_size = calculate_lot_size(symbol, sl_pips)
    if lot_size <= 0:
        log.warning(f"  {symbol}: Lot size calculation failed")
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
        log.info(f"  {symbol}: {direction} placed | "
                 f"SL:{sl_pips}p | TP:{signal.get('tp1_pips')}p"
                 f" | Lot:{lot_size}")
        log_signal({
            'symbol': symbol, 'direction': direction,
            'strategy': strategy_name,
            'ai_score': float(score + ai_score) / 2,
            'was_traded': True, 'skip_reason': None,
            'session': session, 'market_regime': state,
        })

    return placed


if __name__ == "__main__":
    run()
