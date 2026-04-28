# =============================================================
# backtest/engine.py  v2.0
# The main backtest loop — upgraded to match live system.
#
# Key upgrades vs v1.0:
#  1. Builds proper master_report (matches live master_scanner.py schema)
#  2. Updates feature_store (needed by LIQUIDITY_SWEEP_ENTRY)
#  3. Passes master_report to all strategies
#  4. Final score gate (≥ 45) from market + SMC scores
#  5. Dynamic position sizing with conviction levels
#  6. Partial TP + ATR trailing + dynamic TP extension
#  7. Confluence minimum = 6 (matches live)
#  8. Proper pip value per symbol (not hardcoded $1)
# =============================================================

import datetime
import time as time_mod
from dataclasses import dataclass
from core.logger import get_logger
from backtest.data_loader import load_all_data, get_candles_at_time
from backtest.tick_simulator import simulate_order_flow_from_full
from backtest.smc_builder import build_smc_report
from backtest.market_builder import build_market_report
from backtest.trade_tracker import TradeTracker
from backtest.config import (
    SYMBOLS, START_DATE, END_DATE, CACHE_DIR,
    AVG_SPREAD_PIPS, SLIPPAGE_PIPS, PIP_VALUE_PER_LOT,
    STARTING_BALANCE, SCAN_EVERY_N_BARS, STRATEGIES_FILTER,
    PARTIAL_TP_ENABLED, ATR_TRAIL_ENABLED, DYNAMIC_TP_EXTENSION_ENABLED,
    DYNAMIC_SIZING_ENABLED, BASE_RISK_PERCENT, MIN_CONFLUENCE,
    MIN_RR_RATIO, MASTER_MIN_SCORE, MAX_OPEN_TRADES, MAX_PER_SYMBOL,
    CONVICTION_LOW_SCORE_MAX, CONVICTION_MED_SCORE_MAX,
)
from core.pip_utils import get_pip_size

# Relaxed mode overrides
RELAXED_MIN_SCORE = 35
RELAXED_MIN_CONFLUENCE = 4
RELAXED_MIN_RR_RATIO = 1.5
RELAXED_CONSENSUS_GROUPS = 1  # Only 1 strategy group needed

# Strategy feature keys embedded in signal dicts by each strategy
_STRATEGY_FEATURE_KEYS = {
    'TREND_CONTINUATION': '_trend_cont_features',
    'STRUCTURE_ALIGNMENT': '_structure_features',
    'SMC_OB_REVERSAL': '_smc_ob_features',
    'EMA_CROSS_MOMENTUM': '_ema_cross_features',
    'VWAP_MEAN_REVERSION': '_vwap_features',
    'BREAKOUT_MOMENTUM': '_breakout_features',
    'LIQUIDITY_SWEEP_ENTRY': '_liq_sweep_features',
    'DELTA_DIVERGENCE': '_delta_div_features',
    'FVG_REVERSION': '_fvg_features',
    'RSI_DIVERGENCE_SMC': '_rsi_div_features',
}

# Mapping from feature dict keys to store_trade() parameter names
_FEATURE_PARAM_MAP = {
    '_trend_cont_features': 'trend_cont_features',
    '_structure_features': 'structure_features',
    '_smc_ob_features': 'smc_ob_features',
    '_ema_cross_features': 'ema_cross_features',
    '_vwap_features': 'vwap_features',
    '_breakout_features': 'breakout_features',
    '_liq_sweep_features': 'liq_sweep_features',
    '_delta_div_features': 'delta_div_features',
    '_fvg_features': 'fvg_features',
    '_rsi_div_features': 'rsi_div_features',
}


def _extract_strategy_features(signal: dict) -> dict:
    """
    Extract strategy-specific feature dicts from a signal dict.

    Each strategy embeds its features under a key like '_trend_cont_features',
    '_structure_features', etc. This helper extracts them and maps to the
    parameter names expected by store_trade().

    Returns dict suitable for **kwargs unpacking into store_trade().
    """
    if not signal:
        return {}
    features = {}
    for sig_key, param_name in _FEATURE_PARAM_MAP.items():
        val = signal.get(sig_key)
        if val:
            features[param_name] = val
    return features


def _log_shadow_results(trades, label: str, log_fn=None):
    """
    Log shadow trade outcomes to terminal.
    Shows win rate and individual results for model validation.
    """
    if not trades:
        return
    if log_fn is None:
        log_fn = log.info
    wins = [t for t in trades if t.outcome.startswith('WIN')]
    losses = [t for t in trades if t.outcome.startswith('LOSS')]
    total_pips = sum(t.profit_pips for t in trades)
    wr = len(wins) / len(trades) * 100 if trades else 0
    log_fn(f"  [{label}] Shadow Results: {len(trades)} trades | "
           f"{len(wins)}W/{len(losses)}L ({wr:.1f}% WR) | "
           f"PnL: {total_pips:+.1f} pips")
    # Log each shadow trade
    for t in trades:
        emoji = '+' if t.outcome.startswith('WIN') else '-'
        log_fn(f"  [{label}]   {emoji} {t.strategy} {t.direction} "
               f"entry={t.entry_price:.5f} exit={t.exit_price:.5f} "
               f"pips={t.profit_pips:+.1f} R={t.profit_r:+.2f} "
               f"outcome={t.outcome}")


log = get_logger(__name__)


def _open_signal_shadow(tracker, signal, symbol, current_time,
                        bar_close, pip_size, total_slippage,
                        market_report, market_state):
    """
    Open a shadow trade from a signal dict.
    Returns True if the shadow trade was successfully opened.
    """
    sl_p = signal.get('sl_pips', 0)
    tp_p = signal.get('tp1_pips', 0) or signal.get('tp_pips', 0)
    if sl_p <= 0 or tp_p <= 0:
        return False

    sh_entry = float(bar_close)
    if signal['direction'] == 'BUY':
        sh_entry += total_slippage * pip_size
    else:
        sh_entry -= total_slippage * pip_size

    tracker.open_trade(
        symbol=symbol,
        direction=signal['direction'],
        strategy=signal.get('strategy', 'UNKNOWN'),
        entry_time=current_time,
        entry_price=sh_entry,
        sl_price=0, tp_price=0,
        sl_pips=sl_p, tp_pips=tp_p,
        score=signal.get('score', 0),
        confluence=signal.get('confluence', []),
        session=market_report.get('session', 'UNKNOWN'),
        market_state=market_state,
        agreement_groups=1,
        atr_value=0.0,
    )

    st = tracker.open_trades[-1]
    if signal['direction'] == 'BUY':
        st.sl_price = st.entry_price - st.sl_pips * pip_size
        st.tp_price = st.entry_price + st.tp_pips * pip_size
    else:
        st.sl_price = st.entry_price + st.sl_pips * pip_size
        st.tp_price = st.entry_price - st.tp_pips * pip_size
    # Fix: open_trade() sets original_sl/tp from the passed 0 values;
    # correct them so dynamic TP extension works properly
    st.original_sl_price = st.sl_price
    st.original_tp_price = st.tp_price

    return True


@dataclass
class BacktestConfig:
    """Configuration for a single backtest run."""
    symbol: str
    start_date: datetime.datetime
    end_date: datetime.datetime
    scan_every_n_bars: int = 15   # Scan every 15 M1 bars = M15 close
    max_trades: int = 9999        # Max trades to execute (0 = unlimited)
    strategies_filter: list = None  # Empty = all
    relaxed_mode: bool = False     # Lower gates for data collection
    store_db: bool = False         # Store trades/signals in MySQL
    run_id: str = 'default'       # Run identifier for DB grouping
    use_model: bool = False        # Use trained XGBoost model as additional gate
    use_strategy_models: bool = False  # Use Layer 1 per-strategy models
    unlimited_positions: bool = False  # Remove max open position limits


def _build_master_report(symbol: str,
                         market_report: dict,
                         smc_report: dict,
                         flow_data: dict) -> dict:
    """
    Build a master_report dict that matches the live master_scanner.py schema.
    This ensures all strategies receive the same data structure as in live trading.
    """
    market_score = market_report.get('trade_score', 0)
    smc_score = smc_report.get('smc_score', 0)

    # HTF approved?
    htf = smc_report.get('htf_alignment', {})
    htf_approved = htf.get('approved', True)
    htf_penalty = 0 if htf_approved else 30

    # Premium/discount penalty
    pd = smc_report.get('premium_discount', {})
    pd_bias = str(pd.get('bias', ''))
    market_bias = str(market_report.get('combined_bias', 'NEUTRAL'))
    smc_bias = str(smc_report.get('smc_bias', 'NEUTRAL'))
    pd_penalty = 15 if (
        (market_bias == "BULLISH" and pd_bias == "SELL") or
        (market_bias == "BEARISH" and pd_bias == "BUY")
    ) else 0

    # Final score (same formula as live)
    base_score = (market_score * 0.50) + (smc_score * 0.50)
    final_score = max(0, round(base_score - htf_penalty - pd_penalty))

    # Combined bias
    if market_bias == smc_bias and market_bias != "NEUTRAL":
        combined_bias = market_bias
        bias_confidence = "HIGH"
    elif market_bias == "NEUTRAL" or smc_bias == "NEUTRAL":
        combined_bias = market_bias if market_bias != "NEUTRAL" else smc_bias
        bias_confidence = "MODERATE"
    else:
        combined_bias = "CONFLICTED"
        bias_confidence = "LOW"

    # Sweep alignment
    last_sweep = smc_report.get('last_sweep', {})
    sweep_aligned = last_sweep.get('bias') == combined_bias if last_sweep else False

    # Order flow / volume / momentum shortcuts
    of_imb = flow_data.get('order_flow_imbalance', {})
    volume_surge = flow_data.get('volume_surge', {})
    momentum = flow_data.get('momentum', {})

    # Session
    session = market_report.get('session', 'UNKNOWN')

    return {
        "symbol": symbol,
        "timestamp": "",  # filled later
        "session": session,
        "session_quality": 1.0,
        "day_trade_ok": True,
        "block_reason": "",
        "market_report": market_report,
        "smc_report": smc_report,
        "fractal_alignment": {"score": 0, "passed": True},
        "market_score": market_score,
        "smc_score": smc_score,
        "final_score": final_score,
        "market_bias": market_bias,
        "smc_bias": smc_bias,
        "combined_bias": combined_bias,
        "bias_confidence": bias_confidence,
        "market_state": market_report.get('market_state', 'BALANCED'),
        "htf_approved": htf_approved,
        "pd_penalty": pd_penalty,
        "sweep_aligned": sweep_aligned,
        "recommendation": {"action": "TRADE", "reason": "backtest"},
        "order_flow_imbalance": of_imb,
        "volume_surge": volume_surge,
        "momentum": momentum,
        "scalping_signal": {"status": "INSUFFICIENT", "gates_passed": 0},
        "volume_profile": market_report.get('profile', {}),
        "vwap_context": market_report.get('vwap', {}),
    }


def run_backtest(config: BacktestConfig) -> dict:
    """
    Run a full backtest on one symbol.
    Returns performance summary dict.
    """
    symbol = config.symbol
    log.info(f"")
    log.info(f"{'='*60}")
    log.info(f"  BACKTEST v2.0: {symbol}")
    log.info(f"  Period: {config.start_date.date()} to {config.end_date.date()}")
    log.info(f"  Features: PartialTP={PARTIAL_TP_ENABLED} "
             f"Trail={ATR_TRAIL_ENABLED} "
             f"ExtTP={DYNAMIC_TP_EXTENSION_ENABLED} "
             f"DynamicSize={DYNAMIC_SIZING_ENABLED}")
    log.info(f"{'='*60}")

    start_time = time_mod.time()

    # ── Load historical data ──────────────────────────────
    data = load_all_data(symbol, config.start_date, config.end_date, CACHE_DIR)
    if not data:
        log.error(f"No data loaded for {symbol}")
        return {}

    df_m1 = data['M1']
    df_m5 = data['M5']
    df_m15 = data['M15']
    df_h1 = data['H1']
    df_h4 = data['H4']

    log.info(f"  M1: {len(df_m1)} bars, H1: {len(df_h1)} bars, H4: {len(df_h4)} bars")

    # ── Pip size and pip value ────────────────────────────
    pip_size = get_pip_size(symbol)
    pip_value = PIP_VALUE_PER_LOT.get(symbol, PIP_VALUE_PER_LOT['DEFAULT'])

    spread = AVG_SPREAD_PIPS.get(symbol, AVG_SPREAD_PIPS['DEFAULT'])
    total_slippage = spread + SLIPPAGE_PIPS  # Total cost in pips

    log.info(f"  Pip size: {pip_size}, Pip value: ${pip_value}/lot/pip")
    log.info(f"  Spread: {spread}p, Slippage: {SLIPPAGE_PIPS}p, Total cost: {total_slippage}p")

    # ── Initialize trade tracker ──────────────────────────
    max_open = 9999 if config.unlimited_positions else MAX_OPEN_TRADES
    max_per_sym = 9999 if config.unlimited_positions else MAX_PER_SYMBOL
    tracker = TradeTracker(
        starting_balance=STARTING_BALANCE,
        pip_value_per_lot=pip_value,
        max_open=max_open,
        max_per_symbol=max_per_sym,
        partial_tp_enabled=PARTIAL_TP_ENABLED,
        atr_trail_enabled=ATR_TRAIL_ENABLED,
        dynamic_tp_enabled=DYNAMIC_TP_EXTENSION_ENABLED,
        dynamic_sizing_enabled=DYNAMIC_SIZING_ENABLED,
        base_risk_percent=BASE_RISK_PERCENT,
    )

    # ── Import strategy engine ───────────────────────────
    from strategies.strategy_engine import _run_one_strategy, _get_strategy_group
    from strategies.strategy_registry import get_active_strategies
    from data_layer.feature_store import store

    # Determine which strategies to run
    if config.strategies_filter:
        active_strategies = [s for s in config.strategies_filter
                             if s in get_active_strategies()]
    else:
        active_strategies = get_active_strategies()

    log.info(f"  Strategies: {', '.join(active_strategies)}")
    log.info(f"")

    # ── Main backtest loop ────────────────────────────────
    total_m1_bars = len(df_m1)
    signals_found = 0
    signals_blocked_consensus = 0
    signals_blocked_gate = 0
    signals_blocked_score = 0
    signals_blocked_choppy = 0
    signals_blocked_bias = 0
    signals_blocked_confluence = 0
    signals_no_strategy = 0
    trades_executed = 0

    # ── Store feature snapshots per trade (for DB) ─────
    trade_reports = {}  # ticket -> {master_report, market_report, smc_report, flow}

    # ── Shadow trade placeholders (v3.2) ─────────────────
    # Initialized after ML gate check below.
    shadow_tracker = None
    shadow_reports = {}
    shadow_count = 0

    # ── Layer 1 Strategy Model system ────────────────────
    strat_model_mgr = None
    strat_model_shadow_tracker = None
    strat_model_shadow_reports = {}
    strat_model_shadow_count = 0
    strat_model_reject_count = 0

    # ── ML Gate prediction tracking ──────────────────────
    ml_predictions = []  # Track all predictions for distribution analysis

    # ── Load ML Gate model if --use-model ────────────────
    ml_gate_active = False
    model_blocked_count = 0
    model_caution_count = 0
    if config.use_model:
        try:
            from ai_engine.ml_gate import is_model_trained, collect_all_strategy_scores
            if is_model_trained():
                ml_gate_active = True
                log.info(f"  [ML_GATE] Strategy-Informed ML v3.0 loaded — "
                         f"replacing consensus gates")
            else:
                log.warning(f"  [ML_GATE] --use-model but no trained model found — "
                            f"falling back to rule-based gates")
        except Exception as e:
            log.warning(f"  [ML_GATE] Failed to load: {e}")

    # ── Shadow trade system (v3.3) ─────────────────────
    # Simulate CAUTION signals (0.0 <= R < 0.5) without executing.
    # Uses FULL TradeTracker (same partial TP, trailing, TP extension
    # as real trades) so shadow outcomes match real-world outcomes.
    # Only active when model is loaded AND we're storing to DB.
    if config.store_db and ml_gate_active:
        shadow_tracker = TradeTracker(
            starting_balance=STARTING_BALANCE,
            pip_value_per_lot=pip_value,
            max_open=9999, max_per_symbol=9999,
            partial_tp_enabled=PARTIAL_TP_ENABLED,
            atr_trail_enabled=ATR_TRAIL_ENABLED,
            dynamic_tp_enabled=DYNAMIC_TP_EXTENSION_ENABLED,
            dynamic_sizing_enabled=DYNAMIC_SIZING_ENABLED,
            base_risk_percent=BASE_RISK_PERCENT,
        )

    # ── Load Layer 1 Strategy Models if --use-strategy-models ──
    if config.use_strategy_models:
        try:
            from ai_engine.strategy_model import get_strategy_model_manager
            strat_model_mgr = get_strategy_model_manager()
            if strat_model_mgr._active:
                active_list = list(strat_model_mgr._active)
                log.info(f"  [L1_STRAT_MODEL] Loaded {len(active_list)} strategy models: "
                         f"{', '.join(active_list)}")
                # Create shadow tracker for L1 rejections
                if config.store_db:
                    strat_model_shadow_tracker = TradeTracker(
                        starting_balance=STARTING_BALANCE,
                        pip_value_per_lot=pip_value,
                        max_open=9999, max_per_symbol=9999,
                        partial_tp_enabled=PARTIAL_TP_ENABLED,
                        atr_trail_enabled=ATR_TRAIL_ENABLED,
                        dynamic_tp_enabled=DYNAMIC_TP_EXTENSION_ENABLED,
                        dynamic_sizing_enabled=DYNAMIC_SIZING_ENABLED,
                        base_risk_percent=BASE_RISK_PERCENT,
                    )
            else:
                log.info(f"  [L1_STRAT_MODEL] No trained strategy models found — skipping L1")
                strat_model_mgr = None
        except Exception as e:
            log.warning(f"  [L1_STRAT_MODEL] Failed to load: {e}")
            strat_model_mgr = None

    # ── All-Signals Shadow Tracker ─────────────────────────
    # Shadows ALL qualifying signals (not just model rejects) so
    # both L1 and L2 models can learn from every signal's outcome.
    all_signals_shadow_tracker = None
    all_signals_shadow_reports = {}
    all_signals_shadow_count = 0

    if config.store_db:
        all_signals_shadow_tracker = TradeTracker(
            starting_balance=STARTING_BALANCE,
            pip_value_per_lot=pip_value,
            max_open=9999, max_per_symbol=9999,
            partial_tp_enabled=PARTIAL_TP_ENABLED,
            atr_trail_enabled=ATR_TRAIL_ENABLED,
            dynamic_tp_enabled=DYNAMIC_TP_EXTENSION_ENABLED,
            dynamic_sizing_enabled=DYNAMIC_SIZING_ENABLED,
            base_risk_percent=BASE_RISK_PERCENT,
        )
        log.info(f"  [ALL_SIGNALS_SHADOW] Tracking all qualifying signals for training data")

    scan_bar = config.scan_every_n_bars

    for bar_idx in range(scan_bar, total_m1_bars):
        current_bar = df_m1.iloc[bar_idx]
        current_time = current_bar['time']

        # Only scan every N bars (M15 frequency)
        if bar_idx % config.scan_every_n_bars != 0:
            # But still check exits on every bar
            tracker.check_exits(
                current_time,
                float(current_bar['high']),
                float(current_bar['low']),
                float(current_bar['close']),
                pip_size,
                pip_value,
            )
            # Also check shadow trade exits
            if shadow_tracker is not None:
                shadow_tracker.check_exits(
                    current_time,
                    float(current_bar['high']),
                    float(current_bar['low']),
                    float(current_bar['close']),
                    pip_size,
                    pip_value,
                )
            # Also check L1 strategy model shadow exits
            if strat_model_shadow_tracker is not None:
                strat_model_shadow_tracker.check_exits(
                    current_time,
                    float(current_bar['high']),
                    float(current_bar['low']),
                    float(current_bar['close']),
                    pip_size,
                    pip_value,
                )
            # Also check all-signals shadow exits (CRITICAL: was missing)
            if all_signals_shadow_tracker is not None:
                all_signals_shadow_tracker.check_exits(
                    current_time,
                    float(current_bar['high']),
                    float(current_bar['low']),
                    float(current_bar['close']),
                    pip_size,
                    pip_value,
                )
            continue

        # ── Skip if we already have too many trades ───────
        if config.max_trades > 0 and trades_executed >= config.max_trades:
            break

        # ── Get candles UP TO this point (simulates live) ──
        sliced = get_candles_at_time(data, current_time, count=200)

        s_m1  = sliced['M1']
        s_m5  = sliced['M5']
        s_m15 = sliced['M15']
        s_h1  = sliced['H1']
        s_h4  = sliced['H4']

        if len(s_m15) < 30 or len(s_h1) < 30 or len(s_h4) < 10:
            continue

        # ── Build order flow from M1 data ─────────────────
        flow = simulate_order_flow_from_full(s_m1, pip_size)

        # ── Build SMC report ──────────────────────────────
        smc_report = build_smc_report(s_h1, s_h4, current_time)

        # ── Build market report ───────────────────────────
        market_report = build_market_report(s_m15, flow, smc_report, symbol)

        # ── Build master_report (matches live schema) ─────
        master_report = _build_master_report(symbol, market_report, smc_report, flow)
        master_report['timestamp'] = current_time.strftime('%Y-%m-%d %H:%M:%S')

        # ── Update feature_store (needed by LIQUIDITY_SWEEP) ──
        market_data_for_store = {
            'current_price': float(current_bar['close']),
            'atr': float(s_m15.iloc[-1].get('atr', 0)) if len(s_m15) > 0 else 0,
            'delta': flow.get('delta', {}),
            'rolling_delta': flow.get('rolling_delta', {}),
            'order_flow_imbalance': flow.get('order_flow_imbalance', {}),
            'volume_surge': flow.get('volume_surge', {}),
            'momentum': flow.get('momentum', {}),
            'vwap': market_report.get('vwap', {}),
            'profile': market_report.get('profile', {}),
        }
        store.update_symbol_features(symbol, market_data_for_store, smc_report)

        # ── Gate 0: Master score minimum ─────────────────
        final_score = master_report.get('final_score', 0)
        score_gate = RELAXED_MIN_SCORE if config.relaxed_mode else MASTER_MIN_SCORE
        if final_score < score_gate:
            signals_blocked_score += 1
            continue

        # ── Gate 1: Institutional confirmation (SOFT) ──
        # No longer a hard block — passes through for ML Gate to evaluate.
        # ML Gate learns when institutional confirmation matters via features.
        imb_strength = flow.get('order_flow_imbalance', {}).get('strength', 'NONE')
        surge_active = flow.get('volume_surge', {}).get('surge_detected', False)
        has_institutional = imb_strength in ('STRONG', 'EXTREME') or surge_active

        if not has_institutional:
            signals_blocked_gate += 1
            # SOFT: no continue — signal passes to ML Gate

        # ── Gate 2: Choppy market ─────────────────────────
        is_choppy = flow.get('momentum', {}).get('is_choppy', True)
        if is_choppy and not surge_active:
            signals_blocked_choppy += 1
            continue

        # ── Run strategies ────────────────────────────────
        market_state = master_report.get('market_state', 'BALANCED')
        session = master_report.get('session', 'UNKNOWN')
        combined_bias = master_report.get('combined_bias', 'NEUTRAL')

        # ── ML Gate path: collect ALL strategy scores ─────
        # Initialize L1 model vars (used in both ML Gate and rule-based paths)
        strat_model_verdict = None
        strat_model_predicted_r = None

        if ml_gate_active:
            try:
                all_scores = collect_all_strategy_scores(
                    symbol, s_m1, s_m5, s_m15, s_h1, s_h4,
                    smc_report, market_report,
                    market_state, session, master_report)

                # Pick the best signal from strategies that fired
                non_zero_scores = {k: v for k, v in all_scores.items()
                                   if v and v > 0}
                if not non_zero_scores:
                    signals_no_strategy += 1
                    continue

                # Get the actual signal dict for the best strategy
                best_strat_name = max(non_zero_scores,
                                      key=non_zero_scores.get)
                best_signal = None
                try:
                    best_signal = _run_one_strategy(
                        best_strat_name, symbol,
                        s_m1, s_m5, s_m15, s_h1, s_h4,
                        smc_report, market_report,
                        market_state, session, master_report,
                        relaxed=config.relaxed_mode)
                except Exception:
                    pass

                if best_signal is None:
                    signals_no_strategy += 1
                    continue

                signals_found += 1

                # ── Shadow ALL qualifying strategy signals (not just best) ──
                # This gives both L1 and L2 models maximum training data.
                # The best signal will be handled separately (real trade or model shadow).
                if all_signals_shadow_tracker is not None:
                    for strat_name, score in all_scores.items():
                        if score <= 0 or strat_name == best_strat_name:
                            continue
                        try:
                            sig = _run_one_strategy(
                                strat_name, symbol,
                                s_m1, s_m5, s_m15, s_h1, s_h4,
                                smc_report, market_report,
                                market_state, session, master_report,
                                relaxed=config.relaxed_mode)
                            if sig is None:
                                continue
                            if _open_signal_shadow(
                                    all_signals_shadow_tracker, sig,
                                    symbol, current_time,
                                    current_bar['close'], pip_size,
                                    total_slippage, market_report,
                                    market_state):
                                all_signals_shadow_count += 1
                                all_signals_shadow_reports[
                                    all_signals_shadow_tracker.ticket_counter] = {
                                    'master_report': master_report,
                                    'market_report': market_report,
                                    'smc_report': smc_report,
                                    'flow': flow,
                                    'strategy_scores': all_scores,
                                    'predicted_r': None,
                                    'signal': sig,
                                }
                        except Exception:
                            pass

                # ── Layer 1 Strategy Model: per-strategy PASS/REJECT ──
                if strat_model_mgr is not None and strat_model_mgr.has_model(best_strat_name):
                    try:
                        from ai_engine.ml_gate import extract_features
                        l1_features = extract_features(
                            best_signal, master_report, market_report,
                            smc_report, flow, all_scores, symbol, spread)
                        if l1_features is not None:
                            l1_result = strat_model_mgr.evaluate_signal(
                                best_strat_name, l1_features)
                            strat_model_verdict = l1_result.get('verdict', 'NO_MODEL')
                            strat_model_predicted_r = l1_result.get('predicted_r', 0.0)

                            if strat_model_verdict == 'REJECT':
                                strat_model_reject_count += 1
                                best_signal['strategy_model_verdict'] = strat_model_verdict
                                best_signal['strategy_model_predicted_r'] = strat_model_predicted_r

                                # Shadow simulate the L1 rejection
                                if strat_model_shadow_tracker is not None:
                                    sl_p = best_signal.get('sl_pips', 0)
                                    tp_p = (best_signal.get('tp1_pips', 0)
                                            or best_signal.get('tp_pips', 0))
                                    if sl_p > 0 and tp_p > 0:
                                        sh_entry = float(current_bar['close'])
                                        if best_signal['direction'] == 'BUY':
                                            sh_entry += total_slippage * pip_size
                                        else:
                                            sh_entry -= total_slippage * pip_size
                                        strat_model_shadow_tracker.open_trade(
                                            symbol=symbol,
                                            direction=best_signal['direction'],
                                            strategy=best_strat_name,
                                            entry_time=current_time,
                                            entry_price=sh_entry,
                                            sl_price=0, tp_price=0,
                                            sl_pips=sl_p, tp_pips=tp_p,
                                            score=best_signal.get('score', 0),
                                            confluence=best_signal.get('confluence', []),
                                            session=market_report.get('session', 'UNKNOWN'),
                                            market_state=market_state,
                                            agreement_groups=1,
                                            atr_value=0.0,
                                        )
                                        st = strat_model_shadow_tracker.open_trades[-1]
                                        if best_signal['direction'] == 'BUY':
                                            st.sl_price = st.entry_price - st.sl_pips * pip_size
                                            st.tp_price = st.entry_price + st.tp_pips * pip_size
                                        else:
                                            st.sl_price = st.entry_price + st.sl_pips * pip_size
                                            st.tp_price = st.entry_price - st.tp_pips * pip_size
                                        st.original_sl_price = st.sl_price
                                        st.original_tp_price = st.tp_price
                                        strat_model_shadow_count += 1
                                        strat_model_shadow_reports[
                                            strat_model_shadow_tracker.ticket_counter] = {
                                            'master_report': master_report,
                                            'market_report': market_report,
                                            'smc_report': smc_report,
                                            'flow': flow,
                                            'strategy_scores': all_scores or {},
                                            'predicted_r': predicted_r if 'predicted_r' in dir() else None,
                                            'strategy_model_verdict': strat_model_verdict,
                                            'strategy_model_predicted_r': strat_model_predicted_r,
                                            'signal': best_signal,  # Full signal with strategy features
                                        }

                                if strat_model_shadow_count <= 3 or strat_model_shadow_count % 25 == 0:
                                    log.info(f"  [L1_STRAT_MODEL] REJECT {best_strat_name} "
                                             f"{best_signal['direction']} "
                                             f"R={strat_model_predicted_r:.2f}")
                                continue  # L1 REJECT → shadow, don't proceed to L2

                            # L1 PASS → proceed to Layer 2 ML Gate
                            if strat_model_reject_count <= 3 or strat_model_reject_count % 50 == 0:
                                log.info(f"  [L1_STRAT_MODEL] PASS {best_strat_name} "
                                         f"{best_signal['direction']} "
                                         f"R={strat_model_predicted_r:.2f}")
                    except Exception as e:
                        log.debug(f"  [L1_STRAT_MODEL] Error: {e}")

                # ── ML Gate (Layer 2): score the signal ──────────────
                try:
                    from ai_engine.ml_gate import score_signal
                    ml_result = score_signal(
                        best_signal, master_report, market_report,
                        smc_report, flow,
                        all_strategy_scores=all_scores,
                        symbol=symbol,
                        spread_pips=spread,
                    )

                    predicted_r = ml_result.get('predicted_r', 0.0)
                    recommendation = ml_result.get('recommendation', 'SKIP')

                    best_signal['model_predicted_r'] = predicted_r
                    best_signal['ml_recommendation'] = recommendation

                    # Track prediction for distribution analysis
                    ml_predictions.append({
                        'strategy': best_strat_name,
                        'direction': best_signal['direction'],
                        'predicted_r': predicted_r,
                        'recommendation': recommendation,
                    })

                    # ── NEUTRAL = features failed → treat as SKIP (shadow) ──
                    # If the model can't score the signal, don't execute it blind.
                    if recommendation == 'NEUTRAL':
                        recommendation = 'SKIP'
                        log.warning(f"  [ML_GATE] NEUTRAL (features failed) → SKIP+SHADOW "
                                    f"{best_strat_name} {best_signal['direction']}")

                    if recommendation == 'SKIP':
                        model_blocked_count += 1
                    elif recommendation == 'CAUTION':
                        model_caution_count += 1

                    # ── CAUTION or SKIP → SHADOW trade (simulate, don't execute) ──
                    # v3.3: ALL non-TAKE signals are shadowed for training data.
                    # - CAUTION (0.0 <= R < 0.5): model is unsure
                    # - SKIP (R < 0.0): model predicts loss — but shadow verifies
                    # This gives the model learning signal from its rejections too.
                    if recommendation in ('CAUTION', 'SKIP'):
                        if shadow_tracker is not None:
                            sl_p = best_signal.get('sl_pips', 0)
                            tp_p = best_signal.get('tp1_pips', 0) or best_signal.get('tp_pips', 0)
                            if sl_p > 0 and tp_p > 0:
                                sh_entry = float(current_bar['close'])
                                if best_signal['direction'] == 'BUY':
                                    sh_entry += total_slippage * pip_size
                                else:
                                    sh_entry -= total_slippage * pip_size
                                shadow_tracker.open_trade(
                                    symbol=symbol,
                                    direction=best_signal['direction'],
                                    strategy=best_strat_name,
                                    entry_time=current_time,
                                    entry_price=sh_entry,
                                    sl_price=0, tp_price=0,
                                    sl_pips=sl_p, tp_pips=tp_p,
                                    score=best_signal.get('score', 0),
                                    confluence=best_signal.get('confluence', []),
                                    session=market_report.get('session', 'UNKNOWN'),
                                    market_state=market_state,
                                    agreement_groups=1,
                                    atr_value=0.0,
                                )
                                # Fix SL/TP prices after open
                                st = shadow_tracker.open_trades[-1]
                                if best_signal['direction'] == 'BUY':
                                    st.sl_price = st.entry_price - st.sl_pips * pip_size
                                    st.tp_price = st.entry_price + st.tp_pips * pip_size
                                else:
                                    st.sl_price = st.entry_price + st.sl_pips * pip_size
                                    st.tp_price = st.entry_price - st.tp_pips * pip_size
                                shadow_count += 1
                                shadow_reports[shadow_tracker.ticket_counter] = {
                                    'master_report': master_report,
                                    'market_report': market_report,
                                    'smc_report': smc_report,
                                    'flow': flow,
                                    'strategy_scores': all_scores or {},
                                    'predicted_r': predicted_r,
                                    'signal': best_signal,  # Full signal with strategy features
                                }
                        # Log sparingly to avoid spam
                        log_limit = 3 if recommendation == 'CAUTION' else 5
                        if shadow_count <= log_limit or shadow_count % 25 == 0:
                            log.info(f"  [ML_GATE] SHADOW ({recommendation}) {best_strat_name} "
                                     f"{best_signal['direction']} "
                                     f"R={predicted_r:.2f}")
                        continue  # Don't execute as real trade

                    # Model says TAKE — proceed with real execution
                    if model_blocked_count % 20 == 0 or model_blocked_count <= 3:
                        log.info(f"  [ML_GATE] {recommendation} {best_strat_name} "
                                 f"{best_signal['direction']} "
                                 f"R={predicted_r:.2f}")

                    # Build final_signals compatible with existing code
                    best = best_signal
                    best['symbol'] = symbol
                    best['group'] = _get_strategy_group(best_strat_name)
                    final_signals = [best]
                    final_groups = {_get_strategy_group(best_strat_name)}
                    confluence = best.get('confluence', [])

                except Exception as e:
                    log.debug(f"  [ML_GATE] Scoring error: {e}")
                    continue

            except Exception as e:
                log.debug(f"  [ML_GATE] Error: {e}")

        # ── Rule-based path (original gates, when no ML model) ─
        if not ml_gate_active:
            signals = []
            for strategy_name in active_strategies:
                try:
                    signal = _run_one_strategy(
                        strategy_name, symbol,
                        s_m1, s_m5, s_m15, s_h1, s_h4,
                        smc_report, market_report,
                        market_state, session, master_report,
                        relaxed=config.relaxed_mode)

                    if signal is None:
                        continue

                    direction = str(signal.get('direction', ''))
                    score = signal.get('score', 0)

                    # Apply per-strategy minimum score
                    from strategies.strategy_engine import STRATEGY_MIN_SCORES
                    min_score = STRATEGY_MIN_SCORES.get(strategy_name, 70)
                    if score < min_score:
                        continue

                    signal['symbol'] = symbol
                    signal['group'] = _get_strategy_group(strategy_name)
                    signals.append(signal)

                except Exception as e:
                    log.debug(f"  [{symbol}] {strategy_name} error: {e}")
                    continue

            if not signals:
                continue

            signals_found += 1

            # ── Gate 3: Bias direction filter ─────────────
            if not config.relaxed_mode:
                if combined_bias == 'BULLISH':
                    signals = [s for s in signals
                               if s['direction'] == 'BUY']
                elif combined_bias == 'BEARISH':
                    signals = [s for s in signals
                               if s['direction'] == 'SELL']
                if not signals:
                    signals_blocked_bias += 1
                    continue

            # ── Gate 4: Multi-group consensus ──────────────
            buy_groups = set(s['group'] for s in signals
                             if s['direction'] == 'BUY')
            sell_groups = set(s['group'] for s in signals
                              if s['direction'] == 'SELL')

            min_groups = (RELAXED_CONSENSUS_GROUPS
                          if config.relaxed_mode else 2)

            if (len(buy_groups) >= min_groups
                    and len(buy_groups) >= len(sell_groups)):
                final_signals = [s for s in signals
                                 if s['direction'] == 'BUY']
                final_groups = buy_groups
            elif len(sell_groups) >= min_groups:
                final_signals = [s for s in signals
                                 if s['direction'] == 'SELL']
                final_groups = sell_groups
            else:
                signals_blocked_consensus += 1
                continue

            # ── Best signal ──────────────────────────────
            best = max(final_signals, key=lambda s: s['score'])

            # ── Gate 5: Confluence check ──────────────────
            confluence = best.get('confluence', [])
            min_conv = (RELAXED_MIN_CONFLUENCE
                        if config.relaxed_mode else MIN_CONFLUENCE)
            if len(confluence) < min_conv:
                signals_blocked_confluence += 1
                continue

            # Collect ALL strategy scores for ML training data
            if config.store_db:
                try:
                    from ai_engine.ml_gate import collect_all_strategy_scores
                    all_scores = collect_all_strategy_scores(
                        symbol, s_m1, s_m5, s_m15, s_h1, s_h4,
                        smc_report, market_report,
                        market_state, session, master_report)
                except Exception:
                    all_scores = None
            else:
                all_scores = None

            # ── Shadow ALL non-best signals from this scan bar ──
            if all_signals_shadow_tracker is not None:
                for sig in final_signals:
                    if sig is best:
                        continue  # Best signal handled separately
                    if _open_signal_shadow(
                            all_signals_shadow_tracker, sig,
                            symbol, current_time,
                            current_bar['close'], pip_size,
                            total_slippage, market_report,
                            market_state):
                        all_signals_shadow_count += 1
                        all_signals_shadow_reports[
                            all_signals_shadow_tracker.ticket_counter] = {
                            'master_report': master_report,
                            'market_report': market_report,
                            'smc_report': smc_report,
                            'flow': flow,
                            'strategy_scores': all_scores or {},
                            'predicted_r': None,
                            'signal': sig,
                        }

            # ── Layer 1 Strategy Model: filter best signal ──
            best_strat_name = best.get('strategy', '')
            if strat_model_mgr is not None and strat_model_mgr.has_model(best_strat_name):
                try:
                    from ai_engine.ml_gate import extract_features
                    l1_features = extract_features(
                        best, master_report, market_report,
                        smc_report, flow, all_scores or {}, symbol, spread)
                    if l1_features is not None:
                        l1_result = strat_model_mgr.evaluate_signal(
                            best_strat_name, l1_features)
                        strat_model_verdict = l1_result.get('verdict', 'NO_MODEL')
                        strat_model_predicted_r = l1_result.get('predicted_r', 0.0)

                        if strat_model_verdict == 'REJECT':
                            strat_model_reject_count += 1
                            best['strategy_model_verdict'] = strat_model_verdict
                            best['strategy_model_predicted_r'] = strat_model_predicted_r

                            # Shadow simulate the L1 rejection
                            if strat_model_shadow_tracker is not None:
                                sl_p = best.get('sl_pips', 0)
                                tp_p = (best.get('tp1_pips', 0)
                                        or best.get('tp_pips', 0))
                                if sl_p > 0 and tp_p > 0:
                                    sh_entry = float(current_bar['close'])
                                    if best['direction'] == 'BUY':
                                        sh_entry += total_slippage * pip_size
                                    else:
                                        sh_entry -= total_slippage * pip_size
                                    strat_model_shadow_tracker.open_trade(
                                        symbol=symbol,
                                        direction=best['direction'],
                                        strategy=best_strat_name,
                                        entry_time=current_time,
                                        entry_price=sh_entry,
                                        sl_price=0, tp_price=0,
                                        sl_pips=sl_p, tp_pips=tp_p,
                                        score=best.get('score', 0),
                                        confluence=best.get('confluence', []),
                                        session=market_report.get('session', 'UNKNOWN'),
                                        market_state=market_state,
                                        agreement_groups=1,
                                        atr_value=0.0,
                                    )
                                    st = strat_model_shadow_tracker.open_trades[-1]
                                    if best['direction'] == 'BUY':
                                        st.sl_price = st.entry_price - st.sl_pips * pip_size
                                        st.tp_price = st.entry_price + st.tp_pips * pip_size
                                    else:
                                        st.sl_price = st.entry_price + st.sl_pips * pip_size
                                        st.tp_price = st.entry_price - st.tp_pips * pip_size
                                    st.original_sl_price = st.sl_price
                                    st.original_tp_price = st.tp_price
                                    strat_model_shadow_count += 1
                                    strat_model_shadow_reports[
                                        strat_model_shadow_tracker.ticket_counter] = {
                                        'master_report': master_report,
                                        'market_report': market_report,
                                        'smc_report': smc_report,
                                        'flow': flow,
                                        'strategy_scores': all_scores or {},
                                        'predicted_r': None,
                                        'strategy_model_verdict': strat_model_verdict,
                                        'strategy_model_predicted_r': strat_model_predicted_r,
                                        'signal': best,
                                    }

                            if strat_model_shadow_count <= 3 or strat_model_shadow_count % 25 == 0:
                                log.info(f"  [L1_STRAT_MODEL] REJECT {best_strat_name} "
                                         f"{best['direction']} "
                                         f"R={strat_model_predicted_r:.2f}")
                            continue  # L1 REJECT → shadow, skip execution

                        # L1 PASS
                        if strat_model_reject_count <= 3 or strat_model_reject_count % 50 == 0:
                            log.info(f"  [L1_STRAT_MODEL] PASS {best_strat_name} "
                                     f"{best['direction']} "
                                     f"R={strat_model_predicted_r:.2f}")
                except Exception as e:
                    log.debug(f"  [L1_STRAT_MODEL] Error: {e}")

        # ── Check if we can open a trade ─────────────────
        if not tracker.can_open(symbol):
            continue

        # ── Execute trade ────────────────────────────────
        entry_price = float(current_bar['close'])

        # Apply spread + slippage
        if best['direction'] == 'BUY':
            entry_price += total_slippage * pip_size
        else:
            entry_price -= total_slippage * pip_size

        sl_pips = best.get('sl_pips', 0)
        tp1_pips = best.get('tp1_pips', 0)

        # Also check tp_pips (some strategies use this field)
        tp_pips = tp1_pips or best.get('tp_pips', 0)

        # Validate SL/TP
        if sl_pips <= 0 or tp_pips <= 0:
            continue

        min_rr = RELAXED_MIN_RR_RATIO if config.relaxed_mode else MIN_RR_RATIO
        if tp_pips / sl_pips < min_rr:
            continue

        # Set actual SL/TP based on entry price
        if best['direction'] == 'BUY':
            sl_price = entry_price - sl_pips * pip_size
            tp_price = entry_price + tp_pips * pip_size
        else:
            sl_price = entry_price + sl_pips * pip_size
            tp_price = entry_price - tp_pips * pip_size

        # Get ATR for trailing (from M15)
        atr_value = float(s_m15.iloc[-1].get('atr', 0)) if len(s_m15) > 0 else 0.0

        # Agreement groups count for dynamic sizing
        agreement_groups = len(final_groups)

        tracker.open_trade(
            symbol=symbol,
            direction=best['direction'],
            strategy=best.get('strategy', 'UNKNOWN'),
            entry_time=current_time,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            score=best.get('score', 0),
            confluence=confluence,
            session=market_report.get('session', 'UNKNOWN'),
            market_state=market_state,
            agreement_groups=agreement_groups,
            atr_value=atr_value,
        )
        trades_executed += 1

        # ── Save feature snapshot at trade entry (for DB storage) ─
        # Compute Fibonacci confluence for this trade
        fib_data = {}
        if config.store_db:
            try:
                from backtest.fib_builder import build_fib_report, check_fib_confluence
                fib_report = build_fib_report(df_h1=s_h1, df_h4=s_h4,
                                             current_price=entry_price)
                fib_data = check_fib_confluence(entry_price,
                                                 best['direction'], fib_report)
            except Exception as e:
                log.debug(f"[{symbol}] Fib computation skipped: {e}")
                fib_data = {}
        if all_scores is None:
            all_scores = {}
        all_scores['_fib_data'] = fib_data

        trade_reports[tracker.ticket_counter] = {
            'master_report': master_report,
            'market_report': market_report,
            'smc_report': smc_report,
            'flow': flow,
            'strategy_scores': all_scores,
            'predicted_r': best.get('model_predicted_r'),
            'strategy_model_verdict': strat_model_verdict,
            'strategy_model_predicted_r': strat_model_predicted_r,
            'signal': best,  # Full signal dict with embedded strategy features
        }

        # ── Store signal metadata for ML training (no DB write) ────────

        if trades_executed % 5 == 0:
            log.info(f"  [{symbol}] Trade #{trades_executed} | "
                     f"{best['direction']} {best.get('strategy','')} "
                     f"score={best.get('score',0)} "
                     f"groups={final_groups} "
                     f"conviction={tracker.determine_conviction(best.get('score',0), agreement_groups)} "
                     f"| Balance: ${tracker.balance:.2f}")

    # ── Check remaining open trades at end ───────────────
    if tracker.open_trades:
        last_bar = df_m1.iloc[-1]
        tracker.close_remaining_at_end(
            last_bar['time'], float(last_bar['close']), pip_size, pip_value)

    # ── Close remaining shadow trades at end ────────────
    if shadow_tracker and shadow_tracker.open_trades:
        last_bar = df_m1.iloc[-1]
        shadow_tracker.close_remaining_at_end(
            last_bar['time'], float(last_bar['close']), pip_size, pip_value)

    # ── Close remaining L1 strategy model shadow trades at end ──
    if strat_model_shadow_tracker and strat_model_shadow_tracker.open_trades:
        last_bar = df_m1.iloc[-1]
        strat_model_shadow_tracker.close_remaining_at_end(
            last_bar['time'], float(last_bar['close']), pip_size, pip_value)

    # ── Close remaining all-signals shadow trades at end ──
    if all_signals_shadow_tracker and all_signals_shadow_tracker.open_trades:
        last_bar = df_m1.iloc[-1]
        all_signals_shadow_tracker.close_remaining_at_end(
            last_bar['time'], float(last_bar['close']), pip_size, pip_value)

    # ── Log shadow trade results ─────────────────────────
    if shadow_tracker and shadow_tracker.closed_trades:
        _log_shadow_results(shadow_tracker.closed_trades,
                            f"{symbol} L2_SHADOW")
    if strat_model_shadow_tracker and strat_model_shadow_tracker.closed_trades:
        _log_shadow_results(strat_model_shadow_tracker.closed_trades,
                            f"{symbol} L1_SHADOW")
    if all_signals_shadow_tracker and all_signals_shadow_tracker.closed_trades:
        _log_shadow_results(all_signals_shadow_tracker.closed_trades,
                            f"{symbol} ALL_SIGNALS_SHADOW")

    elapsed = time_mod.time() - start_time
    summary = tracker.get_summary()
    summary['symbol'] = symbol
    summary['elapsed_seconds'] = round(elapsed, 1)
    summary['signals_found'] = signals_found
    summary['signals_blocked_consensus'] = signals_blocked_consensus
    summary['signals_blocked_gate'] = signals_blocked_gate
    summary['signals_blocked_score'] = signals_blocked_score
    summary['signals_blocked_choppy'] = signals_blocked_choppy
    summary['signals_blocked_bias'] = signals_blocked_bias
    summary['signals_blocked_confluence'] = signals_blocked_confluence
    summary['signals_no_strategy'] = signals_no_strategy
    summary['trades_executed'] = trades_executed
    summary['final_score_avg'] = final_score if trades_executed > 0 else 0
    summary['relaxed_mode'] = config.relaxed_mode
    summary['run_id'] = config.run_id
    summary['model_blocked'] = model_blocked_count
    summary['model_caution'] = model_caution_count
    summary['shadow_trades'] = shadow_count
    summary['strat_model_rejected'] = strat_model_reject_count
    summary['strat_model_shadow_trades'] = strat_model_shadow_count
    summary['all_signals_shadow_trades'] = all_signals_shadow_count

    # ── ML Gate prediction distribution log ─────────────
    if ml_gate_active and ml_predictions:
        takes = [p for p in ml_predictions if p['recommendation'] == 'TAKE']
        cautions = [p for p in ml_predictions if p['recommendation'] == 'CAUTION']
        skips = [p for p in ml_predictions if p['recommendation'] == 'SKIP']
        neutrals = [p for p in ml_predictions if p['recommendation'] == 'NEUTRAL']
        all_r = [p['predicted_r'] for p in ml_predictions]
        if all_r:
            log.info(f"  [ML_GATE] Prediction distribution: "
                     f"TAKE={len(takes)} CAUTION={len(cautions)} SKIP={len(skips)} NEUTRAL={len(neutrals)} | "
                     f"R: min={min(all_r):.2f} avg={sum(all_r)/len(all_r):.2f} max={max(all_r):.2f}")
        if cautions:
            caution_r = [p['predicted_r'] for p in cautions]
            log.info(f"  [ML_GATE] CAUTION range: R={min(caution_r):.2f} to {max(caution_r):.2f}")
        if skips:
            skip_r = [p['predicted_r'] for p in skips]
            log.info(f"  [ML_GATE] SKIP range: R={min(skip_r):.2f} to {max(skip_r):.2f}")

    # ── Store all completed trades to DB ─
    if config.store_db:
        try:
            from backtest.db_store import store_trade
            spread = AVG_SPREAD_PIPS.get(symbol, AVG_SPREAD_PIPS['DEFAULT'])
            stored = 0
            for trade in tracker.closed_trades:
                # Use per-trade feature snapshot (captured at entry time)
                reports = trade_reports.get(trade.ticket, {})
                strat_feat = _extract_strategy_features(reports.get('signal'))
                store_trade(
                    trade=trade,
                    master_report=reports.get('master_report'),
                    market_report=reports.get('market_report'),
                    smc_report=reports.get('smc_report'),
                    flow_data=reports.get('flow'),
                    run_id=config.run_id,
                    spread_pips=spread,
                    slippage_pips=SLIPPAGE_PIPS,
                    strategy_scores=reports.get('strategy_scores'),
                    model_predicted_r=reports.get('predicted_r'),
                    **strat_feat,
                )
                stored += 1
            log.info(f"  [DB] Stored {stored} trades in MySQL")

            # ── Store shadow trades to DB ──
            if shadow_tracker and shadow_tracker.closed_trades:
                shadow_stored = 0
                for trade in shadow_tracker.closed_trades:
                    reports = shadow_reports.get(trade.ticket, {})
                    strat_feat = _extract_strategy_features(reports.get('signal'))
                    store_trade(
                        trade=trade,
                        master_report=reports.get('master_report'),
                        market_report=reports.get('market_report'),
                        smc_report=reports.get('smc_report'),
                        flow_data=reports.get('flow'),
                        run_id=config.run_id,
                        spread_pips=spread,
                        slippage_pips=SLIPPAGE_PIPS,
                        strategy_scores=reports.get('strategy_scores'),
                        source='SHADOW',
                        model_predicted_r=reports.get('predicted_r'),
                        **strat_feat,
                    )
                    shadow_stored += 1
                if shadow_stored > 0:
                    log.info(f"  [DB] Stored {shadow_stored} shadow trades in MySQL")

            # ── Store L1 strategy model shadow trades to DB ──
            if strat_model_shadow_tracker and strat_model_shadow_tracker.closed_trades:
                l1_shadow_stored = 0
                for trade in strat_model_shadow_tracker.closed_trades:
                    reports = strat_model_shadow_reports.get(trade.ticket, {})
                    strat_feat = _extract_strategy_features(reports.get('signal'))
                    store_trade(
                        trade=trade,
                        master_report=reports.get('master_report'),
                        market_report=reports.get('market_report'),
                        smc_report=reports.get('smc_report'),
                        flow_data=reports.get('flow'),
                        run_id=config.run_id,
                        spread_pips=spread,
                        slippage_pips=SLIPPAGE_PIPS,
                        strategy_scores=reports.get('strategy_scores'),
                        source='SHADOW',
                        model_predicted_r=reports.get('predicted_r'),
                        strategy_model_verdict=reports.get('strategy_model_verdict'),
                        strategy_model_predicted_r=reports.get('strategy_model_predicted_r'),
                        **strat_feat,
                    )
                    l1_shadow_stored += 1
                if l1_shadow_stored > 0:
                    log.info(f"  [DB] Stored {l1_shadow_stored} L1 strategy model shadow trades in MySQL")

            # ── Store all-signals shadow trades to DB ──
            if all_signals_shadow_tracker and all_signals_shadow_tracker.closed_trades:
                all_sig_stored = 0
                for trade in all_signals_shadow_tracker.closed_trades:
                    reports = all_signals_shadow_reports.get(trade.ticket, {})
                    strat_feat = _extract_strategy_features(reports.get('signal'))
                    store_trade(
                        trade=trade,
                        master_report=reports.get('master_report'),
                        market_report=reports.get('market_report'),
                        smc_report=reports.get('smc_report'),
                        flow_data=reports.get('flow'),
                        run_id=config.run_id,
                        spread_pips=spread,
                        slippage_pips=SLIPPAGE_PIPS,
                        strategy_scores=reports.get('strategy_scores'),
                        source='SHADOW',
                        model_predicted_r=reports.get('predicted_r'),
                        **strat_feat,
                    )
                    all_sig_stored += 1
                if all_sig_stored > 0:
                    log.info(f"  [DB] Stored {all_sig_stored} all-signals shadow trades in MySQL")
        except Exception as e:
            log.warning(f"  [DB] Could not store trades: {e}")

    return summary


# =============================================================
# PARALLEL MULTI-SYMBOL BACKTEST — processes all pairs on the
# same M1 timeline, exactly like live trading where all pairs
# are scanned simultaneously every M15 bar.
# =============================================================

def run_parallel_backtest(symbols: list, start_date, end_date,
                          scan_every: int = 15, relaxed_mode: bool = False,
                          store_db: bool = False, run_id: str = 'default',
                          max_trades_per_symbol: int = 9999,
                          use_model: bool = False, unlimited_positions: bool = False) -> list:
    """
    Run all symbols in parallel on the same M1 timeline.
    Each symbol gets its own TradeTracker, strategies scan independently,
    but all pairs share the same clock — like live trading.
    
    Returns list of per-symbol summary dicts.
    """
    import time as time_mod
    from backtest.data_loader import load_all_data, get_candles_at_time
    from backtest.tick_simulator import simulate_order_flow_from_full
    from backtest.smc_builder import build_smc_report
    from backtest.market_builder import build_market_report
    from backtest.trade_tracker import TradeTracker
    from strategies.strategy_engine import _run_one_strategy, _get_strategy_group
    from strategies.strategy_registry import get_active_strategies
    from data_layer.feature_store import store

    start_time = time_mod.time()

    active_strategies = get_active_strategies()
    log.info(f"")
    log.info(f"{'='*65}")
    log.info(f"  PARALLEL BACKTEST v3.0 — {len(symbols)} symbols")
    log.info(f"  Period: {start_date.date()} to {end_date.date()}")
    log.info(f"  Strategies: {', '.join(active_strategies)}")
    log.info(f"  Mode: {'RELAXED' if relaxed_mode else 'STRICT'}")
    log.info(f"{'='*65}")

    # ── Load data for all symbols ──────────────────────────
    symbol_data = {}   # symbol -> {M1, M5, M15, H1, H4}
    symbol_trackers = {}  # symbol -> TradeTracker
    symbol_reports = {}   # symbol -> {ticket -> snapshot}
    symbol_pip = {}       # symbol -> pip_size
    symbol_pipval = {}    # symbol -> pip_value
    symbol_spread = {}    # symbol -> total_slippage in pips
    symbol_stats = {}     # symbol -> {signals_found, blocked_gate, blocked_consensus, blocked_score, executed, model_blocked}
    symbol_shadow_trackers = {}  # symbol -> TradeTracker for shadow trades
    symbol_shadow_reports = {}   # symbol -> {ticket -> snapshot}
    symbol_shadow_count = {}     # symbol -> int
    all_sig_shadow_trackers = {}  # symbol -> TradeTracker for all-signals shadows
    all_sig_shadow_reports = {}   # symbol -> {ticket -> snapshot}
    all_sig_shadow_count = {}     # symbol -> int

    # ── ML Gate prediction tracking (for distribution analysis) ──
    ml_predictions = []  # Track all predictions across all symbols

    # ── Load ML Gate model if --use-model ────────────────
    ml_gate_active = False
    if use_model:
        try:
            from ai_engine.ml_gate import is_model_trained
            if is_model_trained():
                ml_gate_active = True
                log.info(f"  [ML_GATE] Strategy-Informed ML v3.0 loaded")
            else:
                log.warning(f"  [ML_GATE] --use-model but no model found")
        except Exception as e:
            log.warning(f"  [ML_GATE] Failed to load: {e}")

    for sym in symbols:
        data = load_all_data(sym, start_date, end_date, CACHE_DIR)
        if not data:
            log.warning(f"  No data for {sym}, skipping")
            continue

        symbol_data[sym] = data
        pip_size = get_pip_size(sym)
        pip_value = PIP_VALUE_PER_LOT.get(sym, PIP_VALUE_PER_LOT['DEFAULT'])
        spread = AVG_SPREAD_PIPS.get(sym, AVG_SPREAD_PIPS['DEFAULT'])

        symbol_pip[sym] = pip_size
        symbol_pipval[sym] = pip_value
        symbol_spread[sym] = spread + SLIPPAGE_PIPS

        max_open = 9999 if unlimited_positions else MAX_OPEN_TRADES
        max_per_sym = 9999 if unlimited_positions else MAX_PER_SYMBOL
        symbol_trackers[sym] = TradeTracker(
            starting_balance=STARTING_BALANCE,
            pip_value_per_lot=pip_value,
            max_open=max_open,
            max_per_symbol=max_per_sym,
            partial_tp_enabled=False if relaxed_mode else PARTIAL_TP_ENABLED,
            atr_trail_enabled=False if relaxed_mode else ATR_TRAIL_ENABLED,
            dynamic_tp_enabled=False if relaxed_mode else DYNAMIC_TP_EXTENSION_ENABLED,
            dynamic_sizing_enabled=DYNAMIC_SIZING_ENABLED,
            base_risk_percent=BASE_RISK_PERCENT,
        )
        symbol_reports[sym] = {}
        symbol_stats[sym] = {
            'signals_found': 0, 'blocked_gate': 0,
            'blocked_consensus': 0, 'blocked_score': 0,
            'blocked_choppy': 0, 'blocked_bias': 0,
            'blocked_confluence': 0, 'no_strategy': 0,
            'executed': 0, 'model_blocked': 0, 'shadow_trades': 0,
            'model_caution': 0, 'strat_model_rejected': 0,
        }
        # Shadow tracker for this symbol (only if store_db + model)
        # v3.3: Uses FULL TradeTracker (same features as real trades)
        if store_db and ml_gate_active:
            symbol_shadow_trackers[sym] = TradeTracker(
                starting_balance=STARTING_BALANCE,
                pip_value_per_lot=pip_value,
                max_open=9999, max_per_symbol=9999,
                partial_tp_enabled=PARTIAL_TP_ENABLED,
                atr_trail_enabled=ATR_TRAIL_ENABLED,
                dynamic_tp_enabled=DYNAMIC_TP_EXTENSION_ENABLED,
                dynamic_sizing_enabled=DYNAMIC_SIZING_ENABLED,
                base_risk_percent=BASE_RISK_PERCENT,
            )
            symbol_shadow_reports[sym] = {}
            symbol_shadow_count[sym] = 0

        # All-signals shadow tracker for this symbol
        if store_db:
            all_sig_shadow_trackers[sym] = TradeTracker(
                starting_balance=STARTING_BALANCE,
                pip_value_per_lot=pip_value,
                max_open=9999, max_per_symbol=9999,
                partial_tp_enabled=PARTIAL_TP_ENABLED,
                atr_trail_enabled=ATR_TRAIL_ENABLED,
                dynamic_tp_enabled=DYNAMIC_TP_EXTENSION_ENABLED,
                dynamic_sizing_enabled=DYNAMIC_SIZING_ENABLED,
                base_risk_percent=BASE_RISK_PERCENT,
            )
            all_sig_shadow_reports[sym] = {}
            all_sig_shadow_count[sym] = 0

        log.info(f"  Loaded {sym}: M1={len(data['M1'])} bars, H1={len(data['H1'])} bars")

    if not symbol_data:
        return []

    # ── Find the shortest M1 series (determines loop length) ─
    min_len = min(len(d['M1']) for d in symbol_data.values())
    log.info(f"  Timeline: {min_len} M1 bars (~{min_len // 1440} days)")

    # ── Main parallel loop ──────────────────────────────────
    for bar_idx in range(scan_every, min_len):
        # Get the current time from the first symbol (all share same UTC clock)
        ref_sym = list(symbol_data.keys())[0]
        current_time = symbol_data[ref_sym]['M1'].iloc[bar_idx]['time']

        # ── Check exits on EVERY bar for ALL symbols ───────
        for sym in list(symbol_data.keys()):
            data = symbol_data[sym]
            if bar_idx >= len(data['M1']):
                continue
            bar = data['M1'].iloc[bar_idx]
            symbol_trackers[sym].check_exits(
                current_time,
                float(bar['high']),
                float(bar['low']),
                float(bar['close']),
                symbol_pip[sym],
                symbol_pipval[sym],
            )
            # Also check shadow trade exits
            if sym in symbol_shadow_trackers:
                symbol_shadow_trackers[sym].check_exits(
                    current_time,
                    float(bar['high']),
                    float(bar['low']),
                    float(bar['close']),
                    symbol_pip[sym],
                    symbol_pipval[sym],
                )
            # Also check all-signals shadow exits (CRITICAL: was missing)
            if sym in all_sig_shadow_trackers:
                all_sig_shadow_trackers[sym].check_exits(
                    current_time,
                    float(bar['high']),
                    float(bar['low']),
                    float(bar['close']),
                    symbol_pip[sym],
                    symbol_pipval[sym],
                )

        # ── Only run strategy scan every N bars ────────────
        if bar_idx % scan_every != 0:
            continue

        # ── Scan ALL symbols at this timestamp ─────────────
        for sym in list(symbol_data.keys()):
            stats = symbol_stats[sym]
            tracker = symbol_trackers[sym]
            data = symbol_data[sym]

            # Skip if max trades reached
            if max_trades_per_symbol > 0 and stats['executed'] >= max_trades_per_symbol:
                continue

            # Skip if already have open trade for this symbol
            if not tracker.can_open(sym):
                continue

            # Get sliced data up to current time
            if bar_idx >= len(data['M1']):
                continue

            sliced = get_candles_at_time(data, current_time, count=200)
            s_m1, s_m5, s_m15, s_h1, s_h4 = (
                sliced['M1'], sliced['M5'], sliced['M15'],
                sliced['H1'], sliced['H4']
            )

            if len(s_m1) < 5 or len(s_m15) < 30 or len(s_h1) < 30 or len(s_h4) < 10:
                continue

            current_bar = s_m1.iloc[-1]

            # Build reports
            flow = simulate_order_flow_from_full(s_m1, symbol_pip[sym])
            smc_report = build_smc_report(s_h1, s_h4, current_time)
            market_report = build_market_report(s_m15, flow, smc_report, sym)
            master_report = _build_master_report(sym, market_report, smc_report, flow)
            master_report['timestamp'] = current_time.strftime('%Y-%m-%d %H:%M:%S')

            # Update feature store
            market_data_for_store = {
                'current_price': float(current_bar['close']),
                'atr': float(s_m15.iloc[-1].get('atr', 0)) if len(s_m15) > 0 else 0,
                'delta': flow.get('delta', {}),
                'rolling_delta': flow.get('rolling_delta', {}),
                'order_flow_imbalance': flow.get('order_flow_imbalance', {}),
                'volume_surge': flow.get('volume_surge', {}),
                'momentum': flow.get('momentum', {}),
                'vwap': market_report.get('vwap', {}),
                'profile': market_report.get('profile', {}),
            }
            store.update_symbol_features(sym, market_data_for_store, smc_report)

            # Gate 0: Score
            final_score = master_report.get('final_score', 0)
            score_gate = RELAXED_MIN_SCORE if relaxed_mode else MASTER_MIN_SCORE
            if final_score < score_gate:
                stats['blocked_score'] += 1
                continue

            # Gate 1: Institutional (SOFT — no longer blocks)
            imb_strength = flow.get('order_flow_imbalance', {}).get('strength', 'NONE')
            surge_active = flow.get('volume_surge', {}).get('surge_detected', False)
            has_institutional = imb_strength in ('STRONG', 'EXTREME') or surge_active

            if not has_institutional:
                stats['blocked_gate'] += 1
                # SOFT: no continue — signal passes to ML Gate

            # Gate 2: Choppy
            is_choppy = flow.get('momentum', {}).get('is_choppy', True)
            if is_choppy and not surge_active:
                stats['blocked_choppy'] += 1
                continue

            # Run strategies
            market_state = master_report.get('market_state', 'BALANCED')
            session = master_report.get('session', 'UNKNOWN')
            combined_bias = master_report.get('combined_bias', 'NEUTRAL')

            signals = []
            for strategy_name in active_strategies:
                try:
                    signal = _run_one_strategy(
                        strategy_name, sym,
                        s_m1, s_m5, s_m15, s_h1, s_h4,
                        smc_report, market_report,
                        market_state, session, master_report,
                        relaxed=relaxed_mode)
                    if signal is None:
                        continue
                    direction = str(signal.get('direction', ''))
                    score = signal.get('score', 0)
                    from strategies.strategy_engine import STRATEGY_MIN_SCORES
                    min_score = STRATEGY_MIN_SCORES.get(strategy_name, 70)
                    if score < min_score:
                        continue
                    signal['symbol'] = sym
                    signal['group'] = _get_strategy_group(strategy_name)
                    signals.append(signal)
                except Exception:
                    continue

            if not signals:
                continue

            stats['signals_found'] += 1

            # Gate 3: Bias filter (skip in relaxed)
            if not relaxed_mode:
                if combined_bias == 'BULLISH':
                    signals = [s for s in signals if s['direction'] == 'BUY']
                elif combined_bias == 'BEARISH':
                    signals = [s for s in signals if s['direction'] == 'SELL']
                if not signals:
                    stats['blocked_bias'] += 1
                    continue

            # Gate 4: Consensus
            buy_groups = set(s['group'] for s in signals if s['direction'] == 'BUY')
            sell_groups = set(s['group'] for s in signals if s['direction'] == 'SELL')
            min_groups = RELAXED_CONSENSUS_GROUPS if relaxed_mode else 2

            if len(buy_groups) >= min_groups and len(buy_groups) >= len(sell_groups):
                final_signals = [s for s in signals if s['direction'] == 'BUY']
                final_groups = buy_groups
            elif len(sell_groups) >= min_groups:
                final_signals = [s for s in signals if s['direction'] == 'SELL']
                final_groups = sell_groups
            else:
                stats['blocked_consensus'] += 1
                continue

            # Best signal
            best = max(final_signals, key=lambda s: s['score'])

            # Gate 5: Confluence
            confluence = best.get('confluence', [])
            min_conv = RELAXED_MIN_CONFLUENCE if relaxed_mode else MIN_CONFLUENCE
            if len(confluence) < min_conv:
                stats['blocked_confluence'] += 1
                continue

            # Gate 6: ML Gate filter (if --use-model)
            if ml_gate_active:
                try:
                    from ai_engine.ml_gate import score_signal, collect_all_strategy_scores
                    all_scores = collect_all_strategy_scores(
                        sym, s_m1, s_m5, s_m15, s_h1, s_h4,
                        smc_report, market_report,
                        market_state, session, master_report)

                    # ── Shadow ALL non-best qualifying signals ──
                    if sym in all_sig_shadow_trackers:
                        for strat_name, score in all_scores.items():
                            if score <= 0 or strat_name == best.get('strategy', ''):
                                continue
                            try:
                                sig = _run_one_strategy(
                                    strat_name, sym,
                                    s_m1, s_m5, s_m15, s_h1, s_h4,
                                    smc_report, market_report,
                                    market_state, session, master_report,
                                    relaxed=relaxed_mode)
                                if sig is None:
                                    continue
                                if _open_signal_shadow(
                                        all_sig_shadow_trackers[sym], sig,
                                        sym, current_time,
                                        current_bar['close'], symbol_pip[sym],
                                        symbol_spread[sym], market_report,
                                        market_state):
                                    all_sig_shadow_count[sym] += 1
                                    all_sig_shadow_reports[sym][
                                        all_sig_shadow_trackers[sym].ticket_counter] = {
                                        'master_report': master_report,
                                        'market_report': market_report,
                                        'smc_report': smc_report,
                                        'flow': flow,
                                        'strategy_scores': all_scores,
                                        'predicted_r': None,
                                        'signal': sig,
                                    }
                            except Exception:
                                pass

                    ml_result = score_signal(
                        best, master_report, market_report,
                        smc_report, flow,
                        all_strategy_scores=all_scores,
                        symbol=sym, spread_pips=symbol_spread.get(sym, 0))
                    predicted_r = ml_result.get('predicted_r', 0.0)
                    rec = ml_result.get('recommendation', 'SKIP')
                    best['model_predicted_r'] = predicted_r

                    # Track prediction for distribution analysis
                    ml_predictions.append({
                        'symbol': sym,
                        'strategy': best.get('strategy', 'UNKNOWN'),
                        'direction': best['direction'],
                        'predicted_r': predicted_r,
                        'recommendation': rec,
                    })

                    # ── NEUTRAL = features failed → treat as SKIP (shadow) ──
                    if rec == 'NEUTRAL':
                        rec = 'SKIP'

                    if rec == 'SKIP':
                        stats['model_blocked'] += 1
                    elif rec == 'CAUTION':
                        stats['model_caution'] += 1

                    # v3.3: ALL non-TAKE signals → SHADOW (CAUTION + SKIP)
                    if rec in ('CAUTION', 'SKIP'):
                        if sym in symbol_shadow_trackers:
                            sl_p = best.get('sl_pips', 0)
                            tp_p = best.get('tp1_pips', 0) or best.get('tp_pips', 0)
                            if sl_p > 0 and tp_p > 0:
                                sh_entry = float(current_bar['close'])
                                total_slip = symbol_spread[sym]
                                if best['direction'] == 'BUY':
                                    sh_entry += total_slip * symbol_pip[sym]
                                else:
                                    sh_entry -= total_slip * symbol_pip[sym]
                                symbol_shadow_trackers[sym].open_trade(
                                    symbol=sym,
                                    direction=best['direction'],
                                    strategy=best.get('strategy', 'UNKNOWN'),
                                    entry_time=current_time,
                                    entry_price=sh_entry,
                                    sl_price=0, tp_price=0,
                                    sl_pips=sl_p, tp_pips=tp_p,
                                    score=best.get('score', 0),
                                    confluence=best.get('confluence', []),
                                    session=market_report.get('session', 'UNKNOWN'),
                                    market_state=market_state,
                                    agreement_groups=1,
                                    atr_value=0.0,
                                )
                                st = symbol_shadow_trackers[sym].open_trades[-1]
                                if best['direction'] == 'BUY':
                                    st.sl_price = st.entry_price - st.sl_pips * symbol_pip[sym]
                                    st.tp_price = st.entry_price + st.tp_pips * symbol_pip[sym]
                                else:
                                    st.sl_price = st.entry_price + st.sl_pips * symbol_pip[sym]
                                    st.tp_price = st.entry_price - st.tp_pips * symbol_pip[sym]
                                symbol_shadow_count[sym] += 1
                                stats['shadow_trades'] += 1
                                symbol_shadow_reports[sym][symbol_shadow_trackers[sym].ticket_counter] = {
                                    'master_report': master_report,
                                    'market_report': market_report,
                                    'smc_report': smc_report,
                                    'flow': flow,
                                    'strategy_scores': all_scores or {},
                                    'predicted_r': predicted_r,
                                    'signal': best,  # Full signal with strategy features
                                }
                        continue  # Don't execute as real trade
                except Exception:
                    pass

            # Execute trade
            entry_price = float(current_bar['close'])
            total_slip = symbol_spread[sym]
            if best['direction'] == 'BUY':
                entry_price += total_slip * symbol_pip[sym]
            else:
                entry_price -= total_slip * symbol_pip[sym]

            sl_pips = best.get('sl_pips', 0)
            tp_pips = best.get('tp1_pips', 0) or best.get('tp_pips', 0)

            if sl_pips <= 0 or tp_pips <= 0:
                continue

            min_rr = RELAXED_MIN_RR_RATIO if relaxed_mode else MIN_RR_RATIO
            if tp_pips / sl_pips < min_rr:
                continue

            if best['direction'] == 'BUY':
                sl_price = entry_price - sl_pips * symbol_pip[sym]
                tp_price = entry_price + tp_pips * symbol_pip[sym]
            else:
                sl_price = entry_price + sl_pips * symbol_pip[sym]
                tp_price = entry_price - tp_pips * symbol_pip[sym]

            atr_value = float(s_m15.iloc[-1].get('atr', 0)) if len(s_m15) > 0 else 0.0
            agreement_groups = len(final_groups)

            tracker.open_trade(
                symbol=sym,
                direction=best['direction'],
                strategy=best.get('strategy', 'UNKNOWN'),
                entry_time=current_time,
                entry_price=entry_price,
                sl_price=sl_price,
                tp_price=tp_price,
                sl_pips=sl_pips,
                tp_pips=tp_pips,
                score=best.get('score', 0),
                confluence=confluence,
                session=market_report.get('session', 'UNKNOWN'),
                market_state=market_state,
                agreement_groups=agreement_groups,
                atr_value=atr_value,
            )
            stats['executed'] += 1

            # Save snapshot + compute Fibonacci confluence for DB
            fib_data_snap = {}
            if store_db:
                try:
                    from backtest.fib_builder import build_fib_report, check_fib_confluence
                    fib_report = build_fib_report(df_h1=s_h1, df_h4=s_h4,
                                                 current_price=entry_price)
                    fib_data_snap = check_fib_confluence(entry_price,
                                                      best['direction'], fib_report)
                except Exception as e:
                    log.debug(f"[{sym}] Fib computation skipped: {e}")
                    fib_data_snap = {}
            snap_scores = (all_scores if ml_gate_active else None) or {}
            snap_scores['_fib_data'] = fib_data_snap

            symbol_reports[sym][tracker.ticket_counter] = {
                'master_report': master_report,
                'market_report': market_report,
                'smc_report': smc_report,
                'flow': flow,
                # Always pass snap_scores when store_db; the truthy check on
                # _fib_data dict value was causing fib data to be dropped
                # when fib_data_snap was {} (empty dict is falsy in Python).
                'strategy_scores': snap_scores if store_db else (all_scores if ml_gate_active else None),
                'predicted_r': best.get('model_predicted_r'),
                'signal': best,  # Full signal dict with embedded strategy features
            }

            # Trade metadata saved (no DB write for signals)

            log.info(f"  [{sym}] Trade #{stats['executed']} | "
                     f"{best['direction']} {best.get('strategy','')} "
                     f"score={best.get('score',0)} | "
                     f"Balance: ${tracker.balance:.2f}")

    # ── Close remaining open trades ────────────────────────
    for sym in list(symbol_data.keys()):
        tracker = symbol_trackers[sym]
        if tracker.open_trades:
            data = symbol_data[sym]
            last_bar = data['M1'].iloc[-1]
            tracker.close_remaining_at_end(
                last_bar['time'], float(last_bar['close']),
                symbol_pip[sym], symbol_pipval[sym])
        # Close remaining shadow trades
        if sym in symbol_shadow_trackers and symbol_shadow_trackers[sym].open_trades:
            data = symbol_data[sym]
            last_bar = data['M1'].iloc[-1]
            symbol_shadow_trackers[sym].close_remaining_at_end(
                last_bar['time'], float(last_bar['close']),
                symbol_pip[sym], symbol_pipval[sym])
        # Close remaining all-signals shadow trades
        if sym in all_sig_shadow_trackers and all_sig_shadow_trackers[sym].open_trades:
            data = symbol_data[sym]
            last_bar = data['M1'].iloc[-1]
            all_sig_shadow_trackers[sym].close_remaining_at_end(
                last_bar['time'], float(last_bar['close']),
                symbol_pip[sym], symbol_pipval[sym])

    # ── Build per-symbol summaries + store to DB ────────────
    all_results = []
    elapsed = time_mod.time() - start_time

    for sym in list(symbol_data.keys()):
        tracker = symbol_trackers[sym]
        stats = symbol_stats[sym]

        summary = tracker.get_summary()
        summary['symbol'] = sym
        summary['elapsed_seconds'] = round(elapsed, 1)
        summary['signals_found'] = stats['signals_found']
        summary['signals_blocked_consensus'] = stats['blocked_consensus']
        summary['signals_blocked_gate'] = stats['blocked_gate']
        summary['signals_blocked_score'] = stats['blocked_score']
        summary['signals_blocked_choppy'] = stats.get('blocked_choppy', 0)
        summary['signals_blocked_bias'] = stats.get('blocked_bias', 0)
        summary['signals_blocked_confluence'] = stats.get('blocked_confluence', 0)
        summary['signals_no_strategy'] = stats.get('no_strategy', 0)
        summary['trades_executed'] = stats['executed']
        summary['final_score_avg'] = 0
        summary['relaxed_mode'] = relaxed_mode
        summary['run_id'] = run_id
        summary['model_blocked'] = stats.get('model_blocked', 0)
        summary['model_caution'] = stats.get('model_caution', 0)
        summary['shadow_trades'] = stats.get('shadow_trades', 0)
        summary['strat_model_rejected'] = stats.get('strat_model_rejected', 0)

        # Store trades to DB
        if store_db:
            try:
                from backtest.db_store import store_trade
                spread = AVG_SPREAD_PIPS.get(sym, AVG_SPREAD_PIPS['DEFAULT'])
                stored = 0
                for trade in tracker.closed_trades:
                    reports = symbol_reports.get(sym, {}).get(trade.ticket, {})
                    strat_feat = _extract_strategy_features(reports.get('signal'))
                    store_trade(
                        trade=trade,
                        master_report=reports.get('master_report'),
                        market_report=reports.get('market_report'),
                        smc_report=reports.get('smc_report'),
                        flow_data=reports.get('flow'),
                        run_id=run_id,
                        spread_pips=spread,
                        slippage_pips=SLIPPAGE_PIPS,
                        strategy_scores=reports.get('strategy_scores'),
                        model_predicted_r=reports.get('predicted_r'),
                        **strat_feat,
                    )
                    stored += 1
                log.info(f"  [DB] {sym}: Stored {stored} trades in MySQL")

                # Store shadow trades
                if sym in symbol_shadow_trackers and symbol_shadow_trackers[sym].closed_trades:
                    shadow_stored = 0
                    for trade in symbol_shadow_trackers[sym].closed_trades:
                        reports = symbol_shadow_reports.get(sym, {}).get(trade.ticket, {})
                        strat_feat = _extract_strategy_features(reports.get('signal'))
                        store_trade(
                            trade=trade,
                            master_report=reports.get('master_report'),
                            market_report=reports.get('market_report'),
                            smc_report=reports.get('smc_report'),
                            flow_data=reports.get('flow'),
                            run_id=run_id,
                            spread_pips=spread,
                            slippage_pips=SLIPPAGE_PIPS,
                            strategy_scores=reports.get('strategy_scores'),
                            source='SHADOW',
                            model_predicted_r=reports.get('predicted_r'),
                            **strat_feat,
                        )
                        shadow_stored += 1
                    if shadow_stored > 0:
                        log.info(f"  [DB] {sym}: Stored {shadow_stored} shadow trades in MySQL")

                # Store all-signals shadow trades
                if sym in all_sig_shadow_trackers and all_sig_shadow_trackers[sym].closed_trades:
                    all_sig_stored = 0
                    for trade in all_sig_shadow_trackers[sym].closed_trades:
                        reports = all_sig_shadow_reports.get(sym, {}).get(trade.ticket, {})
                        strat_feat = _extract_strategy_features(reports.get('signal'))
                        store_trade(
                            trade=trade,
                            master_report=reports.get('master_report'),
                            market_report=reports.get('market_report'),
                            smc_report=reports.get('smc_report'),
                            flow_data=reports.get('flow'),
                            run_id=run_id,
                            spread_pips=spread,
                            slippage_pips=SLIPPAGE_PIPS,
                            strategy_scores=reports.get('strategy_scores'),
                            source='SHADOW',
                            model_predicted_r=reports.get('predicted_r'),
                            **strat_feat,
                        )
                        all_sig_stored += 1
                    if all_sig_stored > 0:
                        log.info(f"  [DB] {sym}: Stored {all_sig_stored} all-signals shadow trades in MySQL")
            except Exception as e:
                log.warning(f"  [DB] {sym}: Could not store trades: {e}")

        # Log shadow trade results for this symbol
        if sym in symbol_shadow_trackers and symbol_shadow_trackers[sym].closed_trades:
            _log_shadow_results(symbol_shadow_trackers[sym].closed_trades,
                                f"{sym} L2_SHADOW")
        if sym in all_sig_shadow_trackers and all_sig_shadow_trackers[sym].closed_trades:
            _log_shadow_results(all_sig_shadow_trackers[sym].closed_trades,
                                f"{sym} ALL_SIGNALS_SHADOW")

        all_results.append(summary)

    log.info(f"\n  Parallel backtest completed in {elapsed:.1f}s")

    # ── ML Gate prediction distribution (across all symbols) ──
    if ml_gate_active and ml_predictions:
        takes = [p for p in ml_predictions if p['recommendation'] == 'TAKE']
        cautions = [p for p in ml_predictions if p['recommendation'] == 'CAUTION']
        skips = [p for p in ml_predictions if p['recommendation'] == 'SKIP']
        neutrals = [p for p in ml_predictions if p['recommendation'] == 'NEUTRAL']
        all_r = [p['predicted_r'] for p in ml_predictions]
        if all_r:
            log.info(f"  [ML_GATE] Overall prediction distribution: "
                     f"TAKE={len(takes)} CAUTION={len(cautions)} SKIP={len(skips)} NEUTRAL={len(neutrals)} | "
                     f"R: min={min(all_r):.2f} avg={sum(all_r)/len(all_r):.2f} max={max(all_r):.2f}")

    return all_results
