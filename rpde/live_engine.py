# =============================================================
# rpde/live_engine.py  —  Live Trading Orchestrator (RPDE)
#
# PURPOSE: Wires the full RPDE decision chain for live trading:
#
#   M5 tick
#     -> Feature extraction (93 features)
#     -> Pattern Gate (XGB + Pattern Library + TFT fusion)
#     -> RL Agent (PPO policy)
#     -> Safety Guards (non-overridable human rails)
#     -> MT5 Execution Bridge (place_order)
#     -> Monitor (manage_positions, trailing, BE, partial TP)
#     -> Record outcome (ExperienceBuffer)
#
# CONTAINS:
#   1. LiveEngine class — main orchestrator
#   2. MT5 Execution Bridge — converts RL output to broker orders
#   3. RL Mid-Trade Actions — delegates to order_manager + RL early exit
#   4. Real-time P&L -> RL feedback — reads MT5 positions, feeds state
#
# DESIGN DECISIONS:
#   - Graceful degradation: if RL not trained, falls back to Gate-only
#   - Paper mode: default True — logs trades but doesn't execute
#   - Thread safety: locks for shared state mutations
#   - Error handling: any step failing logs + continues, never crashes
#   - Cooldown tracking: pair-level cooldowns prevent overtrading
# =============================================================

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from core.logger import get_logger
from core.pip_utils import get_pip_size

log = get_logger(__name__)

# ── Config imports with safe fallbacks ──────────────────────────
try:
    from rpde.config import (
        PATTERN_COOLDOWN_MINUTES,
        MAX_PATTERN_POSITIONS,
        SPREAD_LIMITS,
        SAFETY_MAX_DAILY_LOSS_PCT,
        SAFETY_MAX_WEEKLY_LOSS_PCT,
        SAFETY_MAX_POSITIONS,
        SAFETY_MAX_CONSECUTIVE_LOSSES,
        RL_RETRAIN_DAYS,
    )
except ImportError:
    log.warning("[LIVE_ENGINE] rpde.config imports failed — using defaults")
    PATTERN_COOLDOWN_MINUTES = 30
    MAX_PATTERN_POSITIONS = 5
    SPREAD_LIMITS = {}
    SAFETY_MAX_DAILY_LOSS_PCT = 3.0
    SAFETY_MAX_WEEKLY_LOSS_PCT = 5.0
    SAFETY_MAX_POSITIONS = 5
    SAFETY_MAX_CONSECUTIVE_LOSSES = 5
    RL_RETRAIN_DAYS = 7

try:
    from config.settings import (
        PAIR_WHITELIST,
        MAX_SPREAD,
        MAGIC_NUMBER,
        RISK_PERCENT_PER_TRADE,
    )
except ImportError:
    PAIR_WHITELIST = ["EURJPY", "GBPJPY"]
    MAX_SPREAD = {"DEFAULT": 4.0}
    MAGIC_NUMBER = 200001
    RISK_PERCENT_PER_TRADE = 1.0


# ════════════════════════════════════════════════════════════════
#  ACTIVE TRADE TRACKING
# ════════════════════════════════════════════════════════════════

@dataclass
class ActiveTrade:
    """
    Tracks a single RPDE-initiated trade from entry to exit.

    Stores everything needed for the ExperienceBuffer when the
    trade closes, plus mid-trade RL evaluation metadata.

    Attributes:
        ticket:           MT5 position ticket ID.
        pair:             Currency pair symbol.
        direction:        "BUY" or "SELL".
        entry_time:       ISO datetime string when trade opened.
        entry_price:      Price at trade entry.
        lot_size:         Lot size executed.
        sl_pips:          Stop-loss distance in pips.
        tp_pips:          Take-profit distance in pips.
        size_r:           R-multiple size from RL agent.
        stop_type:        "tight" or "medium" from RL agent.
        tp_r:             Take-profit R-multiple from RL agent.
        fusion_confidence: Fusion layer confidence at entry.
        fusion_expected_r: Fusion layer expected R at entry.
        gate_recommendation: PatternGate recommendation at entry.
        rl_action_name:   RL agent action name at entry.
        rl_confidence:    RL agent confidence at entry.
        rl_predicted_value: RL agent predicted state value.
        session:          Trading session at entry.
        spread_at_entry:  Spread in pips at entry.
        atr_at_entry:     ATR value at entry in pips.
        strategy:         Strategy tag for order_manager.
        ai_score:         AI score for order_manager.
        peak_profit_pips: Maximum profit observed (for MFE).
        max_adverse_pips: Maximum adverse excursion (for MAE).
        last_rl_check:    Timestamp of last RL mid-trade check.
        closed:           Whether the trade has been closed.
    """
    ticket: int
    pair: str
    direction: str
    entry_time: str
    entry_price: float
    lot_size: float
    sl_pips: float
    tp_pips: float
    size_r: float = 1.0
    stop_type: str = "medium"
    tp_r: float = 2.0
    fusion_confidence: float = 0.0
    fusion_expected_r: float = 0.0
    gate_recommendation: str = ""
    rl_action_name: str = ""
    rl_confidence: float = 0.0
    rl_predicted_value: float = 0.0
    session: str = ""
    spread_at_entry: float = 0.0
    atr_at_entry: float = 0.0
    strategy: str = "RPDE"
    ai_score: float = 0.0
    peak_profit_pips: float = 0.0
    max_adverse_pips: float = 0.0
    last_rl_check: Optional[datetime] = None
    closed: bool = False

    @property
    def hold_time_hours(self) -> float:
        """Duration since entry in hours."""
        try:
            entry = datetime.fromisoformat(self.entry_time)
            now = datetime.now(timezone.utc)
            if entry.tzinfo is None:
                entry = entry.replace(tzinfo=timezone.utc)
            return max(0.0, (now - entry).total_seconds() / 3600.0)
        except Exception:
            return 0.0


# ════════════════════════════════════════════════════════════════
#  LIVE ENGINE
# ════════════════════════════════════════════════════════════════

class LiveEngine:
    """
    Main live trading orchestrator for the RPDE system.

    Wires the complete decision chain:
        M5 Bar -> Features -> PatternGate -> RL Agent -> Safety -> Execute

    Graceful degradation:
        - If RL agent not trained, uses PatternGate output directly
        - If PatternGate has no models, skips (no blind trades)
        - Any subsystem failure is logged and skipped (never crashes)

    Thread safety:
        - All shared state mutations protected by threading.RLock
        - Safe for use in multi-threaded environments (e.g., main
          loop running alongside 1-second position management)

    Usage:
        engine = LiveEngine(pairs=PAIR_WHITELIST, paper_mode=True)
        engine.initialize()

        # On each M5 bar close:
        engine.on_m5_bar(pair, master_report, market_report, smc_report, flow_data)

        # Every 1 second:
        engine.manage_open_positions()

        # Periodically:
        engine.run_learning_cycle()

        # Status check:
        status = engine.get_status()

        # Graceful shutdown:
        engine.shutdown()
    """

    # Minimum interval between RL mid-trade checks (seconds)
    _RL_MID_TRADE_INTERVAL = 30

    # Risk percentage per trade for lot size calculation
    _DEFAULT_RISK_PCT = 0.01  # 1% of equity

    # Maximum daily pattern-based trades counter
    _MAX_DAILY_TRADES = 8

    def __init__(self, pairs: Optional[List[str]] = None,
                 paper_mode: bool = True):
        """
        Initialize the live engine with all subsystem references.

        Args:
            pairs:      List of currency pairs to trade. If None,
                        uses PAIR_WHITELIST from config.
            paper_mode: If True, log trades but don't execute via MT5.
                        Always default to True for safety.
        """
        self.pairs = [p.upper() for p in (pairs or PAIR_WHITELIST)]
        self.paper_mode = paper_mode

        # ── Subsystem references (populated in initialize()) ──
        self._gate = None           # PatternGate
        self._rl_agents: Dict[str, Any] = {}  # {pair: RLDecisionEngine}
        self._safety = None         # SafetyGuardSystem
        self._learning_loop = None  # ContinuousLearningLoop

        # ── State tracking ──
        self._last_bar_time: Dict[str, Optional[datetime]] = {}
        self._active_trades: Dict[int, ActiveTrade] = {}  # ticket -> ActiveTrade
        self._cooldowns: Dict[str, Optional[datetime]] = {}
        self._daily_trade_count: int = 0
        self._daily_reset_date: Optional[str] = None

        # ── Performance tracking ──
        self._consecutive_losses: int = 0
        self._consecutive_wins: int = 0
        self._peak_equity: float = 0.0
        self._daily_pnl_usd: float = 0.0
        self._weekly_pnl_usd: float = 0.0
        self._weekly_start_date: Optional[str] = None

        # ── Execution stats ──
        self._stats = {
            "total_signals": 0,
            "gate_skips": 0,
            "rl_skips": 0,
            "safety_blocks": 0,
            "executed": 0,
            "execution_failures": 0,
            "paper_executed": 0,
            "outcomes_recorded": 0,
        }

        # ── Thread safety ──
        self._lock = threading.RLock()

        # ── MT5 symbol info cache (lot limits, pip values) ──
        self._symbol_info_cache: Dict[str, dict] = {}

        # ── Engine state ──
        self._initialized = False
        self._running = False

        log.info(
            f"[LIVE_ENGINE] Created (pairs={len(self.pairs)}, "
            f"paper_mode={self.paper_mode})"
        )

    # ════════════════════════════════════════════════════════════
    #  INITIALIZATION
    # ════════════════════════════════════════════════════════════

    def initialize(self) -> bool:
        """
        Initialize all subsystems. Call once at startup.

        Loads PatternGate models, RL agents, SafetyGuardSystem,
        and ContinuousLearningLoop. Caches MT5 symbol info.

        Returns:
            True if all critical subsystems initialized successfully.
        """
        with self._lock:
            log.info("[LIVE_ENGINE] Initializing subsystems...")

            # ── Step 1: PatternGate ──
            try:
                from rpde.pattern_gate import PatternGate
                self._gate = PatternGate()
                self._gate.initialize(self.pairs)
                gate_status = self._gate.get_status()
                log.info(
                    f"[LIVE_ENGINE] PatternGate initialized: "
                    f"{gate_status.get('models_loaded', 0)} models, "
                    f"{gate_status.get('patterns_loaded', 0)} patterns"
                )
            except Exception as e:
                log.error(f"[LIVE_ENGINE] PatternGate init FAILED: {e}")
                self._gate = None

            # ── Step 2: RL Agents (per-pair, graceful degradation) ──
            for pair in self.pairs:
                try:
                    from rpde.rl_agent import RLDecisionEngine
                    agent = RLDecisionEngine(pair)
                    loaded = agent.load()
                    if loaded and agent.is_trained:
                        self._rl_agents[pair] = agent
                        log.info(
                            f"[LIVE_ENGINE] RL agent loaded for {pair} "
                            f"(episodes={agent.training_episodes})"
                        )
                    else:
                        log.info(
                            f"[LIVE_ENGINE] No trained RL model for {pair} "
                            f"— will use Gate-only mode"
                        )
                except Exception as e:
                    log.warning(
                        f"[LIVE_ENGINE] RL agent init failed for {pair}: {e} "
                        f"— will use Gate-only mode"
                    )

            # ── Step 3: SafetyGuardSystem ──
            try:
                from rpde.safety_guards import SafetyGuardSystem
                self._safety = SafetyGuardSystem()
                log.info("[LIVE_ENGINE] SafetyGuardSystem initialized")
            except Exception as e:
                log.error(f"[LIVE_ENGINE] SafetyGuardSystem init FAILED: {e}")
                self._safety = None

            # ── Step 4: ContinuousLearningLoop ──
            try:
                from rpde.experience_buffer import ContinuousLearningLoop
                self._learning_loop = ContinuousLearningLoop()
                log.info("[LIVE_ENGINE] ContinuousLearningLoop initialized")
            except Exception as e:
                log.warning(
                    f"[LIVE_ENGINE] ContinuousLearningLoop init failed: {e} "
                    f"— trade recording will be unavailable"
                )
                self._learning_loop = None

            # ── Step 5: Cache MT5 symbol info ──
            self._cache_symbol_info()

            # ── Step 6: Initialize state ──
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            self._daily_reset_date = today
            self._daily_trade_count = 0
            self._weekly_start_date = self._get_week_start()

            self._initialized = True
            self._running = True

            log.info(
                f"[LIVE_ENGINE] Initialization complete — "
                f"{len(self._rl_agents)}/{len(self.pairs)} RL agents loaded, "
                f"paper_mode={self.paper_mode}"
            )
            return True

    def _cache_symbol_info(self):
        """Cache MT5 symbol information for lot size calculations."""
        try:
            import MetaTrader5 as mt5
            for pair in self.pairs:
                info = mt5.symbol_info(pair)
                if info is not None:
                    self._symbol_info_cache[pair] = {
                        "volume_min": info.volume_min,
                        "volume_max": info.volume_max,
                        "volume_step": info.volume_step,
                        "digits": info.digits,
                        "point": info.point,
                        "trade_stops_level": info.trade_stops_level,
                        "contract_size": getattr(info, 'trade_contract_size', 100000),
                        "pip_value": self._estimate_pip_value(pair, info),
                    }
                    log.debug(
                        f"[LIVE_ENGINE] Cached {pair}: "
                        f"lot_min={info.volume_min} lot_step={info.volume_step} "
                        f"contract={self._symbol_info_cache[pair]['contract_size']}"
                    )
                else:
                    log.warning(f"[LIVE_ENGINE] Cannot get symbol info for {pair}")
        except Exception as e:
            log.warning(f"[LIVE_ENGINE] MT5 symbol info caching failed: {e}")

    def _estimate_pip_value(self, pair: str, sym_info) -> float:
        """
        Estimate the USD value of 1 pip for lot size calculations.

        For standard forex pairs: pip_value = pip_size * contract_size
        For JPY pairs: pip_value = 0.01 * contract_size
        For metals: pip_value = 0.01 * contract_size / price

        Args:
            pair: Currency pair string.
            sym_info: MT5 symbol info object.

        Returns:
            Estimated USD pip value per standard lot.
        """
        try:
            contract = getattr(sym_info, 'trade_contract_size', 100000)
            pip_size = get_pip_size(pair)

            # For most forex: pip_value = pip_size * contract_size
            # e.g., EURUSD: 0.0001 * 100000 = $10/pip/lot
            # For JPY pairs: 0.01 * 100000 = $10/pip/lot (approximately,
            # exact value depends on USD/JPY rate)
            if "JPY" in pair.upper():
                # Approximate: need to get current USD/JPY price
                try:
                    import MetaTrader5 as mt5
                    usdjpy_info = mt5.symbol_info_tick("USDJPY")
                    if usdjpy_info:
                        price = (usdjpy_info.bid + usdjpy_info.ask) / 2
                        return (pip_size * contract) / price
                except Exception:
                    pass
                return 6.5  # Approximate JPY pip value

            if "XAG" in pair.upper():
                # Silver: 0.01 * 5000 = $50/pip/lot (5000 oz contract)
                return pip_size * contract

            if "XAU" in pair.upper():
                # Gold: 0.1 * 100 = $10/pip/lot (100 oz contract)
                return pip_size * contract

            return pip_size * contract
        except Exception:
            return 10.0  # Conservative default for standard forex

    # ════════════════════════════════════════════════════════════
    #  MAIN DECISION CHAIN — on_m5_bar()
    # ════════════════════════════════════════════════════════════

    def on_m5_bar(self, pair: str, master_report: dict,
                  market_report: dict, smc_report: dict,
                  flow_data: dict):
        """
        Called on every M5 bar close. This is the main decision chain.

        Steps:
            1. Extract 93 features from reports
            2. Build feature array for PatternGate
            3. Run PatternGate.evaluate() -> gate_result
            4. If gate says SKIP -> return
            5. Build fusion_result dict from gate_result
            6. Build market_state dict (atr, spread, session, etc.)
            7. Build portfolio_state dict (open positions, daily P&L)
            8. Run RL agent.decide() -> rl_decision
            9. If RL says SKIP -> fall back to Gate-only if possible
           10. Build trade_request dict
           11. Build account_state dict (from MT5)
           12. Run safety.check() -> safety_result
           13. If safety fails -> log and return
           14. Execute via MT5 bridge -> place_order()
           15. Track the active trade
           16. Log the full decision chain

        Args:
            pair:          Currency pair symbol (e.g., "EURJPY").
            master_report: Combined master analysis report dict.
            market_report: Market analysis report dict.
            smc_report:    SMC structure report dict.
            flow_data:     Order flow data dict.
        """
        pair_upper = pair.upper()

        if not self._initialized:
            log.debug(f"[LIVE_ENGINE] Not initialized — skipping {pair_upper}")
            return

        if not self._running:
            return

        with self._lock:
            self._stats["total_signals"] += 1

        # ── Step 0: Reset daily counter if new day ──
        self._check_daily_reset()

        # ── Step 0.5: Check cooldown for this pair ──
        if not self._check_cooldown(pair_upper):
            return

        # ── Step 0.6: Check daily trade limit ──
        if self._daily_trade_count >= self._MAX_DAILY_TRADES:
            log.debug(
                f"[LIVE_ENGINE] Daily trade limit reached "
                f"({self._daily_trade_count}/{self._MAX_DAILY_TRADES})"
            )
            return

        # ── Step 1: Extract 93 features ──
        try:
            from rpde.feature_snapshot import extract_snapshot_from_report
            spread_pips = float(
                master_report.get('spread', 0)
                or master_report.get('spread_pips', 0)
                or 0
            )
            snapshot = extract_snapshot_from_report(
                pair=pair_upper,
                master_report=master_report,
                market_report=market_report,
                smc_report=smc_report,
                flow_data=flow_data,
                direction=master_report.get('direction'),
                spread_pips=spread_pips,
            )
            if not snapshot:
                log.debug(f"[LIVE_ENGINE] Feature extraction empty for {pair_upper}")
                return
        except Exception as e:
            log.error(f"[LIVE_ENGINE] Feature extraction failed for {pair_upper}: {e}")
            return

        # ── Step 2: Build feature array for PatternGate ──
        try:
            from ai_engine.ml_gate import FEATURE_NAMES
            features_array = np.array(
                [float(snapshot.get(name, 0.0)) for name in FEATURE_NAMES],
                dtype=np.float32,
            )
        except Exception as e:
            log.error(f"[LIVE_ENGINE] Feature array build failed: {e}")
            return

        # ── Step 3: Run PatternGate ──
        if self._gate is None:
            log.debug(f"[LIVE_ENGINE] PatternGate not available — skipping")
            return

        try:
            gate_result = self._gate.evaluate(pair_upper, features_array, master_report)
        except Exception as e:
            log.error(f"[LIVE_ENGINE] PatternGate evaluation failed: {e}")
            return

        # ── Step 4: If gate says SKIP -> return ──
        recommendation = gate_result.get('recommendation', 'SKIP')
        if recommendation == 'SKIP':
            with self._lock:
                self._stats["gate_skips"] += 1
            log.debug(
                f"[LIVE_ENGINE] {pair_upper} SKIP (Gate: {gate_result.get('reason', '')})"
            )
            return

        # ── Step 5: Build fusion_result from gate_result ──
        fusion_result = self._build_fusion_result(gate_result, pair_upper)

        # ── Step 6: Build market_state ──
        atr_pips = self._get_atr_pips(pair_upper)
        market_state = self._build_market_state(pair_upper, market_report, atr_pips)

        # ── Step 7: Build portfolio_state ──
        account_state = self._build_account_state()
        portfolio_state = self._build_portfolio_state(account_state, market_state)

        # ── Step 8: Run RL agent ──
        rl_decision = None
        rl_available = pair_upper in self._rl_agents

        if rl_available:
            try:
                agent = self._rl_agents[pair_upper]
                rl_decision = agent.decide(fusion_result, market_state, portfolio_state)
            except Exception as e:
                log.error(f"[LIVE_ENGINE] RL agent failed for {pair_upper}: {e}")
                rl_decision = None

        # ── Step 9: If RL says SKIP ──
        if rl_decision and not rl_decision.get('entry', False):
            with self._lock:
                self._stats["rl_skips"] += 1

            # In Gate-only mode (no RL or RL says SKIP),
            # if Gate says TAKE, we can still trade with default params
            if recommendation == 'TAKE' and not rl_available:
                log.info(
                    f"[LIVE_ENGINE] {pair_upper} RL not available — "
                    f"falling back to Gate-only mode (TAKE)"
                )
                direction = gate_result.get('direction')
                if direction:
                    rl_decision = {
                        "action": 0,
                        "action_name": "GATE_ONLY",
                        "entry": True,
                        "direction": direction,
                        "size_r": 1.0,
                        "stop_type": "medium",
                        "tp_r": 2.0,
                        "confidence": gate_result.get('combined_confidence', 0.5),
                        "value": gate_result.get('expected_r', 0.0),
                        "reason": "Gate-only fallback (no RL agent trained)",
                    }
                else:
                    log.debug(
                        f"[LIVE_ENGINE] {pair_upper} SKIP — "
                        f"Gate CAUTION with no direction, RL SKIP"
                    )
                    return
            elif recommendation == 'TAKE':
                log.info(
                    f"[LIVE_ENGINE] {pair_upper} RL SKIP but Gate says TAKE — "
                    f"respecting RL decision (SKIP)"
                )
                return
            else:
                log.debug(
                    f"[LIVE_ENGINE] {pair_upper} Gate={recommendation} RL=SKIP — skipped"
                )
                return
        elif not rl_decision:
            # No RL decision at all — fallback to Gate-only
            if recommendation == 'TAKE':
                direction = gate_result.get('direction')
                if direction:
                    rl_decision = {
                        "action": 0,
                        "action_name": "GATE_ONLY",
                        "entry": True,
                        "direction": direction,
                        "size_r": 1.0,
                        "stop_type": "medium",
                        "tp_r": 2.0,
                        "confidence": gate_result.get('combined_confidence', 0.5),
                        "value": gate_result.get('expected_r', 0.0),
                        "reason": "Gate-only fallback (no RL decision)",
                    }
                else:
                    return
            else:
                return

        # ── Step 10: Build trade_request ──
        trade_request = {
            "pair": pair_upper,
            "direction": rl_decision.get("direction"),
            "size_r": rl_decision.get("size_r", 1.0),
            "stop_type": rl_decision.get("stop_type", "medium"),
            "tp_r": rl_decision.get("tp_r", 2.0),
            "confidence": rl_decision.get("confidence", 0.5),
            "strategy": "RPDE",
            "source": "live_engine",
        }

        # ── Step 11: Build account_state if not already built ──
        if account_state is None:
            account_state = self._build_account_state()

        # ── Step 12: Run safety checks ──
        if self._safety is not None:
            try:
                safety_result = self._safety.check(
                    trade_request, account_state, market_state
                )
                if not safety_result.passed:
                    with self._lock:
                        self._stats["safety_blocks"] += 1
                    log.info(
                        f"[LIVE_ENGINE] {pair_upper} SAFETY BLOCK: "
                        f"{safety_result.guard_name} — {safety_result.message}"
                    )
                    return
            except Exception as e:
                log.error(f"[LIVE_ENGINE] Safety check failed: {e}")
                return
        else:
            log.warning("[LIVE_ENGINE] SafetyGuardSystem not available — "
                        "skipping safety checks (DANGEROUS)")

        # ── Step 13: Execute via MT5 bridge ──
        order_params = self._convert_rl_to_order(
            pair_upper, rl_decision, atr_pips, account_state
        )

        if order_params is None:
            log.warning(
                f"[LIVE_ENGINE] {pair_upper} Order conversion failed — skipping"
            )
            return

        execution_success = False
        ticket = 0

        if self.paper_mode:
            # Paper mode: log but don't execute
            execution_success = True
            ticket = int(time.time() * 1000) % (10**9)  # Fake ticket
            with self._lock:
                self._stats["paper_executed"] += 1
            log.info(
                f"[LIVE_ENGINE] PAPER TRADE {pair_upper} "
                f"{order_params['direction']} "
                f"lot={order_params['lot_size']:.2f} "
                f"sl={order_params['sl_pips']:.1f}p "
                f"tp={order_params['tp_pips']:.1f}p "
                f"(R={rl_decision.get('size_r', 1.0)})"
            )
        else:
            # Live execution
            try:
                from execution.order_manager import place_order
                execution_success = place_order(
                    symbol=order_params['symbol'],
                    direction=order_params['direction'],
                    lot_size=order_params['lot_size'],
                    sl_pips=order_params['sl_pips'],
                    tp_pips=order_params['tp_pips'],
                    strategy=order_params['strategy'],
                    ai_score=order_params['ai_score'],
                    session=market_state.get('session', ''),
                    market_regime=market_state.get('market_regime', ''),
                    rsi=market_state.get('rsi'),
                    atr=atr_pips,
                    spread=market_state.get('spread'),
                )
                if execution_success:
                    with self._lock:
                        self._stats["executed"] += 1
                    # Try to get the ticket from the most recent position
                    ticket = self._get_latest_ticket(pair_upper)
                else:
                    with self._lock:
                        self._stats["execution_failures"] += 1
            except Exception as e:
                log.error(f"[LIVE_ENGINE] Order execution failed: {e}")
                with self._lock:
                    self._stats["execution_failures"] += 1
                return

        # ── Step 14: Track active trade ──
        if execution_success and ticket > 0:
            active_trade = ActiveTrade(
                ticket=ticket,
                pair=pair_upper,
                direction=order_params['direction'],
                entry_time=datetime.now(timezone.utc).isoformat(),
                entry_price=order_params.get('entry_price', 0.0),
                lot_size=order_params['lot_size'],
                sl_pips=order_params['sl_pips'],
                tp_pips=order_params['tp_pips'],
                size_r=rl_decision.get('size_r', 1.0),
                stop_type=rl_decision.get('stop_type', 'medium'),
                tp_r=rl_decision.get('tp_r', 2.0),
                fusion_confidence=fusion_result.get('combined_confidence', 0.0),
                fusion_expected_r=fusion_result.get('combined_expected_r', 0.0),
                gate_recommendation=gate_result.get('recommendation', ''),
                rl_action_name=rl_decision.get('action_name', ''),
                rl_confidence=rl_decision.get('confidence', 0.0),
                rl_predicted_value=rl_decision.get('value', 0.0),
                session=market_state.get('session', ''),
                spread_at_entry=market_state.get('spread', 0.0),
                atr_at_entry=atr_pips,
                strategy=order_params['strategy'],
                ai_score=order_params['ai_score'],
            )
            with self._lock:
                self._active_trades[ticket] = active_trade
                self._daily_trade_count += 1
                self._cooldowns[pair_upper] = datetime.now(timezone.utc)

            # Update safety system's per-pair position count
            if self._safety:
                self._update_safety_positions()

        # ── Step 15: Log full decision chain ──
        log.info(
            f"[LIVE_ENGINE] {pair_upper} {rl_decision.get('direction', '?')} | "
            f"Gate: {recommendation} conf={gate_result.get('combined_confidence', 0):.2f} "
            f"R={gate_result.get('expected_r', 0):.2f} | "
            f"RL: {rl_decision.get('action_name', '?')} "
            f"conf={rl_decision.get('confidence', 0):.2f} | "
            f"Safety: {'PASS' if self._safety else 'N/A'} | "
            f"Execute: {'OK' if execution_success else 'FAIL'}"
        )

    # ════════════════════════════════════════════════════════════
    #  MT5 EXECUTION BRIDGE
    # ════════════════════════════════════════════════════════════

    def _convert_rl_to_order(self, pair: str, rl_decision: dict,
                             atr_pips: float,
                             account_state: dict) -> Optional[dict]:
        """
        MT5 Execution Bridge: Convert RL decision to place_order() params.

        Converts:
          - size_r  -> lot_size (using account equity, risk %, ATR)
          - stop_type -> sl_pips ("tight" = ATR * 0.75, "medium" = ATR * 1.0)
          - tp_r    -> tp_pips (sl_pips * tp_r)

        Lot size formula:
          lot_size = (equity * risk_pct * size_r) / (sl_pips * pip_value)

        Respects broker limits (min lot, max lot, lot step).

        Args:
            pair:          Currency pair string.
            rl_decision:   RL agent decision dict.
            atr_pips:      Current ATR in pips.
            account_state: Account state dict from MT5.

        Returns:
            Dict with place_order() parameters, or None if conversion fails.
        """
        try:
            direction = rl_decision.get('direction')
            if direction not in ('BUY', 'SELL'):
                log.warning(f"[LIVE_ENGINE] Invalid direction: {direction}")
                return None

            size_r = float(rl_decision.get('size_r', 1.0))
            stop_type = rl_decision.get('stop_type', 'medium')
            tp_r = float(rl_decision.get('tp_r', 2.0))

            # ── SL from stop_type ──
            if atr_pips <= 0:
                log.warning(f"[LIVE_ENGINE] Invalid ATR for {pair}: {atr_pips}")
                return None

            if stop_type == "tight":
                sl_pips = atr_pips * 0.75
            else:
                sl_pips = atr_pips * 1.0

            # Floor SL to 3 pips (order_manager minimum)
            sl_pips = max(sl_pips, 3.0)

            # ── TP from R-multiple ──
            tp_pips = sl_pips * tp_r
            tp_pips = max(tp_pips, sl_pips + 1.0)  # TP must exceed SL

            # ── Lot size from risk ──
            equity = float(account_state.get('equity', 0))
            if equity <= 0:
                log.warning(f"[LIVE_ENGINE] Invalid equity: {equity}")
                return None

            risk_pct = self._DEFAULT_RISK_PCT
            pip_value = self._get_pip_value_for_pair(pair)

            if pip_value <= 0 or sl_pips <= 0:
                log.warning(
                    f"[LIVE_ENGINE] Invalid pip_value={pip_value} or sl_pips={sl_pips}"
                )
                return None

            lot_size = (equity * risk_pct * size_r) / (sl_pips * pip_value)

            # ── Respect broker limits ──
            sym_info = self._symbol_info_cache.get(pair)
            if sym_info:
                lot_min = sym_info.get('volume_min', 0.01)
                lot_max = sym_info.get('volume_max', 100.0)
                lot_step = sym_info.get('volume_step', 0.01)

                # Round to lot step
                lot_size = round(lot_size / lot_step) * lot_step

                # Clamp to limits
                lot_size = max(lot_min, min(lot_max, lot_size))

                # If lot_size < min, trade is too small for broker
                if lot_size < lot_min:
                    log.info(
                        f"[LIVE_ENGINE] {pair}: computed lot {lot_size:.4f} "
                        f"< broker min {lot_min} — trade too small, skipping"
                    )
                    return None
            else:
                # Fallback: round to 0.01
                lot_size = round(lot_size, 2)
                if lot_size < 0.01:
                    return None

            # ── Compute AI score for order_manager ──
            confidence = float(rl_decision.get('confidence', 0.5))
            expected_r = float(rl_decision.get('value', 0.0))
            ai_score = min(100, max(50, confidence * 60 + expected_r * 20 + 30))

            # ── Get current price for ActiveTrade tracking ──
            entry_price = 0.0
            try:
                import MetaTrader5 as mt5
                tick = mt5.symbol_info_tick(pair)
                if tick:
                    entry_price = tick.ask if direction == "BUY" else tick.bid
            except Exception:
                pass

            return {
                'symbol': pair,
                'direction': direction,
                'lot_size': lot_size,
                'sl_pips': sl_pips,
                'tp_pips': tp_pips,
                'strategy': 'RPDE',
                'ai_score': ai_score,
                'entry_price': entry_price,
            }

        except Exception as e:
            log.error(f"[LIVE_ENGINE] Order conversion failed for {pair}: {e}")
            return None

    def _get_pip_value_for_pair(self, pair: str) -> float:
        """Get cached pip value for a pair."""
        info = self._symbol_info_cache.get(pair)
        if info:
            return info.get('pip_value', 10.0)
        return 10.0  # Conservative default

    def _get_atr_pips(self, pair: str) -> float:
        """Get current ATR in pips for a pair."""
        try:
            from execution.order_manager import get_atr_for_symbol
            atr = get_atr_for_symbol(pair, 'M5', 50)
            return float(atr) if atr and atr > 0 else 15.0  # Default 15 pips
        except Exception:
            return 15.0

    def _get_latest_ticket(self, pair: str) -> int:
        """Get the most recent position ticket for a pair from MT5."""
        try:
            import MetaTrader5 as mt5
            positions = mt5.positions_get(symbol=pair)
            if positions:
                # Sort by time, newest first
                pos = max(positions, key=lambda p: p.time)
                return pos.ticket
        except Exception:
            pass
        return 0

    # ════════════════════════════════════════════════════════════
    #  STATE BUILDERS
    # ════════════════════════════════════════════════════════════

    def _build_fusion_result(self, gate_result: dict, pair: str) -> dict:
        """
        Build fusion_result dict from gate_result for RL agent input.

        Maps PatternGate output to the format expected by the RL agent's
        observation builder.

        Args:
            gate_result: PatternGate evaluate() output dict.
            pair:        Currency pair string.

        Returns:
            Fusion result dict with keys matching rl_agent expectations.
        """
        direction = gate_result.get('direction')
        recommendation = gate_result.get('recommendation', 'SKIP')

        return {
            'combined_confidence': float(
                gate_result.get('combined_confidence', 0.0)
            ),
            'combined_expected_r': float(
                gate_result.get('expected_r', 0.0)
            ),
            'direction': direction,
            'recommendation': recommendation,
            'tft_contribution': float(
                gate_result.get('tft_contribution', 0.0)
            ),
            'signal_agreement': gate_result.get('signal_agreement', 'PARTIAL'),
            'reversal_warning': bool(gate_result.get('reversal_warning', False)),
            'xgb_confidence': float(
                gate_result.get('model_confidence', 0.0)
            ),
            'xgb_predicted_r': float(
                gate_result.get('model_predicted_r', 0.0)
            ),
            'xgb_available': bool(
                gate_result.get('model_confidence', 0) > 0
            ),
            'tft_available': bool(
                gate_result.get('tft_contribution', 0) > 0
            ),
            'tft_pattern_match': float(
                gate_result.get('tft_pattern_match', 0.0)
            ),
            'tft_momentum': float(gate_result.get('tft_momentum', 0.0)),
            'tft_reversal': float(gate_result.get('tft_reversal', 0.0)),
            'pattern_match_score': float(
                gate_result.get('pattern_match_score', 0.0)
            ),
            'pattern_available': bool(
                gate_result.get('pattern_match_score', 0) > 0.3
            ),
            'agreement_count': 2 if recommendation == 'TAKE' else 1,
            'weights': gate_result.get('fusion_weights', {}),
        }

    def _build_market_state(self, pair: str, market_report: dict,
                            atr_pips: float) -> dict:
        """
        Build market_state dict for RL agent observation.

        Extracts and normalizes market microstructure features from
        the market report and additional computed metrics.

        Args:
            pair:          Currency pair string.
            market_report: Market analysis report dict.
            atr_pips:      Current ATR in pips.

        Returns:
            Market state dict with all RL-expected keys.
        """
        mr = market_report or {}

        # Session computation
        from rpde.feature_snapshot import compute_session
        session = mr.get('session', compute_session(datetime.now(timezone.utc)))

        # Normalize session name for RL agent encoding
        session_normalized = session
        if 'OVERLAP' in session:
            session_normalized = 'NY_LONDON_OVERLAP'
        elif 'LONDON' in session and 'OPEN' in session:
            session_normalized = 'LONDON_OPEN'
        elif 'LONDON' in session:
            session_normalized = 'LONDON'
        elif 'NY' in session or 'NEW_YORK' in session:
            session_normalized = 'NY_AFTERNOON'
        elif 'TOKYO' in session or 'ASIAN' in session:
            session_normalized = 'TOKYO'
        elif 'SYDNEY' in session:
            session_normalized = 'SYDNEY'

        spread = float(
            mr.get('spread', 0)
            or mr.get('spread_pips', 0)
            or 0
        )

        return {
            'atr': atr_pips,
            'atr_percentile': float(mr.get('atr_percentile', 50.0)),
            'momentum_score': float(mr.get('momentum_score', 0.0)),
            'spread': spread,
            'session': session_normalized,
            'trend_strength': float(mr.get('trend_strength', 0.0)),
            'market_regime': str(mr.get('market_regime', 'RANGING')),
            'rsi': float(mr.get('rsi', 50.0)),
            'volatility': float(mr.get('volatility', 0.0)),
            'is_choppy': bool(mr.get('is_choppy', False)),
            'current_spread': spread,
            'average_spread': float(mr.get('avg_spread', spread)),
        }

    def _build_portfolio_state(self, account_state: dict,
                               market_state: dict) -> dict:
        """
        Build portfolio_state dict for RL agent observation.

        This is the Real-time P&L -> RL feedback path. Computes
        current open positions, daily P&L, drawdown, and exposure.

        Args:
            account_state: Account state from MT5.
            market_state:  Current market state dict.

        Returns:
            Portfolio state dict with all RL-expected keys.
        """
        with self._lock:
            open_positions = len(self._active_trades)
            consecutive_losses = self._consecutive_losses
            consecutive_wins = self._consecutive_wins
            daily_pnl_r = self._daily_pnl_r
            max_exposure = float(SAFETY_MAX_POSITIONS)

        balance = float(account_state.get('balance', 0))
        equity = float(account_state.get('equity', balance))
        drawdown = 0.0

        if balance > 0 and equity < balance:
            drawdown = (balance - equity) / balance

        # Update peak equity tracking
        with self._lock:
            if equity > self._peak_equity:
                self._peak_equity = equity
            if self._peak_equity > 0:
                max_drawdown = (self._peak_equity - equity) / self._peak_equity
            else:
                max_drawdown = 0.0

        # Exposure: fraction of max positions currently in use
        exposure = min(1.0, open_positions / max(max_exposure, 1))

        return {
            'open_positions': open_positions,
            'daily_pnl_r': daily_pnl_r,
            'drawdown': drawdown,
            'max_drawdown': max_drawdown,
            'consecutive_losses': consecutive_losses,
            'consecutive_wins': consecutive_wins,
            'exposure': exposure,
            'max_exposure': max_exposure,
            'balance': balance,
            'equity': equity,
        }

    @property
    def _daily_pnl_r(self) -> float:
        """Current daily P&L in R-multiples (approximate)."""
        # Use USD P&L converted to approximate R
        # 1R ≈ 1% of equity (our default risk per trade)
        with self._lock:
            equity = self._peak_equity if self._peak_equity > 0 else 1000.0
            if equity > 0:
                return self._daily_pnl_usd / (equity * self._DEFAULT_RISK_PCT)
            return 0.0

    def _build_account_state(self) -> dict:
        """
        Get current account state from MT5.

        Returns dict with balance, equity, margin, free_margin,
        margin_level, daily/weekly P&L, and open position count.

        If MT5 is not available, returns a safe default dict.
        """
        try:
            import MetaTrader5 as mt5
            info = mt5.account_info()
            if info is None:
                return self._default_account_state()

            balance = float(info.balance)
            equity = float(info.equity)
            margin = float(info.margin)
            free_margin = float(info.free_margin)

            # Compute daily P&L from closed deals today
            daily_pnl = self._compute_daily_pnl()
            weekly_pnl = self._compute_weekly_pnl()

            daily_pnl_pct = (daily_pnl / balance * 100) if balance > 0 else 0.0
            weekly_pnl_pct = (weekly_pnl / balance * 100) if balance > 0 else 0.0

            margin_level = (equity / margin * 100) if margin > 0 else 0.0

            return {
                'balance': balance,
                'equity': equity,
                'margin': margin,
                'free_margin': free_margin,
                'margin_level': margin_level,
                'daily_pnl': daily_pnl,
                'weekly_pnl': weekly_pnl,
                'daily_pnl_pct': daily_pnl_pct,
                'weekly_pnl_pct': weekly_pnl_pct,
                'open_positions': len(self._active_trades),
                'profit': equity - balance,
                'currency': info.currency,
            }
        except Exception as e:
            log.debug(f"[LIVE_ENGINE] MT5 account_info failed: {e}")
            return self._default_account_state()

    def _default_account_state(self) -> dict:
        """Return a safe default account state when MT5 is unavailable."""
        return {
            'balance': 0.0,
            'equity': 0.0,
            'margin': 0.0,
            'free_margin': 0.0,
            'margin_level': 0.0,
            'daily_pnl': 0.0,
            'weekly_pnl': 0.0,
            'daily_pnl_pct': 0.0,
            'weekly_pnl_pct': 0.0,
            'open_positions': len(self._active_trades),
            'profit': 0.0,
            'currency': 'USD',
        }

    def _compute_daily_pnl(self) -> float:
        """Compute today's P&L from closed deals."""
        try:
            import MetaTrader5 as mt5
            from datetime import timedelta
            now = datetime.now(timezone.utc)
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

            deals = mt5.history_deals_get(today_start, now)
            if not deals:
                return 0.0

            pnl = sum(
                d.profit + (d.swap or 0) + (d.commission or 0)
                for d in deals
                if d.magic == MAGIC_NUMBER
                and d.entry == 1  # DEAL_ENTRY_IN
            )
            return float(pnl)
        except Exception:
            return self._daily_pnl_usd

    def _compute_weekly_pnl(self) -> float:
        """Compute this week's P&L from closed deals."""
        try:
            import MetaTrader5 as mt5
            from datetime import timedelta
            now = datetime.now(timezone.utc)
            week_start = now - timedelta(days=now.weekday())
            week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)

            deals = mt5.history_deals_get(week_start, now)
            if not deals:
                return 0.0

            pnl = sum(
                d.profit + (d.swap or 0) + (d.commission or 0)
                for d in deals
                if d.magic == MAGIC_NUMBER
                and d.entry == 1  # DEAL_ENTRY_IN
            )
            return float(pnl)
        except Exception:
            return self._weekly_pnl_usd

    # ════════════════════════════════════════════════════════════
    #  POSITION MANAGEMENT (called every 1 second)
    # ════════════════════════════════════════════════════════════

    def manage_open_positions(self):
        """
        Called every 1 second. Manages all open RPDE positions.

        Steps:
            1. Call order_manager.manage_positions() (trailing, BE, partial TP)
            2. Call order_manager.sync_closed_trades() (sync broker-side closes)
            3. For each active RPDE trade, check if RL wants early exit
            4. Detect closed trades and record outcomes to ExperienceBuffer
            5. Update MFE/MAE tracking for active trades
        """
        if not self._initialized or not self._running:
            return

        # ── Step 1 & 2: Delegate to order_manager ──
        if not self.paper_mode:
            try:
                from execution.order_manager import manage_positions, sync_closed_trades
                manage_positions()
                sync_closed_trades()
            except Exception as e:
                log.debug(f"[LIVE_ENGINE] manage_positions/sync error: {e}")

        # ── Step 3: Check for closed trades and update tracking ──
        now = datetime.now(timezone.utc)
        closed_tickets = []

        with self._lock:
            for ticket, trade in list(self._active_trades.items()):
                if trade.closed:
                    closed_tickets.append(ticket)
                    continue

                # Check if position still exists in MT5
                if not self.paper_mode:
                    is_open = self._is_position_open(trade.pair, ticket)
                    if not is_open:
                        closed_tickets.append(ticket)
                        continue

                # ── Update MFE/MAE ──
                current_pnl = self._get_position_pnl_pips(trade)
                if current_pnl > trade.peak_profit_pips:
                    trade.peak_profit_pips = current_pnl
                if current_pnl < -trade.max_adverse_pips:
                    trade.max_adverse_pips = abs(current_pnl)

                # ── Step 4: RL mid-trade evaluation (every 30s) ──
                if (trade.last_rl_check is None or
                        (now - trade.last_rl_check).total_seconds() >=
                        self._RL_MID_TRADE_INTERVAL):

                    trade.last_rl_check = now
                    self._evaluate_mid_trade(trade)

        # ── Step 5: Process closed trades ──
        for ticket in closed_tickets:
            self._record_closed_trade(ticket)

        # ── Update safety system position counts ──
        if self._safety:
            self._update_safety_positions()

    def _is_position_open(self, pair: str, ticket: int) -> bool:
        """Check if a position is still open in MT5."""
        try:
            import MetaTrader5 as mt5
            positions = mt5.positions_get(symbol=pair)
            if positions:
                return any(p.ticket == ticket for p in positions)
        except Exception:
            pass
        return False

    def _get_position_pnl_pips(self, trade: ActiveTrade) -> float:
        """
        Get current P&L in pips for an active trade.

        For paper mode, returns 0 (no real price data).
        For live mode, queries MT5 for current price.

        Args:
            trade: ActiveTrade instance.

        Returns:
            Current profit/loss in pips (positive = profit).
        """
        if self.paper_mode:
            return 0.0

        try:
            import MetaTrader5 as mt5
            pos = mt5.positions_get(ticket=trade.ticket)
            if pos and len(pos) > 0:
                p = pos[0]
                pip_size = get_pip_size(trade.pair)
                if pip_size <= 0:
                    return 0.0
                if p.type == 0:  # BUY
                    return (p.price_current - p.price_open) / pip_size
                else:  # SELL
                    return (p.price_open - p.price_current) / pip_size
        except Exception:
            pass
        return 0.0

    # ════════════════════════════════════════════════════════════
    #  RL MID-TRADE ACTIONS
    # ════════════════════════════════════════════════════════════

    def _evaluate_mid_trade(self, trade: ActiveTrade):
        """
        RL Mid-Trade Actions: Ask RL agent if we should exit early.

        Uses the RL agent's decision method with a mid-trade observation
        that includes the current profit/loss and time held.

        The agent can return:
          - HOLD:         Keep the position
          - TRAIL_STOP:   Delegate to order_manager trailing (default)
          - EXIT_EARLY:   Close the position now

        Args:
            trade: ActiveTrade instance to evaluate.
        """
        if trade.closed:
            return

        pair = trade.pair
        agent = self._rl_agents.get(pair)
        if agent is None or not agent.is_trained:
            return  # No RL agent — order_manager handles everything

        try:
            # Build mid-trade observation
            current_pnl_r = 0.0
            current_pnl_pips = self._get_position_pnl_pips(trade)

            if trade.sl_pips > 0:
                current_pnl_r = current_pnl_pips / trade.sl_pips

            # Build a modified portfolio state for mid-trade evaluation
            mid_trade_state = {
                'open_positions': len(self._active_trades),
                'daily_pnl_r': self._daily_pnl_r,
                'drawdown': 0.0,
                'max_drawdown': 0.0,
                'consecutive_losses': self._consecutive_losses,
                'consecutive_wins': self._consecutive_wins,
                'exposure': len(self._active_trades) / max(SAFETY_MAX_POSITIONS, 1),
                'max_exposure': float(SAFETY_MAX_POSITIONS),
            }

            # Build a modified market state
            atr_pips = self._get_atr_pips(pair)
            mid_market_state = {
                'atr': atr_pips,
                'atr_percentile': 50.0,
                'momentum_score': 0.0,
                'spread': 0.0,
                'session': trade.session,
                'trend_strength': 0.0,
                'market_regime': 'RANGING',
            }

            # Build a modified fusion_result that reflects current trade state
            mid_fusion = {
                'combined_confidence': trade.fusion_confidence,
                'combined_expected_r': trade.fusion_expected_r,
                'direction': trade.direction,
                'recommendation': trade.gate_recommendation,
                'tft_contribution': 0.0,
                'signal_agreement': 'PARTIAL',
                'reversal_warning': False,
                'xgb_confidence': trade.fusion_confidence,
                'xgb_predicted_r': trade.fusion_expected_r,
                'xgb_available': True,
                'tft_available': False,
                'tft_pattern_match': 0.0,
                'tft_momentum': 0.0,
                'tft_reversal': 0.0,
                'pattern_match_score': 0.0,
                'pattern_available': False,
                'agreement_count': 1,
                'weights': {},
            }

            # Ask RL for decision
            decision = agent.decide(mid_fusion, mid_market_state, mid_trade_state)

            action_name = decision.get('action_name', 'SKIP')

            # Only act on explicit entry actions interpreted as EXIT signals
            # (The RL agent sees the trade is open; if it says SKIP, that's HOLD)
            if action_name == 'SKIP' or not decision.get('entry', False):
                # HOLD — do nothing, order_manager handles trailing/BE
                return

            # If the RL agent suggests a direction OPPOSITE to our trade,
            # consider that an early exit signal
            suggested_dir = decision.get('direction')
            if suggested_dir and suggested_dir != trade.direction:
                if decision.get('confidence', 0) > 0.6:
                    log.info(
                        f"[LIVE_ENGINE] RL MID-TRADE EXIT: {pair} #{trade.ticket} "
                        f"RL suggests {suggested_dir} vs holding {trade.direction} "
                        f"(conf={decision.get('confidence', 0):.2f}, "
                        f"PnL={current_pnl_r:+.2f}R, "
                        f"held={trade.hold_time_hours:.1f}h)"
                    )
                    self._close_trade_early(trade, "RL_EARLY_EXIT")
                    return

            # Otherwise HOLD — order_manager handles trailing, BE, partial TP

        except Exception as e:
            log.debug(
                f"[LIVE_ENGINE] Mid-trade RL evaluation failed for "
                f"{pair} #{trade.ticket}: {e}"
            )

    def _close_trade_early(self, trade: ActiveTrade, reason: str):
        """
        Close a trade early via MT5.

        Args:
            trade:  ActiveTrade to close.
            reason: Reason string for logging/comment.
        """
        if self.paper_mode:
            log.info(
                f"[LIVE_ENGINE] PAPER CLOSE {trade.pair} #{trade.ticket} "
                f"({reason})"
            )
            trade.closed = True
            return

        try:
            import MetaTrader5 as mt5
            positions = mt5.positions_get(ticket=trade.ticket)
            if positions and len(positions) > 0:
                from execution.order_manager import _close_position
                sym_info = mt5.symbol_info(trade.pair)
                _close_position(positions[0], reason, sym_info)
                trade.closed = True
                log.info(
                    f"[LIVE_ENGINE] Closed {trade.pair} #{trade.ticket} "
                    f"({reason})"
                )
        except Exception as e:
            log.error(
                f"[LIVE_ENGINE] Failed to close {trade.pair} #{trade.ticket}: {e}"
            )

    # ════════════════════════════════════════════════════════════
    #  TRADE OUTCOME RECORDING
    # ════════════════════════════════════════════════════════════

    def _record_closed_trade(self, ticket: int):
        """
        Record a closed trade's outcome to the ExperienceBuffer.

        Detects the outcome from MT5 deal history, computes R-multiple,
        and creates a TradeExperience for the continuous learning loop.

        Args:
            ticket: MT5 position ticket of the closed trade.
        """
        with self._lock:
            trade = self._active_trades.pop(ticket, None)

        if trade is None:
            return

        trade.closed = True

        # ── Get outcome from MT5 deal history ──
        outcome_data = self._get_trade_outcome(ticket, trade.pair)

        profit_pips = outcome_data.get('profit_pips', 0.0)
        profit_usd = outcome_data.get('profit_usd', 0.0)
        exit_price = outcome_data.get('exit_price', trade.entry_price)
        outcome = outcome_data.get('outcome', 'LOSS')

        # ── Compute R-multiple ──
        profit_r = 0.0
        if trade.sl_pips > 0:
            profit_r = profit_pips / trade.sl_pips

        # ── Update consecutive win/loss tracking ──
        with self._lock:
            if outcome in ('WIN', 'WIN_TP'):
                self._consecutive_wins += 1
                self._consecutive_losses = 0
                self._daily_pnl_usd += max(0, profit_usd)
            elif outcome in ('LOSS', 'LOSS_SL'):
                self._consecutive_losses += 1
                self._consecutive_wins = 0
                self._daily_pnl_usd += min(0, profit_usd)
            else:
                # BREAKEVEN or MANUAL
                self._consecutive_wins = 0
                self._consecutive_losses = 0
                self._daily_pnl_usd += profit_usd

            self._weekly_pnl_usd += profit_usd

        # ── Record to ExperienceBuffer ──
        if self._learning_loop is not None:
            try:
                from rpde.experience_buffer import TradeExperience

                now = datetime.now(timezone.utc)
                experience = TradeExperience(
                    trade_id=ticket,
                    pair=trade.pair,
                    direction=trade.direction,
                    entry_time=trade.entry_time,
                    exit_time=now.isoformat(),
                    entry_price=trade.entry_price,
                    exit_price=exit_price,
                    profit_pips=profit_pips,
                    profit_r=round(profit_r, 4),
                    profit_usd=round(profit_usd, 2),
                    outcome=outcome,
                    fusion_confidence=trade.fusion_confidence,
                    fusion_expected_r=trade.fusion_expected_r,
                    signal_agreement='N/A',
                    reversal_warning=False,
                    rl_action=0,
                    rl_action_name=trade.rl_action_name,
                    rl_predicted_value=trade.rl_predicted_value,
                    session=trade.session,
                    spread_at_entry=trade.spread_at_entry,
                    atr_at_entry=trade.atr_at_entry,
                    mae_r=round(
                        trade.max_adverse_pips / trade.sl_pips, 4
                    ) if trade.sl_pips > 0 else 0.0,
                    mfe_r=round(
                        trade.peak_profit_pips / trade.sl_pips, 4
                    ) if trade.sl_pips > 0 else 0.0,
                    hold_time_hours=round(trade.hold_time_hours, 2),
                )

                result = self._learning_loop.record_trade(experience)
                with self._lock:
                    self._stats["outcomes_recorded"] += 1

                log.info(
                    f"[LIVE_ENGINE] Recorded outcome: {trade.pair} #{ticket} "
                    f"{outcome} R={profit_r:+.2f} "
                    f"PnL=${profit_usd:+.2f} "
                    f"(MAE={experience.mae_r:.2f}R, "
                    f"MFE={experience.mfe_r:.2f}R, "
                    f"held={experience.hold_time_hours:.1f}h)"
                )

            except Exception as e:
                log.error(
                    f"[LIVE_ENGINE] Failed to record experience for "
                    f"{trade.pair} #{ticket}: {e}"
                )

        # ── Update safety system position counts ──
        if self._safety:
            self._update_safety_positions()

    def _get_trade_outcome(self, ticket: int, pair: str) -> dict:
        """
        Get trade outcome from MT5 deal history.

        Looks up the closing deal for a position and extracts
        profit/loss, exit price, and outcome classification.

        Args:
            ticket: MT5 position ticket.
            pair:   Currency pair symbol.

        Returns:
            Dict with profit_pips, profit_usd, exit_price, outcome.
        """
        default = {
            'profit_pips': 0.0,
            'profit_usd': 0.0,
            'exit_price': 0.0,
            'outcome': 'LOSS',
        }

        if self.paper_mode:
            # For paper trades, use tracked MFE/MAE
            with self._lock:
                trade = self._active_trades.get(ticket)
            if trade:
                # Simulate outcome based on last known PnL
                pnl = self._get_position_pnl_pips(trade)
                outcome = 'WIN' if pnl > 0 else 'LOSS' if pnl < 0 else 'BREAKEVEN'
                return {
                    'profit_pips': max(0, pnl),
                    'profit_usd': pnl * self._get_pip_value_for_pair(pair) * trade.lot_size,
                    'exit_price': trade.entry_price,
                    'outcome': outcome,
                }
            return default

        try:
            import MetaTrader5 as mt5
            from datetime import timedelta

            now = datetime.now(timezone.utc)
            from_dt = now - timedelta(hours=72)  # Look back 3 days

            deals = mt5.history_deals_get(from_dt, now)
            if not deals:
                return default

            # Find the closing deal for this position
            for deal in deals:
                if deal.position_id == ticket and deal.entry == 1:
                    pip_size = get_pip_size(pair)
                    profit_usd = deal.profit + (deal.swap or 0) + (deal.commission or 0)

                    # Determine outcome
                    if profit_usd > 0:
                        comment = str(deal.comment).upper() if deal.comment else ""
                        outcome = 'WIN_TP' if 'TP' in comment else 'WIN'
                    elif profit_usd < 0:
                        comment = str(deal.comment).upper() if deal.comment else ""
                        outcome = 'LOSS_SL' if 'SL' in comment else 'LOSS'
                    else:
                        outcome = 'BREAKEVEN'

                    return {
                        'profit_usd': profit_usd,
                        'exit_price': float(deal.price),
                        'outcome': outcome,
                        'profit_pips': abs(profit_usd) / max(
                            self._get_pip_value_for_pair(pair) * 0.01, 0.001
                        ),
                    }
        except Exception as e:
            log.debug(f"[LIVE_ENGINE] Deal history lookup failed: {e}")

        return default

    # ════════════════════════════════════════════════════════════
    #  UTILITY HELPERS
    # ════════════════════════════════════════════════════════════

    def _check_cooldown(self, pair: str) -> bool:
        """
        Check if pair is in cooldown. Returns True if OK to trade.

        Args:
            pair: Currency pair string.

        Returns:
            True if cooldown has expired (OK to trade), False if still cooling.
        """
        with self._lock:
            last_trade_time = self._cooldowns.get(pair)

        if last_trade_time is None:
            return True

        elapsed = (datetime.now(timezone.utc) - last_trade_time).total_seconds() / 60
        if elapsed < PATTERN_COOLDOWN_MINUTES:
            log.debug(
                f"[LIVE_ENGINE] {pair} cooldown active "
                f"({elapsed:.0f}/{PATTERN_COOLDOWN_MINUTES} min)"
            )
            return False

        return True

    def _check_daily_reset(self):
        """Reset daily counters if we've crossed midnight UTC."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        week_start = self._get_week_start()

        with self._lock:
            if self._daily_reset_date != today:
                log.info(
                    f"[LIVE_ENGINE] Daily reset: {self._daily_reset_date} -> {today} "
                    f"(trades_yesterday={self._daily_trade_count}, "
                    f"daily_pnl=${self._daily_pnl_usd:+.2f})"
                )
                self._daily_trade_count = 0
                self._daily_pnl_usd = 0.0
                self._daily_reset_date = today

            if self._weekly_start_date != week_start:
                log.info(
                    f"[LIVE_ENGINE] Weekly reset: "
                    f"weekly_pnl=${self._weekly_pnl_usd:+.2f}"
                )
                self._weekly_pnl_usd = 0.0
                self._weekly_start_date = week_start

    def _get_week_start(self) -> str:
        """Get the Monday of the current week as a date string."""
        now = datetime.now(timezone.utc)
        monday = now - timedelta(days=now.weekday())
        return monday.strftime("%Y-%m-%d")

    def _update_safety_positions(self):
        """Update safety guard system with current per-pair position counts."""
        if not self._safety:
            return

        pair_counts: Dict[str, int] = defaultdict(int)
        with self._lock:
            for trade in self._active_trades.values():
                if not trade.closed:
                    pair_counts[trade.pair] += 1

        for pair, count in pair_counts.items():
            self._safety.update_positions(pair, count)

        # Also zero-out pairs that no longer have positions
        all_active_pairs = set(pair_counts.keys())
        with self._lock:
            for pair in self.pairs:
                if pair not in all_active_pairs:
                    self._safety.update_positions(pair, 0)

    # ════════════════════════════════════════════════════════════
    #  LEARNING CYCLE
    # ════════════════════════════════════════════════════════════

    def run_learning_cycle(self) -> dict:
        """
        Check and run scheduled learning cycles for all pairs.

        Calls ContinuousLearningLoop.check_all_schedules() and
        executes any due retrains.

        Returns:
            Dict with schedule check results and any errors.
        """
        if self._learning_loop is None:
            return {"status": "skipped", "reason": "Learning loop not initialized"}

        try:
            schedule = self._learning_loop.check_all_schedules(self.pairs)

            if not schedule.get('any_due', False):
                return {
                    "status": "no_action",
                    "schedule": schedule,
                }

            results = {"status": "executed", "schedule": schedule, "errors": []}

            for pair, pair_schedule in schedule.get('per_pair', {}).items():
                try:
                    self._learning_loop.run_learning_cycle(pair)
                except Exception as e:
                    results["errors"].append(f"{pair}: {e}")

            return results

        except Exception as e:
            log.error(f"[LIVE_ENGINE] Learning cycle failed: {e}")
            return {"status": "error", "error": str(e)}

    # ════════════════════════════════════════════════════════════
    #  STATUS & SHUTDOWN
    # ════════════════════════════════════════════════════════════

    def get_status(self) -> dict:
        """
        Get comprehensive engine status for dashboard.

        Returns:
            Dict with subsystem status, active trades, performance
            stats, and safety system state.
        """
        with self._lock:
            active_count = sum(1 for t in self._active_trades.values() if not t.closed)

        # Gate status
        gate_status = {}
        if self._gate:
            try:
                gate_status = self._gate.get_status()
            except Exception:
                gate_status = {"error": "Failed to get gate status"}

        # Safety status
        safety_status = {}
        if self._safety:
            try:
                safety_status = self._safety.get_summary()
            except Exception:
                safety_status = {"error": "Failed to get safety status"}

        # Learning status
        learning_status = {}
        if self._learning_loop:
            try:
                schedule = self._learning_loop.check_all_schedules(self.pairs)
                learning_status = schedule
            except Exception:
                learning_status = {"error": "Failed to check schedules"}

        # Account state
        account = self._build_account_state()

        return {
            'engine': {
                'initialized': self._initialized,
                'running': self._running,
                'paper_mode': self.paper_mode,
                'pairs': self.pairs,
            },
            'subsystems': {
                'pattern_gate': gate_status,
                'rl_agents': {
                    pair: {
                        'loaded': True,
                        'trained': agent.is_trained,
                        'episodes': agent.training_episodes,
                    }
                    for pair, agent in self._rl_agents.items()
                },
                'safety': safety_status,
                'learning': learning_status,
            },
            'active_trades': {
                'count': active_count,
                'details': [
                    {
                        'ticket': t.ticket,
                        'pair': t.pair,
                        'direction': t.direction,
                        'size_r': t.size_r,
                        'hold_hours': round(t.hold_time_hours, 1),
                        'entry_time': t.entry_time,
                    }
                    for t in self._active_trades.values()
                    if not t.closed
                ],
            },
            'performance': {
                'consecutive_losses': self._consecutive_losses,
                'consecutive_wins': self._consecutive_wins,
                'daily_pnl_usd': round(self._daily_pnl_usd, 2),
                'weekly_pnl_usd': round(self._weekly_pnl_usd, 2),
                'daily_trade_count': self._daily_trade_count,
                'peak_equity': round(self._peak_equity, 2),
            },
            'account': {
                'balance': account.get('balance', 0),
                'equity': account.get('equity', 0),
                'margin_level': round(account.get('margin_level', 0), 1),
                'daily_pnl_pct': round(account.get('daily_pnl_pct', 0), 2),
            },
            'stats': dict(self._stats),
        }

    def shutdown(self):
        """
        Graceful shutdown: close positions, save state, cleanup.

        In paper mode, simply stops the engine and logs state.
        In live mode, optionally closes all RPDE positions.
        """
        log.info("[LIVE_ENGINE] Shutting down...")

        with self._lock:
            self._running = False

        # ── Save experience buffers ──
        if self._learning_loop:
            try:
                for pair, buf in self._learning_loop.get_all_buffers().items():
                    buf.save(force=True)
                log.info("[LIVE_ENGINE] Experience buffers saved")
            except Exception as e:
                log.error(f"[LIVE_ENGINE] Failed to save experience buffers: {e}")

        # ── Optionally close all RPDE positions ──
        if not self.paper_mode:
            with self._lock:
                open_trades = {
                    t: trade for t, trade in self._active_trades.items()
                    if not trade.closed
                }

            if open_trades:
                log.warning(
                    f"[LIVE_ENGINE] {len(open_trades)} RPDE positions still open "
                    f"— closing for safety"
                )
                for ticket, trade in open_trades.items():
                    self._close_trade_early(trade, "ENGINE_SHUTDOWN")

        # ── Save RL agents ──
        for pair, agent in self._rl_agents.items():
            try:
                agent.save()
            except Exception as e:
                log.error(f"[LIVE_ENGINE] Failed to save RL agent for {pair}: {e}")

        # ── Final status log ──
        with self._lock:
            stats = dict(self._stats)

        log.info(
            f"[LIVE_ENGINE] Shutdown complete — "
            f"signals={stats['total_signals']} gate_skips={stats['gate_skips']} "
            f"rl_skips={stats['rl_skips']} safety_blocks={stats['safety_blocks']} "
            f"executed={stats['executed']} paper={stats['paper_executed']} "
            f"outcomes={stats['outcomes_recorded']} "
            f"daily_pnl=${self._daily_pnl_usd:+.2f}"
        )

        self._initialized = False
