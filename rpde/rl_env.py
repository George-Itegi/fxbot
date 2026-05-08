# =============================================================
# rpde/rl_env.py  --  RL Trading Environment (Phase 3)
#
# PURPOSE: Gymnasium-compatible trading environment that wraps
# Phase 2 FusionLayer output in a standard RL state/action/reward
# interface for a PPO agent to train on.
#
# ARCHITECTURE:
#
#   FusionLayer output (Phase 2)
#       │  combined_confidence, combined_expected_r, direction,
#       │  signal_agreement, reversal_warning, weights, ...
#       │
#       ▼
#   ┌──────────────────────────────────────────────────────┐
#   │          TRADING ENVIRONMENT (gym.Env)               │
#   │                                                      │
#   │  Market State (from live data feed)                  │
#   │      │  session, hour, spread, ATR, trends, RSI...  │
#   │      ▼                                              │
#   │  Portfolio State (internal simulator)                │
#   │      │  open positions, equity, drawdown, PnL...    │
#   │      ▼                                              │
#   │                                                      │
#   │  Observation (28-dim float32 vector)                 │
#   │  Action Space (7 discrete actions)                   │
#   │  Reward (R-multiple based + shaping)                 │
#   │                                                      │
#   │  Embedded Trade Simulator:                           │
#   │    - Tracks open trades with SL/TP levels            │
#   │    - Simulates PnL from price movements              │
#   │    - Records trade history per episode               │
#   └──────────────────────────────────────────────────────┘
#
# OBSERVATION SPACE (28 features):
#   Fusion Signal  (8): confidence, expected_r, direction_enc,
#                        agreement_enc, reversal, xgb_w, tft_w,
#                        tft_contribution
#   Market State  (12): session, hour, day, spread_ratio,
#                        atr_pctile, vol_regime, momentum_vel,
#                        rsi, m5/m15/h1/h4 trend
#   Portfolio     (8):  positions_ratio, daily_pnl_ratio,
#                        weekly_pnl, equity_ratio, unrealized_pnl,
#                        drawdown, cons_losses, hrs_since_trade
#
# ACTION SPACE (7 discrete):
#   0: SKIP
#   1: ENTER_BUY_05R   2: ENTER_BUY_1R   3: ENTER_BUY_15R
#   4: ENTER_SELL_05R  5: ENTER_SELL_1R  6: ENTER_SELL_15R
#
# REWARD FUNCTION:
#   - Trade close:  profit_r * scaling
#   - Bad entry:    extra penalty if loss > 1R
#   - Good entry:   extra bonus if profit > 1R
#   - Step shaping: -0.001 per 1% drawdown increase
#   - Skip:         0
#
# USAGE:
#   env = TradingEnv("EURJPY")
#   env.receive_signal(fusion_result, market_state, portfolio_state)
#   obs, reward, terminated, truncated, info = env.step(action)
#
# FALLBACK:
#   If gymnasium is not installed, the module will fall back to
#   a lightweight stub that raises ImportError with a helpful
#   message when TradingEnv is instantiated.
# =============================================================

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Gymnasium with graceful fallback ─────────────────────────
try:
    import gymnasium as gym
    from gymnasium import spaces

    _GYMNASIUM_AVAILABLE = True
except ImportError:
    try:
        import gym as gym  # type: ignore[no-redef]
        from gym import spaces  # type: ignore[no-redef]

        _GYMNASIUM_AVAILABLE = True
    except ImportError:
        _GYMNASIUM_AVAILABLE = False
        gym = None  # type: ignore[assignment]
        spaces = None  # type: ignore[assignment]

from core.logger import get_logger
from rpde.config import (
    RL_ACTION_DIM,
    RL_BAD_ENTRY_PENALTY,
    RL_CONSECUTIVE_LOSSES as RL_MAX_CONSECUTIVE_LOSSES,
    RL_DRAWDOWN_SHAPING,
    RL_GOOD_ENTRY_BONUS,
    RL_MAX_DAILY_LOSS_PCT,
    RL_MAX_POSITIONS,
    RL_MAX_STEPS_PER_EPISODE,
    RL_OBS_DIM,
    RL_REWARD_SCALING,
    RL_SIZE_MAP,
)

log = get_logger(__name__)


# ════════════════════════════════════════════════════════════════
#  ACTION ENUM / MAPPINGS
# ════════════════════════════════════════════════════════════════

class Action:
    """
    Discrete action definitions for the trading environment.

    Each action maps to a specific trade decision:
      - 0: Do nothing (skip this signal)
      - 1-3: Enter BUY with 0.5R, 1.0R, or 1.5R size
      - 4-6: Enter SELL with 0.5R, 1.0R, or 1.5R size

    Size determines the stop-loss width via RL_SIZE_MAP in config,
    which implicitly sets the take-profit target (typically 2:1 R:R).
    """

    SKIP = 0
    ENTER_BUY_05R = 1
    ENTER_BUY_1R = 2
    ENTER_BUY_15R = 3
    ENTER_SELL_05R = 4
    ENTER_SELL_1R = 5
    ENTER_SELL_15R = 6

    # Human-readable labels for logging / info dicts
    NAMES: Dict[int, str] = {
        0: "SKIP",
        1: "ENTER_BUY_0.5R",
        2: "ENTER_BUY_1.0R",
        3: "ENTER_BUY_1.5R",
        4: "ENTER_SELL_0.5R",
        5: "ENTER_SELL_1.0R",
        6: "ENTER_SELL_1.5R",
    }

    @classmethod
    def name(cls, action: int) -> str:
        """Return human-readable name for an action index."""
        return cls.NAMES.get(action, f"UNKNOWN({action})")

    @classmethod
    def size_r(cls, action: int) -> float:
        """
        Return the R-size for a trade entry action.

        Args:
            action: Action index (1-6 for entries).

        Returns:
            Position size in R-multiples (0.5, 1.0, or 1.5).
            Returns 0.0 for SKIP.
        """
        _size_map = {
            cls.ENTER_BUY_05R: 0.5,
            cls.ENTER_BUY_1R: 1.0,
            cls.ENTER_BUY_15R: 1.5,
            cls.ENTER_SELL_05R: 0.5,
            cls.ENTER_SELL_1R: 1.0,
            cls.ENTER_SELL_15R: 1.5,
        }
        return _size_map.get(action, 0.0)

    @classmethod
    def direction(cls, action: int) -> Optional[str]:
        """
        Return trade direction for an entry action.

        Returns:
            "BUY" or "SELL" for entry actions, None for SKIP.
        """
        if action in (cls.ENTER_BUY_05R, cls.ENTER_BUY_1R, cls.ENTER_BUY_15R):
            return "BUY"
        if action in (cls.ENTER_SELL_05R, cls.ENTER_SELL_1R, cls.ENTER_SELL_15R):
            return "SELL"
        return None

    @classmethod
    def is_entry(cls, action: int) -> bool:
        """Return True if the action is a trade entry (not SKIP)."""
        return action != cls.SKIP


# ════════════════════════════════════════════════════════════════
#  TRADE SIMULATOR (embedded)
# ════════════════════════════════════════════════════════════════

@dataclass
class SimulatedTrade:
    """
    Represents a single open trade in the internal simulator.

    The simulator tracks entry details and computes unrealized PnL
    as market prices update. When a trade closes (via close_trade or
    SL/TP hit), profit_r is recorded in the trade history.
    """

    trade_id: int
    direction: str          # "BUY" or "SELL"
    size_r: float           # Position size in R-multiples
    entry_price: float      # Price at entry
    sl_price: float         # Stop-loss price level
    tp_price: float         # Take-profit price level
    sl_r: float             # Stop-loss in R-multiples
    tp_r: float             # Take-profit in R-multiples
    entry_time: float       # Unix timestamp of entry
    closed: bool = False
    profit_r: float = 0.0   # R-multiple result (set on close)
    close_price: float = 0.0
    close_time: float = 0.0

    def compute_pnl_r(self, current_price: float) -> float:
        """
        Compute unrealized profit in R-multiples at current price.

        Args:
            current_price: Current market price.

        Returns:
            Unrealized R-multiple. Negative = losing trade.
        """
        if self.closed:
            return self.profit_r

        if self.direction == "BUY":
            # For BUY: profit = (current - entry) / (entry - sl) * size_r
            risk_distance = self.entry_price - self.sl_price
            if risk_distance <= 0:
                return 0.0
            pip_move = current_price - self.entry_price
            return (pip_move / risk_distance) * self.size_r
        else:
            # For SELL: profit = (entry - current) / (sl - entry) * size_r
            risk_distance = self.sl_price - self.entry_price
            if risk_distance <= 0:
                return 0.0
            pip_move = self.entry_price - current_price
            return (pip_move / risk_distance) * self.size_r

    def check_sl_tp(self, current_price: float) -> Optional[Tuple[str, float]]:
        """
        Check if price has hit stop-loss or take-profit.

        Args:
            current_price: Current market price.

        Returns:
            Tuple of ("sl" or "tp", profit_r) if hit, None otherwise.
        """
        if self.closed:
            return None

        pnl_r = self.compute_pnl_r(current_price)

        if self.direction == "BUY":
            if current_price <= self.sl_price:
                return ("sl", -self.sl_r * self.size_r / self.size_r)
            if current_price >= self.tp_price:
                return ("tp", self.tp_r)
        else:
            if current_price >= self.sl_price:
                return ("sl", -self.sl_r)
            if current_price <= self.tp_price:
                return ("tp", self.tp_r)

        return None


# ════════════════════════════════════════════════════════════════
#  MAIN ENVIRONMENT CLASS
# ════════════════════════════════════════════════════════════════

class TradingEnv:
    """
    Gymnasium-compatible trading environment for RL-based trade decisions.

    Wraps Phase 2 FusionLayer output + market state + portfolio state into
    a standard RL interface (obs, action, reward) suitable for PPO or other
    policy-gradient algorithms.

    Episode = one trading day (configurable via max_steps_per_episode).
    Each step = one signal received from the fusion layer.

    The environment maintains an internal trade simulator that tracks open
    positions, computes unrealized PnL, and records trade outcomes.

    Args:
        pair: Currency pair (e.g. "EURJPY"). Used for logging only.
        reward_scaling: Multiplier applied to R-multiple rewards.
        max_steps_per_episode: Maximum signals per episode (trading day).
        max_daily_loss_pct: Hard daily loss limit as pct of initial equity.
        max_positions: Maximum concurrent open positions.
        initial_equity: Starting equity for portfolio state normalization.

    Example::

        env = TradingEnv("EURJPY")
        obs, info = env.reset()

        # External system pushes a new fused signal
        env.receive_signal(fusion_result, market_state, portfolio_state)

        # Agent decides
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
    """

    # ── Class-level flag for gymnasium availability ──────────
    available: bool = _GYMNASIUM_AVAILABLE

    def __init__(
        self,
        pair: str,
        reward_scaling: float = RL_REWARD_SCALING,
        max_steps_per_episode: int = RL_MAX_STEPS_PER_EPISODE,
        max_daily_loss_pct: float = RL_MAX_DAILY_LOSS_PCT,
        max_positions: int = RL_MAX_POSITIONS,
        initial_equity: float = 10000.0,
    ) -> None:
        # ── Guard: gymnasium required ──
        if not _GYMNASIUM_AVAILABLE:
            raise ImportError(
                "gymnasium (or gym) is required for TradingEnv. "
                "Install it with: pip install gymnasium"
            )

        super().__init__()  # type: ignore[misc]

        self.pair = pair.upper()
        self.reward_scaling = reward_scaling
        self.max_steps_per_episode = max_steps_per_episode
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_positions = max_positions
        self.initial_equity = initial_equity

        # ── Gymnasium spaces ──
        # Observation: 28-dim float32 vector, each feature normalized to [-1, 1]
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(RL_OBS_DIM,),
            dtype=np.float32,
        )
        # Action: 7 discrete choices
        self.action_space = spaces.Discrete(RL_ACTION_DIM)

        # ── Internal state (set properly in reset) ──
        self._current_step: int = 0
        self._pending_fusion: Dict[str, Any] = {}
        self._pending_market: Dict[str, Any] = {}
        self._pending_portfolio: Dict[str, Any] = {}

        # Portfolio simulator state
        self._total_equity: float = initial_equity
        self._daily_pnl: float = 0.0
        self._weekly_pnl: float = 0.0
        self._unrealized_pnl: float = 0.0
        self._current_drawdown: float = 0.0
        self._peak_equity: float = initial_equity
        self._consecutive_losses: int = 0
        self._last_trade_time: float = 0.0

        # Trade simulator
        self._open_trades: List[SimulatedTrade] = []
        self._closed_trades: List[SimulatedTrade] = []
        self._next_trade_id: int = 1

        # Previous drawdown for step-level shaping reward
        self._prev_drawdown: float = 0.0

        # Episode statistics
        self._episode_total_reward: float = 0.0
        self._episode_trades: int = 0
        self._episode_skips: int = 0

        # Current ATR value (for computing SL/TP prices)
        self._current_atr: float = 0.0
        self._current_price: float = 0.0

        log.info(
            f"[RL-ENV] {self.pair}: initialized | "
            f"obs_dim={RL_OBS_DIM} action_dim={RL_ACTION_DIM} | "
            f"max_steps={max_steps_per_episode} "
            f"max_loss={max_daily_loss_pct}% "
            f"max_pos={max_positions} "
            f"scaling={reward_scaling}"
        )

    # ──────────────────────────────────────────────────────────
    #  GYMNASIUM API
    # ──────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to start a new episode (trading day).

        Args:
            seed: Random seed for reproducibility.
            options: Optional dict. Supported key "equity" to override
                     initial equity.

        Returns:
            Tuple of (observation, info).
        """
        if seed is not None:
            np.random.seed(seed)

        # Reset step counter
        self._current_step = 0

        # Clear pending signals
        self._pending_fusion = {}
        self._pending_market = {}
        self._pending_portfolio = {}

        # Reset portfolio state
        override_equity = (
            options.get("equity", self.initial_equity)
            if options else self.initial_equity
        )
        self.initial_equity = override_equity
        self._total_equity = override_equity
        self._daily_pnl = 0.0
        self._weekly_pnl = 0.0
        self._unrealized_pnl = 0.0
        self._current_drawdown = 0.0
        self._peak_equity = override_equity
        self._consecutive_losses = 0
        self._last_trade_time = 0.0
        self._prev_drawdown = 0.0

        # Reset trade simulator
        self._open_trades = []
        self._closed_trades = []
        self._next_trade_id = 1

        # Reset episode stats
        self._episode_total_reward = 0.0
        self._episode_trades = 0
        self._episode_skips = 0

        # Reset market data
        self._current_atr = 0.0
        self._current_price = 0.0

        # Return zero-observation with empty info
        obs = np.zeros(RL_OBS_DIM, dtype=np.float32)
        info = {
            "pair": self.pair,
            "episode_start": datetime.now(timezone.utc).isoformat(),
            "reason": "reset",
        }

        log.debug(f"[RL-ENV] {self.pair}: episode reset")
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.

        The agent observes the current state (set via receive_signal),
        chooses an action, and the environment returns the next state,
        reward, and termination flags.

        Args:
            action: Integer action index (0-6). See Action class.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).

            - observation: np.ndarray of shape (28,), dtype float32
            - reward: float — the reward for this step
            - terminated: bool — episode ended (loss limit hit, etc.)
            - truncated: bool — episode ended (max steps reached)
            - info: dict — extra info (trade details, episode stats)
        """
        # ── Validate action ──
        if not (0 <= action < RL_ACTION_DIM):
            log.warning(
                f"[RL-ENV] {self.pair}: invalid action {action}, "
                f"clamping to SKIP"
            )
            action = Action.SKIP

        action_name = Action.name(action)
        reward = 0.0
        trade_outcome_r: Optional[float] = None
        info: Dict[str, Any] = {
            "action": action,
            "action_name": action_name,
            "step": self._current_step,
            "pair": self.pair,
        }

        # ── Process action ──
        if Action.is_entry(action):
            # Check risk constraints before allowing entry
            can_trade = self._check_risk_gates(action)

            if can_trade:
                trade = self._open_trade(action)
                if trade is not None:
                    info["trade_opened"] = {
                        "trade_id": trade.trade_id,
                        "direction": trade.direction,
                        "size_r": trade.size_r,
                        "entry_price": trade.entry_price,
                        "sl_price": trade.sl_price,
                        "tp_price": trade.tp_price,
                    }
                    self._episode_trades += 1
                    self._last_trade_time = time.time()
            else:
                # Risk gate blocked the entry — treat as skip
                info["blocked"] = True
                info["block_reason"] = "risk_gate"
                action = Action.SKIP
                self._episode_skips += 1
        else:
            self._episode_skips += 1

        # ── Update open trades with current price ──
        self._update_open_trades()

        # ── Check if any trades were closed by SL/TP ──
        closed_this_step = self._check_closed_trades()
        if closed_this_step:
            for trade in closed_this_step:
                trade_outcome_r = trade.profit_r
                info.setdefault("trades_closed", []).append({
                    "trade_id": trade.trade_id,
                    "direction": trade.direction,
                    "profit_r": round(trade.profit_r, 4),
                    "close_price": trade.close_price,
                })

        # ── Compute reward ──
        reward = self._compute_reward(action, trade_outcome_r)

        # ── Update portfolio metrics ──
        self._update_portfolio_state()

        # ── Step-level drawdown shaping ──
        dd_change = self._current_drawdown - self._prev_drawdown
        if dd_change > 0:
            reward += RL_DRAWDOWN_SHAPING * dd_change * 100.0
        self._prev_drawdown = self._current_drawdown

        self._episode_total_reward += reward

        # ── Build observation for next step ──
        obs = self._build_observation()

        # ── Check episode termination ──
        terminated, truncation_reason = self._is_episode_over()

        # ── Advance step counter ──
        self._current_step += 1

        # Truncation: max steps exceeded
        truncated = False
        if not terminated and self._current_step >= self.max_steps_per_episode:
            truncated = True
            truncation_reason = "max_steps"

        # ── Populate info dict ──
        info["reward"] = round(reward, 6)
        info["total_reward"] = round(self._episode_total_reward, 6)
        info["open_positions"] = len(self._open_trades)
        info["equity"] = round(self._total_equity, 2)
        info["drawdown_pct"] = round(self._current_drawdown * 100, 2)
        info["episode_trades"] = self._episode_trades
        info["episode_skips"] = self._episode_skips

        if terminated:
            info["termination_reason"] = truncation_reason
            self._log_episode_summary(truncation_reason)
        elif truncated:
            info["truncation_reason"] = truncation_reason
            self._log_episode_summary(truncation_reason)

        return obs, reward, terminated, truncated, info

    # ──────────────────────────────────────────────────────────
    #  EXTERNAL INTERFACE
    # ──────────────────────────────────────────────────────────

    def receive_signal(
        self,
        fusion_result: Dict[str, Any],
        market_state: Dict[str, Any],
        portfolio_state: Dict[str, Any],
    ) -> None:
        """
        Push the latest signal data into the environment.

        Must be called before step() to update the observation.

        Args:
            fusion_result: Dict from FusionLayer.fuse() containing:
                - combined_confidence (float)
                - combined_expected_r (float)
                - direction (str | None): "BUY", "SELL", or None
                - signal_agreement (str): "ALL_AGREE", "XGB_TFT_AGREE",
                  "PARTIAL", "DISAGREE"
                - reversal_warning (bool)
                - weights (dict): {"xgb_weight": float, "tft_weight": float}
                - tft_contribution (float)
            market_state: Dict with current market conditions:
                - session (str): "London", "NewYork", "Asian", "Off"
                - hour_of_day (int): 0-23
                - day_of_week (int): 0-4 (Mon-Fri)
                - current_spread (float): current spread in pips
                - avg_spread (float): average spread for normalization
                - atr_percentile (float): 0-1
                - volatility_regime (str): "ranging", "normal", "quiet"
                - momentum_velocity (float)
                - rsi (float): 0-100
                - m5_trend, m15_trend, h1_trend, h4_trend (str): "up", "down", "flat"
                - current_price (float): latest price
                - atr (float): current ATR value for SL/TP computation
            portfolio_state: Dict with current portfolio status:
                - open_positions (int)
                - daily_pnl (float): realized daily PnL in account currency
                - weekly_pnl (float): realized weekly PnL
                - total_equity (float): current total equity
                - unrealized_pnl (float): unrealized PnL across open trades
                - consecutive_losses (int)
                - hours_since_last_trade (float)
        """
        self._pending_fusion = fusion_result or {}
        self._pending_market = market_state or {}
        self._pending_portfolio = portfolio_state or {}

        # Extract ATR and price for SL/TP computation
        self._current_price = float(market_state.get("current_price", 0.0))
        self._current_atr = float(market_state.get("atr", 0.0))

        log.debug(
            f"[RL-ENV] {self.pair}: signal received | "
            f"conf={fusion_result.get('combined_confidence', 0):.3f} "
            f"R={fusion_result.get('combined_expected_r', 0):.2f} "
            f"dir={fusion_result.get('direction')}"
        )

    def close_trade(self, trade_id: int, profit_r: float) -> None:
        """
        Manually close a trade with a given profit R-multiple.

        Used by the external system when a real trade closes, or during
        backtesting when the backtester determines the outcome.

        Args:
            trade_id: ID of the trade to close.
            profit_r: R-multiple result of the trade.
        """
        for trade in self._open_trades:
            if trade.trade_id == trade_id and not trade.closed:
                trade.closed = True
                trade.profit_r = profit_r
                trade.close_time = time.time()

                # Update portfolio PnL
                self._daily_pnl += profit_r
                self._weekly_pnl += profit_r

                # Track consecutive losses
                if profit_r < 0:
                    self._consecutive_losses += 1
                else:
                    self._consecutive_losses = 0

                # Move to closed list
                self._closed_trades.append(trade)
                self._open_trades.remove(trade)

                log.debug(
                    f"[RL-ENV] {self.pair}: trade #{trade_id} closed | "
                    f"profit_r={profit_r:.3f} dir={trade.direction}"
                )
                return

        log.warning(
            f"[RL-ENV] {self.pair}: trade #{trade_id} not found "
            f"or already closed"
        )

    def get_state(self) -> np.ndarray:
        """
        Get the current observation vector without advancing the env.

        Useful for inference in live trading where you want to peek at
        the state before deciding whether to call step().

        Returns:
            np.ndarray of shape (28,), dtype float32.
        """
        return self._build_observation()

    # ──────────────────────────────────────────────────────────
    #  TRADE SIMULATION
    # ──────────────────────────────────────────────────────────

    def _open_trade(self, action: int) -> Optional[SimulatedTrade]:
        """
        Open a simulated trade based on the chosen action.

        Computes SL and TP prices using the current ATR and the
        size-to-R mapping from config (RL_SIZE_MAP).

        Args:
            action: Entry action (1-6).

        Returns:
            SimulatedTrade if opened successfully, None if blocked.
        """
        direction = Action.direction(action)
        size_r = Action.size_r(action)

        if direction is None or size_r == 0.0:
            return None

        # Look up SL/TP R-multiples from config
        sl_tp = RL_SIZE_MAP.get(size_r)
        if sl_tp is None:
            log.warning(
                f"[RL-ENV] {self.pair}: no SL/TP mapping for size {size_r}"
            )
            return None

        sl_r = sl_tp["sl_r"]
        tp_r = sl_tp["tp_r"]

        # Compute SL/TP prices from ATR
        # If ATR is not available, use a fixed pip estimate (fallback)
        if self._current_atr > 0:
            # SL distance in price = ATR * (sl_r / 1.0)
            # Normalize: 1R = 1 ATR as the base unit
            sl_distance = self._current_atr * sl_r
            tp_distance = self._current_atr * tp_r
        else:
            # Fallback: assume 10 pips per R (conservative)
            pip_size = 0.01 if "JPY" in self.pair else 0.0001
            sl_distance = pip_size * 10.0 * sl_r
            tp_distance = pip_size * 10.0 * tp_r

        price = self._current_price
        if price <= 0:
            log.warning(
                f"[RL-ENV] {self.pair}: cannot open trade, "
                f"invalid price {price}"
            )
            return None

        # Compute SL and TP price levels
        if direction == "BUY":
            sl_price = price - sl_distance
            tp_price = price + tp_distance
        else:
            sl_price = price + sl_distance
            tp_price = price - tp_distance

        trade = SimulatedTrade(
            trade_id=self._next_trade_id,
            direction=direction,
            size_r=size_r,
            entry_price=price,
            sl_price=sl_price,
            tp_price=tp_price,
            sl_r=sl_r,
            tp_r=tp_r,
            entry_time=time.time(),
        )
        self._next_trade_id += 1
        self._open_trades.append(trade)

        log.debug(
            f"[RL-ENV] {self.pair}: opened trade #{trade.trade_id} | "
            f"{direction} {size_r}R @ {price:.5f} | "
            f"SL={sl_price:.5f} TP={tp_price:.5f} | "
            f"sl_r={sl_r} tp_r={tp_r}"
        )

        return trade

    def _check_risk_gates(self, action: int) -> bool:
        """
        Check risk constraints before allowing a trade entry.

        Enforces:
          - Max concurrent positions
          - Max consecutive losses
          - Daily loss limit

        Args:
            action: The entry action being attempted.

        Returns:
            True if the trade is allowed, False if blocked.
        """
        # Gate 1: Max positions
        if len(self._open_trades) >= self.max_positions:
            log.debug(
                f"[RL-ENV] {self.pair}: blocked — max positions "
                f"({len(self._open_trades)}/{self.max_positions})"
            )
            return False

        # Gate 2: Consecutive loss protection
        if self._consecutive_losses >= RL_MAX_CONSECUTIVE_LOSSES:
            log.debug(
                f"[RL-ENV] {self.pair}: blocked — consecutive losses "
                f"({self._consecutive_losses}/{RL_MAX_CONSECUTIVE_LOSSES})"
            )
            return False

        # Gate 3: Daily loss limit
        max_loss = self.initial_equity * (self.max_daily_loss_pct / 100.0)
        if self._daily_pnl < -max_loss:
            log.debug(
                f"[RL-ENV] {self.pair}: blocked — daily loss limit "
                f"({self._daily_pnl:.2f} < {-max_loss:.2f})"
            )
            return False

        return True

    def _update_open_trades(self) -> None:
        """
        Update all open trades with the current price.

        Checks each trade for SL/TP hits and closes any that are
        triggered. Updates unrealized PnL for remaining trades.
        """
        trades_to_close: List[SimulatedTrade] = []

        for trade in self._open_trades:
            if trade.closed:
                continue

            result = trade.check_sl_tp(self._current_price)
            if result is not None:
                reason, profit_r = result
                trade.closed = True
                trade.profit_r = profit_r
                trade.close_price = self._current_price
                trade.close_time = time.time()

                # Update portfolio PnL
                self._daily_pnl += profit_r
                self._weekly_pnl += profit_r

                # Track consecutive losses
                if profit_r < 0:
                    self._consecutive_losses += 1
                else:
                    self._consecutive_losses = 0

                trades_to_close.append(trade)

                log.debug(
                    f"[RL-ENV] {self.pair}: trade #{trade.trade_id} "
                    f"{reason} | profit_r={profit_r:.3f}"
                )

        # Move closed trades
        for trade in trades_to_close:
            self._open_trades.remove(trade)
            self._closed_trades.append(trade)

        # Update unrealized PnL from remaining open trades
        self._unrealized_pnl = sum(
            t.compute_pnl_r(self._current_price) for t in self._open_trades
        )

    def _check_closed_trades(self) -> List[SimulatedTrade]:
        """
        Return trades that were closed during the current step.

        Used by step() to compute rewards for closed trades.

        Returns:
            List of SimulatedTrade objects closed this step.
        """
        # Closed trades are already in self._closed_trades.
        # We return only the ones closed since last step check.
        # This is handled by tracking the length of closed_trades.
        return [
            t for t in self._closed_trades
            if t.close_time > 0
            and (time.time() - t.close_time) < 1.0  # closed within last second
        ]

    # ──────────────────────────────────────────────────────────
    #  OBSERVATION BUILDER
    # ──────────────────────────────────────────────────────────

    def _build_observation(self) -> np.ndarray:
        """
        Build the 28-dimensional observation vector.

        Combines fusion signal (8), market state (12), and portfolio
        state (8) into a single float32 vector with all features
        normalized to approximately [-1, 1].

        Returns:
            np.ndarray of shape (28,), dtype float32.
        """
        obs = np.zeros(RL_OBS_DIM, dtype=np.float32)

        fusion = self._pending_fusion
        market = self._pending_market
        portfolio = self._pending_portfolio

        # ═══════════════════════════════════════════════════════
        # GROUP 1: FUSION SIGNAL (indices 0-7, 8 features)
        # ═══════════════════════════════════════════════════════

        # [0] combined_confidence: already in [0, 1]
        obs[0] = self._safe_float(fusion.get("combined_confidence"), 0.0)

        # [1] combined_expected_r: clamp to [-2, 3] then normalize to [-1, 1]
        expected_r = self._safe_float(fusion.get("combined_expected_r"), 0.0)
        obs[1] = np.clip(expected_r / 3.0, -1.0, 1.0)

        # [2] direction_encoded: BUY=1, SELL=-1, None=0
        direction = fusion.get("direction")
        if direction == "BUY":
            obs[2] = 1.0
        elif direction == "SELL":
            obs[2] = -1.0
        else:
            obs[2] = 0.0

        # [3] signal_agreement_encoded
        agreement = fusion.get("signal_agreement", "DISAGREE")
        _agreement_map = {
            "ALL_AGREE": 1.0,
            "XGB_TFT_AGREE": 0.67,
            "PARTIAL": 0.33,
            "DISAGREE": 0.0,
        }
        obs[3] = _agreement_map.get(str(agreement), 0.0)

        # [4] reversal_warning: 0 or 1
        reversal = fusion.get("reversal_warning", False)
        obs[4] = 1.0 if reversal else 0.0

        # [5] xgb_weight: already in [0, 1]
        weights = fusion.get("weights", {})
        obs[5] = self._safe_float(weights.get("xgb_weight"), 0.55)

        # [6] tft_weight: already in [0, 1]
        obs[6] = self._safe_float(weights.get("tft_weight"), 0.45)

        # [7] tft_contribution: already in [0, 1]
        obs[7] = self._safe_float(fusion.get("tft_contribution"), 0.0)

        # ═══════════════════════════════════════════════════════
        # GROUP 2: MARKET STATE (indices 8-19, 12 features)
        # ═══════════════════════════════════════════════════════

        # [8] session_encoded: London=0.25, NewYork=0.5, Asian=0.75, Off=1.0
        session = market.get("session", "Off")
        _session_map = {
            "London": 0.25,
            "NewYork": 0.5,
            "Asian": 0.75,
            "Off": 1.0,
            "Sydney": 0.875,  # between Asian and Off
        }
        obs[8] = _session_map.get(str(session), 1.0)

        # [9] hour_of_day / 24: normalize to [0, 1]
        hour = self._safe_float(market.get("hour_of_day"), 0.0)
        obs[9] = hour / 24.0

        # [10] day_of_week / 5: normalize to [0, 1] (0=Mon, 4=Fri)
        day = self._safe_float(market.get("day_of_week"), 0.0)
        obs[10] = day / 5.0

        # [11] spread_ratio: current_spread / avg_spread, clamp to [0, 3]
        current_spread = self._safe_float(market.get("current_spread"), 1.0)
        avg_spread = self._safe_float(market.get("avg_spread"), 1.0)
        spread_ratio = current_spread / max(avg_spread, 0.1)
        obs[11] = np.clip(spread_ratio / 3.0, 0.0, 1.0)

        # [12] atr_percentile: already in [0, 1]
        obs[12] = np.clip(
            self._safe_float(market.get("atr_percentile"), 0.5), 0.0, 1.0
        )

        # [13] volatility_regime: ranging=0, normal=0.5, quiet=1.0
        regime = market.get("volatility_regime", "normal")
        _regime_map = {
            "ranging": 0.0,
            "normal": 0.5,
            "quiet": 1.0,
            "volatile": -0.5,
        }
        obs[13] = _regime_map.get(str(regime), 0.5)

        # [14] momentum_velocity: clamp to [-2, 2], then [-1, 1]
        momentum = self._safe_float(market.get("momentum_velocity"), 0.0)
        obs[14] = np.clip(momentum / 2.0, -1.0, 1.0)

        # [15] rsi: normalize from [0, 100] to [0, 1]
        rsi = self._safe_float(market.get("rsi"), 50.0)
        obs[15] = np.clip(rsi / 100.0, 0.0, 1.0)

        # [16-19] Multi-timeframe trends: up=1, down=-1, flat=0
        for i, tf_key in enumerate(["m5_trend", "m15_trend", "h1_trend", "h4_trend"]):
            trend = market.get(tf_key, "flat")
            _trend_map = {"up": 1.0, "down": -1.0, "flat": 0.0}
            obs[16 + i] = _trend_map.get(str(trend), 0.0)

        # ═══════════════════════════════════════════════════════
        # GROUP 3: PORTFOLIO STATE (indices 20-27, 8 features)
        # ═══════════════════════════════════════════════════════

        # [20] open_positions / max_positions: [0, 1]
        open_pos = self._safe_int(portfolio.get("open_positions"), len(self._open_trades))
        obs[20] = open_pos / max(self.max_positions, 1)

        # [21] daily_pnl / max_daily_loss: [-1, 1]
        # Positive PnL caps at 1.0; negative PnL maps to [-1, 0]
        daily_pnl = self._safe_float(portfolio.get("daily_pnl"), self._daily_pnl)
        max_loss = self.initial_equity * (self.max_daily_loss_pct / 100.0)
        if max_loss > 0:
            pnl_ratio = daily_pnl / max_loss
        else:
            pnl_ratio = 0.0
        obs[21] = np.clip(pnl_ratio, -1.0, 1.0)

        # [22] weekly_pnl: normalize to [-1, 1] by initial equity
        weekly_pnl = self._safe_float(portfolio.get("weekly_pnl"), self._weekly_pnl)
        obs[22] = np.clip(
            weekly_pnl / max(self.initial_equity, 1.0), -1.0, 1.0
        )

        # [23] total_equity / initial_equity: normalize (0.5 to 1.5 → -1 to 1)
        equity = self._safe_float(
            portfolio.get("total_equity"), self._total_equity
        )
        equity_ratio = equity / max(self.initial_equity, 1.0)
        # Map [0.5, 1.5] to [-1, 1]
        obs[23] = np.clip((equity_ratio - 1.0) * 2.0, -1.0, 1.0)

        # [24] unrealized_pnl: normalize by initial equity
        unrealized = self._safe_float(
            portfolio.get("unrealized_pnl"), self._unrealized_pnl
        )
        obs[24] = np.clip(
            unrealized / max(self.initial_equity, 1.0), -1.0, 1.0
        )

        # [25] current_drawdown: [0, 1]
        obs[25] = np.clip(self._current_drawdown, 0.0, 1.0)

        # [26] consecutive_losses / max_consecutive: [0, 1]
        cons_losses = self._safe_int(
            portfolio.get("consecutive_losses"), self._consecutive_losses
        )
        obs[26] = cons_losses / max(RL_MAX_CONSECUTIVE_LOSSES, 1)

        # [27] hours_since_last_trade: normalize, cap at 8 hours
        hrs_since = self._safe_float(
            portfolio.get("hours_since_last_trade"), 0.0
        )
        # If no trade yet, compute from _last_trade_time
        if hrs_since == 0.0 and self._last_trade_time > 0:
            hrs_since = (time.time() - self._last_trade_time) / 3600.0
        obs[27] = np.clip(hrs_since / 8.0, 0.0, 1.0)

        return obs

    # ──────────────────────────────────────────────────────────
    #  REWARD FUNCTION
    # ──────────────────────────────────────────────────────────

    def _compute_reward(
        self, action: int, trade_outcome: Optional[float]
    ) -> float:
        """
        Compute the reward for the current step.

        Reward components:
          1. Trade outcome: profit_r * reward_scaling (on trade close)
          2. Bad entry penalty: RL_BAD_ENTRY_PENALTY if loss > 1R
          3. Good entry bonus: RL_GOOD_ENTRY_BONUS if profit > 1R
          4. Skip reward: 0 (neutral — agent learns to skip bad signals)

        Note: Drawdown shaping is handled separately in step() so it
        applies whether the agent traded or skipped.

        Args:
            action: The action taken this step.
            trade_outcome: R-multiple if a trade closed, None otherwise.

        Returns:
            float reward value.
        """
        reward = 0.0

        if trade_outcome is not None:
            # Component 1: Scaled trade outcome
            reward += trade_outcome * self.reward_scaling

            # Component 2: Bad entry penalty (loss > 1R)
            if trade_outcome < -1.0:
                reward += RL_BAD_ENTRY_PENALTY
                log.debug(
                    f"[RL-ENV] {self.pair}: bad entry penalty "
                    f"({trade_outcome:.2f}R loss)"
                )

            # Component 3: Good entry bonus (profit > 1R)
            elif trade_outcome > 1.0:
                reward += RL_GOOD_ENTRY_BONUS
                log.debug(
                    f"[RL-ENV] {self.pair}: good entry bonus "
                    f"({trade_outcome:.2f}R profit)"
                )

        # Component 4: Skip = 0 (implicit, no penalty for skipping)

        return reward

    # ──────────────────────────────────────────────────────────
    #  PORTFOLIO STATE UPDATE
    # ──────────────────────────────────────────────────────────

    def _update_portfolio_state(self) -> None:
        """
        Recalculate portfolio metrics after trade updates.

        Updates:
          - Total equity (initial + realized PnL + unrealized PnL)
          - Peak equity (for drawdown calculation)
          - Current drawdown (peak-to-trough)
        """
        # Sync from external portfolio state if available
        portfolio = self._pending_portfolio
        if portfolio:
            self._daily_pnl = self._safe_float(
                portfolio.get("daily_pnl"), self._daily_pnl
            )
            self._weekly_pnl = self._safe_float(
                portfolio.get("weekly_pnl"), self._weekly_pnl
            )
            self._total_equity = self._safe_float(
                portfolio.get("total_equity"), self._total_equity
            )
            self._consecutive_losses = self._safe_int(
                portfolio.get("consecutive_losses"), self._consecutive_losses
            )

        # Update unrealized PnL from open trades
        self._unrealized_pnl = sum(
            t.compute_pnl_r(self._current_price) for t in self._open_trades
        )

        # Track peak equity and compute drawdown
        if self._total_equity > self._peak_equity:
            self._peak_equity = self._total_equity

        if self._peak_equity > 0:
            self._current_drawdown = (
                (self._peak_equity - self._total_equity) / self._peak_equity
            )
        else:
            self._current_drawdown = 0.0

    # ──────────────────────────────────────────────────────────
    #  EPISODE TERMINATION
    # ──────────────────────────────────────────────────────────

    def _is_episode_over(self) -> Tuple[bool, str]:
        """
        Check if the episode should terminate.

        Termination conditions:
          1. Daily loss limit exceeded
          2. All positions closed and no more signals expected

        Note: Max steps is handled as truncation in step(), not here.

        Returns:
            Tuple of (terminated: bool, reason: str).
        """
        # Condition 1: Daily loss limit hit
        max_loss = self.initial_equity * (self.max_daily_loss_pct / 100.0)
        if self._daily_pnl < -max_loss:
            return True, "daily_loss_limit"

        # Condition 2: Catastrophic drawdown (>10%)
        if self._current_drawdown > 0.10:
            return True, "catastrophic_drawdown"

        return False, ""

    # ──────────────────────────────────────────────────────────
    #  INTERNAL HELPERS
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """
        Safely convert a value to float, with default on failure.

        Handles None, numpy types, and invalid values gracefully.
        """
        if value is None:
            return default
        try:
            result = float(value)
            if np.isnan(result) or np.isinf(result):
                return default
            return result
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        """
        Safely convert a value to int, with default on failure.
        """
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _log_episode_summary(self, reason: str) -> None:
        """
        Log a summary of the completed episode.

        Args:
            reason: Why the episode ended.
        """
        n_trades = len(self._closed_trades)
        wins = sum(1 for t in self._closed_trades if t.profit_r > 0)
        losses = sum(1 for t in self._closed_trades if t.profit_r <= 0)
        total_r = sum(t.profit_r for t in self._closed_trades)
        avg_r = total_r / max(n_trades, 1)
        win_rate = wins / max(n_trades, 1)

        log.info(
            f"[RL-ENV] {self.pair}: episode ended ({reason}) | "
            f"steps={self._current_step} trades={n_trades} "
            f"wins={wins} losses={losses} win_rate={win_rate:.1%} | "
            f"total_r={total_r:.2f} avg_r={avg_r:.2f} | "
            f"episode_reward={self._episode_total_reward:.3f} | "
            f"equity={self._total_equity:.2f} "
            f"drawdown={self._current_drawdown:.1%}"
        )

    # ──────────────────────────────────────────────────────────
    #  PROPERTIES / INSPECTION
    # ──────────────────────────────────────────────────────────

    @property
    def open_trades(self) -> List[SimulatedTrade]:
        """Return list of currently open simulated trades."""
        return list(self._open_trades)

    @property
    def closed_trades(self) -> List[SimulatedTrade]:
        """Return list of trades closed during this episode."""
        return list(self._closed_trades)

    @property
    def episode_stats(self) -> Dict[str, Any]:
        """Return episode statistics dict."""
        n_trades = len(self._closed_trades)
        wins = sum(1 for t in self._closed_trades if t.profit_r > 0)
        total_r = sum(t.profit_r for t in self._closed_trades)

        return {
            "pair": self.pair,
            "steps": self._current_step,
            "trades": n_trades,
            "wins": wins,
            "losses": n_trades - wins,
            "win_rate": wins / max(n_trades, 1),
            "total_r": total_r,
            "avg_r": total_r / max(n_trades, 1),
            "total_reward": self._episode_total_reward,
            "equity": self._total_equity,
            "drawdown": self._current_drawdown,
            "skips": self._episode_skips,
        }


# ════════════════════════════════════════════════════════════════
#  WRAPPER: Make TradingEnv a proper gymnasium.Env subclass
# ════════════════════════════════════════════════════════════════

if _GYMNASIUM_AVAILABLE:

    class GymTradingEnv(gym.Env):  # type: ignore[misc]
        """
        Full gymnasium.Env subclass wrapping TradingEnv.

        This thin wrapper makes TradingEnv compatible with standard
        RL libraries (Stable-Baselines3, CleanRL, etc.) that expect
        a proper gym.Env subclass with spaces defined in __init__.

        All logic is delegated to TradingEnv — this class only
        ensures correct gymnasium protocol compliance.

        Args:
            pair: Currency pair (e.g. "EURJPY").
            reward_scaling: Reward multiplier.
            max_steps_per_episode: Max signals per episode.
            max_daily_loss_pct: Daily loss limit as pct of equity.
            max_positions: Max concurrent positions.
            initial_equity: Starting equity.
        """

        metadata = {"render_modes": []}

        def __init__(
            self,
            pair: str = "EURJPY",
            reward_scaling: float = RL_REWARD_SCALING,
            max_steps_per_episode: int = RL_MAX_STEPS_PER_EPISODE,
            max_daily_loss_pct: float = RL_MAX_DAILY_LOSS_PCT,
            max_positions: int = RL_MAX_POSITIONS,
            initial_equity: float = 10000.0,
        ) -> None:
            super().__init__()

            self._env = TradingEnv(
                pair=pair,
                reward_scaling=reward_scaling,
                max_steps_per_episode=max_steps_per_episode,
                max_daily_loss_pct=max_daily_loss_pct,
                max_positions=max_positions,
                initial_equity=initial_equity,
            )

            # Expose spaces (required by gymnasium)
            self.observation_space = self._env.observation_space
            self.action_space = self._env.action_space

        def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None,
        ) -> Tuple[np.ndarray, Dict[str, Any]]:
            """Reset the environment. Gymnasium API."""
            return self._env.reset(seed=seed, options=options)

        def step(
            self, action: int
        ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
            """Execute one step. Gymnasium API."""
            return self._env.step(action)

        def receive_signal(
            self,
            fusion_result: Dict[str, Any],
            market_state: Dict[str, Any],
            portfolio_state: Dict[str, Any],
        ) -> None:
            """Push signal data. Delegated to inner TradingEnv."""
            self._env.receive_signal(fusion_result, market_state, portfolio_state)

        def close_trade(self, trade_id: int, profit_r: float) -> None:
            """Close a trade. Delegated to inner TradingEnv."""
            self._env.close_trade(trade_id, profit_r)

        def get_state(self) -> np.ndarray:
            """Peek at current observation. Delegated."""
            return self._env.get_state()

        @property
        def inner_env(self) -> TradingEnv:
            """Access the underlying TradingEnv for inspection."""
            return self._env

        @property
        def episode_stats(self) -> Dict[str, Any]:
            """Return current episode statistics."""
            return self._env.episode_stats

    # ══════════════════════════════════════════════════════════
    #  Register environment for gymnasium.make()
    # ══════════════════════════════════════════════════════════
    try:
        from gymnasium.envs.registration import register

        register(
            id="fxbot/Trading-v0",
            entry_point="rpde.rl_env:GymTradingEnv",
            max_episode_steps=RL_MAX_STEPS_PER_EPISODE,
        )
        log.debug("[RL-ENV] registered fxbot/Trading-v0 with gymnasium")
    except Exception as e:
        log.debug(f"[RL-ENV] gymnasium registration skipped: {e}")


else:
    # No gymnasium available — provide a clear error class
    class GymTradingEnv:  # type: ignore[no-redef]
        """
        Stub class when gymnasium is not installed.

        Raises ImportError on instantiation with a helpful message.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "gymnasium (or gym) is required for RL training. "
                "Install with: pip install gymnasium"
            )
