# =============================================================
# rpde/rl_agent.py  —  PPO Decision Engine (Phase 3)
#
# PURPOSE: Learn optimal trading decisions via Proximal Policy
# Optimization (PPO).  Receives fused signals from Phase 2's
# FusionLayer and learns when to enter, what size to use, when
# to skip — all through direct reward feedback from the market.
#
# ARCHITECTURE:
#
#   Shared Feature Extractor:
#     Input(28) -> Dense(128, ReLU) -> Dense(128, ReLU) -> Dense(64, ReLU)
#                                       |
#                       ┌───────────────┼───────────────┐
#                       ▼               ▼               ▼
#                 Actor Head      Critic Head    (auxiliary heads)
#                 Dense(64,ReLU)  Dense(64,ReLU)
#                 Dense(7,softmax) Dense(1)
#                 (policy logits)  (state value)
#
# ACTION SPACE (7 discrete actions):
#   0 = SKIP              (no trade)
#   1 = ENTER_BUY_05R     (buy, 0.5R size, tight stop, 0.5R TP)
#   2 = ENTER_BUY_1R      (buy, 1.0R size, medium stop, 1.0R TP)
#   3 = ENTER_BUY_15R     (buy, 1.5R size, medium stop, 2.0R TP)
#   4 = ENTER_SELL_05R    (sell, 0.5R size, tight stop, 0.5R TP)
#   5 = ENTER_SELL_1R     (sell, 1.0R size, medium stop, 1.0R TP)
#   6 = ENTER_SELL_15R    (sell, 1.5R size, medium stop, 2.0R TP)
#
# OBSERVATION SPACE (28 dims):
#   Fusion layer outputs + market state + portfolio state,
#   flattened into a single feature vector by the engine.
#
# MODEL PERSISTENCE:
#   rpde/models/rl/{PAIR}_rl_agent.pt    — PyTorch state dict
#   rpde/models/rl/{PAIR}_rl_meta.json   — Training stats, config snapshot
#
# GRACEFUL DEGRADATION:
#   If PyTorch is not installed, all public functions return safe
#   SKIP decisions instead of crashing.  The system logs a warning
#   once and continues operating in rule-based-only mode.
#
# USAGE (inference):
#   from rpde.rl_agent import load_rl_agent
#
#   agent = load_rl_agent("EURJPY")
#   decision = agent.decide(fusion_result, market_state, portfolio_state)
#   # decision["action"] = 3  ->  ENTER_BUY_1.5R
#
# USAGE (training):
#   from rpde.rl_agent import train_rl_agent
#   metrics = train_rl_agent("EURJPY", episodes=500)
# =============================================================

import json
import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Generator, List

import numpy as np

# ── Graceful PyTorch import ──────────────────────────────────
_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    optim = None
    Categorical = None

# When PyTorch is not installed, provide a dummy base class so that
# the ActorCriticNetwork (and any other nn.Module subclasses) can
# still be *defined* at module-import time without crashing.
# Instances will never be constructed when _TORCH_AVAILABLE is False.
if nn is None:
    class _DummyModule:
        """Stand-in for nn.Module when PyTorch is unavailable."""
        def __init__(self, *a, **kw):
            pass
        def __call__(self, *a, **kw):
            raise RuntimeError("PyTorch is not installed")
        def parameters(self):
            return iter(())
        def state_dict(self):
            return {}
        def load_state_dict(self, *a, **kw):
            pass
        def to(self, *a, **kw):
            return self
        def eval(self):
            return self
        def train(self, mode=True):
            return self
    nn = type('nn', (), {'Module': _DummyModule})()

# ── Internal imports ─────────────────────────────────────────
from core.logger import get_logger

# ── Config imports with safe fallback defaults ────────────────
# If the RL_* constants have been added to rpde/config.py they
# are used; otherwise we fall back to sensible production defaults
# so this file is self-contained and runnable immediately.
try:
    from rpde.config import (
        RL_OBS_DIM,
        RL_ACTION_DIM,
        RL_ACTOR_HIDDEN,
        RL_CRITIC_HIDDEN,
        RL_LEARNING_RATE,
        RL_GAMMA,
        RL_GAE_LAMBDA,
        RL_CLIP_RATIO,
        RL_VALUE_COEFF,
        RL_ENTROPY_COEFF,
        RL_MAX_GRAD_NORM,
        RL_PPO_EPOCHS,
        RL_MINI_BATCH_SIZE,
        RL_ROLLOUT_STEPS,
        RL_RETRAIN_DAYS,
        RL_MIN_TRAINING_EPISODES,
        RL_DEVICE,
        RL_MIXED_PRECISION,
    )
except ImportError:
    # ── Phase 3 RL defaults (used until config.py is updated) ──
    RL_OBS_DIM              = 28
    RL_ACTION_DIM           = 7
    RL_ACTOR_HIDDEN         = 128
    RL_CRITIC_HIDDEN        = 128
    RL_LEARNING_RATE        = 3e-4
    RL_GAMMA                = 0.99
    RL_GAE_LAMBDA           = 0.95
    RL_CLIP_RATIO           = 0.2
    RL_VALUE_COEFF          = 0.5
    RL_ENTROPY_COEFF        = 0.01
    RL_MAX_GRAD_NORM        = 0.5
    RL_PPO_EPOCHS           = 4
    RL_MINI_BATCH_SIZE      = 64
    RL_ROLLOUT_STEPS        = 2048
    RL_RETRAIN_DAYS         = 7
    RL_MIN_TRAINING_EPISODES = 50
    RL_DEVICE               = "auto"
    RL_MIXED_PRECISION      = True

log = get_logger(__name__)

# ════════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════════

# Base directory for RL model files
_RL_MODELS_DIR = Path(__file__).resolve().parent / "models" / "rl"

# File I/O lock for thread-safe atomic saves
_IO_LOCK = threading.Lock()

# Action-to-trade specification.  Every discrete action maps to a
# concrete set of trade parameters used by the execution layer.
ACTION_MAP: Dict[int, Dict[str, Any]] = {
    0: {
        "name": "SKIP",
        "entry": False,
        "direction": None,
        "size_r": 0.0,
        "stop_type": None,
        "tp_r": 0.0,
    },
    1: {
        "name": "ENTER_BUY_0.5R",
        "entry": True,
        "direction": "BUY",
        "size_r": 0.5,
        "stop_type": "tight",
        "tp_r": 0.5,
    },
    2: {
        "name": "ENTER_BUY_1R",
        "entry": True,
        "direction": "BUY",
        "size_r": 1.0,
        "stop_type": "medium",
        "tp_r": 1.0,
    },
    3: {
        "name": "ENTER_BUY_1.5R",
        "entry": True,
        "direction": "BUY",
        "size_r": 1.5,
        "stop_type": "medium",
        "tp_r": 2.0,
    },
    4: {
        "name": "ENTER_SELL_0.5R",
        "entry": True,
        "direction": "SELL",
        "size_r": 0.5,
        "stop_type": "tight",
        "tp_r": 0.5,
    },
    5: {
        "name": "ENTER_SELL_1R",
        "entry": True,
        "direction": "SELL",
        "size_r": 1.0,
        "stop_type": "medium",
        "tp_r": 1.0,
    },
    6: {
        "name": "ENTER_SELL_1.5R",
        "entry": True,
        "direction": "SELL",
        "size_r": 1.5,
        "stop_type": "medium",
        "tp_r": 2.0,
    },
}


# ════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════

def get_rl_device() -> str:
    """
    Determine the best available compute device for RL training.

    Returns:
        "cuda" if a CUDA GPU is available and torch is loaded,
        "mps"  if Apple Silicon GPU is available,
        "cpu"  otherwise.
    """
    if not _TORCH_AVAILABLE:
        return "cpu"

    if RL_DEVICE != "auto":
        return RL_DEVICE

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_gpu_info_rl() -> dict:
    """
    Return GPU availability and memory info for logging.

    Returns:
        Dict with keys: available, device_name, device_type,
        vram_total_mb, vram_free_mb, vram_used_mb.
    """
    if not _TORCH_AVAILABLE:
        return {
            "available": False,
            "device_name": "N/A (PyTorch not installed)",
            "device_type": "cpu",
            "vram_total_mb": 0,
            "vram_free_mb": 0,
            "vram_used_mb": 0,
        }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total = props.total_mem / (1024 ** 2)
        used = torch.cuda.memory_allocated(0) / (1024 ** 2)
        free = total - used
        return {
            "available": True,
            "device_name": props.name,
            "device_type": "cuda",
            "vram_total_mb": round(total, 1),
            "vram_free_mb": round(free, 1),
            "vram_used_mb": round(used, 1),
        }

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return {
            "available": True,
            "device_name": "Apple Silicon (MPS)",
            "device_type": "mps",
            "vram_total_mb": -1,
            "vram_free_mb": -1,
            "vram_used_mb": -1,
        }

    return {
        "available": False,
        "device_name": "CPU only",
        "device_type": "cpu",
        "vram_total_mb": 0,
        "vram_free_mb": 0,
        "vram_used_mb": 0,
    }


def _build_observation_vector(
    fusion_result: Dict[str, Any],
    market_state: Dict[str, Any],
    portfolio_state: Dict[str, Any],
) -> np.ndarray:
    """
    Convert heterogeneous trading signals into a flat 28-dim
    observation vector suitable for the RL network.

    The observation is designed to be information-rich yet compact:
      [0-7]   Fusion layer (confidence, expected_r, direction, TFT, agreement, reversal, xgb/tft avail)
      [8-14]  Market microstructure (atr, volatility regime, momentum, spread, session, trend strength)
      [15-21] Signal features (xgb confidence, xgb predicted_r, tft pattern_match, tft momentum, tft reversal)
      [22-27] Portfolio context (open positions, daily P&L, drawdown, consecutive wins/losses, exposure)

    Args:
        fusion_result:   Dict from FusionLayer.fuse().
        market_state:    Dict with market metadata (atr, spread, session, etc.).
        portfolio_state: Dict with current portfolio info (positions, P&L, etc.).

    Returns:
        numpy array of shape (28,) with float32 values.
    """
    obs = np.zeros(RL_OBS_DIM, dtype=np.float32)

    # ── Fusion layer features [0-7] ──
    # Combined confidence (clipped to [0, 1])
    obs[0] = float(np.clip(fusion_result.get("combined_confidence", 0.0), 0.0, 1.0))
    # Combined expected R (normalised by dividing by 3.0 so typical range is ~[-1, 1])
    obs[1] = float(np.clip(fusion_result.get("combined_expected_r", 0.0) / 3.0, -1.0, 1.0))
    # Direction encoding: BUY=1.0, SELL=-1.0, None=0.0
    direction = fusion_result.get("direction")
    if direction == "BUY":
        obs[2] = 1.0
    elif direction == "SELL":
        obs[2] = -1.0
    # TFT contribution (already in [0, 1])
    obs[3] = float(np.clip(fusion_result.get("tft_contribution", 0.0), 0.0, 1.0))
    # Signal agreement: ALL_AGREE=1.0, XGB_TFT_AGREE=0.7, PARTIAL=0.3, DISAGREE=0.0
    agreement_map = {"ALL_AGREE": 1.0, "XGB_TFT_AGREE": 0.7, "PARTIAL": 0.3, "DISAGREE": 0.0}
    obs[4] = float(agreement_map.get(fusion_result.get("signal_agreement", "DISAGREE"), 0.0))
    # Reversal warning binary
    obs[5] = 1.0 if fusion_result.get("reversal_warning") else 0.0
    # Recommendation: TAKE=1.0, CAUTION=0.5, SKIP=0.0
    rec_map = {"TAKE": 1.0, "CAUTION": 0.5, "SKIP": 0.0}
    obs[6] = float(rec_map.get(fusion_result.get("recommendation", "SKIP"), 0.0))
    # XGB / TFT weight balance (xgb_weight / (xgb_weight + tft_weight))
    weights = fusion_result.get("weights", {})
    xgb_w = float(weights.get("xgb_weight", 0.5))
    tft_w = float(weights.get("tft_weight", 0.5))
    total_w = xgb_w + tft_w
    obs[7] = xgb_w / total_w if total_w > 0 else 0.5

    # ── Market microstructure features [8-14] ──
    # ATR normalised (typical forex ATR is 5-50 pips; normalise by /50)
    obs[8] = float(np.clip(market_state.get("atr", 0.0) / 50.0, 0.0, 2.0))
    # Volatility percentile [0, 1]
    obs[9] = float(np.clip(market_state.get("atr_percentile", 0.5), 0.0, 1.0))
    # Momentum score [-1, 1]
    obs[10] = float(np.clip(market_state.get("momentum_score", 0.0), -1.0, 1.0))
    # Spread normalised (typical spread 0.5-5 pips; /5)
    obs[11] = float(np.clip(market_state.get("spread", 1.0) / 5.0, 0.0, 1.0))
    # Session encoding: London=0.8, NY_London=1.0, NY=0.6, Tokyo=0.4, Sydney=0.2, Off=0.0
    session_map = {
        "LONDON": 0.8, "NY_LONDON_OVERLAP": 1.0, "NY_AFTERNOON": 0.6,
        "TOKYO": 0.4, "SYDNEY": 0.2, "OFF": 0.0,
    }
    obs[12] = float(session_map.get(market_state.get("session", "OFF"), 0.0))
    # Trend strength [-1, 1] (negative = bearish)
    obs[13] = float(np.clip(market_state.get("trend_strength", 0.0), -1.0, 1.0))
    # Market regime encoding (ranging=0, trending_strong=1, breakout=0.8, volatile=0.3)
    regime_map = {
        "RANGING": 0.0, "TRENDING_STRONG": 1.0, "TRENDING_WEAK": 0.6,
        "BREAKOUT_ACCEPTED": 0.8, "VOLATILE": 0.3, "LOW_LIQUIDITY": 0.1,
    }
    obs[14] = float(regime_map.get(market_state.get("market_regime", "RANGING"), 0.0))

    # ── Signal detail features [15-21] ──
    # XGB confidence
    obs[15] = float(np.clip(fusion_result.get("xgb_confidence", 0.0) if fusion_result.get("xgb_available") else 0.0, 0.0, 1.0))
    # XGB predicted R (normalised)
    obs[16] = float(np.clip(fusion_result.get("xgb_predicted_r", 0.0) / 3.0, -1.0, 1.0)) if fusion_result.get("xgb_available") else 0.0
    # TFT pattern match
    obs[17] = float(np.clip(fusion_result.get("tft_pattern_match", 0.0) if fusion_result.get("tft_available") else 0.0, 0.0, 1.0))
    # TFT momentum score
    obs[18] = float(np.clip(fusion_result.get("tft_momentum", 0.0) if fusion_result.get("tft_available") else 0.0, -1.0, 1.0))
    # TFT reversal probability
    obs[19] = float(np.clip(fusion_result.get("tft_reversal", 0.0) if fusion_result.get("tft_available") else 0.0, 0.0, 1.0))
    # Pattern library match score
    obs[20] = float(np.clip(fusion_result.get("pattern_match_score", 0.0) if fusion_result.get("pattern_available") else 0.0, 0.0, 1.0))
    # Direction agreement count (how many of 3 agree on same direction)
    obs[21] = float(np.clip(fusion_result.get("agreement_count", 0) / 3.0, 0.0, 1.0))

    # ── Portfolio context features [22-27] ──
    # Number of open positions (normalised by /5)
    obs[22] = float(np.clip(portfolio_state.get("open_positions", 0) / 5.0, 0.0, 1.0))
    # Daily P&L in R (clipped to [-5, 5])
    obs[23] = float(np.clip(portfolio_state.get("daily_pnl_r", 0.0), -5.0, 5.0) / 5.0)
    # Current drawdown [0, 1]
    obs[24] = float(np.clip(portfolio_state.get("drawdown", 0.0), 0.0, 1.0))
    # Consecutive losses (normalised by /5)
    obs[25] = float(np.clip(portfolio_state.get("consecutive_losses", 0) / 5.0, 0.0, 1.0))
    # Consecutive wins (normalised by /5)
    obs[26] = float(np.clip(portfolio_state.get("consecutive_wins", 0) / 5.0, 0.0, 1.0))
    # Account exposure (normalised by /max_positions)
    obs[27] = float(np.clip(
        portfolio_state.get("exposure", 0.0) / max(portfolio_state.get("max_exposure", 5.0), 1.0),
        0.0, 1.0
    ))

    return obs


def _skip_decision(reason: str) -> dict:
    """
    Return a safe SKIP decision when RL is unavailable or the
    agent is not yet trained.

    This ensures the system never crashes and always returns a
    valid decision dict, even when PyTorch is missing.
    """
    return {
        "action": 0,
        "action_name": "SKIP",
        "entry": False,
        "direction": None,
        "size_r": 0.0,
        "stop_type": None,
        "tp_r": 0.0,
        "confidence": 0.0,
        "value": 0.0,
        "reason": reason,
    }


# ════════════════════════════════════════════════════════════════
#  ACTOR-CRITIC NETWORK
# ════════════════════════════════════════════════════════════════

class ActorCriticNetwork(nn.Module):
    """
    Shared-feature-extractor actor-critic network for PPO.

    Architecture
    ------------
    Shared backbone:
        Input(obs_dim) -> Linear(128, ReLU) -> Linear(128, ReLU) -> Linear(64, ReLU)

    Actor head (policy):
        Linear(64, ReLU) -> Linear(action_dim)  [logits; softmax applied externally]

    Critic head (value):
        Linear(64, ReLU) -> Linear(1)  [scalar state value]

    The shared backbone is the key design choice: by learning a common
    representation, the critic provides a richer training signal for
    the actor, while the actor's gradient updates improve the shared
    features for the critic.  This is especially important for trading
    where the same market features drive both *what* to do (actor) and
    *how good* the current state is (critic).

    Weights are initialised with orthogonal init (gain=sqrt(2) for ReLU
    layers, gain=1 for the final linear layers) which has been shown to
    improve PPO stability.
    """

    def __init__(self, obs_dim: int = RL_OBS_DIM, action_dim: int = RL_ACTION_DIM,
                 hidden_dim: int = RL_ACTOR_HIDDEN):
        """
        Args:
            obs_dim:    Dimension of the observation vector (default 28).
            action_dim: Number of discrete actions (default 7).
            hidden_dim: Width of the shared hidden layers (default 128).
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # ── Shared feature extractor ──
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),  # 64
            nn.ReLU(),
        )

        # ── Actor head (policy) ──
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

        # ── Critic head (value) ──
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # ── Orthogonal weight initialisation ──
        self._init_weights()

        # Action map reference for logging
        self.action_names = [ACTION_MAP[i]["name"] for i in range(action_dim)]

    def _init_weights(self):
        """Apply orthogonal initialisation to all Linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Final policy layer: small gain to start near-uniform policy
        # This prevents the agent from committing too strongly early on
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)

        # Final value layer: gain=1 is standard
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.zeros_(self.critic[-1].bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the shared backbone and both heads.

        Args:
            obs: Tensor of shape (batch, obs_dim) or (obs_dim,).

        Returns:
            Tuple of (action_logits, value) where:
              - action_logits: (batch, action_dim) raw logits
              - value:         (batch, 1) or (1,) scalar state value
        """
        features = self.shared(obs)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value

    def get_action(self, obs: torch.Tensor,
                   deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the current policy.

        During inference the caller typically passes a single observation
        (1, obs_dim) and receives a single action.

        Args:
            obs:           Tensor of shape (1, obs_dim).
            deterministic: If True, pick the argmax action (no sampling).
                           Used for evaluation / deployment.

        Returns:
            Tuple of (action_int, log_prob, value) where:
              - action_int: Python int, the chosen action index.
              - log_prob:   Tensor, log probability of the chosen action.
              - value:      Tensor, estimated state value V(s).
        """
        action_logits, value = self.forward(obs)

        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
            # Compute log_prob manually for deterministic actions
            log_prob = torch.log(torch.softmax(action_logits, dim=-1).gather(-1, action.unsqueeze(-1)).squeeze(-1) + 1e-8)
        else:
            dist = Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob, value.squeeze(-1)

    def evaluate(self, obs: torch.Tensor,
                 actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate a batch of (obs, action) pairs.

        Used during PPO training to compute log_probs, values, and
        entropy for the current policy over the rollout data.

        Args:
            obs:     Tensor of shape (batch, obs_dim).
            actions: Tensor of shape (batch,) with action indices.

        Returns:
            Tuple of (log_probs, values, entropy):
              - log_probs: (batch,) log probability of each action under
                           the current policy.
              - values:    (batch,) estimated state value V(s).
              - entropy:   () scalar mean entropy (for entropy bonus).
        """
        action_logits, value = self.forward(obs)
        dist = Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, value.squeeze(-1), entropy.mean()


# ════════════════════════════════════════════════════════════════
#  ROLLOUT BUFFER
# ════════════════════════════════════════════════════════════════

class RolloutBuffer:
    """
    Stores transitions for PPO training using a pre-allocated
    numpy ring buffer for memory efficiency.

    After a rollout is collected, call ``compute_gae()`` to compute
    Generalized Advantage Estimation (GAE) advantages and discounted
    returns.  Then iterate over ``get_batches()`` for mini-batch PPO
    updates.

    Attributes:
        obs:        (N, obs_dim) observations
        actions:    (N,) discrete action indices
        log_probs:  (N,) log probabilities of taken actions
        rewards:    (N,) rewards received
        values:     (N,) critic value estimates V(s)
        dones:      (N,) episode termination flags
        advantages: (N,) GAE advantages (computed after rollout)
        returns:    (N,) discounted returns (computed after rollout)
    """

    def __init__(self, rollout_steps: int, obs_dim: int = RL_OBS_DIM):
        """
        Args:
            rollout_steps: Number of transitions per rollout (N).
            obs_dim:       Observation dimension.
        """
        self.rollout_steps = rollout_steps
        self.obs_dim = obs_dim
        self.ptr = 0  # Current write position
        self.full = False  # Whether the buffer has been filled once

        # Pre-allocate numpy arrays (much faster than appending lists)
        self.obs = np.zeros((rollout_steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros(rollout_steps, dtype=np.int64)
        self.log_probs = np.zeros(rollout_steps, dtype=np.float32)
        self.rewards = np.zeros(rollout_steps, dtype=np.float32)
        self.values = np.zeros(rollout_steps, dtype=np.float32)
        self.dones = np.zeros(rollout_steps, dtype=np.float32)

        # Computed after rollout
        self.advantages = np.zeros(rollout_steps, dtype=np.float32)
        self.returns = np.zeros(rollout_steps, dtype=np.float32)

    def add(self, obs: np.ndarray, action: int, log_prob: float,
            reward: float, value: float, done: bool):
        """
        Store a single transition.

        Args:
            obs:      Observation vector of shape (obs_dim,).
            action:   Discrete action index (int).
            log_prob: Log probability of the action under the policy.
            reward:   Scalar reward received.
            value:    Critic's value estimate V(s).
            done:     Whether the episode terminated.
        """
        if self.ptr >= self.rollout_steps:
            raise RuntimeError(
                f"RolloutBuffer overflow: tried to add step {self.ptr} "
                f"but buffer only holds {self.rollout_steps}. "
                f"Call compute_gae() and reset before adding more."
            )
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = float(done)
        self.ptr += 1

    def compute_gae(self, last_value: float, last_done: bool,
                    gamma: float = RL_GAMMA,
                    gae_lambda: float = RL_GAE_LAMBDA):
        """
        Compute Generalized Advantage Estimation (GAE) and discounted
        returns for the entire rollout.

        GAE (λ) balances bias vs variance in advantage estimation:
          - λ=0: TD(0) advantage — low variance, high bias
          - λ=1: Monte Carlo advantage — high variance, low bias
          - λ=0.95 (default): good practical trade-off

        The return is computed as: G_t = A_t + V(s_t)

        Args:
            last_value: Critic's value estimate for the state *after*
                        the last transition (bootstrap value).
            last_done:  Whether the episode terminated after the last
                        transition (no bootstrapping if True).
            gamma:      Discount factor (default 0.99).
            gae_lambda: GAE lambda (default 0.95).
        """
        n = self.ptr
        if n == 0:
            return

        gae = 0.0
        self.advantages[n - 1] = 0.0

        # Walk backwards through the rollout
        for t in reversed(range(n)):
            if t == n - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            # Temporal difference error
            delta = (self.rewards[t]
                     + gamma * next_value * next_non_terminal
                     - self.values[t])

            # GAE running sum
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae

        # Compute returns: R_t = A_t + V(s_t)
        self.returns[:n] = self.advantages[:n] + self.values[:n]

        self.full = True

    def get_batches(self, batch_size: int = RL_MINI_BATCH_SIZE,
                    device: str = "cpu") -> Generator[dict, None, None]:
        """
        Yield shuffled mini-batches from the rollout.

        Each batch is a dict of torch tensors moved to the specified
        device.  This generator is consumed by the PPO update loop.

        Args:
            batch_size: Mini-batch size (default 64).
            device:     Target torch device string.

        Yields:
            Dict with keys: obs, actions, old_log_probs, advantages,
                            returns — all as torch tensors on device.
        """
        n = self.ptr
        if n == 0:
            return

        # Generate random permutation indices
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_indices = indices[start:end]

            yield {
                "obs": torch.tensor(self.obs[batch_indices], dtype=torch.float32, device=device),
                "actions": torch.tensor(self.actions[batch_indices], dtype=torch.int64, device=device),
                "old_log_probs": torch.tensor(self.log_probs[batch_indices], dtype=torch.float32, device=device),
                "advantages": torch.tensor(self.advantages[batch_indices], dtype=torch.float32, device=device),
                "returns": torch.tensor(self.returns[batch_indices], dtype=torch.float32, device=device),
            }

    def reset(self):
        """Clear the buffer for the next rollout."""
        self.ptr = 0
        self.full = False

    def __len__(self) -> int:
        return self.ptr


# ════════════════════════════════════════════════════════════════
#  RL DECISION ENGINE
# ════════════════════════════════════════════════════════════════

class RLDecisionEngine:
    """
    Main interface for RL-based trading decisions.

    This is the production-facing class that wraps the ActorCritic
    network, handles training, persistence, and inference.

    Lifecycle
    ---------
    1. Instantiate: ``engine = RLDecisionEngine("EURJPY")``
    2. Load model:  ``engine.load()``  (or pass ``autoload=True``)
    3. Inference:   ``engine.decide(fusion_result, market_state, portfolio_state)``
    4. Feedback:    ``engine.update_from_trade(trade_id, profit_r)``
    5. Retrain:     ``engine.train(episodes=100)``
    6. Persist:     ``engine.save()``

    Training Pipeline
    -----------------
    1. Collect rollout buffer of N steps from TradingEnv
    2. Compute GAE advantages and returns
    3. For each PPO epoch (default 4):
       a. Sample mini-batches from rollout
       b. Compute actor loss (clipped surrogate objective)
       c. Compute critic loss (MSE between predicted V and return)
       d. Compute entropy bonus (encourages exploration)
       e. Backpropagate total loss = actor - entropy*coeff + critic*coeff
       f. Clip gradients to max norm
    4. Log metrics to the training history

    Safety
    ------
    - If PyTorch is not installed, all methods degrade gracefully
      and return SKIP decisions.
    - An untrained model (``is_trained == False``) always returns SKIP.
    - Atomic saves (write .tmp → rename) prevent corrupt model files.
    - Thread-safe I/O with a module-level lock.
    """

    def __init__(self, pair: str, device: str = None):
        """
        Args:
            pair:   Uppercase currency pair (e.g. "EURJPY").
            device: Compute device override.  None → auto-detect.
        """
        self.pair = pair.upper()
        self.device = device or get_rl_device()

        # ── Network ──
        self.network: Optional[ActorCriticNetwork] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scaler = None  # GradScaler for mixed precision

        # ── Mixed precision ──
        self.use_amp = RL_MIXED_PRECISION and self.device == "cuda" and _TORCH_AVAILABLE

        # ── Training state ──
        self.is_trained = False
        self.training_episodes = 0
        self.total_timesteps = 0
        self.total_updates = 0
        self.best_reward_mean = -float("inf")

        # ── Training history (for metrics / diagnostics) ──
        self.history: List[dict] = []

        # ── Performance tracking ──
        self._trade_outcomes: List[dict] = []  # Recent trades for feedback

        # ── Persistence paths ──
        _RL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self._model_path = _RL_MODELS_DIR / f"{self.pair}_rl_agent.pt"
        self._meta_path = _RL_MODELS_DIR / f"{self.pair}_rl_meta.json"

        # ── Initialise network ──
        if _TORCH_AVAILABLE:
            self._init_network()

        log.info(
            f"[RL] {self.pair}: DecisionEngine initialised "
            f"(device={self.device}, torch={'ok' if _TORCH_AVAILABLE else 'MISSING'}, "
            f"trained={self.is_trained})"
        )

    # ──────────────────────────────────────────────────────────
    #  NETWORK INITIALISATION
    # ──────────────────────────────────────────────────────────

    def _init_network(self):
        """Create the actor-critic network and optimiser."""
        if not _TORCH_AVAILABLE:
            return

        self.network = ActorCriticNetwork(
            obs_dim=RL_OBS_DIM,
            action_dim=RL_ACTION_DIM,
            hidden_dim=RL_ACTOR_HIDDEN,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=RL_LEARNING_RATE,
            eps=1e-5,
        )

        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

    # ──────────────────────────────────────────────────────────
    #  INFERENCE — THE MAIN ENTRY POINT
    # ──────────────────────────────────────────────────────────

    def decide(self, fusion_result: Dict[str, Any],
               market_state: Dict[str, Any],
               portfolio_state: Dict[str, Any]) -> dict:
        """
        Make a trading decision based on current market state.

        This is the primary method called by the live trading loop.
        It converts the heterogeneous input signals into a flat
        observation vector, feeds it through the policy network,
        and maps the chosen action to concrete trade parameters.

        If the agent is not trained or PyTorch is unavailable,
        a safe SKIP decision is returned.

        Args:
            fusion_result:   Dict from FusionLayer.fuse().
            market_state:    Dict with market metadata.
            portfolio_state: Dict with portfolio info.

        Returns:
            Decision dict with keys: action, action_name, entry,
            direction, size_r, stop_type, tp_r, confidence, value,
            reason.
        """
        # ── Guard: PyTorch not available ──
        if not _TORCH_AVAILABLE:
            return _skip_decision("PyTorch not installed — RL engine unavailable")

        # ── Guard: network not initialised ──
        if self.network is None:
            return _skip_decision("RL network not initialised")

        # ── Guard: model not yet trained ──
        if not self.is_trained:
            return _skip_decision("RL model not yet trained — returning SKIP")

        # ── Guard: consecutive-loss protection ──
        # If the portfolio is in drawdown, the agent should be more
        # cautious.  We still run the policy but apply a confidence
        # dampening factor so the gate layer can more easily override.
        consecutive_losses = portfolio_state.get("consecutive_losses", 0)
        drawdown = portfolio_state.get("drawdown", 0.0)

        # ── Build observation ──
        obs_np = _build_observation_vector(fusion_result, market_state, portfolio_state)
        obs_tensor = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        # ── Get action from policy ──
        self.network.eval()
        with torch.no_grad():
            action, log_prob, value = self.network.get_action(obs_tensor, deterministic=True)

        # ── Map action to trade params ──
        trade_spec = ACTION_MAP.get(action, ACTION_MAP[0])
        confidence = torch.exp(log_prob).item()

        # ── Confidence dampening in adverse conditions ──
        if consecutive_losses >= 3 or drawdown > 0.05:
            dampening = max(0.3, 1.0 - consecutive_losses * 0.15 - drawdown * 2.0)
            confidence *= dampening
            log.debug(
                f"[RL] {self.pair}: confidence dampened "
                f"(losses={consecutive_losses}, dd={drawdown:.2%}) "
                f"-> {confidence:.3f}"
            )

        # ── Build reason string ──
        value_item = value.item() if isinstance(value, torch.Tensor) else float(value)
        reason = (
            f"[{self.pair}] RL decision: {trade_spec['name']} "
            f"(confidence={confidence:.3f}, value={value_item:.3f}, "
            f"fusion_conf={fusion_result.get('combined_confidence', 0):.3f})"
        )

        if trade_spec["entry"]:
            log.info(reason)
        else:
            log.debug(reason)

        return {
            "action": action,
            "action_name": trade_spec["name"],
            "entry": trade_spec["entry"],
            "direction": trade_spec["direction"],
            "size_r": trade_spec["size_r"],
            "stop_type": trade_spec["stop_type"],
            "tp_r": trade_spec["tp_r"],
            "confidence": round(confidence, 4),
            "value": round(value_item, 4),
            "reason": reason,
        }

    # ──────────────────────────────────────────────────────────
    #  TRAINING
    # ──────────────────────────────────────────────────────────

    def train(self, episodes: int = 100,
              env=None, callback=None) -> dict:
        """
        Run PPO training for the specified number of episodes.

        Training Loop
        -------------
        For each episode:
          1. Collect a full rollout from the environment (or replay buffer)
          2. Compute GAE advantages
          3. Run PPO update (multiple epochs of mini-batch gradient steps)
          4. Log metrics

        Args:
            episodes: Number of episodes to train.
            env:      TradingEnv instance.  If None, training is skipped
                      with a warning (requires environment setup).
            callback: Optional callable(epoch, metrics) for progress.

        Returns:
            Dict with training metrics: total_episodes, total_timesteps,
            policy_loss, value_loss, entropy, reward_mean, etc.
        """
        if not _TORCH_AVAILABLE:
            log.warning("[RL] PyTorch not available — cannot train")
            return {"status": "error", "reason": "PyTorch not installed"}

        if self.network is None:
            self._init_network()

        if env is None:
            log.warning(
                f"[RL] {self.pair}: no environment provided — "
                f"cannot train.  Pass a TradingEnv instance."
            )
            return {
                "status": "error",
                "reason": "No TradingEnv environment provided",
                "pair": self.pair,
            }

        self.network.train()
        buffer = RolloutBuffer(RL_ROLLOUT_STEPS, RL_OBS_DIM)

        # ── Aggregate metrics across all episodes ──
        all_policy_losses = []
        all_value_losses = []
        all_entropies = []
        all_reward_means = []
        episode_rewards = []

        log.info(
            f"[RL] {self.pair}: starting PPO training "
            f"({episodes} episodes, {RL_ROLLOUT_STEPS} rollout steps, "
            f"device={self.device})"
        )

        for episode_idx in range(episodes):
            # ── 1. Collect rollout ──
            obs = env.reset()
            episode_reward = 0.0
            episode_length = 0

            for step in range(RL_ROLLOUT_STEPS):
                obs_tensor = torch.tensor(
                    obs, dtype=torch.float32, device=self.device
                ).unsqueeze(0)

                with torch.no_grad():
                    action, log_prob, value = self.network.get_action(obs_tensor)

                next_obs, reward, done, info = env.step(action)

                # Store transition (detach tensors to avoid graph buildup)
                buffer.add(
                    obs=obs,
                    action=action,
                    log_prob=log_prob.item(),
                    reward=reward,
                    value=value.item(),
                    done=done,
                )

                episode_reward += reward
                episode_length += 1
                self.total_timesteps += 1
                obs = next_obs

                if done:
                    episode_rewards.append(episode_reward)
                    episode_reward = 0.0
                    episode_length = 0
                    obs = env.reset()

            # ── 2. Compute GAE ──
            # Bootstrap value for the last state
            with torch.no_grad():
                obs_tensor = torch.tensor(
                    obs, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                _, _, last_value = self.network.get_action(obs_tensor)
                last_value = last_value.item()

            buffer.compute_gae(
                last_value=last_value,
                last_done=done,
                gamma=RL_GAMMA,
                gae_lambda=RL_GAE_LAMBDA,
            )

            # ── 3. PPO update ──
            update_metrics = self._ppo_update(buffer)
            all_policy_losses.append(update_metrics["policy_loss"])
            all_value_losses.append(update_metrics["value_loss"])
            all_entropies.append(update_metrics["entropy"])

            reward_mean = float(np.mean(episode_rewards[-10:])) if episode_rewards else 0.0
            all_reward_means.append(reward_mean)

            # ── Track best model ──
            if reward_mean > self.best_reward_mean:
                self.best_reward_mean = reward_mean
                self.save()

            # ── Log ──
            self.training_episodes += 1
            self.total_updates += 1

            epoch_metrics = {
                "episode": episode_idx + 1,
                "total_episodes": self.training_episodes,
                "total_timesteps": self.total_timesteps,
                "reward_mean_10": round(reward_mean, 4),
                "policy_loss": round(update_metrics["policy_loss"], 6),
                "value_loss": round(update_metrics["value_loss"], 6),
                "entropy": round(update_metrics["entropy"], 6),
                "approx_kl": round(update_metrics["approx_kl"], 6),
                "clip_fraction": round(update_metrics["clip_fraction"], 4),
                "buffer_size": len(buffer),
            }
            self.history.append(epoch_metrics)

            if callback:
                callback(episode_idx, epoch_metrics)

            if (episode_idx + 1) % 10 == 0 or episode_idx == 0:
                log.info(
                    f"[RL] {self.pair}: episode {episode_idx + 1}/{episodes} "
                    f"reward_mean={reward_mean:.3f} "
                    f"pi_loss={update_metrics['policy_loss']:.4f} "
                    f"v_loss={update_metrics['value_loss']:.4f} "
                    f"entropy={update_metrics['entropy']:.4f} "
                    f"kl={update_metrics['approx_kl']:.4f}"
                )

            # Reset buffer for next rollout
            buffer.reset()

        # ── Final save ──
        self.is_trained = True
        self.save()

        # ── Summary ──
        summary = {
            "status": "complete",
            "pair": self.pair,
            "episodes_trained": episodes,
            "total_episodes": self.training_episodes,
            "total_timesteps": self.total_timesteps,
            "final_policy_loss": round(float(np.mean(all_policy_losses[-10:])), 6),
            "final_value_loss": round(float(np.mean(all_value_losses[-10:])), 6),
            "final_entropy": round(float(np.mean(all_entropies[-10:])), 6),
            "best_reward_mean": round(self.best_reward_mean, 4),
            "device": self.device,
            "model_path": str(self._model_path),
        }

        log.info(
            f"[RL] {self.pair}: training complete — "
            f"{summary['episodes_trained']} episodes, "
            f"{summary['total_timesteps']} timesteps, "
            f"best_reward_mean={summary['best_reward_mean']:.3f}"
        )

        return summary

    def _ppo_update(self, buffer: RolloutBuffer) -> dict:
        """
        Run PPO update on the collected rollout.

        For each PPO epoch:
          1. Iterate over mini-batches
          2. Compute new log_probs, values, entropy from current policy
          3. Compute ratio = exp(new_log_prob - old_log_prob)
          4. Clipped surrogate objective:
             L_CLIP = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
          5. Value loss: MSE(V(s), GAE_return)
          6. Total loss = -L_CLIP + c1 * value_loss - c2 * entropy
          7. Gradient clipping to max norm

        Returns:
            Dict with policy_loss, value_loss, entropy, approx_kl,
            clip_fraction.
        """
        self.network.train()

        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_entropies = []
        epoch_approx_kls = []
        epoch_clip_fractions = []

        # Normalise advantages for stability (standard PPO practice)
        with torch.no_grad():
            adv_batch = buffer.advantages[:buffer.ptr]
            adv_mean = adv_batch.mean()
            adv_std = adv_batch.std() + 1e-8
            buffer.advantages[:buffer.ptr] = (adv_batch - adv_mean) / adv_std

        for _epoch in range(RL_PPO_EPOCHS):
            for batch in buffer.get_batches(RL_MINI_BATCH_SIZE, self.device):
                obs_batch = batch["obs"]
                actions_batch = batch["actions"]
                old_log_probs_batch = batch["old_log_probs"]
                advantages_batch = batch["advantages"]
                returns_batch = batch["returns"]

                # Evaluate current policy on rollout data
                new_log_probs, new_values, entropy = self.network.evaluate(
                    obs_batch, actions_batch
                )

                # ── Clipped surrogate objective ──
                # ratio = π_new(a|s) / π_old(a|s)
                ratio = torch.exp(new_log_probs - old_log_probs_batch)

                # L_CLIP = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(
                    ratio, 1.0 - RL_CLIP_RATIO, 1.0 + RL_CLIP_RATIO
                ) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                # ── Value loss ──
                value_loss = nn.functional.mse_loss(new_values, returns_batch)

                # ── Total loss ──
                # Negative entropy bonus encourages exploration
                total_loss = (
                    policy_loss
                    + RL_VALUE_COEFF * value_loss
                    - RL_ENTROPY_COEFF * entropy
                )

                # ── Backpropagation ──
                self.optimizer.zero_grad()

                if self.use_amp and self.scaler is not None:
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.network.parameters(), RL_MAX_GRAD_NORM
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.network.parameters(), RL_MAX_GRAD_NORM
                    )
                    self.optimizer.step()

                # ── Diagnostics ──
                with torch.no_grad():
                    # Approximate KL divergence between old and new policy
                    approx_kl = ((ratio - 1.0) - (new_log_probs - old_log_probs_batch)).mean().item()
                    # Fraction of samples where clipping was active
                    clip_fraction = (
                        (abs(ratio - 1.0) > RL_CLIP_RATIO).float().mean().item()
                    )

                epoch_policy_losses.append(policy_loss.item())
                epoch_value_losses.append(value_loss.item())
                epoch_entropies.append(entropy.item())
                epoch_approx_kls.append(approx_kl)
                epoch_clip_fractions.append(clip_fraction)

        return {
            "policy_loss": float(np.mean(epoch_policy_losses)),
            "value_loss": float(np.mean(epoch_value_losses)),
            "entropy": float(np.mean(epoch_entropies)),
            "approx_kl": float(np.mean(epoch_approx_kls)),
            "clip_fraction": float(np.mean(epoch_clip_fractions)),
        }

    # ──────────────────────────────────────────────────────────
    #  TRADE FEEDBACK
    # ──────────────────────────────────────────────────────────

    def update_from_trade(self, trade_id: int, profit_r: float) -> None:
        """
        Record a trade outcome for future training and diagnostics.

        This does NOT trigger online learning updates (PPO requires
        full rollouts).  Instead, it stores the outcome for:
          - Periodic retraining (episodes built from real trade data)
          - Performance tracking and diagnostics
          - Meta-logging for model quality assessment

        Args:
            trade_id: Unique trade identifier from the execution layer.
            profit_r: R-multiple profit (positive = win, negative = loss).
        """
        outcome = {
            "trade_id": trade_id,
            "profit_r": profit_r,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "is_win": profit_r > 0,
        }
        self._trade_outcomes.append(outcome)

        # Keep only the last 500 outcomes to prevent unbounded growth
        if len(self._trade_outcomes) > 500:
            self._trade_outcomes = self._trade_outcomes[-500:]

        log.debug(
            f"[RL] {self.pair}: trade outcome recorded "
            f"(id={trade_id}, profit_r={profit_r:.2f}, "
            f"total_outcomes={len(self._trade_outcomes)})"
        )

    # ──────────────────────────────────────────────────────────
    #  PERSISTENCE
    # ──────────────────────────────────────────────────────────

    def save(self) -> None:
        """
        Persist model weights and metadata to disk.

        Uses atomic writes (write to .tmp then rename) to prevent
        corrupt files if the process is interrupted mid-write.

        Model file:     rpde/models/rl/{PAIR}_rl_agent.pt
        Metadata file:  rpde/models/rl/{PAIR}_rl_meta.json
        """
        if not _TORCH_AVAILABLE or self.network is None:
            log.warning(f"[RL] {self.pair}: cannot save — PyTorch or network unavailable")
            return

        _RL_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        with _IO_LOCK:
            try:
                # ── Save model weights (atomic) ──
                tmp_model = str(self._model_path) + ".tmp"
                torch.save({
                    "network_state_dict": self.network.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
                    "obs_dim": RL_OBS_DIM,
                    "action_dim": RL_ACTION_DIM,
                    "hidden_dim": RL_ACTOR_HIDDEN,
                    "pair": self.pair,
                }, tmp_model)
                os.replace(tmp_model, str(self._model_path))

                # ── Save metadata (atomic) ──
                meta = self._build_meta()
                tmp_meta = str(self._meta_path) + ".tmp"
                with open(tmp_meta, "w") as f:
                    json.dump(meta, f, indent=2, default=str)
                os.replace(tmp_meta, str(self._meta_path))

                log.debug(
                    f"[RL] {self.pair}: model saved "
                    f"({self._model_path.name}, "
                    f"{self.training_episodes} episodes)"
                )

            except Exception as e:
                log.error(f"[RL] {self.pair}: failed to save model: {e}")

    def load(self) -> bool:
        """
        Load model weights and metadata from disk.

        Returns:
            True if a model was successfully loaded, False otherwise.
        """
        if not _TORCH_AVAILABLE:
            log.warning(f"[RL] {self.pair}: cannot load — PyTorch not installed")
            return False

        if not self._model_path.exists():
            log.debug(f"[RL] {self.pair}: no saved model at {self._model_path}")
            return False

        with _IO_LOCK:
            try:
                checkpoint = torch.load(
                    str(self._model_path),
                    map_location=self.device,
                    weights_only=False,
                )

                # ── Validate checkpoint ──
                expected_keys = {"network_state_dict", "obs_dim", "action_dim", "pair"}
                if not expected_keys.issubset(checkpoint.keys()):
                    log.warning(
                        f"[RL] {self.pair}: checkpoint missing expected keys, "
                        f"got {list(checkpoint.keys())}"
                    )
                    return False

                if checkpoint["pair"] != self.pair:
                    log.warning(
                        f"[RL] checkpoint pair mismatch: "
                        f"expected {self.pair}, got {checkpoint['pair']}"
                    )
                    return False

                # ── Rebuild network if dimensions changed ──
                obs_dim = checkpoint.get("obs_dim", RL_OBS_DIM)
                action_dim = checkpoint.get("action_dim", RL_ACTION_DIM)
                hidden_dim = checkpoint.get("hidden_dim", RL_ACTOR_HIDDEN)

                if (obs_dim != RL_OBS_DIM or action_dim != RL_ACTION_DIM
                        or hidden_dim != RL_ACTOR_HIDDEN):
                    log.info(
                        f"[RL] {self.pair}: checkpoint dimensions differ from config "
                        f"(obs={obs_dim} vs {RL_OBS_DIM}, act={action_dim} vs {RL_ACTION_DIM}), "
                        f"rebuilding network"
                    )
                    self.network = ActorCriticNetwork(
                        obs_dim=obs_dim,
                        action_dim=action_dim,
                        hidden_dim=hidden_dim,
                    ).to(self.device)
                    self.optimizer = optim.Adam(
                        self.network.parameters(),
                        lr=RL_LEARNING_RATE,
                        eps=1e-5,
                    )

                # ── Load weights ──
                self.network.load_state_dict(checkpoint["network_state_dict"])

                if checkpoint.get("optimizer_state_dict") and self.optimizer:
                    try:
                        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    except Exception:
                        log.debug(
                            f"[RL] {self.pair}: optimizer state could not be loaded "
                            f"(likely LR or param mismatch) — using fresh optimizer"
                        )

                self.network.to(self.device)
                self.is_trained = True

                # ── Load metadata ──
                self._load_meta()

                log.info(
                    f"[RL] {self.pair}: model loaded "
                    f"({self.training_episodes} episodes, "
                    f"{self.total_timesteps} timesteps, "
                    f"best_reward={self.best_reward_mean:.3f})"
                )
                return True

            except Exception as e:
                log.error(f"[RL] {self.pair}: failed to load model: {e}")
                return False

    def _build_meta(self) -> dict:
        """
        Build the metadata dict for persistence.

        Includes training statistics, performance metrics, and a
        snapshot of the config used for this model (critical for
        reproducibility).
        """
        # Compute recent performance from trade outcomes
        recent = self._trade_outcomes[-50:] if self._trade_outcomes else []
        wins = sum(1 for t in recent if t["is_win"])
        total = len(recent)
        avg_profit = float(np.mean([t["profit_r"] for t in recent])) if recent else 0.0

        return {
            "pair": self.pair,
            "is_trained": self.is_trained,
            "training_episodes": self.training_episodes,
            "total_timesteps": self.total_timesteps,
            "total_updates": self.total_updates,
            "best_reward_mean": self.best_reward_mean,
            "device": self.device,
            "saved_at": datetime.now(timezone.utc).isoformat(),

            # Performance metrics (last 50 trades)
            "performance": {
                "recent_trades": total,
                "recent_wins": wins,
                "recent_win_rate": round(wins / total, 4) if total > 0 else 0.0,
                "recent_avg_profit_r": round(avg_profit, 4),
            },

            # Config snapshot (for reproducibility)
            "config": {
                "obs_dim": RL_OBS_DIM,
                "action_dim": RL_ACTION_DIM,
                "actor_hidden": RL_ACTOR_HIDDEN,
                "critic_hidden": RL_CRITIC_HIDDEN,
                "learning_rate": RL_LEARNING_RATE,
                "gamma": RL_GAMMA,
                "gae_lambda": RL_GAE_LAMBDA,
                "clip_ratio": RL_CLIP_RATIO,
                "value_coeff": RL_VALUE_COEFF,
                "entropy_coeff": RL_ENTROPY_COEFF,
                "max_grad_norm": RL_MAX_GRAD_NORM,
                "ppo_epochs": RL_PPO_EPOCHS,
                "mini_batch_size": RL_MINI_BATCH_SIZE,
                "rollout_steps": RL_ROLLOUT_STEPS,
                "mixed_precision": RL_MIXED_PRECISION,
            },

            # Last 10 history entries (for quick diagnostics)
            "recent_history": self.history[-10:],
        }

    def _load_meta(self) -> None:
        """Load metadata from the JSON sidecar file."""
        if not self._meta_path.exists():
            return

        try:
            with open(self._meta_path, "r") as f:
                meta = json.load(f)

            self.training_episodes = int(meta.get("training_episodes", 0))
            self.total_timesteps = int(meta.get("total_timesteps", 0))
            self.total_updates = int(meta.get("total_updates", 0))
            self.best_reward_mean = float(meta.get("best_reward_mean", -float("inf")))
            self.history = meta.get("history", [])

            # Restore trade outcomes if available
            perf = meta.get("performance", {})
            # Note: individual trade outcomes are not persisted in meta
            # (they come from the live system).  We restore aggregate stats.

        except Exception as e:
            log.warning(f"[RL] {self.pair}: failed to load metadata: {e}")

    # ──────────────────────────────────────────────────────────
    #  STATUS / DIAGNOSTICS
    # ──────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """
        Return comprehensive training status and performance metrics.

        Useful for the dashboard, CLI, and monitoring systems.

        Returns:
            Dict with: pair, is_trained, training_episodes,
            total_timesteps, best_reward_mean, device, model_exists,
            performance, recent_history_summary, gpu_info.
        """
        recent = self._trade_outcomes[-50:] if self._trade_outcomes else []
        wins = sum(1 for t in recent if t["is_win"])
        total = len(recent)
        avg_profit = float(np.mean([t["profit_r"] for t in recent])) if recent else 0.0

        # Last 5 history entries summary
        last_history = self.history[-5:] if self.history else []

        return {
            "pair": self.pair,
            "is_trained": self.is_trained,
            "training_episodes": self.training_episodes,
            "total_timesteps": self.total_timesteps,
            "total_updates": self.total_updates,
            "best_reward_mean": round(self.best_reward_mean, 4),
            "device": self.device,
            "model_exists": self._model_path.exists(),
            "meta_exists": self._meta_path.exists(),
            "model_path": str(self._model_path),
            "meta_path": str(self._meta_path),
            "torch_available": _TORCH_AVAILABLE,
            "mixed_precision": self.use_amp,

            # Live performance
            "performance": {
                "total_trades_tracked": len(self._trade_outcomes),
                "recent_trades": total,
                "recent_wins": wins,
                "recent_win_rate": round(wins / total, 4) if total > 0 else 0.0,
                "recent_avg_profit_r": round(avg_profit, 4),
            },

            # Recent training history
            "recent_history": last_history,

            # GPU info
            "gpu_info": get_gpu_info_rl(),
        }


# ════════════════════════════════════════════════════════════════
#  MODULE-LEVEL CONVENIENCE FUNCTIONS
# ════════════════════════════════════════════════════════════════

# Instance cache to avoid re-creating engines on every call
_agent_cache: Dict[str, RLDecisionEngine] = {}
_agent_cache_lock = threading.Lock()


def train_rl_agent(pair: str, episodes: int = 100,
                   env=None, callback=None, **kwargs) -> dict:
    """
    Convenience function: train an RL agent for a given pair.

    Creates or reuses an RLDecisionEngine, runs training, and
    returns the training metrics dict.

    Args:
        pair:     Currency pair string (e.g. "EURJPY").
        episodes: Number of training episodes.
        env:      TradingEnv instance (required for training).
        callback: Optional callable(epoch, metrics) for progress.
        **kwargs: Additional keyword arguments (device override, etc.).

    Returns:
        Training metrics dict from RLDecisionEngine.train().

    Example::

        from rpde.rl_env import TradingEnv
        env = TradingEnv("EURJPY")
        metrics = train_rl_agent("EURJPY", episodes=500, env=env)
    """
    if not _TORCH_AVAILABLE:
        log.warning("[RL] PyTorch not available — cannot train agent")
        return {"status": "error", "reason": "PyTorch not installed"}

    pair_upper = pair.upper()

    with _agent_cache_lock:
        if pair_upper not in _agent_cache:
            _agent_cache[pair_upper] = RLDecisionEngine(
                pair_upper,
                device=kwargs.get("device"),
            )
        engine = _agent_cache[pair_upper]

    return engine.train(episodes=episodes, env=env, callback=callback)


def load_rl_agent(pair: str, device: str = None) -> RLDecisionEngine:
    """
    Convenience function: load (or create) an RL agent for a pair.

    If a trained model exists on disk it is loaded automatically.
    Otherwise a fresh (untrained) engine is returned that will
    yield SKIP decisions until trained.

    Args:
        pair:   Currency pair string (e.g. "EURJPY").
        device: Optional device override.

    Returns:
        RLDecisionEngine instance (trained if model exists on disk).

    Example::

        agent = load_rl_agent("EURJPY")
        decision = agent.decide(fusion_result, market_state, portfolio_state)
    """
    pair_upper = pair.upper()

    with _agent_cache_lock:
        if pair_upper not in _agent_cache:
            _agent_cache[pair_upper] = RLDecisionEngine(
                pair_upper,
                device=device,
            )
        engine = _agent_cache[pair_upper]

    # Attempt to load saved model
    if not engine.is_trained:
        engine.load()

    return engine
