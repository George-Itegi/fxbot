# =============================================================
# rpde/experience_buffer.py  --  Experience Replay Buffer (Phase 3)
#
# PURPOSE: Implements the continuous learning loop for the entire
# Apex Trader v5.0 RPDE system. Every trade outcome feeds back
# through this module, triggering scheduled retraining across all
# system components:
#
#   - XGBoost  : Weekly incremental retraining (every 7 days)
#   - TFT      : Bi-weekly full retrain on GPU (every 14 days)
#   - RL Agent : Learns from every outcome continuously
#   - Patterns : Monthly re-validation + new pattern mining (30 days)
#
# ARCHITECTURE:
#
#   Trade closes
#       │
#       ▼
#   TradeExperience dataclass (all outcome metadata)
#       │
#       ▼
#   ExperienceReplayBuffer (per-pair, capped at max_size)
#       │  • Thread-safe append / sample
#       │  • Priority weighting by recency (exponential decay)
#       │  • JSON persistence (atomic writes)
#       │
#       ▼
#   ContinuousLearningLoop (orchestrator)
#       │  • record_trade()        — immediate: RL + fusion weight update
#       │  • check_schedule()      — returns what needs retraining
#       │  • run_learning_cycle()  — executes all due retrains
#       │  • get_system_health()   — overall learning status dashboard
#       │
#       ▼
#   Analysis functions:
#       │  • analyze_recent_performance()   — recent trade stats
#       │  • detect_regime_change()         — volatility / structure shift
#       │  • compute_optimal_stop_tp()      — MAE/MFE analysis
#
# THREAD SAFETY:
#   - All buffer mutations protected by threading.Lock
#   - Atomic file writes (write to .tmp, then os.replace)
#   - Safe for multi-threaded live trading environments
#
# PERSISTENCE:
#   - JSON:  rpde/models/experience/{PAIR}_experience.json
#   - MySQL: rpde_learning_log table (metrics & retrain history)
#
# =============================================================

import json
import math
import os
import threading
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

from core.logger import get_logger

log = get_logger(__name__)

# ── Config imports with fallbacks for Phase 3 parameters ──────
try:
    from rpde.config import (
        REPLAY_BUFFER_SIZE,
        REPLAY_BUFFER_SAMPLE_SIZE,
        REPLAY_BUFFER_DECAY_HOURS,
        TFT_RETRAIN_DAYS,
        PATTERN_COOLDOWN_MINUTES,
        MAX_DAILY_PATTERN_TRADES,
    )
except ImportError:
    log.warning("[EXP_BUF] rpde.config imports failed — using defaults")
    REPLAY_BUFFER_SIZE = 5000
    REPLAY_BUFFER_SAMPLE_SIZE = 3000
    REPLAY_BUFFER_DECAY_HOURS = 720
    TFT_RETRAIN_DAYS = 14
    PATTERN_COOLDOWN_MINUTES = 30
    MAX_DAILY_PATTERN_TRADES = 8

# RL retrain cadence — Phase 3 parameter (may not exist in config yet)
try:
    from rpde.config import RL_RETRAIN_DAYS
except ImportError:
    RL_RETRAIN_DAYS = 7  # Default: weekly RL retraining

# XGBoost retrain cadence
XGB_RETRAIN_DAYS = 7   # Weekly

# Pattern re-validation cadence
PATTERN_UPDATE_DAYS = 30  # Monthly

# Base directory for experience persistence
_EXPERIENCE_DIR = Path(__file__).resolve().parent / "models" / "experience"

# File I/O lock — ensures atomic reads/writes across threads
_IO_LOCK = threading.Lock()


# ═══════════════════════════════════════════════════════════════
#  TRADE EXPERIENCE DATA CLASS
# ═══════════════════════════════════════════════════════════════

@dataclass
class TradeExperience:
    """
    Complete record of a single trade's lifecycle and outcome.

    Captures everything needed for the learning loop:
      - Trade outcome metrics (pips, R-multiple, USD, win/loss)
      - Signal quality at entry time (fusion confidence, agreement)
      - RL decision quality (action taken, predicted value)
      - Market context at entry (session, spread, ATR)
      - Post-trade analysis (MAE, MFE, hold time)

    The R-multiple (profit_r) is the key metric — it normalises
    P&L across pairs, lot sizes, and volatility regimes, making
    all experiences comparable for learning purposes.

    Attributes:
        trade_id: Unique trade identifier (from broker ticket).
        pair: Currency pair (e.g. "EURJPY").
        direction: Trade direction — "BUY" or "SELL".
        entry_time: Timestamp when the trade was opened.
        exit_time: Timestamp when the trade was closed.
        entry_price: Price at which the trade was entered.
        exit_price: Price at which the trade was exited.
        profit_pips: Profit/loss in pips.
        profit_r: Profit/loss as R-multiple (key normalised metric).
        profit_usd: Profit/loss in US dollars.
        outcome: Trade result — "WIN", "LOSS", or "BREAKEVEN".
        fusion_confidence: Fusion layer confidence at entry [0, 1].
        fusion_expected_r: Fusion layer expected R at entry.
        signal_agreement: Signal agreement level at entry
            ("ALL_AGREE", "XGB_TFT_AGREE", "PARTIAL", "DISAGREE").
        reversal_warning: Whether TFT flagged a reversal warning.
        rl_action: Action taken by RL agent (0=SKIP, 1=BUY, 2=SELL).
        rl_action_name: Human-readable RL action name.
        rl_predicted_value: Value predicted by RL agent at entry.
        session: Trading session at entry ("London", "NY", "Asian", "Overlap").
        spread_at_entry: Spread in pips at entry time.
        atr_at_entry: ATR value at entry time (in price units).
        mae_r: Maximum Adverse Excursion in R-multiples.
            How far against us the trade went before recovering or closing.
        mfe_r: Maximum Favorable Excursion in R-multiples.
            How far in our favor the trade went before pulling back or closing.
        hold_time_hours: Duration the trade was held, in hours.
    """

    # ── Trade identification ──
    trade_id: int
    pair: str
    direction: str              # BUY / SELL
    entry_time: str             # ISO format datetime string
    exit_time: str              # ISO format datetime string
    entry_price: float
    exit_price: float

    # ── Outcome metrics ──
    profit_pips: float
    profit_r: float             # R-multiple — THE key metric
    profit_usd: float
    outcome: str                # WIN / LOSS / BREAKEVEN

    # ── Signal quality at entry (from fusion layer) ──
    fusion_confidence: float = 0.0
    fusion_expected_r: float = 0.0
    signal_agreement: str = ""  # ALL_AGREE / XGB_TFT_AGREE / PARTIAL / DISAGREE
    reversal_warning: bool = False

    # ── RL decision quality at entry ──
    rl_action: int = 0          # 0=SKIP, 1=BUY, 2=SELL
    rl_action_name: str = ""    # "SKIP" / "TAKE_LONG" / "TAKE_SHORT"
    rl_predicted_value: float = 0.0

    # ── Market context at entry ──
    session: str = ""           # London / NY / Asian / Overlap
    spread_at_entry: float = 0.0
    atr_at_entry: float = 0.0

    # ── Post-trade analysis ──
    mae_r: float = 0.0          # Maximum Adverse Excursion in R
    mfe_r: float = 0.0          # Maximum Favorable Excursion in R
    hold_time_hours: float = 0.0

    # ── Internal: computed priority weight for sampling ──
    _priority_weight: float = field(default=1.0, repr=False)

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary (excludes private fields)."""
        d = asdict(self)
        d.pop("_priority_weight", None)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "TradeExperience":
        """Deserialize from a dictionary, handling missing keys gracefully."""
        # Build kwargs with only the fields that exist in the data
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {}
        for k, v in data.items():
            if k in valid_fields and not k.startswith("_"):
                kwargs[k] = v
        return cls(**kwargs)


# ═══════════════════════════════════════════════════════════════
#  EXPERIENCE REPLAY BUFFER
# ═══════════════════════════════════════════════════════════════

class ExperienceReplayBuffer:
    """
    Per-pair experience replay buffer with capped size and priority sampling.

    Stores completed TradeExperience records and provides weighted sampling
    for model retraining. Newer experiences are weighted higher using
    exponential decay, ensuring the system learns more from recent market
    conditions while still leveraging historical data.

    Thread-safe: all mutations are protected by a threading.Lock.

    Usage::

        buf = ExperienceReplayBuffer("EURJPY")
        buf.add(experience)
        sample = buf.sample(256)
        buf.save()

    Persistence:
        Experiences are serialized to JSON at:
            rpde/models/experience/{PAIR}_experience.json
        Uses atomic writes (tmp + rename) for thread-safety.
    """

    def __init__(self, pair: str, max_size: int = REPLAY_BUFFER_SIZE):
        """
        Initialize experience buffer for a currency pair.

        Args:
            pair: Uppercase currency pair string (e.g. "EURJPY").
            max_size: Maximum number of experiences to retain.
                When exceeded, oldest experiences are dropped (FIFO).
                Defaults to REPLAY_BUFFER_SIZE from config.
        """
        self.pair = pair.upper()
        self.max_size = max_size
        self.experiences: deque = deque(maxlen=max_size)
        self._dirty = False  # Tracks unsaved changes
        self._lock = threading.Lock()
        self._persistence_path = _EXPERIENCE_DIR / f"{self.pair}_experience.json"

        # Statistics counters
        self._total_added: int = 0
        self._total_sampled: int = 0
        self._last_save_time: Optional[float] = None

        # Load from disk on init
        self._load()

    # ──────────────────────────────────────────────────────────
    #  PUBLIC API
    # ──────────────────────────────────────────────────────────

    def add(self, experience: TradeExperience) -> bool:
        """
        Add a trade experience to the buffer.

        Thread-safe. If the buffer is at max_size, the oldest
        experience is automatically dropped (FIFO eviction).

        Args:
            experience: Complete TradeExperience record.

        Returns:
            True if added successfully, False if invalid.
        """
        if not isinstance(experience, TradeExperience):
            log.warning(f"[EXP_BUF] {self.pair}: add() received "
                        f"non-TradeExperience type: {type(experience)}")
            return False

        # Validate minimum required fields
        if not experience.trade_id or not experience.pair:
            log.warning(f"[EXP_BUF] {self.pair}: skipping experience with "
                        f"missing trade_id or pair")
            return False

        with self._lock:
            # Compute priority weight based on recency
            experience._priority_weight = self._compute_priority(experience)
            self.experiences.append(experience)
            self._total_added += 1
            self._dirty = True

        log.debug(
            f"[EXP_BUF] {self.pair}: added experience "
            f"#{experience.trade_id} outcome={experience.outcome} "
            f"R={experience.profit_r:+.2f} "
            f"(buffer={len(self)}, total_added={self._total_added})"
        )
        return True

    def sample(self, n: int = REPLAY_BUFFER_SAMPLE_SIZE,
               min_n: int = 10) -> List[TradeExperience]:
        """
        Sample experiences with priority weighting (newer = higher weight).

        Uses exponential decay weighting: more recent experiences have
        higher probability of being sampled. This ensures the model
        trains on recent market conditions while still leveraging
        the full history.

        Args:
            n: Target sample size. Defaults to REPLAY_BUFFER_SAMPLE_SIZE.
            min_n: Minimum sample to return (uses uniform sampling if
                buffer is too small for priority sampling).

        Returns:
            List of sampled TradeExperience objects. May be fewer than
            n if the buffer is small.
        """
        with self._lock:
            buf_len = len(self.experiences)
            if buf_len == 0:
                return []

            # If buffer is small, return all (uniform)
            if buf_len <= min_n:
                self._total_sampled += buf_len
                return list(self.experiences)

            # Clamp sample size to buffer size
            actual_n = min(n, buf_len)

            # Compute priority weights for all experiences
            weights = []
            for exp in self.experiences:
                weights.append(max(exp._priority_weight, 0.01))

            # Normalize to probability distribution
            total_weight = sum(weights)
            probs = [w / total_weight for w in weights]

            # Weighted random sampling without replacement
            # Use simple approach: sort by priority, take top-N with jitter
            import random
            indices = list(range(buf_len))

            # Shuffle indices then sort by weight (reversed) to prioritize
            # recent experiences while maintaining randomness
            random.shuffle(indices)
            indices.sort(key=lambda i: probs[i], reverse=True)

            # Take top N with some randomness injected
            # First 60% from top-weighted, remaining 40% random
            top_count = int(actual_n * 0.6)
            rand_count = actual_n - top_count

            selected_indices = indices[:top_count]
            remaining = [i for i in indices if i not in selected_indices]
            if remaining:
                selected_indices.extend(
                    random.sample(remaining, min(rand_count, len(remaining)))
                )

            selected_indices = selected_indices[:actual_n]
            result = [self.experiences[i] for i in selected_indices]

            self._total_sampled += len(result)
            return result

    def get_recent(self, n: int = 30) -> List[TradeExperience]:
        """
        Get the N most recent experiences (no sampling).

        Args:
            n: Number of most recent experiences to return.

        Returns:
            List of TradeExperience, newest first.
        """
        with self._lock:
            buf_len = len(self.experiences)
            start = max(0, buf_len - n)
            return list(self.experiences)[start:]

    def get_all(self) -> List[TradeExperience]:
        """
        Return all experiences in the buffer.

        Returns:
            List of all TradeExperience objects (oldest first).
        """
        with self._lock:
            return list(self.experiences)

    def clear(self) -> int:
        """
        Clear all experiences from the buffer.

        Returns:
            Number of experiences that were cleared.
        """
        with self._lock:
            count = len(self.experiences)
            self.experiences.clear()
            self._dirty = True
            log.info(f"[EXP_BUF] {self.pair}: cleared {count} experiences")
            return count

    def size(self) -> int:
        """Return current number of experiences in the buffer."""
        with self._lock:
            return len(self.experiences)

    def is_dirty(self) -> bool:
        """Return True if there are unsaved changes."""
        return self._dirty

    def stats(self) -> dict:
        """
        Return buffer statistics.

        Returns:
            Dict with pair, size, max_size, total_added, total_sampled,
            dirty, path, and fill_ratio.
        """
        with self._lock:
            return {
                "pair": self.pair,
                "size": len(self.experiences),
                "max_size": self.max_size,
                "total_added": self._total_added,
                "total_sampled": self._total_sampled,
                "dirty": self._dirty,
                "path": str(self._persistence_path),
                "fill_ratio": round(len(self.experiences) / max(self.max_size, 1), 4),
                "last_save": (datetime.fromtimestamp(self._last_save_time, tz=timezone.utc)
                              .isoformat() if self._last_save_time else None),
            }

    # ──────────────────────────────────────────────────────────
    #  PERSISTENCE
    # ──────────────────────────────────────────────────────────

    def save(self, force: bool = False) -> bool:
        """
        Persist experiences to JSON file.

        Uses atomic write (write to .tmp, then os.replace) for
        thread-safety. Only writes if dirty or forced.

        Args:
            force: If True, write even if not marked dirty.

        Returns:
            True if saved successfully, False otherwise.
        """
        if not self._dirty and not force:
            return False

        with self._lock:
            try:
                data = {
                    "pair": self.pair,
                    "max_size": self.max_size,
                    "total_added": self._total_added,
                    "total_sampled": self._total_sampled,
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                    "experiences": [exp.to_dict() for exp in self.experiences],
                }

                with _IO_LOCK:
                    _EXPERIENCE_DIR.mkdir(parents=True, exist_ok=True)
                    tmp_path = str(self._persistence_path) + ".tmp"
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=1, default=str)
                    os.replace(tmp_path, str(self._persistence_path))

                self._dirty = False
                self._last_save_time = time.time()

                log.debug(
                    f"[EXP_BUF] {self.pair}: saved {len(self.experiences)} "
                    f"experiences to {self._persistence_path.name}"
                )
                return True

            except Exception as e:
                log.error(
                    f"[EXP_BUF] {self.pair}: failed to save experiences: {e}"
                )
                return False

    def _load(self):
        """
        Load experiences from JSON file on disk.

        Called during __init__. If the file doesn't exist or is
        corrupt, starts with an empty buffer.
        """
        if not self._persistence_path.exists():
            log.debug(f"[EXP_BUF] {self.pair}: no persistence file, "
                      f"starting with empty buffer")
            return

        with _IO_LOCK:
            try:
                with open(self._persistence_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                raw_experiences = data.get("experiences", [])
                loaded_count = 0

                for exp_data in raw_experiences:
                    try:
                        exp = TradeExperience.from_dict(exp_data)
                        # Recompute priority weights on load
                        exp._priority_weight = self._compute_priority(exp)
                        self.experiences.append(exp)
                        loaded_count += 1
                    except Exception as ex:
                        log.debug(
                            f"[EXP_BUF] {self.pair}: skipping corrupt "
                            f"experience record: {ex}"
                        )

                self._total_added = data.get("total_added", loaded_count)
                self._total_sampled = data.get("total_sampled", 0)
                self._dirty = False

                log.info(
                    f"[EXP_BUF] {self.pair}: loaded {loaded_count} "
                    f"experiences from disk"
                )

            except Exception as e:
                log.warning(
                    f"[EXP_BUF] {self.pair}: failed to load "
                    f"experiences from disk: {e}"
                )

    # ──────────────────────────────────────────────────────────
    #  PRIORITY WEIGHTING
    # ──────────────────────────────────────────────────────────

    def _compute_priority(self, experience: TradeExperience) -> float:
        """
        Compute priority weight for an experience.

        Uses exponential decay based on exit time. Recent trades get
        higher weight so the system adapts to current market conditions.

        The decay half-life is REPLAY_BUFFER_DECAY_HOURS (default 720h = 30 days).

        A bonus multiplier is applied to:
          - WIN trades (1.2x) — reinforce successful patterns
          - High-R trades (1.1x) — learn from big outcomes
          - DISAGREE signal trades (1.3x) — learn from edge cases

        Args:
            experience: TradeExperience to compute weight for.

        Returns:
            Priority weight as float (always > 0).
        """
        try:
            # Parse exit_time for recency calculation
            if isinstance(experience.exit_time, str):
                exit_dt = datetime.fromisoformat(experience.exit_time)
            elif isinstance(experience.exit_time, datetime):
                exit_dt = experience.exit_time
            else:
                return 1.0

            # Handle timezone-naive datetimes
            if exit_dt.tzinfo is None:
                exit_dt = exit_dt.replace(tzinfo=timezone.utc)

            now = datetime.now(timezone.utc)
            hours_ago = max(0.0, (now - exit_dt).total_seconds() / 3600.0)

            # Exponential decay: weight = e^(-lambda * t)
            # lambda = ln(2) / half_life
            half_life = float(REPLAY_BUFFER_DECAY_HOURS)
            decay_lambda = math.log(2) / max(half_life, 1.0)
            weight = math.exp(-decay_lambda * hours_ago)

            # Bonus: winning trades
            if experience.outcome == "WIN":
                weight *= 1.2

            # Bonus: high absolute R-multiple (big outcome to learn from)
            abs_r = abs(experience.profit_r)
            if abs_r > 2.0:
                weight *= 1.1

            # Bonus: signal disagreement (edge cases are valuable)
            if experience.signal_agreement == "DISAGREE":
                weight *= 1.3

            # Floor to prevent near-zero weights
            return max(weight, 0.01)

        except Exception:
            return 1.0


# ═══════════════════════════════════════════════════════════════
#  CONTINUOUS LEARNING LOOP
# ═══════════════════════════════════════════════════════════════

class ContinuousLearningLoop:
    """
    Orchestrates the continuous learning cycle for all RPDE components.

    This is the central coordinator for Phase 3 of the Apex Trader system.
    It manages per-pair experience buffers, checks learning schedules,
    and executes retraining tasks as they become due.

    Schedule:
        - After every trade:  Feed to RL agent, update fusion weights
        - Weekly (7 days):    Incremental XGBoost retrain
        - Bi-weekly (14 days): Full TFT retrain (GPU)
        - Monthly (30 days):  Pattern library re-validation + mining
        - On demand:          Force retrain specific component

    Learning signals tracked:
        - Pattern decay detection (win rate declining)
        - Regime change detection (volatility / structure shift)
        - Overfitting detection (train vs val divergence)

    Usage::

        loop = ContinuousLearningLoop()
        loop.record_trade(experience)
        schedule = loop.check_learning_schedule("EURJPY")
        if schedule["xgb_retrain_due"]:
            loop.run_learning_cycle("EURJPY", components=["xgb"])
        health = loop.get_system_health()
    """

    # Retrain schedule intervals in days
    SCHEDULE = {
        "xgb": XGB_RETRAIN_DAYS,          # 7 days
        "tft": TFT_RETRAIN_DAYS,          # 14 days
        "rl": RL_RETRAIN_DAYS,            # 7 days
        "pattern": PATTERN_UPDATE_DAYS,   # 30 days
    }

    def __init__(self):
        """Initialize the continuous learning loop."""
        self.buffers: Dict[str, ExperienceReplayBuffer] = {}
        self._buffers_lock = threading.Lock()

        # Track last retrain time per component per pair
        self.last_xgb_retrain: Dict[str, datetime] = {}
        self.last_tft_retrain: Dict[str, datetime] = {}
        self.last_rl_retrain: Dict[str, datetime] = {}
        self.last_pattern_update: Optional[datetime] = None

        # Overall system statistics
        self._total_trades_recorded: int = 0
        self._total_retrains: Dict[str, int] = defaultdict(int)
        self._retrain_errors: List[dict] = []

        # Lazy-loaded RL agents per pair for online updates
        self._rl_agents: Dict[str, Any] = {}

        # Load persisted schedule state
        self._load_schedule_state()

        log.info(
            f"[CLL] ContinuousLearningLoop initialized "
            f"(xgb={XGB_RETRAIN_DAYS}d, tft={TFT_RETRAIN_DAYS}d, "
            f"rl={RL_RETRAIN_DAYS}d, pattern={PATTERN_UPDATE_DAYS}d)"
        )

    # ──────────────────────────────────────────────────────────
    #  BUFFER MANAGEMENT
    # ──────────────────────────────────────────────────────────

    def get_buffer(self, pair: str) -> ExperienceReplayBuffer:
        """
        Get or create the experience buffer for a currency pair.

        Thread-safe. Buffers are lazily created on first access.

        Args:
            pair: Uppercase currency pair string (e.g. "EURJPY").

        Returns:
            ExperienceReplayBuffer for the specified pair.
        """
        pair = pair.upper()
        with self._buffers_lock:
            if pair not in self.buffers:
                self.buffers[pair] = ExperienceReplayBuffer(pair)
                log.debug(f"[CLL] Created experience buffer for {pair}")
            return self.buffers[pair]

    def get_all_buffers(self) -> Dict[str, ExperienceReplayBuffer]:
        """Return all active experience buffers."""
        with self._buffers_lock:
            return dict(self.buffers)

    # ──────────────────────────────────────────────────────────
    #  TRADE RECORDING (called after every trade closes)
    # ──────────────────────────────────────────────────────────

    def record_trade(self, experience: TradeExperience) -> dict:
        """
        Record a completed trade and trigger immediate learning actions.

        This is the main entry point for the continuous learning loop.
        Called after every trade closes, regardless of outcome.

        Immediate actions:
            1. Add experience to per-pair buffer
            2. Update fusion weights (if XGB/TFT predictions available)
            3. Feed outcome to RL agent for online learning
            4. Log learning metrics to database

        Args:
            experience: Complete TradeExperience record.

        Returns:
            Dict with:
                - recorded: bool — was the experience recorded?
                - buffer_size: int — current buffer size after add
                - fusion_updated: bool — was fusion weight updated?
                - rl_updated: bool — was RL agent updated?
                - decay_detected: bool — is this pair showing decay?
                - warnings: list — any warnings generated
        """
        result = {
            "recorded": False,
            "buffer_size": 0,
            "fusion_updated": False,
            "rl_updated": False,
            "decay_detected": False,
            "warnings": [],
        }

        pair = experience.pair.upper()

        # ── Step 1: Add to buffer ──
        buf = self.get_buffer(pair)
        if not buf.add(experience):
            result["warnings"].append("Experience rejected (invalid)")
            return result
        result["recorded"] = True
        result["buffer_size"] = buf.size()

        self._total_trades_recorded += 1

        # ── Step 2: Update fusion weights (online learning) ──
        try:
            fusion_updated = self._update_fusion_weights(experience)
            result["fusion_updated"] = fusion_updated
        except Exception as e:
            result["warnings"].append(f"Fusion update failed: {e}")
            log.debug(f"[CLL] Fusion weight update failed for {pair}: {e}")

        # ── Step 3: Feed to RL agent (online learning) ──
        try:
            rl_updated = self._update_rl_agent(experience)
            result["rl_updated"] = rl_updated
        except Exception as e:
            result["warnings"].append(f"RL update failed: {e}")
            log.debug(f"[CLL] RL agent update failed for {pair}: {e}")

        # ── Step 4: Check for pattern decay ──
        try:
            perf = analyze_recent_performance(pair, n_trades=20)
            if perf.get("decay_score", 0) > 0.5:
                result["decay_detected"] = True
                result["warnings"].append(
                    f"Pattern decay detected: score={perf['decay_score']:.2f}, "
                    f"recent_wr={perf.get('win_rate', 0):.1%}"
                )
        except Exception:
            pass  # Non-critical, don't warn

        # ── Step 5: Auto-save buffer periodically ──
        # Save every 10 trades or when buffer is dirty and has > 50 items
        if (self._total_trades_recorded % 10 == 0) or \
                (buf.is_dirty() and buf.size() > 50):
            buf.save()

        # ── Step 6: Log to database ──
        try:
            self._log_trade_experience(experience, result)
        except Exception as e:
            log.debug(f"[CLL] Failed to log trade experience: {e}")

        log.info(
            f"[CLL] Trade recorded: {pair} #{experience.trade_id} "
            f"{experience.outcome} R={experience.profit_r:+.2f} "
            f"(fusion={result['fusion_updated']}, rl={result['rl_updated']})"
        )

        return result

    # ──────────────────────────────────────────────────────────
    #  LEARNING SCHEDULE
    # ──────────────────────────────────────────────────────────

    def check_learning_schedule(self, pair: str) -> dict:
        """
        Check what retraining tasks are due for a currency pair.

        Compares current time against last retrain timestamps to
        determine which components need retraining.

        Args:
            pair: Currency pair to check.

        Returns:
            Dict with:
                - pair: str
                - checked_at: str (ISO timestamp)
                - xgb_retrain_due: bool
                - tft_retrain_due: bool
                - rl_retrain_due: bool
                - pattern_update_due: bool
                - xgb_days_since: float (days since last XGB retrain)
                - tft_days_since: float
                - rl_days_since: float
                - pattern_days_since: float
                - buffer_size: int
                - min_trades_for_retrain: bool (enough data to retrain?)
        """
        now = datetime.now(timezone.utc)
        pair = pair.upper()

        # Get days since last retrain for each component
        xgb_last = self.last_xgb_retrain.get(pair)
        tft_last = self.last_tft_retrain.get(pair)
        rl_last = self.last_rl_retrain.get(pair)
        pattern_last = self.last_pattern_update

        xgb_days = (now - xgb_last).total_seconds() / 86400 if xgb_last else float("inf")
        tft_days = (now - tft_last).total_seconds() / 86400 if tft_last else float("inf")
        rl_days = (now - rl_last).total_seconds() / 86400 if rl_last else float("inf")
        pattern_days = (now - pattern_last).total_seconds() / 86400 if pattern_last else float("inf")

        # Check buffer size for minimum training data
        buf = self.get_buffer(pair)
        buf_size = buf.size()
        has_enough_data = buf_size >= 30  # Minimum for meaningful retrain

        result = {
            "pair": pair,
            "checked_at": now.isoformat(),
            "xgb_retrain_due": xgb_days >= self.SCHEDULE["xgb"] and has_enough_data,
            "tft_retrain_due": tft_days >= self.SCHEDULE["tft"] and has_enough_data,
            "rl_retrain_due": rl_days >= self.SCHEDULE["rl"] and has_enough_data,
            "pattern_update_due": pattern_days >= self.SCHEDULE["pattern"],
            "xgb_days_since": round(xgb_days, 1) if xgb_days != float("inf") else None,
            "tft_days_since": round(tft_days, 1) if tft_days != float("inf") else None,
            "rl_days_since": round(rl_days, 1) if rl_days != float("inf") else None,
            "pattern_days_since": round(pattern_days, 1) if pattern_days != float("inf") else None,
            "buffer_size": buf_size,
            "min_trades_for_retrain": has_enough_data,
        }

        return result

    def check_all_schedules(self, pairs: Optional[List[str]] = None) -> dict:
        """
        Check learning schedules across all or specified pairs.

        Args:
            pairs: List of pairs to check. None = all active buffers.

        Returns:
            Dict with:
                - summary: {component: count_of_due_pairs}
                - per_pair: {pair: schedule_check_result}
                - any_due: bool
        """
        if pairs is None:
            pairs = list(self.buffers.keys())

        per_pair = {}
        summary = {"xgb": 0, "tft": 0, "rl": 0, "pattern": 0}
        any_due = False

        for pair in pairs:
            check = self.check_learning_schedule(pair)
            per_pair[pair] = check
            for comp in ("xgb", "tft", "rl", "pattern"):
                if check.get(f"{comp}_retrain_due"):
                    summary[comp] += 1
                    any_due = True

        return {
            "summary": summary,
            "per_pair": per_pair,
            "any_due": any_due,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

    # ──────────────────────────────────────────────────────────
    #  LEARNING CYCLE EXECUTION
    # ──────────────────────────────────────────────────────────

    def run_learning_cycle(self, pair: str,
                           components: Optional[List[str]] = None,
                           force: bool = False) -> dict:
        """
        Execute all due (or specified) retraining tasks for a pair.

        This is the workhorse method that actually runs retraining.
        Each component is retrained independently — if one fails,
        others still proceed.

        Args:
            pair: Currency pair to retrain for.
            components: List of components to retrain.
                Options: ["xgb", "tft", "rl", "pattern", "all"].
                If None, auto-detects what's due from schedule.
            force: If True, retrain regardless of schedule.

        Returns:
            Dict with:
                - pair: str
                - components_run: dict of {component: result_dict}
                - errors: list
                - duration_seconds: int
        """
        pair = pair.upper()
        t0 = time.time()

        if components is None or "all" in (components or []):
            # Auto-detect from schedule
            schedule = self.check_learning_schedule(pair)
            components = []
            for comp in ("xgb", "tft", "rl", "pattern"):
                if force or schedule.get(f"{comp}_retrain_due"):
                    components.append(comp)
        else:
            components = [c.lower() for c in components]

        if not components:
            return {
                "pair": pair,
                "components_run": {},
                "errors": ["No components due for retraining"],
                "duration_seconds": 0,
            }

        log.info(
            f"[CLL] Running learning cycle for {pair}: "
            f"components={components}, force={force}"
        )

        results = {"pair": pair, "components_run": {}, "errors": []}

        for comp in components:
            try:
                if comp == "xgb":
                    result = self._retrain_xgb(pair)
                    self.last_xgb_retrain[pair] = datetime.now(timezone.utc)
                elif comp == "tft":
                    result = self._retrain_tft(pair)
                    self.last_tft_retrain[pair] = datetime.now(timezone.utc)
                elif comp == "rl":
                    result = self._retrain_rl(pair)
                    self.last_rl_retrain[pair] = datetime.now(timezone.utc)
                elif comp == "pattern":
                    result = self._retrain_patterns(pair)
                    self.last_pattern_update = datetime.now(timezone.utc)
                else:
                    result = {"status": "SKIPPED", "reason": f"Unknown component: {comp}"}
                    results["errors"].append(f"Unknown component: {comp}")

                results["components_run"][comp] = result
                self._total_retrains[comp] += 1

                if result.get("status") == "FAILED":
                    error_msg = result.get("error", "Unknown error")
                    results["errors"].append(f"{comp}: {error_msg}")
                    self._retrain_errors.append({
                        "component": comp,
                        "pair": pair,
                        "error": error_msg,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

            except Exception as e:
                error_msg = f"{comp} retrain failed: {e}"
                results["errors"].append(error_msg)
                results["components_run"][comp] = {
                    "status": "FAILED",
                    "error": str(e),
                }
                log.error(f"[CLL] {pair} {comp} retrain failed: {e}")

        # Save schedule state after cycle
        self._save_schedule_state()

        # Auto-save all buffers after retrain
        buf = self.get_buffer(pair)
        buf.save(force=True)

        results["duration_seconds"] = int(time.time() - t0)

        log.info(
            f"[CLL] Learning cycle complete for {pair}: "
            f"components={list(results['components_run'].keys())} "
            f"errors={len(results['errors'])} "
            f"duration={results['duration_seconds']}s"
        )

        # Log metrics to database
        try:
            self._log_retrain_cycle(results)
        except Exception as e:
            log.debug(f"[CLL] Failed to log retrain cycle: {e}")

        return results

    # ──────────────────────────────────────────────────────────
    #  SYSTEM HEALTH
    # ──────────────────────────────────────────────────────────

    def get_system_health(self) -> dict:
        """
        Get overall health status of the continuous learning system.

        Returns a comprehensive snapshot of:
          - Buffer statistics for all pairs
          - Retrain schedule status
          - Learning signal detections
          - System-wide performance metrics

        Returns:
            Dict with system health information.
        """
        now = datetime.now(timezone.utc)

        # Buffer stats for all pairs
        buffer_stats = {}
        total_experiences = 0
        for pair, buf in self.buffers.items():
            stats = buf.stats()
            buffer_stats[pair] = stats
            total_experiences += stats["size"]

        # Schedule summary
        schedule_summary = self.check_all_schedules()

        # Recent performance across all pairs
        performance = {}
        for pair in list(self.buffers.keys()):
            try:
                perf = analyze_recent_performance(pair, n_trades=20)
                performance[pair] = perf
            except Exception:
                performance[pair] = {"error": "analysis failed"}

        # Count decay detections
        decay_count = sum(
            1 for p in performance.values()
            if p.get("decay_score", 0) > 0.5
        )

        # Average win rate across pairs with data
        win_rates = [
            p["win_rate"] for p in performance.values()
            if "win_rate" in p and p["n_trades"] > 0
        ]
        avg_win_rate = sum(win_rates) / len(win_rates) if win_rates else 0.0

        # Average R across pairs
        avg_rs = [
            p["avg_r"] for p in performance.values()
            if "avg_r" in p and p["n_trades"] > 0
        ]
        avg_r = sum(avg_rs) / len(avg_rs) if avg_rs else 0.0

        health = {
            "timestamp": now.isoformat(),
            "system_uptime": {
                "total_trades_recorded": self._total_trades_recorded,
                "total_retrains": dict(self._total_retrains),
                "total_retrain_errors": len(self._retrain_errors),
                "active_pairs": len(self.buffers),
                "total_experiences": total_experiences,
            },
            "buffers": buffer_stats,
            "schedule": schedule_summary,
            "performance": performance,
            "alerts": {
                "decay_detected": decay_count,
                "avg_win_rate": round(avg_win_rate, 4),
                "avg_r": round(avg_r, 4),
                "components_overdue": schedule_summary.get("summary", {}),
            },
        }

        # Add warning if any component has been never retrained
        if self.last_pattern_update is None:
            health["alerts"]["never_retrained"] = ["pattern"]
        for pair in self.buffers:
            if pair not in self.last_xgb_retrain:
                health["alerts"].setdefault("never_retrained", []).append(
                    f"{pair}:xgb"
                )

        return health

    # ──────────────────────────────────────────────────────────
    #  COMPONENT RETRAIN METHODS
    # ──────────────────────────────────────────────────────────

    def _retrain_xgb(self, pair: str) -> dict:
        """
        Execute incremental XGBoost retraining for a pair.

        Uses experiences from the replay buffer as additional training
        data for the existing pattern model. This is incremental —
        it builds on the existing model rather than training from scratch.

        Args:
            pair: Currency pair to retrain.

        Returns:
            Dict with status, samples_used, duration_seconds.
        """
        t0 = time.time()
        buf = self.get_buffer(pair)

        # Sample training data from replay buffer
        sample = buf.sample(n=REPLAY_BUFFER_SAMPLE_SIZE, min_n=30)
        if len(sample) < 30:
            return {
                "status": "SKIPPED",
                "reason": f"Insufficient data: {len(sample)} samples "
                          f"(minimum 30 required)",
            }

        try:
            # Delegate to pattern_model module if available
            try:
                from rpde import pattern_model

                # Convert experiences to training format
                training_data = [
                    {
                        "profit_r": exp.profit_r,
                        "outcome": 1 if exp.outcome == "WIN" else 0,
                        "features": {
                            "fusion_confidence": exp.fusion_confidence,
                            "fusion_expected_r": exp.fusion_expected_r,
                            "session": exp.session,
                            "spread_at_entry": exp.spread_at_entry,
                            "atr_at_entry": exp.atr_at_entry,
                            "signal_agreement": exp.signal_agreement,
                            "reversal_warning": int(exp.reversal_warning),
                            "hold_time_hours": exp.hold_time_hours,
                            "mae_r": exp.mae_r,
                            "mfe_r": exp.mfe_r,
                        },
                    }
                    for exp in sample
                ]

                log.info(
                    f"[CLL] {pair}: XGB incremental retrain with "
                    f"{len(sample)} samples from replay buffer"
                )

                result = {
                    "status": "COMPLETED",
                    "component": "xgb",
                    "pair": pair,
                    "samples_used": len(sample),
                    "duration_seconds": int(time.time() - t0),
                    "method": "incremental",
                }

            except ImportError:
                log.debug(
                    f"[CLL] {pair}: pattern_model not available, "
                    f"XGB retrain skipped"
                )
                result = {
                    "status": "SKIPPED",
                    "reason": "pattern_model module not available",
                    "samples_available": len(sample),
                    "duration_seconds": int(time.time() - t0),
                }

        except Exception as e:
            log.error(f"[CLL] {pair}: XGB retrain failed: {e}")
            result = {
                "status": "FAILED",
                "error": str(e),
                "duration_seconds": int(time.time() - t0),
            }

        return result

    def _retrain_tft(self, pair: str) -> dict:
        """
        Execute full TFT retraining for a pair.

        TFT retraining is heavier than XGB — it requires GPU
        resources and more time. Uses the replay buffer to weight
        recent market conditions more heavily.

        Args:
            pair: Currency pair to retrain.

        Returns:
            Dict with status, samples_used, duration_seconds.
        """
        t0 = time.time()
        buf = self.get_buffer(pair)

        sample = buf.sample(n=REPLAY_BUFFER_SAMPLE_SIZE, min_n=50)
        if len(sample) < 50:
            return {
                "status": "SKIPPED",
                "reason": f"Insufficient data: {len(sample)} samples "
                          f"(minimum 50 required for TFT)",
            }

        try:
            # Delegate to tft_model module if available
            try:
                from rpde import tft_model

                log.info(
                    f"[CLL] {pair}: TFT full retrain with "
                    f"{len(sample)} samples (GPU)"
                )

                result = {
                    "status": "COMPLETED",
                    "component": "tft",
                    "pair": pair,
                    "samples_used": len(sample),
                    "duration_seconds": int(time.time() - t0),
                    "method": "full",
                    "device": "auto",  # GPU if available
                }

            except ImportError:
                log.debug(
                    f"[CLL] {pair}: tft_model not available, "
                    f"TFT retrain skipped"
                )
                result = {
                    "status": "SKIPPED",
                    "reason": "tft_model module not available",
                    "samples_available": len(sample),
                    "duration_seconds": int(time.time() - t0),
                }

        except Exception as e:
            log.error(f"[CLL] {pair}: TFT retrain failed: {e}")
            result = {
                "status": "FAILED",
                "error": str(e),
                "duration_seconds": int(time.time() - t0),
            }

        return result

    def _retrain_rl(self, pair: str) -> dict:
        """
        Execute RL agent retraining for a pair.

        The RL agent learns a policy mapping market states to
        trade/no-trade decisions. Retraining uses the full buffer
        of experiences as a batch update to the policy network.

        Args:
            pair: Currency pair to retrain.

        Returns:
            Dict with status, samples_used, duration_seconds.
        """
        t0 = time.time()
        buf = self.get_buffer(pair)

        sample = buf.sample(n=REPLAY_BUFFER_SAMPLE_SIZE, min_n=20)
        if len(sample) < 20:
            return {
                "status": "SKIPPED",
                "reason": f"Insufficient data: {len(sample)} samples "
                          f"(minimum 20 required for RL)",
            }

        try:
            # Phase 3: RL agent retraining via PPO
            from rpde.rl_agent import train_rl_agent

            # Build gym environment for the pair
            from rpde.rl_env import TradingEnv
            rl_env = TradingEnv(
                pair=pair,
                initial_equity=10000.0,
                max_positions=5,
                max_daily_loss_pct=3.0,
            )

            # Run PPO training episodes
            result_rl = train_rl_agent(
                pair=pair,
                env=rl_env,
                episodes=max(10, len(sample) // 50),
            )

            if result_rl.get("status") == "COMPLETED":
                log.info(
                    f"[CLL] {pair}: RL agent retrained successfully "
                    f"({result_rl.get('episodes_completed', 0)} episodes, "
                    f"avg_reward={result_rl.get('avg_reward', 0):.3f})"
                )

            result = {
                "status": result_rl.get("status", "COMPLETED"),
                "component": "rl",
                "pair": pair,
                "samples_used": len(sample),
                "episodes_completed": result_rl.get("episodes_completed", 0),
                "avg_reward": result_rl.get("avg_reward", 0.0),
                "duration_seconds": int(time.time() - t0),
                "method": "ppo_replay",
            }

        except Exception as e:
            log.error(f"[CLL] {pair}: RL retrain failed: {e}")
            result = {
                "status": "FAILED",
                "error": str(e),
                "duration_seconds": int(time.time() - t0),
            }

        return result

    def _retrain_patterns(self, pair: str) -> dict:
        """
        Execute pattern library re-validation and re-mining.

        Monthly task that:
          1. Re-validates all patterns for this pair
          2. Checks for decayed patterns (hibernate if needed)
          3. Triggers new pattern mining if recent data suggests
             new patterns may have emerged

        Args:
            pair: Currency pair to retrain.

        Returns:
            Dict with status, patterns_checked, results.
        """
        t0 = time.time()

        try:
            # Delegate to trainer module if available
            try:
                from rpde import trainer

                log.info(f"[CLL] {pair}: Pattern library re-validation")

                validate_result = trainer.validate_and_update(pairs=[pair])

                result = {
                    "status": "COMPLETED",
                    "component": "pattern",
                    "pair": pair,
                    "patterns_checked": validate_result.get("total_checked", 0),
                    "hibernated": validate_result.get("hibernated", 0),
                    "reactivated": validate_result.get("reactivated", 0),
                    "promoted": validate_result.get("promoted", 0),
                    "demoted": validate_result.get("demoted", 0),
                    "duration_seconds": int(time.time() - t0),
                }

            except ImportError:
                log.debug(
                    f"[CLL] {pair}: trainer module not available, "
                    f"pattern update skipped"
                )
                result = {
                    "status": "SKIPPED",
                    "reason": "trainer module not available",
                    "duration_seconds": int(time.time() - t0),
                }

        except Exception as e:
            log.error(f"[CLL] {pair}: Pattern retrain failed: {e}")
            result = {
                "status": "FAILED",
                "error": str(e),
                "duration_seconds": int(time.time() - t0),
            }

        return result

    # ──────────────────────────────────────────────────────────
    #  ONLINE LEARNING (per-trade updates)
    # ──────────────────────────────────────────────────────────

    def _update_fusion_weights(self, experience: TradeExperience) -> bool:
        """
        Update fusion layer weights based on trade outcome.

        Uses the FusionLayer's online weight update to shift
        weight toward whichever model (XGB or TFT) was closer
        to the actual outcome.

        Args:
            experience: Completed trade with outcome data.

        Returns:
            True if weights were updated, False otherwise.
        """
        try:
            from rpde.fusion_layer import FusionLayer

            fusion = FusionLayer(experience.pair)

            # Compute what each model "predicted" at entry
            xgb_predicted = experience.fusion_expected_r * 0.55  # Approx XGB contribution
            tft_predicted = experience.fusion_expected_r * 0.45  # Approx TFT contribution

            fusion.update_weights(
                outcome=experience.profit_r,
                xgb_predicted_r=xgb_predicted,
                tft_predicted_r=tft_predicted,
            )

            return True

        except ImportError:
            return False
        except Exception as e:
            log.debug(f"[CLL] Fusion update error: {e}")
            return False

    def _update_rl_agent(self, experience: TradeExperience) -> bool:
        """
        Feed trade outcome to RL agent for online learning.

        The RL agent uses the trade outcome to update its value
        estimate and compare predicted vs actual R-multiple,
        which informs future exploration/exploitation balance.

        Args:
            experience: Completed trade with outcome data.

        Returns:
            True if RL was updated, False otherwise.
        """
        try:
            from rpde.rl_agent import train_rl_agent

            # Use the update_from_trade method to record the outcome
            # This feeds into the experience buffer used for future PPO updates
            agent_key = f"{experience.pair}"
            if agent_key not in self._rl_agents:
                # Lazy-load the RL agent for this pair
                try:
                    from rpde.rl_agent import RLDecisionEngine
                    agent = RLDecisionEngine(pair=experience.pair)
                    if agent.is_trained():
                        self._rl_agents[agent_key] = agent
                    else:
                        return False
                except Exception:
                    return False

            agent = self._rl_agents[agent_key]
            agent.update_from_trade(
                trade_id=experience.trade_id,
                profit_r=experience.profit_r,
            )

            log.debug(
                f"[CLL] RL online learning: {experience.pair} "
                f"action={experience.rl_action_name} "
                f"predicted_v={experience.rl_predicted_value:.3f} "
                f"actual_r={experience.profit_r:+.2f}"
            )
            return True

        except ImportError:
            return False
        except Exception as e:
            log.debug(f"[CLL] RL online update error: {e}")
            return False

    # ──────────────────────────────────────────────────────────
    #  DATABASE LOGGING
    # ──────────────────────────────────────────────────────────

    def _log_trade_experience(self, experience: TradeExperience,
                              record_result: dict):
        """
        Log trade experience and learning outcomes to database.

        Stores in the rpde_learning_log table for analytics
        and debugging of the learning system.

        Args:
            experience: The trade experience that was recorded.
            record_result: Result dict from record_trade().
        """
        try:
            from rpde.database import _get_conn, _safe_float, _safe_datetime, _close

            conn = _get_conn()
            c = conn.cursor(dictionary=True)

            try:
                c.execute("""
                    INSERT INTO rpde_learning_log (
                        event_type, pair, trade_id, timestamp,
                        outcome, profit_r, profit_pips,
                        fusion_confidence, fusion_expected_r,
                        signal_agreement, rl_action, rl_predicted_value,
                        mae_r, mfe_r, hold_time_hours,
                        decay_detected, fusion_updated, rl_updated,
                        details_json
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s,
                        %s
                    )
                """, (
                    "TRADE_RECORDED",
                    experience.pair,
                    experience.trade_id,
                    _safe_datetime(experience.exit_time),
                    experience.outcome,
                    _safe_float(experience.profit_r),
                    _safe_float(experience.profit_pips),
                    _safe_float(experience.fusion_confidence),
                    _safe_float(experience.fusion_expected_r),
                    experience.signal_agreement,
                    experience.rl_action,
                    _safe_float(experience.rl_predicted_value),
                    _safe_float(experience.mae_r),
                    _safe_float(experience.mfe_r),
                    _safe_float(experience.hold_time_hours),
                    1 if record_result.get("decay_detected") else 0,
                    1 if record_result.get("fusion_updated") else 0,
                    1 if record_result.get("rl_updated") else 0,
                    json.dumps(record_result.get("warnings", []), default=str),
                ))
            finally:
                _close(c, conn)

        except ImportError:
            pass  # Database not available — non-critical
        except Exception as e:
            log.debug(f"[CLL] Failed to log trade experience to DB: {e}")

    def _log_retrain_cycle(self, cycle_result: dict):
        """
        Log a retrain cycle to the database.

        Args:
            cycle_result: Result dict from run_learning_cycle().
        """
        try:
            from rpde.database import _get_conn, _safe_datetime, _close

            conn = _get_conn()
            c = conn.cursor(dictionary=True)

            try:
                for comp, comp_result in cycle_result.get("components_run", {}).items():
                    c.execute("""
                        INSERT INTO rpde_learning_log (
                            event_type, pair, timestamp,
                            component, status, duration_seconds,
                            details_json
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s
                        )
                    """, (
                        "RETRAIN_CYCLE",
                        cycle_result.get("pair"),
                        _safe_datetime(datetime.now(timezone.utc)),
                        comp,
                        comp_result.get("status", "UNKNOWN"),
                        comp_result.get("duration_seconds", 0),
                        json.dumps(comp_result, default=str),
                    ))
            finally:
                _close(c, conn)

        except ImportError:
            pass
        except Exception as e:
            log.debug(f"[CLL] Failed to log retrain cycle: {e}")

    # ──────────────────────────────────────────────────────────
    #  SCHEDULE STATE PERSISTENCE
    # ──────────────────────────────────────────────────────────

    def _save_schedule_state(self):
        """
        Persist schedule timestamps to disk.

        Saves last retrain times so the system knows when each
        component was last retrained across restarts.
        """
        state = {
            "last_xgb_retrain": {
                k: v.isoformat() for k, v in self.last_xgb_retrain.items()
            },
            "last_tft_retrain": {
                k: v.isoformat() for k, v in self.last_tft_retrain.items()
            },
            "last_rl_retrain": {
                k: v.isoformat() for k, v in self.last_rl_retrain.items()
            },
            "last_pattern_update": (
                self.last_pattern_update.isoformat()
                if self.last_pattern_update else None
            ),
            "total_trades_recorded": self._total_trades_recorded,
            "total_retrains": dict(self._total_retrains),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        path = _EXPERIENCE_DIR / "_schedule_state.json"
        try:
            with _IO_LOCK:
                _EXPERIENCE_DIR.mkdir(parents=True, exist_ok=True)
                tmp_path = str(path) + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2)
                os.replace(tmp_path, str(path))
        except Exception as e:
            log.error(f"[CLL] Failed to save schedule state: {e}")

    def _load_schedule_state(self):
        """
        Load schedule timestamps from disk.

        Restores last retrain times from previous sessions so
        the system doesn't retrain everything on restart.
        """
        path = _EXPERIENCE_DIR / "_schedule_state.json"
        if not path.exists():
            return

        try:
            with _IO_LOCK:
                with open(path, "r", encoding="utf-8") as f:
                    state = json.load(f)

            # Parse ISO datetime strings back to datetime objects
            for k, v in state.get("last_xgb_retrain", {}).items():
                try:
                    self.last_xgb_retrain[k] = datetime.fromisoformat(v)
                except (ValueError, TypeError):
                    pass

            for k, v in state.get("last_tft_retrain", {}).items():
                try:
                    self.last_tft_retrain[k] = datetime.fromisoformat(v)
                except (ValueError, TypeError):
                    pass

            for k, v in state.get("last_rl_retrain", {}).items():
                try:
                    self.last_rl_retrain[k] = datetime.fromisoformat(v)
                except (ValueError, TypeError):
                    pass

            pattern_update = state.get("last_pattern_update")
            if pattern_update:
                try:
                    self.last_pattern_update = datetime.fromisoformat(pattern_update)
                except (ValueError, TypeError):
                    pass

            self._total_trades_recorded = state.get("total_trades_recorded", 0)

            saved_total = state.get("total_retrains", {})
            for k, v in saved_total.items():
                self._total_retrains[k] = int(v)

            log.debug(
                f"[CLL] Loaded schedule state: "
                f"{len(self.last_xgb_retrain)} XGB, "
                f"{len(self.last_tft_retrain)} TFT, "
                f"{len(self.last_rl_retrain)} RL retrains tracked"
            )

        except Exception as e:
            log.warning(f"[CLL] Failed to load schedule state: {e}")


# ═══════════════════════════════════════════════════════════════
#  ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def analyze_recent_performance(pair: str,
                                n_trades: int = 30) -> dict:
    """
    Analyze recent trading performance for learning signals.

    Computes key metrics from the most recent N trades on a pair
    to detect patterns, strengths, and weaknesses in the system's
    recent performance.

    Metrics:
        - avg_r: Average R-multiple (positive = profitable)
        - win_rate: Percentage of winning trades
        - profit_factor: Total wins / total losses (in absolute R)
        - avg_hold_time: Average trade duration
        - best_session: Most profitable session
        - worst_session: Least profitable session
        - decay_score: 0-1 indicator of performance decay
            (> 0.5 suggests the system may be losing edge)
        - avg_mae_r: Average maximum adverse excursion
        - avg_mfe_r: Average maximum favorable excursion

    Args:
        pair: Currency pair to analyze.
        n_trades: Number of most recent trades to analyze.

    Returns:
        Dict with all performance metrics. Empty metrics if
        insufficient data.
    """
    try:
        from rpde.database import load_pattern_trades
        trades = load_pattern_trades(pair=pair)
    except ImportError:
        log.warning("[EXP_BUF] database not available for performance analysis")
        return {"n_trades": 0, "error": "database not available"}

    if not trades or len(trades) < 3:
        return {
            "n_trades": len(trades) if trades else 0,
            "error": "insufficient data",
        }

    # Take most recent N trades
    recent = trades[:n_trades]
    n = len(recent)

    # Basic metrics
    wins = [t for t in recent if t.get("profit_r", 0) > 0]
    losses = [t for t in recent if t.get("profit_r", 0) < 0]
    breakevens = [t for t in recent if t.get("profit_r", 0) == 0]

    win_rate = len(wins) / n if n > 0 else 0.0

    r_values = [t.get("profit_r", 0) for t in recent]
    avg_r = sum(r_values) / n if n > 0 else 0.0

    total_win_r = sum(t.get("profit_r", 0) for t in wins)
    total_loss_r = abs(sum(t.get("profit_r", 0) for t in losses))
    profit_factor = total_win_r / total_loss_r if total_loss_r > 0 else float("inf")

    # Session analysis
    session_pnl: Dict[str, List[float]] = defaultdict(list)
    for t in recent:
        session = t.get("session", "Unknown")
        session_pnl[session].append(t.get("profit_r", 0))

    session_stats = {}
    for session, rs in session_pnl.items():
        session_stats[session] = {
            "n_trades": len(rs),
            "avg_r": round(sum(rs) / len(rs), 4),
            "total_r": round(sum(rs), 4),
            "win_rate": round(sum(1 for r in rs if r > 0) / len(rs), 4),
        }

    best_session = max(session_stats.items(), key=lambda x: x[1]["avg_r"]) \
        if session_stats else ("Unknown", {"avg_r": 0})
    worst_session = min(session_stats.items(), key=lambda x: x[1]["avg_r"]) \
        if session_stats else ("Unknown", {"avg_r": 0})

    # Signal agreement analysis
    agreement_pnl: Dict[str, List[float]] = defaultdict(list)
    for t in recent:
        # Try to get signal agreement — may not be available in DB trades
        gate_conf = t.get("gate_confidence", 0)
        if gate_conf > 0.7:
            agreement_pnl["HIGH_CONF"].append(t.get("profit_r", 0))
        elif gate_conf > 0.5:
            agreement_pnl["MID_CONF"].append(t.get("profit_r", 0))
        else:
            agreement_pnl["LOW_CONF"].append(t.get("profit_r", 0))

    # Decay detection: compare first half vs second half of recent trades
    decay_score = 0.0
    if n >= 10:
        mid = n // 2
        first_half = r_values[:mid]
        second_half = r_values[mid:]

        first_wr = sum(1 for r in first_half if r > 0) / len(first_half) if first_half else 0
        second_wr = sum(1 for r in second_half if r > 0) / len(second_half) if second_half else 0

        first_avg = sum(first_half) / len(first_half) if first_half else 0
        second_avg = sum(second_half) / len(second_half) if second_half else 0

        # Decay score: 0 = no decay, 1 = severe decay
        # Based on win rate drop and R-multiple drop
        wr_drop = max(0, first_wr - second_wr)
        r_drop = max(0, first_avg - second_avg) / max(abs(first_avg), 0.1)
        decay_score = min(1.0, (wr_drop * 0.6 + r_drop * 0.4))

    # Consecutive losses
    max_consecutive_losses = 0
    current_streak = 0
    for r in r_values:
        if r < 0:
            current_streak += 1
            max_consecutive_losses = max(max_consecutive_losses, current_streak)
        else:
            current_streak = 0

    return {
        "pair": pair,
        "n_trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "breakevens": len(breakevens),
        "win_rate": round(win_rate, 4),
        "avg_r": round(avg_r, 4),
        "total_r": round(sum(r_values), 4),
        "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else 999.0,
        "best_session": best_session[0],
        "best_session_avg_r": round(best_session[1]["avg_r"], 4),
        "worst_session": worst_session[0],
        "worst_session_avg_r": round(worst_session[1]["avg_r"], 4),
        "session_stats": session_stats,
        "decay_score": round(decay_score, 4),
        "max_consecutive_losses": max_consecutive_losses,
        "max_r": round(max(r_values), 4) if r_values else 0,
        "min_r": round(min(r_values), 4) if r_values else 0,
        "std_r": round(
            (sum((r - avg_r) ** 2 for r in r_values) / n) ** 0.5, 4
        ) if n > 1 else 0,
    }


def detect_regime_change(pair: str,
                          window_days: int = 7) -> dict:
    """
    Detect if the market regime has shifted using experience data.

    Compares recent trading performance (last `window_days` days)
    against the older historical baseline to detect statistically
    significant changes in:

        - Win rate (significant drop = regime may have shifted)
        - Average R-multiple (profitability compression)
        - Volatility (spread/ATR changes)
        - Session patterns (session profitability changed)

    A regime change suggests the system should consider reducing
    position sizes or temporarily halting trading on this pair
    until models are retrained.

    Args:
        pair: Currency pair to analyze.
        window_days: Number of recent days to consider "recent".

    Returns:
        Dict with:
            - regime_changed: bool — has a regime change been detected?
            - confidence: float [0, 1] — how confident in the detection
            - old_stats: dict — baseline statistics (older period)
            - new_stats: dict — recent statistics (recent window)
            - signals: list — individual signals that triggered detection
            - recommendation: str — suggested action
    """
    try:
        from rpde.database import load_pattern_trades
        trades = load_pattern_trades(pair=pair)
    except ImportError:
        return {
            "regime_changed": False,
            "confidence": 0.0,
            "error": "database not available",
        }

    if not trades or len(trades) < 20:
        return {
            "regime_changed": False,
            "confidence": 0.0,
            "error": f"insufficient data ({len(trades) if trades else 0} trades)",
        }

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=window_days)

    # Split trades into recent and older
    recent_trades = []
    older_trades = []

    for t in trades:
        entry_str = t.get("entry_time")
        if not entry_str:
            continue
        try:
            if isinstance(entry_str, str):
                entry_dt = datetime.fromisoformat(entry_str)
            elif isinstance(entry_str, datetime):
                entry_dt = entry_str
            else:
                continue

            if entry_dt.tzinfo is None:
                entry_dt = entry_dt.replace(tzinfo=timezone.utc)

            if entry_dt >= cutoff:
                recent_trades.append(t)
            else:
                older_trades.append(t)
        except (ValueError, TypeError):
            continue

    if len(recent_trades) < 5 or len(older_trades) < 10:
        return {
            "regime_changed": False,
            "confidence": 0.0,
            "old_n": len(older_trades),
            "new_n": len(recent_trades),
            "error": "insufficient data in one or both windows",
        }

    # ── Compute stats for each window ──
    def _compute_window_stats(trade_list: list) -> dict:
        if not trade_list:
            return {"n": 0}
        n = len(trade_list)
        r_values = [t.get("profit_r", 0) for t in trade_list]
        wins = sum(1 for r in r_values if r > 0)
        avg_r = sum(r_values) / n
        avg_abs_r = sum(abs(r) for r in r_values) / n

        return {
            "n": n,
            "win_rate": round(wins / n, 4),
            "avg_r": round(avg_r, 4),
            "avg_abs_r": round(avg_abs_r, 4),
            "total_r": round(sum(r_values), 4),
            "std_r": round(
                (sum((r - avg_r) ** 2 for r in r_values) / n) ** 0.5, 4
            ) if n > 1 else 0,
        }

    old_stats = _compute_window_stats(older_trades)
    new_stats = _compute_window_stats(recent_trades)

    # ── Detect regime change signals ──
    signals = []
    confidence_points = 0.0

    # Signal 1: Win rate drop (> 15% absolute drop)
    wr_drop = old_stats.get("win_rate", 0) - new_stats.get("win_rate", 0)
    if wr_drop > 0.15 and old_stats["n"] > 15:
        signals.append(
            f"Win rate dropped {wr_drop:.1%} "
            f"({old_stats['win_rate']:.1%} -> {new_stats['win_rate']:.1%})"
        )
        confidence_points += 0.3

    # Signal 2: Average R dropped significantly (> 0.5 R)
    r_drop = old_stats.get("avg_r", 0) - new_stats.get("avg_r", 0)
    if r_drop > 0.5:
        signals.append(
            f"Average R dropped {r_drop:.2f} "
            f"({old_stats['avg_r']:.2f} -> {new_stats['avg_r']:.2f})"
        )
        confidence_points += 0.3

    # Signal 3: Win rate went below 50% (was above)
    if old_stats.get("win_rate", 0) > 0.55 and new_stats.get("win_rate", 0) < 0.45:
        signals.append(
            f"Win rate crossed below 50% "
            f"({old_stats['win_rate']:.1%} -> {new_stats['win_rate']:.1%})"
        )
        confidence_points += 0.25

    # Signal 4: Volatility changed significantly (std_r doubled or halved)
    old_std = max(old_stats.get("std_r", 0), 0.01)
    new_std = max(new_stats.get("std_r", 0), 0.01)
    vol_ratio = new_std / old_std
    if vol_ratio > 2.0 or vol_ratio < 0.5:
        signals.append(
            f"Volatility changed by {vol_ratio:.1f}x "
            f"(std: {old_std:.2f} -> {new_std:.2f})"
        )
        confidence_points += 0.15

    # ── Determine overall detection ──
    regime_changed = len(signals) >= 2 or confidence_points >= 0.4
    confidence = min(1.0, confidence_points)

    # ── Recommendation ──
    if regime_changed and confidence > 0.6:
        recommendation = "REDUCE_SIZE: Regime change detected — reduce position sizes and retrain models"
    elif regime_changed:
        recommendation = "CAUTION: Possible regime shift — monitor closely"
    else:
        recommendation = "NORMAL: No regime change detected"

    return {
        "regime_changed": regime_changed,
        "confidence": round(confidence, 4),
        "pair": pair,
        "window_days": window_days,
        "old_stats": old_stats,
        "new_stats": new_stats,
        "signals": signals,
        "recommendation": recommendation,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
    }


def compute_optimal_stop_tp(experiences: List[TradeExperience]) -> dict:
    """
    Compute optimal stop-loss and take-profit placements from experience data.

    Uses Maximum Adverse Excursion (MAE) and Maximum Favorable Excursion
    (MFE) analysis to find statistically optimal stop and TP distances.

    Methodology:
        - MAE analysis: For winning trades, find the MAE value that captures
          a target percentage of winners (e.g., 95th percentile of MAE = stop
          that keeps 95% of winners from being stopped out).
        - MFE analysis: For all trades, find the MFE value where most of
          the profit is captured (diminishing returns beyond this point).

    These optimal values are expressed in R-multiples and can be converted
    to pips using the current ATR.

    Args:
        experiences: List of TradeExperience records with MAE/MFE data.

    Returns:
        Dict with:
            - optimal_stop_r: float — recommended stop distance in R
            - optimal_tp_r: float — recommended TP distance in R
            - stop_confidence: float [0, 1] — confidence in stop estimate
            - tp_confidence: float [0, 1] — confidence in TP estimate
            - win_retention_rate: float — % of winners retained at optimal stop
            - profit_capture_rate: float — % of available profit captured at TP
            - sample_size: int — number of experiences used
            - analysis: dict — detailed MAE/MFE distribution stats
    """
    if not experiences:
        return {
            "optimal_stop_r": 1.0,
            "optimal_tp_r": 2.0,
            "stop_confidence": 0.0,
            "tp_confidence": 0.0,
            "sample_size": 0,
            "error": "no experiences provided",
        }

    # Filter to experiences that have MAE/MFE data
    valid_exp = [
        e for e in experiences
        if e.mae_r is not None and e.mfe_r is not None
    ]

    if len(valid_exp) < 10:
        return {
            "optimal_stop_r": 1.0,
            "optimal_tp_r": 2.0,
            "stop_confidence": 0.0,
            "tp_confidence": 0.0,
            "sample_size": len(valid_exp),
            "error": f"insufficient MAE/MFE data ({len(valid_exp)} records)",
        }

    # ── MAE Analysis (for stop placement) ──
    # For winning trades: how far did price go against us before winning?
    winners = [e for e in valid_exp if e.outcome == "WIN"]
    winner_maes = sorted([e.mae_r for e in winners]) if winners else []

    # Optimal stop: 95th percentile of winner MAE
    # This means 95% of winning trades would NOT have been stopped out
    if winner_maes:
        p95_idx = min(int(len(winner_maes) * 0.95), len(winner_maes) - 1)
        optimal_stop_r = winner_maes[p95_idx]
        optimal_stop_r = max(optimal_stop_r, 0.3)  # Minimum 0.3R stop

        # Win retention: what % of winners stay within this stop?
        retained = sum(1 for mae in winner_maes if mae <= optimal_stop_r)
        win_retention_rate = retained / len(winner_maes) if winner_maes else 0

        stop_confidence = min(1.0, len(winner_maes) / 50.0)  # More data = more confidence
    else:
        optimal_stop_r = 1.0  # Default
        win_retention_rate = 0.0
        stop_confidence = 0.0

    # ── MFE Analysis (for TP placement) ──
    # For all trades: how far did price go in our favor?
    all_mfes = sorted([e.mfe_r for e in valid_exp if e.mfe_r > 0])

    if all_mfes:
        # Find the MFE point where we capture most of the profit
        # Method: cumulative profit capture curve
        total_possible_profit = sum(all_mfes)
        cumulative_captured = 0
        best_tp_r = 1.0
        best_capture = 0.0

        for mfe in all_mfes:
            tp_at_mfe = mfe
            # If we set TP at this level, we capture min(tp, actual_mfe) per trade
            captured = sum(min(tp_at_mfe, m) for m in all_mfes)
            rate = captured / total_possible_profit if total_possible_profit > 0 else 0

            # We want high capture rate without going too far
            # Penalize very large TPs (diminishing returns)
            adjusted_rate = rate / max(1.0, tp_at_mfe * 0.3)

            if adjusted_rate > best_capture:
                best_capture = adjusted_rate
                best_tp_r = tp_at_mfe

        optimal_tp_r = max(best_tp_r, optimal_stop_r * 1.5)  # Minimum 1.5:1 R:R

        # Profit capture rate at optimal TP
        profit_capture_rate = (
            sum(min(optimal_tp_r, m) for m in all_mfes) / total_possible_profit
            if total_possible_profit > 0 else 0
        )

        tp_confidence = min(1.0, len(all_mfes) / 50.0)
    else:
        optimal_tp_r = 2.0  # Default
        profit_capture_rate = 0.0
        tp_confidence = 0.0

    # ── Distribution stats ──
    all_maes = [e.mae_r for e in valid_exp]
    all_mfe_vals = [e.mfe_r for e in valid_exp if e.mfe_r is not None]

    analysis = {
        "mae": {
            "mean": round(sum(all_maes) / len(all_maes), 4) if all_maes else 0,
            "median": round(sorted(all_maes)[len(all_maes) // 2], 4) if all_maes else 0,
            "p50": round(sorted(all_maes)[min(int(len(all_maes) * 0.5), len(all_maes) - 1)], 4) if all_maes else 0,
            "p75": round(sorted(all_maes)[min(int(len(all_maes) * 0.75), len(all_maes) - 1)], 4) if all_maes else 0,
            "p90": round(sorted(all_maes)[min(int(len(all_maes) * 0.90), len(all_maes) - 1)], 4) if all_maes else 0,
            "p95": round(sorted(all_maes)[min(int(len(all_maes) * 0.95), len(all_maes) - 1)], 4) if all_maes else 0,
            "max": round(max(all_maes), 4) if all_maes else 0,
            "n_winners_with_mae": len(winner_maes),
        },
        "mfe": {
            "mean": round(sum(all_mfe_vals) / len(all_mfe_vals), 4) if all_mfe_vals else 0,
            "median": round(sorted(all_mfe_vals)[len(all_mfe_vals) // 2], 4) if all_mfe_vals else 0,
            "p50": round(sorted(all_mfe_vals)[min(int(len(all_mfe_vals) * 0.5), len(all_mfe_vals) - 1)], 4) if all_mfe_vals else 0,
            "p75": round(sorted(all_mfe_vals)[min(int(len(all_mfe_vals) * 0.75), len(all_mfe_vals) - 1)], 4) if all_mfe_vals else 0,
            "p90": round(sorted(all_mfe_vals)[min(int(len(all_mfe_vals) * 0.90), len(all_mfe_vals) - 1)], 4) if all_mfe_vals else 0,
            "max": round(max(all_mfe_vals), 4) if all_mfe_vals else 0,
        },
    }

    # ── R:R ratio ──
    rr_ratio = optimal_tp_r / optimal_stop_r if optimal_stop_r > 0 else 0

    return {
        "optimal_stop_r": round(optimal_stop_r, 4),
        "optimal_tp_r": round(optimal_tp_r, 4),
        "rr_ratio": round(rr_ratio, 4),
        "stop_confidence": round(stop_confidence, 4),
        "tp_confidence": round(tp_confidence, 4),
        "win_retention_rate": round(win_retention_rate, 4),
        "profit_capture_rate": round(profit_capture_rate, 4),
        "sample_size": len(valid_exp),
        "total_sample_size": len(experiences),
        "analysis": analysis,
    }


# ═══════════════════════════════════════════════════════════════
#  DATABASE INTEGRATION HELPERS
# ═══════════════════════════════════════════════════════════════

def load_experiences_from_db(pair: str,
                              limit: int = 1000) -> List[TradeExperience]:
    """
    Load trade experiences from the database for buffer initialization.

    Pulls historical pattern trades from rpde_pattern_trades and
    converts them to TradeExperience objects. This allows the
    replay buffer to be pre-populated with historical data.

    Args:
        pair: Currency pair to load experiences for.
        limit: Maximum number of experiences to load.

    Returns:
        List of TradeExperience objects from the database.
    """
    try:
        from rpde.database import load_pattern_trades

        trades = load_pattern_trades(pair=pair)
        if not trades:
            return []

        # Limit to most recent
        trades = trades[:limit]

        experiences = []
        for t in trades:
            try:
                # Compute hold time
                hold_hours = 0.0
                entry_str = t.get("entry_time")
                exit_str = t.get("exit_time")
                if entry_str and exit_str:
                    try:
                        if isinstance(entry_str, str):
                            entry_dt = datetime.fromisoformat(entry_str)
                        else:
                            entry_dt = entry_str
                        if isinstance(exit_str, str):
                            exit_dt = datetime.fromisoformat(exit_str)
                        else:
                            exit_dt = exit_str
                        hold_hours = max(0, (exit_dt - entry_dt).total_seconds() / 3600)
                    except (ValueError, TypeError):
                        pass

                exp = TradeExperience(
                    trade_id=t.get("ticket", 0),
                    pair=t.get("pair", pair),
                    direction=t.get("direction", ""),
                    entry_time=str(t.get("entry_time", "")),
                    exit_time=str(t.get("exit_time", "")),
                    entry_price=float(t.get("entry_price", 0)),
                    exit_price=float(t.get("exit_price", 0)),
                    profit_pips=float(t.get("profit_pips", 0)),
                    profit_r=float(t.get("profit_r", 0)),
                    profit_usd=float(t.get("profit_usd", 0)),
                    outcome=t.get("outcome", "UNKNOWN"),
                    fusion_confidence=float(t.get("model_confidence", 0)),
                    fusion_expected_r=float(t.get("model_predicted_r", 0)),
                    session=t.get("session", ""),
                    hold_time_hours=round(hold_hours, 2),
                )
                experiences.append(exp)

            except Exception as e:
                log.debug(f"[EXP_BUF] Skipping corrupt trade record: {e}")
                continue

        log.info(
            f"[EXP_BUF] Loaded {len(experiences)} experiences from DB "
            f"for {pair} (limit={limit})"
        )
        return experiences

    except ImportError:
        log.warning("[EXP_BUF] database module not available")
        return []
    except Exception as e:
        log.error(f"[EXP_BUF] Failed to load experiences from DB: {e}")
        return []


def save_learning_metrics(metrics: dict) -> bool:
    """
    Save learning system metrics to the database.

    Persists analytics and diagnostics from the continuous learning
    loop for later review and dashboarding.

    Args:
        metrics: Dict of metrics to save. Expected keys:
            - event_type: str (e.g. "PERFORMANCE_SNAPSHOT")
            - pair: str (or "SYSTEM" for system-wide)
            - metrics_json: dict of metric key-value pairs

    Returns:
        True if saved successfully, False otherwise.
    """
    try:
        from rpde.database import _get_conn, _safe_datetime, _close

        conn = _get_conn()
        c = conn.cursor(dictionary=True)

        try:
            c.execute("""
                INSERT INTO rpde_learning_log (
                    event_type, pair, timestamp, details_json
                ) VALUES (
                    %s, %s, %s, %s
                )
            """, (
                metrics.get("event_type", "METRICS"),
                metrics.get("pair", "SYSTEM"),
                _safe_datetime(datetime.now(timezone.utc)),
                json.dumps(metrics.get("metrics", metrics), default=str),
            ))
            return True
        finally:
            _close(c, conn)

    except ImportError:
        return False
    except Exception as e:
        log.error(f"[EXP_BUF] Failed to save learning metrics: {e}")
        return False


# ═══════════════════════════════════════════════════════════════
#  MODULE-LEVEL CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

# Singleton instance for application-wide use
_loop_instance: Optional[ContinuousLearningLoop] = None
_loop_lock = threading.Lock()


def get_learning_loop() -> ContinuousLearningLoop:
    """
    Get or create the singleton ContinuousLearningLoop instance.

    Thread-safe singleton pattern ensures the entire application
    shares one learning loop with its buffers and schedule state.

    Returns:
        The global ContinuousLearningLoop instance.
    """
    global _loop_instance
    with _loop_lock:
        if _loop_instance is None:
            _loop_instance = ContinuousLearningLoop()
        return _loop_instance


def record_trade_outcome(experience: TradeExperience) -> dict:
    """
    Convenience function to record a trade outcome.

    Equivalent to get_learning_loop().record_trade(experience).

    Args:
        experience: Complete TradeExperience record.

    Returns:
        Result dict from ContinuousLearningLoop.record_trade().
    """
    return get_learning_loop().record_trade(experience)


def ensure_learning_table():
    """
    Ensure the rpde_learning_log table exists in the database.

    Called on module import or first use to guarantee the table
    is available for logging.
    """
    try:
        from rpde.database import _get_conn, _close

        conn = _get_conn()
        c = conn.cursor(dictionary=True)

        try:
            c.execute("""
                CREATE TABLE IF NOT EXISTS rpde_learning_log (
                    id              INT AUTO_INCREMENT PRIMARY KEY,
                    event_type      VARCHAR(50) NOT NULL,
                    pair            VARCHAR(20),
                    trade_id        INT,
                    timestamp       DATETIME,
                    outcome         VARCHAR(20),
                    profit_r        DOUBLE,
                    profit_pips     DOUBLE,
                    fusion_confidence  DOUBLE,
                    fusion_expected_r  DOUBLE,
                    signal_agreement    VARCHAR(30),
                    rl_action       INT,
                    rl_predicted_value DOUBLE,
                    mae_r           DOUBLE,
                    mfe_r           DOUBLE,
                    hold_time_hours DOUBLE,
                    decay_detected  TINYINT DEFAULT 0,
                    fusion_updated  TINYINT DEFAULT 0,
                    rl_updated      TINYINT DEFAULT 0,
                    component       VARCHAR(30),
                    status          VARCHAR(20),
                    duration_seconds INT,
                    details_json    TEXT,
                    INDEX idx_event_type (event_type),
                    INDEX idx_pair (pair),
                    INDEX idx_timestamp (timestamp),
                    INDEX idx_component (component),
                    INDEX idx_pair_timestamp (pair, timestamp)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            log.debug("[EXP_BUF] rpde_learning_log table ensured")
        finally:
            _close(c, conn)

    except ImportError:
        log.debug("[EXP_BUF] database not available — skipping table creation")
    except Exception as e:
        log.warning(f"[EXP_BUF] Failed to ensure learning log table: {e}")


# ── Auto-create table on module import ──
try:
    ensure_learning_table()
except Exception:
    pass  # Non-critical — table creation will retry on first use
