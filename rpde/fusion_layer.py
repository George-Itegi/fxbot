# =============================================================
# rpde/fusion_layer.py  --  Pattern Fusion Layer (Phase 2)
#
# PURPOSE: Combine XGBoost (Phase 1) and TFT (Phase 2) predictions
# with Pattern Library matches into a single unified trading signal.
#
# ARCHITECTURE:
#
#   XGBoost result (PatternModel.predict)
#       │  predicted_r, confidence, direction, is_pattern
#       ▼
#   ┌──────────────────────────────────────────────────────┐
#   │              FUSION LAYER (FusionLayer)              │
#   │                                                      │
#   │  TFT result (TFTModelManager.predict)                │
#   │      │  candle_pattern_match, momentum_score,        │
#   │      │  reversal_probability                         │
#   │      ▼                                              │
#   │  Pattern Library match (PatternGate._match_pattern) │
#   │      │  win_rate, expected_r, match_score, tier     │
#   │      ▼                                              │
#   │                                                      │
#   │  1. Derive TFT direction & confidence               │
#   │  2. Check 3-way signal agreement                    │
#   │  3. Apply direction boost / penalty                 │
#   │  4. Compute weighted confidence & expected R        │
#   │  5. Apply reversal warning from TFT                 │
#   │  6. Emit TAKE / CAUTION / SKIP recommendation       │
#   └──────────────────────┬───────────────────────────────┘
#                          │
#                          ▼
#   Fused output:
#     combined_confidence, combined_expected_r, direction,
#     tft_contribution, signal_agreement, reversal_warning,
#     recommendation, weights, reason
#
# WEIGHT LEARNING:
#   - Per-pair persistent weights stored in JSON
#   - Online EMA update after each trade outcome
#   - Clamp to [FUSION_MIN_WEIGHT, FUSION_MAX_WEIGHT]
#   - Atomic file writes for thread-safety
# =============================================================

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

from core.logger import get_logger
from rpde.config import (
    FUSION_DEFAULT_XGB_WEIGHT,
    FUSION_DEFAULT_TFT_WEIGHT,
    FUSION_META_LR,
    FUSION_WEIGHT_SMOOTHING,
    FUSION_MIN_WEIGHT,
    FUSION_MAX_WEIGHT,
    FUSION_DIRECTION_AGREE_BOOST,
    FUSION_DIRECTION_DISAGREE_PENALTY,
    TFT_MIN_PATTERN_MATCH,
    TFT_MIN_MOMENTUM_SCORE,
    TFT_REVERSAL_THRESHOLD,
    GATE_MIN_CONFIDENCE,
    GATE_MIN_PREDICTED_R,
)

log = get_logger(__name__)

# Base directory for per-pair fusion weight files
_FUSION_MODELS_DIR = Path(__file__).resolve().parent / "models" / "fusion"

# File I/O lock — ensures atomic reads/writes across threads
_IO_LOCK = threading.Lock()


# ════════════════════════════════════════════════════════════════
#  NEUTRAL RESULT HELPER
# ════════════════════════════════════════════════════════════════

def _neutral_result(reason: str) -> dict:
    """Return a fully-populated SKIP result when fusion cannot proceed."""
    return {
        "combined_confidence": 0.0,
        "combined_expected_r": 0.0,
        "direction": None,
        "tft_contribution": 0.0,
        "signal_agreement": "DISAGREE",
        "reversal_warning": False,
        "recommendation": "SKIP",
        "weights": {
            "xgb_weight": FUSION_DEFAULT_XGB_WEIGHT,
            "tft_weight": FUSION_DEFAULT_TFT_WEIGHT,
        },
        "reason": reason,
        "xgb_available": False,
        "tft_available": False,
        "pattern_available": False,
    }


# ════════════════════════════════════════════════════════════════
#  FUSION LAYER
# ════════════════════════════════════════════════════════════════

class FusionLayer:
    """
    Combines XGBoost and TFT predictions into a single trading signal.

    The fusion layer is *lightweight* — no ML training, just simple
    per-pair weight adjustment via online EMA updates.

    Gracefully degrades:
      - TFT missing  -> XGB-only mode  (xgb_weight = 1.0)
      - XGB missing  -> TFT-only mode  (tft_weight = 1.0)
      - Both missing -> neutral SKIP

    Usage::

        fusion = FusionLayer("EURJPY")
        result = fusion.fuse(xgb_result, tft_result, pattern_match)

    After a trade closes::

        fusion.update_weights(outcome_r)
    """

    def __init__(self, pair: str):
        """
        Initialise fusion layer for a currency pair.

        Loads or creates per-pair weight file at:
            rpde/models/fusion/{PAIR}_fusion.json

        Args:
            pair: Uppercase currency pair string (e.g. "EURJPY").
        """
        self.pair = pair.upper()
        self._weight_path = _FUSION_MODELS_DIR / f"{self.pair}_fusion.json"

        # Internal weight state
        self._xgb_weight: float = FUSION_DEFAULT_XGB_WEIGHT
        self._tft_weight: float = FUSION_DEFAULT_TFT_WEIGHT
        self._n_updates: int = 0
        self._last_update: Optional[str] = None
        self._performance_history: list = []  # last N (outcome, xgb_r, tft_r)

        # Load persisted weights (if any)
        self.load_weights()

    # ──────────────────────────────────────────────────────────
    #  PUBLIC API
    # ──────────────────────────────────────────────────────────

    def fuse(self,
             xgb_result: Optional[dict],
             tft_result: Optional[dict],
             pattern_match: Optional[dict]) -> dict:
        """
        Main fusion entry point.

        Combines XGBoost, TFT, and Pattern Library signals into a
        unified decision with confidence, expected R, direction, and
        a TAKE / CAUTION / SKIP recommendation.

        Args:
            xgb_result:  Dict from PatternModel.predict():
                predicted_r, confidence, direction, is_pattern
            tft_result:  Dict from TFTModelManager.predict():
                candle_pattern_match, momentum_score, reversal_probability
            pattern_match: Dict from PatternGate._match_pattern_library():
                win_rate, expected_r, match_score, tier, direction

        Returns:
            Full fusion result dict (see module docstring for keys).
        """
        # ── 1. Normalise inputs (handle None / missing keys) ──
        xgb = self._safe_dict(xgb_result)
        tft = self._safe_dict(tft_result)
        plib = self._safe_dict(pattern_match)

        xgb_available = bool(xgb.get("is_pattern")) or (xgb.get("confidence", 0) > 0.1)
        tft_available = (
            tft.get("candle_pattern_match") is not None
            and tft.get("momentum_score") is not None
        )
        pattern_available = (
            plib.get("win_rate") is not None
            and (plib.get("match_score", 0) or 0) >= 0.3
        )

        # ── 2. Graceful fallbacks ──
        if not xgb_available and not tft_available:
            log.debug(f"[FUSION] {self.pair}: no signals available -> SKIP")
            return _neutral_result("No XGB or TFT signals available")

        if not xgb_available:
            return self._tft_only_fusion(tft, plib)

        if not tft_available:
            return self._xgb_only_fusion(xgb, plib)

        # ── 3. Derive TFT direction & confidence ──
        tft_signal = self._derive_tft_signal(tft)
        tft_dir = tft_signal["direction"]
        tft_conf = tft_signal["confidence"]
        tft_r = tft_signal["derived_r"]

        xgb_dir = xgb.get("direction")
        xgb_conf = xgb.get("confidence", 0.0)
        xgb_r = xgb.get("predicted_r", 0.0)
        plib_dir = plib.get("direction")

        # ── 4. Determine effective weights ──
        xgb_w = self._xgb_weight
        tft_w = self._tft_weight

        # ── 5. Check 3-way signal agreement ──
        agreement = self._check_signal_agreement(xgb_dir, tft_dir, plib_dir)

        # ── 6. Choose primary direction ──
        # Prefer XGB (more mature), fallback to TFT when XGB has no direction
        direction = xgb_dir
        if direction is None:
            direction = tft_dir

        # ── 7. Compute weighted confidence ──
        combined_conf = xgb_w * xgb_conf + tft_w * tft_conf

        # Apply agreement boost / penalty
        if agreement["level"] == "ALL_AGREE":
            combined_conf += FUSION_DIRECTION_AGREE_BOOST
            reason_agreement = "all 3 sources agree"
        elif agreement["level"] == "XGB_TFT_AGREE":
            combined_conf += FUSION_DIRECTION_AGREE_BOOST * 0.5
            reason_agreement = "XGB & TFT agree, no pattern match"
        elif agreement["level"] == "PARTIAL":
            combined_conf -= FUSION_DIRECTION_DISAGREE_PENALTY * 0.3
            reason_agreement = "partial agreement"
        else:
            # DISAGREE — penalize but don't kill; prefer XGB
            combined_conf -= FUSION_DIRECTION_DISAGREE_PENALTY
            reason_agreement = f"XGB={xgb_dir} vs TFT={tft_dir} disagree, prefer XGB"

        combined_conf = max(0.0, min(1.0, combined_conf))

        # ── 8. Compute weighted expected R ──
        combined_r = xgb_w * xgb_r + tft_w * tft_r

        # Boost expected R when pattern library also agrees
        if pattern_available and plib_dir == direction:
            plib_r = plib.get("expected_r", 0.0)
            # Blend in a small pattern library contribution (up to 15%)
            pattern_boost_weight = 0.15 * min(plib.get("match_score", 0.0), 1.0)
            combined_r = (1.0 - pattern_boost_weight) * combined_r + pattern_boost_weight * plib_r

        # ── 9. Reversal warning from TFT ──
        reversal_prob = tft.get("reversal_probability", 0.0) or 0.0
        reversal_warning = reversal_prob > TFT_REVERSAL_THRESHOLD

        if reversal_warning:
            # Dampen confidence when reversal is likely
            reversal_dampening = 1.0 - (reversal_prob - TFT_REVERSAL_THRESHOLD) * 0.5
            combined_conf *= max(0.5, reversal_dampening)
            log.debug(
                f"[FUSION] {self.pair}: reversal warning "
                f"(prob={reversal_prob:.2f} > threshold={TFT_REVERSAL_THRESHOLD}), "
                f"dampened confidence to {combined_conf:.2f}"
            )

        # ── 10. TFT contribution metric ──
        # How much of the final confidence comes from TFT
        tft_contribution = round(tft_w * tft_conf / max(combined_conf, 1e-6), 4)
        tft_contribution = min(tft_contribution, 1.0)

        # ── 11. Recommendation ──
        recommendation = self._make_recommendation(
            combined_conf, combined_r, direction, reversal_warning
        )

        # ── 12. Build reason string ──
        reason = (
            f"[{self.pair}] Fusion {recommendation}: "
            f"conf={combined_conf:.3f} R={combined_r:.2f} dir={direction} | "
            f"w(xgb={xgb_w:.2f},tft={tft_w:.2f}) | "
            f"agreement={agreement['level']} ({reason_agreement}) | "
            f"xgb: R={xgb_r:.2f} conf={xgb_conf:.2f}, "
            f"tft: R={tft_r:.2f} conf={tft_conf:.2f}"
            + (f" | REVERSAL WARNING (prob={reversal_prob:.2f})" if reversal_warning else "")
        )

        log.info(f"[FUSION] {reason}")

        return {
            "combined_confidence": round(combined_conf, 4),
            "combined_expected_r": round(combined_r, 4),
            "direction": direction,
            "tft_contribution": tft_contribution,
            "signal_agreement": agreement["level"],
            "reversal_warning": reversal_warning,
            "recommendation": recommendation,
            "weights": {
                "xgb_weight": round(self._xgb_weight, 4),
                "tft_weight": round(self._tft_weight, 4),
            },
            "reason": reason,
            "xgb_available": xgb_available,
            "tft_available": tft_available,
            "pattern_available": pattern_available,
        }

    # ──────────────────────────────────────────────────────────
    #  TFT SIGNAL DERIVATION
    # ──────────────────────────────────────────────────────────

    def _derive_tft_signal(self, tft_result: dict) -> dict:
        """
        Derive overall TFT direction and confidence from the 3 raw outputs.

        Raw TFT outputs:
          - candle_pattern_match: float [0, 1]
          - momentum_score:       float [-1, 1]
          - reversal_probability: float [0, 1]

        Derived:
          - direction:  "BUY" / "SELL" / None
          - confidence: weighted combo (see below)
          - derived_r:  momentum_score * candle_pattern_match * 2.0
        """
        pattern_match = max(0.0, min(1.0, float(tft_result.get("candle_pattern_match", 0.0) or 0.0)))
        momentum = max(-1.0, min(1.0, float(tft_result.get("momentum_score", 0.0) or 0.0)))
        reversal = max(0.0, min(1.0, float(tft_result.get("reversal_probability", 0.0) or 0.0)))

        # Direction from momentum score sign
        if momentum > 0.1:
            direction = "BUY"
        elif momentum < -0.1:
            direction = "SELL"
        else:
            direction = None

        # Confidence: weighted combination
        #   - candle_pattern_match weight: 0.4
        #   - |momentum_score| weight:     0.4
        #   - (1 - reversal_probability):  0.2  (low reversal = high confidence)
        confidence = (
            0.4 * pattern_match
            + 0.4 * abs(momentum)
            + 0.2 * (1.0 - reversal)
        )
        confidence = max(0.0, min(1.0, confidence))

        # Scale to R-like range (momentum * pattern_match gives ~[-1, 1], *2 -> ~[-2, 2])
        derived_r = momentum * pattern_match * 2.0

        return {
            "direction": direction,
            "confidence": confidence,
            "derived_r": derived_r,
            "pattern_match": pattern_match,
            "momentum": momentum,
            "reversal": reversal,
        }

    # ──────────────────────────────────────────────────────────
    #  SIGNAL AGREEMENT
    # ──────────────────────────────────────────────────────────

    def _check_signal_agreement(self,
                                 xgb_dir: Optional[str],
                                 tft_dir: Optional[str],
                                 pattern_dir: Optional[str]) -> dict:
        """
        Check agreement across all 3 signal sources.

        Returns:
            dict with:
              - level: "ALL_AGREE" | "XGB_TFT_AGREE" | "PARTIAL" | "DISAGREE"
              - boost: confidence adjustment multiplier ( informational )
        """
        # Strip None values for counting
        dirs = [d for d in [xgb_dir, tft_dir, pattern_dir] if d is not None]

        if len(dirs) == 0:
            return {"level": "DISAGREE", "boost": 0.0}

        if len(dirs) == 1:
            return {"level": "PARTIAL", "boost": 0.0}

        if len(dirs) == 2:
            # Check the two available
            if xgb_dir is not None and tft_dir is not None:
                if xgb_dir == tft_dir:
                    return {"level": "XGB_TFT_AGREE", "boost": FUSION_DIRECTION_AGREE_BOOST * 0.5}
                else:
                    return {"level": "DISAGREE", "boost": -FUSION_DIRECTION_DISAGREE_PENALTY}
            # One of xgb/tft + pattern only => partial
            return {"level": "PARTIAL", "boost": 0.0}

        # All 3 directions present
        if xgb_dir == tft_dir == pattern_dir:
            return {"level": "ALL_AGREE", "boost": FUSION_DIRECTION_AGREE_BOOST}

        # XGB & TFT agree but pattern doesn't
        if xgb_dir == tft_dir:
            return {"level": "XGB_TFT_AGREE", "boost": FUSION_DIRECTION_AGREE_BOOST * 0.5}

        # Any other combination is partial disagreement
        return {"level": "PARTIAL", "boost": -FUSION_DIRECTION_DISAGREE_PENALTY * 0.3}

    # ──────────────────────────────────────────────────────────
    #  RECOMMENDATION LOGIC
    # ──────────────────────────────────────────────────────────

    def _make_recommendation(self,
                              confidence: float,
                              expected_r: float,
                              direction: Optional[str],
                              reversal_warning: bool) -> str:
        """
        Decide TAKE / CAUTION / SKIP.

        Rules:
          - No direction                -> SKIP
          - Reversal warning + low conf  -> SKIP
          - conf >= GATE_MIN_CONFIDENCE
            AND R >= GATE_MIN_PREDICTED_R -> TAKE
          - R > 0 AND conf > 0.3        -> CAUTION
          - Everything else              -> SKIP
        """
        if direction is None:
            return "SKIP"

        # Strong reversal warning overrides even decent signals
        if reversal_warning and confidence < 0.5:
            return "SKIP"

        if (confidence >= GATE_MIN_CONFIDENCE
                and expected_r >= GATE_MIN_PREDICTED_R):
            if reversal_warning:
                return "CAUTION"  # Downgrade from TAKE
            return "TAKE"

        if expected_r > 0.0 and confidence > 0.3:
            return "CAUTION"

        return "SKIP"

    # ──────────────────────────────────────────────────────────
    #  SINGLE-MODEL FALLBACKS
    # ──────────────────────────────────────────────────────────

    def _xgb_only_fusion(self, xgb: dict, plib: dict) -> dict:
        """Fuse when only XGBoost is available."""
        xgb_conf = xgb.get("confidence", 0.0)
        xgb_r = xgb.get("predicted_r", 0.0)
        xgb_dir = xgb.get("direction")
        plib_dir = plib.get("direction")

        combined_conf = xgb_conf * 0.8  # Reduced for single-model
        combined_r = xgb_r

        # Pattern library agreement boost
        if plib_dir is not None and plib_dir == xgb_dir:
            combined_conf = min(1.0, combined_conf + 0.05)
            plib_r = plib.get("expected_r", 0.0)
            combined_r = 0.85 * combined_r + 0.15 * plib_r

        recommendation = self._make_recommendation(combined_conf, combined_r, xgb_dir, False)

        reason = (
            f"[{self.pair}] XGB-only {recommendation}: "
            f"conf={combined_conf:.3f} R={combined_r:.2f} dir={xgb_dir} | "
            f"xgb: R={xgb_r:.2f} conf={xgb_conf:.2f}"
            + (f" | pattern agrees ({plib.get('tier', '?')} tier)" if plib_dir == xgb_dir else "")
        )
        log.info(f"[FUSION] {reason}")

        return {
            "combined_confidence": round(combined_conf, 4),
            "combined_expected_r": round(combined_r, 4),
            "direction": xgb_dir,
            "tft_contribution": 0.0,
            "signal_agreement": "PARTIAL" if plib_dir else "PARTIAL",
            "reversal_warning": False,
            "recommendation": recommendation,
            "weights": {"xgb_weight": 1.0, "tft_weight": 0.0},
            "reason": reason,
            "xgb_available": True,
            "tft_available": False,
            "pattern_available": plib.get("win_rate") is not None,
        }

    def _tft_only_fusion(self, tft: dict, plib: dict) -> dict:
        """Fuse when only TFT is available."""
        tft_signal = self._derive_tft_signal(tft)
        tft_conf = tft_signal["confidence"]
        tft_r = tft_signal["derived_r"]
        tft_dir = tft_signal["direction"]

        combined_conf = tft_conf * 0.8  # Reduced for single-model
        combined_r = tft_r

        # Pattern library agreement boost
        plib_dir = plib.get("direction")
        if plib_dir is not None and plib_dir == tft_dir:
            combined_conf = min(1.0, combined_conf + 0.05)
            plib_r = plib.get("expected_r", 0.0)
            combined_r = 0.85 * combined_r + 0.15 * plib_r

        reversal_prob = tft.get("reversal_probability", 0.0) or 0.0
        reversal_warning = reversal_prob > TFT_REVERSAL_THRESHOLD
        if reversal_warning:
            combined_conf *= max(0.5, 1.0 - (reversal_prob - TFT_REVERSAL_THRESHOLD) * 0.5)

        recommendation = self._make_recommendation(
            combined_conf, combined_r, tft_dir, reversal_warning
        )

        reason = (
            f"[{self.pair}] TFT-only {recommendation}: "
            f"conf={combined_conf:.3f} R={combined_r:.2f} dir={tft_dir} | "
            f"tft: R={tft_r:.2f} conf={tft_conf:.2f} momentum={tft_signal['momentum']:.2f}"
            + (f" | REVERSAL WARNING" if reversal_warning else "")
        )
        log.info(f"[FUSION] {reason}")

        return {
            "combined_confidence": round(combined_conf, 4),
            "combined_expected_r": round(combined_r, 4),
            "direction": tft_dir,
            "tft_contribution": 1.0,
            "signal_agreement": "PARTIAL" if plib_dir else "PARTIAL",
            "reversal_warning": reversal_warning,
            "recommendation": recommendation,
            "weights": {"xgb_weight": 0.0, "tft_weight": 1.0},
            "reason": reason,
            "xgb_available": False,
            "tft_available": True,
            "pattern_available": plib.get("win_rate") is not None,
        }

    # ──────────────────────────────────────────────────────────
    #  WEIGHT UPDATE (online learning)
    # ──────────────────────────────────────────────────────────

    def update_weights(self, outcome: float,
                       xgb_predicted_r: Optional[float] = None,
                       tft_predicted_r: Optional[float] = None):
        """
        Update fusion weights based on a trade outcome.

        Simple online learning:
          - If TFT was closer to actual outcome than XGB, shift weight
            toward TFT (and vice versa).
          - Uses EMA smoothing to prevent jitter.
          - Clamps weights to [FUSION_MIN_WEIGHT, FUSION_MAX_WEIGHT].

        Args:
            outcome: Actual R-multiple of the closed trade.
            xgb_predicted_r: XGB's predicted R for this trade (if available).
            tft_predicted_r: TFT's predicted R for this trade (if available).
        """
        if xgb_predicted_r is not None and tft_predicted_r is not None:
            # Compare prediction errors
            xgb_error = abs(outcome - xgb_predicted_r)
            tft_error = abs(outcome - tft_predicted_r)

            if tft_error < xgb_error:
                # TFT was more accurate -> shift weight toward TFT
                delta = FUSION_META_LR * (1.0 - tft_error / max(xgb_error, 1e-6))
                new_xgb = self._xgb_weight - delta
                new_tft = self._tft_weight + delta
            elif xgb_error < tft_error:
                # XGB was more accurate -> shift weight toward XGB
                delta = FUSION_META_LR * (1.0 - xgb_error / max(tft_error, 1e-6))
                new_xgb = self._xgb_weight + delta
                new_tft = self._tft_weight - delta
            else:
                # Equal accuracy — no change
                new_xgb = self._xgb_weight
                new_tft = self._tft_weight
        else:
            # Cannot compare — keep current weights
            log.debug(
                f"[FUSION] {self.pair}: skipping weight update, "
                f"missing prediction (xgb_r={xgb_predicted_r}, tft_r={tft_predicted_r})"
            )
            # Still record outcome for history
            self._performance_history.append({
                "outcome": outcome,
                "xgb_r": xgb_predicted_r,
                "tft_r": tft_predicted_r,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            if len(self._performance_history) > 200:
                self._performance_history = self._performance_history[-200:]
            return

        # EMA smoothing
        alpha = 1.0 - FUSION_WEIGHT_SMOOTHING  # 0.1 by default
        smoothed_xgb = FUSION_WEIGHT_SMOOTHING * self._xgb_weight + alpha * new_xgb
        smoothed_tft = FUSION_WEIGHT_SMOOTHING * self._tft_weight + alpha * new_tft

        # Clamp
        smoothed_xgb = max(FUSION_MIN_WEIGHT, min(FUSION_MAX_WEIGHT, smoothed_xgb))
        smoothed_tft = max(FUSION_MIN_WEIGHT, min(FUSION_MAX_WEIGHT, smoothed_tft))

        # Normalize to sum to 1.0
        total = smoothed_xgb + smoothed_tft
        if total > 0:
            smoothed_xgb = smoothed_xgb / total
            smoothed_tft = smoothed_tft / total

        self._xgb_weight = round(smoothed_xgb, 6)
        self._tft_weight = round(smoothed_tft, 6)
        self._n_updates += 1
        self._last_update = datetime.now(timezone.utc).isoformat()

        # Record in performance history
        self._performance_history.append({
            "outcome": outcome,
            "xgb_r": xgb_predicted_r,
            "tft_r": tft_predicted_r,
            "xgb_w": self._xgb_weight,
            "tft_w": self._tft_weight,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if len(self._performance_history) > 200:
            self._performance_history = self._performance_history[-200:]

        self.save_weights()

        log.info(
            f"[FUSION] {self.pair}: weights updated -> "
            f"xgb={self._xgb_weight:.4f} tft={self._tft_weight:.4f} "
            f"(outcome={outcome:.2f}, xgb_err={xgb_error:.2f}, tft_err={tft_error:.2f}, "
            f"updates={self._n_updates})"
        )

    # ──────────────────────────────────────────────────────────
    #  WEIGHT PERSISTENCE
    # ──────────────────────────────────────────────────────────

    def save_weights(self):
        """
        Persist current fusion weights to disk.

        Uses atomic write (write to temp file, then rename) for
        thread-safety.
        """
        data = {
            "pair": self.pair,
            "xgb_weight": self._xgb_weight,
            "tft_weight": self._tft_weight,
            "n_updates": self._n_updates,
            "last_update": self._last_update,
            "performance_history": self._performance_history[-50:],  # Keep last 50
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        with _IO_LOCK:
            try:
                _FUSION_MODELS_DIR.mkdir(parents=True, exist_ok=True)
                tmp_path = str(self._weight_path) + ".tmp"
                with open(tmp_path, "w") as f:
                    json.dump(data, f, indent=2, default=str)
                os.replace(tmp_path, str(self._weight_path))
            except Exception as e:
                log.error(f"[FUSION] {self.pair}: failed to save weights: {e}")

    def load_weights(self):
        """
        Load fusion weights from disk.

        If the file doesn't exist or is corrupt, keeps default weights.
        """
        if not self._weight_path.exists():
            log.debug(f"[FUSION] {self.pair}: no weight file, using defaults")
            return

        with _IO_LOCK:
            try:
                with open(self._weight_path, "r") as f:
                    data = json.load(f)

                self._xgb_weight = float(data.get("xgb_weight", FUSION_DEFAULT_XGB_WEIGHT))
                self._tft_weight = float(data.get("tft_weight", FUSION_DEFAULT_TFT_WEIGHT))
                self._n_updates = int(data.get("n_updates", 0))
                self._last_update = data.get("last_update")
                self._performance_history = data.get("performance_history", [])

                log.debug(
                    f"[FUSION] {self.pair}: loaded weights "
                    f"xgb={self._xgb_weight:.4f} tft={self._tft_weight:.4f} "
                    f"({self._n_updates} prior updates)"
                )
            except Exception as e:
                log.warning(f"[FUSION] {self.pair}: failed to load weights: {e}")

    def get_weights(self) -> dict:
        """
        Return current weights and metadata.

        Returns:
            Dict with xgb_weight, tft_weight, n_updates, last_update,
            pair, and weight_path.
        """
        return {
            "pair": self.pair,
            "xgb_weight": self._xgb_weight,
            "tft_weight": self._tft_weight,
            "n_updates": self._n_updates,
            "last_update": self._last_update,
            "weight_path": str(self._weight_path),
            "performance_history_len": len(self._performance_history),
        }

    # ──────────────────────────────────────────────────────────
    #  INTERNAL HELPERS
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _safe_dict(d: Optional[dict]) -> dict:
        """Return an empty dict if *d* is None or not a dict."""
        if d is None or not isinstance(d, dict):
            return {}
        return d


# ════════════════════════════════════════════════════════════════
#  MODULE-LEVEL CONVENIENCE FUNCTION
# ════════════════════════════════════════════════════════════════

# Simple cache to avoid re-creating FusionLayer on every call
_fusion_cache: Dict[str, FusionLayer] = {}
_fusion_cache_lock = threading.Lock()


def fuse_all_signals(pair: str,
                     xgb_result: Optional[dict],
                     tft_result: Optional[dict],
                     pattern_match: Optional[dict]) -> dict:
    """
    Convenience function for one-shot fusion without managing state.

    Creates or reuses a FusionLayer for *pair*, then calls fuse().

    Used by PatternGate and other callers that don't need to hold a
    long-lived FusionLayer instance.

    Args:
        pair: Currency pair string (e.g. "EURJPY").
        xgb_result: Dict from PatternModel.predict().
        tft_result: Dict from TFTModelManager.predict().
        pattern_match: Dict from PatternGate._match_pattern_library().

    Returns:
        Full fusion result dict.
    """
    pair_upper = pair.upper()

    with _fusion_cache_lock:
        if pair_upper not in _fusion_cache:
            _fusion_cache[pair_upper] = FusionLayer(pair_upper)
        fusion = _fusion_cache[pair_upper]

    return fusion.fuse(xgb_result, tft_result, pattern_match)
