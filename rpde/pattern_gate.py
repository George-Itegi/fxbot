# =============================================================
# rpde/pattern_gate.py  —  RPDE L2: Pattern Gate (Decision Layer)
#
# PURPOSE: Final decision layer that combines per-pair pattern
# model predictions with pattern library confidence to decide
# whether to take a pattern-based trade.
#
# ARCHITECTURE:
#   v4.2: ML Gate → R-multiple prediction → TAKE/CAUTION/SKIP
#   v5.0: Pattern Gate → pattern match + confidence + expected R
#         → TAKE/CAUTION/SKIP
#
# The gate is lightweight — it does NOT train models or compute
# heavy features. It COMBINES signals from:
#   1. PatternModel (L1): XGBoost predicted R-multiple
#   2. Pattern Library: Nearest pattern match with historical WR
#   3. Human Guards: Safety checks that NEVER get overridden by AI
#
# DECISION FLOW:
#   features → PatternModel.predict() → model_result
#   features → Pattern Library match → pattern_match
#   model_result + pattern_match → _combine_signals() → combined
#   combined + master_report → _apply_human_guards() → final
#
# OUTPUT:
#   TAKE    → Good edge, model + pattern agree, guards pass
#   CAUTION → Marginal edge, take with reduced size or shadow
#   SKIP    → No edge, guards fail, or signals disagree
# =============================================================

import numpy as np
from datetime import datetime, timezone
from typing import Optional, List

from core.logger import get_logger
from rpde.pattern_model import PatternModel

log = get_logger(__name__)


class PatternGate:
    """
    L2 Pattern Gate — Final decision layer for RPDE system.

    Combines per-pair pattern model prediction with pattern library
    confidence to make the final TAKE/CAUTION/SKIP decision.

    Usage:
        gate = PatternGate()
        gate.initialize()
        result = gate.evaluate("EURJPY", features, master_report)
        if result['recommendation'] == 'TAKE':
            # Execute trade with result['direction']
    """

    def __init__(self):
        self.models = {}           # {pair: PatternModel}
        self.pattern_cache = {}    # {pair: [active_patterns]}
        self.is_initialized = False

    # ════════════════════════════════════════════════════════════════
    # INITIALIZATION
    # ════════════════════════════════════════════════════════════════

    def initialize(self, pairs: list = None):
        """
        Load all trained pattern models and active patterns.
        Called once at startup.

        Args:
            pairs: List of pair strings. If None, loads PAIR_WHITELIST.
        """
        if pairs is None:
            from config.settings import PAIR_WHITELIST
            pairs = PAIR_WHITELIST

        loaded_models = 0
        loaded_patterns = 0

        for pair in pairs:
            pair_upper = pair.upper()

            # Load model
            model = PatternModel(pair_upper)
            if model.is_trained():
                self.models[pair_upper] = model
                loaded_models += 1

            # Load patterns from library
            try:
                from rpde.database import load_pattern_library
                patterns = load_pattern_library(
                    pair=pair_upper, active_only=True)
                if patterns:
                    self.pattern_cache[pair_upper] = patterns
                    loaded_patterns += len(patterns)
            except Exception as e:
                log.debug(f"[RPDE_GATE] Failed to load patterns for "
                          f"{pair_upper}: {e}")

        self.is_initialized = True
        log.info(
            f"[RPDE_GATE] Initialized: {loaded_models} models, "
            f"{loaded_patterns} patterns across {len(pairs)} pairs")

    # ════════════════════════════════════════════════════════════════
    # MAIN EVALUATION
    # ════════════════════════════════════════════════════════════════

    def evaluate(self, pair: str, features: np.ndarray,
                 master_report: dict = None) -> dict:
        """
        Evaluate whether to take a pattern-based trade.

        Process:
        1. Get pattern model prediction for this pair
        2. Match features against pattern library
        3. Find best matching pattern
        4. Combine model prediction + pattern confidence
        5. Apply human guard checks (from master_report)
        6. Make final decision

        Args:
            pair: Currency pair string (e.g. 'EURJPY')
            features: numpy array of shape (93,) with ML Gate features
            master_report: Optional master report dict with market context

        Returns:
            Evaluation result dict with recommendation, direction,
            confidence, matched pattern, guards, and reason.
        """
        pair_upper = pair.upper()
        mr = master_report or {}

        # ── Step 1: Get pattern model prediction ──
        model_result = self._get_model_prediction(pair_upper, features)

        # ── Step 2: Match against pattern library ──
        pattern_match = self._match_pattern_library(
            pair_upper, features)

        # ── Step 3: Combine signals ──
        combined = self._combine_signals(model_result, pattern_match)

        # ── Step 4: Apply human guards ──
        guards = self._apply_human_guards(pair_upper, mr)

        # ── Step 5: Make final decision ──
        decision = self._make_decision(combined, guards)

        # ── Build result ──
        result = {
            'recommendation': decision['recommendation'],
            'direction': decision['direction'],
            'model_predicted_r': model_result.get('predicted_r', 0.0),
            'model_confidence': model_result.get('confidence', 0.0),
            'matched_pattern': pattern_match.get('pattern_id'),
            'pattern_win_rate': pattern_match.get('win_rate'),
            'pattern_tier': pattern_match.get('tier'),
            'pattern_expected_r': pattern_match.get('expected_r'),
            'pattern_match_score': pattern_match.get('match_score'),
            'combined_confidence': combined.get('combined_confidence', 0.0),
            'expected_r': combined.get('expected_r', 0.0),
            'human_guards': guards,
            'reason': decision['reason'],
            'pair': pair_upper,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }

        if decision['recommendation'] in ('TAKE', 'CAUTION'):
            log.info(
                f"[RPDE_GATE] {pair_upper} → {decision['recommendation']} "
                f"{decision['direction']} "
                f"R={combined.get('expected_r', 0):.2f} "
                f"conf={combined.get('combined_confidence', 0):.2f} "
                f"| {decision['reason']}")

        return result

    # ════════════════════════════════════════════════════════════════
    # MODEL PREDICTION
    # ════════════════════════════════════════════════════════════════

    def _get_model_prediction(self, pair: str,
                               features: np.ndarray) -> dict:
        """
        Get pattern model prediction for this pair.

        Args:
            pair: Uppercase pair string
            features: Feature array (93,)

        Returns:
            Model prediction dict with predicted_r, confidence, direction,
            is_pattern. Falls back to neutral if no model exists.
        """
        model = self.models.get(pair)

        if model is None or not model.is_trained():
            return {
                'predicted_r': 0.0,
                'confidence': 0.0,
                'direction': None,
                'is_pattern': False,
                'model_loaded': False,
            }

        try:
            return model.predict(features)
        except Exception as e:
            log.error(f"[RPDE_GATE] Model prediction failed for {pair}: {e}")
            return {
                'predicted_r': 0.0,
                'confidence': 0.0,
                'direction': None,
                'is_pattern': False,
                'model_loaded': True,
                'error': str(e),
            }

    # ════════════════════════════════════════════════════════════════
    # PATTERN LIBRARY MATCHING
    # ════════════════════════════════════════════════════════════════

    def _match_pattern_library(self, pair: str,
                                features: np.ndarray) -> dict:
        """
        Match current features against stored patterns in the library.

        Uses cosine similarity between current feature vector and
        each pattern's cluster center to find the best match.

        Args:
            pair: Uppercase pair string
            features: Feature array (93,)

        Returns:
            Pattern match dict with pattern_id, win_rate, tier,
            expected_r, match_score, direction. Empty/neutral if
            no patterns loaded or no good match found.
        """
        patterns = self.pattern_cache.get(pair, [])
        if not patterns:
            return {
                'pattern_id': None,
                'win_rate': None,
                'tier': None,
                'expected_r': 0.0,
                'match_score': 0.0,
                'direction': None,
                'n_patterns_loaded': 0,
            }

        # Build feature dict for matching
        from ai_engine.ml_gate import FEATURE_NAMES
        feature_dict = {}
        for i, name in enumerate(FEATURE_NAMES):
            if i < len(features):
                feature_dict[name] = float(features[i])

        best_match = None
        best_score = -1.0

        for pattern in patterns:
            score = self._compute_pattern_similarity(
                feature_dict, pattern)
            if score > best_score:
                best_score = score
                best_match = pattern

        if best_match is None or best_score < 0.3:
            # No pattern matches well enough
            return {
                'pattern_id': None,
                'win_rate': None,
                'tier': None,
                'expected_r': 0.0,
                'match_score': round(best_score, 4) if best_score > 0 else 0.0,
                'direction': None,
                'n_patterns_loaded': len(patterns),
            }

        return {
            'pattern_id': best_match.get('pattern_id'),
            'win_rate': float(best_match.get('win_rate', 0.0)),
            'tier': best_match.get('tier'),
            'expected_r': float(best_match.get('avg_expected_r', 0.0)),
            'profit_factor': float(best_match.get('profit_factor', 0.0)),
            'occurrences': int(best_match.get('occurrences', 0)),
            'match_score': round(best_score, 4),
            'direction': best_match.get('direction'),
            'n_patterns_loaded': len(patterns),
        }

    @staticmethod
    def _compute_pattern_similarity(feature_dict: dict,
                                      pattern: dict) -> float:
        """
        Compute similarity between current features and a pattern's
        feature ranges using normalized distance scoring.

        For each feature in the pattern's ranges, compute how close
        the current value is to the pattern's mean, normalized by std.
        Average across all features to get overall similarity.

        Returns:
            Similarity score in [0.0, 1.0].
        """
        feature_ranges = pattern.get('feature_ranges', {})
        if not feature_ranges:
            return 0.0

        from rpde.config import CLUSTER_FEATURES

        scores = []
        for feat_name in CLUSTER_FEATURES:
            fr = feature_ranges.get(feat_name)
            if fr is None:
                continue

            current_val = feature_dict.get(feat_name)
            if current_val is None:
                continue

            mean_val = fr.get('mean', 0.0)
            std_val = fr.get('std', 0.0)

            if std_val is None or std_val < 1e-8:
                # No variation — use range instead
                min_val = fr.get('min', mean_val - 1.0)
                max_val = fr.get('max', mean_val + 1.0)
                range_val = max_val - min_val
                if range_val < 1e-8:
                    # Constant value — check if current matches
                    scores.append(1.0 if abs(current_val - mean_val) < 1e-6 else 0.0)
                    continue
                # Distance from center of range, normalized to [0, 1]
                center = (min_val + max_val) / 2.0
                half_range = range_val / 2.0
                dist = abs(current_val - center)
                score = max(0.0, 1.0 - dist / half_range)
            else:
                # Z-score based similarity
                z = abs(current_val - mean_val) / std_val
                # Map z-score to similarity: z=0 → 1.0, z=1 → 0.6, z=2 → 0.3, z=3 → 0.1
                score = 1.0 / (1.0 + z * 0.5)

            scores.append(score)

        if not scores:
            return 0.0

        return float(np.mean(scores))

    # ════════════════════════════════════════════════════════════════
    # SIGNAL COMBINATION
    # ════════════════════════════════════════════════════════════════

    def _combine_signals(self, model_result: dict,
                         pattern_match: dict) -> dict:
        """
        Combine pattern model prediction with pattern library match.

        Combination rules:
        - combined_confidence = weighted average of model_confidence
          and pattern-based confidence
        - combined_expected_r = weighted average of model_R and
          pattern_expected_R
        - If direction disagrees between model and pattern → SKIP

        Weights:
        - If only model available: weight = 1.0 model
        - If only pattern available: weight = 1.0 pattern
        - If both: 0.5 model + 0.5 pattern

        Returns:
            Combined signal dict.
        """
        from rpde.config import GATE_MIN_CONFIDENCE

        # Extract model signal
        model_r = model_result.get('predicted_r', 0.0)
        model_conf = model_result.get('confidence', 0.0)
        model_dir = model_result.get('direction')
        model_loaded = model_result.get('model_loaded', False)

        # Extract pattern signal
        pattern_wr = pattern_match.get('win_rate')
        pattern_r = pattern_match.get('expected_r', 0.0)
        pattern_score = pattern_match.get('match_score', 0.0)
        pattern_dir = pattern_match.get('direction')
        pattern_tier = pattern_match.get('tier')

        # Determine availability
        model_available = model_loaded and model_dir is not None
        pattern_available = (
            pattern_wr is not None
            and pattern_score >= 0.3
            and pattern_dir is not None
        )

        # ── Direction disagreement check ──
        if model_available and pattern_available:
            if model_dir != pattern_dir:
                # Signals disagree → force SKIP later via direction=None
                log.debug(
                    f"[RPDE_GATE] Direction disagreement: "
                    f"model={model_dir} vs pattern={pattern_dir}")
                return {
                    'combined_confidence': 0.0,
                    'expected_r': 0.0,
                    'direction': None,
                    'direction_disagree': True,
                    'model_weight': 0.5,
                    'pattern_weight': 0.5,
                    'reason': f'Direction disagreement: model={model_dir} vs pattern={pattern_dir}',
                }

        # ── Compute weights ──
        if model_available and pattern_available:
            # Both available — use pattern tier to adjust weight
            tier_boost = {
                'GOD_TIER': 0.15,      # Trust validated patterns more
                'STRONG': 0.10,
                'VALID': 0.05,
                'PROBATIONARY': 0.0,   # Equal weight for unproven patterns
            }
            boost = tier_boost.get(pattern_tier, 0.0)
            model_weight = 0.5 - boost
            pattern_weight = 0.5 + boost
        elif model_available:
            model_weight = 1.0
            pattern_weight = 0.0
        elif pattern_available:
            model_weight = 0.0
            pattern_weight = 1.0
        else:
            # Neither available
            return {
                'combined_confidence': 0.0,
                'expected_r': 0.0,
                'direction': None,
                'direction_disagree': False,
                'model_weight': 0.0,
                'pattern_weight': 0.0,
                'reason': 'No model or pattern available',
            }

        # ── Pattern confidence from win_rate and match quality ──
        # Higher win_rate → higher confidence
        # Higher match_score → higher confidence
        # Pattern WR maps to confidence: WR=0.55→0.2, WR=0.70→0.6, WR=0.85→0.9
        if pattern_wr is not None:
            pattern_conf = max(0.0, min(1.0, (pattern_wr - 0.45) * 3.0))
            # Scale by match quality
            pattern_conf *= pattern_score
        else:
            pattern_conf = 0.0

        # ── Combined metrics ──
        combined_conf = (model_weight * model_conf
                         + pattern_weight * pattern_conf)
        combined_conf = max(0.0, min(1.0, combined_conf))

        combined_r = (model_weight * model_r
                      + pattern_weight * pattern_r)

        # Direction: prefer model direction, fallback to pattern
        direction = model_dir if model_dir else pattern_dir

        return {
            'combined_confidence': round(combined_conf, 4),
            'expected_r': round(combined_r, 4),
            'direction': direction,
            'direction_disagree': False,
            'model_weight': round(model_weight, 2),
            'pattern_weight': round(pattern_weight, 2),
            'reason': (f'Model R={model_r:.2f} ({model_weight:.0%}), '
                       f'Pattern WR={pattern_wr:.0%} R={pattern_r:.2f} '
                       f'({pattern_weight:.0%})'),
        }

    # ════════════════════════════════════════════════════════════════
    # HUMAN GUARDS
    # ════════════════════════════════════════════════════════════════

    def _apply_human_guards(self, pair: str,
                             master_report: dict) -> dict:
        """
        Apply human-controlled safety guards.

        These NEVER get overridden by AI. If any critical guard fails,
        the trade is blocked regardless of model confidence.

        Guards:
        1. Spread filter: spread < max_spread for this pair
        2. Session filter: must be in SESSION_WHITELIST
        3. Choppy market filter: skip if choppy + no volume surge
        4. Max positions: < MAX_PATTERN_POSITIONS
        5. ADX filter: skip if ADX < 15 (no clear trend)

        Args:
            pair: Uppercase pair string
            master_report: Dict with market context

        Returns:
            Dict with individual guard results and overall pass/fail.
        """
        from config.settings import MAX_SPREAD, SESSION_WHITELIST
        from rpde.config import MAX_PATTERN_POSITIONS, PATTERN_COOLDOWN_MINUTES

        guards = {
            'spread_ok': True,
            'session_ok': True,
            'choppy_filter': True,
            'max_positions_ok': True,
            'adx_filter': True,
            'cooldown_ok': True,
            'all_passed': True,
            'block_reason': None,
        }

        # ── 1. Spread filter ──
        current_spread = float(master_report.get('spread', 0)
                              or master_report.get('spread_pips', 0)
                              or 0)
        max_spread = MAX_SPREAD.get(pair, MAX_SPREAD.get('DEFAULT', 4.0))

        if current_spread > max_spread:
            guards['spread_ok'] = False
            guards['block_reason'] = (
                f'Spread {current_spread:.1f} > max {max_spread:.1f}')

        # ── 2. Session filter ──
        current_session = master_report.get('session', '')
        if current_session and SESSION_WHITELIST:
            if current_session not in SESSION_WHITELIST:
                guards['session_ok'] = False
                guards['block_reason'] = (
                    f'Session {current_session} not in whitelist')

        # ── 3. Choppy market filter ──
        is_choppy = master_report.get('is_choppy', False)
        vol_surge = master_report.get('vol_surge', False)
        if is_choppy and not vol_surge:
            guards['choppy_filter'] = False
            guards['block_reason'] = 'Choppy market without volume surge'

        # ── 4. Max positions ──
        # Check current open pattern-based positions
        try:
            from rpde.database import load_pattern_trades
            open_trades = load_pattern_trades(
                pair=pair, source='LIVE')
            # Filter to only recent trades (within last 24 hours)
            active_positions = 0
            for t in open_trades:
                exit_time = t.get('exit_time')
                outcome = t.get('outcome')
                if outcome in ('WIN', 'LOSS', 'BE'):
                    continue  # Closed trade
                if exit_time is None:
                    active_positions += 1

            if active_positions >= MAX_PATTERN_POSITIONS:
                guards['max_positions_ok'] = False
                guards['block_reason'] = (
                    f'Max pattern positions ({MAX_PATTERN_POSITIONS}) reached')
        except Exception:
            # If we can't check, assume OK
            pass

        # ── 5. ADX filter ──
        adx = float(master_report.get('adx', 0)
                    or master_report.get('atr_adx', 0)
                    or 0)
        if adx > 0 and adx < 15:
            guards['adx_filter'] = False
            guards['block_reason'] = f'ADX {adx:.1f} < 15 (no clear trend)'

        # ── 6. Cooldown check ──
        try:
            from rpde.database import load_pattern_trades
            recent_trades = load_pattern_trades(pair=pair)
            cooldown_ok = True
            for t in recent_trades[:5]:
                entry_time = t.get('entry_time')
                if entry_time is None:
                    continue
                try:
                    if isinstance(entry_time, str):
                        et = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                    else:
                        et = entry_time
                    elapsed = (datetime.now(timezone.utc) - et).total_seconds() / 60
                    if elapsed < PATTERN_COOLDOWN_MINUTES:
                        cooldown_ok = False
                        guards['block_reason'] = (
                            f'Cooldown active ({elapsed:.0f}/{PATTERN_COOLDOWN_MINUTES} min)')
                        break
                except Exception:
                    continue
            guards['cooldown_ok'] = cooldown_ok
        except Exception:
            pass

        # ── Overall pass/fail ──
        guards['all_passed'] = all([
            guards['spread_ok'],
            guards['session_ok'],
            guards['choppy_filter'],
            guards['max_positions_ok'],
            guards['adx_filter'],
            guards['cooldown_ok'],
        ])

        if not guards['all_passed']:
            log.debug(f"[RPDE_GATE] Guard blocked: {guards['block_reason']}")

        return guards

    # ════════════════════════════════════════════════════════════════
    # DECISION MAKING
    # ════════════════════════════════════════════════════════════════

    def _make_decision(self, combined: dict, guards: dict) -> dict:
        """
        Make the final TAKE/CAUTION/SKIP decision.

        Decision rules:
        - Guards fail → SKIP (always, no override)
        - Direction is None → SKIP
        - combined_confidence >= GATE_MIN_CONFIDENCE AND expected_r >= GATE_MIN_PREDICTED_R → TAKE
        - expected_r > 0 but below thresholds → CAUTION
        - expected_r <= 0 → SKIP

        Args:
            combined: Combined signal dict from _combine_signals()
            guards: Guard results dict from _apply_human_guards()

        Returns:
            Dict with recommendation and reason.
        """
        from rpde.config import (GATE_MIN_CONFIDENCE, GATE_MIN_PREDICTED_R)

        # ── Guard veto (always wins) ──
        if not guards['all_passed']:
            return {
                'recommendation': 'SKIP',
                'direction': None,
                'reason': f"Guard blocked: {guards.get('block_reason', 'unknown')}",
            }

        # ── Direction disagreement ──
        if combined.get('direction_disagree', False):
            return {
                'recommendation': 'SKIP',
                'direction': None,
                'reason': combined.get('reason', 'Direction disagreement'),
            }

        direction = combined.get('direction')
        confidence = combined.get('combined_confidence', 0.0)
        expected_r = combined.get('expected_r', 0.0)

        # ── No direction → SKIP ──
        if direction is None:
            return {
                'recommendation': 'SKIP',
                'direction': None,
                'reason': 'No clear directional signal',
            }

        # ── Both thresholds met → TAKE ──
        if (confidence >= GATE_MIN_CONFIDENCE
                and expected_r >= GATE_MIN_PREDICTED_R):
            return {
                'recommendation': 'TAKE',
                'direction': direction,
                'reason': combined.get('reason', f'Edge detected: R={expected_r:.2f}'),
            }

        # ── Marginal edge → CAUTION ──
        if expected_r > 0.0 and confidence > 0.3:
            return {
                'recommendation': 'CAUTION',
                'direction': direction,
                'reason': (f'Marginal edge: R={expected_r:.2f}, '
                           f'conf={confidence:.2f} (need conf>={GATE_MIN_CONFIDENCE}, '
                           f'R>={GATE_MIN_PREDICTED_R})'),
            }

        # ── No edge → SKIP ──
        return {
            'recommendation': 'SKIP',
            'direction': None,
            'reason': (f'No edge: R={expected_r:.2f}, conf={confidence:.2f}'),
        }

    # ════════════════════════════════════════════════════════════════
    # STATUS
    # ════════════════════════════════════════════════════════════════

    def get_status(self) -> dict:
        """
        Get status of all loaded models and patterns.

        Returns:
            Status dict with model info and pattern counts per pair.
        """
        status = {
            'is_initialized': self.is_initialized,
            'models_loaded': len(self.models),
            'patterns_loaded': sum(len(p) for p in self.pattern_cache.values()),
            'per_pair': {},
        }

        all_pairs = set(list(self.models.keys()) + list(self.pattern_cache.keys()))

        for pair in sorted(all_pairs):
            model = self.models.get(pair)
            patterns = self.pattern_cache.get(pair, [])

            pair_status = {
                'model_loaded': model is not None and model.is_trained(),
                'patterns_count': len(patterns),
            }

            if model and model.is_trained():
                info = model.get_info()
                pair_status['model_samples'] = info.get('training_samples')
                pair_status['model_val_corr'] = info.get('val_corr')
                pair_status['model_val_r2'] = info.get('val_r2')
                pair_status['model_trained_at'] = info.get('trained_at')

            if patterns:
                best = max(patterns, key=lambda p: p.get('win_rate', 0))
                pair_status['best_pattern_wr'] = best.get('win_rate')
                pair_status['best_pattern_tier'] = best.get('tier')
                pair_status['best_pattern_id'] = best.get('pattern_id')

            status['per_pair'][pair] = pair_status

        return status

    def reload_patterns(self, pairs: list = None):
        """
        Reload pattern cache from database (e.g. after re-mining).

        Args:
            pairs: List of pairs to reload. If None, reloads all cached pairs.
        """
        if pairs is None:
            pairs = list(self.pattern_cache.keys())

        reloaded = 0
        for pair in pairs:
            pair_upper = pair.upper()
            try:
                from rpde.database import load_pattern_library
                patterns = load_pattern_library(
                    pair=pair_upper, active_only=True)
                self.pattern_cache[pair_upper] = patterns
                reloaded += 1
            except Exception as e:
                log.warning(
                    f"[RPDE_GATE] Failed to reload patterns for "
                    f"{pair_upper}: {e}")

        log.info(f"[RPDE_GATE] Reloaded patterns for {reloaded} pairs")
