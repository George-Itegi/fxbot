# =============================================================
# rpde/tft_dataset.py  —  Multi-Timeframe Dataset Builder for TFT
#
# PURPOSE: Build PyTorch datasets from golden moments for the
# Temporal Fusion Transformer. Each sample corresponds to a
# golden moment (big move discovered by the RPDE scanner) and
# contains multi-timeframe candle sequences ending at that
# moment, plus the 3 TFT training targets.
#
# DATA FLOW:
#   Golden Moments (DB) → fetch multi-TF candles from MT5 →
#   compute 11 per-candle features → compute 3 targets →
#   extract 93-engineered context → PyTorch Dataset
#
# SEPARATION from Phase 1:
#   Phase 1 (XGBoost): 93 engineered features → single snapshot
#   Phase 2 (TFT): raw candle sequences → temporal patterns
#   This module bridges them: 93 features as context, candles
#   as the primary signal the TFT learns from.
#
# INPUTS:
#   - Golden moments from rpde.database.load_golden_moments()
#   - Candle data from data_layer.price_feed.get_candles()
#   - Config from rpde.config (TFT_TIMEFRAMES, TFT_RAW_FEATURES, etc.)
#
# OUTPUTS:
#   - MultiTFDataset: PyTorch Dataset with multi-TF feature tensors
#   - build_live_inputs(): dict of {tf: tensor} for live prediction
# =============================================================

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

from core.logger import get_logger
from rpde.config import (
    TFT_TIMEFRAMES,
    TFT_RAW_FEATURES,
    TFT_MIN_TRAINING_SAMPLES,
    TFT_TRAIN_VAL_SPLIT,
    TFT_SEQUENCE_STRIDE,
    TFT_MIN_PAIR_TRADES,
)

log = get_logger(__name__)

# ── Check torch availability ─────────────────────────────────
_TORCH_AVAILABLE = False
try:
    import torch
    from torch.utils.data import Dataset
    _TORCH_AVAILABLE = True
except ImportError:
    log.warning("[TFT_DATASET] PyTorch not installed — TFT dataset unavailable")

# ── Constants ─────────────────────────────────────────────────
N_FEATURES = len(TFT_RAW_FEATURES)  # 11 features per candle per TF
_MAX_CONTEXT_DIM = 93               # Max engineered context features


# ════════════════════════════════════════════════════════════════
#  TIMEFRAME UTILITY
# ════════════════════════════════════════════════════════════════

def _tf_to_seconds(tf_name: str) -> int:
    """Convert a timeframe name to its duration in seconds.

    Args:
        tf_name: Timeframe string (e.g. 'M5', 'H1', 'H4')

    Returns:
        Duration in seconds
    """
    mapping = {
        'M1': 60, 'M5': 300, 'M15': 900, 'M30': 1800,
        'H1': 3600, 'H4': 14400, 'D1': 86400, 'W1': 604800,
    }
    return mapping.get(tf_name, 300)


def _strip_tz(ts: Any) -> pd.Timestamp:
    """Normalize a timestamp to timezone-naive for comparison.

    Handles pd.Timestamp, datetime, and string inputs.

    Args:
        ts: Any timestamp-like object

    Returns:
        Timezone-naive pd.Timestamp, or NaT on failure
    """
    try:
        result = pd.Timestamp(ts)
        if result.tzinfo is not None:
            result = result.tz_convert(None)
        return result
    except Exception:
        return pd.NaT


# ════════════════════════════════════════════════════════════════
#  CANDLE FEATURE COMPUTATION
# ════════════════════════════════════════════════════════════════

def compute_candle_features(df: pd.DataFrame) -> np.ndarray:
    """Compute 11 features per candle from raw OHLCV data.

    Features produced (in order matching TFT_RAW_FEATURES):
        0. open            — raw open price
        1. high            — raw high price
        2. low             — raw low price
        3. close           — raw close price
        4. tick_volume     — raw tick volume
        5. body_ratio      — |close - open| / (high - low)
        6. upper_wick      — (high - max(open, close)) / (high - low)
        7. lower_wick      — (min(open, close) - low) / (high - low)
        8. range_pct       — (high - low) / close  (normalized range)
        9. return_1        — close / prev_close - 1  (1-bar return)
       10. return_3        — close / close_3_ago - 1  (3-bar return)

    Raw prices are NOT normalized — the TFT's internal LayerNorm
    handles scale differences. Derived features (ratios, returns)
    are already naturally normalized.

    Args:
        df: DataFrame with columns: open, high, low, close, tick_volume.
            Must have at least 1 row. Extra columns are ignored.

    Returns:
        np.ndarray of shape (n_candles, 11), dtype float32.
        Returns empty (0, 11) array if df is empty.
    """
    n = len(df)
    if n == 0:
        return np.zeros((0, N_FEATURES), dtype=np.float32)

    result = np.zeros((n, N_FEATURES), dtype=np.float32)

    # Extract raw OHLCV arrays
    o = df['open'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    c = df['close'].values.astype(np.float64)
    v = df['tick_volume'].values.astype(np.float64)

    # ── Raw features (indices 0-4) ──
    result[:, 0] = o   # open
    result[:, 1] = h   # high
    result[:, 2] = l   # low
    result[:, 3] = c   # close
    result[:, 4] = v   # tick_volume

    # ── Derived features (indices 5-10) ──
    # Protect against zero-range candles (doji / data errors)
    hl_range = h - l
    hl_safe = np.where(np.abs(hl_range) < 1e-10, 1e-10, hl_range)

    # body_ratio: |close - open| / (high - low)
    result[:, 5] = np.abs(c - o) / hl_safe

    # upper_wick: (high - max(open, close)) / (high - low)
    result[:, 6] = (h - np.maximum(o, c)) / hl_safe

    # lower_wick: (min(open, close) - low) / (high - low)
    result[:, 7] = (np.minimum(o, c) - l) / hl_safe

    # range_pct: (high - low) / close  (normalized range)
    close_safe = np.where(np.abs(c) < 1e-10, 1e-10, c)
    result[:, 8] = hl_range / close_safe

    # return_1: close / prev_close - 1
    # First candle has no previous → 0.0
    if n >= 2:
        prev_close = np.where(np.abs(c[:-1]) < 1e-10, 1e-10, c[:-1])
        result[1:, 9] = (c[1:] / prev_close) - 1.0
    # result[0, 9] stays 0.0

    # return_3: close / close_3_bars_ago - 1
    # First 3 candles have no 3-bar-ago → 0.0
    if n >= 4:
        close_3ago = np.where(np.abs(c[:-3]) < 1e-10, 1e-10, c[:-3])
        result[3:, 10] = (c[3:] / close_3ago) - 1.0
    # result[0:3, 10] stays 0.0

    # Replace any inf/nan with 0.0
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    return result.astype(np.float32)


# ════════════════════════════════════════════════════════════════
#  TARGET COMPUTATION
# ════════════════════════════════════════════════════════════════

def compute_targets(golden_moment: dict) -> dict:
    """Compute the 3 TFT training targets from a golden moment.

    Targets:
        candle_pattern_match:
            Binary — 1.0 if the pattern was profitable
            (forward_return > 0.3 R-multiples), else 0.0.

        momentum_score:
            Continuous in [-1, 1] — direction * normalized magnitude.
            sign(forward_return) * min(|forward_return| / 3.0, 1.0)

        reversal_probability:
            Continuous in [0, 1] — estimated probability that the
            move was a reversal rather than continuation.
            Computed from ATR context and momentum velocity from
            the golden moment's engineered features.

    Args:
        golden_moment: Dict from rpde_pattern_scans with keys:
            forward_return (float), direction (str), move_pips (float),
            is_win (int/bool), atr_at_entry (float),
            features (dict, the 93 engineered features or None)

    Returns:
        Dict with keys: candle_pattern_match, momentum_score,
        reversal_probability (all floats).
    """
    forward_return = float(golden_moment.get('forward_return', 0.0))
    direction = str(golden_moment.get('direction', '')).upper()
    move_pips = float(golden_moment.get('move_pips', 0.0))
    atr = float(golden_moment.get('atr_at_entry', 0.0))
    features = golden_moment.get('features')

    # ── Target 1: candle_pattern_match ──
    # Was the pattern profitable? 0.3 R threshold filters noise.
    candle_pattern_match = 1.0 if forward_return > 0.3 else 0.0

    # ── Target 2: momentum_score ──
    # Normalized directional momentum in [-1, 1]
    if abs(forward_return) < 1e-10:
        momentum_score = 0.0
    else:
        sign = 1.0 if forward_return >= 0 else -1.0
        magnitude = min(abs(forward_return) / 3.0, 1.0)
        momentum_score = sign * magnitude

    # ── Target 3: reversal_probability ──
    # Estimate whether the big move was a reversal or continuation.
    # Uses engineered features from the golden moment's feature snapshot.
    reversal_probability = 0.0

    if isinstance(features, dict) and len(features) > 0:
        atr_percentile = float(features.get('ap_atr_percentile', 50.0))
        momentum_velocity = float(features.get('vs_momentum_velocity', 0.0))
        atr_ratio = float(features.get('ap_atr_ratio', 1.0))

        # Normalize momentum velocity to a rough scale
        # Negative velocity = bearish momentum, positive = bullish
        mom_abs = min(abs(momentum_velocity), 3.0)

        # Check if the move direction was AGAINST the prevailing momentum
        # (i.e., it was a reversal)
        if direction == 'BUY' and momentum_velocity < -0.2:
            # Bearish momentum → but price went UP → reversal
            # Higher ATR percentile = more volatile = stronger reversal signal
            reversal_probability = 0.3 + 0.5 * (atr_percentile / 100.0)
            reversal_probability *= (0.5 + 0.5 * min(mom_abs / 2.0, 1.0))

        elif direction == 'SELL' and momentum_velocity > 0.2:
            # Bullish momentum → but price went DOWN → reversal
            reversal_probability = 0.3 + 0.5 * (atr_percentile / 100.0)
            reversal_probability *= (0.5 + 0.5 * min(mom_abs / 2.0, 1.0))

        elif direction in ('BUY', 'SELL'):
            # Move was WITH the prevailing momentum → continuation
            reversal_probability = max(0.0, 0.15 - 0.25 * (atr_percentile / 100.0))

        # ATR ratio adjustment: extreme ATR suggests exhaustion → higher reversal prob
        if atr_ratio > 1.5:
            reversal_probability = min(1.0, reversal_probability + 0.15)

    else:
        # No engineered features available — use simple ATR heuristic
        if atr > 0 and move_pips > 0:
            # Large move relative to ATR could indicate reversal after exhaustion
            move_atr_ratio = move_pips / atr
            reversal_probability = min(0.7, max(0.0, move_atr_ratio * 0.05))

    reversal_probability = float(np.clip(reversal_probability, 0.0, 1.0))

    return {
        'candle_pattern_match': candle_pattern_match,
        'momentum_score': momentum_score,
        'reversal_probability': reversal_probability,
    }


# ════════════════════════════════════════════════════════════════
#  MULTI-TIMEFRAME DATASET
# ════════════════════════════════════════════════════════════════

class MultiTFDataset(Dataset):
    """PyTorch Dataset for multi-timeframe TFT training.

    Each sample corresponds to a golden moment (big move) from the
    RPDE scanner. For each sample, multi-TF candle sequences ending
    at (or before) the golden moment timestamp are fetched from MT5,
    and 11 per-candle features are computed.

    Additionally, the 93 engineered features from the golden moment's
    feature snapshot are included as static context.

    Dataset items are dicts with:
        features: Dict[str, FloatTensor(seq_len, 11)]
            Per-timeframe feature tensors keyed by TF name.
        targets: FloatTensor(3)
            [candle_pattern_match, momentum_score, reversal_probability]
        data_available: FloatTensor(n_timeframes)
            1.0 if candles were available for that TF, 0.0 if padded.
        context: FloatTensor(n_context_features)
            Engineered feature vector from the golden moment snapshot.
            Up to 93 dimensions; 0-padded if fewer features available.

    Args:
        pair: Currency pair string (e.g. 'EURJPY').
        golden_moments: List of dicts from load_golden_moments().
        timeframes: Dict mapping TF name → sequence length.
            Defaults to rpde.config.TFT_TIMEFRAMES.
        feature_names: List of 11 per-candle feature names.
            Defaults to rpde.config.TFT_RAW_FEATURES.
    """

    def __init__(
        self,
        pair: str,
        golden_moments: List[dict],
        timeframes: Optional[Dict[str, int]] = None,
        feature_names: Optional[List[str]] = None,
    ):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for MultiTFDataset")

        self.pair = pair.upper()
        self.golden_moments = golden_moments or []
        self.timeframes = timeframes or TFT_TIMEFRAMES
        self.feature_names = feature_names or TFT_RAW_FEATURES
        self.n_features = len(self.feature_names)
        self.tf_names = list(self.timeframes.keys())
        self.n_timeframes = len(self.tf_names)

        # Discovered context feature names (set during _build_samples)
        self.context_feature_names: List[str] = []
        self.n_context_features: int = 0

        # Pre-compute all samples (candles, targets, context)
        self.samples: List[dict] = []
        if self.golden_moments:
            self.samples = self._build_samples()

        log.info(
            f"[TFT_DATASET] {self.pair}: {len(self.samples)} samples built "
            f"from {len(self.golden_moments)} golden moments "
            f"({self.n_timeframes} TFs, {self.n_features} features, "
            f"{self.n_context_features} context dims)"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Return a single sample dict.

        Returns:
            Dict with keys: features, targets, data_available, context.
            All values are torch.FloatTensor tensors.
        """
        return self.samples[idx]

    # ──────────────────────────────────────────────────────────
    #  SAMPLE CONSTRUCTION
    # ──────────────────────────────────────────────────────────

    def _build_samples(self) -> List[dict]:
        """Build all samples by fetching candles and computing features.

        Optimized to fetch candles once per timeframe, then slice for
        each golden moment. This avoids redundant MT5 calls.

        Returns:
            List of sample dicts ready for __getitem__.
        """
        t0 = time.time()

        # ── Step 1: Parse and sort golden moment timestamps ──
        valid_moments = self._parse_moments()
        if not valid_moments:
            log.warning(f"[TFT_DATASET] {self.pair}: no valid golden moments")
            return []

        moments, timestamps = zip(*valid_moments)
        oldest = min(timestamps)
        newest = max(timestamps)

        log.info(
            f"[TFT_DATASET] {self.pair}: {len(moments)} valid moments, "
            f"range: {oldest} → {newest}"
        )

        # ── Step 2: Fetch candles for each timeframe ──
        candles_cache = self._fetch_all_timeframes(oldest, newest)

        # ── Step 3: Discover context feature names ──
        self.context_feature_names = self._discover_context_features(moments)
        self.n_context_features = len(self.context_feature_names)

        # ── Step 4: Build individual samples ──
        samples = []
        skipped = 0
        for moment, ts in valid_moments:
            sample = self._build_single_sample(moment, ts, candles_cache)
            if sample is not None:
                samples.append(sample)
            else:
                skipped += 1

        duration = round(time.time() - t0, 2)
        log.info(
            f"[TFT_DATASET] {self.pair}: built {len(samples)} samples "
            f"({skipped} skipped) in {duration}s"
        )
        return samples

    def _parse_moments(self) -> List[Tuple[dict, pd.Timestamp]]:
        """Parse and validate golden moment timestamps.

        Returns:
            List of (moment_dict, naive_timestamp) tuples.
            Moments without valid timestamps are excluded.
        """
        valid = []
        for m in self.golden_moments:
            ts = _strip_tz(m.get('bar_timestamp'))
            if pd.notna(ts):
                valid.append((m, ts))

        # Sort chronologically (oldest first) for efficient candle fetching
        valid.sort(key=lambda x: x[1])
        return valid

    def _fetch_all_timeframes(
        self, oldest: pd.Timestamp, newest: pd.Timestamp
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """Fetch candle data for all timeframes, covering the full date range.

        For each timeframe, calculates how many bars are needed to span
        from `oldest` to `newest`, fetches them in one call, and caches
        the DataFrame.

        Args:
            oldest: Earliest golden moment timestamp.
            newest: Latest golden moment timestamp.

        Returns:
            Dict mapping TF name → DataFrame or None.
        """
        from data_layer.price_feed import get_candles

        cache: Dict[str, Optional[pd.DataFrame]] = {}
        time_span_seconds = (newest - oldest).total_seconds()

        for tf_name, seq_len in self.timeframes.items():
            tf_seconds = _tf_to_seconds(tf_name)

            # Calculate bars needed: time span + sequence length + indicator warmup buffer
            # get_candles calls dropna() after indicators, so ~200 bars may be lost
            needed_bars = int(time_span_seconds / tf_seconds) + seq_len + 300
            # MT5 copy_rates_from_pos limit
            needed_bars = min(needed_bars, 100_000)
            # Minimum: at least 3x the sequence length
            needed_bars = max(needed_bars, seq_len * 3)

            try:
                df = get_candles(self.pair, tf_name, needed_bars)

                if df is not None and not df.empty:
                    # Ensure time is datetime
                    df['time'] = pd.to_datetime(df['time'])
                    # Strip timezone for consistent comparison
                    if df['time'].dt.tz is not None:
                        df['time'] = df['time'].dt.tz_convert(None)

                    # Keep only the columns we need
                    required_cols = {'time', 'open', 'high', 'low', 'close', 'tick_volume'}
                    if not required_cols.issubset(set(df.columns)):
                        log.warning(
                            f"[TFT_DATASET] {self.pair} {tf_name}: missing columns, "
                            f"have {list(df.columns)}"
                        )
                        cache[tf_name] = None
                        continue

                    cache[tf_name] = df
                    log.info(
                        f"[TFT_DATASET] {self.pair} {tf_name}: "
                        f"{len(df)} bars "
                        f"({df['time'].iloc[0]} → {df['time'].iloc[-1]})"
                    )
                else:
                    cache[tf_name] = None
                    log.warning(
                        f"[TFT_DATASET] {self.pair} {tf_name}: "
                        f"get_candles returned empty"
                    )

            except Exception as e:
                cache[tf_name] = None
                log.error(
                    f"[TFT_DATASET] {self.pair} {tf_name}: "
                    f"fetch failed: {e}"
                )

        return cache

    def _discover_context_features(
        self, moments: List[dict]
    ) -> List[str]:
        """Discover all engineered feature keys across all golden moments.

        Collects the union of all feature dict keys, sorts them
        alphabetically, and takes the first 93. This ensures a
        consistent feature ordering across all samples.

        Args:
            moments: List of golden moment dicts.

        Returns:
            Sorted list of feature key strings (up to 93).
        """
        all_keys: set = set()
        for m in moments:
            features = m.get('features')
            if isinstance(features, dict):
                # Only include numeric values (skip string encodings, etc.)
                for key, val in features.items():
                    if isinstance(val, (int, float, bool)):
                        all_keys.add(key)
                    elif isinstance(val, str):
                        pass  # Skip string-encoded features
                    # Nested dicts / lists are skipped

        sorted_keys = sorted(all_keys)[:_MAX_CONTEXT_DIM]
        log.info(
            f"[TFT_DATASET] {self.pair}: discovered "
            f"{len(all_keys)} numeric context features, "
            f"using {len(sorted_keys)}"
        )
        return sorted_keys

    def _build_single_sample(
        self,
        moment: dict,
        ts: pd.Timestamp,
        candles_cache: Dict[str, Optional[pd.DataFrame]],
    ) -> Optional[dict]:
        """Build a single training sample from one golden moment.

        For each timeframe, slices candles up to the golden moment
        timestamp and computes 11 per-candle features. If insufficient
        candles are available, zero-pads and marks data_available=0.

        Args:
            moment: Golden moment dict from database.
            ts: Naive timestamp of the golden moment.
            candles_cache: Pre-fetched candle DataFrames per TF.

        Returns:
            Sample dict with features, targets, data_available, context.
            Returns None if the moment is unusable (no candles at all).
        """
        features: Dict[str, 'torch.Tensor'] = {}
        data_available_flags: List[float] = []

        any_tf_available = False

        for tf_name in self.tf_names:
            seq_len = self.timeframes[tf_name]
            df = candles_cache.get(tf_name)

            if df is None or df.empty:
                # No data at all → zero tensor, mark unavailable
                features[tf_name] = torch.zeros(seq_len, self.n_features)
                data_available_flags.append(0.0)
                continue

            # Slice candles up to (and including) the golden moment timestamp
            mask = df['time'] <= ts
            available = df.loc[mask]

            if len(available) < 2:
                # Not even 2 bars → essentially no data
                features[tf_name] = torch.zeros(seq_len, self.n_features)
                data_available_flags.append(0.0)
                continue

            if len(available) >= seq_len:
                # Ideal case: take the last seq_len candles
                window = available.iloc[-seq_len:]
                feat_array = compute_candle_features(window)
                features[tf_name] = torch.from_numpy(feat_array)
                data_available_flags.append(1.0)
                any_tf_available = True
            else:
                # Partial data: use what we have, zero-pad the rest
                feat_array = compute_candle_features(available)
                n_available = len(feat_array)
                padded = np.zeros((seq_len, self.n_features), dtype=np.float32)
                # Place available data at the END of the sequence
                # (most recent candles are the most relevant)
                padded[seq_len - n_available:] = feat_array
                features[tf_name] = torch.from_numpy(padded)

                # Mark as partially available if we have at least 50% of
                # the expected sequence length
                avail_ratio = n_available / seq_len
                data_available_flags.append(1.0 if avail_ratio >= 0.5 else 0.0)
                any_tf_available = True

        # Skip sample if NO timeframe had any data
        if not any_tf_available:
            return None

        # ── Compute targets ──
        targets = compute_targets(moment)
        targets_tensor = torch.FloatTensor([
            targets['candle_pattern_match'],
            targets['momentum_score'],
            targets['reversal_probability'],
        ])

        # ── Extract context features ──
        moment_features = moment.get('features')
        context = np.zeros(self.n_context_features, dtype=np.float32)
        if isinstance(moment_features, dict):
            for i, key in enumerate(self.context_feature_names):
                val = moment_features.get(key, 0.0)
                try:
                    context[i] = float(val)
                except (TypeError, ValueError):
                    context[i] = 0.0
            # Replace any inf/nan
            context = np.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            'features': features,
            'targets': targets_tensor,
            'data_available': torch.FloatTensor(data_available_flags),
            'context': torch.from_numpy(context),
        }


# ════════════════════════════════════════════════════════════════
#  CUSTOM COLLATE FUNCTION
# ════════════════════════════════════════════════════════════════

class MultiTFCollateFn:
    """Custom collate function for DataLoader with MultiTFDataset.

    Handles variable-length sequences per timeframe within a batch
    by padding shorter sequences to the max length in that batch.
    Also creates attention masks so the model can ignore padded tokens.

    Usage:
        dataset = MultiTFDataset(pair, golden_moments)
        loader = DataLoader(dataset, batch_size=32,
                            collate_fn=MultiTFCollateFn())
    """

    def __call__(self, batch: List[dict]) -> dict:
        """Collate a list of sample dicts into batched tensors.

        Args:
            batch: List of dicts from MultiTFDataset.__getitem__().
                Each dict has: features, targets, data_available, context.

        Returns:
            Batched dict with:
                features: Dict[str, FloatTensor(B, max_seq, 11)]
                    Padded feature tensors per timeframe.
                attention_masks: Dict[str, FloatTensor(B, max_seq)]
                    1.0 for real data, 0.0 for padding.
                targets: FloatTensor(B, 3)
                data_available: FloatTensor(B, n_timeframes)
                context: FloatTensor(B, n_context_features)
        """
        batch_size = len(batch)
        tf_names = list(batch[0]['features'].keys())
        n_features = batch[0]['features'][tf_names[0]].size(1)

        collated_features: Dict[str, 'torch.Tensor'] = {}
        collated_masks: Dict[str, 'torch.Tensor'] = {}

        for tf_name in tf_names:
            # Gather all sequences for this timeframe
            sequences = [item['features'][tf_name] for item in batch]
            seq_lens = [s.size(0) for s in sequences]
            max_len = max(seq_lens)

            # Create padded tensors
            padded = torch.zeros(batch_size, max_len, n_features)
            masks = torch.zeros(batch_size, max_len)

            for i, seq in enumerate(sequences):
                seq_len = seq.size(0)
                padded[i, :seq_len, :] = seq
                masks[i, :seq_len] = 1.0

            collated_features[tf_name] = padded
            collated_masks[tf_name] = masks

        # Stack targets: (B, 3)
        targets = torch.stack([item['targets'] for item in batch])

        # Stack data_available: (B, n_timeframes)
        data_available = torch.stack([item['data_available'] for item in batch])

        # Stack context: (B, n_context_features)
        contexts = torch.stack([item['context'] for item in batch])

        return {
            'features': collated_features,
            'attention_masks': collated_masks,
            'targets': targets,
            'data_available': data_available,
            'context': contexts,
        }


# ════════════════════════════════════════════════════════════════
#  TRAINING DATASET BUILDER
# ════════════════════════════════════════════════════════════════

def build_training_dataset(
    pair: str,
    days: Optional[int] = None,
) -> MultiTFDataset:
    """Build a MultiTFDataset for a currency pair from golden moments.

    Loads golden moments from the RPDE database, optionally filters
    by recency, and constructs a PyTorch Dataset with multi-TF
    candle sequences for each moment.

    Args:
        pair: Currency pair string (e.g. 'EURJPY').
        days: Optional maximum age of golden moments in days.
            If None, all golden moments are used.

    Returns:
        MultiTFDataset instance ready for DataLoader.

    Raises:
        RuntimeError: If PyTorch is not installed.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to build TFT dataset")

    from rpde.database import load_golden_moments, load_pattern_trades

    pair_upper = pair.upper()
    t0 = time.time()

    log.info(f"[TFT_DATASET] Building training dataset for {pair_upper}...")

    # ── Load golden moments ──
    golden_moments = load_golden_moments(pair=pair_upper)

    if not golden_moments:
        log.warning(
            f"[TFT_DATASET] {pair_upper}: no golden moments found in database"
        )
        return MultiTFDataset(pair_upper, [])

    # ── Optionally filter by recency ──
    if days is not None and days > 0:
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_ts = pd.Timestamp(cutoff)
        if cutoff_ts.tzinfo is not None:
            cutoff_ts = cutoff_ts.tz_convert(None)

        original_count = len(golden_moments)
        golden_moments = [
            m for m in golden_moments
            if _strip_tz(m.get('bar_timestamp')) >= cutoff_ts
        ]
        filtered = original_count - len(golden_moments)
        if filtered > 0:
            log.info(
                f"[TFT_DATASET] {pair_upper}: filtered {filtered} moments "
                f"older than {days} days, {len(golden_moments)} remaining"
            )

    # ── Check pair trade count ──
    pattern_trades = load_pattern_trades(pair=pair_upper)
    n_trades = len(pattern_trades)
    if n_trades < TFT_MIN_PAIR_TRADES:
        log.warning(
            f"[TFT_DATASET] {pair_upper}: only {n_trades} pattern trades "
            f"(minimum {TFT_MIN_PAIR_TRADES}). Dataset may be unreliable."
        )

    # ── Check minimum sample count ──
    if len(golden_moments) < TFT_MIN_TRAINING_SAMPLES:
        log.warning(
            f"[TFT_DATASET] {pair_upper}: only {len(golden_moments)} golden "
            f"moments (minimum {TFT_MIN_TRAINING_SAMPLES}). "
            f"Training quality may be poor."
        )

    # ── Build dataset ──
    dataset = MultiTFDataset(pair_upper, golden_moments)

    duration = round(time.time() - t0, 2)
    log.info(
        f"[TFT_DATASET] {pair_upper}: training dataset ready — "
        f"{len(dataset)} samples, {n_trades} trades, {duration}s"
    )

    return dataset


# ════════════════════════════════════════════════════════════════
#  LIVE INPUT BUILDER
# ════════════════════════════════════════════════════════════════

def build_live_inputs(pair: str) -> Dict[str, 'torch.Tensor']:
    """Fetch the latest multi-TF candles for live TFT prediction.

    Retrieves the most recent candles for each configured timeframe
    and computes the 11 per-candle features. Returns tensors ready
    for direct model input.

    On any MT5 failure, returns zero-filled tensors for that
    timeframe so the model can still produce a prediction (using
    the data_available flags to know which TFs are missing).

    Args:
        pair: Currency pair string (e.g. 'EURJPY').

    Returns:
        Dict mapping TF name → FloatTensor(seq_len, 11).
        Always contains all configured timeframes.
    """
    if not _TORCH_AVAILABLE:
        log.error("[TFT_DATASET] PyTorch required for live inputs")
        return {
            tf: torch.zeros(seq_len, N_FEATURES)
            for tf, seq_len in TFT_TIMEFRAMES.items()
        }

    from data_layer.price_feed import get_candles

    pair_upper = pair.upper()
    result: Dict[str, 'torch.Tensor'] = {}

    for tf_name, seq_len in TFT_TIMEFRAMES.items():
        try:
            # Fetch extra bars to account for indicator warmup + dropna
            count = seq_len + 250
            df = get_candles(pair_upper, tf_name, count)

            if df is not None and not df.empty:
                # Need at least seq_len usable bars
                if len(df) >= seq_len:
                    window = df.tail(seq_len)
                    feat_array = compute_candle_features(window)
                    result[tf_name] = torch.from_numpy(feat_array)
                    log.debug(
                        f"[TFT_DATASET] Live {pair_upper} {tf_name}: "
                        f"{len(window)} candles"
                    )
                else:
                    # Partial data — zero pad
                    log.warning(
                        f"[TFT_DATASET] Live {pair_upper} {tf_name}: "
                        f"only {len(df)} bars (need {seq_len}), padding"
                    )
                    feat_array = compute_candle_features(df)
                    padded = np.zeros((seq_len, N_FEATURES), dtype=np.float32)
                    n_avail = len(feat_array)
                    padded[seq_len - n_avail:] = feat_array
                    result[tf_name] = torch.from_numpy(padded)
            else:
                log.warning(
                    f"[TFT_DATASET] Live {pair_upper} {tf_name}: "
                    f"no candle data, using zeros"
                )
                result[tf_name] = torch.zeros(seq_len, N_FEATURES)

        except Exception as e:
            log.error(
                f"[TFT_DATASET] Live {pair_upper} {tf_name}: "
                f"fetch error: {e}"
            )
            result[tf_name] = torch.zeros(seq_len, N_FEATURES)

    log.info(
        f"[TFT_DATASET] Live inputs ready for {pair_upper}: "
        f"{list(result.keys())}"
    )
    return result


# ════════════════════════════════════════════════════════════════
#  DATALOADER FACTORY
# ════════════════════════════════════════════════════════════════

def build_dataloaders(
    dataset: MultiTFDataset,
    batch_size: int = 32,
    val_split: float = None,
    num_workers: int = 0,
) -> Tuple[Any, Any, 'MultiTFDataset', 'MultiTFDataset']:
    """Create training and validation DataLoaders from a MultiTFDataset.

    Performs a time-based split: the first (1-val_split) fraction
    of samples goes to training, the rest to validation. This
    preserves temporal ordering and prevents look-ahead bias.

    Args:
        dataset: Built MultiTFDataset instance.
        batch_size: Batch size for both loaders.
        val_split: Validation fraction. Defaults to TFT_TRAIN_VAL_SPLIT.
        num_workers: DataLoader worker processes. Default 0 for safety.

    Returns:
        Tuple of (train_loader, val_loader, train_dataset, val_dataset).
        Returns (None, None, None, None) if PyTorch is unavailable or
        dataset is too small.
    """
    if not _TORCH_AVAILABLE:
        return None, None, None, None

    from torch.utils.data import DataLoader

    if val_split is None:
        val_split = TFT_TRAIN_VAL_SPLIT

    n_total = len(dataset)
    if n_total < 10:
        log.warning(
            f"[TFT_DATASET] Dataset too small for split: {n_total} samples"
        )
        return None, None, None, None

    # Time-based split (golden moments are sorted chronologically
    # in MultiTFDataset._build_samples)
    split_idx = int(n_total * val_split)
    split_idx = max(1, min(split_idx, n_total - 1))

    # Create subset datasets
    train_dataset = MultiTFDataset.__new__(MultiTFDataset)
    train_dataset.pair = dataset.pair
    train_dataset.golden_moments = dataset.golden_moments
    train_dataset.timeframes = dataset.timeframes
    train_dataset.feature_names = dataset.feature_names
    train_dataset.n_features = dataset.n_features
    train_dataset.tf_names = dataset.tf_names
    train_dataset.n_timeframes = dataset.n_timeframes
    train_dataset.context_feature_names = dataset.context_feature_names
    train_dataset.n_context_features = dataset.n_context_features
    train_dataset.samples = dataset.samples[:split_idx]

    val_dataset = MultiTFDataset.__new__(MultiTFDataset)
    val_dataset.pair = dataset.pair
    val_dataset.golden_moments = dataset.golden_moments
    val_dataset.timeframes = dataset.timeframes
    val_dataset.feature_names = dataset.feature_names
    val_dataset.n_features = dataset.n_features
    val_dataset.tf_names = dataset.tf_names
    val_dataset.n_timeframes = dataset.n_timeframes
    val_dataset.context_feature_names = dataset.context_feature_names
    val_dataset.n_context_features = dataset.n_context_features
    val_dataset.samples = dataset.samples[split_idx:]

    collate_fn = MultiTFCollateFn()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    log.info(
        f"[TFT_DATASET] DataLoaders: "
        f"train={len(train_dataset)}, val={len(val_dataset)}, "
        f"batch={batch_size}, split={val_split:.0%}"
    )

    return train_loader, val_loader, train_dataset, val_dataset
