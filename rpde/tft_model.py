# =============================================================
# rpde/tft_model.py  —  Multi-Timeframe Temporal Fusion Transformer
#
# Apex Trader v5.0  |  RPDE Phase 2 — Deep Temporal Architecture
#
# PURPOSE: Learn candle-level temporal patterns across H4, H1, M15,
# and M5 timeframes that the 93-feature XGBoost model cannot capture.
# While XGBoost sees "what the market looks like right now" (a single
# snapshot), the TFT sees "how the candles have been evolving" across
# four granularities simultaneously.
#
# ARCHITECTURE:
#   ┌─────────────────────────────────────────────────────┐
#   │  Per-Timeframe Pipeline (one per TF)                │
#   │  ┌───────────┐   ┌─────────────┐   ┌────────────┐ │
#   │  │    VSN    │ → │ TF Encoder  │ → │ Global Pool │ │
#   │  │ (feature  │   │ (learned    │   │ (mean over  │ │
#   │  │  select.) │   │  pos+causal)│   │  sequence)  │ │
#   │  └───────────┘   └─────────────┘   └──────┬─────┘ │
#   └────────────────────────────────────────────┼───────┘
#                                                │
#            ┌───────────────────────────────────┤
#            │  4 pooled vectors:                │
#            │  (B, hidden) × 4                  │
#            ▼                                   ▼
#   ┌──────────────────────┐     ┌──────────────────────┐
#   │ Cross-TF Attention   │     │   Pair Embedding     │
#   │ (mutual attention    │     │   (learned per pair) │
#   │  between all 4 TFs)  │     └──────────┬───────────┘
#   └──────────┬───────────┘                │
#              │ 4 context-enriched vectors  │
#              ▼                            ▼
#   ┌──────────────────────────────────────────────┐
#   │  Concat: [ctx_H4, ..., ctx_M5,              │
#   │           pair_embed,                        │
#   │           context_encoded (93 features)]     │
#   │        → MLP → ReLU → Dropout → MLP         │
#   └──────────────┬───────────────────────────────┘
#                  │
#       ┌──────────┼──────────┐
#       ▼          ▼          ▼
#   ┌───────┐ ┌────────┐ ┌──────────────┐
#   │ CPM   │ │ MOM    │ │ REV_PROB     │
#   │ (cls) │ │ (reg)  │ │ (prob)       │
#   └───────┘ └────────┘ └──────────────┘
#
# DESIGN PRINCIPLES:
#   - Learned positional encoding (different seq lengths per TF)
#   - Causal masking: candles can only attend to prior candles
#   - Padding masks: padded positions are properly ignored
#   - Static context encoder: 93 engineered features as additional input
#   - VSN: per-timeframe feature importance (interpretability)
#   - Pair embedding: pair-specific personality injection
#   - Multi-task: 3 heads for pattern match, momentum, reversal
#   - GPU-auto with graceful CPU fallback
#   - Mixed precision (fp16) for GPU training
#   - No external deps beyond PyTorch + numpy
# =============================================================

from __future__ import annotations

import os
import json
import math
import time
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any

import numpy as np

from core.logger import get_logger

log = get_logger(__name__)

# ── Check torch availability ──────────────────────────────
_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    _TORCH_AVAILABLE = True
except ImportError:
    log.warning(
        "[TFT_MODEL] PyTorch not installed — TFT unavailable. "
        "Install: pip install torch"
    )

# ── Defaults from rpde.config (used when no config dict provided) ─
_DEFAULT_CONFIG = {
    "TFT_TIMEFRAMES": {"H4": 30, "H1": 48, "M15": 96, "M5": 60},
    "TFT_RAW_FEATURES": 11,
    "TFT_HIDDEN_SIZE": 64,
    "TFT_ATTENTION_HEADS": 4,
    "TFT_NUM_ENCODER_LAYERS": 2,
    "TFT_DROPOUT": 0.15,
    "TFT_LEARNING_RATE": 1e-4,
    "TFT_WEIGHT_DECAY": 1e-5,
    "TFT_BATCH_SIZE": 32,
    "TFT_EPOCHS": 50,
    "TFT_PATIENCE": 8,
    "TFT_GRADIENT_CLIP": 1.0,
    "VSN_ENABLED": True,
    "VSN_HIDDEN_SIZE": 16,
    "CROSS_TF_ATTENTION_ENABLED": True,
    "CROSS_TF_ATTENTION_HEADS": 4,
    "CROSS_TF_ATTENTION_LAYERS": 2,
    "TFT_OUTPUT_HEADS": {
        "candle_pattern_match": {"type": "classification", "hidden_dim": 32},
        "momentum_score": {"type": "regression", "hidden_dim": 32},
        "reversal_probability": {"type": "regression", "hidden_dim": 32},
    },
    "TFT_DEVICE": "auto",
    "TFT_MIXED_PRECISION": True,
    "TFT_NUM_WORKERS": 2,
}


def _cfg(config: Optional[dict], key: str, default: Any = None) -> Any:
    """Safely get a value from config dict with fallback to defaults."""
    return config.get(key, _DEFAULT_CONFIG.get(key, default))


# ════════════════════════════════════════════════════════════════
#  DEVICE MANAGEMENT
# ════════════════════════════════════════════════════════════════

def get_device(config: Optional[dict] = None) -> torch.device:
    """
    Determine the best available device for TFT operations.

    Args:
        config: Optional config dict. Uses TFT_DEVICE setting.

    Returns:
        torch.device — 'cuda' if available and configured, else 'cpu'
    """
    if not _TORCH_AVAILABLE:
        return torch.device("cpu")

    device_str = _cfg(config, "TFT_DEVICE", "auto")
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def get_gpu_info() -> dict:
    """Return GPU availability and specs for logging."""
    if not _TORCH_AVAILABLE or not torch.cuda.is_available():
        return {"available": False, "device": "cpu"}
    return {
        "available": True,
        "device": "cuda",
        "name": torch.cuda.get_device_name(0),
        "memory_total_gb": round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 2
        ),
        "cuda_version": torch.version.cuda,
    }


# ════════════════════════════════════════════════════════════════
#  VARIABLE SELECTION NETWORK (VSN)
#  Per-timeframe feature selection with importance scoring
# ════════════════════════════════════════════════════════════════

class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN) — per-timeframe feature selection.

    For each input feature (column) of one timeframe:
      1. Process through an independent GRU across the sequence dimension
      2. Generate a scalar importance score from the final hidden state
      3. Softmax across all features → selection weights

    The weighted sum of features gives the selected representation.
    A ``variable_importance`` attribute (batch, n_features) is set after
    each forward pass for interpretability.

    Args:
        n_features: Number of input features per candle (e.g. 11)
        hidden_size: GRU hidden dimension for feature processing
        output_dim: Dimension of the output representation
        dropout: Dropout rate
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        output_dim: int,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.output_dim = output_dim

        # Per-feature GRU: each feature is processed independently
        # Input to GRU: 1 → hidden_size (processes single value at each time step)
        self.feature_grus = nn.ModuleList([
            nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True)
            for _ in range(n_features)
        ])

        # Selection weight computation: hidden_size → 1
        self.weight_layer = nn.Linear(hidden_size, 1)

        # Feature projection: n_features → output_dim (weighted combination)
        self.feature_proj = nn.Linear(n_features, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

        # Stores importance after forward pass
        self.variable_importance: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features) — raw feature tensor for one TF

        Returns:
            (batch, seq_len, output_dim) — selected representation
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Process each feature independently through its GRU
        # Each GRU takes a (batch, seq_len, 1) slice
        feature_hidden_states = []
        for i in range(self.n_features):
            feat = x[:, :, i : i + 1]  # (batch, seq_len, 1)
            _, h_n = self.feature_grus[i](feat)  # h_n: (1, batch, hidden_size)
            feature_hidden_states.append(h_n.squeeze(0))  # (batch, hidden_size)

        # Stack: (batch, n_features, hidden_size)
        stacked = torch.stack(feature_hidden_states, dim=1)

        # Compute selection weights: (batch, n_features, hidden_size) → (batch, n_features)
        weights = self.weight_layer(stacked).squeeze(-1)  # (batch, n_features)
        weights = F.softmax(weights, dim=-1)  # (batch, n_features)

        # Store importance for interpretability (detached from computation graph)
        self.variable_importance = weights.detach()

        # Apply weights: (batch, n_features) × (batch, n_features, seq_len) → (batch, n_features, seq_len)
        # Transpose x for weighting: (batch, seq_len, n_features) → (batch, n_features, seq_len)
        x_t = x.transpose(1, 2)  # (batch, n_features, seq_len)
        weighted = weights.unsqueeze(-1) * x_t  # (batch, n_features, seq_len)

        # Project: (batch, n_features, seq_len) → (batch, output_dim, seq_len) → (batch, seq_len, output_dim)
        # We need to project along the feature dimension at each time step
        # Reshape: (batch, seq_len, n_features) to apply linear
        weighted = weighted.transpose(1, 2)  # (batch, seq_len, n_features)
        selected = self.feature_proj(weighted)  # (batch, seq_len, output_dim)
        selected = self.layer_norm(selected)
        selected = self.dropout(selected)

        return selected

    def extra_repr(self) -> str:
        return (
            f"n_features={self.n_features}, "
            f"hidden_size={self.hidden_size}, "
            f"output_dim={self.output_dim}"
        )


# ════════════════════════════════════════════════════════════════
#  STATIC CONTEXT ENCODER
#  Encodes the 93-engineered feature vector into the fusion stage
# ════════════════════════════════════════════════════════════════

class StaticContextEncoder(nn.Module):
    """
    Static Context Encoder — projects the 93-engineered feature vector
    into the TFT's hidden dimension for fusion alongside candle embeddings.

    Rather than forcing the TFT to re-learn market structure from raw OHLCV,
    this encoder lets the model directly consume the rich features your
    engine already computes (delta, ADX, RSI, VWAP position, order flow,
    strategy scores, session info, volatility regime, spread, etc.).

    Architecture: 3-layer MLP with LayerNorm and dropout.
        context (n_context) → Linear → GELU → LN → Dropout
                               → Linear → GELU → LN → Dropout
                               → Linear → hidden_size

    Args:
        n_context: Number of engineered context features (up to 93).
        hidden_size: Output dimension (matches TFT hidden_size).
        dropout: Dropout rate.
    """

    def __init__(self, n_context: int = 93, hidden_size: int = 64,
                 dropout: float = 0.15):
        super().__init__()
        self.n_context = n_context
        self.hidden_size = hidden_size

        # 3-layer projection MLP
        intermediate = max(hidden_size * 2, 128)
        self.context_encoder = nn.Sequential(
            nn.Linear(n_context, intermediate),
            nn.GELU(),
            nn.LayerNorm(intermediate),
            nn.Dropout(dropout),
            nn.Linear(intermediate, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: (batch, n_context) — engineered feature vector.

        Returns:
            (batch, hidden_size) — encoded context representation.
        """
        return self.output_norm(self.context_encoder(context))

    def extra_repr(self) -> str:
        return (f"n_context={self.n_context}, "
                f"hidden_size={self.hidden_size}")


# ════════════════════════════════════════════════════════════════
#  CAUSAL MASK UTILITY
# ════════════════════════════════════════════════════════════════

def _generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Generate an additive causal (upper-triangular) mask.

    Positions can only attend to themselves and prior positions.
    Masked positions get -inf so softmax produces 0 attention weight.

    Args:
        seq_len: Sequence length.
        device: Torch device.

    Returns:
        (seq_len, seq_len) float tensor with -inf in the upper triangle.
    """
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device), diagonal=1
    )
    return mask.masked_fill(mask == 1, float('-inf'))


# ════════════════════════════════════════════════════════════════
#  TIMEFRAME ENCODER
#  Learned positional encoding + causal masking + TransformerEncoder
# ════════════════════════════════════════════════════════════════

class TimeframeEncoder(nn.Module):
    """
    Encodes a single timeframe's VSN-selected feature sequence using
    learned positional encoding and stacked TransformerEncoder layers.

    Uses LEARNED positional embeddings (not sinusoidal) because each
    timeframe has a different sequence length (H4=30, H1=48, M15=96, M5=60).

    Uses CAUSAL MASKING so each candle can only attend to itself and
    prior candles. This prevents the model from learning patterns that
    require hindsight — critical for realistic live trading performance.

    Args:
        seq_len: Sequence length for this timeframe
        hidden_size: Transformer hidden dimension (must match VSN output)
        n_heads: Number of self-attention heads
        n_layers: Number of stacked TransformerEncoder layers
        dropout: Dropout rate
    """

    def __init__(
        self,
        seq_len: int,
        hidden_size: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        # Learned positional encoding: one vector per position
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, seq_len, hidden_size)
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # Stacked TransformerEncoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(hidden_size),
        )

        self.dropout = nn.Dropout(dropout)

        # Causal mask: pre-computed for the max sequence length.
        # Registered as buffer so it moves with the model to the correct device.
        causal = torch.triu(
            torch.ones(seq_len, seq_len), diagonal=1
        )
        causal = causal.masked_fill(causal == 1, float('-inf'))
        self.register_buffer('causal_mask', causal)

        # Initialize positional embedding
        self._init_weights()

    def _init_weights(self):
        """Initialize positional embedding with small values."""
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x: torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_size) — output from VSN
            src_key_padding_mask: Optional (batch, seq_len) bool tensor.
                True = padded position to ignore.

        Returns:
            (batch, seq_len, hidden_size) — encoded representation
        """
        seq_len_actual = x.size(1)

        # Add learned positional encoding
        x = x + self.pos_embedding[:, :seq_len_actual, :]
        x = self.dropout(x)

        # Build causal mask sized to actual sequence length.
        # Slice from the pre-computed full mask (registered buffer).
        causal = self.causal_mask[:seq_len_actual, :seq_len_actual]

        # Combine causal mask with padding mask if provided.
        # PyTorch TransformerEncoder expects:
        #   src_mask: (S, S) additive float mask
        #   src_key_padding_mask: (B, S) bool mask (True = ignore)
        x = self.transformer(
            x,
            mask=causal,
            src_key_padding_mask=src_key_padding_mask,
        )

        return x

    def extra_repr(self) -> str:
        return (
            f"seq_len={self.seq_len}, "
            f"hidden_size={self.hidden_size}"
        )


# ════════════════════════════════════════════════════════════════
#  CROSS-TIMEFRAME ATTENTION
#  Mutual attention between pooled timeframe representations
# ════════════════════════════════════════════════════════════════

class CrossTimeframeAttention(nn.Module):
    """
    Cross-Timeframe Attention — enables information sharing between
    all 4 timeframe representations.

    Pipeline:
      1. Global mean-pool each timeframe sequence → 4 vectors (B, H)
      2. Stack into a single sequence: (B, 4, H) where position = TF index
      3. Apply stacked self-attention layers — all TFs attend to each other
      4. Split back into 4 context-enriched vectors

    Args:
        hidden_size: Hidden dimension
        n_heads: Number of attention heads
        n_layers: Number of stacked attention layers
        dropout: Dropout rate
    """

    def __init__(
        self,
        hidden_size: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Learnable TF-type embeddings so the model knows which position
        # corresponds to which timeframe
        self.tf_type_embedding = nn.Parameter(
            torch.zeros(1, 4, hidden_size)
        )
        nn.init.trunc_normal_(self.tf_type_embedding, std=0.02)

        # Stacked transformer encoder for mutual attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(hidden_size),
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self, tf_sequences: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Args:
            tf_sequences: List of 4 tensors, each (batch, seq_len_i, hidden_size)

        Returns:
            List of 4 tensors, each (batch, hidden_size) — context-enriched
        """
        # Global mean pooling for each TF
        pooled = [seq.mean(dim=1) for seq in tf_sequences]  # each (B, H)

        # Stack: (B, 4, H)
        stacked = torch.stack(pooled, dim=1)

        # Add TF-type embedding (learnable positional encoding for TF index)
        stacked = stacked + self.tf_type_embedding

        # Self-attention: all TFs attend to all TFs
        attended = self.transformer(stacked)  # (B, 4, H)

        # Final layer norm
        attended = self.layer_norm(attended)

        # Split back into individual context vectors
        context_vectors = [attended[:, i, :] for i in range(4)]

        return context_vectors

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}"


# ════════════════════════════════════════════════════════════════
#  TEMPORAL FUSION TRANSFORMER (Main Model)
# ════════════════════════════════════════════════════════════════

class TemporalFusionTransformer(nn.Module):
    """
    Multi-Timeframe Temporal Fusion Transformer — the full model.

    Architecture:
      1. Per-timeframe: VSN → TimeframeEncoder → encoded sequence
      2. Global pool each → Cross-TF Attention → context vectors
      3. Concat with pair embedding → MLP → 3 output heads

    Output heads:
      - candle_pattern_match: (batch, 1) — binary classification
      - momentum_score: (batch, 1) — regression [-1, 1]
      - reversal_probability: (batch, 1) — probability [0, 1]

    Args:
        pair: Currency pair string (e.g. 'EURJPY') for pair embedding
        config: Dict of TFT hyperparameters (from rpde.config or defaults)
    """

    def __init__(self, pair: str, config: Optional[dict] = None):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for TFT model")

        super().__init__()

        self.pair = pair.upper()
        self.config = config or {}

        # Unpack config
        self.timeframes = dict(_cfg(self.config, "TFT_TIMEFRAMES"))
        self.tf_names = list(self.timeframes.keys())  # ["H4", "H1", "M15", "M5"]
        self.n_timeframes = len(self.tf_names)
        self.n_features = _cfg(self.config, "TFT_RAW_FEATURES", 11)
        self.hidden_size = _cfg(self.config, "TFT_HIDDEN_SIZE", 64)
        self.dropout_rate = _cfg(self.config, "TFT_DROPOUT", 0.15)
        self.n_heads = _cfg(self.config, "TFT_ATTENTION_HEADS", 4)
        self.n_encoder_layers = _cfg(self.config, "TFT_NUM_ENCODER_LAYERS", 2)
        self.vsn_enabled = _cfg(self.config, "VSN_ENABLED", True)
        self.vsn_hidden = _cfg(self.config, "VSN_HIDDEN_SIZE", 16)
        self.cross_tf_enabled = _cfg(self.config, "CROSS_TF_ATTENTION_ENABLED", True)
        self.cross_tf_heads = _cfg(self.config, "CROSS_TF_ATTENTION_HEADS", 4)
        self.cross_tf_layers = _cfg(self.config, "CROSS_TF_ATTENTION_LAYERS", 2)
        self.output_heads_cfg = _cfg(self.config, "TFT_OUTPUT_HEADS", {})
        self.n_context_features = _cfg(self.config, "TFT_CONTEXT_FEATURES", 93)

        # ── Pair Embedding ──────────────────────────────────
        # Learned embedding that gives the model pair-specific personality
        self.pair_embed_dim = self.hidden_size
        self.pair_embedding = nn.Embedding(1, self.pair_embed_dim)
        nn.init.trunc_normal_(self.pair_embedding.weight, std=0.02)

        # ── Per-Timeframe VSN + Encoder ─────────────────────
        self.vsns = nn.ModuleDict()
        self.encoders = nn.ModuleDict()

        for tf_name in self.tf_names:
            seq_len = self.timeframes[tf_name]

            # VSN: n_features → hidden_size
            self.vsns[tf_name] = VariableSelectionNetwork(
                n_features=self.n_features,
                hidden_size=self.vsn_hidden,
                output_dim=self.hidden_size,
                dropout=self.dropout_rate,
            )

            # TimeframeEncoder: hidden_size → hidden_size
            self.encoders[tf_name] = TimeframeEncoder(
                seq_len=seq_len,
                hidden_size=self.hidden_size,
                n_heads=self.n_heads,
                n_layers=self.n_encoder_layers,
                dropout=self.dropout_rate,
            )

        # ── Cross-Timeframe Attention ───────────────────────
        self.cross_tf_attn = CrossTimeframeAttention(
            hidden_size=self.hidden_size,
            n_heads=self.cross_tf_heads,
            n_layers=self.cross_tf_layers,
            dropout=self.dropout_rate,
        )

        # ── Static Context Encoder ─────────────────────────
        # Projects the 93-engineered features into hidden_size for fusion
        self.context_encoder = StaticContextEncoder(
            n_context=self.n_context_features,
            hidden_size=self.hidden_size,
            dropout=self.dropout_rate,
        )

        # ── Final Fusion MLP ────────────────────────────────
        # Input: 4 context vectors + pair embedding + context encoding
        fusion_input_dim = (
            self.n_timeframes * self.hidden_size
            + self.pair_embed_dim
            + self.hidden_size  # context encoder output
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.fusion_norm = nn.LayerNorm(self.hidden_size)

        # ── Output Heads ────────────────────────────────────
        self.output_heads = nn.ModuleDict()
        for head_name, head_cfg in self.output_heads_cfg.items():
            head_type = head_cfg.get("type", "regression")
            head_dim = head_cfg.get("hidden_dim", 32)
            self.output_heads[head_name] = nn.Sequential(
                nn.Linear(self.hidden_size, head_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(head_dim, 1),
            )

        # Initialize all linear weights
        self._init_weights()

        # Move pair embedding index register to device-aware buffer
        self.register_buffer("_pair_idx", torch.zeros(1, dtype=torch.long))

    def _init_weights(self):
        """Xavier uniform initialization for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        multi_tf_inputs: Dict[str, torch.Tensor],
        context: Optional[torch.Tensor] = None,
        attention_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the full Temporal Fusion Transformer.

        Args:
            multi_tf_inputs: Dict of {tf_name: (batch, seq_len, n_features)}
                e.g. {"H4": (B, 30, 11), "H1": (B, 48, 11), ...}
            context: Optional (batch, n_context) — 93-engineered features.
                When provided, encoded via StaticContextEncoder and
                concatenated into the fusion MLP alongside the
                cross-timeframe context vectors and pair embedding.
            attention_masks: Optional dict of {tf_name: (batch, seq_len)}
                bool tensors. True = padded position to ignore.
                Produced by MultiTFCollateFn in the dataset pipeline.

        Returns:
            Dict of {head_name: (batch, 1) prediction tensor}
                - candle_pattern_match: (batch, 1)
                - momentum_score: (batch, 1)
                - reversal_probability: (batch, 1)
        """
        device = self._pair_idx.device

        # Step 1: Per-timeframe encoding pipeline (with causal + padding masks)
        encoded_sequences = []
        for tf_name in self.tf_names:
            x = multi_tf_inputs[tf_name]  # (batch, seq_len, n_features)
            x = self.vsns[tf_name](x)  # (batch, seq_len, hidden_size)

            # Get padding mask for this TF if available
            pad_mask = None
            if attention_masks is not None and tf_name in attention_masks:
                pad_mask = attention_masks[tf_name]  # (batch, seq_len)

            x = self.encoders[tf_name](x, src_key_padding_mask=pad_mask)
            encoded_sequences.append(x)

        # Step 2: Cross-timeframe attention
        context_vectors = self.cross_tf_attn(encoded_sequences)

        # Step 3: Concatenate context vectors + pair embedding + static context
        ctx_concat = torch.cat(context_vectors, dim=-1)  # (batch, n_tf*hidden)
        pair_emb = self.pair_embedding(self._pair_idx).expand(
            ctx_concat.size(0), -1
        )  # (batch, pair_embed_dim)

        # Encode static context features (93 engineered features)
        if context is not None:
            ctx_encoded = self.context_encoder(context)  # (batch, hidden)
            fused = torch.cat(
                [ctx_concat, pair_emb, ctx_encoded], dim=-1
            )  # (batch, n_tf*H + embed + H)
        else:
            # Backward compatible: no context features provided
            fused = torch.cat([ctx_concat, pair_emb], dim=-1)

        # Step 4: Fusion MLP
        fused = self.fusion_mlp(fused)  # (batch, hidden)
        fused = self.fusion_norm(fused)  # (batch, hidden)

        # Step 5: Multi-head output
        outputs = {}
        for head_name, head_layer in self.output_heads.items():
            outputs[head_name] = head_layer(fused)  # (batch, 1)

        return outputs

    def get_feature_importance(self) -> Dict[str, torch.Tensor]:
        """
        Get per-timeframe feature importance from VSN modules.

        Returns:
            Dict mapping tf_name → (batch, n_features) importance tensor
        """
        importance = {}
        for tf_name in self.tf_names:
            vsn = self.vsns[tf_name]
            if vsn.variable_importance is not None:
                importance[tf_name] = vsn.variable_importance
        return importance

    def count_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return (
            f"pair={self.pair}, "
            f"timeframes={self.tf_names}, "
            f"hidden_size={self.hidden_size}, "
            f"n_features={self.n_features}, "
            f"n_encoder_layers={self.n_encoder_layers}, "
            f"output_heads={list(self.output_heads.keys())}"
        )

    def __repr__(self) -> str:
        lines = [
            f"TemporalFusionTransformer(",
            f"  pair={self.pair},",
            f"  timeframes={self.tf_names},",
            f"  hidden_size={self.hidden_size},",
            f"  n_features={self.n_features},",
            f"  parameters={self.count_parameters():,},",
            f")",
        ]
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
#  TRAINING FUNCTIONS
# ════════════════════════════════════════════════════════════════

def _compute_loss(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    output_heads_cfg: dict,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute combined multi-task loss.

    Loss per head:
      - classification → BCEWithLogitsLoss
      - regression → MSELoss
      - regression with name "probability" → BCELoss (sigmoid applied)

    Returns:
        (total_loss, per_head_loss_dict)
    """
    total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
    head_losses = {}

    for head_name, head_cfg in output_heads_cfg.items():
        pred = predictions[head_name]  # (B, 1)
        target = targets[head_name]    # (B, 1)
        head_type = head_cfg.get("type", "regression")

        if head_type == "classification":
            # Binary classification: BCE with logits
            loss = F.binary_cross_entropy_with_logits(
                pred.squeeze(-1), target.squeeze(-1).float()
            )
        elif head_name == "reversal_probability":
            # Treated as probability: BCE loss (sigmoid applied)
            pred_sig = torch.sigmoid(pred.squeeze(-1))
            loss = F.binary_cross_entropy(
                pred_sig, target.squeeze(-1).float()
            )
        else:
            # Standard regression: MSE
            loss = F.mse_loss(pred.squeeze(-1), target.squeeze(-1).float())

        total_loss = total_loss + loss
        head_losses[head_name] = loss.item()

    return total_loss, head_losses


def train_tft_model(
    model: TemporalFusionTransformer,
    train_dataset,
    val_dataset,
    config: Optional[dict] = None,
) -> dict:
    """
    Train the TFT model with AdamW, mixed precision, early stopping.

    Args:
        model: TemporalFusionTransformer instance (already on device)
        train_dataset: PyTorch Dataset for training
        val_dataset: PyTorch Dataset for validation
        config: Dict of TFT hyperparameters

    Returns:
        Training result dict with:
            train_loss, val_loss, per_head_losses, epochs_trained,
            best_epoch, best_val_loss, duration_seconds
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for training")

    cfg = config or {}
    device = next(model.parameters()).device

    # Unpack training config
    lr = _cfg(cfg, "TFT_LEARNING_RATE", 1e-4)
    weight_decay = _cfg(cfg, "TFT_WEIGHT_DECAY", 1e-5)
    batch_size = _cfg(cfg, "TFT_BATCH_SIZE", 32)
    max_epochs = _cfg(cfg, "TFT_EPOCHS", 50)
    patience = _cfg(cfg, "TFT_PATIENCE", 8)
    grad_clip = _cfg(cfg, "TFT_GRADIENT_CLIP", 1.0)
    use_mixed_precision = (
        _cfg(cfg, "TFT_MIXED_PRECISION", True)
        and device.type == "cuda"
    )
    num_workers = _cfg(cfg, "TFT_NUM_WORKERS", 2)
    output_heads_cfg = _cfg(cfg, "TFT_OUTPUT_HEADS", {})

    log.info(
        f"[TFT_TRAIN] Training {model.pair} | device={device} | "
        f"lr={lr} | batch={batch_size} | epochs={max_epochs} | "
        f"patience={patience} | amp={use_mixed_precision}"
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # Optimizer + Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(patience // 2, 2),
        min_lr=lr * 1e-3,
        verbose=False,
    )

    # Mixed precision scaler
    scaler = (
        torch.cuda.amp.GradScaler(enabled=use_mixed_precision)
        if use_mixed_precision
        else None
    )

    # Training state
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []
    n_train = len(train_dataset)
    n_val = len(val_dataset)

    log.info(
        f"[TFT_TRAIN] Samples: train={n_train}, val={n_val} | "
        f"Parameters: {model.count_parameters():,}"
    )

    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        # ── Train ────────────────────────────────────────────
        model.train()
        train_loss_sum = 0.0
        train_head_losses_sum: Dict[str, float] = {}
        train_batches = 0

        for batch in train_loader:
            # ── Extract inputs, targets, context, and masks from batch ──
            # CollateFn produces: {features, attention_masks, targets, context}
            batch_features = batch.get("features", {})
            batch_masks = batch.get("attention_masks", {})
            batch_context = batch.get("context", None)
            batch_targets = batch.get("targets")

            # Move TF feature tensors to device
            inputs = {}
            if batch_features:
                for tf_name, tensor in batch_features.items():
                    if isinstance(tensor, torch.Tensor):
                        inputs[tf_name] = tensor.to(device)

            # Move attention masks to device (convert float→bool for PyTorch)
            attn_masks = {}
            if batch_masks:
                for tf_name, tensor in batch_masks.items():
                    if isinstance(tensor, torch.Tensor):
                        # Collate produces 1.0/0.0 floats → convert to bool
                        attn_masks[tf_name] = (tensor.to(device) == 0.0)

            # Move context to device
            ctx_tensor = None
            if batch_context is not None and isinstance(batch_context, torch.Tensor):
                ctx_tensor = batch_context.to(device)

            # Move targets to device
            targets = {}
            if batch_targets is not None:
                if isinstance(batch_targets, torch.Tensor):
                    # (B, 3) tensor — split into per-head targets
                    for i, head_name in enumerate(model.output_heads.keys()):
                        if i < batch_targets.size(1):
                            targets[head_name] = batch_targets[:, i:i+1].to(device)
                elif isinstance(batch_targets, dict):
                    for k, v in batch_targets.items():
                        if isinstance(v, torch.Tensor):
                            targets[k] = v.to(device)

            # Fallback: flat batch format (model tf names directly in batch)
            if not inputs:
                for tf_name in model.tf_names:
                    if tf_name in batch:
                        inputs[tf_name] = batch[tf_name].to(device)
                if not targets:
                    for head_name in model.output_heads.keys():
                        if head_name in batch:
                            targets[head_name] = batch[head_name].to(device)

            optimizer.zero_grad()

            if use_mixed_precision and scaler is not None:
                with torch.cuda.amp.autocast():
                    preds = model(inputs, context=ctx_tensor,
                                 attention_masks=attn_masks)
                    loss, head_losses = _compute_loss(
                        preds, targets, output_heads_cfg
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(inputs, context=ctx_tensor,
                             attention_masks=attn_masks)
                loss, head_losses = _compute_loss(
                    preds, targets, output_heads_cfg
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            train_loss_sum += loss.item()
            for k, v in head_losses.items():
                train_head_losses_sum[k] = (
                    train_head_losses_sum.get(k, 0.0) + v
                )
            train_batches += 1

        avg_train_loss = train_loss_sum / max(train_batches, 1)
        avg_train_heads = {
            k: v / max(train_batches, 1)
            for k, v in train_head_losses_sum.items()
        }

        # ── Validate ─────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_head_losses_sum: Dict[str, float] = {}
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                # Same extraction as training loop
                batch_features = batch.get("features", {})
                batch_masks = batch.get("attention_masks", {})
                batch_context = batch.get("context", None)
                batch_targets = batch.get("targets")

                inputs = {}
                if batch_features:
                    for tf_name, tensor in batch_features.items():
                        if isinstance(tensor, torch.Tensor):
                            inputs[tf_name] = tensor.to(device)

                attn_masks = {}
                if batch_masks:
                    for tf_name, tensor in batch_masks.items():
                        if isinstance(tensor, torch.Tensor):
                            attn_masks[tf_name] = (tensor.to(device) == 0.0)

                ctx_tensor = None
                if batch_context is not None and isinstance(batch_context, torch.Tensor):
                    ctx_tensor = batch_context.to(device)

                targets = {}
                if batch_targets is not None:
                    if isinstance(batch_targets, torch.Tensor):
                        for i, head_name in enumerate(model.output_heads.keys()):
                            if i < batch_targets.size(1):
                                targets[head_name] = batch_targets[:, i:i+1].to(device)
                    elif isinstance(batch_targets, dict):
                        for k, v in batch_targets.items():
                            if isinstance(v, torch.Tensor):
                                targets[k] = v.to(device)

                if not inputs:
                    for tf_name in model.tf_names:
                        if tf_name in batch:
                            inputs[tf_name] = batch[tf_name].to(device)
                    if not targets:
                        for head_name in model.output_heads.keys():
                            if head_name in batch:
                                targets[head_name] = batch[head_name].to(device)

                preds = model(inputs, context=ctx_tensor,
                             attention_masks=attn_masks)
                loss, head_losses = _compute_loss(
                    preds, targets, output_heads_cfg
                )

                val_loss_sum += loss.item()
                for k, v in head_losses.items():
                    val_head_losses_sum[k] = (
                        val_head_losses_sum.get(k, 0.0) + v
                    )
                val_batches += 1

        avg_val_loss = val_loss_sum / max(val_batches, 1)
        avg_val_heads = {
            k: v / max(val_batches, 1)
            for k, v in val_head_losses_sum.items()
        }

        # ── Scheduler step ───────────────────────────────────
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Logging ──────────────────────────────────────────
        epoch_data = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 6),
            "val_loss": round(avg_val_loss, 6),
            "train_heads": {k: round(v, 6) for k, v in avg_train_heads.items()},
            "val_heads": {k: round(v, 6) for k, v in avg_val_heads.items()},
            "lr": round(current_lr, 8),
        }
        history.append(epoch_data)

        if epoch % 5 == 0 or epoch == 1:
            log.debug(
                f"[TFT_TRAIN] {model.pair} E{epoch:03d} | "
                f"train={avg_train_loss:.4f} val={avg_val_loss:.4f} | "
                f"lr={current_lr:.6f}"
            )

        # ── Early stopping ───────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            log.info(
                f"[TFT_TRAIN] {model.pair} Early stop at epoch {epoch} "
                f"(best epoch: {best_epoch}, best val: {best_val_loss:.4f})"
            )
            break

    duration = round(time.time() - t0, 1)
    final_train = history[-1]["train_loss"] if history else 0.0
    final_val = history[-1]["val_loss"] if history else 0.0
    final_heads = history[-1]["val_heads"] if history else {}

    log.info(
        f"[TFT_TRAIN] {model.pair} Training complete: {duration}s | "
        f"best_epoch={best_epoch} best_val={best_val_loss:.4f} | "
        f"epochs_trained={len(history)}"
    )

    return {
        "pair": model.pair,
        "train_loss": round(final_train, 6),
        "val_loss": round(final_val, 6),
        "best_val_loss": round(best_val_loss, 6),
        "per_head_losses": final_heads,
        "epochs_trained": len(history),
        "best_epoch": best_epoch,
        "duration_seconds": duration,
        "history": history,
    }


# ════════════════════════════════════════════════════════════════
#  PERSISTENCE: SAVE / LOAD
# ════════════════════════════════════════════════════════════════

def _get_model_dir(model_dir: Optional[str] = None) -> str:
    """Resolve the default model directory."""
    if model_dir is None:
        return os.path.join(os.path.dirname(__file__), "models", "tft")
    return model_dir


def save_tft_model(
    model: TemporalFusionTransformer,
    pair: str,
    model_dir: Optional[str] = None,
    training_samples: int = 0,
    val_metrics: Optional[dict] = None,
) -> str:
    """
    Save TFT model weights + metadata to disk.

    Saves:
      - model.pt: state_dict + config
      - meta.json: pair, version, trained_at, metrics, config snapshot

    Args:
        model: Trained TemporalFusionTransformer
        pair: Currency pair string
        model_dir: Directory path (default: rpde/models/tft/)
        training_samples: Number of training samples used
        val_metrics: Dict of validation metrics from training

    Returns:
        Path to saved model file
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required")

    mdir = _get_model_dir(model_dir)
    os.makedirs(mdir, exist_ok=True)

    safe_pair = pair.upper().replace("/", "_")
    model_path = os.path.join(mdir, f"{safe_pair}_model.pt")
    meta_path = os.path.join(mdir, f"{safe_pair}_meta.json")

    # Save model weights + config snapshot
    config_snapshot = {}
    for key in _DEFAULT_CONFIG:
        if key in model.config:
            config_snapshot[key] = model.config[key]

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "pair": pair,
            "config": config_snapshot,
            "timeframes": model.timeframes,
            "n_features": model.n_features,
            "hidden_size": model.hidden_size,
        },
        model_path,
    )

    # Save metadata
    meta = {
        "pair": pair,
        "version": "5.0",
        "trained_at": datetime.now().isoformat(),
        "training_samples": training_samples,
        "val_metrics": val_metrics or {},
        "parameters": model.count_parameters(),
        "config_snapshot": config_snapshot,
        "timeframes": model.timeframes,
        "output_heads": list(model.output_heads.keys()),
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    size_kb = round(os.path.getsize(model_path) / 1024, 1)
    log.info(
        f"[TFT_MODEL] Saved {pair} model: {model_path} ({size_kb}KB)"
    )
    return model_path


def load_tft_model(
    pair: str,
    model_dir: Optional[str] = None,
    config: Optional[dict] = None,
) -> Optional[Tuple[TemporalFusionTransformer, dict]]:
    """
    Load a trained TFT model + metadata from disk.

    Args:
        pair: Currency pair string
        model_dir: Directory path (default: rpde/models/tft/)
        config: Optional config dict to merge with saved config

    Returns:
        Tuple of (model, meta_dict) or None if not found / error
    """
    if not _TORCH_AVAILABLE:
        log.warning("[TFT_MODEL] PyTorch not available")
        return None

    mdir = _get_model_dir(model_dir)
    safe_pair = pair.upper().replace("/", "_")
    model_path = os.path.join(mdir, f"{safe_pair}_model.pt")
    meta_path = os.path.join(mdir, f"{safe_pair}_meta.json")

    if not os.path.exists(model_path):
        log.debug(f"[TFT_MODEL] No model for {pair}: {model_path}")
        return None

    # Load metadata
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)

    # Reconstruct config
    saved_config = meta.get("config_snapshot", {})
    merged_config = {**_DEFAULT_CONFIG, **saved_config}
    if config:
        merged_config.update(config)

    device = get_device(merged_config)

    try:
        checkpoint = torch.load(
            model_path, map_location=device, weights_only=False
        )

        # Reconstruct model
        model = TemporalFusionTransformer(
            pair=pair,
            config=merged_config,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        size_kb = round(os.path.getsize(model_path) / 1024, 1)
        log.info(
            f"[TFT_MODEL] Loaded {pair} model: {model_path} "
            f"({size_kb}KB, device={device})"
        )
        return model, meta

    except Exception as e:
        log.error(f"[TFT_MODEL] Failed to load {pair} model: {e}")
        return None


# ════════════════════════════════════════════════════════════════
#  TFT MODEL MANAGER
#  High-level wrapper for production use
# ════════════════════════════════════════════════════════════════

class TFTModelManager:
    """
    High-level manager that wraps the TemporalFusionTransformer.

    Handles:
      - Loading existing models or creating new ones
      - Inference with proper device management
      - Model metadata access
      - Feature importance extraction from VSN
      - GPU OOM graceful fallback

    Usage:
        mgr = TFTModelManager("EURJPY")
        if mgr.is_trained():
            result = mgr.predict(multi_tf_inputs)
            print(result["candle_pattern_match"])
    """

    def __init__(self, pair: str, config: Optional[dict] = None):
        """
        Args:
            pair: Currency pair string
            config: Optional config dict for model creation
        """
        self.pair = pair.upper()
        self.config = config
        self.model: Optional[TemporalFusionTransformer] = None
        self.meta: dict = {}
        self._device = get_device(config)

        # Try to load existing model
        self._load_existing()

    def _load_existing(self) -> None:
        """Attempt to load a trained model from disk."""
        result = load_tft_model(self.pair, config=self.config)
        if result is not None:
            self.model, self.meta = result
            self._device = next(self.model.parameters()).device
            log.debug(
                f"[TFT_MGR] Loaded existing model for {self.pair}"
            )
        else:
            log.debug(
                f"[TFT_MGR] No trained model for {self.pair}, "
                f"call train() before predict()"
            )

    def _create_model(self) -> TemporalFusionTransformer:
        """Create a fresh model instance."""
        model = TemporalFusionTransformer(
            pair=self.pair, config=self.config
        )
        model.to(self._device)
        return model

    def is_trained(self) -> bool:
        """Check if a trained model file exists."""
        safe_pair = self.pair.replace("/", "_")
        mdir = _get_model_dir()
        model_path = os.path.join(mdir, f"{safe_pair}_model.pt")
        return os.path.exists(model_path)

    def train(
        self,
        train_dataset,
        val_dataset,
        config: Optional[dict] = None,
    ) -> dict:
        """
        Train the model and save the result.

        Args:
            train_dataset: PyTorch Dataset for training
            val_dataset: PyTorch Dataset for validation
            config: Optional training config overrides

        Returns:
            Training result dict
        """
        train_cfg = {**(self.config or {}), **(config or {})}

        # Create fresh model
        self.model = self._create_model()

        # Train with OOM fallback
        try:
            result = train_tft_model(
                self.model, train_dataset, val_dataset, train_cfg
            )
        except torch.cuda.OutOfMemoryError:
            log.warning(
                f"[TFT_MGR] GPU OOM for {self.pair}, falling back to CPU"
            )
            self._device = torch.device("cpu")
            self.model = self.model.to(self._device)
            # Retry training config with CPU
            train_cfg["TFT_MIXED_PRECISION"] = False
            result = train_tft_model(
                self.model, train_dataset, val_dataset, train_cfg
            )

        # Save model
        save_tft_model(
            self.model,
            self.pair,
            training_samples=len(train_dataset),
            val_metrics={
                "val_loss": result["val_loss"],
                "best_val_loss": result["best_val_loss"],
                "per_head_losses": result["per_head_losses"],
            },
        )

        # Reload meta
        load_result = load_tft_model(self.pair, config=self.config)
        if load_result is not None:
            self.model, self.meta = load_result

        return result

    @torch.no_grad()
    def predict(
        self,
        multi_tf_inputs: Dict[str, np.ndarray],
    ) -> dict:
        """
        Run inference and return predictions + confidence.

        Args:
            multi_tf_inputs: Dict of {tf_name: (1, seq_len, n_features)} or
                             {tf_name: (seq_len, n_features)} numpy arrays

        Returns:
            Dict with:
                - candle_pattern_match: float (raw logit)
                - momentum_score: float
                - reversal_probability: float (0-1)
                - confidence: float (aggregate confidence 0-1)
                - is_signal: bool
        """
        if self.model is None:
            raise RuntimeError(
                f"No model loaded for {self.pair}. "
                f"Call train() first or ensure model file exists."
            )

        self.model.eval()
        device = next(self.model.parameters()).device

        # Convert numpy inputs to tensors
        tensors = {}
        for tf_name, arr in multi_tf_inputs.items():
            t = torch.tensor(
                arr, dtype=torch.float32, device=device
            )
            if t.dim() == 2:
                t = t.unsqueeze(0)  # add batch dim
            tensors[tf_name] = t

        # Forward pass (with OOM fallback)
        try:
            preds = self.model(tensors)
        except torch.cuda.OutOfMemoryError:
            log.warning(
                f"[TFT_MGR] GPU OOM during predict for {self.pair}, "
                f"falling back to CPU"
            )
            cpu_tensors = {
                k: v.cpu() for k, v in tensors.items()
            }
            preds = self.model(cpu_tensors)

        # Extract predictions
        cpm = float(preds["candle_pattern_match"][0, 0].cpu())
        cpm_prob = float(torch.sigmoid(
            preds["candle_pattern_match"][0, 0]
        ).cpu())
        mom = float(preds["momentum_score"][0, 0].cpu())
        rev = float(torch.sigmoid(
            preds["reversal_probability"][0, 0]
        ).cpu())

        # Confidence: weighted combination
        # Pattern match probability: how certain the pattern is recognized
        # Momentum magnitude: how strong the conviction
        # Reversal complement: 1 - reversal_prob (higher = more directional)
        directional_conf = 1.0 - rev
        momentum_conf = min(1.0, abs(mom))
        confidence = (
            0.45 * cpm_prob
            + 0.35 * momentum_conf
            + 0.20 * directional_conf
        )
        confidence = max(0.0, min(1.0, confidence))

        is_signal = cpm_prob > 0.5 and confidence > 0.6

        return {
            "candle_pattern_match": round(cpm, 6),
            "pattern_match_probability": round(cpm_prob, 4),
            "momentum_score": round(mom, 6),
            "reversal_probability": round(rev, 4),
            "confidence": round(confidence, 4),
            "is_signal": is_signal,
        }

    def get_info(self) -> dict:
        """
        Get model metadata.

        Returns:
            Dict with model info, or empty dict if no model loaded.
        """
        if self.model is None:
            return {
                "pair": self.pair,
                "trained": False,
                "error": "No model loaded",
            }

        return {
            "pair": self.pair,
            "trained": True,
            "version": self.meta.get("version", "unknown"),
            "trained_at": self.meta.get("trained_at", "unknown"),
            "parameters": self.model.count_parameters(),
            "device": str(self._device),
            "timeframes": self.model.tf_names,
            "output_heads": list(self.model.output_heads.keys()),
            "training_samples": self.meta.get("training_samples", 0),
            "val_metrics": self.meta.get("val_metrics", {}),
        }

    def get_feature_importance(self) -> Dict[str, List[float]]:
        """
        Get per-timeframe feature importance from VSN modules.

        Requires that a forward pass has been run (stores VSN weights).

        Returns:
            Dict of {tf_name: [importance_per_feature_float, ...]}
        """
        if self.model is None:
            return {}

        importance = self.model.get_feature_importance()
        result = {}
        for tf_name, tensor in importance.items():
            # Average over batch dimension → (n_features,)
            avg = tensor.mean(dim=0).cpu().numpy()
            result[tf_name] = [round(float(x), 4) for x in avg]
        return result


# ════════════════════════════════════════════════════════════════
#  CONVENIENCE FUNCTIONS
# ════════════════════════════════════════════════════════════════

def create_tft_model(
    pair: str, config: Optional[dict] = None
) -> TemporalFusionTransformer:
    """
    Factory function to create a fresh TFT model.

    Args:
        pair: Currency pair string
        config: Optional config dict

    Returns:
        TemporalFusionTransformer instance on the appropriate device
    """
    model = TemporalFusionTransformer(pair=pair, config=config)
    device = get_device(config)
    model.to(device)
    return model
