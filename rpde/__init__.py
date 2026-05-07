# =============================================================
# RPDE — Reverse Pattern Discovery Engine (v5.0)
#
# A self-evolving trading intelligence system that discovers
# profitable patterns by reverse-engineering historical market
# moves. Instead of hardcoded strategies, the AI scans price
# data for significant moves, extracts feature snapshots at
# those moments, clusters them into patterns, validates each
# pattern per-pair, and builds XGBoost classifiers to recognize
# them in real-time.
#
# Architecture:
#   Phase 1: Pattern Discovery (Offline) — 100% COMPLETE
#     Historical Data → Big Move Scanner → Feature Snapshots →
#     Pattern Miner → Per-Pair Validator → Pattern Library
#     → Per-Pair XGBoost Pattern Model
#
#   Phase 2: Temporal Fusion Transformer (Multi-TF Deep Learning)
#     Multi-TF Raw OHLCV → Separate Timeframe Encoders →
#     Cross-TF Attention → Variable Selection Network →
#     3 Output Heads → Pattern Fusion Layer → Enhanced Gate
#
#   Phase 3: RL Decision Engine (GPU) — 100% COMPLETE
#     Fused Signal + Market State + Portfolio → RL Agent (PPO) →
#     Entry/Size/Stop/TP/Exit/Skip → Safety Guards → EXECUTE
#     → Experience Replay Buffer → Continuous Learning Loop
#
# Data Flow (Live):
#   Raw OHLCV ─┐
#   (4 timeframes) ├→ TFT Model ─┐
#                  │               ├→ Fusion Layer ─→ RL Agent ─┐
#   93 features ─┘               │                          │
#                              │                          ▼
#   Pattern Library ─────────────┘              Safety Guards → Trade
#                                              ──────────────────────
#                                              Experience Buffer → Learn
#
# Key Design Principles:
#   - Per-pair personality: Each pair gets its own model and patterns
#   - Currency-specific patterns: Shared base currencies boost confidence
#   - Statistical validation: Minimum occurrences, win rate, profit factor
#   - Non-overridable safety guards protect capital at all times
#   - Continuous learning from every trade outcome
#   - Zero interference with v4.2 system (separate tables, separate models)
#
# Modules:
#   scanner.py            — Big Move Scanner (finds golden moments)
#   feature_snapshot.py   — Extracts 93 features at golden moments
#   pattern_miner.py      — DBSCAN clustering into candidate patterns
#   pattern_validator.py  — Statistical validation with tier assignment
#   pattern_library.py    — Per-pair pattern storage (CRUD)
#   pattern_model.py      — Per-pair XGBoost regression model (Phase 1 L1)
#   pattern_gate.py       — Final decision layer with TFT fusion (Phase 1+2 L2)
#   tft_dataset.py        — Multi-TF PyTorch dataset builder (Phase 2)
#   tft_model.py          — Multi-TF TFT architecture (Phase 2)
#   fusion_layer.py       — XGB + TFT prediction fusion (Phase 2)
#   rl_env.py             — RL Trading Environment — Gymnasium (Phase 3)
#   rl_agent.py           — PPO Decision Engine — Actor-Critic (Phase 3)
#   experience_buffer.py  — Experience Replay + Continuous Learning (Phase 3)
#   safety_guards.py      — Non-overridable safety rails (Phase 3)
#   live_engine.py        — Live Trading Orchestrator + MT5 Bridge (Phase 3)
#   dashboard_panels.py   — Streamlit dashboard panels (Phase 3)
#   database.py           — Separate RPDE database tables
#   config.py             — All tunable parameters
#   trainer.py            — Full training pipeline orchestrator
#   __main__.py           — CLI entry point
# =============================================================

__version__ = "5.0.0"

# ── Lazy Exports ──────────────────────────────────────────────
# Phase 3: Live Engine
from rpde.live_engine import LiveEngine, ActiveTrade

# Phase 3: Dashboard Panels (optional — Streamlit may not be installed)
try:
    from rpde.dashboard_panels import (
        PatternLibraryPanel,
        FusionSignalPanel,
        RLConfidencePanel,
        SafetyGuardPanel,
        LearningHealthPanel,
        render_rpde_dashboard,
    )
except ImportError:
    # Streamlit not installed — panels unavailable
    pass

__all__ = [
    "LiveEngine",
    "ActiveTrade",
    "PatternLibraryPanel",
    "FusionSignalPanel",
    "RLConfidencePanel",
    "SafetyGuardPanel",
    "LearningHealthPanel",
    "render_rpde_dashboard",
]
