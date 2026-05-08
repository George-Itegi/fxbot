# =============================================================
# rpde/dashboard_panels.py  —  Streamlit Dashboard Panels (Phase 3)
#
# PURPOSE: Streamlit panels for the RPDE system's live monitoring
# dashboard. Each panel is a self-contained Streamlit component
# that can be composed into a larger dashboard layout.
#
# PANELS:
#   1. PatternLibraryPanel   — Browse pattern library (per-pair)
#   2. FusionSignalPanel     — XGB vs TFT signal comparison
#   3. RLConfidencePanel     — RL agent confidence / action probs
#   4. SafetyGuardPanel      — Safety guard pass/skip/shutdown
#   5. LearningHealthPanel   — Retrain schedules & model versions
#
# USAGE:
#   import streamlit as st
#   from rpde.dashboard_panels import render_rpde_dashboard
#
#   # Or individual panels:
#   from rpde.dashboard_panels import PatternLibraryPanel
#   PatternLibraryPanel.render(st)
#
# DESIGN DECISIONS:
#   - Each panel class has a static render(st) method
#   - Data fetching is encapsulated inside each panel
#   - Graceful degradation: missing data shows placeholder
#   - No external dependencies beyond Streamlit
# =============================================================

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ════════════════════════════════════════════════════════════════
#  PATTERN LIBRARY PANEL
# ════════════════════════════════════════════════════════════════

class PatternLibraryPanel:
    """
    Streamlit panel for browsing the RPDE pattern library.

    Shows per-pair patterns with tier, win rate, profit factor,
    expected R, and occurrence count. Supports filtering by pair
    and minimum tier.
    """

    @staticmethod
    def _load_patterns(pair: Optional[str] = None,
                       active_only: bool = True) -> List[Dict]:
        """Load patterns from the pattern library database."""
        try:
            from rpde.pattern_library import PatternLibrary
            lib = PatternLibrary()
            if pair:
                patterns = lib.get_patterns_for_pair(
                    pair, active_only=active_only
                )
            else:
                patterns = lib.get_all_patterns(active_only=active_only)
            return patterns or []
        except Exception:
            return []

    @staticmethod
    def render(st, pair: Optional[str] = None,
               active_only: bool = True) -> None:
        """
        Render the pattern library panel.

        Args:
            st:          Streamlit module or container.
            pair:        Optional pair filter (e.g., "EURJPY").
            active_only: Whether to show only active patterns.
        """
        st.subheader("Pattern Library")

        patterns = PatternLibraryPanel._load_patterns(pair, active_only)

        if not patterns:
            st.info("No patterns found. Run the pipeline first: "
                    "`python -m rpde pipeline`")
            return

        # Summary metrics
        total = len(patterns)
        god_tier = sum(1 for p in patterns if p.get("tier") == "GOD_TIER")
        strong = sum(1 for p in patterns if p.get("tier") == "STRONG")
        valid = sum(1 for p in patterns if p.get("tier") == "VALID")
        probationary = sum(1 for p in patterns if p.get("tier") == "PROBATIONARY")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Patterns", total)
        col2.metric("GOD_TIER", god_tier, delta=None)
        col3.metric("STRONG", strong, delta=None)
        col4.metric("VALID", valid, delta=None)
        col5.metric("PROBATIONARY", probationary, delta=None)

        # Group by pair
        by_pair: Dict[str, List[Dict]] = {}
        for p in patterns:
            p_name = p.get("pair", "UNKNOWN")
            if p_name not in by_pair:
                by_pair[p_name] = []
            by_pair[p_name].append(p)

        tier_rank = {
            "GOD_TIER": 4, "STRONG": 3, "VALID": 2, "PROBATIONARY": 1,
        }

        for pair_name in sorted(by_pair.keys()):
            pair_pats = by_pair[pair_name]

            with st.expander(
                f"{pair_name} ({len(pair_pats)} patterns)", expanded=False
            ):
                pair_pats.sort(
                    key=lambda x: (
                        tier_rank.get(x.get("tier", ""), 0),
                        x.get("win_rate", 0),
                    ),
                    reverse=True,
                )

                for p in pair_pats:
                    pid = p.get("pattern_id", "?")
                    direction = p.get("direction", "?")
                    tier = p.get("tier", "?")
                    occ = p.get("occurrences", 0)
                    wr = p.get("win_rate", 0)
                    pf = p.get("profit_factor", 0)
                    er = p.get("avg_expected_r", 0)

                    col_a, col_b, col_c = st.columns([3, 2, 2])
                    col_a.write(f"**{direction}** | {tier}")
                    col_b.write(
                        f"WR: {wr:.1%} | PF: {pf:.2f} | E[R]: {er:.2f}"
                    )
                    col_c.write(f"Occ: {occ}")


# ════════════════════════════════════════════════════════════════
#  FUSION SIGNAL PANEL
# ════════════════════════════════════════════════════════════════

class FusionSignalPanel:
    """
    Streamlit panel showing XGBoost vs TFT fusion signal comparison.

    Displays the last fusion result for each pair, including
    individual model contributions, agreement status, and the
    fused output (direction, confidence, expected R).
    """

    @staticmethod
    def _load_fusion_history(pairs: Optional[List[str]] = None) -> List[Dict]:
        """Load recent fusion results from database or live state."""
        try:
            from rpde.database import RPDEDatabase
            db = RPDEDatabase()
            rows = db.fetch_recent_fusion_signals(
                pairs=pairs, limit=20
            )
            return rows or []
        except Exception:
            return []

    @staticmethod
    def render(st, pairs: Optional[List[str]] = None) -> None:
        """
        Render the fusion signal panel.

        Args:
            st:    Streamlit module or container.
            pairs: Optional list of pairs to show.
        """
        st.subheader("Fusion Signals (XGB + TFT)")

        signals = FusionSignalPanel._load_fusion_history(pairs)

        if not signals:
            st.info("No fusion signals recorded yet. "
                    "The live engine must process M5 bars first.")
            return

        for sig in signals:
            pair = sig.get("pair", "?")
            direction = sig.get("direction", "?")
            confidence = sig.get("combined_confidence", 0)
            expected_r = sig.get("combined_expected_r", 0)
            agreement = sig.get("signal_agreement", "PARTIAL")
            xgb_conf = sig.get("xgb_confidence", 0)
            tft_contrib = sig.get("tft_contribution", 0)

            with st.expander(
                f"{pair}: {direction} (conf={confidence:.2f}, "
                f"E[R]={expected_r:.2f})"
            ):
                col1, col2, col3 = st.columns(3)
                col1.metric("XGB Confidence", f"{xgb_conf:.3f}")
                col2.metric("TFT Contribution", f"{tft_contrib:.3f}")
                col3.metric("Agreement", agreement)

                if agreement == "AGREE":
                    st.success("XGB and TFT agree on direction")
                elif agreement == "DISAGREE":
                    st.warning("XGB and TFT disagree - reduced confidence")
                else:
                    st.info("Partial agreement")


# ════════════════════════════════════════════════════════════════
#  RL CONFIDENCE PANEL
# ════════════════════════════════════════════════════════════════

class RLConfidencePanel:
    """
    Streamlit panel showing RL agent confidence and action probabilities.

    Displays the latest RL decision for each pair, including the
    chosen action, confidence score, predicted state value, and
    action probability distribution.
    """

    @staticmethod
    def _load_rl_decisions(pairs: Optional[List[str]] = None) -> List[Dict]:
        """Load recent RL decisions from database or live state."""
        try:
            from rpde.database import RPDEDatabase
            db = RPDEDatabase()
            rows = db.fetch_recent_rl_decisions(
                pairs=pairs, limit=20
            )
            return rows or []
        except Exception:
            return []

    @staticmethod
    def render(st, pairs: Optional[List[str]] = None) -> None:
        """
        Render the RL confidence panel.

        Args:
            st:    Streamlit module or container.
            pairs: Optional list of pairs to show.
        """
        st.subheader("RL Agent Confidence")

        decisions = RLConfidencePanel._load_rl_decisions(pairs)

        if not decisions:
            st.info("No RL decisions recorded yet. "
                    "The live engine must process M5 bars first.")
            return

        for dec in decisions:
            pair = dec.get("pair", "?")
            action = dec.get("action_name", "?")
            confidence = dec.get("confidence", 0)
            value = dec.get("value", 0)
            entry = dec.get("entry", False)
            direction = dec.get("direction", "?")
            size_r = dec.get("size_r", 0)

            entry_str = "YES" if entry else "SKIP"

            with st.expander(
                f"{pair}: {action} ({entry_str})"
            ):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Action", action)
                col2.metric("Confidence", f"{confidence:.3f}")
                col3.metric("Value (V(s))", f"{value:.3f}")
                col4.metric("Size (R)", f"{size_r:.1f}R")

                if entry:
                    st.success(
                        f"Entry signal: {direction} @ {size_r:.1f}R "
                        f"(confidence={confidence:.2f})"
                    )
                else:
                    st.warning("RL agent decided to SKIP this signal")

                # Action probabilities if available
                probs = dec.get("action_probabilities")
                if probs and isinstance(probs, dict):
                    st.bar_chart(probs)


# ════════════════════════════════════════════════════════════════
#  SAFETY GUARD PANEL
# ════════════════════════════════════════════════════════════════

class SafetyGuardPanel:
    """
    Streamlit panel showing safety guard status.

    Displays the current state of each safety guard:
    PASS (green), SKIP (yellow), or SHUTDOWN (red).
    Also shows system-wide safety status and shutdown reason
    if applicable.
    """

    @staticmethod
    def _load_safety_status() -> Dict[str, Any]:
        """Load current safety guard system status."""
        try:
            from rpde.safety_guards import SafetyGuardSystem
            system = SafetyGuardSystem()
            return system.get_full_status()
        except Exception:
            return {}

    @staticmethod
    def render(st) -> None:
        """
        Render the safety guard panel.

        Args:
            st: Streamlit module or container.
        """
        st.subheader("Safety Guards")

        status = SafetyGuardPanel._load_safety_status()

        if not status:
            st.info("Safety guard system not available.")
            return

        is_shutdown = status.get("is_shutdown", False)

        if is_shutdown:
            st.error(
                f"SYSTEM SHUTDOWN: {status.get('shutdown_reason', 'Unknown')}"
            )
        else:
            st.success("Safety system: ACTIVE (no shutdown)")

        # Guard-by-guard status
        guards = status.get("guards", [])
        if guards:
            for guard in guards:
                name = guard.get("name", "?")
                state = guard.get("state", "PASS")
                message = guard.get("message", "")
                severity = guard.get("severity", "INFO")

                if state == "PASS":
                    st.markdown(f"- **{name}**: :green[{state}]")
                elif state == "SKIP":
                    st.markdown(f"- **{name}**: :orange[{state}] — {message}")
                elif state == "SHUTDOWN":
                    st.markdown(
                        f"- **{name}**: :red[{state}] — {message}"
                    )
                else:
                    st.markdown(f"- **{name}**: {state}")

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Checks", status.get("total_checks", 0))
        col2.metric("Passes", status.get("passes", 0))
        col3.metric("Blocks", status.get("blocks", 0))


# ════════════════════════════════════════════════════════════════
#  LEARNING HEALTH PANEL
# ════════════════════════════════════════════════════════════════

class LearningHealthPanel:
    """
    Streamlit panel showing continuous learning loop health.

    Displays retrain schedules for each component (XGB, TFT, RL),
    model version info, buffer sizes, and whether retraining is
    currently due.
    """

    @staticmethod
    def _load_learning_status() -> Dict[str, Any]:
        """Load continuous learning loop health status."""
        try:
            from rpde.experience_buffer import ContinuousLearningLoop
            loop = ContinuousLearningLoop()
            return loop.get_system_health()
        except Exception:
            return {}

    @staticmethod
    def render(st) -> None:
        """
        Render the learning health panel.

        Args:
            st: Streamlit module or container.
        """
        st.subheader("Learning Health")

        health = LearningHealthPanel._load_learning_status()

        if not health:
            st.info("Continuous learning loop not available.")
            return

        # Summary
        total_exp = health.get("total_experiences", 0)
        active_bufs = health.get("active_buffers", 0)
        learning_active = health.get("learning_active", False)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Experiences", total_exp)
        col2.metric("Active Buffers", active_bufs)
        col3.metric(
            "Learning Active",
            "YES" if learning_active else "NO",
        )

        # Retrain schedule
        schedule = health.get("schedule", {})
        if schedule:
            st.markdown("### Retrain Schedule")
            for component, info in schedule.items():
                status_str = info.get("status", "?")
                due = info.get("due", False)
                last = info.get("last_retrained", "never")

                if due:
                    st.warning(
                        f"**{component}**: :red[DUE] — "
                        f"last retrained: {last}"
                    )
                elif status_str == "TRAINED":
                    st.markdown(
                        f"- **{component}**: :green[TRAINED] — "
                        f"last: {last}"
                    )
                elif status_str == "NOT_TRAINED":
                    st.markdown(f"- **{component}**: :orange[NOT TRAINED]")
                else:
                    st.markdown(f"- **{component}**: {status_str}")

        # Buffer sizes
        buffers = health.get("buffers", {})
        if buffers:
            st.markdown("### Experience Buffers")
            for pair_name, buf_info in buffers.items():
                total_trades = buf_info.get("total_experiences", 0)
                last_trade = str(
                    buf_info.get("last_trade_time", "never")
                )[:19]
                col1, col2 = st.columns(2)
                col1.write(f"**{pair_name}**: {total_trades} trades")
                col2.write(f"Last: {last_trade}")


# ════════════════════════════════════════════════════════════════
#  COMPOSITE DASHBOARD RENDERER
# ════════════════════════════════════════════════════════════════

def render_rpde_dashboard(st, pairs: Optional[List[str]] = None) -> None:
    """
    Render the full RPDE dashboard with all panels.

    This is the main entry point for the Streamlit dashboard.
    Call it from your Streamlit app:

        import streamlit as st
        from rpde.dashboard_panels import render_rpde_dashboard
        render_rpde_dashboard(st)

    Args:
        st:    Streamlit module.
        pairs: Optional list of pairs to filter panels by.
    """
    st.title("RPDE v5.0 — Reverse Pattern Discovery Engine")
    st.caption("Self-evolving trading intelligence system")

    # Top-level layout: 3 columns
    top_col1, top_col2, top_col3 = st.columns(3)

    with top_col1:
        SafetyGuardPanel.render(st)

    with top_col2:
        RLConfidencePanel.render(st, pairs=pairs)

    with top_col3:
        LearningHealthPanel.render(st)

    # Bottom: wider panels
    bot_col1, bot_col2 = st.columns(2)

    with bot_col1:
        FusionSignalPanel.render(st, pairs=pairs)

    with bot_col2:
        PatternLibraryPanel.render(st)

    # Footer info
    st.divider()
    st.caption("RPDE v5.0 | Phase 1: Pattern Discovery + "
               "Phase 2: TFT + Phase 3: RL Decision Engine")
