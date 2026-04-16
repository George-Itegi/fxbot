# dashboard/components/database_mgmt.py
# Database management — view, clear, export data

import streamlit as st
import pandas as pd
from dashboard.components.db_helper import (
    get_trades, get_signals, clear_all_database_data)


def render():
    st.title("🗄️ Database Management")
    st.warning("⚠️ Be careful — deleting data is permanent and cannot be undone.")

    # Stats
    trades  = get_trades(10000)
    signals = get_signals(10000)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Trades",  len(trades))
    col2.metric("Total Signals", len(signals))
    col3.metric("DB Size",       "—")

    st.markdown("---")

    # ── Export section ─────────────────────────────────────
    st.subheader("📥 Export Data")
    col1, col2 = st.columns(2)
    with col1:
        if not trades.empty:
            csv = trades.to_csv(index=False)
            st.download_button(
                "⬇️ Download Trades CSV",
                data=csv,
                file_name="apex_trades.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with col2:
        if not signals.empty:
            csv2 = signals.to_csv(index=False)
            st.download_button(
                "⬇️ Download Signals CSV",
                data=csv2,
                file_name="apex_signals.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.markdown("---")

    # ── Delete section ─────────────────────────────────────
    st.subheader("🗑️ Delete Data")

    col1, col2 = st.columns(2)
    with col1:
        st.error("**Delete ALL Database Data**\n\n"
                 "This will permanently delete all trades, "
                 "signals, and market snapshots.")

        # Double confirmation required
        confirm1 = st.checkbox("I understand this cannot be undone")
        confirm2 = st.text_input(
            'Type "DELETE ALL" to confirm',
            placeholder="DELETE ALL")

        if st.button("🗑️ DELETE ALL DATA",
                     type="primary",
                     use_container_width=True,
                     disabled=not (confirm1 and confirm2 == "DELETE ALL")):
            with st.spinner("Deleting all data..."):
                result = clear_all_database_data()
            if result['success']:
                st.success("✅ All data deleted successfully!")
                st.json(result['deleted'])
                st.rerun()
            else:
                st.error(f"❌ Delete failed: {result['error']}")

    with col2:
        st.info("**What gets deleted:**\n"
                "- All trade records\n"
                "- All signals (traded + skipped)\n"
                "- All market snapshots\n"
                "- Strategy performance stats\n\n"
                "**What is NOT deleted:**\n"
                "- Trained AI models\n"
                "- Strategy registry\n"
                "- Configuration files\n"
                "- Log files")

    st.markdown("---")

    # Preview data
    st.subheader("👁️ Data Preview")
    tab1, tab2 = st.tabs(["Trades", "Signals"])
    with tab1:
        if trades.empty:
            st.info("No trades in database.")
        else:
            st.dataframe(trades.head(20), use_container_width=True)
    with tab2:
        if signals.empty:
            st.info("No signals in database.")
        else:
            st.dataframe(signals.head(20), use_container_width=True)
