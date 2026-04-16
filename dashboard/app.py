# =============================================================
# dashboard/app.py — APEX TRADER DASHBOARD
# Run with: streamlit run dashboard/app.py
# =============================================================

import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(
    page_title  = "APEX TRADER",
    page_icon   = "📈",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("📈 APEX TRADER")
st.sidebar.markdown("v3.0 | ICMarkets Demo")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [
    "📊 Live Overview",
    "💹 Live Positions",
    "📊 MTF Analysis",
    "📋 Trade Journal",
    "🎯 Strategy Performance",
    "🔍 Market Scanner",
    "🛡️ Risk & Account",
    "🤖 Model Status",
    "🗄️ Database",
])

st.sidebar.markdown("---")

# Live bot status indicator
try:
    import MetaTrader5 as mt5
    from dotenv import load_dotenv
    import os
    load_dotenv()
    mt5.initialize()
    mt5.login(int(os.getenv('MT5_LOGIN', 0)),
              password=os.getenv('MT5_PASSWORD', ''),
              server=os.getenv('MT5_SERVER', ''))
    info = mt5.account_info()
    if info:
        st.sidebar.success(f"🟢 MT5 Connected")
        st.sidebar.metric("Balance", f"${info.balance:,.0f}")
        st.sidebar.metric("Equity",  f"${info.equity:,.0f}",
                          f"{info.equity-info.balance:+.0f}")
    mt5.shutdown()
except Exception:
    st.sidebar.error("🔴 MT5 Disconnected")

# ── Route pages ───────────────────────────────────────────────
if page == "📊 Live Overview":
    from dashboard.components.overview import render
    render()
elif page == "💹 Live Positions":
    from dashboard.components.live_positions import render
    render()
elif page == "📊 MTF Analysis":
    from dashboard.components.mtf_analysis import render
    render()
elif page == "📋 Trade Journal":
    from dashboard.components.trade_journal import render
    render()
elif page == "🎯 Strategy Performance":
    from dashboard.components.strategies import render
    render()
elif page == "🔍 Market Scanner":
    from dashboard.components.scanner import render
    render()
elif page == "🛡️ Risk & Account":
    from dashboard.components.risk import render
    render()
elif page == "🤖 Model Status":
    from dashboard.components.models import render
    render()
elif page == "🗄️ Database":
    from dashboard.components.database_mgmt import render
    render()
