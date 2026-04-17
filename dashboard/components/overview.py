# dashboard/components/overview.py — Page 1: Live Overview

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timezone
from dashboard.components.db_helper import get_trades, get_open_positions


def render():
    st.title("📊 Live Overview")
    st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")

    # ── Account metrics ───────────────────────────────────────
    _show_account_metrics()

    st.markdown("---")

    # ── Open positions ────────────────────────────────────────
    st.subheader("🔴 Open Positions")
    open_pos = get_open_positions()
    if open_pos.empty:
        st.info("No open positions right now.")
    else:
        st.dataframe(open_pos[[
            'symbol','direction','strategy',
            'entry_price','sl_price','tp_price',
            'lot_size','ai_score','session'
        ]], use_container_width=True)

    st.markdown("---")

    # ── P&L curve ─────────────────────────────────────────────
    st.subheader("📈 Cumulative P&L")
    trades = get_trades(200)
    if trades.empty:
        st.info("No completed trades yet. Run the bot to see results.")
    else:
        _show_pnl_curve(trades)
        _show_recent_trades(trades)

    # Auto-refresh every 30 seconds
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()

def _show_account_metrics():
    """Show key account metrics in metric cards."""
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
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Balance",   f"${info.balance:,.2f}")
            col2.metric("Equity",    f"${info.equity:,.2f}",
                        f"{info.equity - info.balance:+.2f}")
            col3.metric("Margin",    f"${info.margin:,.2f}")
            col4.metric("Free Margin",f"${info.margin_free:,.2f}")
            col5.metric("Leverage",  f"1:{info.leverage}")
        mt5.shutdown()
    except Exception as e:
        col1, col2 = st.columns(2)
        col1.warning("MT5 not connected")
        col2.caption(str(e))


def _show_pnl_curve(trades: pd.DataFrame):
    """Plot cumulative P&L over time."""
    df = trades[trades['profit_loss'].notna()].copy()
    if df.empty:
        st.info("No closed trades with P&L data yet.")
        return
    df = df.sort_values('timestamp_open')
    df['cumulative_pnl'] = df['profit_loss'].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp_open'],
        y=df['cumulative_pnl'],
        mode='lines+markers',
        name='Cumulative P&L',
        line=dict(color='#00ff88', width=2),
        fill='tozeroy',
        fillcolor='rgba(0,255,136,0.1)',
    ))
    fig.update_layout(
        title='Cumulative P&L ($)',
        xaxis_title='Date',
        yaxis_title='P&L ($)',
        height=350,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
    )
    st.plotly_chart(fig, use_container_width=True)


def _show_recent_trades(trades: pd.DataFrame):
    """Show the 10 most recent trades."""
    st.subheader("📋 Recent Trades")
    df = trades.head(10)
    if df.empty:
        return
    # Color-code by outcome
    def color_outcome(val):
        if isinstance(val, str) and 'WIN' in val:
            return 'color: #00ff88'
        elif val == 'LOSS':
            return 'color: #ff4444'
        return ''
    cols = ['timestamp_open','symbol','direction','strategy',
            'entry_price','exit_price','profit_loss','outcome']
    display = df[[c for c in cols if c in df.columns]]
    st.dataframe(display, use_container_width=True)
