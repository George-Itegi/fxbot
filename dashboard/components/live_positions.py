# dashboard/components/live_positions.py
# Page: Live Positions — real-time P&L + manual close

import streamlit as st
import pandas as pd
import time
from dashboard.components.db_helper import (
    get_live_positions_mt5, close_position_mt5)


def render():
    st.title("💹 Live Positions")
    st.caption("Real-time view of all open trades with current P&L")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    positions = get_live_positions_mt5()

    if not positions:
        st.info("No open positions right now.")
        _show_daily_summary()
        return

    # Summary metrics
    total_profit = sum(p['profit'] for p in positions)
    total_vol    = sum(p['volume'] for p in positions)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Open Trades", len(positions))
    c2.metric("Total Volume", f"{total_vol:.2f} lots")
    c3.metric("Floating P&L",
              f"${total_profit:+.2f}",
              delta_color="normal")
    c4.metric("Trades",
              f"{sum(1 for p in positions if p['profit']>0)} winning / "
              f"{sum(1 for p in positions if p['profit']<0)} losing")

    st.markdown("---")

    # Individual position cards
    for pos in positions:
        _render_position_card(pos)

    st.markdown("---")
    _show_daily_summary()

def _render_position_card(pos: dict):
    """Render a single position card with P&L and close button."""
    profit    = pos['profit']
    direction = pos['direction']
    symbol    = pos['symbol']
    pnl_color = "🟢" if profit > 0 else "🔴" if profit < 0 else "⚪"

    # Calculate pip P&L
    pip_size = 0.01 if pos['entry_price'] > 50 else 0.0001
    if direction == 'BUY':
        pips = (pos['current_price'] - pos['entry_price']) / pip_size
    else:
        pips = (pos['entry_price'] - pos['current_price']) / pip_size

    with st.expander(
        f"{pnl_color} {symbol} {direction} | "
        f"${profit:+.2f} ({pips:+.1f} pips) | "
        f"Ticket: {pos['ticket']}",
        expanded=profit < -10  # Auto-expand losing trades
    ):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Entry",   f"{pos['entry_price']:.5f}")
        col2.metric("Current", f"{pos['current_price']:.5f}",
                    f"{pips:+.1f}p")
        col3.metric("Volume",  f"{pos['volume']} lots")
        col4.metric("P&L",     f"${profit:+.2f}",
                    delta_color="normal")

        col5, col6, col7 = st.columns(3)
        col5.write(f"**SL:** {pos['sl'] or 'Not set'}")
        col6.write(f"**TP:** {pos['tp'] or 'Not set'}")
        col7.write(f"**Comment:** {pos['comment'] or '—'}")

        st.markdown("---")

        # Close button with confirmation
        col_close, col_info = st.columns([1, 3])
        with col_close:
            close_key = f"close_{pos['ticket']}"
            confirm_key = f"confirm_{pos['ticket']}"

            if st.session_state.get(confirm_key):
                st.warning(f"Close this {direction} {symbol}?")
                c1, c2 = st.columns(2)
                if c1.button("✅ YES CLOSE",
                             key=f"yes_{pos['ticket']}",
                             type="primary"):
                    result = close_position_mt5(pos['ticket'])
                    if result['success']:
                        st.success(f"Closed! Final P&L: ${result.get('profit',0):+.2f}")
                        st.session_state.pop(confirm_key, None)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"Close failed: {result.get('error')}")
                if c2.button("❌ Cancel",
                             key=f"cancel_{pos['ticket']}"):
                    st.session_state.pop(confirm_key, None)
                    st.rerun()
            else:
                if st.button(f"🔴 Close Position",
                             key=close_key,
                             use_container_width=True):
                    st.session_state[confirm_key] = True
                    st.rerun()

        with col_info:
            if profit > 5:
                st.info(f"💰 Trade is profitable — consider partial close or move SL to breakeven")
            elif profit < -15:
                st.warning(f"⚠️ Trade losing — review if market conditions changed")


def _show_daily_summary():
    """Show today's closed trade summary."""
    from dashboard.components.db_helper import get_trades
    import pandas as pd
    trades = get_trades(100)
    if trades.empty:
        return
    st.subheader("Today's Closed Trades")
    today = pd.Timestamp.now().date()
    if 'timestamp_open' in trades.columns:
        today_trades = trades[
            pd.to_datetime(trades['timestamp_open']).dt.date == today
        ]
        if today_trades.empty:
            st.info("No closed trades today.")
            return
        total = len(today_trades)
        wins  = len(today_trades[today_trades.get('outcome','').str.contains('WIN', na=False)])
        pnl   = today_trades['profit_loss'].sum() if 'profit_loss' in today_trades else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Trades Today", total)
        c2.metric("Win Rate",  f"{round(wins/total*100,1)}%" if total > 0 else "0%")
        c3.metric("Daily P&L", f"${pnl:+.2f}")
