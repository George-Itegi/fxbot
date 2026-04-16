# dashboard/components/scanner.py — Page 4: Live Market Scanner

import streamlit as st
from datetime import datetime, timezone
from config.settings import WATCHLIST


def render():
    st.title("🔍 Live Market Scanner")
    st.caption("Runs full master scan on all symbols in your watchlist")

    col1, col2 = st.columns([3, 1])
    with col2:
        run_scan = st.button("▶ Run Scan Now", type="primary",
                             use_container_width=True)

    if not run_scan:
        st.info("Click 'Run Scan Now' to fetch live analysis for all symbols.")
        return

    # Connect MT5
    import MetaTrader5 as mt5
    from dotenv import load_dotenv
    import os
    load_dotenv()

    with st.spinner("Connecting to MT5..."):
        mt5.initialize()
        mt5.login(int(os.getenv('MT5_LOGIN', 0)),
                  password=os.getenv('MT5_PASSWORD', ''),
                  server=os.getenv('MT5_SERVER', ''))

    with st.spinner("Fetching external data..."):
        try:
            from data_layer.external_data.external_scanner import get_external_data
            ext_data = get_external_data(WATCHLIST)
            sess = ext_data.get('session', 'UNKNOWN')
            gate = ext_data.get('day_trade_ok', True)
            st.success(f"Session: **{sess}** | Gate: "
                       f"{'✅ OPEN' if gate else '🛑 BLOCKED'}")
        except Exception as e:
            ext_data = {}
            st.warning(f"External data unavailable: {e}")

    st.markdown("---")

    from data_layer.master_scanner import master_scan
    from strategies.strategy_engine import run_strategies

    for symbol in WATCHLIST:
        with st.spinner(f"Scanning {symbol}..."):
            try:
                master = master_scan(symbol)
                if master is None:
                    st.error(f"{symbol}: Scan failed")
                    continue

                signal = run_strategies(symbol, master, ext_data)
                _render_symbol_card(symbol, master, signal, ext_data)
            except Exception as e:
                st.error(f"{symbol}: Error — {e}")

    mt5.shutdown()

def _render_symbol_card(symbol, master, signal, ext_data):
    """Render one symbol's full analysis card."""
    bias   = master.get('combined_bias', 'NEUTRAL')
    score  = master.get('final_score', 0)
    state  = master.get('market_state', '?')
    action = master.get('recommendation', {}).get('action', 'SKIP')
    m      = master.get('market_report', {})
    s      = master.get('smc_report', {})
    d      = m.get('delta', {})
    rd     = m.get('rolling_delta', {})
    vwap   = m.get('vwap', {})
    prof   = m.get('profile', {})
    pd_z   = s.get('premium_discount', {})

    action_colors = {
        'TRADE':'🟢','WATCH':'🟡','WAIT':'🟠','SKIP':'🔴'}
    bias_icons = {'BULLISH':'📈','BEARISH':'📉','NEUTRAL':'↔️',
                  'CONFLICTED':'⚠️'}

    with st.expander(
        f"{action_colors.get(action,'⚪')} {symbol} | "
        f"{bias_icons.get(bias,'')} {bias} | "
        f"Score: {score}/100 | {state} | → {action}",
        expanded=(action in ('TRADE','WATCH'))
    ):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Order Flow**")
            st.write(f"Full Delta: `{d.get('delta',0):+d}`"
                     f" ({d.get('bias','?')} / {d.get('strength','?')})")
            st.write(f"Roll Delta: `{rd.get('delta',0):+d}`"
                     f" ({rd.get('bias','?')} / {rd.get('strength','?')})")
            st.markdown("**VWAP**")
            st.write(f"VWAP: `{vwap.get('vwap','?')}`"
                     f" ({vwap.get('pip_from_vwap',0):+.1f} pips)")
            st.write(f"Position: `{vwap.get('position','?')}`")

        with col2:
            st.markdown("**Volume Profile**")
            st.write(f"POC: `{prof.get('poc','?')}`"
                     f" ({prof.get('pip_to_poc','?')} pips away)")
            st.write(f"VA Zone: `{prof.get('price_position','?')}`")
            st.markdown("**Premium/Discount**")
            st.write(f"Zone: `{pd_z.get('zone','?')}`"
                     f" ({pd_z.get('position_pct','?')}%)")
            st.write(f"EQ: `{pd_z.get('pips_to_eq',0):+.1f}` pips")

        with col3:
            st.markdown("**SMC Structure**")
            ms = s.get('structure', {})
            st.write(f"Trend: `{ms.get('trend','?')}`")
            bos = ms.get('bos')
            if bos:
                st.write(f"BOS: `{bos['type']}` @ {bos['level']}")
            sw = s.get('last_sweep')
            if sw:
                st.write(f"Sweep: `{sw['type']}` ({sw['reversal_pips']}p)")
            nob = s.get('nearest_ob')
            if nob:
                st.write(f"OB: `{nob['type']}`"
                         f" {nob['bottom']}—{nob['top']}")

        if signal:
            st.success(
                f"✅ **SIGNAL:** {signal['direction']} via "
                f"{signal['strategy']} | Score: {signal['score']} | "
                f"SL: {signal['sl_pips']}p | "
                f"TP1: {signal['tp1_pips']}p | "
                f"TP2: {signal['tp2_pips']}p"
            )
            if signal.get('confluence'):
                st.caption("Confluence: " +
                           " | ".join(signal['confluence']))
