# dashboard/components/scanner.py v2.0
# Updated to reflect: multi-group consensus, institutional gates,
# hard state filters, new strategy groups, EMA21 entry logic

import streamlit as st
from datetime import datetime, timezone
from config.settings import WATCHLIST


def render():
    st.title("🔍 Live Market Scanner")
    st.caption("Multi-group consensus engine — signals require 2+ independent strategy groups")

    col1, col2, col3 = st.columns([3, 1, 1])
    with col2:
        run_scan = st.button("▶ Run Scan Now", type="primary", use_container_width=True)
    with col3:
        show_all = st.checkbox("Show all symbols", value=False)

    if not run_scan:
        st.info("Click **Run Scan Now** to fetch live analysis.\n\n"
                "A signal only fires when:\n"
                "- Master score ≥ 45/100\n"
                "- Order flow OR volume surge confirmed\n"
                "- 2+ strategy groups agree on direction\n"
                "- Bias confirmed by delta + SMC")
        _show_engine_legend()
        return

    import MetaTrader5 as mt5
    from dotenv import load_dotenv
    import os
    load_dotenv()

    with st.spinner("Connecting to MT5..."):
        mt5.initialize()
        mt5.login(int(os.getenv('MT5_LOGIN', 0)),
                  password=os.getenv('MT5_PASSWORD', ''),
                  server=os.getenv('MT5_SERVER', ''))

    from data_layer.master_scanner import master_scan
    from strategies.strategy_engine import run_strategies, STRATEGY_GROUPS

    results = []
    progress = st.progress(0)

    for i, symbol in enumerate(WATCHLIST):
        progress.progress((i + 1) / len(WATCHLIST))
        try:
            master = master_scan(symbol)
            if master is None:
                continue
            signal = run_strategies(symbol, master)
            results.append((symbol, master, signal))
        except Exception as e:
            st.error(f"{symbol}: Error — {e}")

    progress.empty()
    mt5.shutdown()

    # Sort: signals first, then by score descending
    results.sort(key=lambda x: (x[2] is not None, x[1].get('final_score', 0)), reverse=True)

    # Summary bar
    total       = len(results)
    with_signal = sum(1 for _, _, sig in results if sig is not None)
    st.markdown(f"### Scan complete — {with_signal} signal(s) from {total} symbols")

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Symbols scanned", total)
    col_b.metric("Signals fired", with_signal)
    col_c.metric("Avg score", f"{sum(m.get('final_score',0) for _, m, _ in results) // max(total,1)}/100")
    col_d.metric("High conf (≥60)", sum(1 for _, m, _ in results if m.get('final_score',0) >= 60))

    st.markdown("---")

    for symbol, master, signal in results:
        score = master.get('final_score', 0)
        if not show_all and score < 30 and signal is None:
            continue
        _render_symbol_card(symbol, master, signal)

def _show_engine_legend():
    """Explain the signal engine logic visually."""
    st.markdown("---")
    st.markdown("#### How signals are generated")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Signal gates (all must pass):**
1. 🔢 Master score ≥ 45/100
2. 🏦 Order flow STRONG/EXTREME **or** volume surge active
3. 📊 No choppy market (no surge = skip)
4. 🧭 Direction matches combined bias
5. 👥 2+ different strategy groups confirm direction
        """)
    with col2:
        st.markdown("""
**Strategy groups (need 2+ different):**
- 🔵 **Trend Following** — EMA Trend, Trend Continuation
- 🟢 **SMC Structure** — OB Reversal, Liquidity Sweep
- 🟠 **Order Flow** — Delta Div, Flow Exhaustion, Smart Money
- 🔴 **Momentum** — M1 Scalp, ORB
- 🟡 **Mean Reversion** — VWAP

*One group alone = BLOCKED even with 90+ score*
        """)


def _render_symbol_card(symbol, master, signal):
    """Render one symbol's full analysis card."""
    bias   = master.get('combined_bias', 'NEUTRAL')
    score  = master.get('final_score', 0)
    state  = master.get('market_state', '?')
    action = master.get('recommendation', {}).get('action', 'SKIP')
    conf   = master.get('bias_confidence', 'LOW')
    m      = master.get('market_report', {})
    s      = master.get('smc_report', {})
    d      = m.get('delta', {})
    rd     = m.get('rolling_delta', {})
    vwap   = m.get('vwap', {})
    prof   = m.get('profile', {})
    pd_z   = s.get('premium_discount', {})
    of_imb = master.get('order_flow_imbalance', {})
    surge  = master.get('volume_surge', {})
    mom    = master.get('momentum', {})
    sc     = master.get('scalping_signal', {})

    action_icon = {'TRADE':'🟢','WATCH':'🟡','WAIT':'🟠','SKIP':'🔴'}.get(action,'⚪')
    bias_icon   = {'BULLISH':'📈','BEARISH':'📉','NEUTRAL':'↔️','CONFLICTED':'⚠️'}.get(bias,'')
    conf_icon   = {'HIGH':'🟢','MODERATE':'🟡','LOW':'🔴'}.get(conf,'')

    # Institutional gate status
    has_of  = of_imb.get('strength','NONE') in ('STRONG','EXTREME')
    has_vol = surge.get('surge_detected', False)
    inst_ok = has_of or has_vol
    inst_icon = "🏦✅" if inst_ok else "🏦❌"

    label = (f"{action_icon} {symbol} | {bias_icon} {bias} | "
             f"Score:{score}/100 | {state} | {inst_icon}")

    with st.expander(label, expanded=(signal is not None or score >= 65)):
        # ── Signal box at top if present ──────────────────────
        if signal:
            direction = signal.get('direction','?')
            strategy  = signal.get('strategy','?')
            sig_score = signal.get('score', 0)
            sl_p      = signal.get('sl_pips', 0)
            tp_p      = signal.get('tp1_pips', 0)
            entry     = signal.get('entry_price', 0)
            group     = signal.get('group', '?')
            rr        = round(tp_p / sl_p, 2) if sl_p > 0 else 0

            dir_icon = "📈" if direction == "BUY" else "📉"
            st.success(
                f"✅ **{dir_icon} {direction}** via **{strategy}** [{group}]\n\n"
                f"Entry: `{entry:.5f}` | SL: `{sl_p}p` | TP: `{tp_p}p` | "
                f"R:R: `{rr:.1f}:1` | Score: `{sig_score}`"
            )
            conf_list = signal.get('confluence', [])
            if conf_list:
                st.caption("Confluence: " + "  ·  ".join(conf_list))

        elif score >= 45 and not inst_ok:
            st.warning("⚠️ Score OK but no institutional confirmation "
                       "(need order flow STRONG or volume surge)")

        # ── Scores & bias ─────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Master Score", f"{score}/100")
        c2.metric("Bias", f"{bias} {bias_icon}")
        c3.metric("Confidence", f"{conf} {conf_icon}")
        c4.metric("State", state)

        st.markdown("---")

        # ── Institutional gates ────────────────────────────────
        st.markdown("**Institutional Gates**")
        gi1, gi2, gi3 = st.columns(3)
        gi1.metric("Order Flow",
                   of_imb.get('strength', 'NONE'),
                   f"{of_imb.get('imbalance', 0):+.2f}",
                   delta_color="normal")
        gi2.metric("Volume Surge",
                   "✅ ACTIVE" if has_vol else "❌ NONE",
                   f"{surge.get('surge_ratio', 0):.1f}x" if has_vol else "—")
        gi3.metric("Momentum",
                   "SCALPABLE" if mom.get('is_scalpable') else "CHOPPY" if mom.get('is_choppy') else "NORMAL",
                   f"{mom.get('velocity_pips_min', 0):.1f} pips/min")

        st.markdown("---")

        # ── Data columns ──────────────────────────────────────
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Order Flow**")
            st.write(f"Full delta: `{d.get('delta',0):+d}` ({d.get('bias','?')})")
            st.write(f"Rolling: `{rd.get('delta',0):+d}` ({rd.get('strength','?')})")
            st.markdown("**VWAP**")
            st.write(f"`{vwap.get('vwap','?')}` ({vwap.get('pip_from_vwap',0):+.1f}p)")
            st.write(f"Position: `{vwap.get('position','?')}`")

        with col2:
            st.markdown("**Volume Profile**")
            st.write(f"POC: `{prof.get('poc','?')}` ({prof.get('pip_to_poc','?')}p)")
            st.write(f"Zone: `{prof.get('price_position','?')}`")
            st.markdown("**Premium/Discount**")
            st.write(f"`{pd_z.get('zone','?')}` ({pd_z.get('position_pct','?')}%)")
            st.write(f"To EQ: `{pd_z.get('pips_to_eq',0):+.1f}p`")

        with col3:
            st.markdown("**SMC Structure**")
            ms = s.get('structure', {})
            st.write(f"Trend: `{ms.get('trend','?')}`")
            bos = ms.get('bos')
            if bos:
                st.write(f"BOS: `{bos['type']}` @ {bos['level']}")
            choch = ms.get('choch')
            if choch:
                st.write(f"⚠️ CHoCH: `{choch['type']}`")
            sw = s.get('last_sweep')
            if sw:
                aligned = "✅" if master.get('sweep_aligned') else "⚠️"
                st.write(f"Sweep: `{sw['type']}` {aligned} ({sw.get('reversal_pips',0)}p)")
            nob = s.get('nearest_ob')
            if nob:
                st.write(f"OB: `{nob['type']}` {nob['bottom']}—{nob['top']}")
