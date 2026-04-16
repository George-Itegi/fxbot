# =============================================================
# data_layer/master_scanner.py
# PURPOSE: Combines ALL data layer modules into ONE master report.
# Includes: Order Flow, Volume Profile, VWAP, State Machine,
# SMC Structure, Order Blocks, Liquidity, FVGs, Sweeps,
# Premium/Discount, Breaker Blocks, HTF Alignment,
# + External Data: COT, Intermarket, Fear/Greed, News.
# Run standalone to test.
# =============================================================

import MetaTrader5 as mt5
from datetime import datetime, timezone
from dotenv import load_dotenv
import os

from data_layer.market_scanner import scan_symbol
from data_layer.smc.smc_scanner import scan_smc
from data_layer.feature_store import store
from data_layer.tick_aggregator import aggregator, init_aggregator
from data_layer.fractal_alignment import check_fractal_alignment

load_dotenv()

# External data is shared across all symbols — fetched once per cycle
_external_cache = {"data": None, "symbols": []}


def connect():
    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}")
        return False
    if not mt5.login(
        int(os.getenv("MT5_LOGIN")),
        password=os.getenv("MT5_PASSWORD"),
        server=os.getenv("MT5_SERVER")
    ):
        print(f"Login failed: {mt5.last_error()}")
        return False
    print("Connected to MT5\n")
    return True

def master_scan(symbol: str) -> dict | None:
    """Run complete institutional scan. Returns full master report."""
    market = scan_symbol(symbol)
    smc    = scan_smc(symbol, timeframe=mt5.TIMEFRAME_H1)
    if market is None or smc is None:
        return None

    # Update the Feature Store
    store.update_symbol_features(symbol, market, smc)
    
    # Check Fractal Alignment (Macro -> Setup -> Trigger)
    fractal = check_fractal_alignment(symbol, smc, market)
    
    market_score = market.get("trade_score", 0)
    smc_score    = smc.get("smc_score", 0)

    # HTF approved?
    htf          = smc.get("htf_alignment", {})
    htf_approved = htf.get("approved", True)
    htf_penalty  = 0 if htf_approved else 30

    # Premium/discount penalty
    pd           = smc.get("premium_discount", {})
    pd_bias      = pd.get("bias", "")
    market_bias  = market.get("combined_bias", "NEUTRAL")
    smc_bias     = smc.get("smc_bias", "NEUTRAL")
    pd_penalty   = 15 if (
        (market_bias == "BULLISH" and pd_bias == "SELL") or
        (market_bias == "BEARISH" and pd_bias == "BUY")
    ) else 0

    # Final score with all factors (no external data)
    base_score  = (market_score * 0.50) + (smc_score * 0.50)
    final_score = max(0, round(
        base_score - htf_penalty - pd_penalty
    ))

    # Combined bias
    if market_bias == smc_bias and market_bias != "NEUTRAL":
        combined_bias   = market_bias
        bias_confidence = "HIGH"
    elif market_bias == "NEUTRAL" or smc_bias == "NEUTRAL":
        combined_bias   = market_bias if market_bias != "NEUTRAL" else smc_bias
        bias_confidence = "MODERATE"
    else:
        combined_bias   = "CONFLICTED"
        bias_confidence = "LOW"

    last_sweep    = smc.get("last_sweep")
    sweep_aligned = last_sweep.get("bias") == combined_bias if last_sweep else False

    recommendation = _get_recommendation(
        final_score, bias_confidence,
        market.get("market_state"),
        htf_approved, pd_bias, combined_bias,
        sweep_aligned)

    return {
        "symbol":           symbol,
        "timestamp":        datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
        "market_report":    market,
        "smc_report":       smc,
        "fractal_alignment": fractal,
        "market_score":     market_score,
        "smc_score":        smc_score,
        "final_score":      final_score,
        "market_bias":      market_bias,
        "smc_bias":         smc_bias,
        "combined_bias":    combined_bias,
        "bias_confidence":  bias_confidence,
        "market_state":     market.get("market_state"),
        "htf_approved":     htf_approved,
        "pd_penalty":       pd_penalty,
        "sweep_aligned":    sweep_aligned,
        "recommendation":   recommendation,
    }

def _get_recommendation(score, confidence, state,
                        htf_approved, pd_bias,
                        combined_bias, sweep_aligned) -> dict:
    """Translate all factors into one clear bot action."""

    # HTF rejection
    if not htf_approved:
        return {"action": "SKIP",
                "reason": "HTF alignment rejected — trading against higher TF"}

    if confidence == "LOW":
        return {"action": "SKIP",
                "reason": "Conflicted bias between market and SMC scanners"}

    if state == "REVERSAL_RISK":
        return {"action": "SKIP",
                "reason": "Reversal risk detected — signals diverging"}

    if (combined_bias == "BULLISH" and pd_bias == "SELL") and score < 70:
        return {"action": "WAIT",
                "reason": "Bullish but price in premium zone — wait for pullback"}

    if (combined_bias == "BEARISH" and pd_bias == "BUY") and score < 70:
        return {"action": "WAIT",
                "reason": "Bearish but price in discount zone — wait for bounce"}

    if state == "TRENDING_EXTENDED":
        return {"action": "WAIT",
                "reason": "Trend valid but price extended — wait for pullback to OB"}

    if score >= 75 and confidence == "HIGH" and sweep_aligned:
        return {"action": "TRADE",
                "reason": f"Score {score}/100 — sweep confirmed + full alignment"}

    if score >= 75 and confidence == "HIGH":
        return {"action": "TRADE",
                "reason": f"Score {score}/100 — high confluence setup"}

    if score >= 55 and confidence in ("HIGH", "MODERATE"):
        return {"action": "WATCH",
                "reason": f"Score {score}/100 — developing setup"}

    return {"action": "SKIP",
            "reason": f"Score {score}/100 — insufficient confluence"}

def print_master_report(r: dict):
    """Print the complete upgraded master report."""
    if not r:
        return

    bias  = r["combined_bias"]
    conf  = r["bias_confidence"]
    state = r["market_state"]
    rec   = r["recommendation"]
    m     = r["market_report"]
    s     = r["smc_report"]
    ms    = s["structure"]
    nob   = s.get("nearest_ob")
    npool = s.get("nearest_pool")
    qfvg  = s.get("quality_fvgs", [])
    vwap  = m.get("vwap", {})
    prof  = m.get("profile", {})
    d     = m.get("delta", {})
    rd    = m.get("rolling_delta", {})
    pd    = s.get("premium_discount", {})
    htf   = s.get("htf_alignment", {})
    sw    = s.get("last_sweep")
    brk   = s.get("breaker_blocks", [])

    bias_icon = "📈" if bias=="BULLISH" else "📉" if bias=="BEARISH" else "⚠️"
    conf_icon = "🟢" if conf=="HIGH" else "🟡" if conf=="MODERATE" else "🔴"
    act_icon  = {"TRADE":"✅","WAIT":"⏳","WATCH":"👀","SKIP":"⛔"}
    state_icons = {
        "TRENDING_STRONG":"🚀","TRENDING_EXTENDED":"⚠️",
        "BALANCED":"↔️","REVERSAL_RISK":"🔄",
        "BREAKOUT_ACCEPTED":"✅","BREAKOUT_REJECTED":"❌",
    }

    BLOCK_CHAR = "█"
    LINE_CHAR = "─"
    DOUBLE_LINE_CHAR = "═"

    print(f"\n{BLOCK_CHAR*57}")
    print(f"  APEX TRADER — MASTER REPORT")
    print(f"  {r["symbol"]}  |  {r["timestamp"]}")
    print(f"  {BLOCK_CHAR*57}")   print(f"  BIAS        : {bias} {bias_icon}"
          f"  |  CONFIDENCE: {conf} {conf_icon}")
    print(f"  STATE       : {state} {state_icons.get(state,"")}")
    print(f"  SESSION     : {r.get("session","?")}"
          f"  (x{r.get("session_multiplier",1.0)})")
    print(f"  SCORES      : Market={r["market_score"]}/100"
          f"  SMC={r["smc_score"]}/100"
          f"  FINAL={r["final_score"]}/100")
    if r.get("pd_penalty",0) > 0:
        print(f"  ⚠️ PD PENALTY : -{r["pd_penalty"]} pts")
    print(f"  HTF         : {"✅ APPROVED" if r["htf_approved"] else "❌ REJECTED"}"
          f"  |  H4={htf.get("h4_bias")}")
    print(f"  ACTION      : {rec["action"]} {act_icon.get(rec["action"],"")}")
    print(f"  REASON      : {rec["reason"]}")
    print(f"{LINE_CHAR*57}")

    print(f"\n  ── ORDER FLOW ──────────────────────────────")
    print(f"  Delta (full)   : {d.get("delta",0):+d}"
          f"  ({d.get("bias","?")} / {d.get("strength","?")})")
    print(f"  Delta (rolling): {rd.get("delta",0):+d}"
          f"  ({rd.get("bias","?")} / {rd.get("strength","?")})")

    print(f"\n  ── MARKET CONTEXT ──────────────────────────")
    print(f"  VWAP     : {vwap.get("vwap")}"
          f"  ({vwap.get("pip_from_vwap",0):+.1f} pips)"
          f"  → {vwap.get("position")}")
    print(f"  POC      : {prof.get("poc")}"
          f"  ({prof.get("pip_to_poc")} pips away)")
    print(f"  VA Zone  : {prof.get("price_position")}")

    print(f"\n  ── PREMIUM / DISCOUNT ──────────────────────")
    print(f"  Zone     : {pd.get("zone")}"
          f"  ({pd.get("position_pct")}% of range)")
    print(f"  Bias     : {pd.get("bias")}"
          f"  | Pips to EQ: {pd.get("pips_to_eq",0):+.1f}")

    print(f"\n  ── SMC STRUCTURE ───────────────────────────")
    print(f"  Trend    : {ms.get("trend")}"
          f"  HH:{ms.get("hh_count")} HL:{ms.get("hl_count")}"
          f" | LH:{ms.get("lh_count")} LL:{ms.get("ll_count")}")
    bos = ms.get("bos")
    if bos:
        print(f"  BOS      : {bos["type"]} @ {bos["level"]}"
              f"  ({bos.get("break_pips","?")} pips)")
    choch = ms.get("choch")
    if choch:
        print(f"  ⚠️ CHOCH  : {choch["type"]} @ {choch["level"]}")

    print(f"\n  ── LAST SWEEP ──────────────────────────────")
    if sw:
        aligned = "✅ ALIGNED" if r.get("sweep_aligned") else "⚠️ AGAINST"
        icon2   = "📈" if sw["bias"]=="BULLISH" else "📉"
        print(f"  {icon2} {sw["type"]} @ {sw["swept_level"]}"
              f"  {aligned}")
        print(f"  Reversal : {sw["reversal_pips"]} pips"
              f"  | {sw["time"][:16]}")
    else:
        print("  No recent sweep.")

    print(f"\n  ── KEY LEVELS ──────────────────────────────")
    if nob:
        print(f"  Nearest OB    : {nob["type"]}"
              f" | {nob["bottom"]} — {nob["top"]}")
    if brk:
        print(f"  Breakers      : {len(brk)} active"
              f" | Nearest: {brk[0]["type"]}"
              f" {brk[0]["bottom"]}—{brk[0]["top"]}")
    if npool:
        status = "UNSWEPT" if not npool["swept"] else "SWEPT"
        print(f"  Nearest Pool  : {npool["type"]} @ {npool["level"]}"
              f" ({status}, {npool["touches"]} touches)")
    # FVG section — show best quality, or nearest, or status
    nfvg = s.get("nearest_fvg")
    if qfvg:
        f0     = qfvg[0]
        status = "✅ UNFILLED" if not f0["filled"] else "❌ FILLED"
        print(f"  Best FVG      : {status} {f0["type"]}"
              f" | {f0["bottom"]}—{f0["top"]}"
              f" | Q:{f0["quality_score"]}/100"
              f" | {f0["gap_pips"]}p")
    elif nfvg:
        status = "✅ UNFILLED" if not nfvg["filled"] else "❌ FILLED"
        print(f"  Nearest FVG   : {status} {nfvg["type"]}"
              f" | {nfvg["bottom"]}—{nfvg["top"]}"
              f" | {nfvg["gap_pips"]}p")
    else:
        print(f"  FVG           : No significant FVGs detected")

    # Report end
    print(f"{BLOCK_CHAR*57}\n")

# =============================================================
# STANDALONE TEST
# =============================================================
if __name__ == "__main__":
    if not connect():
        exit()

    WATCHLIST = ["EURUSD", "GBPUSD", "XAUUSD"]
    print(f"Running MASTER scan on {len(WATCHLIST)} symbols...\n")

    results = []
    for symbol in WATCHLIST:
        print(f"Scanning {symbol}...")
        report = master_scan(symbol)
        if report:
            results.append(report)
            print_master_report(report)
        else:
            print(f"  ⚠️ Could not scan {symbol}\n")

    # Summary Table
    print(f"\n{DOUBLE_LINE_CHAR*57}")
    print(f"  SUMMARY — Best Opportunities")
    print(f"{DOUBLE_LINE_CHAR*57}")
    print(f"  {"Symbol":<8} {"Bias":<12} {"Score":<7}"
          f" {"State":<22} {"Action"}")
    print(f"  {LINE_CHAR*55}")
    for r in sorted(results, key=lambda x: x["final_score"], reverse=True):
        htf_flag  = "" if r["htf_approved"] else " ❌HTF"
        print(f"  {r["symbol"]:<8} {r["combined_bias"]:<12}"
              f" {r["final_score"]:<7}"
              f" {r["market_state"]:<22}"
              f" {r["recommendation"]["action"]}{htf_flag}")
    print(f"{DOUBLE_LINE_CHAR*57}")

    mt5.shutdown()
