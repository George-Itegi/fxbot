//+------------------------------------------------------------------+
//| GoldScalper.mq5 — Pure MQ5 Gold Scalper v1.0                    |
//| Multi-Regime + Order Flow + ONNX ML + Virtual Stops              |
//|                                                                    |
//| Designed for XAUUSD on M1 timeframe                               |
//| Works WITHOUT Python or external dependencies                     |
//|                                                                    |
//| SETUP:                                                             |
//| 1. Copy GoldScalper/ folder to MQL5/Experts/                     |
//| 2. Open MetaEditor, compile GoldScalper.mq5                      |
//| 3. Attach to XAUUSD M1 chart in MT5                              |
//| 4. Enable "Allow Algo Trading" in toolbar                        |
//| 5. Check "Allow DLL imports" in EA properties                    |
//|                                                                    |
//| WITHOUT VPS: Keep INP_VIRTUAL_SL=false, INP_PHYSICAL_SL=true     |
//| WITH VPS: Set INP_VIRTUAL_SL=true for maximum stealth            |
//+------------------------------------------------------------------+
#property copyright "GoldScalper v1.0"
#property version   "1.00"
#property strict
#property description "Multi-regime Gold Scalper with Order Flow + ML"

#include "Config.mqh"
#include "RegimeDetector.mqh"
#include "OrderFlowAnalyzer.mqh"
#include "SessionManager.mqh"
#include "SpreadFilter.mqh"
#include "PositionSizer.mqh"
#include "FeatureBuilder.mqh"
#include "MLConfidence.mqh"
#include "TradeManager.mqh"

// ─── Global Module Instances ───
CRegimeDetector    *g_regime;
COrderFlowAnalyzer *g_orderflow;
CSessionManager    *g_session;
CSpreadFilter      *g_spread;
CPositionSizer     *g_sizer;
CFeatureBuilder    *g_features;
CMLConfidence      *g_ml;
CTradeManager      *g_trades;

// ─── State Tracking ───
int      g_trades_today     = 0;
datetime g_last_trade_time  = 0;
datetime g_last_day         = 0;
double   g_daily_start_balance = 0;
bool     g_daily_limit_hit = false;
datetime g_last_status_log  = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit() {
    // ─── Symbol check ───
    if(_Symbol != "XAUUSD" && _Symbol != "XAUUSDm" && _Symbol != "GOLD") {
        Alert("GoldScalper is designed for XAUUSD! Current symbol: ", _Symbol);
        Alert("It may work on other symbols but results are untested.");
    }
    
    // ─── Timeframe check ───
    if(_Period != PERIOD_M1) {
        Alert("GoldScalper works best on M1 timeframe. Current: ", EnumToString(_Period));
    }
    
    // ─── Subscribe to Level 2 market depth ───
    if(!MarketBookAdd(_Symbol)) {
        Print("Market depth (Level 2) not available for ", _Symbol);
        Print("Order flow analysis will be limited. Check if your broker supports DOM.");
    }
    
    // ─── Initialize all modules ───
    g_regime    = new CRegimeDetector(INP_HURST_PERIOD, INP_ADX_TREND, INP_ADX_QUIET);
    g_orderflow = new COrderFlowAnalyzer(INP_DOM_LEVELS, 100);
    g_session   = new CSessionManager(INP_LONDON_OPEN, INP_NY_OPEN, INP_SESSION_BUFFER, INP_ASIAN_SESSION);
    g_spread    = new CSpreadFilter(INP_SPREAD_MAX, 200);
    g_sizer     = new CPositionSizer(INP_RISK_PCT, INP_MAX_SESSION_RISK);
    g_features  = new CFeatureBuilder(INP_FEATURE_COUNT);
    g_trades    = new CTradeManager(5, (int)INP_MAGIC, INP_PHYSICAL_SL, INP_PHYSICAL_SL_PIPS);
    
    // ─── Load ONNX model (optional) ───
    g_ml = new CMLConfidence(INP_ONNX_MODEL, INP_FEATURE_COUNT);
    if(INP_USE_ML) {
        if(!g_ml->Load()) {
            Print("WARNING: ML model failed to load. Running in RULE-BASED mode only.");
            Print("To use ML: train a model with ml/train_model.py and copy .onnx to MQL5/Files/");
        }
    } else {
        Print("ML model disabled in config. Running in RULE-BASED mode.");
    }
    
    g_daily_start_balance = AccountInfoDouble(ACCOUNT_BALANCE);
    
    // ─── Startup Banner ───
    Print("═══════════════════════════════════════════════════");
    Print("  GoldScalper v1.0 — Pure MQ5 Gold Scalper");
    Print("  Symbol:    ", _Symbol);
    Print("  Timeframe: M1 (tick-driven)");
    Print("  Mode:      ", INP_VIRTUAL_SL ? "VIRTUAL_SL" : "PHYSICAL_SL", " + ",
                            INP_VIRTUAL_TP ? "VIRTUAL_TP" : "PHYSICAL_TP");
    Print("  ML Model:  ", (INP_USE_ML && g_ml->IsLoaded()) ? "ACTIVE" : "DISABLED (rule-based)");
    Print("  Sessions:  London(", INP_LONDON_OPEN, ":00) + NY(", INP_NY_OPEN, ":00) GMT");
    Print("  Risk:      ", INP_RISK_PCT, "% per trade, max ", INP_MAX_SESSION_RISK, "%/session");
    Print("  Balance:   $", DoubleToString(g_daily_start_balance, 2));
    Print("═══════════════════════════════════════════════════");
    
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
    MarketBookRelease(_Symbol);
    
    // Print session summary
    Print("═══════════════════════════════════════════════════");
    Print("  GoldScalper SESSION SUMMARY");
    Print("  Trades:  ", g_trades->GetTotalTrades());
    Print("  Wins:    ", g_trades->GetWins());
    Print("  Losses:  ", g_trades->GetLosses());
    Print("  Win Rate:", DoubleToString(g_trades->GetWinRate() * 100, 1), "%");
    Print("  Profit:  $", DoubleToString(g_trades->GetTotalProfit(), 2));
    Print("  Loss:    $", DoubleToString(g_trades->GetTotalLoss(), 2));
    Print("  Net:     $", DoubleToString(g_trades->GetTotalProfit() - g_trades->GetTotalLoss(), 2));
    Print("═══════════════════════════════════════════════════");
    
    delete g_regime;
    delete g_orderflow;
    delete g_session;
    delete g_spread;
    delete g_sizer;
    delete g_features;
    delete g_ml;
    delete g_trades;
}

//+------------------------------------------------------------------+
//| Tick function — main trading loop                                  |
//+------------------------------------------------------------------+
void OnTick() {
    // ─── 1. Update all analyzers ───
    g_regime->Update();
    g_orderflow->Update();
    g_session->Update();
    g_spread->Update();
    
    // ─── 2. Manage existing positions (virtual stops, trailing, time stops) ───
    g_trades->OnTick();
    
    // ─── 3. Check if we can open new positions ───
    if(!_CanTrade()) return;
    
    // ─── 4. Get current regime ───
    ENUM_REGIME regime = g_regime->GetRegime();
    
    // NEVER trade in quiet regime — gold is choppy, no edge
    if(regime == REGIME_QUIET) {
        if(INP_VERBOSE_LOG && TimeCurrent() - g_last_status_log > 60) {
            Print("Regime: QUIET (H=", DoubleToString(g_regime->GetHurst(), 2),
                  " ADX=", DoubleToString(g_regime->GetADX(), 1), ") — waiting...");
            g_last_status_log = TimeCurrent();
        }
        return;
    }
    
    // ─── 5. Session filter — only trade during active sessions ───
    if(!g_session->IsActive()) {
        if(INP_VERBOSE_LOG && TimeCurrent() - g_last_status_log > 300) {
            Print("Session: OFF_HOURS — next session in ~", 
                  g_session->MinutesToNext(), " min");
            g_last_status_log = TimeCurrent();
        }
        return;
    }
    
    // ─── 6. Spread filter — don't trade when spread is too wide ───
    if(!g_spread->IsAcceptable()) {
        if(INP_VERBOSE_LOG && g_spread->IsWidening()) {
            Print("Spread too wide: ", DoubleToString(g_spread->GetCurrentSpread(), 1),
                  " points (max: ", INP_SPREAD_MAX, ")");
        }
        return;
    }
    
    // ─── 7. Get order flow signal ───
    int ob_signal = g_orderflow->GetSignal(
        INP_OB_IMBALANCE_STRONG, 
        INP_OB_IMBALANCE_MODERATE,
        INP_SPREAD_COMPRESS
    );
    
    if(ob_signal < 0) {
        // No order flow signal — skip
        return;
    }
    
    // ─── 8. ML confidence check (if enabled) ───
    double confidence = 0.7; // Default confidence when ML is disabled
    
    if(INP_USE_ML && g_ml->IsLoaded()) {
        double features[];
        g_features->Build(g_regime, g_orderflow, g_session, g_spread, features);
        confidence = g_ml->Predict(features);
        
        if(confidence < INP_MIN_CONFIDENCE) {
            if(INP_VERBOSE_LOG) {
                Print("ML confidence too low: ", DoubleToString(confidence * 100, 1),
                      "% (min: ", DoubleToString(INP_MIN_CONFIDENCE * 100, 0), "%)");
            }
            return;
        }
    } else {
        // Rule-based confidence: estimate from signal strength
        confidence = _EstimateConfidence(regime, ob_signal);
        if(confidence < 0.55) return; // Weak signal, skip
    }
    
    // ─── 9. Calculate position size ───
    double atr = g_regime->GetATR();
    if(atr <= 0) {
        // Fallback: use point value
        atr = SymbolInfoDouble(_Symbol, SYMBOL_POINT) * 100;
    }
    
    double sl_distance = atr * INP_SL_ATR_MULT;
    double tp_distance = atr * INP_TP_ATR_MULT;
    
    // ─── 10. Regime-specific adjustments ───
    if(regime == REGIME_VOLATILE) {
        sl_distance *= INP_VOLATILE_SL_MULT;   // Wider SL in volatile regime
    } else if(regime == REGIME_MEAN_REVERTING) {
        tp_distance *= INP_MR_TP_MULT;          // Tighter TP in MR regime
    }
    
    double lots = g_sizer->Calculate(sl_distance, confidence);
    if(lots <= 0) {
        Print("Position size = 0 — risk limit reached or insufficient balance");
        return;
    }
    
    // ─── 11. Calculate virtual SL/TP prices ───
    double entry = (ob_signal == ORDER_TYPE_BUY) ? 
                   SymbolInfoDouble(_Symbol, SYMBOL_ASK) : 
                   SymbolInfoDouble(_Symbol, SYMBOL_BID);
    
    double sl = (ob_signal == ORDER_TYPE_BUY) ? 
                entry - sl_distance : entry + sl_distance;
    double tp = (ob_signal == ORDER_TYPE_BUY) ? 
                entry + tp_distance : entry - tp_distance;
    
    // Partial close target
    double first_tp = 0;
    if(INP_PARTIAL_CLOSE) {
        double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
        first_tp = (ob_signal == ORDER_TYPE_BUY) ? 
                   entry + INP_PARTIAL_TP_PIPS * point :
                   entry - INP_PARTIAL_TP_PIPS * point;
    }
    
    // ─── 12. EXECUTE THE TRADE ───
    bool success = g_trades->OpenPosition(
        ob_signal, lots, sl, tp, first_tp,
        INP_PARTIAL_PCT, INP_TIME_STOP_SEC,
        INP_VIRTUAL_SL, INP_VIRTUAL_TP
    );
    
    if(success) {
        g_trades_today++;
        g_last_trade_time = TimeCurrent();
        g_sizer->RecordTradeRisk(INP_RISK_PCT);
        
        // ─── Trade Log ───
        Print("═══ TRADE #", g_trades_today, " ═══");
        Print("  Direction: ", (ob_signal == ORDER_TYPE_BUY ? "BUY" : "SELL"));
        Print("  Entry:     ", DoubleToString(entry, _Digits));
        Print("  vSL:       ", DoubleToString(sl, _Digits), 
              " (", DoubleToString(sl_distance / SymbolInfoDouble(_Symbol, SYMBOL_POINT), 0), " pts)");
        Print("  vTP:       ", DoubleToString(tp, _Digits), 
              " (", DoubleToString(tp_distance / SymbolInfoDouble(_Symbol, SYMBOL_POINT), 0), " pts)");
        Print("  Lots:      ", DoubleToString(lots, 2));
        Print("  Confidence:", DoubleToString(confidence * 100, 1), "%");
        Print("  Regime:    ", g_regime->RegimeToString(),
              " (H=", DoubleToString(g_regime->GetHurst(), 2),
              " ADX=", DoubleToString(g_regime->GetADX(), 1), ")");
        Print("  Session:   ", g_session->SessionToString());
        Print("  OB Imb:    ", DoubleToString(g_orderflow->GetImbalance(), 2));
        Print("  Spread:    ", DoubleToString(g_spread->GetCurrentSpread(), 1), " pts",
              " (ratio=", DoubleToString(g_spread->GetSpreadRatio(), 2), ")");
        Print("════════════════════════");
    }
}

//+------------------------------------------------------------------+
//| Book event — Level 2 market depth update                         |
//+------------------------------------------------------------------+
void OnBookEvent(string symbol) {
    if(symbol == _Symbol)
        g_orderflow->Update();
}

//+------------------------------------------------------------------+
//| Timer function — periodic status logging                          |
//+------------------------------------------------------------------+
void OnTimer() {
    if(INP_VERBOSE_LOG) {
        double balance = AccountInfoDouble(ACCOUNT_BALANCE);
        double equity  = AccountInfoDouble(ACCOUNT_EQUITY);
        
        Print("STATUS: Balance=$", DoubleToString(balance, 2),
              " Equity=$", DoubleToString(equity, 2),
              " Trades=", g_trades_today,
              " Regime=", g_regime->RegimeToString(),
              " Session=", g_session->SessionToString(),
              " OpenPos=", g_trades->GetOpenPositions());
    }
}

//+------------------------------------------------------------------+
//| Check if new trades are allowed                                   |
//+------------------------------------------------------------------+
bool _CanTrade() {
    datetime today = StringToTime(TimeToString(TimeCurrent(), TIME_DATE));
    
    // Day reset
    if(today != g_last_day) {
        g_trades_today = 0;
        g_daily_start_balance = AccountInfoDouble(ACCOUNT_BALANCE);
        g_daily_limit_hit = false;
        g_sizer->ResetSession();
        g_last_day = today;
    }
    
    // Daily trade limit
    if(g_trades_today >= INP_MAX_TRADES_DAY) return false;
    
    // Daily drawdown limit
    if(!g_daily_limit_hit && g_daily_start_balance > 0) {
        double balance = AccountInfoDouble(ACCOUNT_BALANCE);
        double drawdown_pct = (g_daily_start_balance - balance) / g_daily_start_balance * 100;
        if(drawdown_pct >= INP_MAX_DRAWDOWN_PCT) {
            g_daily_limit_hit = true;
            Print("DAILY DRAWDOWN LIMIT HIT: ", DoubleToString(drawdown_pct, 1), "%");
        }
    }
    if(g_daily_limit_hit) return false;
    
    // Min time between trades
    if(TimeCurrent() - g_last_trade_time < INP_MIN_TRADE_INTERVAL) return false;
    
    // Already in a position (single position mode)
    if(g_trades->GetOpenPositions() > 0) return false;
    
    // Not enough balance
    if(AccountInfoDouble(ACCOUNT_BALANCE) < 100) return false;
    
    return true;
}

//+------------------------------------------------------------------+
//| Estimate confidence from rule-based signals (when ML is off)     |
//+------------------------------------------------------------------+
double _EstimateConfidence(ENUM_REGIME regime, int signal) {
    double conf = 0.5; // Base confidence
    
    // Regime bonus
    if(regime == REGIME_TRENDING && signal == ORDER_TYPE_BUY && g_regime->GetHurst() > 0.6)
        conf += 0.15; // Strong uptrend + buy signal
    else if(regime == REGIME_TRENDING && signal == ORDER_TYPE_SELL && g_regime->GetHurst() > 0.6)
        conf += 0.15; // Strong downtrend + sell signal
    else if(regime == REGIME_MEAN_REVERTING)
        conf += 0.05; // Moderate edge in MR
    else if(regime == REGIME_VOLATILE)
        conf -= 0.05; // Risky in volatile
    
    // Order flow strength
    double imbalance = MathAbs(g_orderflow->GetImbalance());
    if(imbalance > 0.6)
        conf += 0.10; // Strong imbalance
    else if(imbalance > 0.4)
        conf += 0.05; // Moderate imbalance
    
    // Spread compression (leading signal — big move coming)
    if(g_spread->IsCompressed())
        conf += 0.10;
    
    // Session quality
    if(g_session->IsOverlap())
        conf += 0.05; // London-NY overlap is best
    else if(g_session->IsFriday())
        conf -= 0.05; // Friday is risky
    
    // ADX strength
    if(g_regime->GetADX() > 30)
        conf += 0.05; // Strong trend
    
    return MathMax(0.3, MathMin(0.95, conf));
}
//+------------------------------------------------------------------+
