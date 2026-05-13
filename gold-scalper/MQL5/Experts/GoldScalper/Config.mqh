//+------------------------------------------------------------------+
//| Config.mqh — All tunable parameters for GoldScalper v1.0        |
//| Centralized settings — change here, affects entire bot           |
//+------------------------------------------------------------------+
#ifndef CONFIG_MQH
#define CONFIG_MQH

// ─── Bot Identity ───
input string   INP_BOT_NAME        = "GoldScalper v1.0"; // Bot name
input long     INP_MAGIC           = 20240513;            // Magic number (unique per bot)

// ─── Trading Mode ───
input bool     INP_VIRTUAL_SL      = false;  // Virtual SL (brokers can't see) — use FALSE without VPS
input bool     INP_VIRTUAL_TP      = true;   // Virtual TP (brokers can't see) — safe to use always
input bool     INP_PHYSICAL_SL     = true;   // Physical SL on broker (safety net if MT5 disconnects)
input int      INP_PHYSICAL_SL_PIPS= 100;    // Physical SL distance in pips (wider than stop-hunt zone)

// ─── Risk Management ───
input double   INP_RISK_PCT        = 1.0;    // Risk % per trade (of account balance)
input double   INP_MAX_SESSION_RISK= 3.0;    // Max risk % in a single session
input int      INP_MAX_TRADES_DAY  = 30;     // Max trades per day
input double   INP_MAX_DRAWDOWN_PCT= 5.0;    // Stop trading if daily drawdown exceeds this %
input int      INP_MIN_TRADE_INTERVAL = 5;    // Min seconds between trades

// ─── Gold Specific ───
input int      INP_SPREAD_MAX      = 35;     // Max spread (points) to allow entry
input int      INP_ATR_PERIOD      = 14;     // ATR period
input double   INP_SL_ATR_MULT     = 0.5;    // SL = 0.5 x ATR
input double   INP_TP_ATR_MULT     = 1.0;    // TP = 1.0 x ATR

// ─── Session (GMT hours) ───
input int      INP_LONDON_OPEN     = 8;      // London open hour GMT
input int      INP_NY_OPEN         = 13;     // NY open hour GMT
input int      INP_SESSION_BUFFER  = 30;     // Minutes before session to start looking
input bool     INP_ASIAN_SESSION   = false;  // Trade Asian session too? (usually choppy)

// ─── Regime Detection ───
input int      INP_HURST_PERIOD    = 100;    // Hurst exponent lookback
input double   INP_HURST_TREND     = 0.6;    // Hurst > this = trending
input double   INP_HURST_MR        = 0.4;    // Hurst < this = mean-reverting
input int      INP_ADX_PERIOD      = 14;     // ADX period
input double   INP_ADX_TREND       = 25;     // ADX > this = trending
input double   INP_ADX_QUIET       = 15;     // ADX < this = quiet

// ─── Order Flow ───
input double   INP_OB_IMBALANCE_STRONG = 0.55; // Strong imbalance threshold
input double   INP_OB_IMBALANCE_MODERATE = 0.40; // Moderate imbalance threshold
input int      INP_DOM_LEVELS      = 10;     // DOM levels to analyze
input double   INP_SPREAD_COMPRESS = 0.5;    // Spread ratio < this = compression (big move coming)

// ─── ML Confidence ───
input bool     INP_USE_ML          = false;  // Use ONNX ML model (requires .onnx file)
input string   INP_ONNX_MODEL      = "GoldScalper.onnx"; // ONNX model filename
input double   INP_MIN_CONFIDENCE  = 0.65;   // Min ML confidence to trade (0-1)
input int      INP_FEATURE_COUNT   = 30;     // Number of features for ML model

// ─── Time Stop ───
input int      INP_TIME_STOP_SEC   = 30;     // Close if no profit after N seconds
input int      INP_MIN_PROFIT_PIPS = 10;     // Min profit pips to avoid time stop

// ─── Partial Close ───
input bool     INP_PARTIAL_CLOSE   = true;   // Enable partial close at first TP
input double   INP_PARTIAL_PCT     = 0.7;    // Close 70% at first TP
input int      INP_PARTIAL_TP_PIPS = 30;     // First TP in pips for partial close

// ─── Regime Strategy Tweaks ───
input double   INP_VOLATILE_SL_MULT= 1.5;    // Wider SL in volatile regime
input double   INP_MR_TP_MULT      = 0.6;    // Tighter TP in mean-reverting regime

// ─── Logging ───
input bool     INP_VERBOSE_LOG     = true;   // Verbose logging (disable in production for speed)
input bool     INP_LOG_TICKS       = false;  // Log every tick (very verbose, for debugging only)

#endif
//+------------------------------------------------------------------+
