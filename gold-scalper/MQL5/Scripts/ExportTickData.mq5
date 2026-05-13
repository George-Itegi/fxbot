//+------------------------------------------------------------------+
//| ExportTickData.mq5 — Export gold tick data for ML training       |
//|                                                                    |
//| Run this script on XAUUSD M1 chart to export historical data     |
//| with features and labels for training the ONNX model.            |
//|                                                                    |
//| Output: CSV file saved to MQL5/Files/gold_features.csv           |
//| Then use ml/train_model.py to train the ONNX model               |
//+------------------------------------------------------------------+
#property copyright "GoldScalper"
#property version   "1.00"
#property description "Export tick data with features for ML training"
#property script_show_inputs

input int      INP_EXPORT_BARS  = 5000;    // Number of M1 bars to export
input int      INP_FUTURE_BARS  = 5;       // Bars ahead for label (5 = 5 minutes)
input double   INP_MIN_MOVE_PTS = 20;      // Min price move in points for label
input string   INP_FILENAME     = "gold_features.csv"; // Output filename

//+------------------------------------------------------------------+
void OnStart() {
    Print("Exporting ", INP_EXPORT_BARS, " bars of feature data...");
    
    int file_handle = FileOpen(INP_FILENAME, FILE_WRITE | FILE_CSV | FILE_ANSI, ",");
    if(file_handle == INVALID_HANDLE) {
        Print("ERROR: Cannot create file ", INP_FILENAME);
        return;
    }
    
    // ─── Header row ───
    string header = "timestamp,bar_index";
    // Regime features (6)
    header += ",hurst_approx,adx,adx_slope,atr_ratio,volume_zscore,price_position";
    // Price action features (10)
    header += ",mom_5,mom_20,rsi_14,price_vs_vwap,bb_position,range_5,range_20,wick_ratio,consec_dir,bar_body_ratio";
    // Session features (4)
    header += ",session_id,is_overlap,is_friday,is_monday";
    // Spread features (3)
    header += ",spread,spread_avg,spread_ratio";
    // Order flow proxy features (3)
    header += ",vol_ratio,tick_intensity,buy_pressure";
    // Cross features (2)
    header += ",regime_x_vol,momentum_align";
    // Cyclical time (1)
    header += ",time_sin";
    // Label (1)
    header += ",label";
    
    FileWrite(file_handle, header);
    
    // ─── Get indicator handles ───
    int adx_handle  = iADX(_Symbol, PERIOD_M1, 14);
    int atr_handle  = iATR(_Symbol, PERIOD_M1, 14);
    int rsi_handle  = iRSI(_Symbol, PERIOD_M1, 14, PRICE_CLOSE);
    int bb_handle   = iBands(_Symbol, PERIOD_M1, 20, 0, 2.0, PRICE_CLOSE);
    
    int exported = 0;
    int errors = 0;
    
    // Process bars (skip the most recent INP_FUTURE_BARS — no label yet)
    for(int i = INP_FUTURE_BARS; i < INP_EXPORT_BARS - 50; i++) {
        // ─── Get basic OHLCV ───
        double open  = iOpen(_Symbol, PERIOD_M1, i);
        double high  = iHigh(_Symbol, PERIOD_M1, i);
        double low   = iLow(_Symbol, PERIOD_M1, i);
        double close = iClose(_Symbol, PERIOD_M1, i);
        long volume  = iVolume(_Symbol, PERIOD_M1, i);
        
        if(close <= 0) { errors++; continue; }
        
        // ─── Label: Did price move up or down in next N bars? ───
        double future_close = iClose(_Symbol, PERIOD_M1, i - INP_FUTURE_BARS);
        double future_move  = (future_close - close) / SymbolInfoDouble(_Symbol, SYMBOL_POINT);
        int label = 0; // Neutral
        if(future_move > INP_MIN_MOVE_PTS) label = 1;   // Price went up significantly
        if(future_move < -INP_MIN_MOVE_PTS) label = -1;  // Price went down significantly
        
        // ─── Feature calculations ───
        
        // 1. Hurst approximation (simplified for speed)
        double hurst = _QuickHurst(i, 50);
        
        // 2. ADX
        double adx[], adx_prev[];
        ArraySetAsSeries(adx, true);
        ArraySetAsSeries(adx_prev, true);
        CopyBuffer(adx_handle, 0, i, 1, adx);
        CopyBuffer(adx_handle, 0, i + 1, 1, adx_prev);
        double adx_val = (ArraySize(adx) > 0) ? adx[0] : 25;
        double adx_slope = (ArraySize(adx_prev) > 0) ? adx_val - adx_prev[0] : 0;
        
        // 3. ATR ratio
        double atr[], atr_history[];
        ArraySetAsSeries(atr, true);
        ArraySetAsSeries(atr_history, true);
        CopyBuffer(atr_handle, 0, i, 1, atr);
        CopyBuffer(atr_handle, 0, i, 50, atr_history);
        double atr_val = (ArraySize(atr) > 0) ? atr[0] : 0;
        double atr_avg = 0;
        if(ArraySize(atr_history) >= 50) {
            for(int j = 0; j < 50; j++) atr_avg += atr_history[j];
            atr_avg /= 50;
        }
        double atr_ratio = (atr_avg > 0) ? atr_val / atr_avg : 1.0;
        
        // 4. Volume z-score
        long volumes[];
        ArraySetAsSeries(volumes, true);
        CopyTickVolume(_Symbol, PERIOD_M1, i, 50, volumes);
        double vol_mean = 0;
        int vol_count = ArraySize(volumes);
        for(int j = 0; j < vol_count; j++) vol_mean += (double)volumes[j];
        vol_mean /= MathMax(vol_count, 1);
        double vol_var = 0;
        for(int j = 0; j < vol_count; j++) vol_var += MathPow((double)volumes[j] - vol_mean, 2);
        double vol_std = MathSqrt(vol_var / MathMax(vol_count, 1));
        double vol_zscore = (vol_std > 0) ? ((double)volume - vol_mean) / vol_std : 0;
        
        // 5. Price position in range
        double high_50 = iHigh(_Symbol, PERIOD_M1, i);
        double low_50  = iLow(_Symbol, PERIOD_M1, i);
        for(int j = 1; j < 50; j++) {
            double h = iHigh(_Symbol, PERIOD_M1, i + j);
            double l = iLow(_Symbol, PERIOD_M1, i + j);
            if(h > high_50) high_50 = h;
            if(l < low_50)  low_50 = l;
        }
        double price_pos = (high_50 > low_50) ? (close - low_50) / (high_50 - low_50) : 0.5;
        
        // 6. Momentum
        double close_5 = iClose(_Symbol, PERIOD_M1, i + 5);
        double close_20 = iClose(_Symbol, PERIOD_M1, i + 20);
        double mom_5 = (close_5 > 0) ? (close - close_5) / close_5 * 100 : 0;
        double mom_20 = (close_20 > 0) ? (close - close_20) / close_20 * 100 : 0;
        
        // 7. RSI
        double rsi[];
        ArraySetAsSeries(rsi, true);
        CopyBuffer(rsi_handle, 0, i, 1, rsi);
        double rsi_val = (ArraySize(rsi) > 0) ? rsi[0] : 50;
        
        // 8. VWAP position
        double vwap = _ApproxVWAP(i, 20);
        double price_vs_vwap = (atr_val > 0) ? (close - vwap) / atr_val : 0;
        
        // 9. Bollinger position
        double bb_upper[], bb_lower[];
        ArraySetAsSeries(bb_upper, true);
        ArraySetAsSeries(bb_lower, true);
        CopyBuffer(bb_handle, 1, i, 1, bb_upper);
        CopyBuffer(bb_handle, 2, i, 1, bb_lower);
        double bb_range = (ArraySize(bb_upper) > 0 && ArraySize(bb_lower) > 0) ? 
                          bb_upper[0] - bb_lower[0] : 1;
        double bb_pos = (bb_range > 0 && ArraySize(bb_upper) > 0) ? 
                        (close - bb_lower[0]) / bb_range : 0.5;
        
        // 10. Range and wick
        double range_5 = _GetRange(i, 5);
        double range_20 = _GetRange(i, 20);
        double body = MathAbs(close - open);
        double total_bar = high - low;
        double wick_ratio = (total_bar > 0) ? 1.0 - body / total_bar : 0;
        
        // 11. Consecutive direction
        int consec = 0;
        for(int j = 0; j < 10; j++) {
            double o = iOpen(_Symbol, PERIOD_M1, i + j);
            double c = iClose(_Symbol, PERIOD_M1, i + j);
            if(c > o) { if(consec < 0) break; consec++; }
            else if(c < o) { if(consec > 0) break; consec--; }
            else break;
        }
        
        // 12. Bar body ratio
        double bar_body_ratio = (total_bar > 0) ? body / total_bar : 0;
        
        // 13. Session features
        datetime bar_time = iTime(_Symbol, PERIOD_M1, i);
        MqlDateTime dt;
        TimeToStruct(bar_time, dt);
        int session_id = _GetSessionID(dt);
        bool is_overlap = (session_id == 4);
        bool is_friday = (dt.day_of_week == 5);
        bool is_monday = (dt.day_of_week == 1);
        
        // 14. Spread (approximate from bar)
        double spread_pts = (open > 0) ? MathAbs(high - low) / SymbolInfoDouble(_Symbol, SYMBOL_POINT) * 0.1 : 30;
        
        // 15. Order flow proxies
        double vol_ratio = (vol_mean > 0) ? (double)volume / vol_mean : 1.0;
        double tick_intensity = (atr_val > 0) ? (high - low) / atr_val : 1.0;
        double buy_pressure = (total_bar > 0) ? (close - low) / total_bar : 0.5;
        
        // 16. Cross features
        double regime_x_vol = price_pos * MathAbs(vol_zscore / 3.0);
        double momentum_align = ((mom_5 > 0 && buy_pressure > 0.5) || 
                                  (mom_5 < 0 && buy_pressure < 0.5)) ? 1.0 : 0.0;
        
        // 17. Cyclical time
        double hour_angle = 2.0 * M_PI * (dt.hour + dt.min / 60.0) / 24.0;
        double time_sin = MathSin(hour_angle);
        
        // ─── Write row ───
        string row = "";
        row += TimeToString(bar_time) + "," + IntegerToString(i);
        row += "," + DoubleToString(hurst, 4);
        row += "," + DoubleToString(adx_val, 2);
        row += "," + DoubleToString(adx_slope, 2);
        row += "," + DoubleToString(atr_ratio, 4);
        row += "," + DoubleToString(vol_zscore, 4);
        row += "," + DoubleToString(price_pos, 4);
        row += "," + DoubleToString(mom_5, 6);
        row += "," + DoubleToString(mom_20, 6);
        row += "," + DoubleToString(rsi_val, 2);
        row += "," + DoubleToString(price_vs_vwap, 6);
        row += "," + DoubleToString(bb_pos, 4);
        row += "," + DoubleToString(range_5 / SymbolInfoDouble(_Symbol, SYMBOL_POINT), 1);
        row += "," + DoubleToString(range_20 / SymbolInfoDouble(_Symbol, SYMBOL_POINT), 1);
        row += "," + DoubleToString(wick_ratio, 4);
        row += "," + IntegerToString(consec);
        row += "," + DoubleToString(bar_body_ratio, 4);
        row += "," + IntegerToString(session_id);
        row += "," + (is_overlap ? "1" : "0");
        row += "," + (is_friday ? "1" : "0");
        row += "," + (is_monday ? "1" : "0");
        row += "," + DoubleToString(spread_pts, 1);
        row += "," + "30.0"; // avg spread placeholder
        row += "," + DoubleToString(spread_pts / 30.0, 4);
        row += "," + DoubleToString(vol_ratio, 4);
        row += "," + DoubleToString(tick_intensity, 4);
        row += "," + DoubleToString(buy_pressure, 4);
        row += "," + DoubleToString(regime_x_vol, 4);
        row += "," + DoubleToString(momentum_align, 1);
        row += "," + DoubleToString(time_sin, 4);
        row += "," + IntegerToString(label);
        
        FileWrite(file_handle, row);
        exported++;
        
        // Progress
        if(exported % 500 == 0)
            Print("Exported ", exported, " bars...");
    }
    
    // Cleanup
    IndicatorRelease(adx_handle);
    IndicatorRelease(atr_handle);
    IndicatorRelease(rsi_handle);
    IndicatorRelease(bb_handle);
    FileClose(file_handle);
    
    Print("═══════════════════════════════════");
    Print("  EXPORT COMPLETE");
    Print("  Bars exported: ", exported);
    Print("  Errors:        ", errors);
    Print("  File:          MQL5/Files/", INP_FILENAME);
    Print("  Next: Run ml/train_model.py");
    Print("═══════════════════════════════════");
}

//+------------------------------------------------------------------+
double _QuickHurst(int shift, int period) {
    // Very simplified Hurst approximation for export speed
    double returns[];
    ArrayResize(returns, period - 1);
    int valid = 0;
    
    for(int i = 0; i < period - 1; i++) {
        double p0 = iClose(_Symbol, PERIOD_M1, shift + i);
        double p1 = iClose(_Symbol, PERIOD_M1, shift + i + 1);
        if(p0 > 0 && p1 > 0) {
            returns[valid] = MathLog(p0 / p1);
            valid++;
        }
    }
    
    if(valid < 20) return 0.5;
    
    // Simple variance ratio as Hurst proxy
    double var_1 = 0, mean_1 = 0;
    for(int i = 0; i < valid; i++) mean_1 += returns[i];
    mean_1 /= valid;
    for(int i = 0; i < valid; i++) var_1 += MathPow(returns[i] - mean_1, 2);
    var_1 /= valid;
    
    // 2-period returns
    double var_2 = 0, mean_2 = 0;
    int half = valid / 2;
    for(int i = 0; i < half; i++) {
        double r2 = returns[i*2] + returns[i*2+1];
        mean_2 += r2;
    }
    mean_2 /= half;
    for(int i = 0; i < half; i++) {
        double r2 = returns[i*2] + returns[i*2+1];
        var_2 += MathPow(r2 - mean_2, 2);
    }
    var_2 /= half;
    
    if(var_1 <= 0) return 0.5;
    
    double vr = var_2 / (2 * var_1);
    // VR > 1 → trending (H > 0.5), VR < 1 → mean-reverting (H < 0.5)
    double hurst = 0.5 + (vr - 1.0) * 0.5;
    return MathMax(0.1, MathMin(0.9, hurst));
}

double _ApproxVWAP(int shift, int period) {
    double sum_pv = 0, sum_v = 0;
    for(int i = 0; i < period; i++) {
        double typical = (iHigh(_Symbol, PERIOD_M1, shift + i) + 
                         iLow(_Symbol, PERIOD_M1, shift + i) + 
                         iClose(_Symbol, PERIOD_M1, shift + i)) / 3.0;
        long vol = iVolume(_Symbol, PERIOD_M1, shift + i);
        sum_pv += typical * (double)vol;
        sum_v += (double)vol;
    }
    return (sum_v > 0) ? sum_pv / sum_v : iClose(_Symbol, PERIOD_M1, shift);
}

double _GetRange(int shift, int period) {
    double high = iHigh(_Symbol, PERIOD_M1, shift);
    double low = iLow(_Symbol, PERIOD_M1, shift);
    for(int i = 1; i < period; i++) {
        double h = iHigh(_Symbol, PERIOD_M1, shift + i);
        double l = iLow(_Symbol, PERIOD_M1, shift + i);
        if(h > high) high = h;
        if(l < low)  low = l;
    }
    return high - low;
}

int _GetSessionID(MqlDateTime &dt) {
    int min = dt.hour * 60 + dt.min;
    bool in_london = (min >= 450 && min <= 630); // 7:30 - 10:30 GMT
    bool in_ny     = (min >= 750 && min <= 930); // 12:30 - 15:30 GMT
    
    if(in_london && in_ny) return 4; // Overlap
    if(in_london) return 2;
    if(in_ny) return 3;
    if(min <= 360) return 1; // Asian
    return 0;
}
//+------------------------------------------------------------------+
