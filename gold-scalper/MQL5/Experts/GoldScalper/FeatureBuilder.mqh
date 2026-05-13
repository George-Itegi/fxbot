//+------------------------------------------------------------------+
//| FeatureBuilder.mqh — Build ML feature vector from market data   |
//| Constructs a fixed-size feature array for ONNX model inference  |
//+------------------------------------------------------------------+
#ifndef FEATURE_BUILDER_MQH
#define FEATURE_BUILDER_MQH

#include "RegimeDetector.mqh"
#include "OrderFlowAnalyzer.mqh"
#include "SessionManager.mqh"
#include "SpreadFilter.mqh"

class CFeatureBuilder {
private:
    int      m_feature_count;     // Expected number of features
    double   m_features[];        // Output feature array
    
public:
    CFeatureBuilder(int feature_count = 30) : m_feature_count(feature_count) {
        ArrayResize(m_features, m_feature_count);
        ArrayInitialize(m_features, 0);
    }
    
    // Build feature vector from all analyzers
    // Returns reference to internal array (don't store pointer)
    double[]& Build(CRegimeDetector &regime, COrderFlowAnalyzer &orderflow,
                    CSessionManager &session, CSpreadFilter &spread) {
        // ─── Feature Group 1: Regime (6 features) ───
        m_features[0]  = regime.GetHurst();                           // 0-1
        m_features[1]  = regime.GetADX() / 100.0;                    // Normalize to 0-1
        m_features[2]  = regime.GetADXSlope() / 10.0;                // Normalize
        m_features[3]  = regime.GetATRRatio();                        // 0-3+
        m_features[4]  = regime.GetVolumeZScore() / 5.0;             // Normalize
        m_features[5]  = (double)regime.GetRegime() / 3.0;           // 0-1 (4 regimes)
        
        // ─── Feature Group 2: Order Flow (5 features) ───
        m_features[6]  = orderflow.GetImbalance();                    // -1 to +1
        m_features[7]  = MathAbs(orderflow.GetImbalance());           // 0-1
        m_features[8]  = orderflow.GetCumDelta() / 1000.0;           // Normalize
        m_features[9]  = spread.GetSpreadRatio();                     // 0-2+
        m_features[10] = spread.IsCompressed() ? 1.0 : 0.0;         // Binary
        
        // ─── Feature Group 3: Price Action (10 features) ───
        m_features[11] = _GetMomentum(5);                            // 5-bar momentum
        m_features[12] = _GetMomentum(20);                            // 20-bar momentum
        m_features[13] = _GetRSI(14);                                 // RSI 0-100
        m_features[14] = _GetPriceVsVWAP();                           // -1 to +1
        m_features[15] = _GetPriceVsBollinger();                      // 0-1 (position in bands)
        m_features[16] = _GetBarRange(5) / regime.GetATR();           // Normalized range
        m_features[17] = _GetBarRange(20) / regime.GetATR();          // Normalized range
        m_features[18] = _GetWickRatio();                             // 0-1 (wick vs body)
        m_features[19] = _GetConsecutiveDirection();                   // -5 to +5
        
        // ─── Feature Group 4: Session (4 features) ───
        m_features[20] = (double)session.GetSession() / 4.0;         // 0-1
        m_features[21] = session.IsOverlap() ? 1.0 : 0.0;            // Binary
        m_features[22] = session.IsFriday() ? 1.0 : 0.0;             // Binary
        m_features[23] = session.IsMonday() ? 1.0 : 0.0;             // Binary
        
        // ─── Feature Group 5: Spread & Execution (3 features) ───
        m_features[24] = spread.GetCurrentSpread() / 100.0;           // Normalize
        m_features[25] = spread.GetAverageSpread() / 100.0;           // Normalize
        m_features[26] = spread.IsWidening() ? 1.0 : 0.0;            // Binary
        
        // ─── Feature Group 6: Cross-features (2 features) ───
        // Regime x Order Flow interaction
        m_features[27] = m_features[5] * MathAbs(m_features[6]);     // Regime × |OB|
        // Momentum x Imbalance alignment
        m_features[28] = (m_features[11] > 0 && m_features[6] > 0) ? 1.0 :
                         (m_features[11] < 0 && m_features[6] < 0) ? 1.0 : 0.0;
        
        // ─── Feature 29: Time of day (cyclical encoding) ───
        MqlDateTime dt;
        TimeToStruct(TimeGMT(), dt);
        double hour_angle = 2.0 * M_PI * (dt.hour + dt.min / 60.0) / 24.0;
        m_features[29] = MathSin(hour_angle);                         // Cyclical time
        
        // Clip all features to [-10, 10] to prevent extreme values
        for(int i = 0; i < m_feature_count; i++) {
            m_features[i] = MathMax(-10.0, MathMin(10.0, m_features[i]));
            // Replace NaN/Inf with 0
            if(m_features[i] != m_features[i] || MathAbs(m_features[i]) > 1e10)
                m_features[i] = 0;
        }
        
        return m_features;
    }
    
    int GetFeatureCount() const { return m_feature_count; }

private:
    double _GetMomentum(int period) {
        double c0 = iClose(_Symbol, PERIOD_M1, 0);
        double cn = iClose(_Symbol, PERIOD_M1, period);
        if(cn <= 0) return 0;
        return (c0 - cn) / cn * 100; // Percentage momentum
    }
    
    double _GetRSI(int period) {
        int handle = iRSI(_Symbol, PERIOD_M1, period, PRICE_CLOSE);
        if(handle == INVALID_HANDLE) return 50;
        
        double rsi[];
        ArraySetAsSeries(rsi, true);
        int copied = CopyBuffer(handle, 0, 0, 1, rsi);
        IndicatorRelease(handle);
        
        return (copied > 0) ? rsi[0] : 50;
    }
    
    double _GetPriceVsVWAP() {
        // Simplified VWAP approximation using typical price
        int bars = 20;
        double sum_price_vol = 0;
        double sum_vol = 0;
        
        long volumes[];
        ArraySetAsSeries(volumes, true);
        CopyTickVolume(_Symbol, PERIOD_M1, 0, bars, volumes);
        
        for(int i = 0; i < bars; i++) {
            double typical = (iHigh(_Symbol, PERIOD_M1, i) + 
                             iLow(_Symbol, PERIOD_M1, i) + 
                             iClose(_Symbol, PERIOD_M1, i)) / 3.0;
            sum_price_vol += typical * (double)volumes[i];
            sum_vol += (double)volumes[i];
        }
        
        if(sum_vol <= 0) return 0;
        
        double vwap = sum_price_vol / sum_vol;
        double price = iClose(_Symbol, PERIOD_M1, 0);
        double atr = iATR(_Symbol, PERIOD_M1, 14) > 0 ? 
                     iATRValue(14) : 1;
        
        return (atr > 0) ? (price - vwap) / atr : 0;
    }
    
    double _GetPriceVsBollinger() {
        int handle = iBands(_Symbol, PERIOD_M1, 20, 0, 2.0, PRICE_CLOSE);
        if(handle == INVALID_HANDLE) return 0.5;
        
        double upper[], lower[], close_val = iClose(_Symbol, PERIOD_M1, 0);
        ArraySetAsSeries(upper, true);
        ArraySetAsSeries(lower, true);
        
        int c1 = CopyBuffer(handle, 1, 0, 1, upper); // Upper band
        int c2 = CopyBuffer(handle, 2, 0, 1, lower); // Lower band
        IndicatorRelease(handle);
        
        if(c1 > 0 && c2 > 0) {
            double range = upper[0] - lower[0];
            if(range > 0)
                return (close_val - lower[0]) / range; // 0 = at lower, 1 = at upper
        }
        return 0.5;
    }
    
    double _GetBarRange(int period) {
        double high = iHigh(_Symbol, PERIOD_M1, 0);
        double low  = iLow(_Symbol, PERIOD_M1, 0);
        
        for(int i = 1; i < period; i++) {
            double h = iHigh(_Symbol, PERIOD_M1, i);
            double l = iLow(_Symbol, PERIOD_M1, i);
            if(h > high) high = h;
            if(l < low)  low = l;
        }
        
        return high - low;
    }
    
    double _GetWickRatio() {
        double open  = iOpen(_Symbol, PERIOD_M1, 0);
        double close = iClose(_Symbol, PERIOD_M1, 0);
        double high  = iHigh(_Symbol, PERIOD_M1, 0);
        double low   = iLow(_Symbol, PERIOD_M1, 0);
        
        double body = MathAbs(close - open);
        double total = high - low;
        
        if(total <= 0) return 0;
        return 1.0 - (body / total); // High wick ratio = indecision
    }
    
    double _GetConsecutiveDirection() {
        int count = 0;
        for(int i = 0; i < 10; i++) {
            double o = iOpen(_Symbol, PERIOD_M1, i);
            double c = iClose(_Symbol, PERIOD_M1, i);
            if(c > o) { if(count < 0) break; count++; }
            else if(c < o) { if(count > 0) break; count--; }
            else break;
        }
        return count;
    }
    
    double iATRValue(int period) {
        int handle = iATR(_Symbol, PERIOD_M1, period);
        if(handle == INVALID_HANDLE) return 0;
        double atr[];
        ArraySetAsSeries(atr, true);
        int copied = CopyBuffer(handle, 0, 0, 1, atr);
        IndicatorRelease(handle);
        return (copied > 0) ? atr[0] : 0;
    }
};

#endif
//+------------------------------------------------------------------+
