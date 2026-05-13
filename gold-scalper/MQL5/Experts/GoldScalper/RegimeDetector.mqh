//+------------------------------------------------------------------+
//| RegimeDetector.mqh — Real-time market regime classification     |
//| Uses Hurst Exponent + ADX + ATR Ratio + Volume to classify      |
//| gold market into 4 regimes: Trending, Mean-Reverting,           |
//| Volatile, or Quiet.                                              |
//+------------------------------------------------------------------+
#ifndef REGIME_DETECTOR_MQH
#define REGIME_DETECTOR_MQH

// ─── Regime Enum ───
enum ENUM_REGIME {
    REGIME_TRENDING       = 0,  // Strong directional move — momentum breakout
    REGIME_MEAN_REVERTING = 1,  // Range-bound — buy dips, sell rips
    REGIME_VOLATILE       = 2,  // News/wild spikes — wider SL, quick exits
    REGIME_QUIET          = 3   // Dead/choppy — DO NOT TRADE
};

class CRegimeDetector {
private:
    double      m_hurst;            // Current Hurst exponent
    double      m_adx;              // Current ADX value
    double      m_adx_slope;       // ADX slope (rising/falling)
    double      m_atr;              // Current ATR value
    double      m_atr_avg;          // Average ATR over 50 periods
    double      m_atr_ratio;        // Current ATR / Average ATR
    double      m_volume_zscore;    // Volume z-score
    ENUM_REGIME m_regime;           // Current detected regime
    int         m_hurst_period;     // Lookback for Hurst calculation
    double      m_adx_threshold;    // ADX trending threshold
    double      m_adx_quiet;        // ADX quiet threshold
    
    // Price buffer for Hurst calculation
    double      m_close_prices[];
    int         m_buffer_size;
    int         m_data_count;       // How many data points we have
    
    // Previous ADX for slope calculation
    double      m_prev_adx;

public:
    CRegimeDetector(int hurst_period = 100, double adx_trend = 25, double adx_quiet = 15) 
        : m_hurst_period(hurst_period), m_adx_threshold(adx_trend), m_adx_quiet(adx_quiet) {
        m_buffer_size = hurst_period + 50; // Extra room
        ArrayResize(m_close_prices, m_buffer_size);
        ArrayInitialize(m_close_prices, 0);
        m_hurst = 0.5;
        m_adx = 0;
        m_adx_slope = 0;
        m_atr = 0;
        m_atr_avg = 0;
        m_atr_ratio = 1.0;
        m_volume_zscore = 0;
        m_regime = REGIME_QUIET;
        m_data_count = 0;
        m_prev_adx = 0;
    }
    
    void Update() {
        // 1. Shift price buffer and add current close
        _ShiftArray(m_close_prices, m_buffer_size);
        m_close_prices[m_buffer_size - 1] = iClose(_Symbol, PERIOD_M1, 0);
        if(m_data_count < m_buffer_size) m_data_count++;
        
        // 2. Get indicator values
        m_prev_adx = m_adx;
        m_adx = _GetADX(14);
        m_adx_slope = m_adx - m_prev_adx;
        m_atr = _GetATR(14);
        m_atr_avg = _GetATRAverage(50);
        m_atr_ratio = (m_atr_avg > 0) ? m_atr / m_atr_avg : 1.0;
        m_volume_zscore = _GetVolumeZScore(50);
        
        // 3. Calculate Hurst Exponent (only when we have enough data)
        if(m_data_count >= 50)
            m_hurst = _CalcHurst();
        
        // 4. Classify regime
        m_regime = _Classify();
    }
    
    ENUM_REGIME GetRegime()     const { return m_regime; }
    double      GetHurst()      const { return m_hurst; }
    double      GetADX()        const { return m_adx; }
    double      GetADXSlope()   const { return m_adx_slope; }
    double      GetATR()        const { return m_atr; }
    double      GetATRAvg()     const { return m_atr_avg; }
    double      GetATRRatio()   const { return m_atr_ratio; }
    double      GetVolumeZScore() const { return m_volume_zscore; }
    
    string RegimeToString() const {
        switch(m_regime) {
            case REGIME_TRENDING:       return "TRENDING";
            case REGIME_MEAN_REVERTING: return "MEAN_REVERT";
            case REGIME_VOLATILE:       return "VOLATILE";
            case REGIME_QUIET:          return "QUIET";
            default:                    return "UNKNOWN";
        }
    }

private:
    ENUM_REGIME _Classify() {
        // Priority 1: Volatile — ATR expanding rapidly + volume spike
        // This catches news events, NFP, FOMC, etc.
        if(m_atr_ratio > 2.0 && m_volume_zscore > 2.0)
            return REGIME_VOLATILE;
        
        // Also volatile if ATR extremely high regardless of volume
        if(m_atr_ratio > 3.0)
            return REGIME_VOLATILE;
        
        // Priority 2: Quiet — low ADX, tight range, low volume
        // DO NOT TRADE in this regime
        if(m_adx < m_adx_quiet && m_atr_ratio < 0.7 && m_volume_zscore < 0.5)
            return REGIME_QUIET;
        
        // Also quiet if ADX very low and Hurst ~0.5 (random walk)
        if(m_adx < 12 && MathAbs(m_hurst - 0.5) < 0.05)
            return REGIME_QUIET;
        
        // Priority 3: Trending — Hurst > 0.6 AND ADX rising AND ATR expanding
        if(m_hurst > 0.6 && m_adx > m_adx_threshold && m_adx_slope > 0)
            return REGIME_TRENDING;
        
        // Also trending if ADX is very strong (>40) regardless of Hurst
        if(m_adx > 40 && m_atr_ratio > 1.2)
            return REGIME_TRENDING;
        
        // Priority 4: Mean-reverting — Hurst < 0.4 AND ADX < 20
        if(m_hurst < 0.4 && m_adx < 20)
            return REGIME_MEAN_REVERTING;
        
        // Default: if ADX > trending threshold, assume trending
        // Otherwise mean-reverting (gold ranges more than it trends)
        return (m_adx > m_adx_threshold) ? REGIME_TRENDING : REGIME_MEAN_REVERTING;
    }
    
    double _CalcHurst() {
        // Rescaled Range (R/S) Analysis
        // Simplified for real-time use — 3 sub-period sizes
        int n = m_data_count;
        int valid_start = m_buffer_size - n;
        
        if(n < 50) return 0.5;
        
        // Calculate log returns
        double returns[];
        int ret_count = n - 1;
        ArrayResize(returns, ret_count);
        
        for(int i = 0; i < ret_count; i++) {
            double p0 = m_close_prices[valid_start + i];
            double p1 = m_close_prices[valid_start + i + 1];
            if(p0 > 0 && p1 > 0)
                returns[i] = MathLog(p1 / p0);
            else
                returns[i] = 0;
        }
        
        // R/S for 3 sub-period sizes
        int sizes[3];
        sizes[0] = 10; sizes[1] = 20; sizes[2] = 40;
        double log_n[], log_rs[];
        ArrayResize(log_n, 3);
        ArrayResize(log_rs, 3);
        int count = 0;
        
        for(int s = 0; s < 3; s++) {
            int size = sizes[s];
            if(size > ret_count) continue;
            
            int num_blocks = ret_count / size;
            if(num_blocks < 1) continue;
            
            double total_rs = 0;
            int valid_blocks = 0;
            
            for(int b = 0; b < num_blocks; b++) {
                // Mean of block
                double mean = 0;
                for(int i = 0; i < size; i++)
                    mean += returns[b * size + i];
                mean /= size;
                
                // Cumulative deviation from mean
                double max_dev = -DBL_MAX, min_dev = DBL_MAX;
                double cum_dev = 0;
                
                for(int i = 0; i < size; i++) {
                    cum_dev += returns[b * size + i] - mean;
                    if(cum_dev > max_dev) max_dev = cum_dev;
                    if(cum_dev < min_dev) min_dev = cum_dev;
                }
                
                // Standard deviation
                double variance = 0;
                for(int i = 0; i < size; i++)
                    variance += MathPow(returns[b * size + i] - mean, 2);
                double stdev = MathSqrt(variance / size);
                
                if(stdev > 0) {
                    double R = max_dev - min_dev;
                    total_rs += R / stdev;
                    valid_blocks++;
                }
            }
            
            if(valid_blocks > 0) {
                log_n[count] = MathLog(size);
                log_rs[count] = MathLog(total_rs / valid_blocks);
                count++;
            }
        }
        
        if(count < 2) return 0.5;
        
        // Linear regression slope = Hurst exponent
        double sum_x = 0, sum_y = 0;
        for(int i = 0; i < count; i++) { sum_x += log_n[i]; sum_y += log_rs[i]; }
        double mean_x = sum_x / count;
        double mean_y = sum_y / count;
        
        double num = 0, den = 0;
        for(int i = 0; i < count; i++) {
            num += (log_n[i] - mean_x) * (log_rs[i] - mean_y);
            den += (log_n[i] - mean_x) * (log_n[i] - mean_x);
        }
        
        if(den == 0) return 0.5;
        
        double hurst = num / den;
        return MathMax(0.1, MathMin(0.9, hurst));
    }
    
    double _GetADX(int period) {
        int handle = iADX(_Symbol, PERIOD_M1, period);
        if(handle == INVALID_HANDLE) return 0;
        
        double adx[];
        ArraySetAsSeries(adx, true);
        int copied = CopyBuffer(handle, 0, 0, 1, adx);
        IndicatorRelease(handle);
        
        return (copied > 0) ? adx[0] : 0;
    }
    
    double _GetATR(int period) {
        int handle = iATR(_Symbol, PERIOD_M1, period);
        if(handle == INVALID_HANDLE) return 0;
        
        double atr[];
        ArraySetAsSeries(atr, true);
        int copied = CopyBuffer(handle, 0, 0, 1, atr);
        IndicatorRelease(handle);
        
        return (copied > 0) ? atr[0] : 0;
    }
    
    double _GetATRAverage(int period) {
        int handle = iATR(_Symbol, PERIOD_M1, period);
        if(handle == INVALID_HANDLE) return 0;
        
        double atr[];
        ArraySetAsSeries(atr, true);
        int copied = CopyBuffer(handle, 0, 0, period, atr);
        IndicatorRelease(handle);
        
        if(copied < period) return 0;
        
        double sum = 0;
        for(int i = 0; i < period; i++) sum += atr[i];
        return sum / period;
    }
    
    double _GetVolumeZScore(int period) {
        long volumes[];
        ArraySetAsSeries(volumes, true);
        int copied = CopyTickVolume(_Symbol, PERIOD_M1, 0, period, volumes);
        if(copied < period) return 0;
        
        double mean = 0;
        for(int i = 0; i < period; i++) mean += (double)volumes[i];
        mean /= period;
        
        double variance = 0;
        for(int i = 0; i < period; i++)
            variance += MathPow((double)volumes[i] - mean, 2);
        double stdev = MathSqrt(variance / period);
        
        if(stdev > 0)
            return ((double)volumes[0] - mean) / stdev;
        return 0;
    }
    
    void _ShiftArray(double &arr[], int size) {
        for(int i = 0; i < size - 1; i++)
            arr[i] = arr[i + 1];
    }
};

#endif
//+------------------------------------------------------------------+
