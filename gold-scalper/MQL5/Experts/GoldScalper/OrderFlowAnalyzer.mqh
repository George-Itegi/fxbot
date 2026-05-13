//+------------------------------------------------------------------+
//| OrderFlowAnalyzer.mqh — Level 2 Depth of Market analysis        |
//| Reads DOM via MarketBookGet() to detect buy/sell imbalance      |
//| Also tracks spread compression as a leading signal              |
//+------------------------------------------------------------------+
#ifndef ORDERFLOW_ANALYZER_MQH
#define ORDERFLOW_ANALYZER_MQH

class COrderFlowAnalyzer {
private:
    double   m_imbalance;         // -1.0 to +1.0 (positive = buyers dominate)
    double   m_bid_volume;        // Total bid side volume
    double   m_ask_volume;        // Total ask side volume
    double   m_cum_delta;         // Cumulative delta (approximate)
    int      m_dom_levels;        // DOM levels to analyze
    
    double   m_spread_points;     // Current spread in points
    double   m_spread_avg;        // Average spread
    double   m_spread_ratio;      // Current / Average (compression signal)
    
    // Spread history for averaging
    double   m_spread_history[];
    int      m_spread_hist_size;
    int      m_spread_count;      // Valid data points
    
    // Previous mid-price for delta approximation
    double   m_prev_mid;
    
public:
    COrderFlowAnalyzer(int levels = 10, int spread_hist = 100) 
        : m_dom_levels(levels), m_spread_hist_size(spread_hist) {
        ArrayResize(m_spread_history, m_spread_hist_size);
        ArrayInitialize(m_spread_history, 0);
        m_imbalance = 0;
        m_bid_volume = 0;
        m_ask_volume = 0;
        m_cum_delta = 0;
        m_spread_points = 0;
        m_spread_avg = 0;
        m_spread_ratio = 1.0;
        m_spread_count = 0;
        m_prev_mid = 0;
    }
    
    void Update() {
        _AnalyzeDOM();
        _CalcSpread();
        _UpdateCumDelta();
    }
    
    double GetImbalance()    const { return m_imbalance; }
    double GetBidVolume()    const { return m_bid_volume; }
    double GetAskVolume()    const { return m_ask_volume; }
    double GetCumDelta()     const { return m_cum_delta; }
    double GetSpreadPoints() const { return m_spread_points; }
    double GetSpreadAvg()    const { return m_spread_avg; }
    double GetSpreadRatio()  const { return m_spread_ratio; }
    bool   IsSpreadCompressed(double threshold = 0.5) const { return m_spread_ratio < threshold; }
    
    // ─── Signal based on order flow ───
    // Returns: ORDER_TYPE_BUY, ORDER_TYPE_SELL, or -1 (no signal)
    int GetSignal(double strong_threshold = 0.55, double moderate_threshold = 0.40, 
                  double spread_compress = 0.5) const {
        // Strong buy: heavy bid imbalance + spread compression (big move up coming)
        if(m_imbalance > strong_threshold && m_spread_ratio < spread_compress)
            return ORDER_TYPE_BUY;
        
        // Strong sell: heavy ask imbalance + spread compression (big move down coming)
        if(m_imbalance < -strong_threshold && m_spread_ratio < spread_compress)
            return ORDER_TYPE_SELL;
        
        // Moderate buy: bid imbalance only
        if(m_imbalance > moderate_threshold)
            return ORDER_TYPE_BUY;
        
        // Moderate sell: ask imbalance only
        if(m_imbalance < -moderate_threshold)
            return ORDER_TYPE_SELL;
        
        return -1; // No signal
    }

private:
    void _AnalyzeDOM() {
        MqlBookInfo book[];
        if(!MarketBookGet(_Symbol, book)) return;
        
        int size = ArraySize(book);
        if(size == 0) return;
        
        m_bid_volume = 0;
        m_ask_volume = 0;
        
        int levels_used = 0;
        for(int i = 0; i < size && levels_used < m_dom_levels; i++) {
            if(book[i].type == BOOK_TYPE_BID) {
                m_bid_volume += (double)book[i].volume;
                levels_used++;
            }
        }
        
        levels_used = 0;
        for(int i = 0; i < size && levels_used < m_dom_levels; i++) {
            if(book[i].type == BOOK_TYPE_ASK) {
                m_ask_volume += (double)book[i].volume;
                levels_used++;
            }
        }
        
        double total = m_bid_volume + m_ask_volume;
        if(total > 0)
            m_imbalance = (m_bid_volume - m_ask_volume) / total;
    }
    
    void _CalcSpread() {
        double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
        double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        
        if(bid <= 0 || ask <= 0) return;
        
        double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
        if(point <= 0) return;
        
        m_spread_points = (ask - bid) / point;
        
        // Update spread history
        _ShiftArray(m_spread_history, m_spread_hist_size);
        m_spread_history[m_spread_hist_size - 1] = m_spread_points;
        if(m_spread_count < m_spread_hist_size) m_spread_count++;
        
        // Calculate average
        double sum = 0;
        int count = 0;
        for(int i = m_spread_hist_size - m_spread_count; i < m_spread_hist_size; i++) {
            if(m_spread_history[i] > 0) {
                sum += m_spread_history[i];
                count++;
            }
        }
        
        if(count > 0) {
            m_spread_avg = sum / count;
            m_spread_ratio = (m_spread_avg > 0) ? m_spread_points / m_spread_avg : 1.0;
        }
    }
    
    void _UpdateCumDelta() {
        // Approximate cumulative delta from tick trade data
        MqlTick ticks[];
        ArraySetAsSeries(ticks, true);
        int count = CopyTicks(_Symbol, ticks, COPY_TICKS_TRADE, 0, 10);
        
        if(count > 0) {
            for(int i = 0; i < count; i++) {
                if(ticks[i].flags & TICK_FLAG_BUY)
                    m_cum_delta += (double)ticks[i].volume;
                else if(ticks[i].flags & TICK_FLAG_SELL)
                    m_cum_delta -= (double)ticks[i].volume;
            }
        }
        
        // Decay cum_delta slowly to prevent drift
        m_cum_delta *= 0.999;
    }
    
    void _ShiftArray(double &arr[], int size) {
        for(int i = 0; i < size - 1; i++)
            arr[i] = arr[i + 1];
    }
};

#endif
//+------------------------------------------------------------------+
