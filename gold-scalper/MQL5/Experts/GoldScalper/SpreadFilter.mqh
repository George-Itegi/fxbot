//+------------------------------------------------------------------+
//| SpreadFilter.mqh — Monitor and filter by spread                 |
//| Gold spreads can widen dramatically during news/off-hours       |
//| This filter prevents entries when spread is too wide            |
//+------------------------------------------------------------------+
#ifndef SPREAD_FILTER_MQH
#define SPREAD_FILTER_MQH

class CSpreadFilter {
private:
    int      m_max_spread;          // Max spread in points to allow entry
    double   m_current_spread;      // Current spread in points
    double   m_avg_spread;          // Average spread over history
    double   m_min_spread;          // Minimum spread seen (ECN-like)
    double   m_max_seen_spread;     // Maximum spread seen
    
    double   m_spread_history[];
    int      m_history_size;
    int      m_data_count;
    
public:
    CSpreadFilter(int max_spread = 35, int history = 200) 
        : m_max_spread(max_spread), m_history_size(history) {
        ArrayResize(m_spread_history, m_history_size);
        ArrayInitialize(m_spread_history, 0);
        m_current_spread = 0;
        m_avg_spread = 0;
        m_min_spread = DBL_MAX;
        m_max_seen_spread = 0;
        m_data_count = 0;
    }
    
    void Update() {
        double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
        double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
        
        if(bid <= 0 || ask <= 0 || point <= 0) return;
        
        m_current_spread = (ask - bid) / point;
        
        // Track min/max
        if(m_current_spread < m_min_spread) m_min_spread = m_current_spread;
        if(m_current_spread > m_max_seen_spread) m_max_seen_spread = m_current_spread;
        
        // Update history
        _ShiftArray(m_spread_history, m_history_size);
        m_spread_history[m_history_size - 1] = m_current_spread;
        if(m_data_count < m_history_size) m_data_count++;
        
        // Calculate average
        double sum = 0;
        int count = 0;
        for(int i = m_history_size - m_data_count; i < m_history_size; i++) {
            if(m_spread_history[i] > 0) {
                sum += m_spread_history[i];
                count++;
            }
        }
        if(count > 0) m_avg_spread = sum / count;
    }
    
    bool IsAcceptable() const {
        return m_current_spread <= m_max_spread && m_current_spread > 0;
    }
    
    bool IsWidening() const {
        return m_current_spread > m_avg_spread * 1.5;
    }
    
    bool IsCompressed() const {
        return m_avg_spread > 0 && m_current_spread < m_avg_spread * 0.5;
    }
    
    double GetCurrentSpread()    const { return m_current_spread; }
    double GetAverageSpread()    const { return m_avg_spread; }
    double GetMinSpread()        const { return m_min_spread; }
    double GetSpreadRatio()      const { return (m_avg_spread > 0) ? m_current_spread / m_avg_spread : 1.0; }

private:
    void _ShiftArray(double &arr[], int size) {
        for(int i = 0; i < size - 1; i++)
            arr[i] = arr[i + 1];
    }
};

#endif
//+------------------------------------------------------------------+
