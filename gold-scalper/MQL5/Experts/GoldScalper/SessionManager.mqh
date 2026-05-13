//+------------------------------------------------------------------+
//| SessionManager.mqh — Trading session timing for Gold            |
//| Tracks London, NY, and Asian sessions in GMT                    |
//| Only trades during active sessions with configurable buffer     |
//+------------------------------------------------------------------+
#ifndef SESSION_MANAGER_MQH
#define SESSION_MANAGER_MQH

enum ENUM_SESSION {
    SESSION_NONE    = 0,   // Off-hours — no trading
    SESSION_ASIAN   = 1,   // Tokyo/Sydney (usually choppy for gold)
    SESSION_LONDON  = 2,   // London open — good moves
    SESSION_NY      = 3,   // New York open — best moves
    SESSION_OVERLAP = 4    // London-NY overlap — BEST time for gold
};

class CSessionManager {
private:
    int           m_london_open;    // GMT hour
    int           m_ny_open;        // GMT hour
    int           m_buffer_min;     // Minutes before session
    bool          m_trade_asian;    // Trade Asian session?
    ENUM_SESSION  m_current_session;
    bool          m_is_active;
    int           m_minutes_to_next; // Minutes until next session
    bool          m_is_friday;       // Friday — reduced trading
    bool          m_is_monday;       // Monday — cautious start
    
public:
    CSessionManager(int london = 8, int ny = 13, int buffer = 30, bool asian = false) 
        : m_london_open(london), m_ny_open(ny), m_buffer_min(buffer), m_trade_asian(asian) {
        m_current_session = SESSION_NONE;
        m_is_active = false;
        m_minutes_to_next = 0;
        m_is_friday = false;
        m_is_monday = false;
    }
    
    void Update() {
        MqlDateTime dt;
        TimeToStruct(TimeGMT(), dt);
        
        int current_min = dt.hour * 60 + dt.min;
        m_is_friday = (dt.day_of_week == 5);
        m_is_monday = (dt.day_of_week == 1);
        
        // Session boundaries (in minutes from midnight GMT)
        int london_start = m_london_open * 60 - m_buffer_min;
        int london_end   = m_london_open * 60 + 120 + m_buffer_min; // ~2hr after open
        int ny_start     = m_ny_open * 60 - m_buffer_min;
        int ny_end       = m_ny_open * 60 + 120 + m_buffer_min;
        
        // Weekend check — no trading on Saturday/Sunday
        if(dt.day_of_week == 0 || dt.day_of_week == 6) {
            m_current_session = SESSION_NONE;
            m_is_active = false;
            m_minutes_to_next = 0;
            return;
        }
        
        // Check which session we're in
        bool in_london = (current_min >= london_start && current_min <= london_end);
        bool in_ny     = (current_min >= ny_start && current_min <= ny_end);
        
        // Asian session (0:00 - 6:00 GMT) — usually choppy for gold
        bool in_asian  = (current_min >= 0 && current_min <= 360);
        
        if(in_london && in_ny) {
            m_current_session = SESSION_OVERLAP;  // Best time!
            m_is_active = true;
        } else if(in_london) {
            m_current_session = SESSION_LONDON;
            m_is_active = true;
        } else if(in_ny) {
            m_current_session = SESSION_NY;
            m_is_active = true;
        } else if(in_asian && m_trade_asian) {
            m_current_session = SESSION_ASIAN;
            m_is_active = true;
        } else {
            m_current_session = SESSION_NONE;
            m_is_active = false;
        }
        
        // Calculate minutes to next session
        if(!m_is_active) {
            if(current_min < london_start)
                m_minutes_to_next = london_start - current_min;
            else if(current_min < ny_start)
                m_minutes_to_next = ny_start - current_min;
            else
                m_minutes_to_next = (24 * 60 - current_min) + london_start; // Next day
        } else {
            m_minutes_to_next = 0;
        }
        
        // Friday late session — reduce risk (market closes, spreads widen)
        if(m_is_friday && dt.hour >= 20) {
            m_is_active = false;
            m_current_session = SESSION_NONE;
        }
    }
    
    bool         IsActive()         const { return m_is_active; }
    ENUM_SESSION GetSession()       const { return m_current_session; }
    int          MinutesToNext()    const { return m_minutes_to_next; }
    bool         IsFriday()         const { return m_is_friday; }
    bool         IsMonday()         const { return m_is_monday; }
    bool         IsOverlap()        const { return m_current_session == SESSION_OVERLAP; }
    
    string SessionToString() const {
        switch(m_current_session) {
            case SESSION_ASIAN:   return "ASIAN";
            case SESSION_LONDON:  return "LONDON";
            case SESSION_NY:      return "NEW_YORK";
            case SESSION_OVERLAP: return "LONDON+NY";
            default:              return "OFF_HOURS";
        }
    }
};

#endif
//+------------------------------------------------------------------+
