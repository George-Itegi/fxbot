# =============================================================
# data_layer/external_data/news_sentiment.py
# PURPOSE: Fetch forex-related news and score sentiment.
# Uses Finnhub free API — 60 calls/minute, no cost.
# Detects bullish/bearish news before price reacts.
# Run standalone to test.
# Get free API key at: https://finnhub.io/register
# =============================================================

import requests
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import os

load_dotenv()

FINNHUB_KEY = os.getenv("FINNHUB_KEY", "")
BASE_URL    = "https://finnhub.io/api/v1"

# Forex symbols in Finnhub format
FINNHUB_SYMBOLS = {
    "EURUSD": "OANDA:EUR_USD",
    "GBPUSD": "OANDA:GBP_USD",
    "USDJPY": "OANDA:USD_JPY",
    "AUDUSD": "OANDA:AUD_USD",
    "USDCAD": "OANDA:USD_CAD",
    "NZDUSD": "OANDA:NZD_USD",
    "USDCHF": "OANDA:USD_CHF",
    "GBPJPY": "OANDA:GBP_JPY",
    "EURJPY": "OANDA:EUR_JPY",
    "AUDJPY": "OANDA:AUD_JPY",
    "CADJPY": "OANDA:CAD_JPY",
    "NZDJPY": "OANDA:NZD_JPY",
    "EURGBP": "OANDA:EUR_GBP",
    "GBPAUD": "OANDA:GBP_AUD",
    "GBPNZD": "OANDA:GBP_NZD",
    "XAUUSD": "OANDA:XAU_USD",
    "XAGUSD": "OANDA:XAG_USD",
}

# Keywords that indicate high market impact
HIGH_IMPACT_KEYWORDS = [
    'federal reserve', 'fed', 'interest rate', 'inflation', 'cpi',
    'nfp', 'non-farm', 'gdp', 'recession', 'central bank', 'ecb',
    'bank of england', 'boe', 'boj', 'rba', 'rate hike', 'rate cut',
    'quantitative', 'fomc', 'powell', 'lagarde', 'tariff', 'trade war'
]

# ---------------------------------------------------------------
# ECONOMIC CALENDAR — most critical for day trading
# ---------------------------------------------------------------
CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

def get_economic_calendar() -> list:
    """
    Fetch this week's high-impact economic events.
    FREE from ForexFactory — no API key needed.
    Returns only HIGH impact events.
    """
    try:
        r = requests.get(CALENDAR_URL, timeout=10)
        r.raise_for_status()
        events = r.json()
        high   = [e for e in events if e.get('impact') == 'High']
        return high
    except Exception as e:
        print(f"  Calendar fetch failed: {e}")
        return []


def is_news_blackout(window_minutes: int = 30) -> dict:
    """
    Check if we are within a news blackout window.
    For day trading: pause ALL trading 30 min before
    and 15 min after any high-impact news event.
    This is the single most important protection for day traders.
    """
    events = get_economic_calendar()
    now    = datetime.now(timezone.utc)

    upcoming = []
    active   = []
    recent   = []

    for event in events:
        try:
            raw_date  = event.get('date', '')
            # Parse various date formats
            for fmt in ('%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%dT%H:%M:%S',
                        '%m/%d/%Y %I:%M%p'):
                try:
                    event_time = datetime.strptime(raw_date, fmt)
                    if event_time.tzinfo is None:
                        event_time = event_time.replace(tzinfo=timezone.utc)
                    break
                except Exception:
                    continue
            else:
                continue

            diff_mins = (event_time - now).total_seconds() / 60

            if 0 < diff_mins <= window_minutes:
                upcoming.append({
                    'title':    event.get('title', ''),
                    'country':  event.get('country', ''),
                    'mins_away':round(diff_mins, 0),
                })
            elif -15 <= diff_mins <= 0:
                active.append({
                    'title':    event.get('title', ''),
                    'country':  event.get('country', ''),
                    'mins_ago': round(abs(diff_mins), 0),
                })
            elif -60 <= diff_mins < -15:
                recent.append({
                    'title':    event.get('title', ''),
                    'country':  event.get('country', ''),
                    'mins_ago': round(abs(diff_mins), 0),
                })
        except Exception:
            continue

    blackout = len(upcoming) > 0 or len(active) > 0

    return {
        'blackout':  blackout,
        'upcoming':  upcoming,
        'active':    active,
        'recent':    recent,
        'reason':    (f"HIGH IMPACT NEWS: {upcoming[0]['title']} "
                      f"in {upcoming[0]['mins_away']:.0f} mins"
                      if upcoming else
                      f"NEWS JUST RELEASED: {active[0]['title']}"
                      if active else
                      'No news blackout active'),
    }

def fetch_forex_news(category: str = "forex",
                     min_relevance: float = 0.5) -> list:
    """
    Fetch latest forex news from Finnhub.
    Returns list of articles with sentiment scores.
    Falls back to general financial news if no key.
    """
    if not FINNHUB_KEY:
        print("  No FINNHUB_KEY in .env — using newsapi fallback")
        return _fetch_newsapi_fallback()

    try:
        url    = f"{BASE_URL}/news?category={category}"
        params = {"token": FINNHUB_KEY}
        r      = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        articles = r.json()
        print(f"  Finnhub: {len(articles)} articles fetched")
        return articles[:20]  # Last 20 articles
    except Exception as e:
        print(f"  Finnhub failed: {e} — trying fallback")
        return _fetch_newsapi_fallback()


def _fetch_newsapi_fallback() -> list:
    """
    Fallback: fetch forex news from free RSS/public APIs.
    No API key needed.
    """
    try:
        # Use GNews free API — no key needed for basic use
        url    = ("https://gnews.io/api/v4/search"
                  "?q=forex+interest+rates+central+bank"
                  "&lang=en&max=10&apikey=free")
        r      = requests.get(url, timeout=10)
        data   = r.json()
        arts   = data.get('articles', [])
        # Normalize to same format as Finnhub
        result = []
        for a in arts:
            result.append({
                'headline': a.get('title', ''),
                'summary':  a.get('description', ''),
                'url':      a.get('url', ''),
                'datetime': a.get('publishedAt', ''),
                'source':   a.get('source', {}).get('name', ''),
            })
        print(f"  GNews fallback: {len(result)} articles")
        return result
    except Exception as e:
        print(f"  All news sources failed: {e}")
        return []

def score_article_sentiment(headline: str,
                            summary: str = "") -> dict:
    """
    Score a news article as BULLISH, BEARISH, or NEUTRAL for USD.
    Uses keyword matching — no AI needed for basic scoring.
    """
    text = (headline + " " + summary).lower()

    # Bullish USD keywords
    bull_keywords = [
        'rate hike', 'hawkish', 'strong jobs', 'beat expectations',
        'higher than expected', 'inflation rises', 'fed raises',
        'dollar surges', 'usd gains', 'risk off', 'safe haven bid',
        'strong gdp', 'economy grows', 'labour market strong',
    ]

    # Bearish USD keywords
    bear_keywords = [
        'rate cut', 'dovish', 'weak jobs', 'miss expectations',
        'lower than expected', 'inflation falls', 'fed cuts',
        'dollar drops', 'usd falls', 'risk on', 'recession fears',
        'gdp shrinks', 'economy slows', 'layoffs', 'unemployment rises',
    ]

    bull_score = sum(1 for kw in bull_keywords if kw in text)
    bear_score = sum(1 for kw in bear_keywords if kw in text)

    # High impact check
    is_high_impact = any(kw in text for kw in HIGH_IMPACT_KEYWORDS)

    if bull_score > bear_score:
        sentiment = 'BULLISH_USD'
        score     = bull_score
    elif bear_score > bull_score:
        sentiment = 'BEARISH_USD'
        score     = bear_score
    else:
        sentiment = 'NEUTRAL'
        score     = 0

    return {
        'sentiment':       sentiment,
        'score':           score,
        'is_high_impact':  is_high_impact,
        'bull_signals':    bull_score,
        'bear_signals':    bear_score,
    }


def analyze_news_sentiment(articles: list) -> dict:
    """
    Analyze all articles and return aggregate sentiment.
    """
    if not articles:
        return {
            'overall':       'NEUTRAL',
            'bull_count':    0,
            'bear_count':    0,
            'neutral_count': 0,
            'high_impact':   0,
            'articles':      [],
            'note':          'No articles available',
        }

    scored    = []
    bull_count= 0
    bear_count= 0
    neut_count= 0
    hi_impact = 0

    for art in articles:
        headline = art.get('headline', art.get('title', ''))
        summary  = art.get('summary', art.get('description', ''))
        result   = score_article_sentiment(headline, summary)

        if result['sentiment'] == 'BULLISH_USD':
            bull_count += 1
        elif result['sentiment'] == 'BEARISH_USD':
            bear_count += 1
        else:
            neut_count += 1

        if result['is_high_impact']:
            hi_impact += 1

        scored.append({
            'headline':   headline[:80],
            'sentiment':  result['sentiment'],
            'score':      result['score'],
            'high_impact':result['is_high_impact'],
        })

    # Overall bias
    if bull_count > bear_count * 1.5:
        overall = 'BULLISH_USD'
        note    = 'News flow bullish for USD'
    elif bear_count > bull_count * 1.5:
        overall = 'BEARISH_USD'
        note    = 'News flow bearish for USD'
    else:
        overall = 'NEUTRAL'
        note    = 'Mixed or neutral news flow'

    return {
        'overall':       overall,
        'bull_count':    bull_count,
        'bear_count':    bear_count,
        'neutral_count': neut_count,
        'high_impact':   hi_impact,
        'total':         len(articles),
        'articles':      scored[:5],  # Top 5 for display
        'note':          note,
    }

def get_volatility_risk(blackout_result: dict,
                        sentiment_result: dict) -> dict:
    """
    For day trading — convert news data into a volatility risk score.
    High volatility risk = avoid scalping, widen stops, reduce size.
    This is more useful than sentiment direction for day traders.

    Returns score 0-100 (0=calm, 100=extreme volatility risk)
    """
    score  = 0
    flags  = []

    # Active blackout = highest risk
    if blackout_result.get('blackout'):
        score += 60
        flags.append('HIGH IMPACT NEWS ACTIVE OR IMMINENT')

    # Upcoming news within 60 mins = elevated risk
    upcoming = blackout_result.get('upcoming', [])
    if upcoming:
        mins_away = upcoming[0].get('mins_away', 999)
        if mins_away <= 15:
            score += 50; flags.append(f"NEWS IN {mins_away:.0f} MINS — EXTREME")
        elif mins_away <= 30:
            score += 35; flags.append(f"NEWS IN {mins_away:.0f} MINS — HIGH")
        elif mins_away <= 60:
            score += 20; flags.append(f"NEWS IN {mins_away:.0f} MINS — MODERATE")

    # Recent news = aftermath volatility
    recent = blackout_result.get('recent', [])
    if recent:
        score += 15
        flags.append(f"NEWS RELEASED {recent[0].get('mins_ago',0):.0f} mins ago")

    # High impact article count
    hi_count = sentiment_result.get('high_impact', 0)
    if hi_count >= 3:
        score += 15; flags.append(f"{hi_count} high impact headlines")
    elif hi_count >= 1:
        score += 8; flags.append(f"{hi_count} high impact headline")

    score = min(score, 100)

    if score >= 70:
        level = 'EXTREME'
        advice= 'DO NOT TRADE — news volatility too high'
    elif score >= 40:
        level = 'HIGH'
        advice= 'Reduce size, widen stops, avoid new entries'
    elif score >= 20:
        level = 'MODERATE'
        advice= 'Be cautious — news could spike price'
    else:
        level = 'LOW'
        advice= 'News conditions clear for trading'

    return {
        'volatility_score': score,
        'level':            level,
        'advice':           advice,
        'flags':            flags,
    }


def get_news_sentiment() -> dict:
    """Master function — fetch news, analyze sentiment, check blackout, score volatility."""
    print("Fetching forex news...")
    articles   = fetch_forex_news()
    sentiment  = analyze_news_sentiment(articles)
    blackout   = is_news_blackout(window_minutes=30)
    volatility = get_volatility_risk(blackout, sentiment)
    sentiment['blackout']   = blackout
    sentiment['volatility'] = volatility
    return sentiment


# =============================================================
# STANDALONE TEST
# =============================================================
if __name__ == "__main__":
    print("Running News Sentiment Analysis...\n")
    result = get_news_sentiment()

    print(f"\n{'='*55}")
    print(f"  NEWS SENTIMENT REPORT")
    print(f"{'='*55}")
    print(f"  Overall Bias  : {result['overall']}")
    print(f"  Total Articles: {result['total']}")
    print(f"  Bullish USD   : {result['bull_count']}")
    print(f"  Bearish USD   : {result['bear_count']}")
    print(f"  Neutral       : {result['neutral_count']}")
    print(f"  High Impact   : {result['high_impact']}")
    print(f"  Note          : {result['note']}")

    if result['articles']:
        print(f"\n  TOP ARTICLES:")
        for i, art in enumerate(result['articles'], 1):
            icon = "📈" if art['sentiment'] == 'BULLISH_USD' else \
                   "📉" if art['sentiment'] == 'BEARISH_USD' else "↔️"
            hi   = " ⚡HIGH IMPACT" if art['high_impact'] else ""
            print(f"\n  {i}. {icon} {art['sentiment']}{hi}")
            print(f"     {art['headline']}")
    print(f"{'='*55}")

    if not FINNHUB_KEY:
        print(f"\n  TIP: Add FINNHUB_KEY to .env for better results.")
        print(f"  Free key at: https://finnhub.io/register")
