# =============================================================
# data_layer/external_data.py
# Fetches institutional-grade data from free APIs:
# Fear & Greed, VIX, DXY, Gold, Oil, Bonds, News sentiment
# This runs every hour and feeds the AI scoring engine.
# =============================================================

import requests
import yfinance as yf
from datetime import datetime
from core.logger import get_logger

log = get_logger(__name__)


def get_fear_greed() -> dict:
    """
    CNN Fear & Greed Index — 0=Extreme Fear, 100=Extreme Greed.
    Extreme fear = potential buy. Extreme greed = potential sell.
    """
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        data = r.json()
        score = data['fear_and_greed']['score']
        rating = data['fear_and_greed']['rating']
        log.info(f"[FEAR/GREED] Score: {score:.1f} | {rating}")
        return {"score": score, "rating": rating}
    except Exception as e:
        log.warning(f"[FEAR/GREED] Failed: {e}")
        return {"score": None, "rating": "unknown"}


def get_intermarket_data() -> dict:
    """
    Fetch key intermarket instruments via yfinance.
    These show macro risk appetite and USD strength.
    """
    tickers = {
        "vix":    "^VIX",      # Volatility / fear gauge
        "dxy":    "DX-Y.NYB",  # US Dollar Index
        "gold":   "GC=F",      # Gold futures
        "oil":    "CL=F",      # Crude oil
        "sp500":  "^GSPC",     # S&P 500
        "bonds":  "^TNX",      # 10-year US Treasury yield
        "nikkei": "^N225",     # Japanese market (JPY proxy)
    }
    result = {}
    try:
        for key, ticker in tickers.items():
            data = yf.download(ticker, period="2d", interval="1d",
                               progress=False, auto_adjust=True)
            if not data.empty:
                result[key] = round(float(data['Close'].iloc[-1]), 4)
                prev         = float(data['Close'].iloc[-2])
                result[f"{key}_change_pct"] = round(
                    (result[key] - prev) / prev * 100, 3)
        log.info(f"[INTERMARKET] VIX={result.get('vix')} | "
                 f"DXY={result.get('dxy')} | Gold={result.get('gold')}")
    except Exception as e:
        log.warning(f"[INTERMARKET] Failed: {e}")
    return result

def get_news_sentiment(keywords: list = None) -> dict:
    """
    Fetch recent forex news headlines and score overall sentiment.
    Uses Alpha Vantage free tier (500 calls/day).
    Set ALPHAVANTAGE_KEY in your .env file.
    """
    import os
    api_key = os.getenv("ALPHAVANTAGE_KEY", "demo")
    if keywords is None:
        keywords = ["forex", "federal reserve", "interest rate", "inflation"]

    try:
        topics = "financial_markets"
        url = (f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT"
               f"&topics={topics}&apikey={api_key}&limit=20")
        r = requests.get(url, timeout=10)
        data = r.json()
        feed = data.get("feed", [])
        if not feed:
            return {"score": None, "label": "neutral", "article_count": 0}

        scores = [float(a.get("overall_sentiment_score", 0)) for a in feed]
        avg = sum(scores) / len(scores)
        label = "bullish" if avg > 0.15 else "bearish" if avg < -0.15 else "neutral"
        log.info(f"[NEWS] Sentiment: {avg:.3f} ({label}) from {len(feed)} articles")
        return {"score": round(avg, 4), "label": label, "article_count": len(feed)}
    except Exception as e:
        log.warning(f"[NEWS] Sentiment fetch failed: {e}")
        return {"score": None, "label": "neutral", "article_count": 0}


def get_economic_calendar() -> list:
    """
    Returns high-impact events for this week from ForexFactory.
    Used to pause trading 30 minutes before/after major events.
    """
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        r = requests.get(url, timeout=10)
        events = r.json()
        high_impact = [e for e in events if e.get('impact') == 'High']
        log.info(f"[CALENDAR] {len(high_impact)} high-impact events this week.")
        return high_impact
    except Exception as e:
        log.warning(f"[CALENDAR] Failed to fetch calendar: {e}")
        return []


def get_all_external_data() -> dict:
    """Master function — call this once per hour to get everything."""
    log.info("[EXTERNAL DATA] Fetching all external data sources...")
    fear_greed   = get_fear_greed()
    intermarket  = get_intermarket_data()
    news         = get_news_sentiment()
    calendar     = get_economic_calendar()

    return {
        "fear_greed":      fear_greed,
        "vix":             intermarket.get("vix"),
        "vix_change":      intermarket.get("vix_change_pct"),
        "dxy":             intermarket.get("dxy"),
        "dxy_change":      intermarket.get("dxy_change_pct"),
        "gold_price":      intermarket.get("gold"),
        "oil_price":       intermarket.get("oil"),
        "sp500":           intermarket.get("sp500"),
        "bond_yield_10y":  intermarket.get("bonds"),
        "news_sentiment":  news.get("score"),
        "news_label":      news.get("label"),
        "high_impact_news":calendar,
        "fetched_at":      datetime.now().isoformat(),
    }
