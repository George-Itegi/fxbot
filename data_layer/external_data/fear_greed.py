# =============================================================
# data_layer/external_data/fear_greed.py
# PURPOSE: Fetch Fear & Greed Index — measures market emotion.
# 0 = Extreme Fear (buy opportunity)
# 100 = Extreme Greed (sell opportunity)
# Sources: Alternative.me (crypto proxy) + CNN (equity market)
# Run standalone to test.
# =============================================================

import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


def fetch_alternative_me() -> dict:
    """
    Fetch Fear & Greed from Alternative.me.
    Originally crypto but highly correlated with
    overall market risk sentiment.
    Free, no API key needed.
    """
    try:
        url = "https://api.alternative.me/fng/?limit=2&format=json"
        r   = requests.get(url, timeout=10)
        r.raise_for_status()
        data    = r.json()
        entries = data.get('data', [])
        if not entries:
            return {'error': 'No data from Alternative.me'}

        latest = entries[0]
        prev   = entries[1] if len(entries) > 1 else entries[0]

        score      = int(latest['value'])
        prev_score = int(prev['value'])
        label      = latest['value_classification']
        change     = score - prev_score

        return {
            'source':     'alternative.me',
            'score':      score,
            'prev_score': prev_score,
            'change':     change,
            'label':      label,
            'timestamp':  latest.get('timestamp', ''),
        }
    except Exception as e:
        return {'error': str(e)}

def fetch_cnn_fear_greed() -> dict:
    """
    Fetch CNN Fear & Greed Index — equity market focused.
    Measures 7 indicators: momentum, strength, breadth,
    put/call ratio, junk bonds, market volatility, safe havens.
    Free, no API key needed.
    """
    try:
        url = ("https://production.dataviz.cnn.io/index/"
               "fearandgreed/graphdata")
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()

        fg   = data.get('fear_and_greed', {})
        prev = data.get('fear_and_greed_historical', {})

        score = float(fg.get('score', 0))
        label = fg.get('rating', 'unknown')

        # Previous close score for comparison
        prev_score = 0
        if prev and 'data' in prev and len(prev['data']) > 1:
            prev_score = float(prev['data'][-2].get('y', 0))

        change = round(score - prev_score, 1)

        return {
            'source':     'cnn',
            'score':      round(score, 1),
            'prev_score': round(prev_score, 1),
            'change':     change,
            'label':      label,
        }
    except Exception as e:
        return {'error': str(e), 'source': 'cnn'}


def interpret_fear_greed(score: float) -> dict:
    """
    Interpret a Fear & Greed score into trading signals.
    Contrarian: extreme fear = buy opportunity,
                extreme greed = sell opportunity.
    """
    if score <= 20:
        zone   = 'EXTREME_FEAR'
        signal = 'BUY'
        note   = 'Extreme fear — historically strong buy opportunity'
    elif score <= 40:
        zone   = 'FEAR'
        signal = 'BUY'
        note   = 'Fear dominant — lean bullish on dips'
    elif score <= 60:
        zone   = 'NEUTRAL'
        signal = 'NEUTRAL'
        note   = 'Balanced sentiment — no contrarian edge'
    elif score <= 80:
        zone   = 'GREED'
        signal = 'SELL'
        note   = 'Greed dominant — lean bearish on rallies'
    else:
        zone   = 'EXTREME_GREED'
        signal = 'SELL'
        note   = 'Extreme greed — historically strong sell opportunity'

    return {'zone': zone, 'signal': signal, 'note': note}

def get_day_trading_gate(fg_result: dict) -> dict:
    """
    For day trading — use Fear/Greed as a caution gate not a blocker.
    Extreme zones reduce position confidence but don't block trades
    unless the score is static (no momentum = dangerous).
    """
    score  = fg_result.get('score', 50)
    change = fg_result.get('change', 0)
    zone   = fg_result.get('zone', 'NEUTRAL')

    # Extreme zones with no recovery = dangerous
    if zone == 'EXTREME_FEAR' and change <= 0:
        return {
            'caution':  True,
            'reduce_size': True,
            'reason': f'Extreme fear ({score}) still falling — reduce size on longs',
        }
    if zone == 'EXTREME_GREED' and change >= 0:
        return {
            'caution':  True,
            'reduce_size': True,
            'reason': f'Extreme greed ({score}) still rising — reduce size on shorts',
        }
    # Recovering from extreme = actually a good signal
    if zone == 'EXTREME_FEAR' and change > 3:
        return {
            'caution':     False,
            'reduce_size': False,
            'reason': f'Fear recovering (+{change}) — good long opportunity',
        }
    if zone == 'EXTREME_GREED' and change < -3:
        return {
            'caution':     False,
            'reduce_size': False,
            'reason': f'Greed falling ({change}) — good short opportunity',
        }
    return {
        'caution':     False,
        'reduce_size': False,
        'reason': f'Sentiment neutral ({score}) — no impact on sizing',
    }


def get_fear_greed() -> dict:
    """
    Master function — fetch from both sources and combine.
    Returns unified fear/greed report.
    """
    alt = fetch_alternative_me()
    cnn = fetch_cnn_fear_greed()

    # Use CNN as primary, alternative.me as backup
    if 'error' not in cnn:
        primary       = cnn
        primary_label = 'CNN'
    elif 'error' not in alt:
        primary       = alt
        primary_label = 'Alternative.me'
    else:
        return {
            'score':  50,
            'zone':   'NEUTRAL',
            'signal': 'NEUTRAL',
            'note':   'Both sources failed — defaulting to neutral',
            'error':  True,
        }

    score  = primary['score']
    interp = interpret_fear_greed(score)

    result = {
        'score':        score,
        'prev_score':   primary.get('prev_score', score),
        'change':       primary.get('change', 0),
        'label':        primary.get('label', ''),
        'zone':         interp['zone'],
        'signal':       interp['signal'],
        'note':         interp['note'],
        'source':       primary_label,
        'alt_score':    alt.get('score') if 'error' not in alt else None,
        'cnn_score':    cnn.get('score') if 'error' not in cnn else None,
    }
    result['day_gate'] = get_day_trading_gate(result)
    return result


# =============================================================
# STANDALONE TEST
# =============================================================
if __name__ == "__main__":
    print("Fetching Fear & Greed Index...\n")

    # Test individual sources
    print("--- Alternative.me ---")
    alt = fetch_alternative_me()
    if 'error' not in alt:
        print(f"  Score  : {alt['score']} ({alt['label']})")
        print(f"  Change : {alt['change']:+d} from previous")
    else:
        print(f"  Error  : {alt['error']}")

    print("\n--- CNN Fear & Greed ---")
    cnn = fetch_cnn_fear_greed()
    if 'error' not in cnn:
        print(f"  Score  : {cnn['score']} ({cnn['label']})")
        print(f"  Change : {cnn['change']:+.1f} from previous")
    else:
        print(f"  Error  : {cnn['error']}")

    # Combined report
    print("\n--- COMBINED REPORT ---")
    result = get_fear_greed()
    print(f"{'='*50}")
    print(f"  FEAR & GREED INDEX")
    print(f"{'='*50}")
    print(f"  Score    : {result['score']}")
    print(f"  Change   : {result.get('change', 0):+.1f} from yesterday")
    print(f"  Label    : {result['label']}")
    print(f"  Zone     : {result['zone']}")
    print(f"  Signal   : {result['signal']}")
    print(f"  Note     : {result['note']}")
    print(f"  Source   : {result['source']}")
    if result.get('alt_score'):
        print(f"  Alt.me   : {result['alt_score']}")
    if result.get('cnn_score'):
        print(f"  CNN      : {result['cnn_score']}")
    print(f"{'='*50}")
