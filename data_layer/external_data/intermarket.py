# =============================================================
# data_layer/external_data/intermarket.py
# PURPOSE: Fetch key correlated markets that drive forex prices.
# VIX, DXY, Gold, Oil, S&P500, Bonds — the macro picture.
# Run standalone to test.
# =============================================================

import yfinance as yf
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# Instruments to track and their yfinance tickers
TICKERS = {
    'vix':   '^VIX',       # Fear gauge — risk off = USD up, JPY up
    'dxy':   'DX-Y.NYB',   # US Dollar Index — drives all USD pairs
    'gold':  'GC=F',       # Gold — risk off asset, inverse USD
    'oil':   'CL=F',       # Crude oil — drives CAD, NOK
    'sp500': '^GSPC',      # S&P 500 — risk appetite
    'bonds': '^TNX',       # US 10Y yield — drives USD strength
    'nikkei':'^N225',      # Japan market — JPY proxy
}

def fetch_intermarket() -> dict:
    """
    Fetch latest price and daily change for all key instruments.
    Returns clean dict with current values and % changes.
    """
    results = {}
    print("Fetching intermarket data...")

    for key, ticker in TICKERS.items():
        try:
            data = yf.download(ticker, period='5d',
                               interval='1d',
                               progress=False,
                               auto_adjust=True)
            if data.empty or len(data) < 2:
                results[key] = {'error': 'No data'}
                continue

            # Handle multi-index columns from newer yfinance versions
            close = data['Close']
            high  = data['High']
            low   = data['Low']
            if hasattr(close, 'columns'):
                close = close.iloc[:, 0]
                high  = high.iloc[:, 0]
                low   = low.iloc[:, 0]

            current = float(close.iloc[-1])
            prev    = float(close.iloc[-2])
            change  = round((current - prev) / prev * 100, 3)
            high5d  = float(high.max())
            low5d   = float(low.min())

            results[key] = {
                'ticker':    ticker,
                'current':   round(current, 4),
                'prev':      round(prev, 4),
                'change_pct':change,
                'high_5d':   round(high5d, 4),
                'low_5d':    round(low5d, 4),
                'direction': 'UP' if change > 0 else 'DOWN',
            }
            print(f"  {key:<8}: {current:.4f}  ({change:+.2f}%)")
        except Exception as e:
            results[key] = {'error': str(e)}
            print(f"  {key:<8}: FAILED — {e}")

    return results


def interpret_intermarket(data: dict) -> dict:
    """
    Interpret intermarket data into forex trading signals.
    Returns context for each major currency pair.
    """
    signals = {}

    vix    = data.get('vix', {})
    dxy    = data.get('dxy', {})
    gold   = data.get('gold', {})
    oil    = data.get('oil', {})
    sp500  = data.get('sp500', {})
    bonds  = data.get('bonds', {})

    vix_val    = vix.get('current', 20)
    vix_chg    = vix.get('change_pct', 0)
    dxy_chg    = dxy.get('change_pct', 0)
    gold_chg   = gold.get('change_pct', 0)
    sp500_chg  = sp500.get('change_pct', 0)
    bonds_chg  = bonds.get('change_pct', 0)

    # Risk environment
    if vix_val > 30:
        risk_env = 'RISK_OFF'
        risk_note = 'High VIX — fear dominant, safe havens favored'
    elif vix_val > 20:
        risk_env = 'CAUTIOUS'
        risk_note = 'Elevated VIX — uncertain, be selective'
    else:
        risk_env = 'RISK_ON'
        risk_note = 'Low VIX — risk appetite healthy'

    # USD bias from DXY + bonds
    if dxy_chg > 0.3 and bonds_chg > 0:
        usd_bias = 'STRONG_BULL'
    elif dxy_chg > 0.1:
        usd_bias = 'BULL'
    elif dxy_chg < -0.3 and bonds_chg < 0:
        usd_bias = 'STRONG_BEAR'
    elif dxy_chg < -0.1:
        usd_bias = 'BEAR'
    else:
        usd_bias = 'NEUTRAL'

    # Implied signals per pair
    signals['EURUSD'] = _usd_pair_signal('EURUSD', usd_bias, inverse=True)
    signals['GBPUSD'] = _usd_pair_signal('GBPUSD', usd_bias, inverse=True)
    signals['AUDUSD'] = _risk_pair_signal('AUDUSD', risk_env, sp500_chg)
    signals['NZDUSD'] = _risk_pair_signal('NZDUSD', risk_env, sp500_chg)
    signals['USDJPY'] = _safe_haven_signal('USDJPY', risk_env, usd_bias)
    signals['USDCHF'] = _safe_haven_signal('USDCHF', risk_env, usd_bias)
    signals['USDCAD'] = _oil_signal('USDCAD', oil.get('change_pct', 0))
    signals['XAUUSD'] = _gold_signal(gold_chg, risk_env, usd_bias)

    return {
        'risk_environment': risk_env,
        'risk_note':        risk_note,
        'usd_bias':         usd_bias,
        'vix':              vix_val,
        'vix_change':       vix_chg,
        'dxy_change':       dxy_chg,
        'bonds_change':     bonds_chg,
        'gold_change':      gold_chg,
        'oil_change':       oil.get('change_pct', 0),
        'sp500_change':     sp500_chg,
        'pair_signals':     signals,
        'session':          get_current_session(),
        'day_trade_ok':     is_good_for_day_trading(risk_env, vix_val),
    }


def get_current_session() -> str:
    """Return current trading session based on UTC time."""
    hour = datetime.now(timezone.utc).hour
    if 7 <= hour < 10:
        return 'LONDON_OPEN'
    elif 10 <= hour < 12:
        return 'LONDON_MID'
    elif 12 <= hour < 16:
        return 'NY_LONDON_OVERLAP'
    elif 16 <= hour < 20:
        return 'NY_SESSION'
    elif 0 <= hour < 7:
        return 'ASIAN_SESSION'
    else:
        return 'DEAD_ZONE'


def is_good_for_day_trading(risk_env: str, vix: float) -> dict:
    """
    Tell the bot whether current macro conditions
    are suitable for day trading right now.
    """
    if vix > 40:
        return {
            'allowed':  False,
            'reason':   f'VIX={vix:.1f} — extreme volatility, spreads wide, skip',
        }
    if risk_env == 'RISK_OFF' and vix > 30:
        return {
            'allowed':  True,
            'reason':   f'Risk-off but VIX manageable — trade safe havens only',
        }
    # DEAD_ZONE block DISABLED for testing — all sessions tradable
    # if risk_env == 'DEAD_ZONE':
    #     return {
    #         'allowed':  False,
    #         'reason':   'Dead zone — low liquidity, avoid trading',
    #     }
    return {
        'allowed':  True,
        'reason':   f'Conditions OK — {risk_env}, VIX={vix:.1f}',
    }

def _usd_pair_signal(symbol, usd_bias, inverse=False) -> dict:
    """Signal for USD-denominated pairs."""
    if inverse:
        bias = 'BULLISH' if 'BEAR' in usd_bias else \
               'BEARISH' if 'BULL' in usd_bias else 'NEUTRAL'
    else:
        bias = 'BULLISH' if 'BULL' in usd_bias else \
               'BEARISH' if 'BEAR' in usd_bias else 'NEUTRAL'
    return {'bias': bias, 'driver': f'DXY {usd_bias}'}


def _risk_pair_signal(symbol, risk_env, sp500_chg) -> dict:
    """AUD, NZD follow risk appetite."""
    if risk_env == 'RISK_ON' and sp500_chg > 0:
        return {'bias': 'BULLISH', 'driver': 'Risk-on environment'}
    elif risk_env == 'RISK_OFF':
        return {'bias': 'BEARISH', 'driver': 'Risk-off — AUD/NZD weaken'}
    return {'bias': 'NEUTRAL', 'driver': 'Mixed risk signals'}


def _safe_haven_signal(symbol, risk_env, usd_bias) -> dict:
    """JPY and CHF strengthen in risk-off."""
    if risk_env == 'RISK_OFF':
        bias = 'BEARISH'  # USDJPY falls when JPY strengthens
        driver = 'Risk-off — JPY/CHF safe haven demand'
    elif 'BULL' in usd_bias:
        bias = 'BULLISH'
        driver = 'Strong USD dominates'
    else:
        bias = 'NEUTRAL'
        driver = 'No clear driver'
    return {'bias': bias, 'driver': driver}


def _oil_signal(symbol, oil_chg) -> dict:
    """CAD follows oil prices — oil up = CAD up = USDCAD down."""
    if oil_chg > 1.0:
        return {'bias': 'BEARISH', 'driver': f'Oil +{oil_chg:.1f}% → CAD strength'}
    elif oil_chg < -1.0:
        return {'bias': 'BULLISH', 'driver': f'Oil {oil_chg:.1f}% → CAD weakness'}
    return {'bias': 'NEUTRAL', 'driver': 'Oil flat'}


def _gold_signal(gold_chg, risk_env, usd_bias) -> dict:
    """Gold rises in risk-off and USD weakness."""
    if risk_env == 'RISK_OFF' or 'BEAR' in usd_bias:
        return {'bias': 'BULLISH', 'driver': 'Risk-off or USD weakness'}
    elif risk_env == 'RISK_ON' and 'BULL' in usd_bias:
        return {'bias': 'BEARISH', 'driver': 'Risk-on + strong USD = gold pressure'}
    return {'bias': 'NEUTRAL', 'driver': 'Mixed signals'}


# =============================================================
# STANDALONE TEST
# =============================================================
if __name__ == "__main__":
    print("Fetching intermarket data...\n")
    raw  = fetch_intermarket()
    interp = interpret_intermarket(raw)

    print(f"\n{'='*55}")
    print(f"  INTERMARKET REPORT")
    print(f"{'='*55}")
    print(f"  Risk Env   : {interp['risk_environment']}")
    print(f"  Note       : {interp['risk_note']}")
    print(f"  USD Bias   : {interp['usd_bias']}")
    print(f"  VIX        : {interp['vix']:.2f}  ({interp['vix_change']:+.2f}%)")
    print(f"  DXY Change : {interp['dxy_change']:+.3f}%")
    print(f"  Bonds Chg  : {interp['bonds_change']:+.3f}%")
    print(f"  Gold Chg   : {interp['gold_change']:+.3f}%")
    print(f"  Oil Chg    : {interp['oil_change']:+.3f}%")
    print(f"  SP500 Chg  : {interp['sp500_change']:+.3f}%")
    print(f"\n  PAIR SIGNALS:")
    print(f"  {'Pair':<10} {'Bias':<12} Driver")
    print(f"  {'-'*50}")
    for pair, sig in interp['pair_signals'].items():
        icon = "📈" if sig['bias']=='BULLISH' else \
               "📉" if sig['bias']=='BEARISH' else "↔️"
        print(f"  {pair:<10} {sig['bias']:<12} {icon}  {sig['driver']}")
    print(f"{'='*55}")
