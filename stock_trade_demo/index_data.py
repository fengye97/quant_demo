"""
CSI 1000 (sh000852) index data fetcher.

Fetches daily K-line data from Sina Finance API, computes monthly returns,
and caches to .cache/csi1000_monthly.csv to avoid re-fetching on every run.

Usage:
    from index_data import get_index_returns
    monthly_rets = get_index_returns()  # Series indexed by month-end date
"""

import os
import json
import urllib.request
import pandas as pd
import numpy as np

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')
CACHE_FILE = os.path.join(CACHE_DIR, 'csi1000_monthly.csv')

SINA_URL = (
    'https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/'
    'CN_MarketData.getKLineData?symbol=sh000852&scale=240&datalen=8000'
)


def _fetch_daily_kline():
    """
    Fetch CSI 1000 daily K-line data from Sina API.

    Returns DataFrame with columns: date, open, high, low, close, volume.
    """
    req = urllib.request.Request(SINA_URL)
    req.add_header('User-Agent',
                   'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) '
                   'Chrome/120.0.0.0 Safari/537.36')
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode('utf-8')

    data = json.loads(raw)

    records = []
    for item in data:
        records.append({
            'date': item['day'],
            'open': float(item['open']),
            'high': float(item['high']),
            'low': float(item['low']),
            'close': float(item['close']),
            'volume': float(item['volume']),
        })

    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def _compute_monthly_returns(df_daily):
    """
    Compute monthly returns from daily close prices.

    Uses daily pct_change() then resamples to month-end by compounding
    daily returns within each calendar month.

    Returns Series indexed by month-end date.
    """
    df = df_daily.set_index('date').sort_index()
    daily_returns = df['close'].pct_change()
    # Compound daily returns within each calendar month
    monthly = (1 + daily_returns).resample('M').prod() - 1
    monthly = monthly.dropna()
    monthly.name = 'csi1000_return'
    return monthly


def get_index_returns(force_refetch=False):
    """
    Get CSI 1000 monthly returns, with file-based caching.

    On first call: fetches from Sina API, computes monthly returns,
    caches to .cache/csi1000_monthly.csv.
    On subsequent calls: reads from cache.

    Parameters:
        force_refetch: if True, ignore cache and re-fetch from API.

    Returns:
        pd.Series indexed by month-end datetime, values are monthly
        returns as decimals (e.g. 0.05 = 5%).
    """
    if not force_refetch and os.path.exists(CACHE_FILE):
        s = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
        # When saved from a named Series, the CSV has a header; read back as Series
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s.name = 'csi1000_return'
        print(f"[index_data] Loaded {len(s)} months from cache: {CACHE_FILE}")
        return s

    os.makedirs(CACHE_DIR, exist_ok=True)

    try:
        print("[index_data] Fetching CSI 1000 daily K-line from Sina...")
        df_daily = _fetch_daily_kline()
        print(f"[index_data] Fetched {len(df_daily)} daily bars "
              f"({df_daily['date'].min().strftime('%Y-%m-%d')} to "
              f"{df_daily['date'].max().strftime('%Y-%m-%d')})")

        monthly = _compute_monthly_returns(df_daily)
        print(f"[index_data] Computed {len(monthly)} monthly returns "
              f"({monthly.index.min().strftime('%Y-%m-%d')} to "
              f"{monthly.index.max().strftime('%Y-%m-%d')})")

        monthly.to_csv(CACHE_FILE, header=True)
        print(f"[index_data] Cached to {CACHE_FILE}")
        return monthly

    except Exception as e:
        print(f"[index_data] ERROR fetching data: {e}")
        if os.path.exists(CACHE_FILE):
            print("[index_data] Falling back to cached data")
            s = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            s.name = 'csi1000_return'
            return s
        raise RuntimeError(
            f"Failed to fetch CSI 1000 data and no cache exists: {e}"
        )


# ═══════════════════════════════════════════════════════════════════
# Utility: build index-by-period lookup for date alignment
# ═══════════════════════════════════════════════════════════════════

def build_period_lookup(index_returns):
    """
    Build a dict mapping (year, month) → index return for fast lookup.

    This handles the common case where strategy trading dates may not be
    exactly month-end, but we want to align by calendar month.

    Parameters:
        index_returns: Series indexed by month-end dates.

    Returns:
        dict: {(year_int, month_int): return_float}
    """
    lookup = {}
    for idx, val in index_returns.items():
        ts = pd.to_datetime(idx)
        lookup[(ts.year, ts.month)] = float(val)
    return lookup


def get_index_return_for_date(date, period_lookup):
    """
    Look up the CSI 1000 monthly return for a given date's calendar month.

    Parameters:
        date:          datetime or date-like object
        period_lookup: dict from build_period_lookup()

    Returns:
        float: monthly return (0.0 if no data for that month)
    """
    ts = pd.to_datetime(date)
    return period_lookup.get((ts.year, ts.month), 0.0)
