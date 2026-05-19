"""
Benchmark index data fetcher.

Fetches daily K-line data from Sina Finance API, computes monthly returns,
and caches them under .cache/ to avoid re-fetching on every run.
"""

import os
import json
import urllib.request
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')

INDEX_CONFIGS = {
    'csi1000': {
        'name': 'CSI 1000',
        'symbol': 'sh000852',
        'cache_file': 'csi1000_monthly.csv',
        'daily_cache_file': 'csi1000_daily.csv',
        'series_name': 'csi1000_return',
    },
    'chinext': {
        'name': '创业板指数',
        'symbol': 'sz399006',
        'cache_file': 'chinext_monthly.csv',
        'daily_cache_file': 'chinext_daily.csv',
        'series_name': 'chinext_return',
    },
    'star50': {
        'name': '科创50指数',
        'symbol': 'sh000688',
        'cache_file': 'star50_monthly.csv',
        'daily_cache_file': 'star50_daily.csv',
        'series_name': 'star50_return',
    },
}


def _cache_path(index_id):
    return os.path.join(CACHE_DIR, INDEX_CONFIGS[index_id]['cache_file'])


def _daily_cache_path(index_id):
    return os.path.join(CACHE_DIR, INDEX_CONFIGS[index_id]['daily_cache_file'])


def _sina_url(symbol):
    return (
        'https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/'
        f'CN_MarketData.getKLineData?symbol={symbol}&scale=240&datalen=8000'
    )


def _fetch_daily_kline(symbol):
    """Fetch daily K-line data from Sina API."""
    req = urllib.request.Request(_sina_url(symbol))
    req.add_header('User-Agent',
                   'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) '
                   'Chrome/120.0.0.0 Safari/537.36')
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode('utf-8')

    data = json.loads(raw)
    records = [{
        'date': item['day'],
        'open': float(item['open']),
        'high': float(item['high']),
        'low': float(item['low']),
        'close': float(item['close']),
        'volume': float(item['volume']),
    } for item in data]

    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)


def get_index_daily(index_id='csi1000', force_refetch=False):
    """Get cached daily K-line data for a configured benchmark index."""
    if index_id not in INDEX_CONFIGS:
        raise ValueError(f'Unknown index_id: {index_id}')

    cfg = INDEX_CONFIGS[index_id]
    cache_file = _daily_cache_path(index_id)

    if not force_refetch and os.path.exists(cache_file):
        df = pd.read_csv(cache_file, parse_dates=['date'])
        return df.sort_values('date').reset_index(drop=True)

    os.makedirs(CACHE_DIR, exist_ok=True)

    try:
        print(f"[index_data] Fetching {cfg['name']} daily K-line from Sina...")
        df_daily = _fetch_daily_kline(cfg['symbol'])
        df_daily.to_csv(cache_file, index=False)
        print(f"[index_data] Cached daily K-line to {cache_file}")
        return df_daily
    except Exception as e:
        print(f"[index_data] ERROR fetching {cfg['name']} daily K-line: {e}")
        if os.path.exists(cache_file):
            print(f"[index_data] Falling back to cached daily K-line for {cfg['name']}")
            df = pd.read_csv(cache_file, parse_dates=['date'])
            return df.sort_values('date').reset_index(drop=True)
        raise RuntimeError(f"Failed to fetch {cfg['name']} daily data and no cache exists: {e}")


def build_index_panel(index_ids=None, force_refetch=False):
    """Build a daily wide panel for configured indexes."""
    index_ids = index_ids or list(INDEX_CONFIGS.keys())
    panel = None

    for index_id in index_ids:
        cfg = INDEX_CONFIGS[index_id]
        daily = get_index_daily(index_id=index_id, force_refetch=force_refetch).copy()
        renamed = daily.rename(columns={
            'open': f'{index_id}_open',
            'high': f'{index_id}_high',
            'low': f'{index_id}_low',
            'close': f'{index_id}_close',
            'volume': f'{index_id}_volume',
        })
        renamed[f'{index_id}_name'] = cfg['name']
        keep_cols = [
            'date',
            f'{index_id}_open', f'{index_id}_high', f'{index_id}_low',
            f'{index_id}_close', f'{index_id}_volume', f'{index_id}_name',
        ]
        renamed = renamed[keep_cols]
        if panel is None:
            panel = renamed
        else:
            panel = panel.merge(renamed, on='date', how='outer')

    if panel is None:
        return pd.DataFrame(columns=['date'])

    panel = panel.sort_values('date').reset_index(drop=True)
    panel['交易日期'] = pd.to_datetime(panel['date'])
    return panel


def _compute_monthly_returns(df_daily, series_name):
    """Compute monthly returns from daily close prices."""
    df = df_daily.set_index('date').sort_index()
    daily_returns = df['close'].pct_change()
    monthly = (1 + daily_returns).resample('M').prod() - 1
    monthly = monthly.dropna()
    monthly.name = series_name
    return monthly


def get_index_returns(index_id='csi1000', force_refetch=False):
    """Get monthly returns for a configured benchmark index."""
    if index_id not in INDEX_CONFIGS:
        raise ValueError(f'Unknown index_id: {index_id}')

    cfg = INDEX_CONFIGS[index_id]
    cache_file = _cache_path(index_id)

    if not force_refetch and os.path.exists(cache_file):
        s = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s.name = cfg['series_name']
        print(f"[index_data] Loaded {cfg['name']} {len(s)} months from cache: {cache_file}")
        return s

    os.makedirs(CACHE_DIR, exist_ok=True)

    try:
        df_daily = get_index_daily(index_id=index_id, force_refetch=force_refetch)
        print(f"[index_data] Fetched {len(df_daily)} daily bars "
              f"({df_daily['date'].min().strftime('%Y-%m-%d')} to "
              f"{df_daily['date'].max().strftime('%Y-%m-%d')})")

        monthly = _compute_monthly_returns(df_daily, cfg['series_name'])
        print(f"[index_data] Computed {len(monthly)} monthly returns "
              f"({monthly.index.min().strftime('%Y-%m-%d')} to "
              f"{monthly.index.max().strftime('%Y-%m-%d')})")

        monthly.to_csv(cache_file, header=True)
        print(f"[index_data] Cached to {cache_file}")
        return monthly

    except Exception as e:
        print(f"[index_data] ERROR fetching {cfg['name']}: {e}")
        if os.path.exists(cache_file):
            print(f"[index_data] Falling back to cached {cfg['name']} data")
            s = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            s.name = cfg['series_name']
            return s
        raise RuntimeError(f"Failed to fetch {cfg['name']} data and no cache exists: {e}")


# ═══════════════════════════════════════════════════════════════════
# Utility: build index-by-period lookup for date alignment
# ═══════════════════════════════════════════════════════════════════


def build_period_lookup(index_returns):
    """Build a dict mapping (year, month) → index return for fast lookup."""
    lookup = {}
    for idx, val in index_returns.items():
        ts = pd.to_datetime(idx)
        lookup[(ts.year, ts.month)] = float(val)
    return lookup


def get_index_return_for_date(date, period_lookup):
    """Look up the monthly return for a given date's calendar month."""
    ts = pd.to_datetime(date)
    return period_lookup.get((ts.year, ts.month), 0.0)
