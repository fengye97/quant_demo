"""
Benchmark index data fetcher.

Fetches daily K-line data from Sina Finance API, computes monthly returns,
and caches them under .cache/ to avoid re-fetching on every run.
"""

import os
import json
import logging
import urllib.request
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')
TIMING_ETF_CACHE_DIR = os.path.join(CACHE_DIR, 'timing_etf')

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
    'nasdaq': {
        'name': '纳指ETF',
        'symbol': 'sz159941',
        'cache_file': 'nasdaq_monthly.csv',
        'daily_cache_file': 'nasdaq_daily.csv',
        'series_name': 'nasdaq_return',
    },
    'sp500': {
        'name': '标普500ETF',
        'symbol': 'sh513500',
        'cache_file': 'sp500_monthly.csv',
        'daily_cache_file': 'sp500_daily.csv',
        'series_name': 'sp500_return',
    },
}

# A股指数（不含美股ETF代理，仅用于A股benchmark对比）
A_SHARE_INDEX_IDS = {'csi1000', 'chinext', 'star50'}

TIMING_ETF_CONFIGS = {
    'csi1000': {
        'name': '中证1000ETF',
        'code': '510980',
        'symbol': 'sh510980',
        'daily_cache_file': 'csi1000_etf_daily.csv',
        # T+1 交收：A股 ETF 当日买入次日才能卖出（信号 t -> 成交 t+1 open -> 标记 t+1 close）。
        'settlement': 'T+1',
        # 涨跌停幅度（沪市中证1000ETF 为 10%）。
        'limit_pct': 0.10,
        'market': 'SH',
        # 复权方式：使用 akshare 的前复权（qfq）拿日线，避免分红日被误判为跳水。
        'dividend_adjust_method': 'qfq',
    },
    'chinext': {
        'name': '创业板ETF',
        'code': '159205',
        'symbol': 'sz159205',
        'daily_cache_file': 'chinext_etf_daily.csv',
        'settlement': 'T+1',
        # 创业板 ETF 跟踪创业板成份，单日涨跌停 20%。
        'limit_pct': 0.20,
        'market': 'SZ',
        'dividend_adjust_method': 'qfq',
    },
    'star50': {
        'name': '科创50ETF',
        'code': '589850',
        'symbol': 'sh589850',
        'daily_cache_file': 'star50_etf_daily.csv',
        'settlement': 'T+1',
        # 科创板 ETF 单日涨跌停 20%。
        'limit_pct': 0.20,
        'market': 'SH',
        'dividend_adjust_method': 'qfq',
    },
    'nasdaq': {
        'name': '纳指ETF',
        'code': '159941',
        'symbol': 'sz159941',
        'daily_cache_file': 'nasdaq_etf_daily.csv',
        # 跨境 ETF（QDII）按 T+0 回转交易；无 A 股式涨跌停限制（盘中可能因溢价熔断停牌但非固定百分比）。
        'settlement': 'T+0',
        'limit_pct': None,
        'market': 'SZ',
        'dividend_adjust_method': 'qfq',
    },
    'sp500': {
        'name': '标普500ETF',
        'code': '513500',
        'symbol': 'sh513500',
        'daily_cache_file': 'sp500_etf_daily.csv',
        'settlement': 'T+0',
        'limit_pct': None,
        'market': 'SH',
        'dividend_adjust_method': 'qfq',
    },
}


def _cache_path(index_id):
    return os.path.join(CACHE_DIR, INDEX_CONFIGS[index_id]['cache_file'])


def _daily_cache_path(index_id):
    return os.path.join(CACHE_DIR, INDEX_CONFIGS[index_id]['daily_cache_file'])


def _ensure_cache_dir(path):
    os.makedirs(path, exist_ok=True)


def _etf_daily_cache_path(index_id, adjust='qfq'):
    """ETF daily cache path. Cache key 必须包含 adjust，避免与未复权旧缓存撞键。"""
    base = TIMING_ETF_CONFIGS[index_id]['daily_cache_file']
    if adjust:
        root, ext = os.path.splitext(base)
        base = f'{root}_{adjust}{ext}'
    return os.path.join(TIMING_ETF_CACHE_DIR, base)


def _legacy_etf_daily_cache_path(index_id):
    """旧的未复权缓存路径（Sina 抓取产物），仅用于 fallback。"""
    return os.path.join(CACHE_DIR, TIMING_ETF_CONFIGS[index_id]['daily_cache_file'])


def _legacy_etf_daily_cache_path_in_subdir(index_id):
    """位于 timing_etf 子目录的旧未复权缓存（升级前的中间态）。"""
    return os.path.join(TIMING_ETF_CACHE_DIR, TIMING_ETF_CONFIGS[index_id]['daily_cache_file'])


def _sina_url(symbol):
    return (
        'https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/'
        f'CN_MarketData.getKLineData?symbol={symbol}&scale=240&datalen=8000'
    )


def _fetch_daily_kline(symbol):
    """Fetch daily K-line data from Sina API (未复权, 用于指数 / fallback)."""
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


def _fetch_etf_daily_akshare(code, adjust='qfq'):
    """通过 akshare 抓取 ETF 日线（默认前复权 qfq）。

    akshare.fund_etf_hist_em 返回中文列：'日期','开盘','收盘','最高','最低',
    '成交量','成交额','振幅','涨跌幅','涨跌额','换手率'。这里重命名为现有
    调用方期望的英文 schema，保持向后兼容。

    严格遵守 CLAUDE.md 第 11 条：不做 forward/backfill，不补 ETF 上市前数据；
    akshare 返回什么就用什么。
    """
    import akshare as ak  # 延迟导入，避免无 akshare 环境下加载本模块即失败

    raw = ak.fund_etf_hist_em(symbol=code, period='daily', adjust=adjust)
    if raw is None or len(raw) == 0:
        raise RuntimeError(f'akshare returned empty frame for ETF {code} (adjust={adjust})')

    rename_map = {
        '日期': 'date',
        '开盘': 'open',
        '收盘': 'close',
        '最高': 'high',
        '最低': 'low',
        '成交量': 'volume',
        '成交额': 'amount',
    }
    keep = [c for c in rename_map if c in raw.columns]
    df = raw[keep].rename(columns=rename_map).copy()

    required = {'date', 'open', 'high', 'low', 'close', 'volume'}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f'akshare ETF frame missing columns {missing} for {code}')

    df['date'] = pd.to_datetime(df['date'])
    for col in ('open', 'high', 'low', 'close', 'volume'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'amount' in df.columns:
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

    df = df.dropna(subset=['date', 'open', 'high', 'low', 'close']).sort_values('date').reset_index(drop=True)
    return df


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


def get_timing_etf_daily(index_id='csi1000', force_refetch=False, adjust=None):
    """Get cached daily K-line data for the user-specified ETF mapped to a timing index.

    主路径：akshare.fund_etf_hist_em(adjust='qfq')，前复权，分红日不会被误判为下跌。
    Fallback：原 Sina 未复权路径（仅在 akshare 抓取失败时使用，并在 logger 中 warn）。
    缓存 key 包含 adjust，避免与旧未复权缓存撞键。
    """
    if index_id not in TIMING_ETF_CONFIGS:
        raise ValueError(f'Unknown ETF timing index_id: {index_id}')

    cfg = TIMING_ETF_CONFIGS[index_id]
    if adjust is None:
        adjust = cfg.get('dividend_adjust_method', 'qfq')

    cache_file = _etf_daily_cache_path(index_id, adjust=adjust)
    legacy_cache_file = _legacy_etf_daily_cache_path(index_id)
    legacy_sub_cache_file = _legacy_etf_daily_cache_path_in_subdir(index_id)

    if not force_refetch and os.path.exists(cache_file):
        df = pd.read_csv(cache_file, parse_dates=['date'])
        return df.sort_values('date').reset_index(drop=True)

    _ensure_cache_dir(CACHE_DIR)
    _ensure_cache_dir(TIMING_ETF_CACHE_DIR)

    # 主路径：akshare 前复权
    try:
        print(f"[index_data] Fetching {cfg['name']} ({cfg['code']}) daily K-line via akshare (adjust={adjust})...")
        df_daily = _fetch_etf_daily_akshare(cfg['code'], adjust=adjust)
        df_daily.to_csv(cache_file, index=False)
        print(f"[index_data] Cached ETF daily K-line to {cache_file} ({len(df_daily)} rows)")
        return df_daily
    except Exception as ak_err:
        logger.warning(
            "[index_data] akshare fetch failed for %s (%s, adjust=%s): %s; falling back to Sina (un-adjusted).",
            cfg['name'], cfg['code'], adjust, ak_err,
        )
        print(f"[index_data] WARN akshare failed for {cfg['name']} ({cfg['code']}): {ak_err}; "
              f"falling back to Sina un-adjusted feed")

    # Fallback 1：Sina 未复权抓取（注意：未复权，分红日会有跳水，仅 best-effort）
    try:
        print(f"[index_data] Fetching {cfg['name']} ({cfg['code']}) daily K-line from Sina (fallback)...")
        df_daily = _fetch_daily_kline(cfg['symbol'])
        # 不写入 qfq 的 cache_file，避免污染前复权缓存；写到未复权 legacy path 以便复用
        df_daily.to_csv(legacy_sub_cache_file, index=False)
        print(f"[index_data] Cached Sina un-adjusted ETF daily K-line to {legacy_sub_cache_file}")
        return df_daily
    except Exception as sina_err:
        logger.warning(
            "[index_data] Sina fallback also failed for %s: %s",
            cfg['name'], sina_err,
        )
        print(f"[index_data] ERROR Sina fallback failed for {cfg['name']}: {sina_err}")

    # Fallback 2：磁盘上任何已有缓存
    for candidate in (cache_file, legacy_sub_cache_file, legacy_cache_file):
        if os.path.exists(candidate):
            print(f"[index_data] Falling back to existing cached ETF daily K-line: {candidate}")
            df = pd.read_csv(candidate, parse_dates=['date'])
            return df.sort_values('date').reset_index(drop=True)

    raise RuntimeError(
        f"Failed to fetch {cfg['name']} ETF daily data via akshare and Sina, and no cache exists."
    )


def refresh_all_timing_etf_daily():
    refreshed = {}
    for index_id in TIMING_ETF_CONFIGS:
        refreshed[index_id] = get_timing_etf_daily(index_id=index_id, force_refetch=True)
    return refreshed


def build_index_panel(index_ids=None, force_refetch=False):
    """Build a daily wide panel for A-share indexes (excludes US ETF proxies by default)."""
    index_ids = index_ids or list(A_SHARE_INDEX_IDS)
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


def _compute_weekly_returns(df_daily, series_name):
    """Compute weekly returns from daily close prices using each week's last trading day."""
    df = df_daily.set_index('date').sort_index()
    weekly_close = df['close'].resample('W-FRI').last().dropna()
    weekly = weekly_close.pct_change().dropna()
    weekly.name = series_name
    return weekly


def get_index_returns(index_id='csi1000', force_refetch=False, frequency='monthly'):
    """Get benchmark returns for a configured index."""
    if index_id not in INDEX_CONFIGS:
        raise ValueError(f'Unknown index_id: {index_id}')
    if frequency not in {'monthly', 'weekly'}:
        raise ValueError(f'Unsupported frequency: {frequency}')

    cfg = INDEX_CONFIGS[index_id]
    cache_file = _cache_path(index_id)
    if frequency == 'weekly':
        cache_file = cache_file.replace('_monthly.csv', '_weekly.csv')

    if not force_refetch and os.path.exists(cache_file):
        s = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s.name = cfg['series_name']
        print(f"[index_data] Loaded {cfg['name']} {len(s)} {frequency} returns from cache: {cache_file}")
        return s

    os.makedirs(CACHE_DIR, exist_ok=True)

    try:
        df_daily = get_index_daily(index_id=index_id, force_refetch=force_refetch)
        print(f"[index_data] Fetched {len(df_daily)} daily bars "
              f"({df_daily['date'].min().strftime('%Y-%m-%d')} to "
              f"{df_daily['date'].max().strftime('%Y-%m-%d')})")

        returns = _compute_monthly_returns(df_daily, cfg['series_name']) if frequency == 'monthly' else _compute_weekly_returns(df_daily, cfg['series_name'])
        print(f"[index_data] Computed {len(returns)} {frequency} returns "
              f"({returns.index.min().strftime('%Y-%m-%d')} to "
              f"{returns.index.max().strftime('%Y-%m-%d')})")

        returns.to_csv(cache_file, header=True)
        print(f"[index_data] Cached to {cache_file}")
        return returns

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


US_INDEX_IDS = {'nasdaq', 'sp500'}


def build_us_index_panel(force_refetch=False):
    """Build a daily wide panel for US ETF proxies (nasdaq, sp500)."""
    return build_index_panel(index_ids=list(US_INDEX_IDS), force_refetch=force_refetch)


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
