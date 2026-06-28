#!/usr/bin/env python3
# 输出已归档至 .cache/archive/，重跑可重生成
import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from get_stock_info import fetch_daily_batch, compute_all_indicators  # noqa: E402
from backtest import select_and_backtest, strategy_evaluate  # noqa: E402
from index_data import get_index_returns  # noqa: E402
from strategies.original import OriginalStrategy  # noqa: E402

WEEKLY_DATA_PATH = os.path.join(ROOT, 'stock_data_weekly_experiment.parquet')
WEEKLY_DATA_FALLBACK_PATH = os.path.join(ROOT, 'stock_data_weekly_experiment.pkl')
MONTHLY_DATA_PATH = os.path.join(ROOT, 'stock_data.parquet')
MONTHLY_DATA_FALLBACK_PATH = os.path.join(ROOT, 'stock_data.csv')
START_DATE = '2021-06-01'
END_DATE = '2026-05-22'
DAILY_BAR_LIMIT = 1200
CACHE_PATH = os.path.join(ROOT, '.cache', 'weekly_daily_2021_06_2026_05.pkl')
OUTPUT_JSON = os.path.join(ROOT, '.cache', 'weekly_experiment_result.json')
DEFAULT_BATCH_SIZE = 200
DEFAULT_MAX_WORKERS = 20
CSV_HEADERS = [
    '交易日期', '股票代码', '股票名称', '是否交易',
    '开盘价', '最高价', '最低价', '收盘价', 'VWAP',
    '成交额', '流通市值', '总市值', '上市至今交易天数',
    '财报季度', '财报年份',
    '归母净利润', '归母净利润_ttm', '归母净利润_ttm同比',
    '归母净利润_单季', '归母净利润_单季同比', '归母净利润_单季环比',
    '经营活动产生的现金流量净额', '经营活动产生的现金流量净额_ttm',
    '经营活动产生的现金流量净额_ttm同比',
    '经营活动产生的现金流量净额_单季',
    '经营活动产生的现金流量净额_单季同比',
    '经营活动产生的现金流量净额_单季环比',
    '净资产',
    '涨跌幅_10', '涨跌幅_20',
    'bias_5', 'bias_10', 'bias_20',
    '振幅_5', '振幅_10', '振幅_20',
    '涨跌幅std_5', '涨跌幅std_10', '涨跌幅std_20',
    '成交额std_5', '成交额std_10', '成交额std_20',
    'K', 'D', 'J',
    'DIF', 'DEA', 'MACD',
    '市盈率倒数', '市净率倒数',
    '新版申万一级行业名称', '新版申万二级行业名称', '新版申万三级行业名称',
    '涨跌幅', '下周期每天涨跌幅',
]
ASOF_COLS = [
    '股票名称', '流通市值', '总市值', '上市至今交易天数',
    '财报季度', '财报年份',
    '归母净利润', '归母净利润_ttm', '归母净利润_ttm同比',
    '归母净利润_单季', '归母净利润_单季同比', '归母净利润_单季环比',
    '经营活动产生的现金流量净额', '经营活动产生的现金流量净额_ttm',
    '经营活动产生的现金流量净额_ttm同比', '经营活动产生的现金流量净额_单季',
    '经营活动产生的现金流量净额_单季同比', '经营活动产生的现金流量净额_单季环比',
    '净资产', '市盈率倒数', '市净率倒数',
    '新版申万一级行业名称', '新版申万二级行业名称', '新版申万三级行业名称',
]


def safe_float(v, default=0.0):
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return default
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default



def last_valid(values: List[float]) -> float:
    for v in reversed(values):
        try:
            if not math.isnan(v):
                return float(v)
        except TypeError:
            continue
    return float('nan')



def load_monthly_base() -> pd.DataFrame:
    if os.path.exists(MONTHLY_DATA_PATH):
        try:
            df = pd.read_parquet(MONTHLY_DATA_PATH)
            df['交易日期'] = pd.to_datetime(df['交易日期'])
            return df.sort_values(['股票代码', '交易日期']).reset_index(drop=True)
        except Exception as exc:
            print(f'[weekly] parquet load failed, falling back to CSV: {exc}')

    if not os.path.exists(MONTHLY_DATA_FALLBACK_PATH):
        raise FileNotFoundError(
            f'Neither {MONTHLY_DATA_PATH} nor {MONTHLY_DATA_FALLBACK_PATH} exists'
        )

    df = pd.read_csv(
        MONTHLY_DATA_FALLBACK_PATH,
        encoding='gbk',
        parse_dates=['交易日期'],
        low_memory=False,
    )
    return df.sort_values(['股票代码', '交易日期']).reset_index(drop=True)



def load_daily_cache(
    stock_codes: List[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_workers: int = DEFAULT_MAX_WORKERS,
    retry_failed: bool = False,
) -> Dict[str, Optional[List[Dict[str, object]]]]:
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    cache: Dict[str, Optional[List[Dict[str, object]]]] = {}
    if os.path.exists(CACHE_PATH):
        cache = pd.read_pickle(CACHE_PATH)

    if retry_failed:
        for code in stock_codes:
            if code in cache and not cache[code]:
                cache.pop(code, None)

    missing = [code for code in stock_codes if code not in cache]
    if not missing:
        print(f'[weekly] daily cache already covers {len(stock_codes)} requested stocks')
        return {code: cache.get(code) for code in stock_codes}

    total_batches = math.ceil(len(missing) / batch_size)
    for batch_idx, start in enumerate(range(0, len(missing), batch_size), start=1):
        batch = missing[start:start + batch_size]
        print(
            f'[weekly] fetching batch {batch_idx}/{total_batches} '
            f'({len(batch)} stocks, workers={max_workers})'
        )
        fetched = fetch_daily_batch(
            batch,
            datalen=DAILY_BAR_LIMIT,
            max_workers=max_workers,
            verbose=True,
        )
        success = 0
        for code in batch:
            data = fetched.get(code)
            cache[code] = data if data else None
            if data:
                success += 1
        pd.to_pickle(cache, CACHE_PATH)
        print(
            f'[weekly] batch {batch_idx}/{total_batches} saved cache: '
            f'{success}/{len(batch)} succeeded, {len(batch) - success} failed'
        )

    return {code: cache.get(code) for code in stock_codes}



def build_week_groups(df_daily: pd.DataFrame) -> List[pd.DataFrame]:
    week_end = df_daily['date'].dt.to_period('W-FRI').apply(lambda p: p.end_time.normalize())
    groups = []
    for _, grp in df_daily.assign(week_end=week_end).groupby('week_end', sort=True):
        grp = grp.sort_values('date').reset_index(drop=True)
        if len(grp) > 0:
            groups.append(grp)
    return groups



def compute_next_week_daily_returns(next_week: Optional[pd.DataFrame]) -> List[float]:
    if next_week is None or len(next_week) == 0:
        return []
    closes = next_week['close'].astype(float).tolist()
    prev_closes = [safe_float(next_week.iloc[0]['pre_close'])] + closes[:-1]
    rets = []
    for prev_close, close in zip(prev_closes, closes):
        if prev_close and prev_close != 0:
            rets.append(float(close / prev_close - 1.0))
    return rets



def build_weekly_rows_for_stock(
    code: str,
    daily_rows: Optional[List[Dict[str, object]]],
    monthly_stock: pd.DataFrame,
) -> List[dict]:
    if not daily_rows:
        return []
    df_daily = pd.DataFrame(daily_rows)
    if len(df_daily) == 0:
        return []

    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily = df_daily[
        (df_daily['date'] >= pd.Timestamp(START_DATE)) &
        (df_daily['date'] <= pd.Timestamp(END_DATE))
    ].copy()
    if len(df_daily) < 80:
        return []

    df_daily = df_daily.sort_values('date').reset_index(drop=True)
    df_daily['pre_close'] = df_daily['close'].shift(1)
    date_to_idx = {ts.normalize(): i for i, ts in enumerate(df_daily['date'])}

    week_groups = build_week_groups(df_daily)
    if len(week_groups) < 3:
        return []

    full_dates = df_daily['date'].dt.strftime('%Y-%m-%d').tolist()
    full_opens = df_daily['open'].astype(float).tolist()
    full_highs = df_daily['high'].astype(float).tolist()
    full_lows = df_daily['low'].astype(float).tolist()
    full_closes = df_daily['close'].astype(float).tolist()
    full_volumes = df_daily['volume'].astype(float).tolist()
    indicators = compute_all_indicators(full_dates, full_opens, full_highs, full_lows, full_closes, full_volumes)

    df_ind = pd.DataFrame({'date': df_daily['date']})
    for key, vals in indicators.items():
        df_ind[key] = vals

    monthly_stock = monthly_stock.sort_values('交易日期').reset_index(drop=True)
    rows = []
    for idx, grp in enumerate(week_groups):
        week_end_date = pd.Timestamp(grp.iloc[-1]['date']).normalize()
        asof_src = monthly_stock[monthly_stock['交易日期'] <= week_end_date]
        if len(asof_src) == 0:
            continue
        asof_row = asof_src.iloc[-1]

        ind_row = df_ind[df_ind['date'] == week_end_date]
        if len(ind_row) == 0:
            continue
        ind_row = ind_row.iloc[0]

        prev_week = week_groups[idx - 1] if idx > 0 else None
        prev_week_close = (
            safe_float(prev_week.iloc[-1]['close'])
            if prev_week is not None and len(prev_week) > 0
            else safe_float(grp.iloc[0]['pre_close'])
        )
        week_close = safe_float(grp.iloc[-1]['close'])
        week_return = (week_close / prev_week_close - 1.0) if prev_week_close else 0.0

        estimated_amount = 0.0
        for _, r in grp.iterrows():
            typical_price = (safe_float(r['high']) + safe_float(r['low']) + safe_float(r['close'])) / 3.0
            estimated_amount += safe_float(r['volume']) * typical_price
        weekly_volume = float(grp['volume'].astype(float).sum())
        vwap = estimated_amount / weekly_volume if weekly_volume > 0 else week_close

        next_week = week_groups[idx + 1] if idx + 1 < len(week_groups) else None
        next_daily_returns = compute_next_week_daily_returns(next_week)

        day_idx = date_to_idx.get(week_end_date)
        if day_idx is None:
            continue

        row = {col: '' for col in CSV_HEADERS}
        row['交易日期'] = week_end_date
        row['股票代码'] = code
        row['股票名称'] = str(asof_row['股票名称'])
        row['是否交易'] = 1
        row['开盘价'] = safe_float(grp.iloc[0]['open'])
        row['最高价'] = float(grp['high'].astype(float).max())
        row['最低价'] = float(grp['low'].astype(float).min())
        row['收盘价'] = week_close
        row['VWAP'] = vwap
        row['成交额'] = estimated_amount
        for col in ASOF_COLS:
            row[col] = asof_row[col]
        row['涨跌幅_10'] = last_valid(indicators['涨跌幅_10'][:day_idx + 1])
        row['涨跌幅_20'] = last_valid(indicators['涨跌幅_20'][:day_idx + 1])
        for col in [
            'bias_5', 'bias_10', 'bias_20',
            '振幅_5', '振幅_10', '振幅_20',
            '涨跌幅std_5', '涨跌幅std_10', '涨跌幅std_20',
            '成交额std_5', '成交额std_10', '成交额std_20',
            'K', 'D', 'J', 'DIF', 'DEA', 'MACD',
        ]:
            row[col] = ind_row[col]
        row['涨跌幅'] = week_return
        row['下周期每天涨跌幅'] = json.dumps(next_daily_returns, ensure_ascii=False)
        rows.append(row)
    return rows



def save_weekly_dataset(df: pd.DataFrame) -> str:
    try:
        df.to_parquet(WEEKLY_DATA_PATH, index=False)
        print(f'[weekly] saved dataset to {WEEKLY_DATA_PATH} rows={len(df)}')
        return WEEKLY_DATA_PATH
    except Exception as exc:
        print(f'[weekly] parquet save failed, falling back to pickle: {exc}')
        df.to_pickle(WEEKLY_DATA_FALLBACK_PATH)
        print(f'[weekly] saved dataset to {WEEKLY_DATA_FALLBACK_PATH} rows={len(df)}')
        return WEEKLY_DATA_FALLBACK_PATH



def _format_eta(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return 'unknown'
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f'{h}h {m}m {s}s'
    if m > 0:
        return f'{m}m {s}s'
    return f'{s}s'



def build_weekly_dataset(
    max_stocks: Optional[int] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_workers: int = DEFAULT_MAX_WORKERS,
    retry_failed: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, object], List[str]]:
    monthly = load_monthly_base()
    monthly = monthly[monthly['交易日期'] <= pd.Timestamp(END_DATE)].copy()
    monthly['股票代码'] = monthly['股票代码'].astype(str).str.strip()
    eligible = monthly[monthly['交易日期'] >= pd.Timestamp(START_DATE)].copy()

    stock_codes = sorted([
        code for code in eligible['股票代码'].unique().tolist()
        if 'bj' not in str(code)
    ])
    if max_stocks is not None:
        stock_codes = stock_codes[:max_stocks]
    print(f'[weekly] requested stock universe size: {len(stock_codes)}')

    daily_map = load_daily_cache(
        stock_codes,
        batch_size=batch_size,
        max_workers=max_workers,
        retry_failed=retry_failed,
    )

    all_rows = []
    built_codes = []
    skipped_no_daily = 0
    skipped_no_weekly_rows = 0
    started_at = time.time()
    progress_every = 100
    total_codes = len(stock_codes)
    for i, code in enumerate(stock_codes, start=1):
        monthly_stock = monthly[monthly['股票代码'] == code].copy()
        daily_rows = daily_map.get(code)
        if not daily_rows:
            skipped_no_daily += 1
        else:
            rows = build_weekly_rows_for_stock(code, daily_rows, monthly_stock)
            if not rows:
                skipped_no_weekly_rows += 1
            else:
                built_codes.append(code)
                all_rows.extend(rows)

        if i % progress_every == 0 or i == total_codes:
            elapsed = time.time() - started_at
            rate = i / elapsed if elapsed > 0 else 0.0
            remaining = total_codes - i
            eta = remaining / rate if rate > 0 else float('inf')
            pct = i / total_codes * 100 if total_codes > 0 else 100.0
            print(
                f'[weekly][progress] {i}/{total_codes} ({pct:.1f}%) '
                f'elapsed={_format_eta(elapsed)} eta={_format_eta(eta)} '
                f'built={len(built_codes)} no_daily={skipped_no_daily} no_rows={skipped_no_weekly_rows} '
                f'rows={len(all_rows)}'
            )

    weekly = pd.DataFrame(all_rows)
    if len(weekly) == 0:
        raise RuntimeError('No weekly rows built')

    weekly['股票代码'] = weekly['股票代码'].astype(str).str.strip()
    weekly = weekly.sort_values(['交易日期', '股票代码']).reset_index(drop=True)
    saved_path = save_weekly_dataset(weekly)
    stats = {
        'requested_stock_codes': len(stock_codes),
        'fetchable_daily_codes': int(sum(1 for code in stock_codes if daily_map.get(code))),
        'weekly_stock_codes': len(built_codes),
        'weekly_rows': int(len(weekly)),
        'skipped_no_daily': int(skipped_no_daily),
        'skipped_no_weekly_rows': int(skipped_no_weekly_rows),
        'saved_dataset_path': saved_path,
    }
    return weekly, stats, built_codes



def build_market_state(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df = df.copy()
    mkt_ret = pd.to_numeric(df['涨跌幅'], errors='coerce').fillna(0.0).groupby(df['交易日期']).mean()
    mkt_cum = (1 + mkt_ret).cumprod()
    mkt_ma = mkt_cum.rolling(window).mean()
    state = (mkt_cum > mkt_ma).map({True: 'bull', False: 'bear'})
    df['市场状态'] = pd.to_datetime(df['交易日期']).map(state)
    return df



def evaluate_original(df: pd.DataFrame, weekly: bool) -> Dict[str, object]:
    strategy = OriginalStrategy(val_history_periods=52 if weekly else 12)
    df = build_market_state(df, 52 if weekly else 12)
    df = strategy.run(df)
    index_returns = get_index_returns('csi1000', frequency='weekly' if weekly else 'monthly')
    result = select_and_backtest(df, strategy)
    ev = strategy_evaluate(result, index_returns=index_returns)
    label = 'weekly_original' if weekly else 'monthly_original'
    metrics = ev.iloc[:, 0].to_dict()
    tp_hits = 0
    total_periods = len(result)
    for payload in result['买入个股收益']:
        try:
            items = json.loads(payload)
        except Exception:
            items = []
        for item in items:
            sell_price = item.get('sell_price')
            buy_price = item.get('buy_price')
            if sell_price is not None and buy_price is not None and buy_price > 0 and sell_price / buy_price - 1 > 0.20:
                tp_hits += 1
    return {
        'label': label,
        'periods': total_periods,
        'input_stock_codes': int(df['股票代码'].astype(str).str.strip().nunique()),
        'input_trade_dates': int(pd.to_datetime(df['交易日期']).nunique()),
        'metrics': metrics,
        'tp_hits_proxy': tp_hits,
    }



def parse_args():
    parser = argparse.ArgumentParser(description='Run the true weekly vs monthly experiment on the largest practical fetchable universe.')
    parser.add_argument('--max-stocks', type=int, default=None, help='Optional cap on the number of stock codes to process.')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='Daily fetch batch size for cache persistence.')
    parser.add_argument('--max-workers', type=int, default=DEFAULT_MAX_WORKERS, help='Concurrent daily fetch workers.')
    parser.add_argument('--retry-failed', action='store_true', help='Retry stock codes previously cached as failed/empty.')
    return parser.parse_args()



def main():
    args = parse_args()

    weekly_df, universe_stats, built_codes = build_weekly_dataset(
        max_stocks=args.max_stocks,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        retry_failed=args.retry_failed,
    )

    if not built_codes:
        raise RuntimeError('No fetchable weekly universe remained after dataset construction')

    monthly_df = load_monthly_base()
    monthly_df['交易日期'] = pd.to_datetime(monthly_df['交易日期'])
    monthly_df['股票代码'] = monthly_df['股票代码'].astype(str).str.strip()
    monthly_df = monthly_df[
        (monthly_df['交易日期'] >= pd.Timestamp(START_DATE)) &
        (monthly_df['交易日期'] <= pd.Timestamp(END_DATE)) &
        (monthly_df['股票代码'].isin(set(built_codes)))
    ].copy()

    monthly_result = evaluate_original(monthly_df, weekly=False)
    weekly_result = evaluate_original(weekly_df, weekly=True)
    summary = {
        'start_date': START_DATE,
        'end_date': END_DATE,
        'note': 'Weekly experiment is limited by Sina daily history depth (~1200 bars), so coverage starts around 2021-06.',
        'comparison_on_same_fetchable_universe': True,
        'universe': universe_stats,
        'monthly_original': monthly_result,
        'weekly_original': weekly_result,
    }
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == '__main__':
    main()
