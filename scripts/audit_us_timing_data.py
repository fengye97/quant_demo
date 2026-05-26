#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STD_DIR = os.path.join(REPO_ROOT, 'stock_trade_demo')
sys.path.insert(0, STD_DIR)

from index_data import describe_timing_etf_cache  # noqa: E402

US_IDS = ['nasdaq', 'sp500']
FRED_KEYS = [
    'FedFundsRate',
    'YieldCurve_10Y2Y',
    'CPI_core',
    'Unemployment',
    'VIX',
    'HighYieldSpread',
    'Treasury10Y',
]


def _load_cache_df(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=['date'])


def _summarize_price_file(path: str) -> dict:
    if not os.path.exists(path):
        return {'exists': False, 'path': path}
    df = pd.read_csv(path)
    out = {
        'exists': True,
        'path': path,
        'rows': int(len(df)),
        'columns': list(df.columns),
    }
    if 'date' in df.columns and len(df) > 0:
        s = pd.to_datetime(df['date'], errors='coerce').dropna()
        if len(s) > 0:
            out['start_date'] = s.min().strftime('%Y-%m-%d')
            out['end_date'] = s.max().strftime('%Y-%m-%d')
    return out


def _compare_signal_vs_etf(signal_path: str, etf_path: str) -> dict:
    signal_df = _load_cache_df(signal_path)
    etf_df = _load_cache_df(etf_path)
    if signal_df is None or etf_df is None:
        return {
            'signal_exists': signal_df is not None,
            'etf_exists': etf_df is not None,
            'overlap_rows': 0,
        }

    merged = signal_df.merge(etf_df, on='date', suffixes=('_signal', '_etf'))
    out = {
        'signal_exists': True,
        'etf_exists': True,
        'signal_rows': int(len(signal_df)),
        'etf_rows': int(len(etf_df)),
        'overlap_rows': int(len(merged)),
    }
    if len(merged) > 0:
        for col in ('open', 'close'):
            ratio = (merged[f'{col}_etf'] / merged[f'{col}_signal']).dropna()
            if len(ratio) > 0:
                out[f'{col}_ratio_min'] = float(ratio.min())
                out[f'{col}_ratio_max'] = float(ratio.max())
                out[f'{col}_ratio_latest'] = float(ratio.iloc[-1])
    return out


def _summarize_fred(name: str) -> dict:
    path = os.path.join(REPO_ROOT, 'data', f'fred_{name}.csv')
    if not os.path.exists(path):
        return {'name': name, 'exists': False, 'path': path}
    df = pd.read_csv(path)
    dcol = df.columns[0]
    vcol = df.columns[1] if len(df.columns) > 1 else None
    s = pd.to_datetime(df[dcol], errors='coerce').dropna()
    return {
        'name': name,
        'exists': True,
        'path': path,
        'rows': int(len(df)),
        'start_date': s.min().strftime('%Y-%m-%d') if len(s) else None,
        'end_date': s.max().strftime('%Y-%m-%d') if len(s) else None,
        'nulls': int(df[vcol].isna().sum()) if vcol else 0,
        'value_col': vcol,
    }


def audit_one(index_id: str) -> None:
    info = describe_timing_etf_cache(index_id)
    signal_cache_path = os.path.join(STD_DIR, '.cache', f'{index_id}_daily.csv')
    legacy_sub_path = os.path.join(STD_DIR, '.cache', 'timing_etf', f'{index_id}_etf_daily.csv')
    qfq_path = os.path.join(STD_DIR, '.cache', 'timing_etf', f'{index_id}_etf_daily_qfq.csv')

    signal_summary = _summarize_price_file(signal_cache_path)
    compare_legacy = _compare_signal_vs_etf(signal_cache_path, legacy_sub_path)
    compare_qfq = _compare_signal_vs_etf(signal_cache_path, qfq_path)

    print(f'=== {index_id} ({info["code"]}, {info["symbol"]}) ===')
    print(f'default_adjust: {info["default_adjust"]}')
    print(f'preferred_runtime_path: {info["preferred_runtime_path"]}')
    print('signal_cache:')
    print(f'  {signal_summary}')
    print('timing_etf_candidates:')
    for item in info['candidates']:
        print(f'  {item}')
    print('signal_vs_legacy_timing_etf:')
    print(f'  {compare_legacy}')
    print('signal_vs_qfq_timing_etf:')
    print(f'  {compare_qfq}')
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', action='append', default=None)
    args = parser.parse_args()

    target_ids = args.only or US_IDS
    print('=== US timing ETF cache audit ===')
    for index_id in target_ids:
        audit_one(index_id)

    print('=== FRED inputs used by MacroV32 ===')
    for key in FRED_KEYS:
        print(_summarize_fred(key))

    print('\nNotes:')
    print('- signal_cache 指的是 build_us_index_panel() 当前使用的 .cache/{index}_daily.csv')
    print('- preferred_runtime_path 指的是 get_timing_etf_daily() 在不 force_refetch 时优先命中的 timing ETF 缓存')
    print('- 若 preferred_runtime_path 仍是 legacy 文件而不是 *_qfq.csv，则说明 qfq 主链尚未真正落盘生效')


if __name__ == '__main__':
    main()
