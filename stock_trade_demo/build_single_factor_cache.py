#!/usr/bin/env python3
"""Offline builder for the single-factor backtest cache.

Per CLAUDE.md rule #12, the Flask web app is a read-only viewer. This script
runs all single-factor backtests once and persists the result to
`.cache/single_factor_results.pkl`, which `web_app.py` then loads at startup.

Usage:
    cd stock_trade_demo
    python3 build_single_factor_cache.py [--top-k N]

The default top_k=5 matches what the web frontend requests.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of stocks per period (must match the web frontend, default 5)')
    args = parser.parse_args()

    # Import lazily so that --help works without loading the entire web stack.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from web_app import (
        FACTOR_BACKTEST_CACHE_FILE,
        _run_single_factor_backtest,
        ensure_stock_data_loaded,
    )

    print(f'[build] 加载行情数据 ...')
    ensure_stock_data_loaded()

    print(f'[build] 开始运行单因子回测 (top_k={args.top_k}) ...')
    t0 = time.time()
    factor_results = _run_single_factor_backtest(top_k=args.top_k)
    elapsed = time.time() - t0
    print(f'[build] 回测完成，共 {len(factor_results)} 个因子，耗时 {elapsed:.1f}s')

    payload = {
        'version': 1,
        'top_k': args.top_k,
        'saved_at': time.time(),
        'factors': factor_results,
    }
    os.makedirs(os.path.dirname(FACTOR_BACKTEST_CACHE_FILE), exist_ok=True)
    with open(FACTOR_BACKTEST_CACHE_FILE, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_kb = os.path.getsize(FACTOR_BACKTEST_CACHE_FILE) / 1024
    print(f'[build] 写入 {FACTOR_BACKTEST_CACHE_FILE} ({size_kb:.1f} KB)')

    summary = ', '.join(
        f"{r['name']}={r['annual_return']}" for r in factor_results
    )
    print(f'[build] 年化摘要: {summary}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
