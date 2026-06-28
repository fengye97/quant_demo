#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STD_DIR = os.path.join(REPO_ROOT, 'stock_trade_demo')
sys.path.insert(0, STD_DIR)

from index_data import TIMING_ETF_CONFIGS, get_timing_etf_daily, describe_timing_etf_cache  # noqa: E402

US_IDS = ['nasdaq', 'sp500']


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', action='append', default=None)
    args = parser.parse_args()

    target_ids = args.only or US_IDS
    for index_id in target_ids:
        cfg = TIMING_ETF_CONFIGS[index_id]
        print(f'=== rebuild {index_id} ({cfg["code"]}, {cfg["symbol"]}) ===')
        df = get_timing_etf_daily(index_id=index_id, force_refetch=True, adjust='qfq')
        print(f'rows={len(df)} start={df["date"].min().strftime("%Y-%m-%d")} end={df["date"].max().strftime("%Y-%m-%d")}')
        summary = describe_timing_etf_cache(index_id=index_id, adjust='qfq')
        print(f'preferred_runtime_path={summary["preferred_runtime_path"]}')
        for item in summary['candidates']:
            print(item)
        print()


if __name__ == '__main__':
    main()
