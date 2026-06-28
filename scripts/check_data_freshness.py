#!/usr/bin/env python3
"""数据新鲜度 & 跨源对齐校验脚本（Protection 4 from plan_protections.md）。

把 CLAUDE.md 「指数 / ETF 日线同步保护」里的几条不可见隐式约束
固化成显式断言，便于在 `POST /api/update_index_data` 之外离线/CI 跑：

  1. A股指数日线 max_date 与对应 ETF 日线 max_date 必须一致（freshness guard 双向）
  2. 美股指数日线 max_date 与对应 ETF 日线 max_date 必须一致
  3. stock_data 月度面板的 max(YYYY-MM) 与 csi1000_monthly.csv 的 max(YYYY-MM) 必须一致
  4. ETF 日期集合应为 A 股交易日历的子集（A 股 ETF；美股 ETF 不做这个约束）
  5. A 股 ETF 起点不得早于 A 股指数日线起点（避免 ETF 抢跑指数日线缓存）

Exit code:
  0  全部通过
  1  至少一条失败（stderr 输出明细）

设计原则：脚本只读，不修改任何 cache；用 print() + exit code 表达结果，
方便接到 CI 或 `data-refresh-all` skill 的末尾。
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import pandas as pd

# stock_trade_demo 在 repo 根目录下
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_DIR = os.path.join(_REPO_ROOT, 'stock_trade_demo')
sys.path.insert(0, _PROJECT_DIR)

from index_data import (  # noqa: E402
    A_SHARE_CALENDAR_CACHE_FILE,
    A_SHARE_INDEX_IDS,
    CACHE_DIR,
    INDEX_CONFIGS,
    TIMING_ETF_CACHE_DIR,
    TIMING_ETF_CONFIGS,
    US_INDEX_IDS,
)
from utils.atomic_io import read_meta  # noqa: E402


def _written_at(path: str) -> Tuple[Optional[float], str]:
    """优先读 ``<path>.meta.json``，缺失时回退到 mtime。

    返回 ``(unix_ts, source)``，``source ∈ {'meta', 'mtime', 'missing'}``。
    """
    meta = read_meta(path)
    if meta and isinstance(meta.get('written_at_unix'), (int, float)):
        return float(meta['written_at_unix']), 'meta'
    if os.path.exists(path):
        try:
            return float(os.path.getmtime(path)), 'mtime'
        except OSError:
            pass
    return None, 'missing'


def _produced_by(path: str) -> Optional[str]:
    meta = read_meta(path)
    if meta:
        return meta.get('produced_by')
    return None

STOCK_DATA_CSV = os.path.join(_PROJECT_DIR, 'stock_data.csv')
STOCK_DATA_PARQUET = os.path.join(_PROJECT_DIR, 'stock_data.parquet')
CSI1000_MONTHLY = os.path.join(CACHE_DIR, INDEX_CONFIGS['csi1000']['cache_file'])


# ── helpers ──────────────────────────────────────────────────────────
def _read_dates(path: str, column: str = 'date') -> Optional[pd.DatetimeIndex]:
    """读取一份 CSV 的日期列；文件缺失返回 None（由 caller 决定是否算失败）。"""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, usecols=[column])
    return pd.to_datetime(df[column], errors='coerce').dropna().sort_values()


def _max_date(path: str, column: str = 'date') -> Optional[pd.Timestamp]:
    idx = _read_dates(path, column=column)
    if idx is None or len(idx) == 0:
        return None
    return idx.iloc[-1] if isinstance(idx, pd.Series) else idx[-1]


def _min_date(path: str, column: str = 'date') -> Optional[pd.Timestamp]:
    idx = _read_dates(path, column=column)
    if idx is None or len(idx) == 0:
        return None
    return idx.iloc[0] if isinstance(idx, pd.Series) else idx[0]


def _stock_data_max_month() -> Optional[pd.Period]:
    """读取 stock_data 月度面板里最新的 YYYY-MM。优先 parquet，回退 GBK CSV。"""
    if os.path.exists(STOCK_DATA_PARQUET):
        df = pd.read_parquet(STOCK_DATA_PARQUET, columns=['交易日期'])
    elif os.path.exists(STOCK_DATA_CSV):
        df = pd.read_csv(STOCK_DATA_CSV, usecols=['交易日期'], encoding='gbk')
    else:
        return None
    dates = pd.to_datetime(df['交易日期'], errors='coerce').dropna()
    if dates.empty:
        return None
    return dates.max().to_period('M')


def _csi1000_monthly_max_month() -> Optional[pd.Period]:
    ts = _max_date(CSI1000_MONTHLY)
    return ts.to_period('M') if ts is not None else None


# ── checks ───────────────────────────────────────────────────────────
def _check_a_share_index_etf_alignment() -> List[Tuple[str, bool, str]]:
    out: List[Tuple[str, bool, str]] = []
    for idx_id in sorted(A_SHARE_INDEX_IDS):
        idx_daily = os.path.join(CACHE_DIR, INDEX_CONFIGS[idx_id]['daily_cache_file'])
        etf_daily = os.path.join(TIMING_ETF_CACHE_DIR, TIMING_ETF_CONFIGS[idx_id]['daily_cache_file'])
        idx_max = _max_date(idx_daily)
        etf_max = _max_date(etf_daily)
        if idx_max is None or etf_max is None:
            out.append((
                f'a_share/{idx_id}: index/etf daily 文件缺失',
                False,
                f'index={idx_max} ({idx_daily}) | etf={etf_max} ({etf_daily})',
            ))
            continue
        ok = idx_max == etf_max
        msg = f'index_max={idx_max.date()} etf_max={etf_max.date()}'
        if not ok:
            msg += '  ←  必须一致（CLAUDE.md 指数/ETF 同步保护）'
        out.append((f'a_share/{idx_id}: index/etf daily max_date 对齐', ok, msg))
    return out


def _check_us_index_etf_alignment() -> List[Tuple[str, bool, str]]:
    out: List[Tuple[str, bool, str]] = []
    for idx_id in sorted(US_INDEX_IDS):
        idx_daily = os.path.join(CACHE_DIR, INDEX_CONFIGS[idx_id]['daily_cache_file'])
        etf_daily = os.path.join(TIMING_ETF_CACHE_DIR, TIMING_ETF_CONFIGS[idx_id]['daily_cache_file'])
        idx_max = _max_date(idx_daily)
        etf_max = _max_date(etf_daily)
        if idx_max is None or etf_max is None:
            out.append((
                f'us/{idx_id}: index/etf daily 文件缺失',
                False,
                f'index={idx_max} ({idx_daily}) | etf={etf_max} ({etf_daily})',
            ))
            continue
        ok = idx_max == etf_max
        msg = f'index_max={idx_max.date()} etf_max={etf_max.date()}'
        if not ok:
            msg += '  ←  美股指数/ETF 应同步至同一交易日'
        out.append((f'us/{idx_id}: index/etf daily max_date 对齐', ok, msg))
    return out


def _check_stock_data_vs_csi1000_monthly() -> List[Tuple[str, bool, str]]:
    sd = _stock_data_max_month()
    cs = _csi1000_monthly_max_month()
    if sd is None or cs is None:
        return [(
            'stock_data vs csi1000_monthly: 月度面板对齐',
            False,
            f'stock_data={sd} csi1000_monthly={cs} （文件缺失）',
        )]
    ok = sd == cs
    msg = f'stock_data_max_month={sd} csi1000_monthly_max_month={cs}'
    if not ok:
        msg += '  ←  月度面板与基准月度收益的最大月份应一致（否则 benchmark 对齐会错位）'
    return [('stock_data vs csi1000_monthly: 月度面板对齐', ok, msg)]


def _check_a_share_etf_subset_of_calendar() -> List[Tuple[str, bool, str]]:
    """A 股 ETF 日期集合应是 A 股交易日历的子集。"""
    out: List[Tuple[str, bool, str]] = []
    cal = _read_dates(A_SHARE_CALENDAR_CACHE_FILE)
    if cal is None:
        return [(
            'a_share_calendar: 文件缺失',
            False,
            f'缺失 {A_SHARE_CALENDAR_CACHE_FILE}',
        )]
    cal_set = set(cal.tolist() if isinstance(cal, pd.Series) else list(cal))
    for idx_id in sorted(A_SHARE_INDEX_IDS):
        etf_daily = os.path.join(TIMING_ETF_CACHE_DIR, TIMING_ETF_CONFIGS[idx_id]['daily_cache_file'])
        etf_dates = _read_dates(etf_daily)
        if etf_dates is None:
            out.append((f'a_share/{idx_id}: ETF 文件缺失', False, f'缺失 {etf_daily}'))
            continue
        etf_set = set(etf_dates.tolist() if isinstance(etf_dates, pd.Series) else list(etf_dates))
        extras = sorted(etf_set - cal_set)
        ok = not extras
        msg = f'{len(etf_set)} 条 ETF 日期'
        if not ok:
            sample = [d.date().isoformat() for d in extras[:5]]
            msg = f'{len(extras)} 条 ETF 日期不在 A 股交易日历内，前几个: {sample}'
        out.append((f'a_share/{idx_id}: ETF 日期 ⊂ A 股交易日历', ok, msg))
    return out


def _check_a_share_etf_starts_after_index() -> List[Tuple[str, bool, str]]:
    """ETF 起始日 >= 指数日线起始日（防止 ETF 抢跑指数缓存）。"""
    out: List[Tuple[str, bool, str]] = []
    for idx_id in sorted(A_SHARE_INDEX_IDS):
        idx_daily = os.path.join(CACHE_DIR, INDEX_CONFIGS[idx_id]['daily_cache_file'])
        etf_daily = os.path.join(TIMING_ETF_CACHE_DIR, TIMING_ETF_CONFIGS[idx_id]['daily_cache_file'])
        idx_min = _min_date(idx_daily)
        etf_min = _min_date(etf_daily)
        if idx_min is None or etf_min is None:
            out.append((f'a_share/{idx_id}: index/etf 起点检查', False, f'index_min={idx_min} etf_min={etf_min}'))
            continue
        ok = etf_min >= idx_min
        msg = f'index_min={idx_min.date()} etf_min={etf_min.date()}'
        if not ok:
            msg += '  ←  ETF 起点早于指数起点；可能指数日线缓存被截断'
        out.append((f'a_share/{idx_id}: ETF 起点不早于指数起点', ok, msg))
    return out


# ── lineage（informational：优先 meta.json，回退 mtime，永远不致 fail） ─
def _lineage_lines() -> List[Tuple[str, bool, str]]:
    """汇总核心数据文件的 ``produced_by`` 与 ``written_at``。

    仅作展示，不影响 exit code：meta.json 是 Pillar 2 Step 6 引入的血缘
    sidecar，旧文件可能没有 sidecar——这里降级到 mtime 显示，方便人工
    判断"这个 cache 是谁写的、多久前写的"。
    """
    out: List[Tuple[str, bool, str]] = []
    paths_of_interest: List[Tuple[str, str]] = []
    for idx_id in sorted(A_SHARE_INDEX_IDS | US_INDEX_IDS):
        paths_of_interest.append((
            f'index/{idx_id}/daily',
            os.path.join(CACHE_DIR, INDEX_CONFIGS[idx_id]['daily_cache_file']),
        ))
    for idx_id in sorted(A_SHARE_INDEX_IDS | US_INDEX_IDS):
        if idx_id in TIMING_ETF_CONFIGS:
            paths_of_interest.append((
                f'etf/{idx_id}/daily',
                os.path.join(TIMING_ETF_CACHE_DIR, TIMING_ETF_CONFIGS[idx_id]['daily_cache_file']),
            ))
    paths_of_interest.append(('a_share_calendar', A_SHARE_CALENDAR_CACHE_FILE))
    paths_of_interest.append(('stock_data.parquet', STOCK_DATA_PARQUET))
    paths_of_interest.append(('stock_data.csv', STOCK_DATA_CSV))

    now = datetime.now(tz=timezone.utc).timestamp()
    for label, path in paths_of_interest:
        ts, source = _written_at(path)
        if ts is None:
            out.append((label, True, f'(missing)  {path}'))
            continue
        age_hours = (now - ts) / 3600.0
        when = datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
        prod = _produced_by(path) or '—'
        out.append((
            label, True,
            f'[{source:<5}] {when}  age={age_hours:6.1f}h  produced_by={prod}'
        ))
    return out


# ── main ────────────────────────────────────────────────────────────
def run_all_checks() -> int:
    sections: List[Tuple[str, List[Tuple[str, bool, str]]]] = [
        ('1. A 股指数 vs ETF 日线 max_date 对齐', _check_a_share_index_etf_alignment()),
        ('2. 美股指数 vs ETF 日线 max_date 对齐', _check_us_index_etf_alignment()),
        ('3. stock_data 月度面板 vs csi1000_monthly', _check_stock_data_vs_csi1000_monthly()),
        ('4. A 股 ETF 日期 ⊂ A 股交易日历', _check_a_share_etf_subset_of_calendar()),
        ('5. A 股 ETF 起点不早于指数起点', _check_a_share_etf_starts_after_index()),
        ('6. 数据血缘 lineage（meta.json 优先 / mtime 回退；仅展示，不影响 exit code）', _lineage_lines()),
    ]

    total = 0
    passed = 0
    failed_lines: List[str] = []
    for header, results in sections:
        print(f'\n── {header} ──')
        for name, ok, detail in results:
            total += 1
            mark = '✓' if ok else '✗'
            line = f'  {mark} {name}: {detail}'
            print(line)
            if ok:
                passed += 1
            else:
                failed_lines.append(line)

    print('\n' + '=' * 60)
    print(f'  {passed}/{total} checks passed')
    print('=' * 60)
    if failed_lines:
        print('\n[FAIL] 以下校验未通过：', file=sys.stderr)
        for line in failed_lines:
            print(line, file=sys.stderr)
        return 1
    print('[OK] 所有数据新鲜度/对齐校验通过')
    return 0


if __name__ == '__main__':
    sys.exit(run_all_checks())
