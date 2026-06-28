"""Regression tests for the acf2ca2 fixes：

1. `index_data.get_index_daily(force_refetch=True)` 必须带 freshness guard：抓回的
   max_date < 本地缓存 max_date 时禁止覆盖（之前 stale 数据会被强行写入，污染 A 股
   交易日历，导致持仓区间显示漂移）。
2. `web_app._check_a_share_index_etf_alignment()` 必须能识别"指数日线 max_date 落后
   于 ETF 日线 max_date"的不同步状态——CLAUDE.md 明确规定这种情况下绝不允许静默
   显示"刷新完成"。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import index_data


# ───────────────────────────── get_index_daily freshness guard ─────────────────────────────

def _write_index_cache(cache_dir: Path, index_id: str, dates: list[str], closes=None) -> Path:
    """Write a fake daily index CSV at the path get_index_daily would look up."""
    closes = closes if closes is not None else [10.0] * len(dates)
    fname = index_data.INDEX_CONFIGS[index_id]['daily_cache_file']
    path = cache_dir / fname
    df = pd.DataFrame({
        'date': pd.to_datetime(dates),
        'open': closes, 'high': closes, 'low': closes,
        'close': closes, 'volume': [100] * len(dates),
    })
    df.to_csv(path, index=False)
    return path


def test_get_index_daily_freshness_guard_blocks_stale_overwrite(tmp_path, monkeypatch):
    """force_refetch=True 时，新抓回的指数 max_date < 本地缓存 max_date → 必须保留缓存。

    没有该 guard 时，stale 数据会写入磁盘（覆盖 A 股交易日历真值源），导致 UI 持仓
    区间漂移。回归这条保护的核心断言是：cache 文件内容（max_date）不应被改写。
    """
    monkeypatch.setattr(index_data, 'CACHE_DIR', str(tmp_path))

    # 本地缓存最新到 2026-05-25
    cache_path = _write_index_cache(
        tmp_path, 'csi1000',
        dates=['2026-05-23', '2026-05-24', '2026-05-25'],
    )
    cached_max_before = pd.read_csv(cache_path, parse_dates=['date'])['date'].max()
    assert cached_max_before == pd.Timestamp('2026-05-25')

    # mock 网络抓取返回更旧的数据
    def _stale_fetch(symbol):
        return pd.DataFrame({
            'date': pd.to_datetime(['2026-05-10', '2026-05-11']),
            'open': [9.0, 9.1], 'high': [9.2, 9.3],
            'low': [8.8, 8.9], 'close': [9.0, 9.1],
            'volume': [100, 110],
        })
    monkeypatch.setattr(index_data, '_fetch_daily_kline', _stale_fetch)

    returned = index_data.get_index_daily('csi1000', force_refetch=True)

    # 1) 返回值应该是本地缓存（max_date 2026-05-25），不是网络新数据
    assert returned['date'].max() == pd.Timestamp('2026-05-25'), \
        f'stale fetch 不应替换本地新缓存，实际 max_date = {returned["date"].max()}'

    # 2) 磁盘 cache 必须保持原样
    after = pd.read_csv(cache_path, parse_dates=['date'])
    assert after['date'].max() == cached_max_before
    assert len(after) == 3  # 仍是原来 3 行，没有被 stale 2 行覆盖


def test_get_index_daily_accepts_fresher_fetch(tmp_path, monkeypatch):
    """正常路径：抓回的 max_date > 本地缓存 max_date → 应当覆盖缓存。"""
    monkeypatch.setattr(index_data, 'CACHE_DIR', str(tmp_path))

    _write_index_cache(
        tmp_path, 'csi1000',
        dates=['2026-05-23', '2026-05-24'],
    )

    def _fresh_fetch(symbol):
        return pd.DataFrame({
            'date': pd.to_datetime(['2026-05-23', '2026-05-24', '2026-05-25', '2026-05-26']),
            'open': [9.0, 9.1, 9.2, 9.3], 'high': [9.5, 9.6, 9.7, 9.8],
            'low': [8.8, 8.9, 9.0, 9.1], 'close': [9.0, 9.1, 9.2, 9.3],
            'volume': [100, 110, 120, 130],
        })
    monkeypatch.setattr(index_data, '_fetch_daily_kline', _fresh_fetch)

    returned = index_data.get_index_daily('csi1000', force_refetch=True)
    assert returned['date'].max() == pd.Timestamp('2026-05-26')

    # disk 也应被覆盖到最新
    cache_path = tmp_path / index_data.INDEX_CONFIGS['csi1000']['daily_cache_file']
    on_disk = pd.read_csv(cache_path, parse_dates=['date'])
    assert on_disk['date'].max() == pd.Timestamp('2026-05-26')


def test_get_index_daily_equal_dates_skips_overwrite(tmp_path, monkeypatch, capsys):
    """边界：fetched_max == cached_max 且 cached 不更短 → 不应重写磁盘。"""
    monkeypatch.setattr(index_data, 'CACHE_DIR', str(tmp_path))
    cache_path = _write_index_cache(
        tmp_path, 'csi1000',
        dates=['2026-05-23', '2026-05-24', '2026-05-25'],
    )
    mtime_before = os.path.getmtime(cache_path)

    def _equal_fetch(symbol):
        return pd.DataFrame({
            'date': pd.to_datetime(['2026-05-23', '2026-05-24', '2026-05-25']),
            'open': [9.0, 9.1, 9.2], 'high': [9.5, 9.6, 9.7],
            'low': [8.8, 8.9, 9.0], 'close': [9.0, 9.1, 9.2],
            'volume': [100, 110, 120],
        })
    monkeypatch.setattr(index_data, '_fetch_daily_kline', _equal_fetch)

    returned = index_data.get_index_daily('csi1000', force_refetch=True)
    assert returned['date'].max() == pd.Timestamp('2026-05-25')
    # mtime 不应变（skip overwrite 分支）
    assert os.path.getmtime(cache_path) == mtime_before


# ───────────────────────────── A 股指数 vs ETF 对齐检查 ─────────────────────────────

@pytest.fixture
def alignment_check():
    """延迟导入 web_app（heavy），但模块本身没有数据加载副作用，import 即可拿到函数。"""
    import web_app
    return web_app


def _df_with_dates(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({'date': pd.to_datetime(dates), 'close': [10.0] * len(dates)})


def test_alignment_returns_mismatch_when_index_behind_etf(alignment_check, monkeypatch):
    """指数 max_date < ETF max_date → 必须出现在返回的 mismatches 里。

    在调用方（`_run_index_data_update`）里这会被 raise，但 alignment 函数本身只负责
    汇总。测试只需要确认它正确识别了不同步状态。
    """
    def _idx(index_id, *args, **kw):
        # 三个 A 股指数都比 ETF 落后一天
        return _df_with_dates(['2026-05-23', '2026-05-24'])
    def _etf(index_id, *args, **kw):
        return _df_with_dates(['2026-05-23', '2026-05-24', '2026-05-25'])

    monkeypatch.setattr(alignment_check, 'get_index_daily', _idx)
    monkeypatch.setattr(alignment_check, 'get_timing_etf_daily', _etf)

    mismatches = alignment_check._check_a_share_index_etf_alignment()
    assert mismatches, '指数落后 ETF 应被识别为 mismatch'
    by_id = {m['index_id']: m for m in mismatches}
    # 三个 A 股指数都应被报告
    assert set(by_id) == {'csi1000', 'chinext', 'star50'}
    for m in mismatches:
        assert m['index_max_date'] == '2026-05-24'
        assert m['etf_max_date'] == '2026-05-25'


def test_alignment_passes_when_dates_equal(alignment_check, monkeypatch):
    """正常路径：max_date 相等 → 不应报告 mismatch。"""
    def _same(index_id, *args, **kw):
        return _df_with_dates(['2026-05-23', '2026-05-24', '2026-05-25'])
    monkeypatch.setattr(alignment_check, 'get_index_daily', _same)
    monkeypatch.setattr(alignment_check, 'get_timing_etf_daily', _same)
    mismatches = alignment_check._check_a_share_index_etf_alignment()
    assert mismatches == [], f'相等 max_date 不应报告 mismatch，实际 {mismatches}'


def test_alignment_ok_when_etf_behind_index(alignment_check, monkeypatch):
    """ETF 比 index 落后 1 天属正常刷新节奏（同日/次日的小滞后）→ 不应报告。

    容差设计：etf_max < idx_max 且滞后 <= STALE_TOLERANCE_DAYS(2) 不报，
    避免在每个刷新前同日误报。只有大滞后（>2 天）才报，见
    test_alignment_flags_etf_behind_index_large_lag。
    """
    def _idx(index_id, *args, **kw):
        return _df_with_dates(['2026-05-23', '2026-05-24', '2026-05-25'])
    def _etf(index_id, *args, **kw):
        return _df_with_dates(['2026-05-23', '2026-05-24'])

    monkeypatch.setattr(alignment_check, 'get_index_daily', _idx)
    monkeypatch.setattr(alignment_check, 'get_timing_etf_daily', _etf)

    mismatches = alignment_check._check_a_share_index_etf_alignment()
    assert mismatches == []


def test_alignment_flags_etf_behind_index_large_lag(alignment_check, monkeypatch):
    """ETF 长期落后 index（>2 天，典型：非 qfq 已新但 qfq 停留数周）→ 必须报告。

    回归 P0-2：之前只检 idx<etf 单向，导致 ETF 陈旧 5 周被静默放行，择时信号/结算价
    停在旧日期却按今日 as_of 展示。现在必须以 direction='etf_behind' 报出。
    """
    def _idx(index_id, *args, **kw):
        return _df_with_dates(['2026-06-24', '2026-06-25', '2026-06-26'])
    def _etf(index_id, *args, **kw):
        return _df_with_dates(['2026-05-23', '2026-05-24', '2026-05-26'])

    monkeypatch.setattr(alignment_check, 'get_index_daily', _idx)
    monkeypatch.setattr(alignment_check, 'get_timing_etf_daily', _etf)

    mismatches = alignment_check._check_a_share_index_etf_alignment()
    assert mismatches, 'ETF 大滞后于 index 应被识别为 mismatch'
    by_id = {m['index_id']: m for m in mismatches}
    assert set(by_id) == {'csi1000', 'chinext', 'star50'}
    for m in mismatches:
        assert m['direction'] == 'etf_behind'
        assert m['index_max_date'] == '2026-06-26'
        assert m['etf_max_date'] == '2026-05-26'
        assert m['lag_days'] == 31


def test_alignment_only_one_index_behind(alignment_check, monkeypatch):
    """只有一个指数落后 → 只报告那一个，其余两个不报告。"""
    def _idx(index_id, *args, **kw):
        if index_id == 'chinext':
            return _df_with_dates(['2026-05-23'])  # 落后
        return _df_with_dates(['2026-05-25'])
    def _etf(index_id, *args, **kw):
        return _df_with_dates(['2026-05-25'])

    monkeypatch.setattr(alignment_check, 'get_index_daily', _idx)
    monkeypatch.setattr(alignment_check, 'get_timing_etf_daily', _etf)

    mismatches = alignment_check._check_a_share_index_etf_alignment()
    assert len(mismatches) == 1
    assert mismatches[0]['index_id'] == 'chinext'
    assert mismatches[0]['index_max_date'] == '2026-05-23'
    assert mismatches[0]['etf_max_date'] == '2026-05-25'
