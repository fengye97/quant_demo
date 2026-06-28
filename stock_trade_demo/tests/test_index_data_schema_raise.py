"""Tests for `index_data.py` SchemaError 拆分（Task #4a）.

CLAUDE.md 红线：schema 校验是为了保护数据。如果上游返回脏数据，必须
**硬失败**，不能被 try/except 静默吞掉降级到旧缓存——那等于反过来阻止
新数据落盘，掩盖问题。

本文件 reproduce 两种场景：
  1. fetch 返回脏数据（open<=0） → 必须 raise SchemaError / SchemaErrors，
     **不**返回旧缓存
  2. fetch 抛网络错（HTTPError） → 仍允许 fall back to cache（原有兜底语义保留）
"""
from __future__ import annotations

import sys
import urllib.error
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandera as pa  # noqa: E402
from pandera.errors import SchemaError, SchemaErrors  # noqa: E402

import index_data  # noqa: E402


def _good_frame():
    return pd.DataFrame({
        'date': pd.to_datetime(['2026-05-23', '2026-05-24']),
        'open': [1.0, 1.1],
        'high': [1.2, 1.3],
        'low': [0.9, 1.0],
        'close': [1.1, 1.2],
        'volume': [100.0, 200.0],
    })


def _bad_frame_open_zero():
    df = _good_frame()
    df.loc[0, 'open'] = 0.0  # 触发 INDEX_DAILY_SCHEMA gt(0) 失败
    return df


def test_schema_error_on_fetch_raises_not_silent_fallback(monkeypatch, tmp_path):
    """脏数据 fetch 必须 raise，不得静默回退到缓存。"""
    # 注入一份比 fetch 日期更早的"良好"缓存，避免被 freshness guard 提前 short-circuit
    cache_file = tmp_path / 'csi1000_daily.csv'
    old_cache = pd.DataFrame({
        'date': ['2020-01-02', '2020-01-03'],
        'open': [1.0, 1.1], 'high': [1.2, 1.3], 'low': [0.9, 1.0],
        'close': [1.1, 1.2], 'volume': [100.0, 200.0],
    })
    old_cache.to_csv(cache_file, index=False)

    # 把 INDEX_CONFIGS / CACHE_DIR 都骗到 tmp_path 下
    monkeypatch.setattr(index_data, 'CACHE_DIR', str(tmp_path))
    # 让 _fetch_daily_kline 返回脏数据（open=0）
    monkeypatch.setattr(index_data, '_fetch_daily_kline',
                        lambda symbol: _bad_frame_open_zero())

    with pytest.raises((SchemaError, SchemaErrors)):
        index_data.get_index_daily('csi1000', force_refetch=True)

    # 缓存文件必须保留原样（schema 校验在 atomic_write_csv 内部，写入未发生）
    assert cache_file.exists()
    after = pd.read_csv(cache_file)
    assert len(after) == 2
    # 没有出现 open=0 的行（不可能被写入），且仍是 2020 的旧缓存内容
    assert (after['open'] > 0).all()
    assert after['date'].iloc[0] == '2020-01-02'


def test_network_error_on_fetch_still_falls_back_to_cache(monkeypatch, tmp_path):
    """网络错（HTTPError 等）仍然允许 fall back to cache，原有语义不变。

    隔离要点：mock 整条 fetch 入口 `_fetch_daily_kline_with_fallback`（Sina→东财双线），
    而不是只 mock Sina 的 `_fetch_daily_kline`。否则当东财 push2his.eastmoney.com
    真实可达时，未 mock 的东财线会用真实数据（5215 行）覆盖缓存，使测试依赖外部
    网络可达性、在东财恢复后误判失败。
    """
    cache_file = tmp_path / 'csi1000_daily.csv'
    seeded = _good_frame().assign(date=lambda d: d['date'].dt.strftime('%Y-%m-%d'))
    seeded.to_csv(cache_file, index=False)

    monkeypatch.setattr(index_data, 'CACHE_DIR', str(tmp_path))

    def _network_fail(symbol):
        raise urllib.error.HTTPError('http://x', 500, 'boom', {}, None)

    monkeypatch.setattr(index_data, '_fetch_daily_kline_with_fallback', _network_fail)

    returned = index_data.get_index_daily('csi1000', force_refetch=True)
    # 没 raise，落回了缓存
    assert returned is not None
    assert len(returned) == 2
