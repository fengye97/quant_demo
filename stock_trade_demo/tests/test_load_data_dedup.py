"""Regression tests for `load_data()` dedup invariants.

防的是 eb4d369：月中 supplement 产生多个 canonical 日期，groupby('交易日期') 会把它们
当独立换仓期，silent 多次换仓。dedup 现在带 invariant assert，若 dedup 逻辑被绕过，
load 阶段就 raise。
"""
from __future__ import annotations

import os
import tempfile

import pandas as pd
import pytest

from backtest import load_data


def _write_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, encoding='gbk', index=False)


def _minimal_row(date: str, code: str, ret: float = 0.01) -> dict:
    """A bare-minimum row matching stock_data.csv schema needs for load_data."""
    return {
        '交易日期': date,
        '股票代码': code,
        '涨跌幅': ret,
        '总市值': 1.0e9,
        '收盘价': 10.0,
    }


def test_load_data_dedup_keeps_latest_day_per_month(tmp_path):
    rows = [
        _minimal_row('2025-05-11', '000001'),
        _minimal_row('2025-05-12', '000001'),
        _minimal_row('2025-05-25', '000001'),
        _minimal_row('2025-05-11', '000002'),
        _minimal_row('2025-05-25', '000002'),
        _minimal_row('2025-06-25', '000001'),
        _minimal_row('2025-06-25', '000002'),
    ]
    csv_path = tmp_path / 'mini.csv'
    _write_csv(pd.DataFrame(rows), str(csv_path))

    df = load_data(str(csv_path))

    dates = sorted(df['交易日期'].dt.strftime('%Y-%m-%d').unique().tolist())
    assert dates == ['2025-05-25', '2025-06-25'], dates
    assert len(df) == 4  # 2 stocks × 2 months


def test_load_data_invariant_raises_if_dedup_bypassed(tmp_path, monkeypatch):
    """If dedup is bypassed and multiple canonical days survive per month,
    the invariant assert must raise instead of returning a bad df."""
    rows = [
        _minimal_row('2025-05-11', '000001'),
        _minimal_row('2025-05-25', '000001'),
    ]
    csv_path = tmp_path / 'mini.csv'
    _write_csv(pd.DataFrame(rows), str(csv_path))

    import backtest as bt
    from pandas.core.groupby.generic import SeriesGroupBy

    real_transform = SeriesGroupBy.transform

    def broken_transform(self, func, *a, **kw):
        if func == 'max':
            return self.obj
        return real_transform(self, func, *a, **kw)

    monkeypatch.setattr(SeriesGroupBy, 'transform', broken_transform)

    with pytest.raises(AssertionError, match='canonical'):
        bt.load_data(str(csv_path))


def test_load_data_single_day_per_month_passes(tmp_path):
    rows = [_minimal_row(f'2025-{m:02d}-25', '000001') for m in range(1, 6)]
    csv_path = tmp_path / 'mini.csv'
    _write_csv(pd.DataFrame(rows), str(csv_path))
    df = load_data(str(csv_path))
    assert df['交易日期'].nunique() == 5
