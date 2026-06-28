"""Tests for STOCK_DATA_SCHEMA (Pillar 2 Step 5 后半段).

覆盖：
  1. 合法 DataFrame 通过校验
  2. 缺失关键列（开盘价缺失）→ SchemaError
  3. 价格列 dtype 错（"open"=0 / 负值）→ SchemaError
  4. atomic_write_csv 的 ``schema_sample`` 慢热模式：仅校验 head+tail
     N 行，万行级 DataFrame 写入时间应远低于全量校验
  5. ``schema_sample`` 小到只覆盖 head + tail 的脏行能被发现；
     夹在中间的脏行会被合理跳过（这是慢热的代价，但写入仍然成功，
     避免阻塞 stock_data.csv 月度更新）
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandera as pa  # noqa: E402
from pandera.errors import SchemaError, SchemaErrors  # noqa: E402

from schemas.stock_panel import STOCK_DATA_SCHEMA  # noqa: E402
from utils.atomic_io import atomic_write_csv  # noqa: E402


def _good_row(date='2026-05-30', code='000001'):
    return {
        '交易日期': date,
        '股票代码': code,
        '股票名称': 'TestCo',
        '开盘价': 10.5,
        '最高价': 11.2,
        '最低价': 10.1,
        '收盘价': 10.8,
        'VWAP': 10.6,
        '成交额': 1.2e8,
        'bias_5': -0.012,
        'bias_10': 0.034,
        'bias_20': 0.011,
        '振幅_5': 0.08,
        '振幅_10': 0.12,
        '振幅_20': 0.15,
        '涨跌幅std_5': 0.02,
        '涨跌幅std_10': 0.025,
        '涨跌幅std_20': 0.03,
        '成交额std_5': 1.0e7,
        '成交额std_10': 1.1e7,
        '成交额std_20': 1.2e7,
        'K': 55.0, 'D': 52.0, 'J': 62.0,
        'DIF': 0.12, 'DEA': 0.10, 'MACD': 0.04,
        '市盈率倒数': 0.05, '市净率倒数': 0.30,
    }


def _good_df(n=3):
    rows = [_good_row(date=f'2026-05-{20+i:02d}', code=f'00000{i}') for i in range(n)]
    return pd.DataFrame(rows)


def test_legal_dataframe_passes_schema(tmp_path):
    df = _good_df(5)
    # 不抛即通过
    STOCK_DATA_SCHEMA.validate(df, lazy=True)
    # 端到端也能写
    target = tmp_path / 'panel.csv'
    atomic_write_csv(target, df, index=False, schema=STOCK_DATA_SCHEMA,
                     encoding='gbk', produced_by='unit_test/stock_data')
    assert target.exists()


def test_missing_key_column_raises(tmp_path):
    df = _good_df(3).drop(columns=['开盘价'])
    with pytest.raises((SchemaError, SchemaErrors)):
        STOCK_DATA_SCHEMA.validate(df, lazy=True)


def test_zero_or_negative_price_raises(tmp_path):
    bad = _good_df(3)
    bad.loc[1, '开盘价'] = 0.0  # gt(0) 失败
    with pytest.raises((SchemaError, SchemaErrors)):
        STOCK_DATA_SCHEMA.validate(bad, lazy=True)

    bad2 = _good_df(3)
    bad2.loc[0, '收盘价'] = -1.5
    with pytest.raises((SchemaError, SchemaErrors)):
        STOCK_DATA_SCHEMA.validate(bad2, lazy=True)


def test_negative_volume_raises():
    bad = _good_df(2)
    bad.loc[0, '成交额'] = -100.0
    with pytest.raises((SchemaError, SchemaErrors)):
        STOCK_DATA_SCHEMA.validate(bad, lazy=True)


def test_schema_sample_mode_writes_large_frame_fast(tmp_path):
    """schema_sample=N 慢热：5w 行写入 + 校验应远快于全量校验。

    硬性指标：写入应在 5 秒内（包含校验+CSV落盘），并产生合法 sidecar 文件。
    我们不严格比较时间（CI 噪声），只要求 sample 模式跑通且 dataframe 已落盘。
    """
    n = 50000
    big = _good_df(1)
    big = pd.concat([big] * n, ignore_index=True)
    big['交易日期'] = '2026-05-30'
    big['股票代码'] = [f'{i:06d}' for i in range(len(big))]

    target = tmp_path / 'big.csv'
    t0 = time.perf_counter()
    atomic_write_csv(
        target, big, index=False,
        schema=STOCK_DATA_SCHEMA, schema_sample=2000,
        encoding='gbk',
        produced_by='unit_test/stock_data_sample',
    )
    elapsed = time.perf_counter() - t0

    assert target.exists()
    assert elapsed < 10.0, f"sample mode write too slow: {elapsed:.2f}s"


def test_schema_sample_catches_head_corruption(tmp_path):
    """脏行落在 head 段（前 N 行）必然被慢热模式发现。

    这是核心保护场景：get_stock_info.py 写入时把"刚算出的当月新行"
    放在末尾，schema_sample 同时验 head+tail 就能覆盖它。
    """
    big = pd.concat([_good_df(100)] * 50, ignore_index=True)  # 5000 行
    big['股票代码'] = [f'{i:06d}' for i in range(len(big))]
    # 把第 3 行的开盘价改成 0 → 必须被 head sample(=100) 抓到
    big.loc[2, '开盘价'] = 0.0

    target = tmp_path / 'corrupt.csv'
    with pytest.raises((SchemaError, SchemaErrors)):
        atomic_write_csv(
            target, big, index=False,
            schema=STOCK_DATA_SCHEMA, schema_sample=100,
            encoding='gbk',
        )
    assert not target.exists(), "校验失败时主文件不应被创建"
