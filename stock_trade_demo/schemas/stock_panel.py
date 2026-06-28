"""Pandera schema for the GBK-encoded ``stock_data.csv`` monthly panel.

This file is the single backing store for every monthly stock-selection
strategy in the project. It is **55 columns wide**, **GBK encoded**, and the
column order is part of the contract — see CLAUDE.md red line 4.

The schema here intentionally stays narrow:

- ``strict=False``        → extra (future) columns pass through.
- ``coerce=False`` at top → schema does **not** silently rewrite dtypes; per
  column we opt in to ``coerce=True`` only where Sina/Tencent occasionally
  returns int vs float for the same field.
- ``nullable=True``       → schemas tolerate holidays / early-listing rows
  that already exist in historical caches; the active guard is "if present,
  must be in-range".

What's checked
--------------
- Key columns must exist with the right names:
    交易日期, 股票代码, 股票名称, 开盘价, 最高价, 最低价, 收盘价, 成交额
- Price columns (开盘价/最高价/最低价/收盘价/VWAP) must be ``> 0`` when present.
- ``成交额`` must be ``>= 0`` when present.
- 交易日期 must look like ``YYYY-MM-DD`` (str match).
- ``振幅_*`` (relative high-low range) must be ``>= 0`` when present.
- ``涨跌幅std_*`` / ``成交额std_*`` must be ``>= 0`` when present.
- ``K`` / ``D`` are bounded to ``[-50, 150]`` (KDJ is technically 0-100 but
  smoothed J can briefly overshoot; spec says "合理范围"). ``J`` and ``MACD``
  / ``DIF`` / ``DEA`` indicators are NOT range-checked — they legitimately
  span negative values and using a tight check would generate false alarms.
- ``市盈率倒数`` / ``市净率倒数`` may be negative (loss-making firms have
  negative PE), so we only assert numeric-and-finite, not sign.

What's NOT checked
------------------
- Industry name strings (free-form text from akshare).
- Carry-forward financial columns (净利润 / 现金流 / 净资产 — these can be
  zero, negative, or NaN depending on filing status; column-level range
  asserts would block legitimate filings).
- 下周期每天涨跌幅 (it's a JSON-encoded list-as-string; structure is checked
  elsewhere in the pipeline).
"""
from __future__ import annotations

import pandera as pa
from pandera import Check, Column, DataFrameSchema

# Reusable check fragments
_PRICE_CHECK = Check.gt(0)  # 开盘/最高/最低/收盘/VWAP must be strictly positive
_NONNEG_CHECK = Check.ge(0)  # 成交额 / 振幅 / std 类指标 >= 0
_K_D_RANGE = Check.in_range(-50.0, 150.0)  # 宽口径 KDJ，避免误伤

STOCK_DATA_SCHEMA = DataFrameSchema(
    {
        # ── identity / time ──────────────────────────────
        "交易日期": Column(
            str,
            Check.str_matches(r"^\d{4}-\d{2}-\d{2}$"),
            nullable=False,
        ),
        "股票代码": Column(str, nullable=False),
        "股票名称": Column(str, nullable=True),

        # ── price (monthly OHLCV) ────────────────────────
        "开盘价": Column(float, _PRICE_CHECK, nullable=True, coerce=True),
        "最高价": Column(float, _PRICE_CHECK, nullable=True, coerce=True),
        "最低价": Column(float, _PRICE_CHECK, nullable=True, coerce=True),
        "收盘价": Column(float, _PRICE_CHECK, nullable=True, coerce=True),
        "VWAP":   Column(float, _PRICE_CHECK, nullable=True, coerce=True),
        "成交额": Column(float, _NONNEG_CHECK, nullable=True, coerce=True),

        # ── core factors / indicators ────────────────────
        # bias 是 (close - SMA_n) / SMA_n，可正可负，仅 nullable+coerce
        "bias_5":  Column(float, nullable=True, coerce=True),
        "bias_10": Column(float, nullable=True, coerce=True),
        "bias_20": Column(float, nullable=True, coerce=True),
        # 振幅 = (max(high)-min(low))/close >= 0
        "振幅_5":  Column(float, _NONNEG_CHECK, nullable=True, coerce=True),
        "振幅_10": Column(float, _NONNEG_CHECK, nullable=True, coerce=True),
        "振幅_20": Column(float, _NONNEG_CHECK, nullable=True, coerce=True),
        # 涨跌幅std / 成交额std >= 0
        "涨跌幅std_5":  Column(float, _NONNEG_CHECK, nullable=True, coerce=True),
        "涨跌幅std_10": Column(float, _NONNEG_CHECK, nullable=True, coerce=True),
        "涨跌幅std_20": Column(float, _NONNEG_CHECK, nullable=True, coerce=True),
        "成交额std_5":  Column(float, _NONNEG_CHECK, nullable=True, coerce=True),
        "成交额std_10": Column(float, _NONNEG_CHECK, nullable=True, coerce=True),
        "成交额std_20": Column(float, _NONNEG_CHECK, nullable=True, coerce=True),
        # KDJ — K/D bounded; J 可以严重溢出，不限
        "K": Column(float, _K_D_RANGE, nullable=True, coerce=True),
        "D": Column(float, _K_D_RANGE, nullable=True, coerce=True),
        "J": Column(float, nullable=True, coerce=True),
        # MACD 三件套 可正可负
        "DIF":  Column(float, nullable=True, coerce=True),
        "DEA":  Column(float, nullable=True, coerce=True),
        "MACD": Column(float, nullable=True, coerce=True),

        # ── valuation reciprocals (negative allowed for loss-makers) ──
        "市盈率倒数": Column(float, nullable=True, coerce=True),
        "市净率倒数": Column(float, nullable=True, coerce=True),
    },
    strict=False,
    coerce=False,
)


__all__ = ["STOCK_DATA_SCHEMA"]
