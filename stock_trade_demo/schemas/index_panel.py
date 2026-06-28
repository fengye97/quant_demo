"""Pandera schemas for index / ETF daily panels.

This module declares lightweight ``DataFrameSchema`` objects that
``atomic_write_csv(..., schema=...)`` can validate against **before** any
tmp file is written. Any schema failure raises and leaves the target file
untouched (no partial / silently-bad CSV ever lands on disk).

Schemas intentionally stay narrow:
- ``strict=False``  → only fail on declared columns; extra cols pass through.
- ``coerce=False``  → caller must supply correct dtypes; we don't auto-cast.
- ``nullable=True`` on price/volume columns to tolerate exchange holidays /
  early-listing blanks that already exist in historical caches; the active
  guard is "if present, must be > 0".
"""
from __future__ import annotations

import pandera as pa
from pandera import Check, Column, DataFrameSchema

# A 股指数/ETF 日线通用 schema（csi1000/chinext/star50/nasdaq/sp500 的 daily CSV）
INDEX_DAILY_SCHEMA = DataFrameSchema(
    {
        "date": Column(
            str,
            Check.str_matches(r"^\d{4}-\d{2}-\d{2}$"),
            nullable=False,
        ),
        # 价格/成交量统一 coerce=True：上游 Sina 返回的 volume 是 int64，
        # akshare 返回的是 float64；不让 schema 因为 dtype 微差异 reject
        # （reject 后会被 index_data.py 的 try/except 误兜底为 "stale cache"）。
        "open": Column(float, Check.gt(0), nullable=True, coerce=True),
        "high": Column(float, Check.gt(0), nullable=True, coerce=True),
        "low": Column(float, Check.gt(0), nullable=True, coerce=True),
        "close": Column(float, Check.gt(0), nullable=True, coerce=True),
        "volume": Column(float, Check.ge(0), nullable=True, coerce=True),
    },
    strict=False,
    coerce=False,
)

# 月度指数 schema（暂仅校验 date 列存在；后续可逐步扩展）
#
# TODO(load-only/schema): 本 schema 当前 **0 callers**，是为
# ``index_data.get_index_returns(..., frequency='monthly')`` 预留的校验位。
# 接入时机：等 `index_data._compute_monthly_returns` 的输出列稳定
# （目前是 ``date`` + ``{series_name}_return``，但 series_name 因 index_id 而异，
# 不能用一张固定 schema 直接校验），再把 schema 扩成 "date + 任意 *_return"
# 或者改成 Per-index 工厂函数；那一步会触发 SchemaError 边界，必须连带
# 给 atomic_write_csv 的调用点补 try/except + meta.json 失效逻辑，
# 因此本 sprint 仅保留预留，不做接入。接入后请删掉本 TODO。
INDEX_MONTHLY_SCHEMA = DataFrameSchema(
    {
        "date": Column(str, nullable=False),
    },
    strict=False,
    coerce=False,
)


__all__ = ["INDEX_DAILY_SCHEMA", "INDEX_MONTHLY_SCHEMA"]
