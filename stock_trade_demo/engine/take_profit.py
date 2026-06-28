"""Take-profit / stop-loss math, pure functions.

Extracted verbatim from ``backtest.py`` (Pillar 1 Step 7). No behavior
change — every floating-point output must be byte-equivalent to the
pre-extraction version (verified by re-running ``compare_strategies.py``
and ``choose_stock.py --strategy original`` against pre-/post-extraction
golden outputs).

Public API
----------
- :func:`apply_take_profit` — single-stock take-profit / stop-loss replay
  over a sequence of daily returns.
- :func:`build_period_daily_curve` — equal-weight portfolio daily curve
  for a single rebalance period, with per-stock TP/SL applied and a
  buy-cost haircut on day 1.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


def apply_take_profit(
    daily_returns: Sequence[float],
    tp_pct: float,
    sell_cost: float,
    sl_pct: Optional[float] = None,
) -> Tuple[List[float], bool]:
    """对单只股票的下周期日收益序列应用止盈规则。

    参数:
      daily_returns — list[float]，每天的涨跌幅
      tp_pct        — 止盈阈值（如 0.30 = 30%）
      sell_cost     — 卖出成本率（手续费+印花税）
      sl_pct        — 止损阈值（如 -0.20 = -20%），None 表示不启用止损

    返回:
      (modified_returns, triggered)
        modified_returns — 考虑止盈后的涨跌幅序列
        triggered        — 是否触发了止盈

    逻辑：
      逐日累积。一旦累积收益超过止盈阈值，当日扣除卖出成本后平仓，
      后续日期收益置零（资金闲置不参与市场波动）。
      如果到期末未触发止盈，最后一天扣除卖出成本。
    """
    cumret = 1.0
    result: List[float] = []
    triggered = False
    for r in daily_returns:
        if triggered:
            result.append(0.0)         # 已平仓，后续不参与
            continue
        cumret *= (1 + r)
        result.append(r)
        if cumret - 1 > tp_pct:
            triggered = True
            result[-1] = result[-1] - sell_cost  # 触发日扣卖出成本
        elif sl_pct is not None and cumret - 1 < sl_pct:
            triggered = True
            result[-1] = result[-1] - sell_cost
    return result, triggered


def build_period_daily_curve(
    daily_lists: Sequence[Sequence[float]],
    tp_pct: float,
    sell_cost: float,
    buy_cost: float,
    sl_pct: Optional[float] = None,
) -> List[float]:
    """基于单股逐日收益构造单个调仓周期的组合日线。"""
    period_len = max((len(x) for x in daily_lists if isinstance(x, list)), default=0)
    if period_len == 0:
        return [round(float(1 - buy_cost), 6)] if daily_lists else []

    stock_curves = []
    for daily_ret in daily_lists:
        if not isinstance(daily_ret, list) or len(daily_ret) == 0:
            stock_curves.append(np.ones(period_len))
            continue

        modified, triggered = apply_take_profit(daily_ret, tp_pct, sell_cost, sl_pct=sl_pct)
        curve = np.cumprod(np.array(modified, dtype=float) + 1.0)
        if len(curve) > 0 and not triggered:
            curve[-1] *= (1 - sell_cost)
        if len(curve) < period_len:
            curve = np.pad(curve, (0, period_len - len(curve)), constant_values=curve[-1])
        stock_curves.append(curve)

    portfolio_curve = np.mean(np.vstack(stock_curves), axis=0) * (1 - buy_cost)
    return [round(float(v), 6) for v in portfolio_curve.tolist()]


__all__ = ["apply_take_profit", "build_period_daily_curve"]
