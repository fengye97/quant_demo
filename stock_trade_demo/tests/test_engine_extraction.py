"""Tests for the engine/ subpackage (Pillar 1 Step 7 extraction).

Pure-function 引擎 = ``stock_trade_demo/engine/{costs,take_profit}.py``.

什么需要保护：
  - apply_take_profit / build_period_daily_curve 的字节等价输出
    （compare_strategies.py / choose_stock.py 的 golden diff 已在另一处覆盖）
  - 边界：空 list / 单 day / tp 不触发 / sl 触发 / sl=None / 空 daily_lists
  - CommissionModel.buy_cost / sell_cost 数值正确

  - backtest.py 通过 re-export 暴露同名符号 → 旧调用方零改动；这里也加一条
    身份测试以防被破坏。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import backtest  # noqa: E402
from engine import CommissionModel, apply_take_profit, build_period_daily_curve  # noqa: E402
from engine.take_profit import (  # noqa: E402
    apply_take_profit as _atp_direct,
    build_period_daily_curve as _bpdc_direct,
)


# ── apply_take_profit ──────────────────────────────────────────────────
def test_apply_take_profit_no_trigger_simple():
    """累积涨幅未到 tp_pct，应保留原序列，末日不扣 sell_cost（由 caller 收尾）。"""
    daily = [0.01, 0.02, -0.005]
    result, triggered = apply_take_profit(daily, tp_pct=0.30, sell_cost=0.001)
    assert triggered is False
    assert result == daily
    # 注意 apply_take_profit 自身不在 untriggered 末日扣 sell_cost；
    # 那是 build_period_daily_curve 的工作。


def test_apply_take_profit_trigger_mid_period():
    """tp 在第三天触发：第三天的 r 被扣 sell_cost，后续日置零。"""
    daily = [0.1, 0.2, 0.3, 0.5]  # cumret 在 day3 末 = 1.1*1.2*1.3 = 1.716 > 1+0.5
    result, triggered = apply_take_profit(daily, tp_pct=0.50, sell_cost=0.001)
    assert triggered is True
    assert result[0] == 0.1
    assert result[1] == 0.2
    assert result[2] == pytest.approx(0.3 - 0.001)
    assert result[3] == 0.0


def test_apply_take_profit_stop_loss_trigger():
    """sl_pct 触发：第二天累积 -25% < -20%，扣 sell_cost 后平仓。"""
    daily = [-0.10, -0.15, 0.50]
    result, triggered = apply_take_profit(daily, tp_pct=0.30, sell_cost=0.001, sl_pct=-0.20)
    assert triggered is True
    assert result[0] == -0.10
    # day2: cumret = 0.9*0.85 = 0.765, drop -0.235 < -0.20 → trigger
    assert result[1] == pytest.approx(-0.15 - 0.001)
    assert result[2] == 0.0


def test_apply_take_profit_sl_none_disables_stop_loss():
    daily = [-0.30, -0.30, -0.30]
    result, triggered = apply_take_profit(daily, tp_pct=0.50, sell_cost=0.001, sl_pct=None)
    assert triggered is False
    assert result == daily


def test_apply_take_profit_empty_daily():
    result, triggered = apply_take_profit([], tp_pct=0.30, sell_cost=0.001)
    assert result == []
    assert triggered is False


# ── build_period_daily_curve ──────────────────────────────────────────
def test_build_period_curve_empty_daily_lists():
    """空持仓 list → 空曲线。"""
    assert build_period_daily_curve([], tp_pct=0.30, sell_cost=0.001, buy_cost=0.0001) == []


def test_build_period_curve_all_empty_inner_lists():
    """有 outer list 但每只股票 daily list 是空 → 只返回首日 (1-buy_cost)。"""
    out = build_period_daily_curve(
        [[], []], tp_pct=0.30, sell_cost=0.001, buy_cost=0.0001
    )
    assert out == [round(1 - 0.0001, 6)]


def test_build_period_curve_basic_two_stocks():
    """两只股票 / 同样 3 天 / 不触发 tp：等权平均、首日扣买入成本、末日扣 sell_cost。"""
    daily = [[0.01, 0.0, 0.02], [0.0, 0.01, -0.01]]
    curve = build_period_daily_curve(daily, tp_pct=0.30, sell_cost=0.001, buy_cost=0.0001)
    assert len(curve) == 3
    # 末值约 = (1.01*1.0*1.02*(1-sell) + 1.0*1.01*0.99*(1-sell)) / 2 * (1-buy)
    expected_last = (
        1.01 * 1.0 * 1.02 * (1 - 0.001) +
        1.0 * 1.01 * 0.99 * (1 - 0.001)
    ) / 2 * (1 - 0.0001)
    assert curve[-1] == pytest.approx(round(expected_last, 6), abs=1e-6)


def test_build_period_curve_handles_uneven_lengths_via_pad():
    """长度不齐：短曲线用末值 pad 到最长长度。"""
    daily = [[0.05, 0.05, 0.05], [0.10]]  # 第二只只有 1 天
    curve = build_period_daily_curve(daily, tp_pct=1.0, sell_cost=0.001, buy_cost=0.0)
    assert len(curve) == 3
    # 第二只 day0 收益 0.10 然后到末日扣 sell_cost → 1.10*(1-0.001) = 1.09890
    # 然后 pad 到长度 3：[1.0989, 1.0989, 1.0989]
    # 第一只逐日累积 [1.05, 1.1025, 1.157625]，末日触发 sell_cost
    # 末值 = mean([1.157625*(1-0.001), 1.0989]) * (1-0) = (1.156467375 + 1.0989) / 2
    expected_last = (1.05 * 1.05 * 1.05 * (1 - 0.001) + 1.10 * (1 - 0.001)) / 2
    assert curve[-1] == pytest.approx(round(expected_last, 6), abs=1e-6)


# ── CommissionModel ────────────────────────────────────────────────────
def test_commission_model_default_values_match_legacy_floats():
    """新 dataclass 必须与 backtest 历史默认值 (c=万1, t=千1) 完全一致。"""
    cm = CommissionModel.default()
    assert cm.c_rate == 1.0 / 10000
    assert cm.t_rate == 1.0 / 1000
    assert cm.buy_cost == 1.0 / 10000
    assert cm.sell_cost == pytest.approx(1.0 / 10000 + 1.0 / 1000)


def test_commission_model_custom_rates():
    cm = CommissionModel(c_rate=2.0 / 10000, t_rate=5.0 / 10000)
    assert cm.buy_cost == 2.0 / 10000
    assert cm.sell_cost == pytest.approx(7.0 / 10000)


# ── back-compat：backtest.py re-export ────────────────────────────────
def test_backtest_module_reexports_engine_symbols_identically():
    """旧调用方 import 路径不可被破坏：backtest.apply_take_profit 必须是同一对象。"""
    assert backtest.apply_take_profit is _atp_direct
    assert backtest.build_period_daily_curve is _bpdc_direct
    assert backtest.CommissionModel is CommissionModel
