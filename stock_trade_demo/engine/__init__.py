"""Pure-function engine primitives extracted from ``backtest.py``.

Pillar 1 Step 7: move the math-only pieces (take-profit logic, period daily
curve construction, commission model) into a dedicated subpackage so future
strategies / backtesters can import them without pulling the whole
``backtest`` module (and its pandas-heavy data-loading side effects).

Backward compatibility: ``stock_trade_demo/backtest.py`` re-exports
``apply_take_profit`` and ``build_period_daily_curve`` from here, so every
existing call site (``compare_strategies.py``, ``choose_stock.py``,
``scripts/*.py``, web app) keeps working unchanged.
"""
from __future__ import annotations

from .costs import CommissionModel
from .take_profit import apply_take_profit, build_period_daily_curve

__all__ = [
    "CommissionModel",
    "apply_take_profit",
    "build_period_daily_curve",
]
