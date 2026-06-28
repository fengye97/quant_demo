"""Commission / transaction-cost model.

Today the costs in ``select_and_backtest`` reduce to two rates:

  - ``c_rate``  — broker commission (default ``1/10000``)
  - ``t_rate``  — stamp duty / 印花税 (default ``1/1000``)
  - ``sell_cost = c_rate + t_rate`` is paid on every sell
  - ``buy_cost  = c_rate`` is paid on every buy

This dataclass is a thin, side-effect-free wrapper so future work
(过户费 / 滑点 / 异常停牌补偿) can be added without changing every call
site. Existing callers (backtest.select_and_backtest) keep passing
``c_rate`` / ``t_rate`` floats — they're unaffected.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommissionModel:
    """Fee schedule for one round-trip stock trade.

    Attributes
    ----------
    c_rate : float
        Broker commission rate, applied on both buy and sell (default 万 1).
    t_rate : float
        Stamp duty / 印花税, applied only on sell (default 千 1).
    """

    c_rate: float = 1.0 / 10000
    t_rate: float = 1.0 / 1000

    @property
    def buy_cost(self) -> float:
        """Per-unit-capital cost incurred on each buy leg."""
        return float(self.c_rate)

    @property
    def sell_cost(self) -> float:
        """Per-unit-capital cost incurred on each sell leg
        (commission + stamp duty)."""
        return float(self.c_rate + self.t_rate)

    @classmethod
    def default(cls) -> "CommissionModel":
        """A-share default: 万 1 佣金 + 千 1 印花税."""
        return cls()


__all__ = ["CommissionModel"]
