"""
策略库 — 不同量化风格的选股策略类。

历史上这里把所有具体策略类全部 import 到包顶层，方便 `from strategies import OriginalStrategy`
之类的写法。但这会让 `import strategies.registry`（一个本应零依赖的小模块）也被迫触发整套
策略类装载，从而和 `timing.strategies` 形成循环（original_ensemble → timing.strategies → timing.base
→ strategies.registry → strategies/__init__ → original_ensemble）。

经检查全仓没有任何 `from strategies import X` 用法，所有调用方都走 `strategies.<module>`
的子模块路径，所以这里只保留 BaseStrategy 的便捷 re-export，避免顶层 import 副作用。

策略列表（按子模块路径直接 import）:
  strategies.original.OriginalStrategy           — 原版策略（行业估值 + bias反转 + 小市值），历史收益最高
  strategies.original_ensemble.OriginalEnsembleStrategy — 多窗口投票增强版
  strategies.chan_enhanced.ChanEnhancedStrategy  — 缠论增强策略 v1.1
  strategies.chan_only.ChanOnlyStrategy          — 纯缠论策略
  strategies.method_a.MethodAStrategy            — Method A v2.0（日线缠论流水线）
  strategies.quality_value.QualityValueStrategy  — 质量价值小盘策略 v3.1
  strategies.sector_heat.SectorHeatStrategy      — 行业热度选股
"""

from strategies.base import BaseStrategy

__all__ = [
    'BaseStrategy',
]
