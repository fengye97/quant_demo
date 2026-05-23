"""
策略库 — 不同量化风格的选股策略类。

策略列表:
  OriginalStrategy  — 原版策略（行业估值 + bias反转 + 小市值），历史收益最高
  ChanEnhancedStrategy — 缠论增强策略 v1.1（原版过滤 + 缠论代理因子边际加成）
  ChanOnlyStrategy  — 纯缠论策略（仅缠论因子，无行业/bias过滤）
  MethodAStrategy   — Method A v2.0（日线缠论流水线 → 月度聚合因子）
  QualityValueStrategy — 质量价值小盘策略 v3.1（规模+价值+质量+反操纵 Z-score复合，size权重65%+10亿下限+5只持仓）
"""

from strategies.base import BaseStrategy
from strategies.original import OriginalStrategy
from strategies.original_ensemble import OriginalEnsembleStrategy
from strategies.chan_enhanced import ChanEnhancedStrategy
from strategies.chan_only import ChanOnlyStrategy
from strategies.method_a import MethodAStrategy
from strategies.quality_value import QualityValueStrategy

__all__ = [
    'BaseStrategy',
    'OriginalStrategy',
    'OriginalEnsembleStrategy',
    'ChanEnhancedStrategy',
    'ChanOnlyStrategy',
    'MethodAStrategy',
    'QualityValueStrategy',
]
