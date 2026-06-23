"""无副作用的策略注册表（Pillar 1 Step 5）。

这个模块**只**定义三个 registry dict 与 `_register_strategy()` 辅助函数，
不 import 任何具体策略类，也不 import `strategies` 包本身。

目的：避免 strategies/__init__.py 与 timing/strategies.py 在 import 时形成循环。
- strategies/__init__.py 会再次 import 各子模块（如 original_ensemble），
  而 original_ensemble 又会 import timing.strategies；
- timing/strategies.py 在 class 定义时需要触发 `_register_strategy(cls)`；
- 如果注册函数住在 `strategies.base` 里，timing 端去 import 它就会走
  `strategies/__init__.py`，引发部分初始化的循环 ImportError。

把注册函数下沉到这个零依赖模块后，两边都可以无副作用地 import。

下游兼容：`strategies.base` 重新 export 同名符号，老代码 `from strategies.base import
STRATEGY_REGISTRY` 仍可工作。
"""
from __future__ import annotations

STRATEGY_REGISTRY = {}      # key: strategy_id -> class（registry='select'）
TIMING_REGISTRY = {}        # key: strategy_id -> class（registry='timing'，A 股择时）
US_TIMING_REGISTRY = {}     # key: strategy_id -> class（registry='us_timing'，美股择时）
COMMODITY_REGISTRY = {}     # key: strategy_id -> class（registry='commodity'，大宗商品择时）

_REGISTRY_MAP = {
    'select': STRATEGY_REGISTRY,
    'timing': TIMING_REGISTRY,
    'us_timing': US_TIMING_REGISTRY,
    'commodity': COMMODITY_REGISTRY,
}


def _register_strategy(cls):
    """供 BaseStrategy / BaseTimingStrategy 的 __init_subclass__ 调用。

    只有同时具备 strategy_id（非空）与 registry（合法值）的子类才会被注册。
    重复注册同一个类视为幂等（pytest --reload 等场景）；不同类抢同一 id 直接 raise。
    """
    sid = getattr(cls, 'strategy_id', None)
    registry = getattr(cls, 'registry', None)
    if not sid or not registry:
        return  # 抽象基类 / 历史遗留类（NasdaqTimingStrategy 等）跳过
    target = _REGISTRY_MAP.get(registry)
    if target is None:
        raise ValueError(
            f"{cls.__name__}.registry={registry!r} 无效，必须是 'select'/'timing'/'us_timing'"
        )
    existing = target.get(sid)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"strategy_id={sid!r} 重复注册：{cls.__name__} vs {existing.__name__}"
        )
    target[sid] = cls
