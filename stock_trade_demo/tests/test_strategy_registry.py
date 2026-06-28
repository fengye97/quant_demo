"""Pillar 1 Step 5 — 策略自动注册回归测试。

覆盖：
  1. 三个 registry 都被 web/state.py import 触发后正确填充
  2. 同一 strategy_id 重复注册（不同类）会 raise
  3. 缺少 strategy_id 或 registry 的抽象类不会被注册
  4. registry 非法值会 raise
  5. changelog_meta 通过 _collect_changelog_meta 汇集到旧 dict
  6. web.state 顶层 STRATEGY_MAP 与 registry 是同一对象（alias）
"""
from __future__ import annotations

import pytest

from strategies.base import (
    BaseStrategy,
    STRATEGY_REGISTRY,
    TIMING_REGISTRY,
    US_TIMING_REGISTRY,
)
from strategies.registry import _register_strategy


def test_registry_populated_after_state_import():
    # 触发 web/state.py 里全部策略 import（同时拉起 timing.strategies）
    from web import state  # noqa: F401
    # A 股选股
    for sid in ('original', 'original_ensemble', 'chan_enhanced', 'chan_only',
                'method_a', 'quality_value', 'sector_heat'):
        assert sid in STRATEGY_REGISTRY, f'select registry missing {sid}'
    # A 股择时
    for sid in ('csi1000_timing', 'star50_timing', 'chinext_timing'):
        assert sid in TIMING_REGISTRY, f'timing registry missing {sid}'
    # 美股择时
    for sid in ('macro_v32_timing', 'sp500_timing'):
        assert sid in US_TIMING_REGISTRY, f'us_timing registry missing {sid}'


def test_duplicate_strategy_id_raises():
    # 借用一个已注册的 strategy_id，造一个不同类，期望 __init_subclass__ 直接 raise
    sid = next(iter(STRATEGY_REGISTRY.keys()))

    with pytest.raises(ValueError, match='重复注册'):
        class _Dup(BaseStrategy):
            strategy_id = sid


def test_abstract_subclass_without_strategy_id_skipped():
    class _Abstract(BaseStrategy):
        # 不声明 strategy_id → registry 跳过
        pass

    # 不应抛异常，也不该出现在任何 registry
    assert '' not in STRATEGY_REGISTRY
    # _Abstract 也不该出现
    assert _Abstract not in STRATEGY_REGISTRY.values()


def test_invalid_registry_name_raises():
    class _Stub:
        strategy_id = '__unit_test_invalid_registry__'
        registry = 'nonexistent_bucket'

    with pytest.raises(ValueError, match='无效'):
        _register_strategy(_Stub)


def test_changelog_meta_collected_into_legacy_dicts():
    from web import state
    # csi1000_timing 在 timing/strategies.py 里挂了 changelog_meta
    assert 'csi1000_timing' in state.TIMING_CHANGELOG_META
    meta = state.TIMING_CHANGELOG_META['csi1000_timing']
    assert meta.get('market_group') == 'csi1000'
    # 美股两个策略都有 changelog_meta
    assert 'macro_v32_timing' in state.US_TIMING_CHANGELOG_META
    assert 'sp500_timing' in state.US_TIMING_CHANGELOG_META


def test_state_maps_are_registry_aliases():
    from web import state
    # 强约束：state.STRATEGY_MAP 必须是同一个 dict 对象（alias），
    # 不是 copy；否则新增策略不会自动出现在 web 层。
    assert state.STRATEGY_MAP is STRATEGY_REGISTRY
    assert state.TIMING_STRATEGY_MAP is TIMING_REGISTRY
    assert state.US_TIMING_STRATEGY_MAP is US_TIMING_REGISTRY
