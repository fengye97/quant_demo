"""选股策略相关 API：/api/info /api/strategy_list /api/backtest /api/factors /api/factor_overview /api/status。"""
from __future__ import annotations

import time
import pandas as pd
from flask import Blueprint, request, jsonify

from web import state
from web.serializers import (
    SPLIT_DATE, DEFAULT_BENCHMARK_ID,
    _normalize_benchmark_id, _get_benchmark_series,
    result_to_json,
)
from backtest import strategy_evaluate

bp = Blueprint('select_api', __name__)


@bp.route('/api/status')
def api_status():
    elapsed = 0
    if state._LOAD_STATUS['start_time'] is not None:
        elapsed = time.time() - state._LOAD_STATUS['start_time']
    return jsonify({
        'stage': state._LOAD_STATUS['stage'],
        'message': state._LOAD_STATUS['message'],
        'elapsed_sec': round(elapsed, 1),
        'loading': state._LOAD_STATUS['loading'],
        'ready': state._DATA_READY.is_set(),
    })


@bp.route('/api/info')
def api_info():
    state.ensure_stock_data_loaded()
    if state.DATA_DF is None:
        return jsonify({'error': '数据未加载'}), 500
    max_date = pd.to_datetime(state.DATA_DF['交易日期'].max())
    min_date = pd.to_datetime(state.DATA_DF['交易日期'].min())
    return jsonify({
        'data_min_date': min_date.strftime('%Y-%m-%d'),
        'data_max_date': max_date.strftime('%Y-%m-%d'),
        'training_cutoff': state.TRAINING_CUTOFF,
        'holdout_start': state.HOLDOUT_START,
    })


@bp.route('/api/strategy_list')
def api_strategy_list():
    strategy_id = state.get_focused_strategy_id()
    strategy = state.build_strategy(strategy_id)
    cumulative_return = None
    cached = state.BACKTEST_CACHE.get(strategy_id)
    if cached is not None:
        result, _ = cached
        if len(result) > 0:
            cumulative_return = f"{float(result['累积净值'].iloc[-1]):.2f}x"
    return jsonify([
        {
            'id': strategy_id,
            'name': strategy.get_display_name(),
            'description': strategy.get_strategy_description(),
            'cumulative_return': cumulative_return,
            'best': True,
            'focus_only': True,
        }
    ])


@bp.route('/api/factors')
def api_factors():
    state.ensure_stock_data_loaded()
    strategy_name = request.args.get('strategy', state.get_focused_strategy_id())
    strategy = state.build_strategy(strategy_name)
    return jsonify(strategy.get_factor_metadata())


@bp.route('/api/factor_overview')
def api_factor_overview():
    state.ensure_stock_data_loaded()
    strategy_name = request.args.get('strategy', state.get_focused_strategy_id())
    return jsonify(state.build_factor_overview_payload(strategy_name))


@bp.route('/api/backtest')
def api_backtest():
    state.init_cache()
    strategy = request.args.get('strategy', 'original')
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    compact = request.args.get('compact', '0') in {'1', 'true', 'yes'}
    benchmark_id = _normalize_benchmark_id(request.args.get('benchmark', DEFAULT_BENCHMARK_ID))

    # ── 缓存命中判断（使用 effective_defaults，含 float tolerance）──
    # 与 timing_api.py 同样使用 effective defaults：选股策略目前不走 best_profile，
    # effective 等于裸 _CACHE_DEFAULTS，但走统一通道便于将来扩展，且兼容 float 序列化抖动。
    use_cache = strategy in state.BACKTEST_CACHE
    if use_cache:
        effective_defaults = state.get_effective_select_defaults(strategy)
        if effective_defaults:
            FLOAT_ATOL = 1e-9
            for key, default_val in effective_defaults.items():
                raw_val = request.args.get(key)
                if raw_val is None:
                    continue
                try:
                    if isinstance(default_val, str):
                        val = raw_val
                        if val != default_val:
                            use_cache = False
                            break
                    elif isinstance(default_val, bool):
                        # 当前 select defaults 不含 bool；保留 future-proof 分支
                        from web.state import _parse_realism_bool
                        v = _parse_realism_bool(raw_val)
                        if v is None or v != default_val:
                            use_cache = False
                            break
                    elif isinstance(default_val, int):
                        val = int(float(raw_val))
                        if val != int(default_val):
                            use_cache = False
                            break
                    else:
                        val = float(raw_val)
                        if abs(val - float(default_val)) > FLOAT_ATOL:
                            use_cache = False
                            break
                except (TypeError, ValueError):
                    use_cache = False
                    break

    try:
        if not use_cache:
            # Pillar 1 Step 6：请求路径只读缓存。任何与缓存默认参数不一致的查询、
            # 或缓存缺失，都明确返回 400 并指向离线脚本，绝不在 Web 进程里现算。
            return jsonify({
                'error': 'cache_miss',
                'strategy': strategy,
                'message': f'选股策略 {strategy} 的缓存缺失或参数与默认值不一致，请运行离线脚本重建后再查询。',
                'build_script': 'python scripts/build_select_cache.py',
            }), 400

        result, _ = state.BACKTEST_CACHE[strategy]
        result = result.copy()

        active_benchmark_id, active_benchmark_series = _get_benchmark_series(benchmark_id)

        if start_date or end_date:
            result, ev = state.filter_by_date(result, start_date, end_date, benchmark_id=active_benchmark_id)
            if result is None:
                return jsonify({'error': '所选日期范围内无数据'}), 400
            return jsonify(result_to_json(result, ev, split_date=None, benchmark_id=active_benchmark_id, compact=compact))
        else:
            ev = strategy_evaluate(result, index_returns=active_benchmark_series)
            return jsonify(result_to_json(result, ev, split_date=SPLIT_DATE, benchmark_id=active_benchmark_id, compact=compact))
    except Exception as e:
        return jsonify({'error': str(e)}), 500
