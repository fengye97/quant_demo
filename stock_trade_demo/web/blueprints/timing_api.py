"""A 股择时 API：/api/timing/info /strategy_list /params /latest_signal /signals /backtest /explore_compare。"""
from __future__ import annotations

import time
import pandas as pd
from flask import Blueprint, request, jsonify

from web import state
from web.params import TimingParams
from web.serializers import (
    DEFAULT_BENCHMARK_ID,
    _normalize_benchmark_id, _get_benchmark_series, _get_benchmark_meta,
    _compute_single_benchmark_curve, _compute_benchmark_curves,
)
from timing import (
    run_timing_backtest, evaluate_timing_result, timing_result_to_json,
    filter_timing_result, summarize_timing_windows,
)

bp = Blueprint('timing_api', __name__)


@bp.route('/api/timing/info')
def api_timing_info():
    try:
        state.ensure_timing_panel_loaded()
    except Exception as e:
        return jsonify({'error': f'指数数据加载失败: {e}'}), 500
    if state.TIMING_PANEL is None or len(state.TIMING_PANEL) == 0:
        return jsonify({'error': '指数数据未加载'}), 500
    max_date = pd.to_datetime(state.TIMING_PANEL['交易日期'].max())
    min_date = pd.to_datetime(state.TIMING_PANEL['交易日期'].min())
    return jsonify({
        'data_min_date': min_date.strftime('%Y-%m-%d'),
        'data_max_date': max_date.strftime('%Y-%m-%d'),
        'indexes': [
            {'id': strategy_id, 'name': strategy_cls().get_display_name(), 'index_name': strategy_cls().get_index_name()}
            for strategy_id, strategy_cls in state.TIMING_STRATEGY_MAP.items()
        ],
    })


@bp.route('/api/timing/strategy_list')
def api_timing_strategy_list():
    try:
        state.init_timing_cache()
    except Exception as e:
        return jsonify({'error': f'择时缓存初始化失败: {e}'}), 500
    payload = []
    for strategy_id in state.TIMING_STRATEGY_MAP.keys():
        strategy = state.build_timing_strategy(strategy_id)
        cached = state.TIMING_CACHE.get(strategy_id)
        cumulative_return = None
        current_action = None
        current_as_of_date = None
        if cached is not None and len(cached) > 0:
            cumulative_return = round(float(cached['累积净值'].iloc[-1]), 2)
            latest_signal = state._build_latest_signal(strategy_id, strategy, cached, state._load_best_profile(strategy_id))
            current_action = latest_signal.get('current_action')
            current_as_of_date = latest_signal.get('as_of_date')
        payload.append({
            'id': strategy_id,
            'name': strategy.get_display_name(),
            'description': strategy.get_strategy_description(),
            'index_name': strategy.get_index_name(),
            'cumulative_return': cumulative_return,
            'current_action': current_action,
            'current_as_of_date': current_as_of_date,
            **state.TIMING_CHANGELOG_META.get(strategy_id, {}),
        })
    return jsonify(payload)


@bp.route('/api/timing/params')
def api_timing_params():
    strategy_name = request.args.get('strategy', 'csi1000_timing')
    strategy = state.build_timing_strategy(strategy_name)
    payload = strategy.get_signal_metadata()
    profile_view = state.get_best_profile_view(strategy_name)
    if profile_view is not None:
        payload['best_profile'] = profile_view
    return jsonify(payload)


@bp.route('/api/timing/latest_signal')
def api_timing_latest_signal():
    strategy_name = request.args.get('strategy', 'csi1000_timing')
    if strategy_name not in state.TIMING_STRATEGY_MAP:
        return jsonify({'error': f'未知策略: {strategy_name}'}), 404
    try:
        state.init_timing_cache()
    except Exception as e:
        return jsonify({'error': f'择时缓存初始化失败: {e}'}), 500
    strategy = state.build_timing_strategy(strategy_name)
    result = state.TIMING_CACHE.get(strategy_name)
    profile = state._load_best_profile(strategy_name)
    return jsonify(state._build_latest_signal(strategy_name, strategy, result, profile))


@bp.route('/api/timing/signals')
def api_timing_signals():
    payload = []
    for strategy_id in state.TIMING_STRATEGY_MAP.keys():
        strategy = state.build_timing_strategy(strategy_id)
        result = state.TIMING_CACHE.get(strategy_id)
        if result is None or len(result) == 0:
            payload.append({
                'id': strategy_id,
                'name': strategy.get_display_name(),
                'index_name': strategy.get_index_name(),
                'date': None,
                'settled_as_of_date': None,
                'action': None,
                'position': 0,
                'reason_summary': '加载中',
                'nav': None,
            })
            continue
        latest_signal = state._build_latest_signal(strategy_id, strategy, result, state._load_best_profile(strategy_id))
        payload.append({
            'id': strategy_id,
            'name': strategy.get_display_name(),
            'index_name': strategy.get_index_name(),
            'date': latest_signal.get('as_of_date'),
            'settled_as_of_date': latest_signal.get('settled_as_of_date'),
            'action': latest_signal.get('current_action'),
            'position': latest_signal.get('current_position', 0),
            'reason_summary': latest_signal.get('reason_summary'),
            'nav': latest_signal.get('settled_nav'),
        })
    return jsonify(payload)


@bp.route('/api/timing/backtest')
def api_timing_backtest():
    strategy_name = request.args.get('strategy', 'csi1000_timing')
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    compact = request.args.get('compact', '0') in {'1', 'true', 'yes'}
    benchmark_id = _normalize_benchmark_id(request.args.get('benchmark', DEFAULT_BENCHMARK_ID))

    # ── 缓存命中判断（参数值与缓存「实际生效」默认值匹配则走缓存）──
    # 注意：缓存由 build_timing_strategy 跑出来，使用 _TIMING_CACHE_DEFAULTS + best_profile.all_params。
    # 前端 /api/timing/params 拿到的 default 来自 strategy 实例属性（同样是 merged 后的值），
    # 所以这里必须用 effective_defaults 比较，否则「前端默认值回放 URL」就会被误判成 cache_miss。
    _t0 = time.time()
    try:
        # 保护6：统一通过 TimingParams 解析 query string
        timing_params = TimingParams.from_query(request.args)
        params = timing_params.to_kwargs()

        use_cache = strategy_name in state.TIMING_CACHE
        if use_cache:
            effective_defaults = state.get_effective_timing_defaults(strategy_name)
            if effective_defaults:
                diffs = timing_params.diff_from_defaults(effective_defaults)
                if diffs:
                    use_cache = False
                    print(f'[timing/backtest] strategy={strategy_name} cache_miss diffs={diffs}')
        print(f'[timing/backtest] strategy={strategy_name} cache_hit={use_cache} compact={compact} params={params}')

        if not use_cache:
            # Pillar 1 Step 6：请求路径只读缓存。任何与缓存默认参数不一致的查询、
            # 或缓存缺失，都明确返回 400 并指向离线脚本，绝不在 Web 进程里现算。
            return jsonify({
                'error': 'cache_miss',
                'strategy': strategy_name,
                'message': f'A 股择时策略 {strategy_name} 的缓存缺失或参数与默认值不一致，请运行离线脚本重建后再查询。',
                'build_script': 'python scripts/build_timing_cache.py',
            }), 400

        _t = time.time()
        result = state.TIMING_CACHE[strategy_name].copy()
        strategy = state.build_timing_strategy(strategy_name)
        print(f'[timing/backtest] cache copy: {(time.time()-_t)*1000:.0f}ms  rows={len(result)}')

        active_benchmark_id, active_benchmark_series = _get_benchmark_series(benchmark_id)
        full_history_start = pd.to_datetime(result['交易日期'].min()) if len(result) else None
        result = filter_timing_result(result, start_date=start_date, end_date=end_date)
        if len(result) == 0:
            return jsonify({'error': '所选日期范围内无数据'}), 400

        _t = time.time()
        metrics = evaluate_timing_result(result, benchmark_returns=active_benchmark_series, reset_capital=True)
        print(f'[timing/backtest] evaluate: {(time.time()-_t)*1000:.0f}ms')

        _t = time.time()
        bm_curve = _compute_single_benchmark_curve(result, active_benchmark_series)
        bm_curves = [] if compact else _compute_benchmark_curves(result, state.INDEX_RETURNS_MAP)
        print(f'[timing/backtest] benchmark curves: {(time.time()-_t)*1000:.0f}ms (compact={compact})')

        _t = time.time()
        payload = timing_result_to_json(
            result,
            metrics,
            benchmark_meta=_get_benchmark_meta(active_benchmark_id),
            benchmark_curve=bm_curve,
            benchmark_curves=bm_curves,
            compact=compact,
        )
        payload['interval_windows'] = summarize_timing_windows(
            result,
            benchmark_returns=active_benchmark_series,
            full_history_start=full_history_start,
        )
        payload['current_signal'] = state._build_latest_signal(strategy_name, strategy, state.TIMING_CACHE[strategy_name], state._load_best_profile(strategy_name))
        print(f'[timing/backtest] to_json: {(time.time()-_t)*1000:.0f}ms')
        print(f'[timing/backtest] total: {(time.time()-_t0)*1000:.0f}ms')
        return jsonify(payload)
    except Exception as e:
        print(f'[timing/backtest] ERROR after {(time.time()-_t0)*1000:.0f}ms: {e}')
        return jsonify({'error': str(e)}), 500


@bp.route('/api/timing/explore_compare')
def api_timing_explore_compare():
    strategy_name = request.args.get('strategy', 'csi1000_timing')
    benchmark_id = _normalize_benchmark_id(request.args.get('benchmark', DEFAULT_BENCHMARK_ID))
    start_date = request.args.get('start')
    end_date = request.args.get('end')

    staged_defaults = {
        'exposure_mode': request.args.get('exposure_mode', 'staged'),
        'enter_threshold': request.args.get('enter_threshold', type=float) or 0.55,
        'add_threshold': request.args.get('add_threshold', type=float) or 0.75,
        'trim_threshold': request.args.get('trim_threshold', type=float) or 0.35,
        'exit_threshold': request.args.get('exit_threshold', type=float) or 0.15,
        'confirm_days': request.args.get('confirm_days', type=int) or 1,
        'max_entry_exposure': request.args.get('max_entry_exposure', type=float) or 0.5,
    }
    for key in ['fast_window', 'slow_window', 'momentum_window', 'breakout_window', 'exit_window',
                'trend_window', 'momentum_short_window', 'momentum_long_window', 'momentum_threshold']:
        val = request.args.get(key, type=float)
        if val is not None:
            staged_defaults[key] = int(val) if key != 'momentum_threshold' else float(val)

    strategy_defaults = dict(state._TIMING_CACHE_DEFAULTS.get(strategy_name, {}))
    strategy_defaults.update({k: v for k, v in staged_defaults.items() if v is not None})

    try:
        # Pillar 1 Step 6：本端点天然依赖非默认参数 fresh-run，与"请求路径只读缓存"
        # 直接冲突；前端目前未使用该接口（grep 0 hits）。保留路由但要求显式 force=1
        # 才会跑离线复算，否则直接 cache_miss 返回。
        if request.args.get('force') not in {'1', 'true', 'yes'}:
            return jsonify({
                'error': 'cache_miss',
                'strategy': strategy_name,
                'message': 'explore_compare 需要双参数现算，已被 load-only 改造禁用；如需调试请加 ?force=1，并仅在离线环境运行。',
                'build_script': 'python scripts/build_timing_cache.py',
            }), 400

        _, benchmark_series = _get_benchmark_series(benchmark_id)
        binary_result, binary_metrics, _ = state.run_timing_backtest_fresh(strategy_name, benchmark_id=benchmark_id)
        staged_result, staged_metrics, _ = state.run_timing_backtest_fresh(strategy_name, benchmark_id=benchmark_id, **strategy_defaults)
        return jsonify({
            'strategy': strategy_name,
            'interval_policy': {
                'windows': ['recent_1m', 'recent_1q', 'recent_6m'],
                'history_bucket': 'pre_6m_history',
                'shared_params': True,
                'reset_capital': True,
            },
            'baseline_binary': {
                'mode': 'binary',
                **state._build_timing_compare_payload(binary_result, binary_metrics, benchmark_series, start_date=start_date, end_date=end_date),
            },
            'candidate_staged': {
                'mode': strategy_defaults.get('exposure_mode', 'staged'),
                **state._build_timing_compare_payload(staged_result, staged_metrics, benchmark_series, params=strategy_defaults, start_date=start_date, end_date=end_date),
            },
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
