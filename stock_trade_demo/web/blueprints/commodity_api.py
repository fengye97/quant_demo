"""大宗商品择时 API：/api/commodity/info /strategy_list /params /latest_signal /signals /backtest。"""
from __future__ import annotations

import time
import pandas as pd
from flask import Blueprint, request, jsonify

from web import state
from web.params import TimingParams
from timing import (
    run_timing_backtest, evaluate_timing_result, timing_result_to_json,
    filter_timing_result, summarize_timing_windows,
)

bp = Blueprint('commodity_api', __name__)


@bp.route('/api/commodity/info')
def api_commodity_info():
    try:
        state.ensure_commodity_panel_loaded()
    except Exception as e:
        return jsonify({'error': f'大宗商品数据加载失败: {e}'}), 500
    if state.COMMODITY_PANEL is None or len(state.COMMODITY_PANEL) == 0:
        return jsonify({'error': '大宗商品数据未加载'}), 500
    max_date = pd.to_datetime(state.COMMODITY_PANEL['交易日期'].max())
    min_date = pd.to_datetime(state.COMMODITY_PANEL['交易日期'].min())
    return jsonify({
        'data_min_date': min_date.strftime('%Y-%m-%d'),
        'data_max_date': max_date.strftime('%Y-%m-%d'),
        'indexes': [
            {'id': strategy_id, 'name': strategy_cls().get_display_name(), 'index_name': strategy_cls().get_index_name()}
            for strategy_id, strategy_cls in state.COMMODITY_STRATEGY_MAP.items()
        ],
    })


@bp.route('/api/commodity/strategy_list')
def api_commodity_strategy_list():
    try:
        state.init_commodity_cache()
    except Exception as e:
        return jsonify({'error': f'大宗商品择时缓存初始化失败: {e}'}), 500
    payload = []
    for strategy_id in state.COMMODITY_STRATEGY_MAP.keys():
        strategy = state.build_commodity_strategy(strategy_id)
        params = strategy.get_parameter_definitions()
        shared_params = strategy.get_shared_parameter_definitions()
        payload.append({
            'id': strategy_id,
            'name': strategy.get_display_name(),
            'description': strategy.get_strategy_description(),
            'index_name': strategy.get_index_name(),
            'principle_summary': strategy.get_principle_summary(),
            'formula_blocks': strategy.get_formula_blocks(),
            'parameters': params,
            'shared_parameters': shared_params,
        })
    return jsonify(payload)


@bp.route('/api/commodity/params')
def api_commodity_params():
    strategy_name = request.args.get('strategy', 'gold_timing')
    if strategy_name not in state.COMMODITY_STRATEGY_MAP:
        return jsonify({'error': f'未知策略: {strategy_name}'}), 404
    strategy = state.build_commodity_strategy(strategy_name)
    payload = strategy.get_signal_metadata()
    profile_view = state.get_best_profile_view(strategy_name)
    if profile_view is not None:
        payload['best_profile'] = profile_view
    return jsonify(payload)


@bp.route('/api/commodity/latest_signal')
def api_commodity_latest_signal():
    strategy_name = request.args.get('strategy', 'gold_timing')
    if strategy_name not in state.COMMODITY_STRATEGY_MAP:
        return jsonify({'error': f'未知策略: {strategy_name}'}), 404
    try:
        state.init_commodity_cache()
    except Exception as e:
        return jsonify({'error': f'大宗商品择时缓存初始化失败: {e}'}), 500
    strategy = state.build_commodity_strategy(strategy_name)
    result = state.COMMODITY_CACHE.get(strategy_name)
    profile = state._load_best_profile(strategy_name)
    return jsonify(state._build_latest_signal(strategy_name, strategy, result, profile))


@bp.route('/api/commodity/signals')
def api_commodity_signals():
    payload = []
    for strategy_id in state.COMMODITY_STRATEGY_MAP.keys():
        strategy = state.build_commodity_strategy(strategy_id)
        result = state.COMMODITY_CACHE.get(strategy_id)
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
            'action': latest_signal.get('current_action', 'hold'),
            'position': latest_signal.get('current_position', 0),
            'reason_summary': latest_signal.get('current_reason', ''),
            'nav': latest_signal.get('nav'),
        })
    return jsonify(payload)


@bp.route('/api/commodity/backtest')
def api_commodity_backtest():
    strategy_name = request.args.get('strategy', 'gold_timing')
    if strategy_name not in state.COMMODITY_STRATEGY_MAP:
        return jsonify({'error': f'未知策略: {strategy_name}'}), 404
    try:
        state.init_commodity_cache()
    except Exception as e:
        return jsonify({'error': f'大宗商品择时缓存初始化失败: {e}'}), 500
    strategy = state.build_commodity_strategy(strategy_name)
    result = state.COMMODITY_CACHE.get(strategy_name)
    if result is None:
        return jsonify({'error': f'策略 {strategy_name} 回测结果未就绪'}), 500

    result = result.copy()
    start_date = request.args.get('start')
    end_date = request.args.get('end')

    filtered = filter_timing_result(result, start_date=start_date, end_date=end_date)
    if len(filtered) == 0:
        return jsonify({'error': '所选日期范围内无数据'}), 400

    metrics = evaluate_timing_result(filtered, reset_capital=True)
    windows = summarize_timing_windows(filtered)

    payload = timing_result_to_json(filtered, metrics, compact=False)
    payload['interval_windows'] = windows
    payload['current_signal'] = state._build_latest_signal(strategy_name, strategy, result, state._load_best_profile(strategy_name))
    return jsonify(payload)
