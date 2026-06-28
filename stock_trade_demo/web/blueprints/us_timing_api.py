"""美股择时 API：/api/us_timing/info /strategy_list /params /latest_signal /backtest。"""
from __future__ import annotations

import time
import pandas as pd
from flask import Blueprint, request, jsonify

from web import state
from web.params import TimingParams
from web.serializers import _build_etf_monthly_returns
from timing import (
    run_timing_backtest, evaluate_timing_result, timing_result_to_json,
    filter_timing_result, summarize_timing_windows,
)

bp = Blueprint('us_timing_api', __name__)


@bp.route('/api/us_timing/info')
def api_us_timing_info():
    try:
        state.ensure_us_timing_panel_loaded()
    except Exception as e:
        return jsonify({'error': f'美股指数数据加载失败: {e}'}), 500
    if state.US_TIMING_PANEL is None or len(state.US_TIMING_PANEL) == 0:
        return jsonify({'error': '美股指数数据未加载'}), 500
    max_date = pd.to_datetime(state.US_TIMING_PANEL['交易日期'].max())
    min_date = pd.to_datetime(state.US_TIMING_PANEL['交易日期'].min())
    return jsonify({
        'data_min_date': min_date.strftime('%Y-%m-%d'),
        'data_max_date': max_date.strftime('%Y-%m-%d'),
        'indexes': [
            {'id': strategy_id, 'name': strategy_cls().get_display_name(), 'index_name': strategy_cls().get_index_name()}
            for strategy_id, strategy_cls in state.US_TIMING_STRATEGY_MAP.items()
        ],
    })


@bp.route('/api/us_timing/strategy_list')
def api_us_timing_strategy_list():
    try:
        state.ensure_us_timing_panel_loaded()
    except Exception as e:
        return jsonify({'error': f'美股指数数据加载失败: {e}'}), 500
    data_max_date = None
    if state.US_TIMING_PANEL is not None and len(state.US_TIMING_PANEL) > 0:
        data_max_date = pd.to_datetime(state.US_TIMING_PANEL['交易日期'].max()).strftime('%Y-%m-%d')

    def _build_perf_delta(current_id, baseline_id=None):
        if not baseline_id:
            return None
        current = state.US_TIMING_CACHE.get(current_id)
        baseline = state.US_TIMING_CACHE.get(baseline_id)
        if current is None or baseline is None or len(current) == 0 or len(baseline) == 0:
            return None

        def _last(df, col):
            if col not in df.columns or len(df[col]) == 0:
                return None
            val = df[col].iloc[-1]
            return float(val) if pd.notna(val) else None

        current_nav = _last(current, '累积净值')
        baseline_nav = _last(baseline, '累积净值')
        current_capital = _last(current, '总资金')
        baseline_capital = _last(baseline, '总资金')
        current_mdd = current.attrs.get('metrics', {}).get('最大回撤')
        baseline_mdd = baseline.attrs.get('metrics', {}).get('最大回撤')
        current_annual = current.attrs.get('metrics', {}).get('年化收益')
        baseline_annual = baseline.attrs.get('metrics', {}).get('年化收益')

        payload = {}
        if current_nav is not None and baseline_nav is not None:
            payload['cumulative_return_diff'] = round(current_nav - baseline_nav, 4)
        if current_capital is not None and baseline_capital is not None:
            payload['final_capital_diff'] = round(current_capital - baseline_capital, 2)
        if current_mdd is not None and baseline_mdd is not None:
            payload['max_drawdown_improvement'] = round(float(current_mdd) - float(baseline_mdd), 2)
        if current_annual is not None and baseline_annual is not None:
            payload['annual_return_diff'] = round(float(current_annual) - float(baseline_annual), 2)
        payload['baseline_strategy'] = baseline_id
        return payload or None

    try:
        state.init_us_timing_cache()
    except Exception as e:
        return jsonify({'error': f'美股择时缓存初始化失败: {e}'}), 500

    payload = []
    for strategy_id in state.US_TIMING_PAGE_STRATEGY_IDS:
        strategy = state.build_us_timing_strategy(strategy_id)
        cached = state.US_TIMING_CACHE.get(strategy_id)
        cumulative_return = None
        current_action = None
        current_as_of_date = None
        total_return_pct = None
        annual_return = None
        max_drawdown = None
        if cached is not None and len(cached) > 0:
            cumulative_return = round(float(cached['累积净值'].iloc[-1]), 4)
            latest_signal = state._build_latest_signal(strategy_id, strategy, cached, state._load_best_profile(strategy_id))
            current_action = latest_signal.get('current_action')
            current_as_of_date = latest_signal.get('as_of_date')
            metrics = cached.attrs.get('metrics', {})
            total_return_pct = metrics.get('总收益率')
            annual_return = metrics.get('年化收益')
            max_drawdown = metrics.get('最大回撤')

        changelog_meta = dict(state.US_TIMING_CHANGELOG_META.get(strategy_id, {}))
        perf_delta = _build_perf_delta(strategy_id, changelog_meta.get('supersedes'))
        payload.append({
            'id': strategy_id,
            'name': strategy.get_display_name(),
            'description': strategy.get_strategy_description(),
            'index_name': strategy.get_index_name(),
            'cumulative_return': cumulative_return,
            'current_action': current_action,
            'current_as_of_date': current_as_of_date,
            'total_return_pct': total_return_pct,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'data_max_date': data_max_date,
            'is_page_winner': True,
            **changelog_meta,
            'performance_delta': perf_delta,
        })
    return jsonify(payload)


@bp.route('/api/us_timing/params')
def api_us_timing_params():
    strategy_name = request.args.get('strategy', 'macro_v32_timing')
    strategy = state.build_us_timing_strategy(strategy_name)
    payload = strategy.get_signal_metadata()
    profile_view = state.get_best_profile_view(strategy_name)
    if profile_view is not None:
        payload['best_profile'] = profile_view
    return jsonify(payload)


@bp.route('/api/us_timing/latest_signal')
def api_us_timing_latest_signal():
    strategy_name = request.args.get('strategy', 'macro_v32_timing')
    if strategy_name not in state.US_TIMING_STRATEGY_MAP:
        return jsonify({'error': f'未知策略: {strategy_name}'}), 404
    try:
        state.init_us_timing_cache()
    except Exception as e:
        return jsonify({'error': f'美股择时缓存初始化失败: {e}'}), 500
    strategy = state.build_us_timing_strategy(strategy_name)
    result = state.US_TIMING_CACHE.get(strategy_name)
    profile = state._load_best_profile(strategy_name)
    return jsonify(state._build_latest_signal(strategy_name, strategy, result, profile))


@bp.route('/api/us_timing/backtest')
def api_us_timing_backtest():
    strategy_name = request.args.get('strategy', 'macro_v32_timing')
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    compact = request.args.get('compact', '0') in {'1', 'true', 'yes'}

    _t0 = time.time()
    try:
        # 保护6：统一通过 TimingParams 解析 query string
        timing_params = TimingParams.from_query(request.args)
        params = timing_params.to_kwargs()

        # 美股 timing cache 是 lazy load 的：确保已从 .cache/us_timing/*.pkl 读入
        try:
            state.init_us_timing_cache()
        except Exception as e:
            print(f'[us_timing/backtest] init_us_timing_cache failed: {e}')

        # ── 缓存命中判断（用 effective_defaults，覆盖 best_profile.all_params） ──
        # 见 timing_api.py 同段注释。
        use_cache = strategy_name in state.US_TIMING_CACHE
        if use_cache:
            effective_defaults = state.get_effective_us_timing_defaults(strategy_name)
            if effective_defaults:
                diffs = timing_params.diff_from_defaults(effective_defaults)
                if diffs:
                    use_cache = False
                    print(f'[us_timing/backtest] strategy={strategy_name} cache_miss diffs={diffs}')

        if not use_cache:
            # Pillar 1 Step 6：请求路径只读缓存。任何与缓存默认参数不一致的查询、
            # 或缓存缺失，都明确返回 400 并指向离线脚本，绝不在 Web 进程里现算。
            return jsonify({
                'error': 'cache_miss',
                'strategy': strategy_name,
                'message': f'美股择时策略 {strategy_name} 的缓存缺失或参数与默认值不一致，请运行离线脚本重建后再查询。',
                'build_script': 'python scripts/build_us_timing_cache.py',
            }), 400

        result = state.US_TIMING_CACHE[strategy_name].copy()
        strategy = state.build_us_timing_strategy(strategy_name)

        full_history_start = pd.to_datetime(result['交易日期'].min()) if len(result) else None
        result = filter_timing_result(result, start_date=start_date, end_date=end_date)
        if len(result) == 0:
            return jsonify({'error': '所选日期范围内无数据'}), 400

        etf_benchmark_returns = _build_etf_monthly_returns(result)
        metrics = evaluate_timing_result(result, benchmark_returns=etf_benchmark_returns, reset_capital=True)
        bm_curve = []
        payload = timing_result_to_json(result, metrics, benchmark_curve=bm_curve, compact=compact)
        payload['interval_windows'] = summarize_timing_windows(
            result,
            benchmark_returns=etf_benchmark_returns,
            full_history_start=full_history_start,
        )
        payload['current_signal'] = state._build_latest_signal(strategy_name, strategy, state.US_TIMING_CACHE[strategy_name], state._load_best_profile(strategy_name))
        print(f'[us_timing/backtest] strategy={strategy_name} cache_hit={use_cache} total={(time.time()-_t0)*1000:.0f}ms')
        return jsonify(payload)
    except Exception as e:
        print(f'[us_timing/backtest] ERROR: {e}')
        return jsonify({'error': str(e)}), 500
