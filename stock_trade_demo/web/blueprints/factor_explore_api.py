"""因子研究 API：/api/sector_heat /api/factor_single_backtest。"""
from __future__ import annotations

import sys
from flask import Blueprint, request, jsonify

from web import state

bp = Blueprint('factor_explore_api', __name__)


@bp.route('/api/sector_heat')
def api_sector_heat():
    n_weeks = request.args.get('weeks', 8, type=int)
    df = state._load_sector_heat()
    if df is None:
        return jsonify({
            'error': f'找不到 {state._SECTOR_HEAT_FILE}，请先运行 scripts/compute_sector_weekly_heat.py'
        }), 503

    latest_weeks = sorted(df['week_label'].unique())[-n_weeks:]
    sub = df[df['week_label'].isin(latest_weeks)].copy()

    industries = sorted(sub['industry'].unique(), key=lambda x: (
        sub[sub['industry'] == x]['weekly_ret_pct'].mean()
    ), reverse=True)
    week_list = sorted(latest_weeks)
    ind_idx = {ind: i for i, ind in enumerate(industries)}
    week_idx = {w: j for j, w in enumerate(week_list)}

    # 旧 CSV 没有 is_partial 列时按全 False 处理；同一 week_label 下所有行 is_partial 一致。
    # 关键约束：UI 上只能有“最后一周”标记为 (进行中)。若历史数据里意外残留多个 partial 周，
    # 强制只保留时间上最后一个为 partial，避免横轴出现两个“进行中”。
    has_partial_col = 'is_partial' in sub.columns
    has_ndays_col = 'n_days_in_week' in sub.columns
    partial_map = {}
    ndays_map = {}
    for w in week_list:
        rows = sub[sub['week_label'] == w]
        if has_partial_col and len(rows):
            partial_map[w] = bool(rows['is_partial'].iloc[0])
        else:
            partial_map[w] = False
        if has_ndays_col and len(rows):
            ndays_map[w] = int(rows['n_days_in_week'].iloc[0])
        else:
            ndays_map[w] = 5

    partial_weeks = [w for w in week_list if partial_map.get(w, False)]
    if len(partial_weeks) > 1:
        last_partial = partial_weeks[-1]
        for w in partial_weeks[:-1]:
            partial_map[w] = False
        # 非最后一周一律视作完整周，避免 UI 上重复显示“进行中”
        print(f'[sector_heat] WARN multiple partial weeks found {partial_weeks}; keep only latest {last_partial} as partial', file=sys.stderr)

    heat_data = []
    for _, row in sub.iterrows():
        i = ind_idx.get(row['industry'])
        j = week_idx.get(row['week_label'])
        if i is not None and j is not None:
            heat_data.append([j, i, round(float(row['weekly_ret_pct']), 2)])

    # 近 4 周排名按策略口径：排除 partial 周。partial 不足时退回到所有可用完整周。
    complete_weeks = [w for w in week_list if not partial_map.get(w, False)]
    recent_4w = complete_weeks[-4:] if complete_weeks else week_list[-4:]
    ranking = (
        sub[sub['week_label'].isin(recent_4w)]
        .groupby('industry')['weekly_ret_pct']
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    ranking.columns = ['industry', 'avg_ret_4w']
    ranking['rank'] = range(1, len(ranking) + 1)

    return jsonify({
        'weeks': week_list,
        'weeks_partial': [partial_map.get(w, False) for w in week_list],
        'weeks_n_days': [ndays_map.get(w, 5) for w in week_list],
        'industries': industries,
        'data': heat_data,
        'latest_ranking': ranking.to_dict(orient='records'),
    })


@bp.route('/api/factor_single_backtest')
def api_factor_single_backtest():
    """单因子回测结果只读 API：仅返回离线脚本预生成的缓存，绝不在线计算。"""
    top_k = request.args.get('top_k', 5, type=int)
    cache_key = f'top_k={top_k}'

    if cache_key in state.FACTOR_BACKTEST_CACHE:
        return jsonify(state.FACTOR_BACKTEST_CACHE[cache_key])

    if state._load_factor_backtest_cache() and cache_key in state.FACTOR_BACKTEST_CACHE:
        return jsonify(state.FACTOR_BACKTEST_CACHE[cache_key])

    return jsonify({
        'error': '单因子回测缓存缺失',
        'detail': (
            f'未找到 {state.FACTOR_BACKTEST_CACHE_FILE}（或请求的 top_k={top_k} 不在缓存里）。\n'
            f'web_app 不会在线计算这份回测，请先运行离线脚本生成：\n'
            f'  python3 {state.FACTOR_BACKTEST_BUILD_SCRIPT}\n'
            f'生成后无需重启 Flask，再次刷新即可。'
        ),
        'cache_file': state.FACTOR_BACKTEST_CACHE_FILE,
        'build_script': state.FACTOR_BACKTEST_BUILD_SCRIPT,
    }), 503
