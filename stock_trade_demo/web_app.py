"""
量化策略 Web 可视化 — Flask 后端。

启动:
  python3 web_app.py
  然后访问 http://localhost:8080

架构:
  - 启动时预加载数据并预运行回测（缓存全量结果）
  - API 请求时按日期范围过滤缓存的回测结果
  - 因子参数变化时重新运行回测
  - 训练集/测试集拆分: 训练集 ≤ 2026-02-28, 测试集 > 2026-02-28
"""

import os
import sys
import json
import warnings
import inspect
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategies.original import OriginalStrategy
from strategies.original_ensemble import OriginalEnsembleStrategy
from strategies.chan_enhanced import ChanEnhancedStrategy
from strategies.chan_only import ChanOnlyStrategy
from strategies.method_a import MethodAStrategy
from strategies.quality_value import QualityValueStrategy
from backtest import load_data, select_and_backtest, strategy_evaluate, compute_alpha_beta
from index_data import INDEX_CONFIGS, get_index_returns, build_index_panel, build_period_lookup, get_index_return_for_date
from timing import (
    CSI1000TimingStrategy,
    Star50TimingStrategy,
    ChiNextTimingStrategy,
    run_timing_backtest,
    evaluate_timing_result,
    timing_result_to_json,
)

warnings.filterwarnings('ignore')
pd.set_option('expand_frame_repr', False)

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), 'web', 'templates'))

# 全局缓存
DATA_DF = None
INDEX_RETURNS = None  # CSI 1000 月度收益 Series（用于归因主基准）
INDEX_RETURNS_MAP = {}  # key: index_id -> monthly returns series
BACKTEST_CACHE = {}  # key: strategy_name → (result_df, eval_df)
TIMING_PANEL = None
TIMING_CACHE = {}  # key: strategy_name -> result_df

STRATEGY_MAP = {
    'original': OriginalStrategy,
    'original_ensemble': OriginalEnsembleStrategy,
    'chan_enhanced': ChanEnhancedStrategy,
    'chan_only': ChanOnlyStrategy,
    'method_a': MethodAStrategy,
    'quality_value': QualityValueStrategy,
}

TIMING_STRATEGY_MAP = {
    'csi1000_timing': CSI1000TimingStrategy,
    'star50_timing': Star50TimingStrategy,
    'chinext_timing': ChiNextTimingStrategy,
}

FACTOR_OVERVIEW = [
    {
        'name': '市场因子',
        'core_fields': '市场组合收益率、无风险收益率',
        'sort_direction': '不适用',
        'long_short': '不适用',
        'double_sort': '不适用',
        'book_recommended': '是',
        'category': '风险归因',
    },
    {
        'name': '规模因子',
        'core_fields': '总市值',
        'sort_direction': '从低到高',
        'long_short': 'Small - Big',
        'double_sort': '否',
        'book_recommended': '是',
        'category': '核心选股',
    },
    {
        'name': '价值因子',
        'core_fields': 'BM',
        'sort_direction': '从低到高',
        'long_short': 'High - Low',
        'double_sort': '是',
        'book_recommended': '是',
        'category': '核心选股',
    },
    {
        'name': '动量因子',
        'core_fields': '过去 11 个月累计收益',
        'sort_direction': '从低到高',
        'long_short': 'High - Low',
        'double_sort': '是',
        'book_recommended': '否，A 股不稳',
        'category': '核心选股',
    },
    {
        'name': '盈利因子',
        'core_fields': 'ROE(TTM)',
        'sort_direction': '从低到高',
        'long_short': 'High - Low',
        'double_sort': '是',
        'book_recommended': '是',
        'category': '核心选股',
    },
    {
        'name': '投资因子',
        'core_fields': '总资产同比增长率',
        'sort_direction': '从低到高',
        'long_short': 'Low - High',
        'double_sort': '是，但仍受污染',
        'book_recommended': '否，证据较弱',
        'category': '核心选股',
    },
    {
        'name': '换手率因子',
        'core_fields': '异常换手率',
        'sort_direction': '从低到高',
        'long_short': 'Low - High',
        'double_sort': '是',
        'book_recommended': '是，A 股很强',
        'category': '交易行为',
    },
    {
        'name': '缠论背驰因子',
        'core_fields': '收盘价、MACD柱',
        'sort_direction': '从低到高',
        'long_short': 'High - Low',
        'double_sort': '是',
        'book_recommended': '否，缠论扩展',
        'category': '缠论扩展',
    },
    {
        'name': '缠论中枢位置因子',
        'core_fields': '最高价、最低价、收盘价',
        'sort_direction': '从低到高',
        'long_short': 'High - Low',
        'double_sort': '是',
        'book_recommended': '否，缠论扩展',
        'category': '缠论扩展',
    },
    {
        'name': '缠论分型因子',
        'core_fields': '最高价、最低价',
        'sort_direction': '从低到高',
        'long_short': 'High - Low',
        'double_sort': '是',
        'book_recommended': '否，缠论扩展',
        'category': '缠论扩展',
    },
    {
        'name': '缠论笔强度因子',
        'core_fields': '收盘价、涨跌幅_20',
        'sort_direction': '从低到高',
        'long_short': 'High - Low',
        'double_sort': '是',
        'book_recommended': '否，缠论扩展',
        'category': '缠论扩展',
    },
    {
        'name': '缠论买卖点信号因子',
        'core_fields': '收盘价、最高价、最低价、MACD',
        'sort_direction': '从低到高',
        'long_short': 'High - Low',
        'double_sort': '是',
        'book_recommended': '否，缠论扩展',
        'category': '缠论扩展',
    },
]

FOCUSED_STRATEGY_ID = 'original_ensemble'

# 训练/测试集拆分日期
SPLIT_DATE = pd.to_datetime('2026-03-31')
DEFAULT_BENCHMARK_ID = 'csi1000'


def _normalize_benchmark_id(benchmark_id):
    if benchmark_id in INDEX_CONFIGS:
        return benchmark_id
    if DEFAULT_BENCHMARK_ID in INDEX_CONFIGS:
        return DEFAULT_BENCHMARK_ID
    return next(iter(INDEX_CONFIGS.keys()), None)


def _get_benchmark_series(benchmark_id):
    normalized_id = _normalize_benchmark_id(benchmark_id)
    if normalized_id is None:
        return None, None
    series = INDEX_RETURNS_MAP.get(normalized_id)
    if series is not None:
        return normalized_id, series
    fallback_id = DEFAULT_BENCHMARK_ID if DEFAULT_BENCHMARK_ID in INDEX_RETURNS_MAP else None
    if fallback_id is not None:
        return fallback_id, INDEX_RETURNS_MAP[fallback_id]
    if INDEX_RETURNS_MAP:
        first_id = next(iter(INDEX_RETURNS_MAP.keys()))
        return first_id, INDEX_RETURNS_MAP[first_id]
    return normalized_id, None


def _get_benchmark_meta(benchmark_id):
    normalized_id = _normalize_benchmark_id(benchmark_id)
    if normalized_id is None:
        return None
    cfg = INDEX_CONFIGS.get(normalized_id, {})
    return {
        'id': normalized_id,
        'name': cfg.get('name', normalized_id),
    }


def _infer_market_label(code):
    code = str(code or '').strip()
    digits = ''.join(ch for ch in code if ch.isdigit())
    if digits.startswith(('688', '689')):
        return '科创板'
    if digits.startswith(('300', '301')):
        return '创业板'
    if digits.startswith(('600', '601', '603', '605')):
        return '上证主板'
    if digits.startswith(('000', '001', '002', '003')):
        return '深证主板'
    return '其他'


def ensure_index_returns_loaded():
    global INDEX_RETURNS, INDEX_RETURNS_MAP
    if INDEX_RETURNS_MAP:
        return
    index_returns_map = {}
    for index_id, cfg in INDEX_CONFIGS.items():
        try:
            series = get_index_returns(index_id=index_id)
            index_returns_map[index_id] = series
            print(f"[init] {cfg['name']} 指数收益加载完成，{len(series)} 个月")
        except Exception as e:
            print(f"[WARN] 无法加载 {cfg['name']} 指数收益: {e}")
    INDEX_RETURNS_MAP = index_returns_map
    INDEX_RETURNS = INDEX_RETURNS_MAP.get('csi1000')
    if INDEX_RETURNS is None:
        print('[WARN] CSI 1000 不可用，业绩归因功能将不可用')


def ensure_stock_data_loaded():
    global DATA_DF
    if DATA_DF is not None:
        return
    csv_path = os.path.join(os.path.dirname(__file__), 'stock_data.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'数据文件不存在: {csv_path}')
    print('[init] 加载数据中 (823MB)...')
    DATA_DF = load_data(csv_path)
    print('[init] 数据加载完成')


def init_cache():
    """按需预热选股缓存。"""
    ensure_index_returns_loaded()
    ensure_stock_data_loaded()
    if BACKTEST_CACHE:
        return

    for sid, cls in [('original', OriginalStrategy),
                     ('original_ensemble', OriginalEnsembleStrategy),
                     ('chan_enhanced', ChanEnhancedStrategy),
                     ('chan_only', ChanOnlyStrategy),
                     ('method_a', MethodAStrategy),
                     ('quality_value', QualityValueStrategy)]:
        try:
            print(f"[init] 预运行 {sid} 策略...")
            s = cls()
            df = s.run(DATA_DF.copy())
            result = select_and_backtest(df, s,
                                         c_rate=s.c_rate,
                                         t_rate=s.t_rate,
                                         bull_tp=s.bull_tp,
                                         bear_tp=s.bear_tp,
                                         bull_n=s.bull_n,
                                         bear_n=s.bear_n,
                                         initial_capital=s.initial_capital)
            if hasattr(s, '_profile_summary'):
                result.attrs['strategy_meta'] = {
                    'profile_summary': getattr(s, '_profile_summary', []),
                }
            ev = strategy_evaluate(result, index_returns=INDEX_RETURNS)
            BACKTEST_CACHE[sid] = (result, ev)
            print(f"[init] {sid} 完成, 累积净值: {result['累积净值'].iloc[-1]:.2f}")
        except Exception as e:
            print(f"[init] {sid} 失败: {e}")


def ensure_timing_panel_loaded():
    global TIMING_PANEL
    ensure_index_returns_loaded()
    if TIMING_PANEL is not None:
        return
    try:
        TIMING_PANEL = build_index_panel()
    except Exception as e:
        print(f"[WARN] 无法加载指数日线面板: {e}")
        TIMING_PANEL = None
        raise


def init_timing_cache():
    ensure_timing_panel_loaded()
    global TIMING_CACHE
    if TIMING_CACHE:
        return

    TIMING_CACHE = {}
    for sid, cls in TIMING_STRATEGY_MAP.items():
        try:
            strategy = cls()
            signal_df = strategy.run(TIMING_PANEL.copy())
            result = run_timing_backtest(signal_df, strategy, benchmark_returns=INDEX_RETURNS_MAP.get(strategy.get_index_id()))
            TIMING_CACHE[sid] = result
            print(f"[init] {sid} 择时策略完成, 累积净值: {result['累积净值'].iloc[-1]:.2f}")
        except Exception as e:
            print(f"[init] {sid} 择时策略失败: {e}")


def build_strategy(strategy_name='original', **params):
    strat_cls = STRATEGY_MAP.get(strategy_name, OriginalStrategy)
    sig = inspect.signature(strat_cls.__init__)
    valid_params = {k: v for k, v in params.items()
                    if k in sig.parameters and v is not None}
    valid_params.pop('self', None)
    return strat_cls(**valid_params)


def build_timing_strategy(strategy_name='csi1000_timing', **params):
    strat_cls = TIMING_STRATEGY_MAP.get(strategy_name, CSI1000TimingStrategy)
    sig = inspect.signature(strat_cls.__init__)
    valid_params = {k: v for k, v in params.items()
                    if k in sig.parameters and v is not None}
    valid_params.pop('self', None)
    return strat_cls(**valid_params)


def run_timing_backtest_fresh(strategy_name='csi1000_timing', benchmark_id=None, **params):
    ensure_timing_panel_loaded()
    strategy = build_timing_strategy(strategy_name, **params)
    _, benchmark_series = _get_benchmark_series(benchmark_id)
    signal_df = strategy.run(TIMING_PANEL.copy())
    result = run_timing_backtest(signal_df, strategy, benchmark_returns=benchmark_series)
    metrics = evaluate_timing_result(result, benchmark_returns=benchmark_series)
    return result, metrics, strategy


def get_focused_strategy_id():
    return FOCUSED_STRATEGY_ID if FOCUSED_STRATEGY_ID in STRATEGY_MAP else 'original'


def build_factor_overview_payload(strategy_name=None):
    strategy_id = strategy_name or get_focused_strategy_id()
    strategy = build_strategy(strategy_id)
    if hasattr(strategy, '_resolve_profiles') and DATA_DF is not None and not getattr(strategy, '_profile_summary', None):
        try:
            strategy._resolve_profiles(DATA_DF.copy())
        except Exception:
            pass
    active_tags = set(strategy.get_factor_overview_tags() or [])
    items = []
    for row in FACTOR_OVERVIEW:
        item = dict(row)
        item['active'] = item['name'] in active_tags
        items.append(item)
    payload = {
        'strategy_id': strategy_id,
        'strategy_name': strategy.get_display_name(),
        'active_factor_names': sorted(active_tags),
        'items': items,
    }
    if hasattr(strategy, '_profile_summary'):
        payload['profile_summary'] = getattr(strategy, '_profile_summary', [])
    return payload


def run_backtest_fresh(strategy_name='original', benchmark_id=None, **params):
    """重新运行回测（参数不同于默认值时使用）。返回: (result_df, eval_df)"""
    if DATA_DF is None:
        raise RuntimeError("数据未加载，请确认 stock_data.csv 存在")

    strategy = build_strategy(strategy_name, **params)
    _, benchmark_series = _get_benchmark_series(benchmark_id)

    df = strategy.run(DATA_DF.copy())
    result = select_and_backtest(df, strategy,
                                 c_rate=strategy.c_rate,
                                 t_rate=strategy.t_rate,
                                 bull_tp=strategy.bull_tp,
                                 bear_tp=strategy.bear_tp,
                                 bull_n=strategy.bull_n,
                                 bear_n=strategy.bear_n,
                                 initial_capital=strategy.initial_capital)
    if hasattr(strategy, '_profile_summary'):
        result.attrs['strategy_meta'] = {
            'profile_summary': getattr(strategy, '_profile_summary', []),
        }
    ev = strategy_evaluate(result, index_returns=benchmark_series)
    return result, ev



def filter_by_date(result, start_date, end_date, benchmark_id=None):
    """按日期范围过滤回测结果并重新计算累积净值和评估指标。
    初始资金始终重置为 100,000——自定义日期范围视为独立回测区间。"""
    original = result.copy()
    original_dates = list(original['交易日期'])
    _, benchmark_series = _get_benchmark_series(benchmark_id)
    if start_date:
        result = result[result['交易日期'] >= pd.to_datetime(start_date)].copy()
    if end_date:
        result = result[result['交易日期'] <= pd.to_datetime(end_date)].copy()
    if len(result) == 0:
        return None, None

    # 重新计算累积净值
    result['累积净值'] = (1 + result['选股下周期涨跌幅']).cumprod()
    result['资金曲线'] = result['累积净值']

    # 自定义日期范围：初始资金重置为 100,000
    start_capital = float(result.attrs.get('initial_capital', 100000))

    capital = float(start_capital)
    capitals = []
    pnls = []
    cum_caps = []
    for _, row in result.iterrows():
        capitals.append(capital)
        pnl = capital * row['选股下周期涨跌幅']
        pnls.append(pnl)
        capital += pnl
        cum_caps.append(capital)

    result['当期本金'] = capitals
    result['当期盈亏'] = pnls
    result['累计资金'] = cum_caps
    result.attrs['initial_capital'] = start_capital

    if 'period_daily_curves' in original.attrs:
        curve_lookup = {
            pd.to_datetime(dt).strftime('%Y-%m-%d'): curve
            for dt, curve in zip(original_dates, original.attrs.get('period_daily_curves', []))
        }
        result.attrs['period_daily_curves'] = [
            curve_lookup.get(pd.to_datetime(dt).strftime('%Y-%m-%d'), [])
            for dt in result['交易日期']
        ]
    if 'daily_equity_curve' in original.attrs:
        result.attrs['daily_equity_curve'] = _build_daily_curve_slice(result, original.attrs.get('daily_equity_curve', []))

    ev = strategy_evaluate(result, initial_capital=start_capital,
                          index_returns=benchmark_series)
    return result, ev


def _build_daily_curve_slice(result_df, full_daily_curve, base_value=1.0):
    """按结果区间切分并重置日线净值基准。"""
    if not full_daily_curve or len(result_df) == 0:
        return []

    period_curves = result_df.attrs.get('period_daily_curves', [])
    if period_curves:
        daily_curve = []
        running_value = float(base_value)
        for period_idx, period_curve in enumerate(period_curves):
            trade_date = pd.to_datetime(result_df.iloc[period_idx]['交易日期'])
            if not period_curve:
                daily_curve.append({
                    'date': trade_date.strftime('%Y-%m-%d'),
                    'value': round(running_value, 6),
                    'return': 0.0,
                })
                continue
            prev = 1.0
            for day_idx, period_value in enumerate(period_curve, start=1):
                day_ret = 0.0 if prev == 0 else float(period_value / prev - 1)
                running_value *= (1 + day_ret)
                daily_curve.append({
                    'date': (trade_date + pd.Timedelta(days=day_idx)).strftime('%Y-%m-%d'),
                    'value': round(running_value, 6),
                    'return': round(day_ret, 6),
                })
                prev = period_value
        return daily_curve

    first_value = float(full_daily_curve[0].get('value', base_value))
    if first_value == 0:
        first_value = base_value

    return [{
        'date': p.get('date'),
        'value': round(float(p.get('value', 1.0)) / first_value * base_value, 6),
        'return': round(float(p.get('return', 0.0)), 6),
    } for p in full_daily_curve]


def _build_stock_payload(raw_stock, cap_per_stock, period_capital, stock_count):
    ret = raw_stock.get('return', 0)
    buy_price = raw_stock.get('buy_price', 0)
    sell_price = raw_stock.get('sell_price', None)
    code = raw_stock.get('code', '')
    if buy_price > 0:
        shares = int(cap_per_stock / buy_price / 100) * 100
        actual_invested = shares * buy_price
    else:
        shares = 0
        actual_invested = cap_per_stock
    pnl = round(float(actual_invested * ret), 2) if actual_invested > 0 else round(float(cap_per_stock * ret), 2)
    latest_price = sell_price if sell_price is not None else buy_price
    position_market_value = round(float(shares * latest_price), 2) if shares > 0 and latest_price is not None else None
    position_weight = round(position_market_value / period_capital, 6) if position_market_value is not None and period_capital > 0 else raw_stock.get('weight', round(1.0 / stock_count, 4))
    return {
        'code': code,
        'name': raw_stock.get('name', ''),
        'weight': raw_stock.get('weight', round(1.0 / stock_count, 4)),
        'position_weight': position_weight,
        'return': ret,
        'pnl': pnl,
        'buy_price': buy_price,
        'sell_price': sell_price,
        'shares': shares,
        'position_market_value': position_market_value,
        'factor_score': raw_stock.get('factor_score'),
        'rank': raw_stock.get('rank'),
        'industry_l2': raw_stock.get('industry_l2', ''),
        'market_label': raw_stock.get('market_label') or _infer_market_label(code),
        'pe': raw_stock.get('pe'),
        'pb': raw_stock.get('pb'),
        'market_cap': raw_stock.get('market_cap'),
        'selection_reason_summary': raw_stock.get('selection_reason_summary', ''),
        'selection_reason_detail': raw_stock.get('selection_reason_detail', []),
        'selection_fundamentals': raw_stock.get('selection_fundamentals', []),
        'selection_factor_breakdown': raw_stock.get('selection_factor_breakdown', []),
    }


def _compute_single_benchmark_curve(result, index_returns):
    if index_returns is None:
        return []
    lookup = build_period_lookup(index_returns)
    bm = []
    cum = 1.0
    for _, row in result.iterrows():
        idx_ret = get_index_return_for_date(row['交易日期'], lookup)
        cum *= (1 + idx_ret)
        bm.append({
            'date': row['交易日期'].strftime('%Y-%m-%d'),
            'value': round(cum, 4),
        })
    return bm


def _compute_benchmark_curves(result, index_returns_map):
    curves = []
    for index_id, cfg in INDEX_CONFIGS.items():
        series = (index_returns_map or {}).get(index_id)
        if series is None:
            continue
        curves.append({
            'id': index_id,
            'name': cfg['name'],
            'curve': _compute_single_benchmark_curve(result, series),
        })
    return curves


def _build_period_benchmark_returns(trade_date, index_returns_map=None):
    benchmark_returns = []
    for index_id, cfg in INDEX_CONFIGS.items():
        series = (index_returns_map or {}).get(index_id)
        if series is None:
            continue
        period_lookup = build_period_lookup(series)
        benchmark_returns.append({
            'id': index_id,
            'name': cfg['name'],
            'return': round(float(get_index_return_for_date(trade_date, period_lookup)), 6),
        })
    return benchmark_returns


def compute_split_metrics(result, split_date=SPLIT_DATE, index_returns=None, benchmark_id=None):
    """
    将回测结果拆分为训练集和测试集，分别计算指标。

    训练集: 交易日 ≤ split_date
    测试集: 交易日 > split_date

    如果提供 index_returns，还会计算每段的 alpha/beta 归因指标和基准曲线。
    如果 split_date 为 None，返回空的 train/test（表示不拆分）。

    返回 dict:
      train: {metrics, months, win_rate, monthly_returns, benchmark_curve, attribution}
      test:  {metrics, months, win_rate, monthly_returns, benchmark_curve, attribution}
      split_date: str
    """
    if split_date is None or pd.isna(split_date):
        return {'train': None, 'test': None, 'split_date': None}

    train = result[result['交易日期'] <= split_date].copy()
    test = result[result['交易日期'] > split_date].copy()

    full_period_curves = result.attrs.get('period_daily_curves', [])
    if full_period_curves:
        period_curve_lookup = {
            pd.to_datetime(dt).strftime('%Y-%m-%d'): curve
            for dt, curve in zip(result['交易日期'], full_period_curves)
        }
        train.attrs['period_daily_curves'] = [
            period_curve_lookup.get(pd.to_datetime(dt).strftime('%Y-%m-%d'), [])
            for dt in train['交易日期']
        ]
        test.attrs['period_daily_curves'] = [
            period_curve_lookup.get(pd.to_datetime(dt).strftime('%Y-%m-%d'), [])
            for dt in test['交易日期']
        ]

    # 确定初始本金
    initial_capital = result.attrs.get('initial_capital', 100000)
    if '当期本金' in result.columns and len(result) > 0:
        initial_capital = result['当期本金'].iloc[0]

    def compute_period(df, start_capital, include_holdings=False):
        if len(df) == 0:
            return None, start_capital
        df = df.copy()
        df['累积净值'] = (1 + df['选股下周期涨跌幅']).cumprod()

        # 重新计算绝对资金
        capital = float(start_capital)
        capitals = []
        pnls = []
        cum_caps = []
        for _, row in df.iterrows():
            capitals.append(capital)
            pnl = capital * row['选股下周期涨跌幅']
            pnls.append(pnl)
            capital += pnl
            cum_caps.append(capital)
        df['当期本金'] = capitals
        df['当期盈亏'] = pnls
        df['累计资金'] = cum_caps

        ev = strategy_evaluate(df, initial_capital=start_capital,
                              index_returns=index_returns)
        win_rate = round(float((df['选股下周期涨跌幅'] > 0).mean()), 4)
        monthly = [{'date': r['交易日期'].strftime('%Y-%m-%d'),
                     'value': round(float(r['选股下周期涨跌幅']), 6)}
                   for _, r in df.iterrows()]
        final_capital = float(df['累计资金'].iloc[-1]) if len(df) > 0 else start_capital

        period_result = {
            'metrics': {
                'cumulative_return': str(ev.loc['累积净值'].values[0]) if '累积净值' in ev.index else 'N/A',
                'annual_return': str(ev.loc['年化收益'].values[0]) if '年化收益' in ev.index else 'N/A',
                'max_drawdown': str(ev.loc['最大回撤'].values[0]) if '最大回撤' in ev.index else 'N/A',
                'max_dd_start': str(ev.loc['最大回撤开始'].values[0]) if '最大回撤开始' in ev.index else 'N/A',
                'max_dd_end': str(ev.loc['最大回撤结束'].values[0]) if '最大回撤结束' in ev.index else 'N/A',
                'calmar_ratio': str(ev.loc['年化收益/回撤比'].values[0]) if '年化收益/回撤比' in ev.index else 'N/A',
                'final_capital': str(ev.loc['最终资金'].values[0]) if '最终资金' in ev.index else 'N/A',
                'total_return_pct': str(ev.loc['总收益率'].values[0]) if '总收益率' in ev.index else 'N/A',
                'total_pnl': str(ev.loc['总盈亏'].values[0]) if '总盈亏' in ev.index else 'N/A',
            },
            'win_rate': win_rate,
            'months': len(df),
            'monthly_returns': monthly,
            'daily_equity_curve': _build_daily_curve_slice(df, result.attrs.get('daily_equity_curve', [])),
            'initial_capital': start_capital,
            'final_capital': final_capital,
            'date_range': {
                'start': df['交易日期'].min().strftime('%Y-%m-%d'),
                'end': df['交易日期'].max().strftime('%Y-%m-%d'),
            },
        }
        # 持股明细
        if include_holdings:
            holdings = []
            for _, row in df.iterrows():
                raw_stocks = []
                if '买入个股收益' in df.columns:
                    try:
                        raw_stocks = json.loads(row['买入个股收益'])
                    except (json.JSONDecodeError, TypeError):
                        pass
                period_capital = float(row.get('当期本金', start_capital))
                n = len(raw_stocks) if raw_stocks else 1
                stocks = []
                for s in raw_stocks:
                    target_weight = float(s.get('weight', round(1.0 / n, 4))) if n > 0 else 0
                    cap_per_stock = period_capital * target_weight
                    stocks.append(_build_stock_payload(s, cap_per_stock, period_capital, n))
                if not stocks:
                    codes = str(row.get('买入股票代码', '')).strip().split()
                    names = str(row.get('买入股票名称', '')).strip().split()
                    stocks = [{'code': c, 'name': names[i] if i < len(names) else '',
                               'weight': round(1.0/len(codes), 4) if codes else 0,
                               'position_weight': round(1.0/len(codes), 4) if codes else 0,
                               'return': 0, 'pnl': 0, 'buy_price': 0,
                               'sell_price': None, 'shares': 0, 'position_market_value': None,
                               'factor_score': None, 'rank': None,
                               'industry_l2': '', 'pe': None, 'pb': None, 'market_cap': None}
                              for i, c in enumerate(codes)]
                holdings.append({
                    'date': row['交易日期'].strftime('%Y-%m-%d'),
                    'period_return': round(float(row['选股下周期涨跌幅']), 6),
                    'period_pnl': round(float(row.get('当期盈亏', 0)), 2),
                    'capital': round(period_capital, 2),
                    'stocks': stocks,
                    'stock_count': len(stocks),
                    'benchmark_returns': _build_period_benchmark_returns(row['交易日期'], INDEX_RETURNS_MAP),
                })
            period_result['holdings'] = holdings

        # ── 归因指标 ──
        if index_returns is not None:
            attr_keys_map = {
                'Beta': 'beta', '年化Alpha': 'annual_alpha',
                '信息比率': 'information_ratio', 'R-squared': 'r_squared',
                '上行捕获率': 'up_capture', '下行捕获率': 'down_capture',
            }
            for ev_key, json_key in attr_keys_map.items():
                if ev_key in ev.index:
                    period_result['metrics'][json_key] = str(ev.loc[ev_key].values[0])

        period_result['benchmark_curve'] = _compute_single_benchmark_curve(df, index_returns)
        period_result['benchmark_curves'] = _compute_benchmark_curves(df, INDEX_RETURNS_MAP)
        period_result['active_benchmark'] = _get_benchmark_meta(benchmark_id)

        return period_result, final_capital

    train_result, train_final_cap = compute_period(train, initial_capital)
    test_result, _ = compute_period(test,
                                    train_final_cap if train_final_cap is not None else initial_capital,
                                    include_holdings=True)

    return {
        'train': train_result,
        'test': test_result,
        'split_date': split_date.strftime('%Y-%m-%d'),
        'initial_capital': initial_capital,
    }


def result_to_json(result, ev, split_date=SPLIT_DATE, benchmark_id=None):
    """将回测结果 DataFrame 转为前端 JSON，包含训练/测试集拆分"""
    # 资金曲线（倍数）
    equity_curve = [{'date': r['交易日期'].strftime('%Y-%m-%d'),
                     'value': round(float(r['累积净值']), 4),
                     'return': round(float(r['选股下周期涨跌幅']), 6)}
                    for _, r in result.iterrows()]

    # 绝对资金曲线
    capital_curve = []
    if '累计资金' in result.columns:
        capital_curve = [{'date': r['交易日期'].strftime('%Y-%m-%d'),
                          'value': round(float(r['累计资金']), 2),
                          'capital_start': round(float(r.get('当期本金', 0)), 2),
                          'pnl': round(float(r.get('当期盈亏', 0)), 2),
                          'return': round(float(r['选股下周期涨跌幅']), 6)}
                         for _, r in result.iterrows()]

    # 回撤
    cum = result['累积净值'].values
    peak = np.maximum.accumulate(cum)
    dd = cum / peak - 1
    drawdown = [{'date': result.iloc[i]['交易日期'].strftime('%Y-%m-%d'),
                 'value': round(float(dd[i]), 6)}
                for i in range(len(result))]

    # 年度收益
    yr = result.copy()
    yr['年份'] = yr['交易日期'].dt.year
    yearly = yr.groupby('年份')['选股下周期涨跌幅'].apply(
        lambda x: (1 + x).prod() - 1)
    yearly_returns = [{'year': int(y), 'value': round(float(v), 6)}
                      for y, v in yearly.items()]

    # 月度收益
    monthly = [{'date': r['交易日期'].strftime('%Y-%m-%d'),
                'value': round(float(r['选股下周期涨跌幅']), 6)}
               for _, r in result.iterrows()]
    daily_equity_curve = _build_daily_curve_slice(result, result.attrs.get('daily_equity_curve', []))

    # 持仓明细（含个股仓位占比和盈亏）
    holdings = []
    if '买入个股收益' in result.columns:
        for _, r in result.iterrows():
            try:
                raw_stocks = json.loads(r['买入个股收益'])
            except (json.JSONDecodeError, TypeError):
                raw_stocks = []
            capital = float(r.get('当期本金', 100000))
            n = len(raw_stocks) if raw_stocks else 1
            stocks = []
            for s in raw_stocks:
                target_weight = float(s.get('weight', round(1.0 / n, 4))) if n > 0 else 0
                cap_per_stock = capital * target_weight
                stocks.append(_build_stock_payload(s, cap_per_stock, capital, n))
            holdings.append({
                'date': r['交易日期'].strftime('%Y-%m-%d'),
                'period_return': round(float(r['选股下周期涨跌幅']), 6),
                'period_pnl': round(float(r.get('当期盈亏', 0)), 2),
                'capital': round(capital, 2),
                'stocks': stocks,
                'stock_count': len(stocks),
                'benchmark_returns': _build_period_benchmark_returns(r['交易日期'], INDEX_RETURNS_MAP),
            })

    def g(m):
        if m not in ev.index:
            return 'N/A'
        value = ev.loc[m].values[0]
        if value is None:
            return 'N/A'
        text = str(value).strip()
        if text.lower() in {'nan', 'nan%', 'none', 'undefined'}:
            return 'N/A'
        return text

    # 初始本金和费率信息
    initial_capital = float(result.attrs.get('initial_capital', 100000))

    active_benchmark_id, active_benchmark_series = _get_benchmark_series(benchmark_id)

    # 训练/测试集拆分
    split = compute_split_metrics(result, split_date,
                                  index_returns=active_benchmark_series,
                                  benchmark_id=active_benchmark_id)

    # 分别构建训练集和测试集的资金曲线（各自从 1 开始）
    train_curve = []
    test_curve = []
    train_capital_curve = []
    test_capital_curve = []
    if split_date and split and split.get('train') and split.get('test'):
        train_df = result[result['交易日期'] <= split_date].copy()
        train_df['累积净值'] = (1 + train_df['选股下周期涨跌幅']).cumprod()
        train_curve = [{'date': r['交易日期'].strftime('%Y-%m-%d'),
                        'value': round(float(r['累积净值']), 4),
                        'return': round(float(r['选股下周期涨跌幅']), 6)}
                       for _, r in train_df.iterrows()]

        test_df = result[result['交易日期'] > split_date].copy()
        test_df['累积净值'] = (1 + test_df['选股下周期涨跌幅']).cumprod()
        test_curve = [{'date': r['交易日期'].strftime('%Y-%m-%d'),
                       'value': round(float(r['累积净值']), 4),
                       'return': round(float(r['选股下周期涨跌幅']), 6)}
                      for _, r in test_df.iterrows()]

        # 训练/测试集的绝对资金曲线
        if split['train'] and 'final_capital' in split['train']:
            train_initial = split['train'].get('initial_capital', initial_capital)
            cap = float(train_initial)
            for _, r in train_df.iterrows():
                pnl = cap * r['选股下周期涨跌幅']
                cap_before = cap
                cap += pnl
                train_capital_curve.append({
                    'date': r['交易日期'].strftime('%Y-%m-%d'),
                    'value': round(cap, 2),
                    'pnl': round(pnl, 2),
                })

        if split['test'] and 'initial_capital' in split['test']:
            test_initial = split['test'].get('initial_capital', initial_capital)
            cap = float(test_initial)
            for _, r in test_df.iterrows():
                pnl = cap * r['选股下周期涨跌幅']
                cap_before = cap
                cap += pnl
                test_capital_curve.append({
                    'date': r['交易日期'].strftime('%Y-%m-%d'),
                    'value': round(cap, 2),
                    'pnl': round(pnl, 2),
                })

    profile_summary = []
    strategy_meta = result.attrs.get('strategy_meta', {}) if hasattr(result, 'attrs') else {}
    if isinstance(strategy_meta, dict):
        profile_summary = strategy_meta.get('profile_summary', []) or []

    return {
        'equity_curve': equity_curve,
        'daily_equity_curve': daily_equity_curve,
        'capital_curve': capital_curve,
        'train_equity_curve': train_curve,
        'test_equity_curve': test_curve,
        'train_daily_equity_curve': split.get('train', {}).get('daily_equity_curve', []) if split and split.get('train') else [],
        'test_daily_equity_curve': split.get('test', {}).get('daily_equity_curve', []) if split and split.get('test') else [],
        'train_capital_curve': train_capital_curve,
        'test_capital_curve': test_capital_curve,
        'drawdown': drawdown,
        'yearly_returns': yearly_returns,
        'monthly_returns': monthly,
        'holdings': holdings,
        'metrics': {
            'cumulative_return': g('累积净值'),
            'annual_return': g('年化收益'),
            'max_drawdown': g('最大回撤'),
            'max_dd_start': g('最大回撤开始'),
            'max_dd_end': g('最大回撤结束'),
            'calmar_ratio': g('年化收益/回撤比'),
            'final_capital': g('最终资金'),
            'total_return_pct': g('总收益率'),
            'total_pnl': g('总盈亏'),
            'beta': g('Beta'),
            'annual_alpha': g('年化Alpha'),
            'information_ratio': g('信息比率'),
            'r_squared': g('R-squared'),
            'up_capture': g('上行捕获率'),
            'down_capture': g('下行捕获率'),
        },
        'initial_capital': initial_capital,
        'fee_info': {
            'c_rate': result.attrs.get('c_rate', 1.0 / 10000),
            't_rate': result.attrs.get('t_rate', 1 / 1000),
            'sell_cost': result.attrs.get('sell_cost', 1.0 / 10000 + 1 / 1000),
            'total_buy_fees': round(result.attrs.get('total_buy_fees', 0), 2),
            'total_sell_fees': round(result.attrs.get('total_sell_fees', 0), 2),
            'total_fees': round(result.attrs.get('total_fees', 0), 2),
        },
        'win_rate': round(float((result['选股下周期涨跌幅'] > 0).mean()), 4),
        'date_range': {
            'start': result['交易日期'].min().strftime('%Y-%m-%d'),
            'end': result['交易日期'].max().strftime('%Y-%m-%d'),
        },
        'total_months': len(result),
        'split': split,
        'profile_summary': profile_summary,
        'active_benchmark': _get_benchmark_meta(active_benchmark_id),
        'benchmark_curve': _compute_single_benchmark_curve(result, active_benchmark_series),
        'benchmark_curves': _compute_benchmark_curves(result, INDEX_RETURNS_MAP),
    }


# ═══════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/timing')
def timing_page():
    return render_template('timing.html')


# 各策略默认参数值（用于缓存命中判断，值对应前端 slider 默认值）
_CACHE_DEFAULTS = {
    'original': {'val_pct_cutoff': 0.68, 'bias_pct': 0.52, 'vol_pct': 0.78},
    'original_ensemble': {'weight_3y': 0.5, 'weight_5y': 0.3, 'weight_full': 0.2, 'vote_top_k': 12, 'board_tilt_strength': 0.4, 'growth_hold_days': 4, 'growth_top_n': 2},
    'chan_enhanced': {'val_pct_cutoff': 0.68, 'bias_pct': 0.52, 'vol_pct': 0.78, 'chan_tilt': 0.03},
    'chan_only': {'chan_weight': 0.70},
    'method_a': {'val_pct_cutoff': 0.68, 'bias_pct': 0.52, 'vol_pct': 0.78, 'chan_tilt': 0.05},
    'quality_value': {
        'size_weight': 0.50, 'bm_weight': 0.25, 'roe_weight': 0.15,
        'turnover_weight': 0.10, 'min_market_cap': 20, 'min_turnover': 0.5,
        'select_stock_num': 3, 'bias_pct': 0.52, 'vol_pct': 0.78,
    },
}

_TIMING_CACHE_DEFAULTS = {
    'csi1000_timing': {'fast_window': 20, 'slow_window': 60, 'momentum_window': 60},
    'star50_timing': {'breakout_window': 20, 'exit_window': 10, 'trend_window': 60},
    'chinext_timing': {'momentum_short_window': 20, 'momentum_long_window': 60, 'trend_window': 60, 'momentum_threshold': 0.0},
}


@app.route('/api/backtest')
def api_backtest():
    init_cache()
    strategy = request.args.get('strategy', 'original')
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    benchmark_id = _normalize_benchmark_id(request.args.get('benchmark', DEFAULT_BENCHMARK_ID))

    # ── 缓存命中判断 ──
    use_cache = strategy in BACKTEST_CACHE
    if use_cache and strategy in _CACHE_DEFAULTS:
        for key, default_val in _CACHE_DEFAULTS[strategy].items():
            val = request.args.get(key, type=float)
            if val is not None and val != default_val:
                use_cache = False
                break

    try:
        if use_cache:
            result, _ = BACKTEST_CACHE[strategy]
            result = result.copy()
        else:
            # 收集前端 slider 传来的所有参数
            params = {}

            # 通用参数（key 与策略 __init__ 参数名一致）
            for key in ['val_pct_cutoff', 'bias_pct', 'vol_pct', 'chan_tilt',
                        'chan_weight', 'size_weight', 'bm_weight', 'roe_weight', 'turnover_weight',
                        'weight_3y', 'weight_5y', 'weight_full', 'vote_top_k', 'board_tilt_strength']:
                val = request.args.get(key, type=float)
                if val is not None:
                    params[key] = val

            growth_timing_mode = request.args.get('growth_timing_mode')
            if growth_timing_mode:
                params['growth_timing_mode'] = growth_timing_mode

            for key in ['select_stock_num', 'growth_hold_days', 'growth_top_n']:
                val = request.args.get(key, type=int)
                if val is not None:
                    params[key] = val

            # min_market_cap / min_turnover：前端以"亿"为单位，转换为元
            min_market_cap_raw = request.args.get('min_market_cap', type=float)
            if min_market_cap_raw is not None:
                params['min_market_cap'] = min_market_cap_raw * 1e8

            min_turnover_raw = request.args.get('min_turnover', type=float)
            if min_turnover_raw is not None:
                params['min_turnover'] = min_turnover_raw * 1e8

            result, _ = run_backtest_fresh(strategy, benchmark_id=benchmark_id, **params)

        active_benchmark_id, active_benchmark_series = _get_benchmark_series(benchmark_id)

        # 日期过滤（如果用户选了特定日期范围）
        if start_date or end_date:
            result, ev = filter_by_date(result, start_date, end_date, benchmark_id=active_benchmark_id)
            if result is None:
                return jsonify({'error': '所选日期范围内无数据'}), 400
            # 用户自定义日期范围时不显示训练/测试拆分
            return jsonify(result_to_json(result, ev, split_date=None, benchmark_id=active_benchmark_id))
        else:
            # 全量数据：包含训练/测试集拆分
            if use_cache:
                ev = strategy_evaluate(result, index_returns=active_benchmark_series)
            else:
                ev = strategy_evaluate(result, index_returns=active_benchmark_series)
            return jsonify(result_to_json(result, ev, split_date=SPLIT_DATE, benchmark_id=active_benchmark_id))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/factors')
def api_factors():
    ensure_stock_data_loaded()
    strategy_name = request.args.get('strategy', get_focused_strategy_id())
    strategy = build_strategy(strategy_name)
    return jsonify(strategy.get_factor_metadata())


@app.route('/api/factor_overview')
def api_factor_overview():
    ensure_stock_data_loaded()
    strategy_name = request.args.get('strategy', get_focused_strategy_id())
    return jsonify(build_factor_overview_payload(strategy_name))


@app.route('/api/info')
def api_info():
    """返回数据库基本信息，包括最新日期范围"""
    ensure_stock_data_loaded()
    if DATA_DF is None:
        return jsonify({'error': '数据未加载'}), 500
    max_date = pd.to_datetime(DATA_DF['交易日期'].max())
    min_date = pd.to_datetime(DATA_DF['交易日期'].min())
    return jsonify({
        'data_min_date': min_date.strftime('%Y-%m-%d'),
        'data_max_date': max_date.strftime('%Y-%m-%d'),
    })


@app.route('/api/timing/info')
def api_timing_info():
    ensure_timing_panel_loaded()
    if TIMING_PANEL is None or len(TIMING_PANEL) == 0:
        return jsonify({'error': '指数数据未加载'}), 500
    max_date = pd.to_datetime(TIMING_PANEL['交易日期'].max())
    min_date = pd.to_datetime(TIMING_PANEL['交易日期'].min())
    return jsonify({
        'data_min_date': min_date.strftime('%Y-%m-%d'),
        'data_max_date': max_date.strftime('%Y-%m-%d'),
        'indexes': [
            {'id': strategy_id, 'name': strategy_cls().get_display_name(), 'index_name': strategy_cls().get_index_name()}
            for strategy_id, strategy_cls in TIMING_STRATEGY_MAP.items()
        ],
    })


@app.route('/api/timing/strategy_list')
def api_timing_strategy_list():
    init_timing_cache()
    payload = []
    for strategy_id, strategy_cls in TIMING_STRATEGY_MAP.items():
        strategy = strategy_cls()
        cached = TIMING_CACHE.get(strategy_id)
        cumulative_return = None
        current_action = None
        if cached is not None and len(cached) > 0:
            cumulative_return = round(float(cached['累积净值'].iloc[-1]), 2)
            current_action = str(cached['signal_action'].iloc[-1])
        payload.append({
            'id': strategy_id,
            'name': strategy.get_display_name(),
            'description': strategy.get_strategy_description(),
            'index_name': strategy.get_index_name(),
            'cumulative_return': cumulative_return,
            'current_action': current_action,
        })
    return jsonify(payload)


@app.route('/api/timing/params')
def api_timing_params():
    strategy_name = request.args.get('strategy', 'csi1000_timing')
    strategy = build_timing_strategy(strategy_name)
    return jsonify(strategy.get_signal_metadata())


@app.route('/api/timing/signals')
def api_timing_signals():
    payload = []
    for strategy_id, strategy_cls in TIMING_STRATEGY_MAP.items():
        strategy = strategy_cls()
        result = TIMING_CACHE.get(strategy_id)
        if result is None or len(result) == 0:
            payload.append({
                'id': strategy_id,
                'name': strategy.get_display_name(),
                'index_name': strategy.get_index_name(),
                'date': None,
                'action': None,
                'position': 0,
                'reason_summary': '加载中',
                'nav': None,
            })
            continue
        latest = result.iloc[-1]
        payload.append({
            'id': strategy_id,
            'name': strategy.get_display_name(),
            'index_name': strategy.get_index_name(),
            'date': pd.to_datetime(latest['交易日期']).strftime('%Y-%m-%d'),
            'action': latest['signal_action'],
            'position': int(latest['position']),
            'reason_summary': latest['reason_summary'],
            'nav': round(float(latest['累积净值']), 4),
        })
    return jsonify(payload)


@app.route('/api/timing/backtest')
def api_timing_backtest():
    strategy_name = request.args.get('strategy', 'csi1000_timing')
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    compact = request.args.get('compact', '0') in {'1', 'true', 'yes'}
    benchmark_id = _normalize_benchmark_id(request.args.get('benchmark', DEFAULT_BENCHMARK_ID))

    use_cache = strategy_name in TIMING_CACHE
    if use_cache and strategy_name in _TIMING_CACHE_DEFAULTS:
        for key, default_val in _TIMING_CACHE_DEFAULTS[strategy_name].items():
            val = request.args.get(key, type=float)
            if val is not None and val != default_val:
                use_cache = False
                break

    try:
        params = {}
        for key in ['fast_window', 'slow_window', 'momentum_window', 'breakout_window', 'exit_window', 'trend_window', 'momentum_short_window', 'momentum_long_window', 'momentum_threshold']:
            val = request.args.get(key, type=float)
            if val is not None:
                params[key] = int(val) if key != 'momentum_threshold' else float(val)

        if use_cache:
            result = TIMING_CACHE[strategy_name].copy()
            strategy = build_timing_strategy(strategy_name)
        else:
            result, _, strategy = run_timing_backtest_fresh(strategy_name, benchmark_id=benchmark_id, **params)

        active_benchmark_id, active_benchmark_series = _get_benchmark_series(benchmark_id)
        if start_date:
            result = result[result['交易日期'] >= pd.to_datetime(start_date)].copy()
        if end_date:
            result = result[result['交易日期'] <= pd.to_datetime(end_date)].copy()
        if len(result) == 0:
            return jsonify({'error': '所选日期范围内无数据'}), 400

        metrics = evaluate_timing_result(result, benchmark_returns=active_benchmark_series)
        return jsonify(timing_result_to_json(
            result,
            metrics,
            benchmark_meta=_get_benchmark_meta(active_benchmark_id),
            benchmark_curve=_compute_single_benchmark_curve(result, active_benchmark_series),
            benchmark_curves=_compute_benchmark_curves(result, INDEX_RETURNS_MAP),
            compact=compact,
        ))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/strategy_list')
def api_strategy_list():
    strategy_id = get_focused_strategy_id()
    strategy = build_strategy(strategy_id)
    return jsonify([
        {
            'id': strategy_id,
            'name': strategy.get_display_name(),
            'description': strategy.get_strategy_description(),
            'cumulative_return': '894.67x',
            'best': True,
            'focus_only': True,
        }
    ])


if __name__ == '__main__':
    print("=" * 60)
    print("  量化策略 Web 可视化")
    print("  访问 http://localhost:8080")
    print("=" * 60)
    app.run(debug=False, host='0.0.0.0', port=8080)
