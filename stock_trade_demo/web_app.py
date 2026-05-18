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
from strategies.chan_enhanced import ChanEnhancedStrategy
from strategies.chan_only import ChanOnlyStrategy
from strategies.method_a import MethodAStrategy
from strategies.quality_value import QualityValueStrategy
from backtest import load_data, select_and_backtest, strategy_evaluate, compute_alpha_beta
from index_data import get_index_returns, build_period_lookup, get_index_return_for_date

warnings.filterwarnings('ignore')
pd.set_option('expand_frame_repr', False)

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), 'web', 'templates'))

# 全局缓存
DATA_DF = None
INDEX_RETURNS = None  # CSI 1000 月度收益 Series
BACKTEST_CACHE = {}  # key: strategy_name → (result_df, eval_df)

STRATEGY_MAP = {
    'original': OriginalStrategy,
    'chan_enhanced': ChanEnhancedStrategy,
    'chan_only': ChanOnlyStrategy,
    'method_a': MethodAStrategy,
    'quality_value': QualityValueStrategy,
}

# 训练/测试集拆分日期
SPLIT_DATE = pd.to_datetime('2026-02-28')


def init_cache():
    """启动时预加载数据、指数收益并运行默认参数回测"""
    global DATA_DF, INDEX_RETURNS
    csv_path = os.path.join(os.path.dirname(__file__), 'stock_data.csv')
    if not os.path.exists(csv_path):
        print(f"[WARN] 数据文件不存在: {csv_path}")
        return
    print("[init] 加载数据中 (823MB)...")
    DATA_DF = load_data(csv_path)
    print("[init] 数据加载完成")

    # 加载 CSI 1000 指数月度收益
    try:
        INDEX_RETURNS = get_index_returns()
        print(f"[init] CSI 1000 指数收益加载完成，{len(INDEX_RETURNS)} 个月")
    except Exception as e:
        print(f"[WARN] 无法加载指数收益数据: {e}")
        print("[WARN] 业绩归因功能将不可用")
        INDEX_RETURNS = None

    for sid, cls in [('original', OriginalStrategy),
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
            ev = strategy_evaluate(result, index_returns=INDEX_RETURNS)
            BACKTEST_CACHE[sid] = (result, ev)
            print(f"[init] {sid} 完成, 累积净值: {result['累积净值'].iloc[-1]:.2f}")
        except Exception as e:
            print(f"[init] {sid} 失败: {e}")


def build_strategy(strategy_name='original', **params):
    strat_cls = STRATEGY_MAP.get(strategy_name, OriginalStrategy)
    sig = inspect.signature(strat_cls.__init__)
    valid_params = {k: v for k, v in params.items()
                    if k in sig.parameters and v is not None}
    valid_params.pop('self', None)
    return strat_cls(**valid_params)


def run_backtest_fresh(strategy_name='original', **params):
    """重新运行回测（参数不同于默认值时使用）。返回: (result_df, eval_df)"""
    if DATA_DF is None:
        raise RuntimeError("数据未加载，请确认 stock_data.csv 存在")

    strategy = build_strategy(strategy_name, **params)

    df = strategy.run(DATA_DF.copy())
    result = select_and_backtest(df, strategy,
                                 c_rate=strategy.c_rate,
                                 t_rate=strategy.t_rate,
                                 bull_tp=strategy.bull_tp,
                                 bear_tp=strategy.bear_tp,
                                 bull_n=strategy.bull_n,
                                 bear_n=strategy.bear_n,
                                 initial_capital=strategy.initial_capital)
    ev = strategy_evaluate(result, index_returns=INDEX_RETURNS)
    return result, ev



def filter_by_date(result, start_date, end_date):
    """按日期范围过滤回测结果并重新计算累积净值和评估指标。
    初始资金始终重置为 100,000——自定义日期范围视为独立回测区间。"""
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

    ev = strategy_evaluate(result, initial_capital=start_capital,
                          index_returns=INDEX_RETURNS)
    return result, ev


def compute_split_metrics(result, split_date=SPLIT_DATE, index_returns=None):
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
                cap_per_stock = period_capital / n
                stocks = []
                for s in raw_stocks:
                    ret = s.get('return', 0)
                    buy_price = s.get('buy_price', 0)
                    # 基于当前期本金重算持股数量（A股100股整数倍）
                    if buy_price > 0:
                        shares = int(cap_per_stock / buy_price / 100) * 100
                        actual_invested = shares * buy_price
                    else:
                        shares = 0
                        actual_invested = cap_per_stock
                    pnl = round(float(actual_invested * ret), 2) if actual_invested > 0 else round(float(cap_per_stock * ret), 2)
                    stocks.append({
                        'code': s.get('code', ''),
                        'name': s.get('name', ''),
                        'weight': s.get('weight', round(1.0 / n, 4)),
                        'return': ret,
                        'pnl': pnl,
                        'buy_price': s.get('buy_price', 0),
                        'sell_price': s.get('sell_price', None),
                        'shares': shares,
                        'factor_score': s.get('factor_score'),
                        'rank': s.get('rank'),
                        'industry_l2': s.get('industry_l2', ''),
                        'pe': s.get('pe'),
                        'pb': s.get('pb'),
                        'market_cap': s.get('market_cap'),
                    'selection_reason_summary': s.get('selection_reason_summary', ''),
                    'selection_reason_detail': s.get('selection_reason_detail', []),
                    'selection_fundamentals': s.get('selection_fundamentals', []),
                    'selection_factor_breakdown': s.get('selection_factor_breakdown', []),
                    })
                if not stocks:
                    codes = str(row.get('买入股票代码', '')).strip().split()
                    names = str(row.get('买入股票名称', '')).strip().split()
                    stocks = [{'code': c, 'name': names[i] if i < len(names) else '',
                               'weight': round(1.0/len(codes), 4) if codes else 0,
                               'return': 0, 'pnl': 0, 'buy_price': 0,
                               'sell_price': None, 'shares': 0,
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

            # 基准曲线（从 1 开始累积）
            lookup = build_period_lookup(index_returns)
            bm_curve = []
            cum = 1.0
            for _, row in df.iterrows():
                idx_ret = get_index_return_for_date(row['交易日期'], lookup)
                cum *= (1 + idx_ret)
                bm_curve.append({
                    'date': row['交易日期'].strftime('%Y-%m-%d'),
                    'value': round(cum, 4),
                })
            period_result['benchmark_curve'] = bm_curve

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


def result_to_json(result, ev, split_date=SPLIT_DATE):
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
            cap_per_stock = capital / n
            stocks = []
            for s in raw_stocks:
                ret = s.get('return', 0)
                buy_price = s.get('buy_price', 0)
                sell_price = s.get('sell_price', None)
                # 基于当前期本金重算持股数量（A股100股整数倍）
                if buy_price > 0:
                    shares = int(cap_per_stock / buy_price / 100) * 100
                    actual_invested = shares * buy_price
                else:
                    shares = 0
                    actual_invested = cap_per_stock
                # 重算个股盈亏
                if actual_invested > 0:
                    pnl = round(float(actual_invested * ret), 2)
                else:
                    pnl = round(float(cap_per_stock * ret), 2)
                stocks.append({
                    'code': s.get('code', ''),
                    'name': s.get('name', ''),
                    'weight': s.get('weight', round(1.0 / n, 4)),
                    'return': ret,
                    'pnl': pnl,
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'shares': shares,
                    'factor_score': s.get('factor_score'),
                    'rank': s.get('rank'),
                    'industry_l2': s.get('industry_l2', ''),
                    'pe': s.get('pe'),
                    'pb': s.get('pb'),
                    'market_cap': s.get('market_cap'),
                    'selection_reason_summary': s.get('selection_reason_summary', ''),
                    'selection_reason_detail': s.get('selection_reason_detail', []),
                    'selection_fundamentals': s.get('selection_fundamentals', []),
                    'selection_factor_breakdown': s.get('selection_factor_breakdown', []),
                })
            holdings.append({
                'date': r['交易日期'].strftime('%Y-%m-%d'),
                'period_return': round(float(r['选股下周期涨跌幅']), 6),
                'period_pnl': round(float(r.get('当期盈亏', 0)), 2),
                'capital': round(capital, 2),
                'stocks': stocks,
                'stock_count': len(stocks),
            })

    def g(m):
        return str(ev.loc[m].values[0]) if m in ev.index else 'N/A'

    # 初始本金和费率信息
    initial_capital = float(result.attrs.get('initial_capital', 100000))

    # 训练/测试集拆分
    split = compute_split_metrics(result, split_date, index_returns=INDEX_RETURNS)

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

    return {
        'equity_curve': equity_curve,
        'capital_curve': capital_curve,
        'train_equity_curve': train_curve,
        'test_equity_curve': test_curve,
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
        'benchmark_curve': _compute_benchmark_curve(result, INDEX_RETURNS),
    }


def _compute_benchmark_curve(result, index_returns):
    """为全量结果计算基准曲线（CSI 1000），从 1 开始累积。"""
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


# ═══════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


# 各策略默认参数值（用于缓存命中判断，值对应前端 slider 默认值）
_CACHE_DEFAULTS = {
    'original': {'val_pct_cutoff': 0.68, 'bias_pct': 0.52, 'vol_pct': 0.78},
    'chan_enhanced': {'val_pct_cutoff': 0.68, 'bias_pct': 0.52, 'vol_pct': 0.78, 'chan_tilt': 0.03},
    'chan_only': {'chan_weight': 0.70},
    'method_a': {'val_pct_cutoff': 0.68, 'bias_pct': 0.52, 'vol_pct': 0.78, 'chan_tilt': 0.05},
    'quality_value': {
        'size_weight': 0.50, 'bm_weight': 0.25, 'roe_weight': 0.15,
        'turnover_weight': 0.10, 'min_market_cap': 20, 'min_turnover': 0.5,
        'select_stock_num': 3, 'bias_pct': 0.52, 'vol_pct': 0.78,
    },
}


@app.route('/api/backtest')
def api_backtest():
    strategy = request.args.get('strategy', 'original')
    start_date = request.args.get('start')
    end_date = request.args.get('end')

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
                        'chan_weight', 'size_weight', 'bm_weight', 'roe_weight', 'turnover_weight']:
                val = request.args.get(key, type=float)
                if val is not None:
                    params[key] = val

            # select_stock_num: 整数参数
            ssn = request.args.get('select_stock_num', type=int)
            if ssn is not None:
                params['select_stock_num'] = ssn

            # min_market_cap / min_turnover：前端以"亿"为单位，转换为元
            min_market_cap_raw = request.args.get('min_market_cap', type=float)
            if min_market_cap_raw is not None:
                params['min_market_cap'] = min_market_cap_raw * 1e8

            min_turnover_raw = request.args.get('min_turnover', type=float)
            if min_turnover_raw is not None:
                params['min_turnover'] = min_turnover_raw * 1e8

            result, _ = run_backtest_fresh(strategy, **params)

        # 日期过滤（如果用户选了特定日期范围）
        if start_date or end_date:
            result, ev = filter_by_date(result, start_date, end_date)
            if result is None:
                return jsonify({'error': '所选日期范围内无数据'}), 400
            # 用户自定义日期范围时不显示训练/测试拆分
            return jsonify(result_to_json(result, ev, split_date=None))
        else:
            # 全量数据：包含训练/测试集拆分
            ev = strategy_evaluate(result)
            return jsonify(result_to_json(result, ev, split_date=SPLIT_DATE))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/factors')
def api_factors():
    strategy_name = request.args.get('strategy', 'original')
    strategy = build_strategy(strategy_name)
    return jsonify(strategy.get_factor_metadata())


@app.route('/api/info')
def api_info():
    """返回数据库基本信息，包括最新日期范围"""
    if DATA_DF is None:
        return jsonify({'error': '数据未加载'}), 500
    max_date = pd.to_datetime(DATA_DF['交易日期'].max())
    min_date = pd.to_datetime(DATA_DF['交易日期'].min())
    return jsonify({
        'data_min_date': min_date.strftime('%Y-%m-%d'),
        'data_max_date': max_date.strftime('%Y-%m-%d'),
    })


@app.route('/api/strategy_list')
def api_strategy_list():
    return jsonify([
        {'id': 'original', 'name': '原版策略',
         'description': '行业估值 + bias反转 + 小市值',
         'cumulative_return': '5223x', 'best': True},
        {'id': 'chan_enhanced', 'name': '缠论增强 v1.1',
         'description': '原版过滤 + 缠论代理因子',
         'cumulative_return': '5131x', 'best': False},
        {'id': 'chan_only', 'name': '纯缠论 v1.2',
         'description': '仅缠论因子',
         'cumulative_return': '16x', 'best': False},
        {'id': 'method_a', 'name': 'Method A v2.0',
         'description': '日线缠论流水线聚合',
         'cumulative_return': '5223x', 'best': False},
        {'id': 'quality_value', 'name': '质量价值小盘 v3.0',
         'description': '规模+价值+质量+反操纵 Z-score复合',
         'cumulative_return': 'TBD', 'best': False},
    ])


if __name__ == '__main__':
    print("=" * 60)
    print("  量化策略 Web 可视化")
    print("  访问 http://localhost:8080")
    print("=" * 60)
    init_cache()
    app.run(debug=False, host='0.0.0.0', port=8080)
