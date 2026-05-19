import json

import numpy as np
import pandas as pd

from backtest import compute_alpha_beta
from index_data import INDEX_CONFIGS, build_period_lookup, get_index_return_for_date


def run_timing_backtest(signal_df, strategy, benchmark_returns=None):
    df = signal_df.copy().sort_values('交易日期').reset_index(drop=True)
    df['index_return'] = df['close'].pct_change().fillna(0.0)

    prev_position = df['position'].shift(1).fillna(0).astype(int)
    trade_cost = np.where(
        (prev_position == 0) & (df['position'] == 1), strategy.buy_cost,
        np.where((prev_position == 1) & (df['position'] == 0), strategy.sell_cost, 0.0)
    )
    df['trade_cost'] = trade_cost
    df['strategy_return'] = df['index_return'] * df['position'] - df['trade_cost']
    df['累积净值'] = (1 + df['strategy_return']).cumprod()

    capital = float(strategy.initial_capital)
    capitals = []
    pnls = []
    cum_capitals = []
    for _, row in df.iterrows():
        capitals.append(capital)
        pnl = capital * float(row['strategy_return'])
        pnls.append(pnl)
        capital += pnl
        cum_capitals.append(capital)

    df['当期本金'] = capitals
    df['当期盈亏'] = pnls
    df['累计资金'] = cum_capitals
    df.attrs['initial_capital'] = strategy.initial_capital
    df.attrs['buy_cost'] = strategy.buy_cost
    df.attrs['sell_cost'] = strategy.sell_cost
    df.attrs['benchmark_returns'] = benchmark_returns
    return df


def evaluate_timing_result(result_df, benchmark_returns=None):
    result = result_df.copy()
    metrics = {}
    if len(result) == 0:
        return metrics

    initial_capital = float(result.attrs.get('initial_capital', 100000))
    final_nav = float(result['累积净值'].iloc[-1])
    final_capital = float(result['累计资金'].iloc[-1])

    metrics['累积净值'] = round(final_nav, 4)
    date_delta = pd.to_datetime(result['交易日期'].iloc[-1]) - pd.to_datetime(result['交易日期'].iloc[0])
    days = max(getattr(date_delta, 'days', 0), 1)
    annual_return = final_nav ** (365.0 / days) - 1 if days > 0 else 0.0
    metrics['年化收益'] = f"{round(annual_return * 100, 2)}%"

    cum = result['累积净值'].astype(float)
    peak = cum.cummax()
    dd = cum / peak - 1
    max_drawdown = float(dd.min()) if len(dd) else 0.0
    end_idx = int(dd.idxmin()) if len(dd) else 0
    end_date = pd.to_datetime(result.iloc[end_idx]['交易日期'])
    start_subset = result[result['交易日期'] <= end_date].copy()
    start_idx = start_subset['累积净值'].astype(float).idxmax() if len(start_subset) else 0
    start_date = pd.to_datetime(result.loc[start_idx, '交易日期']) if len(result) else end_date

    metrics['最大回撤'] = format(max_drawdown, '.2%')
    metrics['最大回撤开始'] = start_date.strftime('%Y-%m-%d')
    metrics['最大回撤结束'] = end_date.strftime('%Y-%m-%d')
    metrics['年化收益/回撤比'] = round(annual_return / abs(max_drawdown), 2) if max_drawdown != 0 else 0
    metrics['最终资金'] = round(final_capital, 2)
    metrics['总收益率'] = f"{round((final_capital / initial_capital - 1) * 100, 2)}%"
    metrics['总盈亏'] = round(final_capital - initial_capital, 2)

    if benchmark_returns is not None:
        strategy_rets = result.set_index('交易日期')['strategy_return'].resample('M').apply(lambda x: (1 + x).prod() - 1)
        attr = compute_alpha_beta(strategy_rets, benchmark_returns)
        if 'error' not in attr:
            metrics['Beta'] = attr['beta']
            metrics['月度Alpha'] = f"{round(attr['alpha_monthly'] * 100, 4)}%"
            metrics['年化Alpha'] = f"{round(attr['alpha_annualized'] * 100, 2)}%"
            metrics['信息比率'] = attr['information_ratio']
            metrics['R-squared'] = attr['r_squared']
            metrics['上行捕获率'] = f"{round(attr['up_capture'] * 100, 1)}%" if attr['up_capture'] is not None else 'N/A'
            metrics['下行捕获率'] = f"{round(attr['down_capture'] * 100, 1)}%" if attr['down_capture'] is not None else 'N/A'
    return metrics


def _compute_single_benchmark_curve(result_df, index_returns):
    if index_returns is None:
        return []
    lookup = build_period_lookup(index_returns)
    curve = []
    cum = 1.0
    monthly_dates = result_df['交易日期'].drop_duplicates().sort_values()
    for date in monthly_dates:
        ret = get_index_return_for_date(date, lookup)
        cum *= (1 + ret)
        curve.append({'date': pd.to_datetime(date).strftime('%Y-%m-%d'), 'value': round(float(cum), 4)})
    return curve


def _compute_benchmark_curves(result_df, index_returns_map):
    curves = []
    for index_id, cfg in INDEX_CONFIGS.items():
        series = (index_returns_map or {}).get(index_id)
        if series is None:
            continue
        curves.append({
            'id': index_id,
            'name': cfg['name'],
            'curve': _compute_single_benchmark_curve(result_df, series),
        })
    return curves


def timing_result_to_json(result_df, metrics, benchmark_meta=None, benchmark_curve=None, benchmark_curves=None, compact=False):
    equity_curve = [
        {'date': pd.to_datetime(r['交易日期']).strftime('%Y-%m-%d'), 'value': round(float(r['累积净值']), 4), 'return': round(float(r['strategy_return']), 6)}
        for _, r in result_df.iterrows()
    ]
    daily_equity_curve = list(equity_curve) if not compact else []

    cum = result_df['累积净值'].values
    peak = np.maximum.accumulate(cum)
    dd = cum / peak - 1
    drawdown = [
        {'date': pd.to_datetime(result_df.iloc[i]['交易日期']).strftime('%Y-%m-%d'), 'value': round(float(dd[i]), 6)}
        for i in range(len(result_df))
    ] if not compact else []

    monthly_returns = [
        {'date': pd.to_datetime(r['交易日期']).strftime('%Y-%m-%d'), 'value': round(float(r['strategy_return']), 6)}
        for _, r in result_df.iterrows()
    ] if not compact else []

    yearly = result_df.copy()
    yearly['年份'] = pd.to_datetime(yearly['交易日期']).dt.year
    yearly_returns = yearly.groupby('年份')['strategy_return'].apply(lambda x: (1 + x).prod() - 1)
    yearly_payload = [{'year': int(y), 'value': round(float(v), 6)} for y, v in yearly_returns.items()] if not compact else []

    signal_rows = result_df.tail(20) if compact else result_df
    signals = []
    trades = []
    for _, row in signal_rows.iterrows():
        item = {
            'date': pd.to_datetime(row['交易日期']).strftime('%Y-%m-%d'),
            'action': row['signal_action'],
            'position': int(row['position']),
            'reason_summary': row['reason_summary'],
            'reason_detail': row['reason_detail'],
            'score': round(float(row.get('signal_score', 0) or 0), 6),
            'close': round(float(row['close']), 2),
        }
        signals.append(item)
        if not compact and row['signal_action'] in {'buy', 'sell'}:
            trades.append(item)

    latest = result_df.iloc[-1]
    snapshot_fields = {}
    for col in ['close', 'ma_fast', 'ma_slow', 'momentum_long', 'momentum_short', 'trend_ma', 'breakout_high', 'exit_low']:
        if col in result_df.columns and pd.notna(latest.get(col)):
            snapshot_fields[col] = round(float(latest[col]), 6)

    def g(key):
        return metrics.get(key, 'N/A')

    return {
        'equity_curve': equity_curve,
        'daily_equity_curve': daily_equity_curve,
        'drawdown': drawdown,
        'yearly_returns': yearly_payload,
        'monthly_returns': monthly_returns,
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
        'initial_capital': float(result_df.attrs.get('initial_capital', 100000)),
        'win_rate': round(float((result_df['strategy_return'] > 0).mean()), 4),
        'date_range': {
            'start': pd.to_datetime(result_df['交易日期'].min()).strftime('%Y-%m-%d'),
            'end': pd.to_datetime(result_df['交易日期'].max()).strftime('%Y-%m-%d'),
        },
        'total_months': len(result_df),
        'signals': signals,
        'trades': trades,
        'signal_summary': {
            'current_action': latest['signal_action'],
            'current_position': int(latest['position']),
            'current_reason': latest['reason_summary'],
        },
        'active_index': {
            'id': latest['index_id'],
            'name': latest['index_name'],
        },
        'indicator_snapshots': snapshot_fields,
        'active_benchmark': benchmark_meta,
        'benchmark_curve': benchmark_curve or [],
        'benchmark_curves': benchmark_curves or [],
        'fee_info': {
            'buy_cost': result_df.attrs.get('buy_cost', 0.001),
            'sell_cost': result_df.attrs.get('sell_cost', 0.001),
            'total_trade_cost': round(float(result_df['trade_cost'].sum() * float(result_df.attrs.get('initial_capital', 100000))), 2),
        },
    }
