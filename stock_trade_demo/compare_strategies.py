"""
策略对比入口 — 运行所有策略并对比分析。

用法:
  python3 compare_strategies.py --plot compare  # 详细对比图
  python3 compare_strategies.py --plot raw      # 简洁风格
  python3 compare_strategies.py --plot both     # 两者都出
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np

from strategies.original import OriginalStrategy
from strategies.chan_enhanced import ChanEnhancedStrategy
from strategies.chan_only import ChanOnlyStrategy
from strategies.method_a import MethodAStrategy
from backtest import load_data, select_and_backtest, strategy_evaluate
from visualization import plot_strategy_comparison, plot_raw_style

warnings.filterwarnings('ignore')
pd.set_option('expand_frame_repr', False)


def parse_arg(flag, default='compare'):
    try:
        idx = sys.argv.index(flag)
        return sys.argv[idx + 1]
    except (ValueError, IndexError):
        return default


def main():
    plot_style = parse_arg('--plot', 'compare')

    print("=" * 75)
    print("  策略对比分析 — 原版 vs 缠论增强 vs 纯缠论 vs Method A")
    print("=" * 75)

    df_full = load_data('stock_data.csv')

    # 注册所有策略
    strategies_cfg = {
        '原版策略': OriginalStrategy(),
        '缠论增强': ChanEnhancedStrategy(),
        '纯缠论': ChanOnlyStrategy(),
        'Method A': MethodAStrategy(),
    }

    strategies = {}
    eval_results = {}

    for name, strat in strategies_cfg.items():
        print(f"\n>>> [{name}]")
        df = strat.run(df_full.copy())
        result = select_and_backtest(df, strat)
        result.to_csv(f'选股策略详情_{name}.csv', encoding='gbk', index=False)
        ev = strategy_evaluate(result)
        print(f"\n[{name} 评估]")
        print(ev.to_string())
        strategies[name] = result
        eval_results[name] = ev

    # 对比分析
    print("\n" + "=" * 75)
    print("  策略对比分析")
    print("=" * 75)

    strategy_names = list(strategies_cfg.keys())

    def get_metric(eval_res, name, metric):
        return eval_res[name].loc[metric].values[0]

    metrics = {}
    for name in strategy_names:
        metrics[name] = {
            'cumret': get_metric(eval_results, name, '累积净值'),
            'ann_ret': get_metric(eval_results, name, '年化收益'),
            'max_dd': get_metric(eval_results, name, '最大回撤'),
            'calmar': get_metric(eval_results, name, '年化收益/回撤比'),
        }

    # 对比表
    header = f"{'指标':<20}" + "".join(f"{n:>18}" for n in strategy_names)
    print(f"\n{header}")
    print("-" * (20 + 18 * len(strategy_names)))
    print(f"{'累积净值':<20}" + "".join(f"{metrics[n]['cumret']:>18.4f}" for n in strategy_names))
    print(f"{'年化收益':<20}" + "".join(f"{metrics[n]['ann_ret']:>18}" for n in strategy_names))
    print(f"{'最大回撤':<20}" + "".join(f"{metrics[n]['max_dd']:>18}" for n in strategy_names))
    print(f"{'收益/回撤比':<20}" + "".join(f"{metrics[n]['calmar']:>18}" for n in strategy_names))

    # 相对改善
    orig_cumret = metrics['原版策略']['cumret']
    if orig_cumret:
        for name in strategy_names[1:]:
            ratio = metrics[name]['cumret'] / orig_cumret - 1
            print(f"\n  {name} vs 原版: 累积净值 {ratio:+.2%}")

    # 月度胜率
    print(f"\n[月度胜率]")
    for name in strategy_names:
        win_rate = (strategies[name]['选股下周期涨跌幅'] > 0).mean()
        print(f"  {name}: {win_rate:.2%}")

    # 策略相关性
    print(f"\n[策略相关性]")
    merged_all = None
    for i, name in enumerate(strategy_names):
        s = strategies[name][['交易日期', '选股下周期涨跌幅']].copy()
        s.columns = ['交易日期', name]
        if merged_all is None:
            merged_all = s
        else:
            merged_all = pd.merge(merged_all, s, on='交易日期')
    corr_matrix = merged_all[[n for n in strategy_names]].corr()
    print(corr_matrix.to_string())

    # 结论
    print(f"\n[结论]")
    for name in strategy_names[1:]:
        if metrics[name]['cumret'] > orig_cumret:
            print(f"  ✅ {name} 累积净值 ({metrics[name]['cumret']:.4f}) > 原版 ({orig_cumret:.4f})")
        else:
            print(f"  ❌ {name} 累积净值 ({metrics[name]['cumret']:.4f}) < 原版 ({orig_cumret:.4f})")

    print("\n" + "=" * 75)

    # 可视化
    if plot_style in ('compare', 'both'):
        plot_strategy_comparison(strategies, eval_results)
    if plot_style in ('raw', 'both'):
        plot_raw_style(strategies, eval_results)


if __name__ == '__main__':
    main()
