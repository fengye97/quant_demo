"""
选股策略 — 原版策略（行业估值 + bias反转 + 小市值）。

历史回测累积净值 5223x，为当前策略库中收益最高的策略。

策略库位于 strategies/ 目录：
  OriginalStrategy     — 原版策略（当前使用，历史收益最高）
  ChanEnhancedStrategy — 缠论增强 v1.1
  ChanOnlyStrategy     — 纯缠论 v1.2
  MethodAStrategy      — Method A 日线流水线 v2.0

可视化：
  python3 choose_stock.py --plot compare   # 详细对比图（默认）
  python3 choose_stock.py --plot raw       # 简洁风格
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np

from strategies.original import OriginalStrategy
from backtest import load_data, select_and_backtest, strategy_evaluate
from visualization import plot_strategy_comparison, plot_raw_style

warnings.filterwarnings('ignore')
pd.set_option('expand_frame_repr', False)


def parse_arg(flag, default='compare'):
    """解析 --flag value 形式的命令行参数"""
    try:
        idx = sys.argv.index(flag)
        return sys.argv[idx + 1]
    except (ValueError, IndexError):
        return default


def main():
    plot_style = parse_arg('--plot', 'compare')

    print("=" * 60)
    print("  选股策略 — 原版（行业估值 + bias反转 + 小市值）")
    print("  历史回测累积净值: 5223x")
    print("=" * 60)

    # 加载数据
    df = load_data('stock_data.csv')

    # 原版策略
    strategy = OriginalStrategy()
    df = strategy.run(df)

    # 回测
    result = select_and_backtest(df, strategy)
    result.to_csv('选股策略详情.csv', encoding='gbk', index=False)

    # 评估
    ev = strategy_evaluate(result)
    print("\n[策略评估]")
    print(ev.to_string())

    # 可视化
    strategies = {'原版策略': result}
    eval_results = {'原版策略': ev}

    if plot_style in ('compare', 'both'):
        plot_strategy_comparison(strategies, eval_results)
    if plot_style in ('raw', 'both'):
        plot_raw_style(strategies, eval_results)


if __name__ == '__main__':
    main()
