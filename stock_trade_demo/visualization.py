"""
可视化模块 — 策略回测结果图表。

两种风格:
  plot_strategy_comparison  — 详细对比图（3面板：资金曲线+回撤+年度收益）
  plot_raw_style            — 简洁风格（单面板资金曲线，同 choose_stock_raw.py）
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# 中文字体
plt.rcParams['font.sans-serif'] = [
    'PingFang SC', 'Heiti SC', 'STHeiti',
    'Arial Unicode MS', 'SimHei', 'sans-serif'
]
plt.rcParams['axes.unicode_minus'] = False


def plot_strategy_comparison(strategies, eval_results, save_path=None):
    """
    详细对比图 — 3 面板。

    Panel 1: 资金曲线（log scale），含牛熊背景标记
    Panel 2: 回撤曲线，标注最大回撤
    Panel 3: 年度收益柱状图对比
    底部附关键指标摘要表
    """
    if save_path is None:
        save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '策略对比图表.png'
        )
    if not strategies:
        return

    strategy_names = list(strategies.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    color_map = dict(zip(strategy_names, colors[:len(strategy_names)]))

    fig, axes = plt.subplots(3, 1, figsize=(18, 14))
    fig.suptitle('选股策略对比分析', fontsize=16, fontweight='bold', y=0.98)

    # ── Panel 1: 累积净值曲线 (log scale) ──
    ax1 = axes[0]
    for name in strategy_names:
        s = strategies[name]
        ax1.plot(s['交易日期'], s['累积净值'], color=color_map[name],
                 linewidth=1.2, alpha=0.9, label=name)
    ax1.set_ylabel('累积净值 (log)', fontsize=11)
    ax1.set_title('资金曲线', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax1.set_yscale('log')
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    # 牛熊背景（基于第一个策略的资金曲线）
    for name in strategy_names:
        s = strategies[name].copy()
        if '选股下周期涨跌幅' not in s.columns:
            continue
        s['cum_ret'] = (1 + s['选股下周期涨跌幅']).cumprod()
        s['ma12'] = s['cum_ret'].rolling(12, min_periods=1).mean()
        bear_starts = s[s['cum_ret'] < s['ma12']]
        if len(bear_starts) > 0:
            for date in bear_starts['交易日期']:
                ax1.axvspan(date, date + pd.Timedelta(days=28),
                           color='red', alpha=0.03, lw=0)
        break

    # ── Panel 2: 回撤曲线 ──
    ax2 = axes[1]
    for name in strategy_names:
        s = strategies[name]
        cum = s['累积净值'].values
        peak = np.maximum.accumulate(cum)
        dd = (cum / peak - 1) * 100
        ax2.fill_between(s['交易日期'], 0, dd, color=color_map[name],
                         alpha=0.35, linewidth=0.8, label=name)
        ax2.plot(s['交易日期'], dd, color=color_map[name],
                 linewidth=0.8, alpha=0.8)
    ax2.set_ylabel('回撤 (%)', fontsize=11)
    ax2.set_title('回撤曲线', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower left', fontsize=9, framealpha=0.9)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.grid(True, alpha=0.3)

    # 标注各策略最大回撤
    for name in strategy_names:
        s = strategies[name]
        cum = s['累积净值'].values
        peak = np.maximum.accumulate(cum)
        dd = cum / peak - 1
        min_idx = np.argmin(dd)
        ax2.annotate(f'{name}\n{dd[min_idx]:.1%}',
                     xy=(s['交易日期'].iloc[min_idx], dd[min_idx] * 100),
                     fontsize=7, color=color_map[name], alpha=0.8,
                     ha='center', va='top')

    # ── Panel 3: 年度收益柱状图 ──
    ax3 = axes[2]
    merged_returns = None
    for name in strategy_names:
        s = strategies[name][['交易日期', '选股下周期涨跌幅']].copy()
        s.columns = ['交易日期', name]
        if merged_returns is None:
            merged_returns = s
        else:
            merged_returns = pd.merge(merged_returns, s, on='交易日期', how='outer')
    merged_returns = merged_returns.dropna()
    merged_returns['年份'] = merged_returns['交易日期'].dt.year
    years = sorted(merged_returns['年份'].unique())

    tick_years = years[::3] if len(years) > 25 else years
    x = np.arange(len(years))
    bar_width = 0.8 / len(strategy_names)

    for i, name in enumerate(strategy_names):
        yearly = merged_returns.groupby('年份')[name].apply(
            lambda x: (1 + x).prod() - 1
        )
        yearly = yearly.reindex(years, fill_value=0)
        offset = (i - len(strategy_names) / 2 + 0.5) * bar_width
        ax3.bar(x + offset, yearly.values * 100, bar_width,
                color=color_map[name], alpha=0.8, label=name)
        for j, (yr, val) in enumerate(zip(years, yearly.values)):
            if abs(val) > 0.5:
                ax3.annotate(f'{yr}', (x[j] + offset, val * 100),
                            fontsize=5, ha='center',
                            va='bottom' if val > 0 else 'top',
                            rotation=90, alpha=0.6)

    ax3.set_ylabel('年度收益 (%)', fontsize=11)
    ax3.set_title('年度收益对比', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(y) for y in tick_years], rotation=45, fontsize=8)
    ax3.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax3.grid(True, alpha=0.3, axis='y')

    # 关键指标表
    table_text = "关键指标:\n"
    for name in strategy_names:
        if name in eval_results:
            er = eval_results[name]
            cumret = er.loc['累积净值'].values[0] if '累积净值' in er.index else 'N/A'
            ann_ret = er.loc['年化收益'].values[0] if '年化收益' in er.index else 'N/A'
            max_dd = er.loc['最大回撤'].values[0] if '最大回撤' in er.index else 'N/A'
            table_text += f"\n{name}: 净值={cumret}  年化={ann_ret}  DD={max_dd}"

    fig.text(0.02, 0.01, table_text, fontsize=7, family='monospace',
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n[图表已保存] → {save_path}")
    plt.close()


def plot_raw_style(strategies, eval_results=None, save_dir=None):
    """
    简洁风格 — 单面板资金曲线。

    与 choose_stock_raw.py 的 plt.plot() 风格一致：
    线性坐标、简单线条、关键指标标注在左上角。

    每个策略生成一张独立图表。
    """
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(__file__))

    for name, s in strategies.items():
        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(s['交易日期'], s['资金曲线'],
                color='#1f77b4', linewidth=1.2)
        ax.set_title(f'{name} — 资金曲线', fontsize=13, fontweight='bold')
        ax.set_ylabel('资金曲线', fontsize=11)
        ax.axhline(y=1, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.grid(True, alpha=0.3)

        # 标注关键指标
        if eval_results and name in eval_results:
            er = eval_results[name]
            cumret = er.loc['累积净值'].values[0] if '累积净值' in er.index else 'N/A'
            ann_ret = er.loc['年化收益'].values[0] if '年化收益' in er.index else 'N/A'
            max_dd = er.loc['最大回撤'].values[0] if '最大回撤' in er.index else 'N/A'
            info = f"累积净值: {cumret}  年化收益: {ann_ret}  最大回撤: {max_dd}"
            ax.text(0.02, 0.97, info, transform=ax.transAxes, fontsize=8,
                    verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        path = os.path.join(save_dir, f'选股对比图_{name}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"[图表已保存] → {path}")
        plt.close()
