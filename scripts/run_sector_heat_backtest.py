"""
离线对比：原版策略 vs SectorHeatStrategy（top / bottom 两种模式）。

运行：
  cd stock_trade_demo
  python ../scripts/run_sector_heat_backtest.py

输出：
  strategy/backtest_sector_heat.csv   — 逐期净值对比
  strategy/sector_heat_summary.md     — 性能摘要 Markdown
"""

import sys
import os

DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'stock_trade_demo')
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
os.chdir(DEMO_DIR)
sys.path.insert(0, DEMO_DIR)
sys.path.insert(0, ROOT_DIR)

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from backtest import load_data, select_and_backtest, strategy_evaluate
from strategies.original import OriginalStrategy
from strategies.sector_heat import SectorHeatStrategy


def ev_get(ev_df, key):
    """从 strategy_evaluate 返回的转置 DataFrame 中取指标值。"""
    if key in ev_df.index:
        val = ev_df.loc[key, 0]
        if isinstance(val, str):
            # 去掉 % 号后尝试转 float
            try:
                return float(val.replace('%', '')) / 100.0
            except Exception:
                return float('nan')
        try:
            return float(val)
        except Exception:
            return float('nan')
    return float('nan')


def parse_cumret(ev_df):
    """从 '累积净值' 取总收益倍数。"""
    return ev_get(ev_df, '累积净值')


def run_one(df, strategy, label):
    df_ready = strategy.run(df.copy())
    result = select_and_backtest(df_ready, strategy)
    ev = strategy_evaluate(result)   # returns DataFrame.T (index=指标名, columns=[0])

    ann  = ev_get(ev, '年化收益')
    mdd  = ev_get(ev, '最大回撤')
    calmar = ev_get(ev, '年化收益/回撤比')
    cum  = parse_cumret(ev)

    print(
        f"  {label:<30} | 年化={ann:.1%}  最大回撤={mdd:.1%}  "
        f"Calmar={calmar:.2f}  累积净值={cum:.1f}x"
    )
    nav = result.set_index('交易日期')['累积净值']
    nav.name = label
    return nav, ev


def fmt_pct(v):
    if np.isnan(v): return 'N/A'
    return f'{v:.1%}'


def fmt_f(v, d=2):
    if np.isnan(v): return 'N/A'
    return f'{v:.{d}f}'


def main():
    print("加载数据...")
    df = load_data()
    print(f"数据范围: {df['交易日期'].min().date()} ~ {df['交易日期'].max().date()}")

    configs = [
        ('原版策略 (baseline)',       OriginalStrategy()),
        ('行业热度-top-30%',          SectorHeatStrategy(sector_heat_cutoff=0.30, sector_heat_mode='top')),
        ('行业热度-top-40%',          SectorHeatStrategy(sector_heat_cutoff=0.40, sector_heat_mode='top')),
        ('行业热度-top-50%',          SectorHeatStrategy(sector_heat_cutoff=0.50, sector_heat_mode='top')),
        ('行业热度-bottom-30%',       SectorHeatStrategy(sector_heat_cutoff=0.30, sector_heat_mode='bottom')),
        ('行业热度-bottom-40%',       SectorHeatStrategy(sector_heat_cutoff=0.40, sector_heat_mode='bottom')),
    ]

    print("\n===== 全历史回测结果 =====")
    navs = {}
    evals = {}
    for label, strategy in configs:
        nav, ev = run_one(df, strategy, label)
        navs[label] = nav
        evals[label] = ev

    nav_df = pd.DataFrame(navs).sort_index()
    nav_df.index = pd.to_datetime(nav_df.index)

    out_nav = os.path.join(ROOT_DIR, 'strategy', 'backtest_sector_heat.csv')
    nav_df.reset_index().rename(columns={'index': '交易日期'}).to_csv(
        out_nav, index=False, encoding='utf-8-sig')
    print(f"\n已保存净值曲线 → {out_nav}")

    # ── 摘要表 ──
    rows = []
    for label, ev in evals.items():
        ann  = ev_get(ev, '年化收益')
        mdd  = ev_get(ev, '最大回撤')
        cal  = ev_get(ev, '年化收益/回撤比')
        cum  = parse_cumret(ev)
        rows.append({
            '策略': label,
            '年化收益': fmt_pct(ann),
            '最大回撤': fmt_pct(mdd),
            'Calmar': fmt_f(cal),
            '累积净值倍': fmt_f(cum, 1),
        })

    # 手动生成 Markdown 表（避免 tabulate 依赖）
    summary_df = pd.DataFrame(rows)
    cols = summary_df.columns.tolist()
    header = '| ' + ' | '.join(cols) + ' |'
    sep    = '| ' + ' | '.join(['---'] * len(cols)) + ' |'
    body   = '\n'.join(
        '| ' + ' | '.join(str(summary_df.iloc[i][c]) for c in cols) + ' |'
        for i in range(len(summary_df))
    )
    table_md = '\n'.join([header, sep, body])

    # 自动结论
    base_ann = ev_get(evals['原版策略 (baseline)'], '年化收益')
    variants = [(k, ev_get(v, '年化收益')) for k, v in evals.items() if k != '原版策略 (baseline)']
    best_label, best_ann = max(variants, key=lambda x: x[1] if not np.isnan(x[1]) else -999)
    delta = best_ann - base_ann if not (np.isnan(best_ann) or np.isnan(base_ann)) else float('nan')

    if not np.isnan(delta):
        sign = '+' if delta >= 0 else ''
        conclusion_lines = [
            f'- 最优配置: **{best_label}**，年化 {fmt_pct(best_ann)}，相对原版 {sign}{fmt_pct(delta)}\n',
        ]
        if delta > 0.005:
            conclusion_lines.append('- 结论: 行业热度过滤有**正向贡献**，建议纳入候选策略池。\n')
        elif delta > -0.005:
            conclusion_lines.append('- 结论: 行业热度过滤效果**中性**，增益不显著。\n')
        else:
            conclusion_lines.append('- 结论: 行业热度过滤**拖累**了收益，需调整方向或弃用。\n')
    else:
        conclusion_lines = ['- 结论: 无法计算有效结论（数值 nan）\n']

    md_content = (
        '# Sector Heat Strategy — 回测摘要\n\n'
        f'数据: {df["交易日期"].min().date()} ~ {df["交易日期"].max().date()}\n\n'
        + table_md + '\n\n'
        '## 结论\n\n'
        + ''.join(conclusion_lines)
    )

    out_md = os.path.join(ROOT_DIR, 'strategy', 'sector_heat_summary.md')
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"已保存摘要报告 → {out_md}")

    # 打印摘要
    print('\n' + md_content)


if __name__ == '__main__':
    main()
