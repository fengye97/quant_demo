import math
from pathlib import Path

import pandas as pd

import web_app
from timing import run_timing_backtest, evaluate_timing_result, summarize_timing_windows


OUTPUT_PATH = Path(__file__).resolve().parent / 'probe_entry_report.md'

A_SHARE_CASES = [
    ('csi1000_timing', '中证1000ETF'),
    ('star50_timing', '科创50ETF'),
    ('chinext_timing', '创业板ETF'),
]

US_CASES = [
    ('nasdaq_timing', '纳指ETF'),
    ('sp500_timing', '标普500ETF'),
]

PROBE_OVERRIDES = {
    'probe_entry_exposure': 0.25,
    'probe_confirm_days': 1,
    'exposure_mode': 'staged',
}


def _to_float(v):
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        if not s or s == '—':
            return None
        s = s.replace('%', '').replace(',', '')
        try:
            return float(s)
        except ValueError:
            return None
    if isinstance(v, float) and math.isnan(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fmt_pct(v):
    n = _to_float(v)
    if n is None:
        return '—'
    return f"{n:.2f}%"


def fmt_num(v, digits=4):
    n = _to_float(v)
    if n is None:
        return '—'
    return f"{n:.{digits}f}"


def extract_metrics(result, benchmark_returns=None):
    metrics = evaluate_timing_result(result, benchmark_returns=benchmark_returns)
    windows = summarize_timing_windows(result, benchmark_returns=benchmark_returns)
    trade_count = int((result.get('signal_action') == 'buy').sum() + (result.get('signal_action') == 'sell').sum())
    return {
        'full': {
            'cumulative_return': metrics.get('累积净值'),
            'total_return_pct': metrics.get('总收益率'),
            'annual_return': metrics.get('年化收益'),
            'max_drawdown': metrics.get('最大回撤'),
            'calmar_ratio': metrics.get('年化收益/回撤比'),
            'avg_exposure': metrics.get('平均仓位'),
            'rebalance_count': metrics.get('调仓次数'),
            'fee_ratio': metrics.get('手续费占比'),
            'trade_count': trade_count,
        },
        'train': windows.get('train', {}).get('metrics', {}),
        'test': windows.get('validation_all', {}).get('metrics', {}),
        'april': windows.get('validation_april', {}).get('metrics', {}),
        'may': windows.get('validation_may', {}).get('metrics', {}),
    }


def run_a_share_case(strategy_id):
    result_base, _, strategy_base = web_app.run_timing_backtest_fresh(strategy_id)
    base = extract_metrics(result_base, benchmark_returns=result_base.attrs.get('benchmark_returns'))

    result_probe, _, strategy_probe = web_app.run_timing_backtest_fresh(strategy_id, **PROBE_OVERRIDES)
    probe = extract_metrics(result_probe, benchmark_returns=result_probe.attrs.get('benchmark_returns'))
    return strategy_base.get_display_name(), base, probe, strategy_base, strategy_probe


def run_us_case(strategy_id):
    web_app.ensure_us_timing_panel_loaded()
    panel = web_app.US_TIMING_PANEL.copy()

    strategy_base = web_app.build_us_timing_strategy(strategy_id)
    result_base = run_timing_backtest(strategy_base.run(panel.copy()), strategy_base)
    base = extract_metrics(result_base)

    strategy_probe = web_app.build_us_timing_strategy(strategy_id, **PROBE_OVERRIDES)
    result_probe = run_timing_backtest(strategy_probe.run(panel.copy()), strategy_probe)
    probe = extract_metrics(result_probe)
    return strategy_base.get_display_name(), base, probe, strategy_base, strategy_probe


def diff(a, b, key):
    av = _to_float(a.get(key))
    bv = _to_float(b.get(key))
    if av is None or bv is None:
        return '—'
    return fmt_num(bv - av, 4)


def render_table(title, rows):
    lines = [f'## {title}', '', '| 策略 | 版本 | 全样本收益 | 全样本回撤 | 训练集收益 | 测试集收益 | 测试集回撤 | 调仓次数 | 平均仓位 | 手续费占比 |', '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        label, version, summary = row
        full = summary['full']
        train = summary['train']
        test = summary['test']
        lines.append(
            f"| {label} | {version} | {fmt_pct(full.get('total_return_pct'))} | {fmt_pct(full.get('max_drawdown'))} | {fmt_pct(train.get('total_return_pct'))} | {fmt_pct(test.get('total_return_pct'))} | {fmt_pct(test.get('max_drawdown'))} | {fmt_num(full.get('rebalance_count'), 0)} | {fmt_num(full.get('avg_exposure'), 2)} | {fmt_pct(full.get('fee_ratio'))} |"
        )
    lines.append('')
    return lines


def render_delta_notes(group_name, case_reports):
    lines = [f'## {group_name}结论', '']
    for label, base, probe in case_reports:
        full_base = base['full']
        full_probe = probe['full']
        test_base = base['test']
        test_probe = probe['test']
        lines.append(
            f"- **{label}**：测试集收益 {fmt_pct(test_base.get('total_return_pct'))} → {fmt_pct(test_probe.get('total_return_pct'))}，"
            f"测试集回撤 {fmt_pct(test_base.get('max_drawdown'))} → {fmt_pct(test_probe.get('max_drawdown'))}，"
            f"全样本收益变化 {diff(full_base, full_probe, 'total_return_pct')} pct，"
            f"调仓次数变化 {diff(full_base, full_probe, 'rebalance_count')}。"
        )
    lines.append('')
    return lines


def main():
    web_app.ensure_timing_panel_loaded()
    web_app.ensure_us_timing_panel_loaded()

    content = [
        '# Probe Entry Trade-off Report',
        '',
        '对比对象：baseline（当前默认参数） vs probe_entry（共享 staged 试探建仓：0.25 初始试探仓位，1 天确认）。',
        '',
        '> 说明：区间展示会切片，但策略状态与指标 warm-up 仍基于区间起点之前的完整历史。',
        '',
    ]

    a_rows = []
    a_reports = []
    for strategy_id, label in A_SHARE_CASES:
        display_name, base, probe, _, _ = run_a_share_case(strategy_id)
        a_rows.append((label, 'baseline', base))
        a_rows.append((label, 'probe_entry', probe))
        a_reports.append((label, base, probe))

    us_rows = []
    us_reports = []
    for strategy_id, label in US_CASES:
        display_name, base, probe, _, _ = run_us_case(strategy_id)
        us_rows.append((label, 'baseline', base))
        us_rows.append((label, 'probe_entry', probe))
        us_reports.append((label, base, probe))

    content.extend(render_table('A股策略对比', a_rows))
    content.extend(render_delta_notes('A股', a_reports))
    content.extend(render_table('美股策略对比', us_rows))
    content.extend(render_delta_notes('美股', us_reports))

    OUTPUT_PATH.write_text('\n'.join(content), encoding='utf-8')
    print(OUTPUT_PATH)


if __name__ == '__main__':
    main()
