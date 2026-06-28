#!/usr/bin/env python
"""Phase 2.2: holdout 报告生成

读取 strategy/best_profile_{name}.json，用对应参数在完整 panel（不做 cutoff 截断）
上跑一次回测，然后用 filter_timing_result(start=2025-12-01) 触发 cold-start
重算 holdout 区间的资金路径，输出 strategy/holdout_report_{name}.md。

holdout 报告纯展示，不参与选优；CLAUDE.md Rule 13 (capital reset) 与
"holdout 严格只读" 的承诺都在这里落地。

运行方式:
    /Users/fatcat/opt/anaconda3/bin/python scripts/build_holdout_reports.py
    /Users/fatcat/opt/anaconda3/bin/python scripts/build_holdout_reports.py --only csi1000_timing
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_DEMO_ROOT = os.path.join(_REPO_ROOT, 'stock_trade_demo')
sys.path.insert(0, _DEMO_ROOT)
from utils.atomic_io import atomic_write_text as _atomic_write_text, atomic_writer as _atomic_writer

# 复用 walk_forward_train 中的 spec 与默认 realism
sys.path.insert(0, _HERE)
from walk_forward_train import (  # noqa: E402
    SHARED_REALISM,
    STRATEGY_SPECS,
    TRAINING_CUTOFF,
    _extract_metric_floats,
    _maybe_apply_us_clean_window,
    _sanitize_nan,
)

from index_data import build_index_panel, build_us_index_panel  # noqa: E402
from timing.backtest import (  # noqa: E402
    evaluate_timing_result,
    filter_timing_result,
    run_timing_backtest,
)


HOLDOUT_START = pd.Timestamp('2025-12-01')
OUTPUT_DIR = os.path.join(_REPO_ROOT, 'strategy')


def _load_best_profile(strategy_id: str):
    fp = os.path.join(OUTPUT_DIR, f'best_profile_{strategy_id}.json')
    if not os.path.exists(fp):
        return None
    with open(fp) as f:
        return json.load(f)


def _build_strategy_from_profile(strategy_id: str, profile: dict):
    spec = STRATEGY_SPECS[strategy_id]
    all_params = dict(profile['all_params'])
    instance = spec['cls'](**all_params)
    for k, v in SHARED_REALISM.items():
        setattr(instance, k, v)
    return instance


def _render_md(strategy_id: str, profile: dict, holdout_metrics: dict,
               holdout_rows: int, holdout_window: tuple) -> str:
    tuned = profile.get('tuned_params', {})
    train_score = profile.get('score')
    train_windows = profile.get('window_metrics', {})

    lines = []
    lines.append(f"# Holdout 报告 — {strategy_id}")
    lines.append('')
    lines.append(f"- 生成时间: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- 训练 cutoff: {profile.get('training_cutoff')}")
    lines.append(f"- Holdout 区间: **{holdout_window[0].strftime('%Y-%m-%d')} ~ "
                 f"{holdout_window[1].strftime('%Y-%m-%d')}**  ({holdout_rows} bars)")
    lines.append(f"- Profile 来源: strategy/best_profile_{strategy_id}.json")
    lines.append('')

    lines.append('## 调参网格选出的最优参数')
    lines.append('| 参数 | 取值 |')
    lines.append('|------|------|')
    for k, v in tuned.items():
        lines.append(f'| `{k}` | `{v}` |')
    lines.append('')
    if strategy_id == 'macro_v32_timing':
        lines.append('> Regime 说明：`fed_block_weight` 控制 Fed 因子块权重；`restrictive_threshold` 表示 Fed 仍偏紧的判定线；`pivot_relief` 表示政策转松时对危机扣分的缓冲强度。')
        lines.append('')

    lines.append('## 训练区评分（来自 walk-forward 选优阶段）')
    lines.append(f"- 评分公式: `{profile.get('score_formula')}`")
    lines.append(f"- 综合分: **{train_score:.4f}**  (maxDD 阈值 {profile.get('maxdd_threshold')})")
    lines.append('')
    lines.append('| 窗口 | Calmar | 年化收益 | 最大回撤 | 平均仓位 | 调仓次数 |')
    lines.append('|------|--------|----------|----------|----------|----------|')
    for win_name, wm in train_windows.items():
        if wm is None:
            continue
        lines.append(
            f"| {win_name} | {wm['calmar']:.3f} | {wm['annual_return']:.2%} | "
            f"{wm['max_drawdown']:.2%} | {wm['avg_exposure']:.2%} | {wm['rebalance_count']} |"
        )
    lines.append('')

    lines.append('## Holdout 区间表现（**只读，未参与选优**）')
    if holdout_metrics is None:
        lines.append('> Holdout 窗口无可交易 bar，跳过。')
    else:
        lines.append('| 指标 | 取值 |')
        lines.append('|------|------|')
        lines.append(f"| 累积净值 | {holdout_metrics['final_nav']:.4f} |")
        lines.append(f"| 年化收益 | {holdout_metrics['annual_return']:.2%} |")
        lines.append(f"| 最大回撤 | {holdout_metrics['max_drawdown']:.2%} |")
        lines.append(f"| Calmar | {holdout_metrics['calmar']:.3f} |")
        lines.append(f"| 平均仓位 | {holdout_metrics['avg_exposure']:.2%} |")
        lines.append(f"| 调仓次数 | {holdout_metrics['rebalance_count']} |")
        lines.append('')
        lines.append('### 训练区 vs Holdout Calmar 对比')
        lines.append('| 窗口 | Calmar |')
        lines.append('|------|--------|')
        for win_name, wm in train_windows.items():
            if wm is None:
                continue
            lines.append(f"| 训练区 {win_name} | {wm['calmar']:.3f} |")
        lines.append(f"| **Holdout** | **{holdout_metrics['calmar']:.3f}** |")
        lines.append('')
        lines.append('> 如果 holdout Calmar 显著低于训练区，说明该参数对训练区过拟合；')
        lines.append('> 如果接近或更高，说明 walk-forward 选出的参数在 OOS 上稳定。')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('*由 `scripts/build_holdout_reports.py` 自动生成。Holdout 报告只读不参与选优；')
    lines.append('如需调整 holdout 起点，请同步修改 `web_app.py` 的 `HOLDOUT_START`。*')
    return '\n'.join(lines) + '\n'


def _evaluate_holdout(strategy_id: str, profile: dict, panel: pd.DataFrame):
    instance = _build_strategy_from_profile(strategy_id, profile)
    signal_df = instance.run(panel.copy())
    full_result = run_timing_backtest(signal_df, instance, benchmark_returns=None)
    holdout_end = pd.to_datetime(full_result['交易日期'].max())
    sliced = filter_timing_result(full_result, start_date=HOLDOUT_START, end_date=holdout_end)
    if len(sliced) == 0:
        return None, 0, (HOLDOUT_START, holdout_end)
    metrics = evaluate_timing_result(sliced, benchmark_returns=None, reset_capital=True)
    return _extract_metric_floats(metrics), len(sliced), (HOLDOUT_START, holdout_end)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', action='append', default=None)
    args = parser.parse_args()

    target_ids = args.only or list(STRATEGY_SPECS.keys())
    cn_ids = [sid for sid in target_ids if STRATEGY_SPECS[sid]['panel'] == 'cn']
    us_ids = [sid for sid in target_ids if STRATEGY_SPECS[sid]['panel'] == 'us']

    cn_panel = us_panel = None
    if cn_ids:
        print('[load] CN panel ...', flush=True)
        cn_panel = build_index_panel()
        cn_panel['交易日期'] = pd.to_datetime(cn_panel['交易日期'])
    if us_ids:
        print('[load] US panel ...', flush=True)
        us_panel = build_us_index_panel()
        us_panel['交易日期'] = pd.to_datetime(us_panel['交易日期'])

    summary = []
    for sid in target_ids:
        profile = _load_best_profile(sid)
        if profile is None:
            print(f"[{sid}] best_profile 缺失，跳过；请先运行 walk_forward_train.py")
            continue
        panel = cn_panel if STRATEGY_SPECS[sid]['panel'] == 'cn' else us_panel
        if panel is None:
            print(f"[{sid}] panel 未加载，跳过")
            continue
        clean_window_meta = None
        if STRATEGY_SPECS[sid]['panel'] == 'us':
            panel, clean_window_meta = _maybe_apply_us_clean_window(panel, sid)
        print(f"[{sid}] 评估 holdout (start={HOLDOUT_START.strftime('%Y-%m-%d')}) ...")
        if clean_window_meta:
            print(f"  [clean-window] index={clean_window_meta['index_id']} start={clean_window_meta['clean_start']} path={clean_window_meta['preferred_runtime_path']}")
        holdout_metrics, holdout_rows, holdout_window = _evaluate_holdout(sid, profile, panel)
        md = _render_md(sid, profile, holdout_metrics, holdout_rows, holdout_window)
        out_path = os.path.join(OUTPUT_DIR, f'holdout_report_{sid}.md')
        _atomic_write_text(out_path, md)
        # 同时落结构化 JSON：web /api/*/latest_signal 直接读，不依赖 md 解析
        json_payload = {
            'strategy_id': sid,
            'generated_at': datetime.now().isoformat(timespec='seconds'),
            'training_cutoff': profile.get('training_cutoff'),
            'holdout_start': holdout_window[0].strftime('%Y-%m-%d'),
            'holdout_end': holdout_window[1].strftime('%Y-%m-%d'),
            'holdout_bars': holdout_rows,
            'training_score': profile.get('score'),
            'training_window_metrics': profile.get('window_metrics', {}),
            'holdout_metrics': holdout_metrics,
        }
        json_path = os.path.join(OUTPUT_DIR, f'holdout_report_{sid}.json')
        with _atomic_writer(json_path, 'w', encoding='utf-8') as _f:
            json.dump(_sanitize_nan(json_payload), _f, ensure_ascii=False, indent=2, default=float)
        print(f"  -> {out_path}")
        print(f"  -> {json_path}")
        if holdout_metrics is not None:
            print(f"     holdout: calmar={holdout_metrics['calmar']:.3f}  "
                  f"maxDD={holdout_metrics['max_drawdown']:.3%}  "
                  f"annRet={holdout_metrics['annual_return']:.3%}  "
                  f"final_nav={holdout_metrics['final_nav']:.4f}")
        summary.append({
            'strategy': sid,
            'rows': holdout_rows,
            'metrics': holdout_metrics,
        })

    print('\n=== Holdout summary ===')
    for s in summary:
        m = s['metrics']
        if m is None:
            print(f"  {s['strategy']:>18}  rows={s['rows']}  no metrics")
            continue
        print(f"  {s['strategy']:>18}  rows={s['rows']:>4}  "
              f"calmar={m['calmar']:>6.3f}  maxDD={m['max_drawdown']:>7.2%}  "
              f"annRet={m['annual_return']:>7.2%}  nav={m['final_nav']:.4f}")


if __name__ == '__main__':
    main()
