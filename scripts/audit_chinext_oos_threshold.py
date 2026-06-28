#!/usr/bin/env python
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_DEMO_ROOT = os.path.join(_REPO_ROOT, 'stock_trade_demo')
sys.path.insert(0, _DEMO_ROOT)
sys.path.insert(0, _HERE)

from build_holdout_reports import HOLDOUT_START  # noqa: E402
from index_data import build_index_panel  # noqa: E402
from timing.backtest import evaluate_timing_result, filter_timing_result, run_timing_backtest  # noqa: E402
from walk_forward_train import (  # noqa: E402
    FULL_NAV_FLOOR_RATIO,
    SHARED_REALISM,
    STRATEGY_SPECS,
    _evaluate_one,
    _extract_metric_floats,
    _slice_panel_pre_cutoff,
)

STRATEGY_ID = 'chinext_timing'
OUTPUT_DIR = os.path.join(_REPO_ROOT, 'strategy')
PROFILE_PATH = os.path.join(OUTPUT_DIR, f'best_profile_{STRATEGY_ID}.json')
CSV_PATH = os.path.join(OUTPUT_DIR, 'chinext_threshold_oos_sensitivity.csv')
MD_PATH = os.path.join(OUTPUT_DIR, 'chinext_threshold_oos_sensitivity.md')
THRESHOLDS = [0.0, 0.005, 0.01, 0.015, 0.02]


def load_profile() -> dict:
    with open(PROFILE_PATH) as f:
        return json.load(f)


def build_temp_spec(profile: dict) -> dict:
    base_params = dict(profile['all_params'])
    base_params.pop('momentum_threshold', None)
    return {
        'cls': STRATEGY_SPECS[STRATEGY_ID]['cls'],
        'panel': 'cn',
        'base': base_params,
        'grid': {'momentum_threshold': THRESHOLDS},
    }


def build_strategy(profile: dict, threshold: float):
    cls = STRATEGY_SPECS[STRATEGY_ID]['cls']
    params = dict(profile['all_params'])
    params['momentum_threshold'] = threshold
    instance = cls(**params)
    for k, v in SHARED_REALISM.items():
        setattr(instance, k, v)
    return instance


def evaluate_holdout(profile: dict, panel_full: pd.DataFrame, threshold: float) -> tuple[dict | None, int, str]:
    instance = build_strategy(profile, threshold)
    signal_df = instance.run(panel_full.copy())
    full_result = run_timing_backtest(signal_df, instance, benchmark_returns=None)
    holdout_end = pd.to_datetime(full_result['交易日期'].max())
    sliced = filter_timing_result(full_result, start_date=HOLDOUT_START, end_date=holdout_end)
    if len(sliced) == 0:
        return None, 0, f'{HOLDOUT_START.strftime("%Y-%m-%d")} ~ {holdout_end.strftime("%Y-%m-%d")}'
    metrics = evaluate_timing_result(sliced, benchmark_returns=None, reset_capital=True)
    return _extract_metric_floats(metrics), len(sliced), f'{HOLDOUT_START.strftime("%Y-%m-%d")} ~ {holdout_end.strftime("%Y-%m-%d")}'


def render_md(profile: dict, report_df: pd.DataFrame, holdout_window: str) -> str:
    best_thr = profile['tuned_params']['momentum_threshold']
    lines = [f'# Chinext momentum_threshold OOS sensitivity — {datetime.now().isoformat(timespec="seconds")}', '']
    lines.append('## Context')
    lines.append(f'- training best params: `{profile["tuned_params"]}`')
    lines.append(f'- current best threshold: **{best_thr}**')
    lines.append(f'- training floor: **{FULL_NAV_FLOOR_RATIO:.2f} × default_full_nav = {profile["default_full_nav"]:.4f}**')
    lines.append(f'- holdout window: **{holdout_window}**')
    lines.append('')
    lines.append('> 说明：由于 best 值是 `0.0`，这里不用“乘法 ±20%”，而改成绝对扰动 `0.000 / 0.005 / 0.010 / 0.015 / 0.020`。')
    lines.append('> holdout 指标只读展示，不参与选优，不回写 `best_profile`。')
    lines.append('')
    lines.append('## Training + holdout comparison')
    lines.append('| threshold | train_score | train_discarded | train_6m_calmar | train_1y_calmar | train_full_calmar | holdout_calmar | holdout_annret | holdout_maxdd | holdout_final_nav | holdout_rows |')
    lines.append('| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for _, row in report_df.iterrows():
        discarded = row['train_discarded'] or ''
        lines.append(
            f"| {row['momentum_threshold']:.3f} | {row['train_score']} | {discarded} | {row['train_recent_6m_calmar']} | {row['train_recent_1y_calmar']} | {row['train_full_pre_cutoff_calmar']} | {row['holdout_calmar']} | {row['holdout_annret']} | {row['holdout_maxdd']} | {row['holdout_final_nav']} | {int(row['holdout_rows'])} |"
        )
    lines.append('')

    best_row = report_df.loc[report_df['momentum_threshold'] == best_thr].iloc[0]
    top_holdout = report_df.sort_values('holdout_calmar', ascending=False).iloc[0]
    lines.append('## Takeaways')
    if float(top_holdout['momentum_threshold']) == float(best_thr):
        lines.append(f'- 在这组最小扰动里，**训练 best 的 `{best_thr:.3f}` 同时也是 holdout Calmar 最优**。')
    else:
        lines.append(f'- 在这组最小扰动里，holdout Calmar 最优的是 `{top_holdout["momentum_threshold"]:.3f}`，而训练 best 是 `{best_thr:.3f}`。')
    lines.append(f'- 训练 best `{best_thr:.3f}` 的 holdout: `Calmar={best_row["holdout_calmar"]}`, `annRet={best_row["holdout_annret"]}`, `maxDD={best_row["holdout_maxdd"]}`, `final_nav={best_row["holdout_final_nav"]}`。')
    lines.append('- 若某个阈值在训练区已被 `full_nav_floor` 或 `maxDD` 约束淘汰，即使 holdout 看起来更好，也**不能**据此回写为 best。')
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    profile = load_profile()
    spec = build_temp_spec(profile)

    panel_full = build_index_panel()
    panel_full['交易日期'] = pd.to_datetime(panel_full['交易日期'])
    panel_pre_cutoff = _slice_panel_pre_cutoff(panel_full)

    rows = []
    holdout_window = ''
    for threshold in THRESHOLDS:
        train_res = _evaluate_one(spec, {'momentum_threshold': threshold}, panel_pre_cutoff, default_full_nav=profile['default_full_nav'])
        holdout_metrics, holdout_rows, holdout_window = evaluate_holdout(profile, panel_full, threshold)
        holdout_metrics = holdout_metrics or {}
        rows.append({
            'momentum_threshold': threshold,
            'train_score': train_res['score'],
            'train_discarded': train_res['discarded'] or '',
            'train_recent_6m_calmar': (train_res['windows'].get('recent_6m') or {}).get('calmar'),
            'train_recent_1y_calmar': (train_res['windows'].get('recent_1y') or {}).get('calmar'),
            'train_full_pre_cutoff_calmar': (train_res['windows'].get('full_pre_cutoff') or {}).get('calmar'),
            'holdout_calmar': holdout_metrics.get('calmar'),
            'holdout_annret': holdout_metrics.get('annual_return'),
            'holdout_maxdd': holdout_metrics.get('max_drawdown'),
            'holdout_final_nav': holdout_metrics.get('final_nav'),
            'holdout_avg_exposure': holdout_metrics.get('avg_exposure'),
            'holdout_rebalance_count': holdout_metrics.get('rebalance_count'),
            'holdout_rows': holdout_rows,
        })

    report_df = pd.DataFrame(rows).sort_values('momentum_threshold').reset_index(drop=True)
    report_df.to_csv(CSV_PATH, index=False)
    with open(MD_PATH, 'w') as f:
        f.write(render_md(profile, report_df, holdout_window))

    print(f'[ok] csv -> {CSV_PATH}')
    print(f'[ok] md  -> {MD_PATH}')
    print(report_df[['momentum_threshold', 'train_score', 'train_discarded', 'holdout_calmar', 'holdout_annret', 'holdout_maxdd', 'holdout_final_nav']].to_string(index=False))


if __name__ == '__main__':
    main()
