#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from typing import Iterable

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRATEGY_DIR = os.path.join(REPO_ROOT, 'strategy')

DEFAULT_IDS = ['chinext_timing', 'csi1000_timing', 'star50_timing']


def load_profile(strategy_id: str) -> dict:
    path = os.path.join(STRATEGY_DIR, f'best_profile_{strategy_id}.json')
    with open(path) as f:
        return json.load(f)


def load_log(strategy_id: str) -> pd.DataFrame:
    path = os.path.join(STRATEGY_DIR, f'walk_forward_log_{strategy_id}.csv')
    return pd.read_csv(path)


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce')
    return out


def build_local_slice(df: pd.DataFrame, tuned_params: dict) -> pd.DataFrame:
    out = df.copy()
    for key, best_val in tuned_params.items():
        if key not in out.columns:
            continue
        col = pd.to_numeric(out[key], errors='coerce')
        best_num = pd.to_numeric(pd.Series([best_val]), errors='coerce').iloc[0]
        if pd.notna(best_num):
            unique_vals = sorted(v for v in col.dropna().unique())
            if len(unique_vals) <= 1:
                continue
            try:
                idx = unique_vals.index(best_num)
            except ValueError:
                continue
            if best_num == 0:
                allowed = set(unique_vals)
            else:
                allowed = set(unique_vals[max(0, idx - 1): min(len(unique_vals), idx + 2)])
            out = out[col.isin(allowed)]
        else:
            out = out[out[key] == best_val]
    return out


def summarize_by_param(local_df: pd.DataFrame, tuned_params: dict) -> pd.DataFrame:
    rows = []
    for key, best_val in tuned_params.items():
        if key not in local_df.columns:
            continue
        grp = local_df.groupby(key, dropna=False)
        for val, seg in grp:
            valid = seg[seg['discarded'].fillna('') == ''].copy()
            best_score = valid['score'].max() if not valid.empty else float('nan')
            mean_score = valid['score'].mean() if not valid.empty else float('nan')
            top = valid.sort_values('score', ascending=False).head(1) if not valid.empty else seg.head(1)
            row = top.iloc[0].to_dict()
            rows.append({
                'parameter': key,
                'value': val,
                'is_best_value': val == best_val,
                'grid_points': int(len(seg)),
                'valid_points': int(len(valid)),
                'best_score': best_score,
                'mean_score': mean_score,
                'discarded_points': int(len(seg) - len(valid)),
                'top_recent_6m_calmar': row.get('recent_6m_calmar'),
                'top_recent_1y_calmar': row.get('recent_1y_calmar'),
                'top_full_pre_cutoff_calmar': row.get('full_pre_cutoff_calmar'),
                'top_discarded': row.get('discarded', ''),
            })
    return pd.DataFrame(rows)


def render_report(strategy_id: str, profile: dict, local_df: pd.DataFrame, summary_df: pd.DataFrame) -> str:
    tuned = profile['tuned_params']
    lines = [f'# Timing sensitivity audit — {strategy_id}', '']
    lines.append('## Best profile')
    lines.append(f"- best params: `{tuned}`")
    lines.append(f"- score: **{profile.get('score')}**")
    lines.append(f"- floor ratio: **{profile.get('full_nav_floor_ratio')}** × default_full_nav `{profile.get('default_full_nav')}`")
    lines.append('')

    if strategy_id == 'chinext_timing':
        lines.append('> 注：`momentum_threshold=0.0` 不能做乘法 ±20% 扰动；本报告改为比较离散邻域值（如 0.0 / 0.01 / 0.02）。')
        lines.append('')

    lines.append('## Local neighborhood rows')
    lines.append(f'- rows in local neighborhood: **{len(local_df)}**')
    lines.append('')
    lines.append('| rank | score | discarded | recent_6m | recent_1y | full_pre_cutoff | params |')
    lines.append('| --- | ---: | --- | ---: | ---: | ---: | --- |')
    ranked = local_df.sort_values(['score'], ascending=[False]).head(12).reset_index(drop=True)
    for idx, row in ranked.iterrows():
        params = {k: row[k] for k in tuned.keys() if k in row}
        discarded = row.get('discarded', '')
        if pd.isna(discarded):
            discarded = ''
        lines.append(
            f"| {idx+1} | {row.get('score')} | {discarded} | {row.get('recent_6m_calmar')} | {row.get('recent_1y_calmar')} | {row.get('full_pre_cutoff_calmar')} | `{params}` |"
        )
    lines.append('')

    lines.append('## One-parameter stability summary')
    lines.append('| parameter | value | best? | grid_points | valid_points | discarded_points | best_score | mean_score | top_recent_6m | top_recent_1y | top_full |')
    lines.append('| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['parameter']} | {row['value']} | {'Y' if row['is_best_value'] else ''} | {int(row['grid_points'])} | {int(row['valid_points'])} | {int(row['discarded_points'])} | {row['best_score']} | {row['mean_score']} | {row['top_recent_6m_calmar']} | {row['top_recent_1y_calmar']} | {row['top_full_pre_cutoff_calmar']} |"
        )
    lines.append('')
    return '\n'.join(lines) + '\n'


def audit_one(strategy_id: str) -> None:
    profile = load_profile(strategy_id)
    df = load_log(strategy_id)
    tuned = profile.get('tuned_params', {})
    metric_cols = ['score', 'recent_6m_calmar', 'recent_1y_calmar', 'full_pre_cutoff_calmar']
    df = _coerce_numeric(df, list(tuned.keys()) + metric_cols)
    local_df = build_local_slice(df, tuned)
    summary_df = summarize_by_param(local_df, tuned)

    csv_path = os.path.join(STRATEGY_DIR, f'sensitivity_{strategy_id}.csv')
    md_path = os.path.join(STRATEGY_DIR, f'sensitivity_{strategy_id}.md')
    local_df.sort_values('score', ascending=False).to_csv(csv_path, index=False)
    with open(md_path, 'w') as f:
        f.write(render_report(strategy_id, profile, local_df, summary_df))
    print(f'[ok] {strategy_id}: {csv_path}')
    print(f'[ok] {strategy_id}: {md_path}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', action='append', default=None)
    args = parser.parse_args()
    target_ids = args.only or DEFAULT_IDS
    for strategy_id in target_ids:
        audit_one(strategy_id)


if __name__ == '__main__':
    main()
