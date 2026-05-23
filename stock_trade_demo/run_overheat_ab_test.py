#!/usr/bin/env python3
import json
import os
import sys
from typing import Dict, Optional

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from backtest import load_data, select_and_backtest, strategy_evaluate  # noqa: E402
from index_data import get_index_returns  # noqa: E402
from strategies.original import OriginalStrategy  # noqa: E402
from strategies.original_ensemble import OriginalEnsembleStrategy  # noqa: E402

OUTPUT_JSON = os.path.join(ROOT, '.cache', 'overheat_ab_test_result.json')
FULL_METRICS = ['累积净值', '年化收益', '最大回撤', '年化收益/回撤比', '最终资金']


def ensure_cache_dir():
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)


def build_strategies():
    return {
        'original_baseline': OriginalStrategy(),
        'ensemble_control': OriginalEnsembleStrategy(
            growth_timing_mode='off',
            board_tilt_strength=0.4,
            board_tilt_gate_pct=0.7,
            overheat_penalty=0.0,
            overheat_bias_pct=0.80,
        ),
        'ensemble_treatment': OriginalEnsembleStrategy(
            growth_timing_mode='off',
            board_tilt_strength=0.4,
            board_tilt_gate_pct=0.7,
            overheat_penalty=0.5,
            overheat_bias_pct=0.80,
        ),
    }


def run_strategy(df_full: pd.DataFrame, strategy, index_returns: pd.Series) -> Dict[str, object]:
    ranked = strategy.run(df_full.copy())
    result = select_and_backtest(ranked, strategy)
    evaluation = strategy_evaluate(result.copy(), index_returns=index_returns)
    metrics = evaluation.iloc[:, 0].to_dict()
    win_rate = float((result['选股下周期涨跌幅'] > 0).mean()) if len(result) else 0.0
    return {
        'result': result,
        'metrics': metrics,
        'win_rate': win_rate,
        'periods': int(len(result)),
    }


def rebase_result_period(result: pd.DataFrame, start: Optional[str] = None, end: Optional[str] = None,
                        initial_capital: float = 100000.0) -> Optional[pd.DataFrame]:
    period = result.copy()
    period['交易日期'] = pd.to_datetime(period['交易日期'])
    if start is not None:
        period = period[period['交易日期'] >= pd.Timestamp(start)]
    if end is not None:
        period = period[period['交易日期'] <= pd.Timestamp(end)]
    period = period.sort_values('交易日期').reset_index(drop=True)
    if len(period) == 0:
        return None

    returns = pd.to_numeric(period['选股下周期涨跌幅'], errors='coerce').fillna(0.0)
    period['资金曲线'] = (returns + 1.0).cumprod()
    period['累积净值'] = period['资金曲线']

    capitals = []
    pnls = []
    cum_capitals = []
    capital = float(initial_capital)
    for ret in returns.tolist():
        capitals.append(capital)
        pnl = capital * float(ret)
        pnls.append(pnl)
        capital += pnl
        cum_capitals.append(capital)

    period['当期本金'] = capitals
    period['当期盈亏'] = pnls
    period['累计资金'] = cum_capitals
    period.attrs = dict(result.attrs)
    period.attrs['initial_capital'] = float(initial_capital)
    return period


def summarize_period(result: pd.DataFrame, index_returns: pd.Series,
                     start: Optional[str] = None, end: Optional[str] = None) -> Optional[Dict[str, object]]:
    period = rebase_result_period(result, start=start, end=end)
    if period is None or len(period) == 0:
        return None
    evaluation = strategy_evaluate(period.copy(), index_returns=index_returns)
    metrics = evaluation.iloc[:, 0].to_dict()
    win_rate = float((period['选股下周期涨跌幅'] > 0).mean()) if len(period) else 0.0
    return {
        'metrics': metrics,
        'win_rate': win_rate,
        'periods': int(len(period)),
        'start': str(period['交易日期'].iloc[0])[:10],
        'end': str(period['交易日期'].iloc[-1])[:10],
    }


def build_full_summary(run_output: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for label, payload in run_output.items():
        metrics = payload['metrics']
        rows.append({
            'strategy': label,
            '累积净值': metrics.get('累积净值'),
            '年化收益': metrics.get('年化收益'),
            '最大回撤': metrics.get('最大回撤'),
            '年化收益/回撤比': metrics.get('年化收益/回撤比'),
            '最终资金': metrics.get('最终资金'),
            '月度胜率': f"{payload['win_rate'] * 100:.2f}%",
            '周期数': payload.get('periods'),
        })
    return pd.DataFrame(rows)


def build_window_summary(window_results: Dict[str, Optional[Dict[str, object]]]) -> pd.DataFrame:
    rows = []
    for label, payload in window_results.items():
        if not payload:
            continue
        metrics = payload['metrics']
        rows.append({
            'strategy': label,
            '窗口': f"{payload['start']} ~ {payload['end']}",
            '累积净值': metrics.get('累积净值'),
            '年化收益': metrics.get('年化收益'),
            '最大回撤': metrics.get('最大回撤'),
            '年化收益/回撤比': metrics.get('年化收益/回撤比'),
            '最终资金': metrics.get('最终资金'),
            '月度胜率': f"{payload['win_rate'] * 100:.2f}%",
            '周期数': payload.get('periods'),
        })
    return pd.DataFrame(rows)


def numeric_value(value):
    try:
        return float(value)
    except Exception:
        return None


def build_deltas(run_output: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, Optional[float]]]:
    original = run_output['original_baseline']['metrics']
    control = run_output['ensemble_control']['metrics']
    treatment = run_output['ensemble_treatment']['metrics']
    return {
        'treatment_vs_control': {
            '累积净值差值': round(numeric_value(treatment.get('累积净值')) - numeric_value(control.get('累积净值')), 4),
            '收益回撤比差值': round(numeric_value(treatment.get('年化收益/回撤比')) - numeric_value(control.get('年化收益/回撤比')), 4),
            '最终资金差值': round(numeric_value(treatment.get('最终资金')) - numeric_value(control.get('最终资金')), 2),
            '月度胜率差值pct': round((run_output['ensemble_treatment']['win_rate'] - run_output['ensemble_control']['win_rate']) * 100, 2),
        },
        'treatment_vs_original': {
            '累积净值差值': round(numeric_value(treatment.get('累积净值')) - numeric_value(original.get('累积净值')), 4),
            '收益回撤比差值': round(numeric_value(treatment.get('年化收益/回撤比')) - numeric_value(original.get('年化收益/回撤比')), 4),
            '最终资金差值': round(numeric_value(treatment.get('最终资金')) - numeric_value(original.get('最终资金')), 2),
            '月度胜率差值pct': round((run_output['ensemble_treatment']['win_rate'] - run_output['original_baseline']['win_rate']) * 100, 2),
        },
    }


def main():
    ensure_cache_dir()
    print('[overheat] loading dataset...')
    df_full = load_data(os.path.join(ROOT, 'stock_data.csv'))
    index_returns = get_index_returns('csi1000')
    strategies = build_strategies()

    run_output: Dict[str, Dict[str, object]] = {}
    for label, strategy in strategies.items():
        print(f'[overheat] running {label} ...')
        run_output[label] = run_strategy(df_full, strategy, index_returns)

    full_summary = build_full_summary(run_output)
    recent_6m = {
        label: summarize_period(payload['result'], index_returns, start='2025-12-01', end='2026-05-31')
        for label, payload in run_output.items()
    }
    recent_2m = {
        label: summarize_period(payload['result'], index_returns, start='2026-04-01', end='2026-05-31')
        for label, payload in run_output.items()
    }
    recent_6m_df = build_window_summary(recent_6m)
    recent_2m_df = build_window_summary(recent_2m)
    deltas = build_deltas(run_output)

    print('\n=== Full history summary ===')
    print(full_summary.to_string(index=False))
    print('\n=== Recent 6 months ===')
    print(recent_6m_df.to_string(index=False))
    print('\n=== 2026-04 ~ 2026-05 ===')
    print(recent_2m_df.to_string(index=False))
    print('\n=== Deltas ===')
    print(json.dumps(deltas, ensure_ascii=False, indent=2))

    payload = {
        'full_history': full_summary.to_dict(orient='records'),
        'recent_6m': recent_6m_df.to_dict(orient='records'),
        'recent_2m': recent_2m_df.to_dict(orient='records'),
        'deltas': deltas,
        'raw_metrics': {
            label: {
                'metrics': data['metrics'],
                'win_rate': round(data['win_rate'], 6),
                'periods': data['periods'],
            }
            for label, data in run_output.items()
        },
    }
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f'\n[overheat] saved summary to {OUTPUT_JSON}')


if __name__ == '__main__':
    main()
