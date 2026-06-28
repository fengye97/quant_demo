import json
import pandas as pd
import web_app
from timing.backtest import summarize_timing_windows

CANDIDATES = [
    {"name": "baseline", "params": {}},
    {"name": "faster_exit_1", "params": {"fast_window": 10, "slow_window": 50, "momentum_window": 40, "enter_threshold": 0.58, "add_threshold": 0.80, "trim_threshold": 0.45, "exit_threshold": 0.22, "confirm_days": 1, "max_entry_exposure": 0.5, "probe_entry_exposure": 0.25, "probe_confirm_days": 1}},
    {"name": "faster_exit_2", "params": {"fast_window": 8, "slow_window": 45, "momentum_window": 30, "enter_threshold": 0.60, "add_threshold": 0.82, "trim_threshold": 0.48, "exit_threshold": 0.25, "confirm_days": 1, "max_entry_exposure": 0.5, "probe_entry_exposure": 0.25, "probe_confirm_days": 1}},
    {"name": "slower_entry_fast_exit", "params": {"fast_window": 12, "slow_window": 55, "momentum_window": 40, "enter_threshold": 0.62, "add_threshold": 0.84, "trim_threshold": 0.46, "exit_threshold": 0.24, "confirm_days": 2, "max_entry_exposure": 0.5, "probe_entry_exposure": 0.25, "probe_confirm_days": 2}},
    {"name": "mid_trend_guard", "params": {"fast_window": 15, "slow_window": 70, "momentum_window": 45, "enter_threshold": 0.60, "add_threshold": 0.82, "trim_threshold": 0.44, "exit_threshold": 0.22, "confirm_days": 1, "max_entry_exposure": 0.5, "probe_entry_exposure": 0.2, "probe_confirm_days": 2}},
]

rows = []
for item in CANDIDATES:
    result, metrics, strategy = web_app.run_timing_backtest_fresh('csi1000_timing', **item['params'])
    windows = summarize_timing_windows(result)
    trades = result.loc[result.get('trade_quantity', pd.Series(dtype=float)).abs() > 1e-8, ['交易日期', 'rebalance_action', 'target_exposure', 'prev_exposure']].tail(12).copy()
    trades['交易日期'] = trades['交易日期'].astype(str)
    rows.append({
        'name': item['name'],
        'params': item['params'],
        'metrics': metrics,
        'windows': windows,
        'recent_trades': trades.to_dict('records'),
    })

print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
