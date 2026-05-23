"""Offline builder for US timing strategy backtest cache.

For each strategy in US_TIMING_STRATEGY_MAP, runs the full backtest at the
strategy's default parameters and pickles the resulting DataFrame to
stock_trade_demo/.cache/us_timing/<strategy_id>.pkl

The web app loads these pickles on startup instead of recomputing.
Run this script whenever strategy code or default params change.

Usage:
    python scripts/build_us_timing_cache.py
"""
import os
import sys
import pickle
import inspect
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STD_DIR = os.path.join(REPO_ROOT, 'stock_trade_demo')
sys.path.insert(0, STD_DIR)

from index_data import build_us_index_panel
from timing import (
    run_timing_backtest,
    MacroV32TimingStrategy,
    NasdaqTimingStrategy,
    SP500TimingStrategy,
)

CACHE_DIR = os.path.join(STD_DIR, '.cache', 'us_timing')

# 与 web_app.US_TIMING_STRATEGY_MAP / _US_TIMING_CACHE_DEFAULTS 保持一致
US_TIMING_STRATEGY_MAP = {
    'macro_v32_timing': MacroV32TimingStrategy,
    'nasdaq_timing': NasdaqTimingStrategy,
    'sp500_timing': SP500TimingStrategy,
}

_US_TIMING_CACHE_DEFAULTS = {
    'macro_v32_timing': {
        'sigmoid_k': 1.5, 'max_leverage': 1.4, 'base_position': 0.5,
        'inertia': 0.03, 'crisis_vix': 35.0,
        'exposure_mode': 'staged', 'enter_threshold': 0.55, 'add_threshold': 0.75,
        'trim_threshold': 0.35, 'exit_threshold': 0.15, 'confirm_days': 1,
        'max_entry_exposure': 1.0,
    },
    'nasdaq_timing': {
        'fast_window': 20, 'slow_window': 120, 'momentum_window': 120,
        'exposure_mode': 'binary', 'enter_threshold': 0.55, 'add_threshold': 0.75,
        'trim_threshold': 0.35, 'exit_threshold': 0.15, 'confirm_days': 2,
        'max_entry_exposure': 0.5,
    },
    'sp500_timing': {
        'fast_window': 20, 'slow_window': 125, 'momentum_window': 100,
        'exposure_mode': 'staged', 'enter_threshold': 0.5, 'add_threshold': 0.72,
        'trim_threshold': 0.32, 'exit_threshold': 0.14, 'confirm_days': 2,
        'max_entry_exposure': 0.5,
    },
}


def build_strategy(strategy_name, strat_cls):
    sig = inspect.signature(strat_cls.__init__)
    defaults = _US_TIMING_CACHE_DEFAULTS.get(strategy_name, {})
    valid = {k: v for k, v in defaults.items()
             if k in sig.parameters and v is not None}
    return strat_cls(**valid)


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"[build] cache dir: {CACHE_DIR}")
    panel = build_us_index_panel()
    print(f"[build] US ETF panel rows: {len(panel)}")

    for sid, cls in US_TIMING_STRATEGY_MAP.items():
        out_path = os.path.join(CACHE_DIR, f"{sid}.pkl")
        try:
            strategy = build_strategy(sid, cls)
            signal_df = strategy.run(panel.copy())
            result = run_timing_backtest(signal_df, strategy)
            with open(out_path, 'wb') as f:
                pickle.dump(result, f)
            final_nv = result['累积净值'].iloc[-1]
            print(f"[build] {sid} OK | rows={len(result)} | 累积净值={final_nv:.4f} -> {out_path}")
        except Exception as e:
            print(f"[build] {sid} FAILED: {e}")
            traceback.print_exc()


if __name__ == '__main__':
    main()
