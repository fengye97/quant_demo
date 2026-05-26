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
import json
import pickle
import inspect
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STD_DIR = os.path.join(REPO_ROOT, 'stock_trade_demo')
BEST_PROFILE_DIR = os.path.join(REPO_ROOT, 'strategy')
sys.path.insert(0, STD_DIR)

from index_data import build_us_index_panel, describe_timing_etf_cache
from timing import (
    run_timing_backtest,
    MacroV32TimingStrategy,
    SP500TimingStrategy,
)

CACHE_DIR = os.path.join(STD_DIR, '.cache', 'us_timing')

# 与 web_app.US_TIMING_STRATEGY_MAP / _US_TIMING_CACHE_DEFAULTS 保持一致
US_TIMING_STRATEGY_MAP = {
    'macro_v32_timing': MacroV32TimingStrategy,
    'sp500_timing': SP500TimingStrategy,
}

LEGACY_CLEAN_STARTS = {
    'nasdaq': '2022-07-06',
    'sp500': '2022-03-31',
}

_US_TIMING_CACHE_DEFAULTS = {
    'macro_v32_timing': {
        'sigmoid_k': 1.2, 'max_leverage': 1.4, 'base_position': 0.45,
        'inertia': 0.05, 'crisis_vix': 40.0,
        'fed_block_weight': 0.25, 'restrictive_threshold': 0.40, 'pivot_relief': 0.60,
        'exposure_mode': 'staged', 'enter_threshold': 0.55, 'add_threshold': 0.75,
        'trim_threshold': 0.35, 'exit_threshold': 0.15, 'confirm_days': 1,
        'max_entry_exposure': 1.0,
        'base_floor': 0.0,
    },
    'sp500_timing': {
        'fast_window': 20, 'slow_window': 125, 'momentum_window': 100,
        'exposure_mode': 'staged', 'enter_threshold': 0.5, 'add_threshold': 0.72,
        'trim_threshold': 0.32, 'exit_threshold': 0.14, 'confirm_days': 2,
        'max_entry_exposure': 0.5,
        'base_floor': 0.0,
    },
}


def _load_best_profile(strategy_name):
    fp = os.path.join(BEST_PROFILE_DIR, f'best_profile_{strategy_name}.json')
    if not os.path.exists(fp):
        return None
    try:
        with open(fp) as f:
            return json.load(f)
    except Exception as e:
        print(f"[build] WARN: failed to load best_profile for {strategy_name}: {e}")
        return None


def build_strategy(strategy_name, strat_cls):
    sig = inspect.signature(strat_cls.__init__)
    defaults = dict(_US_TIMING_CACHE_DEFAULTS.get(strategy_name, {}))
    profile = _load_best_profile(strategy_name)
    if profile is not None:
        all_params = profile.get('all_params') or {}
        overrides = {k: v for k, v in all_params.items() if v is not None}
        if overrides:
            print(f"[build] {strategy_name} overrides from best_profile: {sorted(overrides.keys())}")
        defaults.update(overrides)
    valid = {k: v for k, v in defaults.items()
             if k in sig.parameters and v is not None}
    return strat_cls(**valid)


def _apply_clean_window(panel, strategy):
    index_id = strategy.get_index_id()
    info = describe_timing_etf_cache(index_id=index_id)
    preferred = (info or {}).get('preferred_runtime_path') or ''
    clean_start = LEGACY_CLEAN_STARTS.get(index_id)
    if preferred.endswith('_qfq.csv') or not clean_start:
        return panel, None
    out = panel.copy()
    out['交易日期'] = out['交易日期'].astype('datetime64[ns]')
    out = out[out['交易日期'] >= clean_start].reset_index(drop=True)
    return out, {
        'index_id': index_id,
        'clean_start': clean_start,
        'preferred_runtime_path': preferred,
    }


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"[build] cache dir: {CACHE_DIR}")
    panel = build_us_index_panel()
    print(f"[build] US ETF panel rows: {len(panel)}")

    for sid, cls in US_TIMING_STRATEGY_MAP.items():
        out_path = os.path.join(CACHE_DIR, f"{sid}.pkl")
        try:
            strategy = build_strategy(sid, cls)
            strategy_panel, clean_meta = _apply_clean_window(panel, strategy)
            if clean_meta:
                print(f"[build] {sid} uses clean-window start={clean_meta['clean_start']} path={clean_meta['preferred_runtime_path']}")
            signal_df = strategy.run(strategy_panel.copy())
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
