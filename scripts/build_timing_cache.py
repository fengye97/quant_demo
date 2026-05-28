"""离线预跑：所有 A 股择时策略 × 默认参数 → web_cache.pkl 的 timing bucket。

Pillar 1 Step 6：Web 请求路径只读缓存，不再做策略重算。/api/timing/backtest
命中缓存时直接 copy；否则返回 HTTP 400 cache_miss，提示运行本脚本重建。

使用：
    python scripts/build_timing_cache.py
    # 或限定子集
    python scripts/build_timing_cache.py --strategies csi1000_timing

注意：美股择时（macro_v32_timing/sp500_timing）由 scripts/build_us_timing_cache.py
单独维护，落盘到 .cache/us_timing/<sid>.pkl，不走本脚本。
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STD_DIR = os.path.join(REPO_ROOT, 'stock_trade_demo')
sys.path.insert(0, STD_DIR)


def main() -> int:
    parser = argparse.ArgumentParser(description='Offline rebuild of A-share timing strategy cache.')
    parser.add_argument('--strategies', nargs='*', default=None,
                        help='只重算指定 strategy_id 列表（不传则全跑）。')
    parser.add_argument('--no-save', action='store_true',
                        help='只重算不写 pkl（用于 dry run / 验证）。')
    args = parser.parse_args()

    from web import state
    from timing import run_timing_backtest
    import pandas as pd

    state.ensure_index_returns_loaded()
    state.ensure_timing_panel_loaded()
    if state.TIMING_PANEL is None or len(state.TIMING_PANEL) == 0:
        print('[build_timing] ERROR: 指数日线面板未加载，无法继续。')
        return 2

    target_ids = list(args.strategies) if args.strategies else list(state.TIMING_REGISTRY.keys())
    print(f'[build_timing] 将重算 {len(target_ids)} 个择时策略：{target_ids}')

    failed = []
    for sid in target_ids:
        if sid not in state.TIMING_REGISTRY:
            print(f'[build_timing] {sid} 未注册到 TIMING_REGISTRY，跳过。')
            failed.append(sid)
            continue
        try:
            print(f'[build_timing] 预运行 {sid} ...')
            strategy = state.build_timing_strategy(sid)
            signal_df = strategy.run(state.TIMING_PANEL.copy())
            if sid == 'csi1000_timing':
                state.CSI1000_SIGNAL_SERIES = pd.Series(
                    pd.to_numeric(signal_df['target_exposure'], errors='coerce').fillna(0.0).values,
                    index=pd.to_datetime(signal_df['交易日期']),
                ).sort_index()
            result = run_timing_backtest(
                signal_df, strategy,
                benchmark_returns=state.INDEX_RETURNS_MAP.get(strategy.get_index_id()),
            )
            state.TIMING_CACHE[sid] = result
            nv = float(result['累积净值'].iloc[-1]) if len(result) else float('nan')
            print(f'[build_timing] {sid} OK，累积净值={nv:.3f}，行数={len(result)}')
        except Exception as e:
            print(f'[build_timing] {sid} 失败: {e}')
            traceback.print_exc()
            failed.append(sid)

    if args.no_save:
        print('[build_timing] --no-save 跳过磁盘落盘。')
    else:
        # 与 build_select_cache 同理：落盘前确保 backtest bucket 已加载，避免覆盖。
        # 注意 _load_disk_cache 内部对 timing dict 用 .update() 覆盖（cache_store.py:164），
        # 直接调用会把刚算出的新 timing 值用磁盘旧值压回去 → 本次 rebuild 完全失效。
        # 解决：load 前 snap 当前进程刚算的 timing dict，load 后再 update 回去。
        if not state.BACKTEST_CACHE:
            timing_snap = dict(state.TIMING_CACHE)
            state._load_disk_cache()
            state.TIMING_CACHE.update(timing_snap)
        state._save_disk_cache()
        print('[build_timing] web_cache.pkl 已落盘。')

    if failed:
        print(f'[build_timing] 完成（失败: {failed}）。')
        return 1
    print('[build_timing] 全部完成。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
