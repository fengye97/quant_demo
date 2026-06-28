"""离线预跑：所有选股策略 × 默认参数 → web_cache.pkl 的 backtest bucket。

Pillar 1 Step 6：Web 请求路径只读缓存，不再做策略重算。所有选股策略的默认参数
回测产物由本脚本生产并落盘；blueprint `/api/backtest` 命中缓存时直接 copy，
否则返回 HTTP 400 `cache_miss`，提示运行本脚本重建。

使用：
    python scripts/build_select_cache.py
    # 或限定子集
    python scripts/build_select_cache.py --strategies original original_ensemble

新增策略：
    1. 在策略类上声明 strategy_id + registry='select'
    2. 在 stock_trade_demo/web/state.py 顶部 import 该模块（让 __init_subclass__ 触发）
    3. 跑本脚本一遍。脚本会按 _CACHE_DEFAULTS 选默认参数（未列出的策略走类默认构造）。
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
    parser = argparse.ArgumentParser(description='Offline rebuild of select strategy backtest cache.')
    parser.add_argument('--strategies', nargs='*', default=None,
                        help='只重算指定 strategy_id 列表（不传则全跑）。')
    parser.add_argument('--no-save', action='store_true',
                        help='只重算不写 pkl（用于 dry run / 验证）。')
    args = parser.parse_args()

    # 触发策略类注册 + 拉起 web.state（panel/index 数据加载、_CACHE_DEFAULTS 等）
    from web import state
    from backtest import select_and_backtest, strategy_evaluate

    state.ensure_index_returns_loaded()
    state.ensure_stock_data_loaded()
    if state.DATA_DF is None:
        print('[build_select] ERROR: 月度数据未加载，无法继续。请先确认 stock_data.csv 存在。')
        return 2

    target_ids = list(args.strategies) if args.strategies else list(state.STRATEGY_REGISTRY.keys())
    print(f'[build_select] 将重算 {len(target_ids)} 个选股策略：{target_ids}')

    failed = []
    for sid in target_ids:
        cls = state.STRATEGY_REGISTRY.get(sid)
        if cls is None:
            print(f'[build_select] {sid} 未注册，跳过。')
            failed.append(sid)
            continue
        try:
            print(f'[build_select] 预运行 {sid} ...')
            defaults = state._CACHE_DEFAULTS.get(sid, {})
            # 默认参数仅注入 __init__ 接受的字段，避免类不支持的 kwargs 引发异常
            import inspect
            sig = inspect.signature(cls.__init__)
            init_kwargs = {k: v for k, v in defaults.items() if k in sig.parameters}
            strategy = cls(**init_kwargs)
            df = strategy.run(state.DATA_DF.copy())
            result = select_and_backtest(
                df, strategy,
                c_rate=strategy.c_rate, t_rate=strategy.t_rate,
                bull_tp=strategy.bull_tp, bear_tp=strategy.bear_tp,
                bull_n=strategy.bull_n, bear_n=strategy.bear_n,
                initial_capital=strategy.initial_capital,
            )
            if hasattr(strategy, '_profile_summary'):
                result.attrs['strategy_meta'] = {
                    'profile_summary': getattr(strategy, '_profile_summary', []),
                }
                state._PROFILE_SUMMARY_CACHE[sid] = getattr(strategy, '_profile_summary', [])
            ev = strategy_evaluate(result, index_returns=state.INDEX_RETURNS)
            state.BACKTEST_CACHE[sid] = (result, ev)
            print(f'[build_select] {sid} OK，累积净值={float(result["累积净值"].iloc[-1]):.3f}，行数={len(result)}')
        except Exception as e:
            print(f'[build_select] {sid} 失败: {e}')
            traceback.print_exc()
            failed.append(sid)

    if args.no_save:
        print('[build_select] --no-save 跳过磁盘落盘。')
    else:
        # web_cache.pkl 同时承载 backtest_cache / timing_cache；调用 _save_disk_cache
        # 会用当前进程内存里的两份 dict 一起写出。如果当前进程没有跑 init_timing_cache，
        # state.TIMING_CACHE 可能仍是空 dict，这种情况下 timing bucket 会被清空。
        # 解决：先尝试从磁盘把现有 timing bucket 读回内存，再写。
        if not state.TIMING_CACHE:
            state._load_disk_cache()
        state._save_disk_cache()
        print('[build_select] web_cache.pkl 已落盘。')

    if failed:
        print(f'[build_select] 完成（失败: {failed}）。')
        return 1
    print('[build_select] 全部完成。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
