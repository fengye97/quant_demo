from __future__ import annotations

import pandas as pd

from web import state


def _fake_result(last_date: str) -> pd.DataFrame:
    df = pd.DataFrame({
        '交易日期': pd.to_datetime(['2026-06-01', last_date]),
        '累积净值': [1.0, 1.02],
        'signal_action': ['hold', 'buy'],
        'position': [0, 1],
        'reason_summary': ['', ''],
        'reason_detail': [[], []],
        'signal_score': [0.0, 0.0],
        'strength_score': [0.0, 0.0],
        'target_exposure': [0.0, 1.0],
        'prev_exposure': [0.0, 0.0],
        'exposure_change': [0.0, 1.0],
        'rebalance_action': ['flat', 'enter'],
        'close': [100.0, 101.0],
        'etf_close': [10.0, 10.2],
        'trade_quantity': [0.0, 1.0],
        'strategy_return': [0.0, 0.02],
        'index_id': ['sp500', 'sp500'],
        'index_name': ['标普500', '标普500'],
        'etf_code': ['513500', '513500'],
        'etf_name': ['标普500ETF', '标普500ETF'],
        'holding_units': [0.0, 1.0],
        'holding_value': [0.0, 10.2],
        'cash_balance': [50000.0, 49989.8],
        'entry_price': [None, 10.1],
        'unrealized_pnl': [None, 0.1],
        'invested_amount': [0.0, 10.1],
    })
    df.attrs['metrics'] = {'最大回撤': -1.0, '年化收益': 5.0}
    return df


def test_run_index_data_update_force_reloads_us_panel(monkeypatch, tmp_path):
    saved_index_returns_map = dict(state.INDEX_RETURNS_MAP)
    saved_timing_cache = dict(state.TIMING_CACHE)
    saved_us_timing_cache = dict(state.US_TIMING_CACHE)
    saved_us_panel = state.US_TIMING_PANEL
    saved_timing_panel = state.TIMING_PANEL
    saved_status = dict(state._INDEX_UPDATE_STATUS)
    saved_effective_us = dict(state._EFFECTIVE_US_TIMING_DEFAULTS_CACHE)

    state.US_TIMING_PANEL = pd.DataFrame({'交易日期': pd.to_datetime(['2026-05-26'])})
    state.US_TIMING_CACHE.clear()
    state.US_TIMING_CACHE['macro_v32_timing'] = _fake_result('2026-05-26')
    state.TIMING_CACHE.clear()
    state.TIMING_PANEL = pd.DataFrame({'交易日期': pd.to_datetime(['2026-06-01'])})
    state.INDEX_RETURNS_MAP.clear()
    state._EFFECTIVE_US_TIMING_DEFAULTS_CACHE.clear()
    state._EFFECTIVE_US_TIMING_DEFAULTS_CACHE['macro_v32_timing'] = {'foo': 1}

    force_reload_flags = []

    def _fake_index_daily(index_id, force_refetch=False):
        return pd.DataFrame({'date': pd.to_datetime(['2026-06-04', '2026-06-05']), 'close': [1.0, 1.1]})

    def _fake_index_returns(index_id, force_refetch=False):
        return pd.Series([0.01], index=pd.to_datetime(['2026-06-30']))

    def _fake_refresh_all_timing_etf_daily():
        return None

    def _fake_get_timing_etf_daily(index_id):
        return pd.DataFrame({'date': pd.to_datetime(['2026-06-04', '2026-06-05']), 'close': [1.0, 1.1]})

    def _fake_check_alignment():
        return []

    def _fake_ensure_timing_panel_loaded():
        state.TIMING_PANEL = pd.DataFrame({'交易日期': pd.to_datetime(['2026-06-05'])})

    def _fake_ensure_us_timing_panel_loaded(force_reload=False):
        force_reload_flags.append(force_reload)
        state.US_TIMING_PANEL = pd.DataFrame({'交易日期': pd.to_datetime(['2026-06-05'])})

    def _fake_build_timing_strategy(sid):
        class _S:
            def get_index_id(self):
                return 'csi1000'
        return _S()

    def _fake_build_us_timing_strategy(sid):
        class _S:
            def get_index_id(self):
                return 'sp500'
        return _S()

    def _fake_run_strategy(panel):
        return pd.DataFrame({'交易日期': pd.to_datetime(['2026-06-04', '2026-06-05'])})

    def _fake_run_timing_backtest(signal_df, strategy, benchmark_returns=None):
        idx = strategy.get_index_id()
        last_date = '2026-06-05' if idx == 'sp500' else '2026-06-04'
        return _fake_result(last_date)

    def _fake_save_disk_cache():
        return None

    monkeypatch.setattr(state, 'get_index_daily', _fake_index_daily)
    monkeypatch.setattr(state, 'get_index_returns', _fake_index_returns)
    monkeypatch.setattr(state, 'refresh_all_timing_etf_daily', _fake_refresh_all_timing_etf_daily)
    monkeypatch.setattr(state, 'get_timing_etf_daily', _fake_get_timing_etf_daily)
    monkeypatch.setattr(state, '_check_a_share_index_etf_alignment', _fake_check_alignment)
    monkeypatch.setattr(state, 'ensure_timing_panel_loaded', _fake_ensure_timing_panel_loaded)
    monkeypatch.setattr(state, 'ensure_us_timing_panel_loaded', _fake_ensure_us_timing_panel_loaded)
    monkeypatch.setattr(state, 'build_timing_strategy', _fake_build_timing_strategy)
    monkeypatch.setattr(state, 'build_us_timing_strategy', _fake_build_us_timing_strategy)
    monkeypatch.setattr(state, 'run_timing_backtest', _fake_run_timing_backtest)
    monkeypatch.setattr(state, '_save_disk_cache', _fake_save_disk_cache)
    monkeypatch.setattr(state, 'TIMING_STRATEGY_MAP', {'csi1000_timing': object()})
    monkeypatch.setattr(state, 'US_TIMING_STRATEGY_MAP', {'macro_v32_timing': object()})
    monkeypatch.setattr(state, 'INDEX_CONFIGS', {'csi1000': {'name': '中证1000'}, 'sp500': {'name': '标普500'}})
    monkeypatch.setattr(state, 'A_SHARE_INDEX_IDS', [])
    monkeypatch.setattr(state, '_CACHE_DIR', str(tmp_path))

    class _StrategyWithRun:
        def get_index_id(self):
            return 'sp500'
        def run(self, panel):
            return _fake_run_strategy(panel)

    class _TimingStrategyWithRun:
        def get_index_id(self):
            return 'csi1000'
        def run(self, panel):
            return _fake_run_strategy(panel)

    monkeypatch.setattr(state, 'build_timing_strategy', lambda sid: _TimingStrategyWithRun())
    monkeypatch.setattr(state, 'build_us_timing_strategy', lambda sid: _StrategyWithRun())

    try:
        state._run_index_data_update()
        assert force_reload_flags == [True], '美股 panel 重建应显式 force_reload=True'
        assert state.US_TIMING_PANEL is not None
        assert pd.to_datetime(state.US_TIMING_PANEL['交易日期']).max() == pd.Timestamp('2026-06-05')
        assert 'macro_v32_timing' in state.US_TIMING_CACHE
        assert pd.to_datetime(state.US_TIMING_CACHE['macro_v32_timing']['交易日期']).max() == pd.Timestamp('2026-06-05')
        assert state._EFFECTIVE_US_TIMING_DEFAULTS_CACHE == {}, '重建前应清空 effective defaults 缓存'
    finally:
        state.INDEX_RETURNS_MAP.clear()
        state.INDEX_RETURNS_MAP.update(saved_index_returns_map)
        state.TIMING_CACHE.clear()
        state.TIMING_CACHE.update(saved_timing_cache)
        state.US_TIMING_CACHE.clear()
        state.US_TIMING_CACHE.update(saved_us_timing_cache)
        state.US_TIMING_PANEL = saved_us_panel
        state.TIMING_PANEL = saved_timing_panel
        state._INDEX_UPDATE_STATUS.clear()
        state._INDEX_UPDATE_STATUS.update(saved_status)
        state._EFFECTIVE_US_TIMING_DEFAULTS_CACHE.clear()
        state._EFFECTIVE_US_TIMING_DEFAULTS_CACHE.update(saved_effective_us)
