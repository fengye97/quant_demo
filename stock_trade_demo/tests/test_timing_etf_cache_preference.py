from __future__ import annotations

import pandas as pd

import index_data


def _write_csv(path, dates):
    df = pd.DataFrame({
        'date': dates,
        'open': [1.0] * len(dates),
        'high': [1.0] * len(dates),
        'low': [1.0] * len(dates),
        'close': [1.0] * len(dates),
        'volume': [100] * len(dates),
    })
    df.to_csv(path, index=False)


def test_describe_timing_etf_cache_keeps_qfq_even_when_legacy_is_fresher(tmp_path, monkeypatch):
    monkeypatch.setattr(index_data, 'CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(index_data, 'TIMING_ETF_CACHE_DIR', str(tmp_path / 'timing_etf'))
    (tmp_path / 'timing_etf').mkdir(parents=True, exist_ok=True)

    qfq_path = tmp_path / 'timing_etf' / 'nasdaq_etf_daily_qfq.csv'
    legacy_sub_path = tmp_path / 'timing_etf' / 'nasdaq_etf_daily.csv'

    _write_csv(qfq_path, ['2026-05-27', '2026-05-28'])
    _write_csv(legacy_sub_path, ['2026-06-04', '2026-06-05'])

    info = index_data.describe_timing_etf_cache('nasdaq', adjust='qfq')
    assert info['preferred_runtime_path'] == str(qfq_path)


def test_describe_timing_etf_cache_prefers_qfq_when_same_freshness(tmp_path, monkeypatch):
    monkeypatch.setattr(index_data, 'CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(index_data, 'TIMING_ETF_CACHE_DIR', str(tmp_path / 'timing_etf'))
    (tmp_path / 'timing_etf').mkdir(parents=True, exist_ok=True)

    qfq_path = tmp_path / 'timing_etf' / 'sp500_etf_daily_qfq.csv'
    legacy_sub_path = tmp_path / 'timing_etf' / 'sp500_etf_daily.csv'

    shared_dates = ['2026-06-04', '2026-06-05']
    _write_csv(qfq_path, shared_dates)
    _write_csv(legacy_sub_path, shared_dates)

    info = index_data.describe_timing_etf_cache('sp500', adjust='qfq')
    assert info['preferred_runtime_path'] == str(qfq_path)
