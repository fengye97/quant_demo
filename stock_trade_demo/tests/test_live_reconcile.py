from __future__ import annotations

import pandas as pd
import pytest

from web.app import create_app
from web import state
from services import live_trades as lt


@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def temp_ledger(tmp_path, monkeypatch):
    fake_dir = tmp_path / 'data'
    fake_file = fake_dir / 'live_trades.csv'
    monkeypatch.setattr(lt, 'LIVE_DATA_DIR', str(fake_dir))
    monkeypatch.setattr(lt, 'LIVE_TRADES_FILE', str(fake_file))
    monkeypatch.setattr(lt, '_LOCKFILE_PATH', str(fake_file) + '.lock')
    return fake_file


def _sample_result():
    df = pd.DataFrame({
        '交易日期': pd.to_datetime(['2026-06-01', '2026-06-02']),
        'execution_date': pd.to_datetime(['2026-06-02', '2026-06-03']),
        '累积净值': [1.00, 1.05],
        'target_exposure': [0.5, 0.5],
        'etf_open': [10.0, 11.0],
        'etf_close': [10.5, 11.5],
        'signal_action': ['buy', 'hold'],
        'position': [1, 1],
        'reason_summary': ['建仓', '继续持有'],
        'etf_code': ['510980', '510980'],
        'etf_name': ['中证1000ETF', '中证1000ETF'],
    })
    return df


def test_live_reconcile_uses_execution_date_and_exec_price(client, temp_ledger, monkeypatch):
    saved_cache = dict(state.TIMING_CACHE)
    try:
        state.TIMING_CACHE.clear()
        state.TIMING_CACHE['csi1000_timing'] = _sample_result()
        monkeypatch.setattr(state, 'init_timing_cache', lambda: None)
        lt.append_record({
            'date': '2026-06-02',
            'strategy': 'csi1000_timing',
            'signal_target': '0.5',
            'actual_position': '0.5',
            'exec_price': '10.0000',
            'capital': '50000.00',
            'notes': '',
            'created_at': '2026-06-02 10:00:00',
            'shares': '2000',
        })

        resp = client.get('/api/live/reconcile?strategy=csi1000_timing')
        assert resp.status_code == 200, resp.data
        body = resp.get_json()
        assert body['approximate_mode'] is False
        assert len(body['series']) == 2
        first = body['series'][0]
        second = body['series'][1]
        assert first['date'] == '2026-06-02'
        assert first['signal_date'] == '2026-06-01'
        assert first['mark_price'] == 10.5
        assert first['share_units'] == pytest.approx(2000.0)
        assert first['live_nav'] == pytest.approx(1.02, rel=1e-3)
        assert second['date'] == '2026-06-03'
        assert second['mark_price'] == 11.5
        assert second['live_nav'] == pytest.approx(1.06, rel=1e-3)
    finally:
        state.TIMING_CACHE.clear()
        state.TIMING_CACHE.update(saved_cache)


def test_live_reconcile_marks_approximate_mode_without_exec_price(client, temp_ledger, monkeypatch):
    saved_cache = dict(state.TIMING_CACHE)
    try:
        state.TIMING_CACHE.clear()
        state.TIMING_CACHE['csi1000_timing'] = _sample_result()
        monkeypatch.setattr(state, 'init_timing_cache', lambda: None)
        lt.append_record({
            'date': '2026-06-02',
            'strategy': 'csi1000_timing',
            'signal_target': '0.5',
            'actual_position': '0.5',
            'exec_price': '',
            'capital': '50000.00',
            'notes': '',
            'created_at': '2026-06-02 10:00:00',
            'shares': '',
        })

        resp = client.get('/api/live/reconcile?strategy=csi1000_timing')
        assert resp.status_code == 200, resp.data
        body = resp.get_json()
        assert body['approximate_mode'] is True
        assert body['series'][0]['date'] == '2026-06-02'
        assert body['series'][0]['signal_date'] == '2026-06-01'
    finally:
        state.TIMING_CACHE.clear()
        state.TIMING_CACHE.update(saved_cache)
