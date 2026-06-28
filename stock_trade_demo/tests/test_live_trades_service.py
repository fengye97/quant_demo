"""Regression tests for services.live_trades.

Covers CLAUDE.md rule 15 invariants:
  - never seeded with demo rows
  - atomic write (tmp + os.replace)
  - schema backwards-compatibility on append
  - lock prevents interleaved writes
  - delete returns False if absent (no silent rebuild)
"""
from __future__ import annotations

import csv
import os
import threading

import pytest

from services import live_trades as lt


@pytest.fixture
def temp_ledger(tmp_path, monkeypatch):
    """Point services.live_trades at a tmp file so tests never touch
    the protected data/live_trades.csv."""
    fake_dir = tmp_path / 'data'
    fake_file = fake_dir / 'live_trades.csv'
    monkeypatch.setattr(lt, 'LIVE_DATA_DIR', str(fake_dir))
    monkeypatch.setattr(lt, 'LIVE_TRADES_FILE', str(fake_file))
    monkeypatch.setattr(lt, '_LOCKFILE_PATH', str(fake_file) + '.lock')
    monkeypatch.setattr(lt, '_INTRAPROCESS_LOCK', threading.RLock())
    return fake_file


def test_empty_state_header_only(temp_ledger):
    """First read on a fresh deploy creates header only — never seeds rows."""
    rows = lt.read_all()
    assert rows == []
    assert temp_ledger.exists()
    with open(temp_ledger, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    assert len(lines) == 1
    assert lines[0] == ','.join(lt.LIVE_TRADES_COLUMNS)


def test_append_record_assigns_monotonic_id(temp_ledger):
    r1 = lt.append_record({'date': '2026-01-15', 'strategy': 'csi1000_timing'})
    r2 = lt.append_record({'date': '2026-01-16', 'strategy': 'csi1000_timing'})
    r3 = lt.append_record({'date': '2026-01-17', 'strategy': 'csi1000_timing'})
    assert r1['record_id'] == '1'
    assert r2['record_id'] == '2'
    assert r3['record_id'] == '3'
    all_rows = lt.read_all()
    assert len(all_rows) == 3


def test_append_persists_unknown_columns_as_blank(temp_ledger):
    stored = lt.append_record({
        'date': '2026-02-01', 'strategy': 'sp500_timing',
        'made_up_key': 'should be dropped',
    })
    assert 'made_up_key' not in stored
    for col in lt.LIVE_TRADES_COLUMNS:
        assert col in stored


def test_delete_record_returns_false_when_absent(temp_ledger):
    lt.append_record({'date': '2026-03-01', 'strategy': 'sp500_timing'})
    assert lt.delete_record(9999) is False
    assert lt.delete_record(1) is True
    assert lt.read_all() == []


def test_delete_does_not_remove_file_when_emptied(temp_ledger):
    """CLAUDE.md rule 15: deleting all records must NOT delete the file."""
    lt.append_record({'date': '2026-03-01', 'strategy': 'sp500_timing'})
    lt.delete_record(1)
    assert temp_ledger.exists()
    rows = lt.read_all()
    assert rows == []


def test_latest_position_default_zero(temp_ledger):
    assert lt.latest_position('csi1000_timing') == 0.0


def test_latest_position_picks_most_recent_date(temp_ledger):
    lt.append_record({'date': '2026-04-01', 'strategy': 'csi1000_timing', 'actual_position': '0.5'})
    lt.append_record({'date': '2026-04-10', 'strategy': 'csi1000_timing', 'actual_position': '0.8'})
    lt.append_record({'date': '2026-04-05', 'strategy': 'sp500_timing', 'actual_position': '0.3'})
    assert lt.latest_position('csi1000_timing') == pytest.approx(0.8)
    assert lt.latest_position('sp500_timing') == pytest.approx(0.3)
    assert lt.latest_position('unknown_strategy') == 0.0


def test_schema_backward_compat_old_row_missing_shares(temp_ledger):
    """An old CSV without the 'shares' column (added later) should still load
    cleanly — missing column reads as ''."""
    os.makedirs(temp_ledger.parent, exist_ok=True)
    old_cols = [c for c in lt.LIVE_TRADES_COLUMNS if c != 'shares']
    with open(temp_ledger, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=old_cols)
        w.writeheader()
        w.writerow({c: ('1' if c == 'record_id' else 'x') for c in old_cols})
    rows = lt.read_all()
    assert len(rows) == 1
    assert rows[0].get('shares', '') == ''
    # Now append → file rewritten with full schema
    lt.append_record({'date': '2026-05-01', 'strategy': 'csi1000_timing'})
    rows = lt.read_all()
    assert all('shares' in r for r in rows)


def test_write_is_atomic_no_tmp_left_behind(temp_ledger):
    lt.append_record({'date': '2026-05-15', 'strategy': 'csi1000_timing'})
    assert not (temp_ledger.parent / 'live_trades.csv.tmp').exists()


def test_concurrent_appends_serialize(temp_ledger):
    """Threads racing append_record must all succeed with distinct IDs."""
    def worker(i):
        lt.append_record({'date': f'2026-06-{i+1:02d}', 'strategy': 'csi1000_timing'})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rows = lt.read_all()
    assert len(rows) == 10
    ids = sorted(int(r['record_id']) for r in rows)
    assert ids == list(range(1, 11))


def test_transaction_context_persists_mutations(temp_ledger):
    lt.append_record({'date': '2026-07-01', 'strategy': 'csi1000_timing', 'actual_position': '0.4'})
    with lt.transaction() as rows:
        for r in rows:
            r['notes'] = 'reviewed'
    rows = lt.read_all()
    assert all(r['notes'] == 'reviewed' for r in rows)
