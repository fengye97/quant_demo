"""Tests for meta.json lineage sidecars on atomic writers (Pillar 2 Step 6).

Behavior under test:
  1. ``produced_by=None`` (default) writes NO sidecar (backward compat).
  2. ``produced_by="..."`` writes ``<path>.meta.json`` containing the
     mandatory keys (produced_by, written_at_iso, written_at_unix,
     sha256_first_8k) and ``row_count_if_df`` when the payload is a DataFrame.
  3. The sidecar is written **after** the data file is renamed in place — a
     failed write leaves NO meta file behind (no stale lineage).
  4. ``read_meta`` returns ``None`` for legacy files with no sidecar.
  5. Sidecar write is best-effort: a corrupt directory does not break the
     primary write semantics.
  6. All four typed writers (csv/json/pickle/text) write a sidecar when
     ``produced_by`` is set.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.atomic_io import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    atomic_write_pickle,
    atomic_write_text,
    read_meta,
)


def _good_df():
    return pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'open': [1.0, 1.1, 1.2],
    })


def test_default_produced_by_writes_no_sidecar(tmp_path):
    target = tmp_path / 'out.csv'
    atomic_write_csv(target, _good_df(), index=False)
    assert target.exists()
    assert not (tmp_path / 'out.csv.meta.json').exists(), (
        "default produced_by=None must NOT emit a sidecar (back-compat)"
    )
    assert read_meta(target) is None


def test_csv_with_produced_by_writes_full_sidecar(tmp_path):
    target = tmp_path / 'out.csv'
    df = _good_df()
    atomic_write_csv(target, df, index=False, produced_by='unit_test/csv')

    meta_path = tmp_path / 'out.csv.meta.json'
    assert meta_path.exists(), "sidecar should be written when produced_by is set"
    payload = json.loads(meta_path.read_text(encoding='utf-8'))
    assert payload['produced_by'] == 'unit_test/csv'
    assert isinstance(payload['written_at_iso'], str) and payload['written_at_iso']
    assert isinstance(payload['written_at_unix'], (int, float))
    # Spec: 32 hex chars (truncated SHA-256) — short enough for sidecars,
    # long enough for silent-corruption detection.
    assert isinstance(payload['sha256_first_8k'], str) and len(payload['sha256_first_8k']) == 32
    assert payload['row_count_if_df'] == len(df)

    # read_meta wrapper should round-trip
    via_helper = read_meta(target)
    assert via_helper == payload


def test_failed_write_leaves_no_sidecar(tmp_path):
    """Schema failure → no data file → no meta sidecar."""
    import pandera as pa
    from pandera import Column, DataFrameSchema, Check

    target = tmp_path / 'out.csv'
    bad = _good_df()
    bad.loc[0, 'open'] = 0.0
    schema = DataFrameSchema({
        'open': Column(float, Check.gt(0), coerce=True),
    }, strict=False, coerce=False)

    with pytest.raises(Exception):  # SchemaError / SchemaErrors
        atomic_write_csv(target, bad, index=False,
                         schema=schema, produced_by='unit_test/should_fail')

    assert not target.exists(), "failed validation must leave no data file"
    assert not (tmp_path / 'out.csv.meta.json').exists(), (
        "no sidecar may be written if the data write failed"
    )


def test_read_meta_returns_none_for_legacy_files(tmp_path):
    """Files without a sidecar (legacy) return None — caller falls back to mtime."""
    legacy = tmp_path / 'legacy.csv'
    legacy.write_text("a,b\n1,2\n", encoding='utf-8')
    assert read_meta(legacy) is None


def test_json_pickle_text_all_write_sidecar(tmp_path):
    """The three non-DataFrame writers each emit a sidecar when produced_by set."""
    j = tmp_path / 'cfg.json'
    atomic_write_json(j, {'k': 'v'}, produced_by='unit_test/json')
    assert (tmp_path / 'cfg.json.meta.json').exists()
    j_meta = read_meta(j)
    assert j_meta['produced_by'] == 'unit_test/json'
    assert 'row_count_if_df' not in j_meta  # not a DataFrame

    p = tmp_path / 'obj.pkl'
    atomic_write_pickle(p, {'x': 1}, produced_by='unit_test/pickle')
    assert read_meta(p)['produced_by'] == 'unit_test/pickle'

    t = tmp_path / 'note.txt'
    atomic_write_text(t, 'hello', produced_by='unit_test/text')
    assert read_meta(t)['produced_by'] == 'unit_test/text'


def test_meta_write_failure_does_not_break_main_file(tmp_path, monkeypatch):
    """meta sidecar 是 best-effort：写 sidecar 出错绝不能让主文件回滚或抛错。

    主文件已经 rename 落地 → 它是 source of truth；sidecar 只是 lineage 辅助。
    """
    target = tmp_path / 'main.csv'
    df = _good_df()

    # Make the sidecar write blow up — patch the helper directly.
    import utils.atomic_io as aio

    real_replace = os.replace
    calls = {'n': 0}

    def _replace_failing_for_meta(src, dst):
        # Allow main-file replace through; explode on meta sidecar replace.
        if str(dst).endswith('.meta.json'):
            calls['n'] += 1
            raise OSError('synthetic meta-replace failure')
        return real_replace(src, dst)

    monkeypatch.setattr(aio.os, 'replace', _replace_failing_for_meta)

    # MUST NOT raise — meta failure is swallowed/logged only.
    atomic_write_csv(target, df, index=False, produced_by='unit_test/meta_fail')

    assert target.exists(), "main data file must survive sidecar failure"
    assert pd.read_csv(target).equals(df), "main data content must be intact"
    assert not (tmp_path / 'main.csv.meta.json').exists(), \
        "broken sidecar should not have landed"
    assert calls['n'] >= 1, "sidecar write should have been attempted at least once"


def test_sidecar_sha_changes_when_content_changes(tmp_path):
    """Two different payloads written to same path → sha256_first_8k differs."""
    target = tmp_path / 'evolving.csv'
    atomic_write_csv(target, _good_df(), index=False, produced_by='unit_test/v1')
    sha1 = read_meta(target)['sha256_first_8k']

    df2 = _good_df()
    df2.loc[0, 'open'] = 9.99
    atomic_write_csv(target, df2, index=False, produced_by='unit_test/v2')
    meta2 = read_meta(target)
    assert meta2['produced_by'] == 'unit_test/v2'
    assert meta2['sha256_first_8k'] != sha1, "fingerprint must move when content moves"
