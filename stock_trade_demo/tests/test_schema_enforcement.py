"""Tests for pandera schema enforcement on `atomic_write_csv`.

Covers:
  1. Legal DataFrame writes successfully (file appears, content correct).
  2. Illegal row (`open == 0`) → SchemaError, target file untouched.
  3. Illegal `date` format (not YYYY-MM-DD) → SchemaError, target untouched.
  4. `schema=None` (default) keeps backward-compat: no validation.

The "target untouched" property is what matters most — schema validation
must run **before** any tmp file is created, so a previously-good cache
file cannot be silently broken by a bad fetch.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandera as pa  # noqa: E402
from pandera.errors import SchemaError, SchemaErrors  # noqa: E402

from schemas.index_panel import INDEX_DAILY_SCHEMA  # noqa: E402
from utils.atomic_io import atomic_write_csv  # noqa: E402


def _good_df():
    return pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'open': [1.0, 1.1, 1.2],
        'high': [1.5, 1.6, 1.7],
        'low': [0.5, 0.6, 0.7],
        'close': [1.2, 1.3, 1.4],
        'volume': [100.0, 200.0, 300.0],
    })


def test_legal_dataframe_writes_successfully(tmp_path):
    target = tmp_path / 'index_daily.csv'
    df = _good_df()
    atomic_write_csv(target, df, index=False, schema=INDEX_DAILY_SCHEMA)
    assert target.exists()
    roundtrip = pd.read_csv(target)
    assert list(roundtrip.columns) == list(df.columns)
    assert len(roundtrip) == 3


def test_zero_open_row_blocks_write_and_keeps_existing_file(tmp_path):
    target = tmp_path / 'index_daily.csv'
    # Seed with a known-good prior file.
    prior = _good_df()
    atomic_write_csv(target, prior, index=False, schema=INDEX_DAILY_SCHEMA)
    assert target.exists()
    prior_bytes = target.read_bytes()

    # Now attempt to write a bad frame (open==0 on row 1).
    bad = _good_df()
    bad.loc[1, 'open'] = 0.0
    with pytest.raises((SchemaError, SchemaErrors)):
        atomic_write_csv(target, bad, index=False, schema=INDEX_DAILY_SCHEMA)

    # Target file must still contain the prior good bytes exactly.
    assert target.read_bytes() == prior_bytes, (
        "schema validation must run BEFORE atomic replace; "
        "bad write must not corrupt the existing cache file"
    )
    # No stray tmp file left behind.
    leftovers = [p for p in target.parent.iterdir() if str(p).endswith('.tmp')]
    assert not leftovers, f'tmp file leaked despite validation failure: {leftovers}'


def test_invalid_date_format_blocks_write(tmp_path):
    target = tmp_path / 'index_daily.csv'
    bad = _good_df()
    bad.loc[0, 'date'] = '2024/01/01'  # wrong separator
    with pytest.raises((SchemaError, SchemaErrors)):
        atomic_write_csv(target, bad, index=False, schema=INDEX_DAILY_SCHEMA)
    assert not target.exists(), "no target file should be created when validation fails"
    leftovers = [p for p in target.parent.iterdir() if str(p).endswith('.tmp')]
    assert not leftovers


def test_schema_none_keeps_backward_compat(tmp_path):
    """schema=None must bypass validation entirely (backward compat with
    every existing call site that does NOT pass schema=...)."""
    target = tmp_path / 'index_daily.csv'
    bad = _good_df()
    bad.loc[1, 'open'] = 0.0  # would fail under schema, must pass when schema=None
    atomic_write_csv(target, bad, index=False)  # no schema kwarg
    assert target.exists()
    roundtrip = pd.read_csv(target)
    # The zero-open row should be present, proving validation was skipped.
    assert float(roundtrip.iloc[1]['open']) == 0.0


def test_negative_volume_blocks_write(tmp_path):
    """Volume column has `ge(0)` — negative volume must fail."""
    target = tmp_path / 'index_daily.csv'
    bad = _good_df()
    bad.loc[2, 'volume'] = -1.0
    with pytest.raises((SchemaError, SchemaErrors)):
        atomic_write_csv(target, bad, index=False, schema=INDEX_DAILY_SCHEMA)
    assert not target.exists()
