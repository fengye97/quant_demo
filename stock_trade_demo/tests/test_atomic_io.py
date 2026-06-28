"""Tests for stock_trade_demo/utils/atomic_io.py."""
from __future__ import annotations

import io
import json
import os
import pickle
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

# conftest.py sits one level up and adds stock_trade_demo/ to sys.path; that
# lets us import `utils.atomic_io` regardless of the cwd pytest was invoked
# from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import atomic_io  # noqa: E402


# ──────────────────────────────────────────────
# Round-trip
# ──────────────────────────────────────────────
def test_atomic_write_text_roundtrip(tmp_path):
    target = tmp_path / "hello.txt"
    atomic_io.atomic_write_text(target, "你好\nworld\n", encoding="utf-8")
    assert target.read_text(encoding="utf-8") == "你好\nworld\n"


def test_atomic_write_bytes_roundtrip(tmp_path):
    target = tmp_path / "blob.bin"
    payload = b"\x00\x01\x02hello\xff"
    atomic_io.atomic_write_bytes(target, payload)
    assert target.read_bytes() == payload


def test_atomic_write_pickle_roundtrip(tmp_path):
    target = tmp_path / "obj.pkl"
    obj = {"a": [1, 2, 3], "b": ("x", "y")}
    atomic_io.atomic_write_pickle(target, obj)
    with open(target, "rb") as f:
        assert pickle.load(f) == obj


def test_atomic_write_json_roundtrip(tmp_path):
    target = tmp_path / "obj.json"
    obj = {"中文": True, "k": [1, 2]}
    atomic_io.atomic_write_json(target, obj, ensure_ascii=False, indent=2)
    text = target.read_text(encoding="utf-8")
    assert json.loads(text) == obj
    assert "中文" in text  # ensure_ascii=False respected
    assert "  " in text   # indent=2 respected


def test_atomic_write_csv_roundtrip_gbk(tmp_path):
    """Match the stock_data.csv use case: GBK encoding, index=False."""
    target = tmp_path / "panel.csv"
    df = pd.DataFrame({"代码": ["000001", "000002"], "收盘价": [10.5, 22.3]})
    atomic_io.atomic_write_csv(
        target, df, index=False, encoding="gbk"
    )
    loaded = pd.read_csv(target, encoding="gbk", dtype={"代码": str})
    pd.testing.assert_frame_equal(loaded, df)


def test_atomic_write_csv_byte_equivalent_to_to_csv(tmp_path):
    """Atomic write must be byte-identical to direct df.to_csv for the same kwargs."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    direct = tmp_path / "direct.csv"
    atomic = tmp_path / "atomic.csv"
    df.to_csv(direct, index=False, encoding="utf-8", lineterminator="\n")
    atomic_io.atomic_write_csv(
        atomic, df, index=False, encoding="utf-8", lineterminator="\n"
    )
    assert direct.read_bytes() == atomic.read_bytes()


def test_atomic_writer_byte_equivalent_for_csv_writer_gbk(tmp_path):
    """The streaming path used by get_stock_info.py for stock_data.csv must
    produce bytes identical to the legacy open()+csv.writer()+os.replace path.

    This is the core safety property the team-lead spec calls out: stock_data.csv
    is GBK and downstream code (backtest.py) reads it by column index, so any
    encoding / line terminator / row formatting drift would corrupt the panel.
    """
    import csv as _csv

    rows = [
        ["2026-05-30", "000001", "平安银行", "12.5", "[]", "中文字段"],
        ["2026-05-30", "000002", "万科A", "8.3", "[0.01, -0.02]", "OK"],
    ]
    legacy = tmp_path / "legacy.csv"
    new = tmp_path / "new.csv"

    # Legacy path: identical to pre-refactor stock_data.csv writer.
    tmp = str(legacy) + ".tmp"
    with open(tmp, "w", encoding="gbk", newline="") as f:
        w = _csv.writer(f)
        for r in rows:
            w.writerow(r)
    os.replace(tmp, legacy)

    # New path: atomic_writer + csv.writer.
    with atomic_io.atomic_writer(new, mode="w", encoding="gbk", newline="") as f:
        w = _csv.writer(f)
        for r in rows:
            w.writerow(r)

    assert legacy.read_bytes() == new.read_bytes()


def test_atomic_writer_byte_equivalent_realistic_stock_data_shape(tmp_path):
    """High-fidelity regression case: mimics the real stock_data.csv shape
    (55 cols, Chinese headers, mixed int/float/str, NaN/empty fields, bracket
    strings in 下周期每天涨跌幅, GBK encoding) and asserts the atomic_writer
    path used by supplement_csv_incremental is byte-identical to the legacy
    open()+csv.writer()+os.replace path on a realistic input.

    Why this case beyond the simple gbk roundtrip above: stock_data.csv is
    824MB / 55 cols / mixed dtypes, and downstream backtest.py accesses by
    column index — any drift in encoding, bracket escaping, NaN/empty
    serialization, or scientific-notation rounding would corrupt the panel.
    """
    import csv as _csv
    import filecmp

    # Reproduce the real stock_data.csv header (55 cols, all Chinese).
    header = [
        "交易日期", "股票代码", "股票名称", "是否交易", "开盘价", "最高价",
        "最低价", "收盘价", "成交量", "成交额", "换手率", "涨跌幅_今日",
        "市盈率倒数", "市净率倒数", "总市值", "流通市值",
    ]
    for i in range(len(header), 54):
        header.append(f"因子_{i:02d}")
    header.append("下周期每天涨跌幅")
    assert len(header) == 55

    # 50 rows of realistic content: int / float / empty / bracket-string lists
    # / Chinese names / leading-zero codes / mixed-precision decimals.
    rows = []
    for i in range(50):
        row = [
            f"2026-04-{(i % 28) + 1:02d}",
            f"00{i % 10}00{i % 10}",
            "中信证券" if i % 2 else "平安银行",
            "1" if i % 7 else "0",
            f"{10 + i * 0.13:.4f}",
            f"{11 + i * 0.17:.4f}",
            f"{9 + i * 0.11:.4f}",
            f"{10.5 + i * 0.15:.4f}",
            str(100000 + i * 1234),
            f"{1234567.89 + i * 1000:.2f}",
            "" if i % 11 == 0 else f"{0.012 * i:.6f}",
            f"{(-0.05 + i * 0.001):.6f}",
            "" if i % 13 == 0 else f"{0.05 + i * 0.001:.8f}",
            f"{0.1 + i * 0.0001:.10f}",
            f"{1e9 + i * 1e6:.2f}",
            f"{5e8 + i * 5e5:.2f}",
        ]
        for j in range(len(row), 54):
            row.append("" if (i + j) % 17 == 0 else f"{(j - i) * 0.007:.5f}")
        row.append(
            "[]" if i % 5 == 0
            else f"[{0.01 * i:.4f}, {-0.02 * i:.4f}, {0.003 * i:.6f}]"
        )
        assert len(row) == 55
        rows.append(row)

    legacy = tmp_path / "stock_data_legacy.csv"
    atomic = tmp_path / "stock_data_atomic.csv"

    # Legacy: exactly what supplement_csv_incremental did pre-refactor.
    tmp = str(legacy) + ".tmp"
    with open(tmp, "w", encoding="gbk", newline="") as f:
        w = _csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    os.replace(tmp, legacy)

    # New: atomic_writer + csv.writer (the refactored path).
    with atomic_io.atomic_writer(atomic, mode="w", encoding="gbk", newline="") as f:
        w = _csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

    # filecmp.cmp(shallow=False) does a true byte-by-byte compare.
    assert filecmp.cmp(legacy, atomic, shallow=False), (
        "atomic_writer must produce byte-identical output to legacy writer"
    )
    a = legacy.read_bytes()
    b = atomic.read_bytes()
    assert a == b, (
        f"byte mismatch: legacy={len(a)}B, atomic={len(b)}B; "
        f"first diff @ "
        f"{next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), 'N/A')}"
    )
    # Realism guardrail: ensure we tested non-trivial byte volume.
    assert len(a) > 5000, f"realism check: file too small ({len(a)} bytes)"


def test_atomic_write_parquet_roundtrip(tmp_path):
    pytest.importorskip("pyarrow")
    target = tmp_path / "panel.parquet"
    df = pd.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    atomic_io.atomic_write_parquet(target, df)
    loaded = pd.read_parquet(target)
    pd.testing.assert_frame_equal(loaded, df)


# ──────────────────────────────────────────────
# Crash safety
# ──────────────────────────────────────────────
def test_atomic_writer_raises_leaves_old_intact(tmp_path):
    target = tmp_path / "panel.csv"
    target.write_text("OLD-CONTENT", encoding="utf-8")
    with pytest.raises(RuntimeError, match="boom"):
        with atomic_io.atomic_writer(target, mode="w", encoding="utf-8") as f:
            f.write("HALF")
            raise RuntimeError("boom")
    # Original content must be intact, no .tmp sibling left behind.
    assert target.read_text(encoding="utf-8") == "OLD-CONTENT"
    siblings = list(tmp_path.glob("panel.csv.*.tmp"))
    assert siblings == [], f"dangling tmp files left: {siblings}"


def test_atomic_writer_raises_no_prior_file(tmp_path):
    target = tmp_path / "newfile.txt"
    with pytest.raises(ValueError):
        with atomic_io.atomic_writer(target, mode="w", encoding="utf-8") as f:
            f.write("partial")
            raise ValueError("simulated crash")
    assert not target.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_csv_raises_leaves_old_intact(tmp_path, monkeypatch):
    target = tmp_path / "panel.csv"
    target.write_text("OLD\n", encoding="utf-8")

    class BoomDF:
        def to_csv(self, tmp, **kw):  # noqa: D401
            # Simulate pandas writing a partial tmp file then crashing.
            with open(tmp, "w") as f:
                f.write("HALF")
            raise IOError("disk full")

    with pytest.raises(IOError):
        atomic_io.atomic_write_csv(target, BoomDF(), index=False)
    assert target.read_text(encoding="utf-8") == "OLD\n"
    assert list(tmp_path.glob("panel.csv.*.tmp")) == []


# ──────────────────────────────────────────────
# Stale tmp tolerance
# ──────────────────────────────────────────────
def test_atomic_writer_tolerates_preexisting_tmp(tmp_path, monkeypatch):
    """A stale same-name tmp from a previous crash must not break the next write.

    We force the tmp suffix to a fixed value so a pre-existing tmp at exactly
    that path is hit by the next call. The atomic writer should overwrite it
    (open mode 'w' truncates) and rename succeed.
    """
    target = tmp_path / "panel.txt"
    fixed_tmp = str(target) + ".999.deadbeef.tmp"
    Path(fixed_tmp).write_text("PREVIOUS-CRASH", encoding="utf-8")

    monkeypatch.setattr(atomic_io, "_tmp_path", lambda p: fixed_tmp)

    atomic_io.atomic_write_text(target, "FRESH", encoding="utf-8")
    assert target.read_text(encoding="utf-8") == "FRESH"
    # tmp gone after replace
    assert not Path(fixed_tmp).exists()


# ──────────────────────────────────────────────
# PROTECTED_PATHS guard
# ──────────────────────────────────────────────
def test_protected_paths_blocks_write(tmp_path, monkeypatch):
    target = tmp_path / "live_trades.csv"
    target.write_text("date,code\n", encoding="utf-8")
    real = os.path.realpath(target)
    monkeypatch.setattr(atomic_io, "PROTECTED_PATHS", frozenset({real}))

    with pytest.raises(PermissionError, match="protected"):
        atomic_io.atomic_write_text(target, "MALICIOUS", encoding="utf-8")
    with pytest.raises(PermissionError):
        atomic_io.atomic_write_bytes(target, b"x")
    with pytest.raises(PermissionError):
        atomic_io.atomic_write_pickle(target, {})
    with pytest.raises(PermissionError):
        atomic_io.atomic_write_json(target, {})
    with pytest.raises(PermissionError):
        atomic_io.atomic_write_csv(target, pd.DataFrame({"a": [1]}), index=False)
    with pytest.raises(PermissionError):
        with atomic_io.atomic_writer(target, mode="w", encoding="utf-8") as f:
            f.write("nope")

    assert target.read_text(encoding="utf-8") == "date,code\n"


def test_protected_paths_force_override_allows_write(tmp_path, monkeypatch):
    target = tmp_path / "live_trades.csv"
    target.write_text("date,code\n", encoding="utf-8")
    real = os.path.realpath(target)
    monkeypatch.setattr(atomic_io, "PROTECTED_PATHS", frozenset({real}))

    atomic_io.atomic_write_text(
        target, "AUTHORIZED", encoding="utf-8", _force_override=True
    )
    assert target.read_text(encoding="utf-8") == "AUTHORIZED"


# ──────────────────────────────────────────────
# sweep_dangling_tmps
# ──────────────────────────────────────────────
def test_sweep_dangling_tmps_removes_only_stale(tmp_path):
    stale = tmp_path / "old.csv.123.aaa.tmp"
    fresh = tmp_path / "new.csv.456.bbb.tmp"
    stale.write_text("stale", encoding="utf-8")
    fresh.write_text("fresh", encoding="utf-8")

    # Make stale's mtime 2 hours ago.
    two_hours_ago = time.time() - 7200
    os.utime(stale, (two_hours_ago, two_hours_ago))

    removed = atomic_io.sweep_dangling_tmps(tmp_path, max_age_seconds=3600)
    assert removed == 1
    assert not stale.exists()
    assert fresh.exists()


def test_sweep_dangling_tmps_missing_dir_returns_zero(tmp_path):
    nonexistent = tmp_path / "no-such-dir"
    assert atomic_io.sweep_dangling_tmps(nonexistent) == 0


def test_sweep_dangling_tmps_no_tmps(tmp_path):
    (tmp_path / "real.csv").write_text("ok", encoding="utf-8")
    assert atomic_io.sweep_dangling_tmps(tmp_path) == 0
