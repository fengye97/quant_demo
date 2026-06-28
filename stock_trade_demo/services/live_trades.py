"""Live-trades persistence service.

Owns ALL read/write/lock/atomic-rename logic for `data/live_trades.csv`.
CLAUDE.md rule 15 is enforced here：
  - never delete / clear / rebuild the file
  - never seed example rows; if absent, create with header only
  - schema extensions append columns with default '' (backwards compatible)
  - writes use tmp + fsync + os.replace + intra-process lock + fcntl cross-process lock

Public API:
  LIVE_TRADES_FILE      — canonical path
  LIVE_TRADES_COLUMNS   — current schema
  read_all()            — return list[dict] (locked)
  write_all(rows)       — atomic replace whole file (locked)
  append_record(row)    — assign next record_id, append, persist (locked)
  delete_record(rid)    — remove by record_id, persist (locked); returns bool
  latest_position(strategy_id) -> float
  transaction()         — context manager yielding mutable rows list; commit on exit
"""
from __future__ import annotations

import csv as _csv
import errno
import fcntl
import os
import threading
from contextlib import contextmanager
from typing import Iterator, List, Optional

LIVE_DATA_DIR: str = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data')
)
LIVE_TRADES_FILE: str = os.path.join(LIVE_DATA_DIR, 'live_trades.csv')

LIVE_TRADES_COLUMNS: List[str] = [
    'record_id', 'date', 'strategy', 'signal_target', 'actual_position',
    'exec_price', 'capital', 'notes', 'created_at',
    # Appended later (向后兼容，缺失视为空字符串)：成交股数；用于按 价格×股数/初始资金 反算 actual_position
    'shares',
]

_INTRAPROCESS_LOCK = threading.RLock()
_LOCKFILE_PATH = LIVE_TRADES_FILE + '.lock'


def _ensure_file_exists() -> None:
    """If the canonical CSV is missing, create it with ONLY the header.

    CLAUDE.md rule 15: never seed with example/demo rows; the live ledger
    starts from an empty book.
    """
    os.makedirs(LIVE_DATA_DIR, exist_ok=True)
    if not os.path.exists(LIVE_TRADES_FILE):
        with open(LIVE_TRADES_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = _csv.DictWriter(f, fieldnames=LIVE_TRADES_COLUMNS)
            writer.writeheader()


@contextmanager
def _file_lock() -> Iterator[None]:
    """Acquire intra-process RLock + inter-process fcntl lock.

    Wraps every read/modify/write cycle so multi-worker WSGI deployments don't
    interleave writes. fcntl is best-effort on macOS/Linux; if it fails (e.g.
    on read-only filesystems), we still hold the RLock so the in-process
    handlers serialize correctly.
    """
    with _INTRAPROCESS_LOCK:
        os.makedirs(LIVE_DATA_DIR, exist_ok=True)
        lock_fd = None
        try:
            lock_fd = os.open(_LOCKFILE_PATH, os.O_CREAT | os.O_RDWR, 0o644)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
            except OSError as e:
                if e.errno not in (errno.ENOSYS, errno.EOPNOTSUPP, errno.EINVAL):
                    raise
            yield
        finally:
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                try:
                    os.close(lock_fd)
                except OSError:
                    pass


def _read_all_unlocked() -> List[dict]:
    _ensure_file_exists()
    rows: List[dict] = []
    with open(LIVE_TRADES_FILE, 'r', newline='', encoding='utf-8') as f:
        reader = _csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def _write_all_unlocked(rows: List[dict]) -> None:
    """Atomic write: tmp + fsync + os.replace.

    Never deletes or truncates LIVE_TRADES_FILE in place; the only way it ever
    changes is via this function's atomic rename.
    """
    _ensure_file_exists()
    tmp_path = LIVE_TRADES_FILE + '.tmp'
    with open(tmp_path, 'w', newline='', encoding='utf-8') as f:
        writer = _csv.DictWriter(f, fieldnames=LIVE_TRADES_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in LIVE_TRADES_COLUMNS})
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp_path, LIVE_TRADES_FILE)


def read_all() -> List[dict]:
    """Read the full ledger; safe to call concurrently."""
    with _file_lock():
        return _read_all_unlocked()


def write_all(rows: List[dict]) -> None:
    """Replace the full ledger atomically."""
    with _file_lock():
        _write_all_unlocked(rows)


@contextmanager
def transaction() -> Iterator[List[dict]]:
    """Lock, read, yield mutable rows; auto-persist on clean exit.

    Use inside route handlers that need read-modify-write semantics:

        with services.live_trades.transaction() as rows:
            rows.append(new_row)
    """
    with _file_lock():
        rows = _read_all_unlocked()
        yield rows
        _write_all_unlocked(rows)


def _next_record_id(rows: List[dict]) -> int:
    max_id = 0
    for r in rows:
        try:
            max_id = max(max_id, int(r.get('record_id') or 0))
        except (TypeError, ValueError):
            continue
    return max_id + 1


def append_record(row: dict) -> dict:
    """Append a single record; assigns record_id atomically; returns the stored row."""
    with _file_lock():
        rows = _read_all_unlocked()
        new_id = _next_record_id(rows)
        stored = {k: row.get(k, '') for k in LIVE_TRADES_COLUMNS}
        stored['record_id'] = str(new_id)
        rows.append(stored)
        _write_all_unlocked(rows)
        return stored


def delete_record(record_id) -> bool:
    """Delete a record by record_id; returns True if removed, False if absent."""
    target = str(record_id)
    with _file_lock():
        rows = _read_all_unlocked()
        kept = [r for r in rows if str(r.get('record_id')) != target]
        if len(kept) == len(rows):
            return False
        _write_all_unlocked(kept)
        return True


def latest_position(strategy_id: str) -> float:
    """Latest `actual_position` for a strategy; 0.0 if empty or unparseable.

    Used by the "最新入市信号卡" to phrase 操作建议 as 实盘当前仓 vs 今仓目标
    instead of strategy yesterday vs strategy today.
    """
    try:
        rows = read_all()
    except Exception:
        return 0.0
    rows = [r for r in rows if r.get('strategy') == strategy_id and r.get('date')]
    if not rows:
        return 0.0
    rows.sort(key=lambda r: r['date'])
    try:
        return float(rows[-1].get('actual_position') or 0.0)
    except (TypeError, ValueError):
        return 0.0
