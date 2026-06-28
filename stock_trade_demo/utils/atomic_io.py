"""Atomic file write utilities.

All public APIs in this module write to a sibling ``.tmp`` file, ``flush()`` +
``os.fsync()`` it, then ``os.replace`` it onto the target path. This guarantees
that a crashed or Ctrl-C'd process can only leave the on-disk file in one of
two states: the **old** intact content, or the **new** intact content. There is
no half-written byte range visible to readers.

API
---
- :func:`atomic_write_text` / :func:`atomic_write_bytes`
- :func:`atomic_write_pickle` / :func:`atomic_write_json`
- :func:`atomic_write_csv` (transparently forwards all ``DataFrame.to_csv``
  kwargs including ``encoding``, ``index``, ``lineterminator``, ``quoting`` …)
- :func:`atomic_write_parquet` (transparently forwards all
  ``DataFrame.to_parquet`` kwargs)
- :func:`atomic_writer` — streaming context manager
- :func:`sweep_dangling_tmps` — cleanup of orphaned ``*.tmp`` files

Protected paths
---------------
``PROTECTED_PATHS`` is a frozenset of absolute ``os.path.realpath`` strings
that the atomic writers refuse to touch. It starts empty; Step 4 of the data
migration plan (`docs/plan_data.md`) populates it with
``data/live_trades.csv``. Callers that legitimately need to write a protected
path must pass ``_force_override=True`` (only the dedicated live-trades
service should do so).
"""

from __future__ import annotations

import contextlib
import errno
import glob as _glob
import hashlib as _hashlib
import json as _json
import logging
import os
import pickle as _pickle
import time as _time
import uuid as _uuid
from datetime import datetime as _datetime, timezone as _timezone
from typing import Any, Iterator, Optional, Union

PathLike = Union[str, "os.PathLike[str]"]

logger = logging.getLogger(__name__)

# Populated by `docs/plan_data.md` Step 4. Keep as a module-level frozenset so
# callers can monkeypatch it in tests without touching the implementation.
_LIVE_TRADES_PATH = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "live_trades.csv")
)
PROTECTED_PATHS: "frozenset[str]" = frozenset({_LIVE_TRADES_PATH})


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def _to_str(path: PathLike) -> str:
    return os.fspath(path)


def _check_protected(path: str, force_override: bool) -> None:
    """Raise :class:`PermissionError` if writing ``path`` is forbidden."""
    if not PROTECTED_PATHS:
        return
    try:
        real = os.path.realpath(path)
    except OSError:
        real = os.path.abspath(path)
    if real in PROTECTED_PATHS and not force_override:
        raise PermissionError(
            f"Refusing to overwrite protected path {real!r}. "
            "Pass _force_override=True only from the dedicated service."
        )


def _tmp_path(path: str) -> str:
    """Build a unique sibling tmp name (same directory, same filesystem)."""
    suffix = f".{os.getpid()}.{_uuid.uuid4().hex[:8]}.tmp"
    return path + suffix


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _fsync_fileno(fileno: int) -> None:
    """Best-effort ``os.fsync``. EINVAL (non-syncable fd) is tolerated."""
    try:
        os.fsync(fileno)
    except OSError as e:
        # Some filesystems / platforms (notably tmpfs on certain configs) don't
        # support fsync on regular files; we still want the rename to win.
        if e.errno not in (errno.EINVAL, errno.ENOTSUP):
            raise


def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError:
        logger.debug("failed to remove tmp file %s", path, exc_info=True)


# ──────────────────────────────────────────────
# meta.json lineage (Pillar 2 Step 6)
# ──────────────────────────────────────────────
def _sha256_first_8k(path: str) -> Optional[str]:
    """Return SHA-256 hex digest over the first 8 KiB of ``path``.

    Cheap fingerprint — sufficient for detecting silent corruption between
    runs without paying full-file hashing cost on multi-MB CSVs."""
    try:
        with open(path, "rb") as f:
            buf = f.read(8192)
    except OSError:
        return None
    # Truncate to 32 hex chars — sufficient for silent-corruption detection
    # without storing a full-length digest in every sidecar.
    return _hashlib.sha256(buf).hexdigest()[:32]


def _row_count_if_df(df: Any) -> Optional[int]:
    try:
        return int(len(df))
    except Exception:
        return None


def _write_meta_sidecar(
    path: str,
    produced_by: Optional[str],
    df: Any = None,
) -> None:
    """Write ``<path>.meta.json`` next to a just-replaced data file.

    Best-effort: failures are logged but never propagated — meta lineage is
    a debugging aid, not a correctness guarantee, and must not break the
    write that already succeeded.
    """
    if not produced_by:
        return
    meta_path = path + ".meta.json"
    now = _time.time()
    payload: dict = {
        "produced_by": produced_by,
        "written_at_iso": _datetime.fromtimestamp(now, tz=_timezone.utc).isoformat(),
        "written_at_unix": now,
        "sha256_first_8k": _sha256_first_8k(path),
    }
    rc = _row_count_if_df(df)
    if rc is not None:
        payload["row_count_if_df"] = rc
    try:
        # Write the sidecar atomically too — same tmp+rename pattern.
        tmp = _tmp_path(meta_path)
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(_json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
            f.flush()
            try:
                _fsync_fileno(f.fileno())
            except (AttributeError, ValueError):
                pass
        os.replace(tmp, meta_path)
    except Exception:
        logger.debug(
            "failed to write meta sidecar for %s (produced_by=%r)",
            path, produced_by, exc_info=True,
        )


def read_meta(path: PathLike) -> Optional[dict]:
    """Read the ``<path>.meta.json`` sidecar if present.

    Returns ``None`` if the sidecar does not exist or cannot be parsed —
    callers (e.g. ``check_data_freshness.py``) should fall back to mtime."""
    meta_path = _to_str(path) + ".meta.json"
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return _json.load(f)
    except (OSError, ValueError):
        return None


# ──────────────────────────────────────────────
# Streaming context manager
# ──────────────────────────────────────────────
@contextlib.contextmanager
def atomic_writer(
    path: PathLike,
    mode: str = "w",
    encoding: Optional[str] = None,
    *,
    newline: Optional[str] = None,
    produced_by: Optional[str] = None,
    _force_override: bool = False,
    **open_kwargs: Any,
) -> Iterator[Any]:
    """Yield a file handle that writes to a tmp file and atomically replaces.

    On normal exit: ``flush`` + ``fsync`` + ``os.replace(tmp, path)``.
    On exception: the tmp file is removed and ``path`` is left untouched.

    ``mode`` and ``encoding`` follow built-in :func:`open` semantics; binary
    modes (``"wb"`` / ``"ab"``) automatically ignore ``encoding``.
    """
    path_s = _to_str(path)
    _check_protected(path_s, _force_override)
    _ensure_parent(path_s)
    tmp = _tmp_path(path_s)

    binary = "b" in mode
    open_kwargs = dict(open_kwargs)
    if not binary and encoding is not None:
        open_kwargs["encoding"] = encoding
    if newline is not None:
        open_kwargs["newline"] = newline

    f = open(tmp, mode, **open_kwargs)
    try:
        yield f
        f.flush()
        try:
            _fsync_fileno(f.fileno())
        except (AttributeError, ValueError):
            # Some wrappers (BufferedWriter on closed file etc.) — best effort.
            pass
        f.close()
        os.replace(tmp, path_s)
        _write_meta_sidecar(path_s, produced_by)
    except BaseException:
        try:
            f.close()
        finally:
            _safe_remove(tmp)
        raise


# ──────────────────────────────────────────────
# Simple value writers
# ──────────────────────────────────────────────
def atomic_write_text(
    path: PathLike,
    content: str,
    *,
    encoding: str = "utf-8",
    produced_by: Optional[str] = None,
    _force_override: bool = False,
) -> None:
    """Atomically write ``content`` to ``path``."""
    with atomic_writer(
        path,
        mode="w",
        encoding=encoding,
        produced_by=produced_by,
        _force_override=_force_override,
    ) as f:
        f.write(content)


def atomic_write_bytes(
    path: PathLike,
    content: bytes,
    *,
    produced_by: Optional[str] = None,
    _force_override: bool = False,
) -> None:
    """Atomically write ``content`` (bytes) to ``path``."""
    with atomic_writer(
        path, mode="wb", produced_by=produced_by, _force_override=_force_override
    ) as f:
        f.write(content)


def atomic_write_pickle(
    path: PathLike,
    obj: Any,
    *,
    protocol: int = _pickle.HIGHEST_PROTOCOL,
    produced_by: Optional[str] = None,
    _force_override: bool = False,
) -> None:
    """Pickle ``obj`` to ``path`` atomically (defaults to HIGHEST_PROTOCOL)."""
    with atomic_writer(
        path, mode="wb", produced_by=produced_by, _force_override=_force_override
    ) as f:
        _pickle.dump(obj, f, protocol=protocol)


def atomic_write_json(
    path: PathLike,
    obj: Any,
    *,
    ensure_ascii: bool = False,
    indent: Optional[int] = None,
    sort_keys: bool = False,
    encoding: str = "utf-8",
    produced_by: Optional[str] = None,
    _force_override: bool = False,
) -> None:
    """Serialize ``obj`` as JSON and write atomically.

    Uses ``allow_nan=False`` to raise ``ValueError`` if the payload contains
    float NaN / Inf — these are not valid JSON (RFC 8259) and would silently
    produce malformed files that break browser ``JSON.parse``.  Callers must
    sanitize NaN values before passing to this function.
    """
    payload = _json.dumps(
        obj, ensure_ascii=ensure_ascii, indent=indent, sort_keys=sort_keys,
        allow_nan=False,
    )
    with atomic_writer(
        path,
        mode="w",
        encoding=encoding,
        produced_by=produced_by,
        _force_override=_force_override,
    ) as f:
        f.write(payload)


# ──────────────────────────────────────────────
# pandas writers
# ──────────────────────────────────────────────
def atomic_write_csv(
    path: PathLike,
    df: "Any",
    *,
    schema: Any = None,
    schema_sample: Optional[int] = None,
    produced_by: Optional[str] = None,
    _force_override: bool = False,
    **to_csv_kwargs: Any,
) -> None:
    """Atomically write a DataFrame to CSV.

    All keyword arguments are forwarded **verbatim** to :meth:`DataFrame.to_csv`
    (``encoding``, ``index``, ``lineterminator``, ``quoting``, ``sep`` …). This
    is required for byte-equivalent migration of existing call sites such as
    ``stock_data.csv`` (GBK + ``index=False``).

    If ``schema`` is provided (a ``pandera.DataFrameSchema``), the DataFrame
    is validated **before** any tmp file is created. A validation failure
    raises ``pandera.errors.SchemaError`` /
    ``pandera.errors.SchemaErrors`` and the on-disk target file is left
    completely untouched. Pass ``schema=None`` (default) to skip validation
    (backward-compatible).

    ``schema_sample`` (Pillar 2 Step 5 慢热模式): when set, validate only the
    first ``schema_sample`` rows + a tail sample of the same size — this keeps
    schema checks under ~100 ms even on multi-million-row panels like
    ``stock_data.csv`` while still catching most regressions (latest-month
    rows + earliest history). ``schema_sample=None`` validates the entire
    DataFrame (default).
    """
    path_s = _to_str(path)
    _check_protected(path_s, _force_override)
    if schema is not None:
        # Validate first; on failure no tmp file is even created.
        # ``lazy=True`` collects all errors instead of failing on the first.
        if schema_sample is None or len(df) <= 2 * schema_sample:
            to_check = df
        else:
            # Head + tail sample — cheap, covers the two corners we care about
            # (latest month just written + early history that should never
            # silently mutate).
            import pandas as _pd_local  # local import to avoid hard dep at top
            to_check = _pd_local.concat([df.head(schema_sample), df.tail(schema_sample)])
        schema.validate(to_check, lazy=True)
    _ensure_parent(path_s)
    tmp = _tmp_path(path_s)
    try:
        df.to_csv(tmp, **to_csv_kwargs)
        # to_csv closed its own handle; explicitly fsync the rebuilt fd.
        fd = os.open(tmp, os.O_RDONLY)
        try:
            _fsync_fileno(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path_s)
        _write_meta_sidecar(path_s, produced_by, df=df)
    except BaseException:
        _safe_remove(tmp)
        raise


def atomic_write_parquet(
    path: PathLike,
    df: "Any",
    *,
    produced_by: Optional[str] = None,
    _force_override: bool = False,
    **to_parquet_kwargs: Any,
) -> None:
    """Atomically write a DataFrame to Parquet."""
    path_s = _to_str(path)
    _check_protected(path_s, _force_override)
    _ensure_parent(path_s)
    tmp = _tmp_path(path_s)
    try:
        df.to_parquet(tmp, **to_parquet_kwargs)
        fd = os.open(tmp, os.O_RDONLY)
        try:
            _fsync_fileno(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path_s)
        _write_meta_sidecar(path_s, produced_by, df=df)
    except BaseException:
        _safe_remove(tmp)
        raise


# ──────────────────────────────────────────────
# Maintenance
# ──────────────────────────────────────────────
def sweep_dangling_tmps(
    directory: PathLike,
    *,
    max_age_seconds: int = 3600,
    recursive: bool = False,
) -> int:
    """Delete ``*.tmp`` files older than ``max_age_seconds`` in ``directory``.

    Used at the entry of long-running supplement jobs so that crashed previous
    runs don't leave half-written tmp siblings lying around. Returns the
    number of files actually removed.
    """
    directory_s = _to_str(directory)
    if not os.path.isdir(directory_s):
        return 0
    pattern = "**/*.tmp" if recursive else "*.tmp"
    candidates = _glob.glob(os.path.join(directory_s, pattern), recursive=recursive)
    now = _time.time()
    removed = 0
    for path in candidates:
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        if (now - mtime) < max_age_seconds:
            continue
        try:
            os.remove(path)
            removed += 1
            logger.info("swept dangling tmp file %s (age %.0fs)", path, now - mtime)
        except OSError:
            logger.debug("failed to sweep tmp %s", path, exc_info=True)
    return removed


__all__ = [
    "PROTECTED_PATHS",
    "atomic_writer",
    "atomic_write_text",
    "atomic_write_bytes",
    "atomic_write_pickle",
    "atomic_write_json",
    "atomic_write_csv",
    "atomic_write_parquet",
    "sweep_dangling_tmps",
    "read_meta",
]
