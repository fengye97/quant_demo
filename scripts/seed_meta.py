#!/usr/bin/env python3
"""Backfill ``<file>.meta.json`` sidecars for legacy data files.

Pillar 2 Step 6 introduces ``atomic_write_*(produced_by=...)`` which writes
a JSON sidecar next to every new data file. Files that existed before that
change have no sidecar; this script walks the project caches and seeds a
``produced_by="unknown_legacy"`` sidecar using the file's mtime as
``written_at_unix`` so ``check_data_freshness.py`` can prefer meta.json
across the entire fleet.

Idempotent: existing ``*.meta.json`` files are NEVER overwritten — we only
fill in the missing ones. Re-running is a no-op after the first pass.

Scope (configurable via CLI ``--root`` flags):
  - stock_trade_demo/.cache/
  - data/
  - strategy/

Extensions covered: ``.csv``, ``.parquet``, ``.pkl``, ``.json``
(``*.meta.json`` files are skipped, as is the protected
``data/live_trades.csv`` per CLAUDE.md red line).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOTS = [
    REPO_ROOT / "stock_trade_demo" / ".cache",
    REPO_ROOT / "data",
    REPO_ROOT / "strategy",
]

DATA_EXTENSIONS = {".csv", ".parquet", ".pkl", ".json"}

# CLAUDE.md red line: data/live_trades.csv is sacred — do NOT touch.
PROTECTED = {
    str((REPO_ROOT / "data" / "live_trades.csv").resolve()),
}


def _is_meta_sidecar(p: Path) -> bool:
    # Two suffixes — pathlib only sees the last one (".json"), so check name.
    return p.name.endswith(".meta.json")


def _is_lock_or_tmp(p: Path) -> bool:
    name = p.name
    return name.endswith(".tmp") or name.endswith(".lock")


def _sha256_first_8k(path: Path) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            buf = f.read(8192)
    except OSError:
        return None
    # Truncated SHA-256: match utils.atomic_io._sha256_first_8k (32 hex chars).
    return hashlib.sha256(buf).hexdigest()[:32]


def _row_count_if_obvious(path: Path) -> Optional[int]:
    """Best-effort row count for CSV / Parquet. Heavy formats skipped silently."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            # Cheap line count; subtract 1 for header. Don't decode payload.
            with open(path, "rb") as f:
                n = sum(1 for _ in f)
            return max(n - 1, 0)
        if suffix == ".parquet":
            try:
                import pyarrow.parquet as pq  # type: ignore

                return int(pq.ParquetFile(path).metadata.num_rows)
            except Exception:
                return None
    except OSError:
        return None
    return None


def _iter_candidates(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for fname in filenames:
                p = Path(dirpath) / fname
                if _is_meta_sidecar(p) or _is_lock_or_tmp(p):
                    continue
                if p.suffix.lower() not in DATA_EXTENSIONS:
                    continue
                try:
                    if str(p.resolve()) in PROTECTED:
                        continue
                except OSError:
                    continue
                yield p


def _seed_one(
    data_path: Path,
    *,
    produced_by: str,
    dry_run: bool,
) -> Tuple[bool, str]:
    """Return (wrote, reason). ``wrote=False`` means skipped or dry-run."""
    meta_path = data_path.with_name(data_path.name + ".meta.json")
    if meta_path.exists():
        return False, "exists"
    try:
        st = data_path.stat()
    except OSError as e:
        return False, f"stat-failed:{e}"
    mtime = float(st.st_mtime)
    payload = {
        "produced_by": produced_by,
        "written_at_iso": datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(),
        "written_at_unix": mtime,
        "sha256_first_8k": _sha256_first_8k(data_path),
        "_seeded_by": "scripts/seed_meta.py",
    }
    rc = _row_count_if_obvious(data_path)
    if rc is not None:
        payload["row_count_if_df"] = rc
    if dry_run:
        return False, "would-seed"
    try:
        tmp = meta_path.with_name(meta_path.name + f".{os.getpid()}.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, meta_path)
    except OSError as e:
        return False, f"write-failed:{e}"
    return True, "seeded"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", action="append", default=None,
        help="Root directory to scan (repeatable). Default: cache+data+strategy.",
    )
    parser.add_argument(
        "--produced-by", default="unknown_legacy",
        help='Value to write for legacy files (default: "unknown_legacy")',
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Walk but do not write any sidecars; print what would be done.",
    )
    args = parser.parse_args(argv)

    roots = [Path(r) for r in args.root] if args.root else DEFAULT_ROOTS

    print(f"[seed_meta] scanning roots: {[str(r) for r in roots]}", flush=True)
    print(f"[seed_meta] produced_by   : {args.produced_by!r}", flush=True)
    print(f"[seed_meta] dry_run       : {args.dry_run}", flush=True)
    seeded = 0
    skipped_exist = 0
    other = 0
    total = 0
    for p in _iter_candidates(roots):
        total += 1
        wrote, reason = _seed_one(
            p, produced_by=args.produced_by, dry_run=args.dry_run
        )
        if wrote:
            seeded += 1
            print(f"  + {p.relative_to(REPO_ROOT)}", flush=True)
        elif reason == "exists":
            skipped_exist += 1
        elif reason == "would-seed":
            seeded += 1  # count for dry-run summary
            print(f"  ? {p.relative_to(REPO_ROOT)} (dry-run)", flush=True)
        else:
            other += 1
            print(f"  ! {p.relative_to(REPO_ROOT)} → {reason}", flush=True)

    print("\n" + "=" * 60)
    print(f"  scanned    : {total}")
    print(f"  seeded     : {seeded}{' (dry-run)' if args.dry_run else ''}")
    print(f"  skipped    : {skipped_exist} (already had meta.json)")
    print(f"  errors     : {other}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
