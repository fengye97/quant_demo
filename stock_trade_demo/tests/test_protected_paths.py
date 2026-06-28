"""Tests for PROTECTED_PATHS guard in atomic_io.

Strategy:
  - Monkeypatch ``utils.atomic_io.PROTECTED_PATHS`` to point at a tmpdir path
    so the real ``data/live_trades.csv`` is never touched.
  - Verify that calling any atomic_write_* towards the monkeypatched path raises
    PermissionError without writing anything.
  - Verify that the same call with ``_force_override=True`` succeeds.
"""
from __future__ import annotations

import os
import pandas as pd
import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def fake_protected(tmp_path, monkeypatch):
    """Return a path inside tmp_path and monkeypatch PROTECTED_PATHS to contain it."""
    import stock_trade_demo.utils.atomic_io as _aio
    target = str(os.path.realpath(tmp_path / "fake_live_trades.csv"))
    monkeypatch.setattr(_aio, "PROTECTED_PATHS", frozenset({target}))
    return target


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2], "b": [3, 4]})


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestProtectedPathsBlocked:
    """Every atomic_write_* must raise PermissionError for a protected path."""

    def test_csv_raises(self, fake_protected):
        from stock_trade_demo.utils.atomic_io import atomic_write_csv
        with pytest.raises(PermissionError, match="protected path"):
            atomic_write_csv(fake_protected, _empty_df(), index=False)
        assert not os.path.exists(fake_protected), "file must not be created"

    def test_parquet_raises(self, fake_protected):
        from stock_trade_demo.utils.atomic_io import atomic_write_parquet
        target = fake_protected.replace(".csv", ".parquet")
        # monkeypatch for this different extension path
        import stock_trade_demo.utils.atomic_io as _aio
        import pytest as _pytest
        with _pytest.MonkeyPatch().context() as mp:
            mp.setattr(_aio, "PROTECTED_PATHS", frozenset({os.path.realpath(target)}))
            with pytest.raises(PermissionError, match="protected path"):
                atomic_write_parquet(target, _empty_df())
        assert not os.path.exists(target), "file must not be created"

    def test_pickle_raises(self, fake_protected):
        from stock_trade_demo.utils.atomic_io import atomic_write_pickle
        target = fake_protected.replace(".csv", ".pkl")
        import stock_trade_demo.utils.atomic_io as _aio
        import pytest as _pytest
        with _pytest.MonkeyPatch().context() as mp:
            mp.setattr(_aio, "PROTECTED_PATHS", frozenset({os.path.realpath(target)}))
            with pytest.raises(PermissionError, match="protected path"):
                atomic_write_pickle(target, {"x": 1})
        assert not os.path.exists(target), "file must not be created"

    def test_json_raises(self, fake_protected):
        from stock_trade_demo.utils.atomic_io import atomic_write_json
        target = fake_protected.replace(".csv", ".json")
        import stock_trade_demo.utils.atomic_io as _aio
        import pytest as _pytest
        with _pytest.MonkeyPatch().context() as mp:
            mp.setattr(_aio, "PROTECTED_PATHS", frozenset({os.path.realpath(target)}))
            with pytest.raises(PermissionError, match="protected path"):
                atomic_write_json(target, {"key": "value"})
        assert not os.path.exists(target), "file must not be created"

    def test_text_raises(self, fake_protected):
        from stock_trade_demo.utils.atomic_io import atomic_write_text
        target = fake_protected.replace(".csv", ".txt")
        import stock_trade_demo.utils.atomic_io as _aio
        import pytest as _pytest
        with _pytest.MonkeyPatch().context() as mp:
            mp.setattr(_aio, "PROTECTED_PATHS", frozenset({os.path.realpath(target)}))
            with pytest.raises(PermissionError, match="protected path"):
                atomic_write_text(target, "hello")
        assert not os.path.exists(target), "file must not be created"

    def test_writer_ctx_raises(self, fake_protected):
        from stock_trade_demo.utils.atomic_io import atomic_writer
        with pytest.raises(PermissionError, match="protected path"):
            with atomic_writer(fake_protected, "w") as f:
                f.write("data")
        assert not os.path.exists(fake_protected), "file must not be created"


class TestProtectedPathsForceOverride:
    """_force_override=True must bypass the guard and write successfully."""

    def test_csv_force_override(self, fake_protected):
        from stock_trade_demo.utils.atomic_io import atomic_write_csv
        atomic_write_csv(fake_protected, _empty_df(), index=False, _force_override=True)
        assert os.path.exists(fake_protected)
        result = pd.read_csv(fake_protected)
        assert list(result.columns) == ["a", "b"]
        assert len(result) == 2

    def test_writer_ctx_force_override(self, fake_protected):
        from stock_trade_demo.utils.atomic_io import atomic_writer
        with atomic_writer(fake_protected, "w", _force_override=True) as f:
            f.write("allowed")
        with open(fake_protected) as fh:
            assert fh.read() == "allowed"


class TestRealLiveTradesPathInProtectedPaths:
    """PROTECTED_PATHS must contain the realpath of data/live_trades.csv."""

    def test_live_trades_path_present(self):
        from stock_trade_demo.utils.atomic_io import PROTECTED_PATHS
        expected = os.path.realpath(
            os.path.join(
                os.path.dirname(__file__),
                "..", "..", "data", "live_trades.csv",
            )
        )
        assert expected in PROTECTED_PATHS, (
            f"Expected {expected!r} in PROTECTED_PATHS but got {PROTECTED_PATHS!r}"
        )

    def test_live_trades_not_writable_via_atomic_csv(self, tmp_path, monkeypatch):
        """Even if live_trades.csv doesn't exist on disk, atomic_write_csv must refuse."""
        from stock_trade_demo.utils.atomic_io import PROTECTED_PATHS, atomic_write_csv
        # Pick one path from the real PROTECTED_PATHS
        real_protected = next(iter(PROTECTED_PATHS))
        with pytest.raises(PermissionError):
            atomic_write_csv(real_protected, _empty_df())
