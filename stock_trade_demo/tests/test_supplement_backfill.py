"""Regression tests for `get_stock_info.supplement_csv_incremental` col54 backfill.

防的是 eb4d369：supplement 时应该把"上个月最后一行"的 col54（下周期每天涨跌幅）
回填为本月的日收益。原 bug 是 `stock_data[code][-1]` 在当月已有行（restart-safe
supplement 的中间态）时指向当月行而不是上月行，`startswith(prev_month)` 永远 False，
silent skip。

修复方式：从尾部 `reversed(stock_data[code])` 搜索找到 prev_month 那一行才回填。

本测试也覆盖保护 2：plan_ratio / apply_ratio < 0.90 必须 raise（CLAUDE.md 提到的
expected/actual 覆盖率硬约束）。
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import pytest

# get_stock_info.py 在 repo root（CLAUDE.md：/Users/fatcat/Desktop/quant），
# 而 conftest 默认只把 stock_trade_demo/ 加进 sys.path。
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import get_stock_info as gsi


CSV_HEADER_LEN = 55  # 与 get_stock_info.CSV_HEADERS 对齐
COL54_INITIAL = '[]'


def _make_row(date: str, code: str, col54: str = COL54_INITIAL) -> list[str]:
    row = [''] * CSV_HEADER_LEN
    row[0] = date
    row[1] = code
    row[2] = f'name_{code}'
    row[10] = '1.0e9'
    row[11] = '1.0e9'
    row[12] = '20'
    row[7] = '10.0'
    row[54] = col54
    return row


def _write_csv(rows: list[list[str]], path: Path) -> None:
    with open(path, 'w', encoding='gbk', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(gsi.CSV_HEADERS)
        for r in rows:
            writer.writerow(r)


def _read_csv(path: Path) -> list[list[str]]:
    with open(path, 'r', encoding='gbk') as f:
        reader = csv.reader(f)
        next(reader)  # headers
        return list(reader)


def _stub_build_row_factory(daily_returns_value: object):
    """Return a mock for `_build_target_month_row` that emits a deterministic row
    plus a sentinel daily_returns at the end (popped by the caller)."""
    def _mock(code, daily, rt, stock_rows, target_year, target_month):
        target_date = f'{target_year}-{target_month:02d}-25'
        row = _make_row(target_date, code, col54=COL54_INITIAL)
        row.append(daily_returns_value)  # sentinel (popped by supplement_csv_incremental)
        return row
    return _mock


@pytest.fixture
def patched_fetch(monkeypatch):
    """Replace network calls with deterministic stubs."""
    def _fetch_daily(stocks, cache_dir, today_str=None, min_history_days=120):
        # Return a non-empty list of dummy daily bars per stock so the eligibility
        # / truthiness checks pass without ever hitting the network.
        return {code: [{'date': '2025-05-01'}] for code in stocks}

    def _fetch_rt(stocks):
        return {code: {} for code in stocks}

    monkeypatch.setattr(gsi, 'fetch_daily_batch_incremental', _fetch_daily)
    monkeypatch.setattr(gsi, 'fetch_realtime_quotes_batch', _fetch_rt)
    return _fetch_daily, _fetch_rt


def test_backfill_finds_prev_month_when_target_month_row_already_exists(
    tmp_path, monkeypatch, patched_fetch
):
    """eb4d369 主修复：当 supplement 是 restart 的第二次（已经写过当月行），
    backfill 仍然必须命中上月行。原 bug 让 [-1] 指向当月行，silent skip。
    """
    csv_path = tmp_path / 'mini.csv'
    rows = [
        # Stock A: 已经有当月行（restart 中间态），同时也有上月行。
        _make_row('2025-04-25', 'A', col54=COL54_INITIAL),
        _make_row('2025-05-25', 'A', col54=COL54_INITIAL),
        # Stock B: 只有上月行（typical case）
        _make_row('2025-04-25', 'B', col54=COL54_INITIAL),
        # Stock C: 只有当月行，没有上月行 → 不应被回填（negative case）
        _make_row('2025-05-25', 'C', col54=COL54_INITIAL),
    ]
    _write_csv(rows, csv_path)

    sentinel = [0.011, 0.022, -0.033]  # mock daily returns for target month
    monkeypatch.setattr(gsi, '_build_target_month_row',
                        _stub_build_row_factory(sentinel))

    written = gsi.supplement_csv_incremental(
        csv_path=str(csv_path),
        target_year=2025,
        target_month=5,
        cache_dir=str(tmp_path / 'cache'),
    )
    assert written == 3  # A/B/C 都新写一行当月数据（A/C 被替换）

    final = _read_csv(csv_path)
    by_key = {(r[1], r[0]): r for r in final}

    # A 上月行 col54 必须被回填为 sentinel；当月行 col54 保留 _build_target_month_row
    # 默认的 '[]'（不应被回填错误覆盖）。
    assert by_key[('A', '2025-04-25')][54] == str(sentinel), \
        f"A 2025-04 row col54 应被回填，实际 {by_key[('A', '2025-04-25')][54]!r}"
    assert by_key[('A', '2025-05-25')][54] == COL54_INITIAL, \
        f"A 2025-05 row col54 不应被回填覆盖，实际 {by_key[('A', '2025-05-25')][54]!r}"

    # B 上月行同样被回填
    assert by_key[('B', '2025-04-25')][54] == str(sentinel)

    # C 没有上月行 → 不应产生任何回填条目，当月行 col54 保持初始
    assert by_key[('C', '2025-05-25')][54] == COL54_INITIAL
    # 也不应出现 C 的 2025-04 行
    assert ('C', '2025-04-25') not in by_key


def test_backfill_skipped_when_code_has_no_prev_month_row(
    tmp_path, monkeypatch, patched_fetch
):
    """Negative case：纯当月行的 stock 不应触发回填，也不应吃掉别人的 sentinel。"""
    csv_path = tmp_path / 'mini.csv'
    rows = [
        _make_row('2025-05-25', 'NEWBIE', col54=COL54_INITIAL),
        _make_row('2025-04-25', 'OLD', col54=COL54_INITIAL),
    ]
    _write_csv(rows, csv_path)
    monkeypatch.setattr(gsi, '_build_target_month_row',
                        _stub_build_row_factory([0.05]))

    gsi.supplement_csv_incremental(
        csv_path=str(csv_path),
        target_year=2025,
        target_month=5,
        cache_dir=str(tmp_path / 'cache'),
    )
    final = _read_csv(csv_path)
    by_key = {(r[1], r[0]): r for r in final}
    assert by_key[('OLD', '2025-04-25')][54] == '[0.05]'
    # NEWBIE 没有上月行；当月新行 col54 保持初始（默认值）。
    assert by_key[('NEWBIE', '2025-05-25')][54] == COL54_INITIAL


def test_plan_ratio_below_threshold_raises(tmp_path, monkeypatch, patched_fetch):
    """保护 2：plan_ratio < 0.90 必须 raise（CLAUDE.md 硬约束）。

    模拟回填规划完全失效（旧 bug 等价场景）：让 _build_target_month_row 返回的
    sentinel daily_returns 永远 falsy，于是 backfill_prev_col54 永远不会被填充，
    但所有 eligible stock 又都满足 expected_backfill_eligible（有上月行），
    导致 plan_ratio = 0 / N → 触发硬约束 raise。
    """
    csv_path = tmp_path / 'mini.csv'
    # 10 个 stock，都有上月行 → expected = 10
    rows = [_make_row('2025-04-25', f'S{i:03d}') for i in range(10)]
    _write_csv(rows, csv_path)
    # sentinel 设置为 falsy（空 list） → 全部不进 backfill_prev_col54
    monkeypatch.setattr(gsi, '_build_target_month_row',
                        _stub_build_row_factory([]))

    with pytest.raises(RuntimeError, match=r'col54 回填规划覆盖率'):
        gsi.supplement_csv_incremental(
            csv_path=str(csv_path),
            target_year=2025,
            target_month=5,
            cache_dir=str(tmp_path / 'cache'),
        )


def test_plan_ratio_full_passes(tmp_path, monkeypatch, patched_fetch, capsys):
    """正常路径：所有 eligible 都被规划回填 → ratio=1.0，应有日志且不 raise。"""
    csv_path = tmp_path / 'mini.csv'
    rows = [_make_row('2025-04-25', f'S{i:03d}') for i in range(5)]
    _write_csv(rows, csv_path)
    monkeypatch.setattr(gsi, '_build_target_month_row',
                        _stub_build_row_factory([0.01]))

    gsi.supplement_csv_incremental(
        csv_path=str(csv_path),
        target_year=2025,
        target_month=5,
        cache_dir=str(tmp_path / 'cache'),
    )
    out = capsys.readouterr()
    # 期望日志含 "col54 backfill plan: 5/5"
    assert 'col54 backfill plan' in out.err
    assert '5/5' in out.err
