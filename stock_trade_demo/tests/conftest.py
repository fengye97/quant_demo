"""Pytest fixtures shared by timing realism tests.

Ensures `stock_trade_demo/` is on sys.path so that absolute imports like
`from timing.backtest import ...` and `from index_data import ...` resolve
the same way they do when running scripts from the project root.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _base_panel(dates, *, etf_open, etf_close, etf_prev_close=None,
                has_real_etf_bar=None, target_exposure=None,
                strength_score=None, close=None,
                etf_code='510980', etf_name='中证1000ETF',
                index_id='csi1000', index_name='中证1000'):
    """Build a fully-resolved panel matching what `_attach_etf_prices` would emit.

    The replay engine expects: 交易日期, etf_open, etf_close, etf_prev_close,
    has_real_etf_bar, target_exposure, position, prev_exposure, exposure_change,
    rebalance_action, signal_action, close, plus the misc index_id/index_name/etc.
    `run_timing_backtest` re-derives prev/exposure_change/position/signal/rebalance
    via `_rebuild_timing_actions`; for replay-only tests we precompute them here.
    """
    n = len(dates)
    if has_real_etf_bar is None:
        has_real_etf_bar = [True] * n
    if target_exposure is None:
        target_exposure = [0.0] * n
    if close is None:
        close = list(etf_close)
    if strength_score is None:
        strength_score = [0.0] * n
    target = np.array(target_exposure, dtype=float)
    prev = np.concatenate([[0.0], target[:-1]])
    change = target - prev
    position = (target > 1e-8).astype(int)
    signal_action = []
    rebalance_action = []
    for p, t in zip(prev, target):
        prev_on = p > 1e-8
        target_on = t > 1e-8
        if not prev_on and target_on:
            signal_action.append('buy')
            rebalance_action.append('enter')
        elif prev_on and not target_on:
            signal_action.append('sell')
            rebalance_action.append('exit')
        elif target_on and t > p + 1e-8:
            signal_action.append('hold')
            rebalance_action.append('add')
        elif target_on and t + 1e-8 < p:
            signal_action.append('hold')
            rebalance_action.append('trim')
        elif target_on:
            signal_action.append('hold')
            rebalance_action.append('hold')
        else:
            signal_action.append('flat')
            rebalance_action.append('flat')
    df = pd.DataFrame({
        '交易日期': pd.to_datetime(dates),
        'close': close,
        'etf_open': etf_open,
        'etf_close': etf_close,
        'etf_prev_close': etf_prev_close if etf_prev_close is not None
                           else [None] + list(etf_close[:-1]),
        'has_real_etf_bar': has_real_etf_bar,
        'target_exposure': target,
        'prev_exposure': prev,
        'exposure_change': change,
        'position': position,
        'signal_action': signal_action,
        'rebalance_action': rebalance_action,
        'strength_score': strength_score,
        'index_id': index_id,
        'index_name': index_name,
        'etf_code': etf_code,
        'etf_name': etf_name,
        'reason_summary': ['' for _ in range(n)],
        'reason_detail': [[] for _ in range(n)],
        'signal_score': [0.0] * n,
    })
    return df


@pytest.fixture
def make_panel():
    """Factory fixture returning a ready-to-replay timing panel."""
    return _base_panel


@pytest.fixture
def trading_dates():
    """Helper to build a sequence of business days starting from a given date."""
    def _make(start, periods):
        return pd.bdate_range(start=start, periods=periods)
    return _make
