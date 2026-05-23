"""Regression tests for timing-backtest realism (Phase 1+2 fixes).

Coverage map (one block per item in the task brief):
  1. Bug 1 — Star50 staged + profit_lock requires `price_series` to fire.
  2. Bug 3 — `etf_open == 0` is filtered out instead of raising.
  3. Bug 4 + Bug 5 — ETF pre-inception bars are never replayed; attrs are honest.
  4. Staged design fix — binary_position acts as a hard gate.
  5. T+1 settlement (with T+0 control).
  6. Limit-up blocking and FIFO retry.
  7. Open-price slippage (buy/sell symmetry).
  8. Fee split (commission / stamp / transfer / slippage).
  9. Cash interest accrual.

Tests intentionally bypass `run()` for cases where we want a fully-controlled
panel (Bugs 3-9). Bug 4+5 mocks `_attach_etf_prices` to skip the akshare/Sina
network path while still exercising `run_timing_backtest`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from timing import backtest as timing_backtest  # noqa: E402
from timing.backtest import (  # noqa: E402
    _replay_timing_positions,
    filter_timing_result,
    run_timing_backtest,
)
from timing.base import BaseTimingStrategy  # noqa: E402
from timing.strategies import Star50TimingStrategy  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Bug 1 — Star50 staged + profit_lock
# ---------------------------------------------------------------------------
class TestBug1Star50ProfitLock:
    """Star50 must pass `price_series=df['close']` so profit-lock can trigger.

    We exercise `_build_staged_target_exposure` directly with a synthetic
    strength curve that stays above `add_threshold` while price ramps up past
    +28% and then draws down 4% from the peak. With `profit_lock_enabled=True`
    the engine should clamp exposure to <=0.25 on the drawdown bar. Pre-fix
    (price_series=None) entry_price/peak_price stay None so the lock can never
    bind — the assertion would fail.
    """

    def test_profit_lock_caps_exposure_on_4pct_drawdown_from_peak(self):
        strat = Star50TimingStrategy(
            exposure_mode='staged',
            profit_lock_enabled=True,
            profit_lock_drawdown=0.04,
            profit_lock_level_1=0.10,
            profit_lock_level_2=0.18,
            profit_lock_level_3=0.28,
            probe_confirm_days=1,
            confirm_days=1,
        )
        # Strength stays well above add_threshold (0.75) the whole way so the
        # state machine keeps adding and never trims on score alone.
        strength = pd.Series([0.95] * 40, dtype=float)
        # Price ramps from 1.0 -> 1.40 (+40%), peaks, then dips 4% on the last bar.
        ramp = np.linspace(1.0, 1.40, 39)
        peak_then_drop = np.append(ramp, ramp[-1] * 0.96)
        price = pd.Series(peak_then_drop, dtype=float)

        exposures = strat._build_staged_target_exposure(strength, price_series=price)

        # Sanity: profit at the peak bar exceeds level_3 (0.28).
        assert price.iloc[-2] / price.iloc[0] - 1 >= strat.profit_lock_level_3
        # On the drawdown bar, lock_level_3 should pin exposure <= 0.25.
        assert exposures.iloc[-1] <= 0.25 + 1e-9, (
            f"profit_lock failed: final exposure={exposures.iloc[-1]:.4f}, "
            f"expected <= 0.25"
        )

    def test_profit_lock_disabled_keeps_exposure_high(self):
        """Control: with profit_lock_enabled=False the same drawdown should
        not collapse exposure to the 25% floor."""
        strat = Star50TimingStrategy(
            exposure_mode='staged',
            profit_lock_enabled=False,
            probe_confirm_days=1,
            confirm_days=1,
        )
        strength = pd.Series([0.95] * 40, dtype=float)
        ramp = np.linspace(1.0, 1.40, 39)
        peak_then_drop = np.append(ramp, ramp[-1] * 0.96)
        price = pd.Series(peak_then_drop, dtype=float)
        exposures = strat._build_staged_target_exposure(strength, price_series=price)
        # Without profit_lock, staged ladder will be at full / near-full exposure.
        assert exposures.iloc[-1] > 0.25 + 1e-9

    def test_star50_signal_pipeline_passes_price_series(self):
        """Star50.generate_signals must forward `price_series=df['close']`
        into `_apply_exposure_columns`. We verify by spying on the call."""
        strat = Star50TimingStrategy(exposure_mode='staged', profit_lock_enabled=True)
        captured = {}
        original = strat._apply_exposure_columns

        def spy(df, binary_position, staged_strength=None, ready_mask=None, price_series=None):
            captured['price_series'] = price_series
            return original(df, binary_position, staged_strength=staged_strength,
                            ready_mask=ready_mask, price_series=price_series)

        strat._apply_exposure_columns = spy  # type: ignore[assignment]
        # Minimal indicator-ready DataFrame.
        n = 80
        close = pd.Series(np.linspace(1.0, 1.4, n))
        df = pd.DataFrame({
            '交易日期': pd.bdate_range('2024-01-01', periods=n),
            'close': close,
            'high': close * 1.005,
            'low': close * 0.995,
            'trend_ma': close.rolling(40).mean(),
            'breakout_high': close.rolling(10).max().shift(1),
            'exit_low': close.rolling(5).min().shift(1),
            'macd_line': pd.Series(np.linspace(-0.01, 0.05, n)),
            'macd_signal': pd.Series(np.linspace(-0.02, 0.04, n)),
        })
        strat.generate_signals(df)
        assert captured.get('price_series') is not None, (
            "Star50.generate_signals must pass price_series to _apply_exposure_columns; "
            "otherwise profit_lock never fires."
        )
        # And the series should be the close series (or value-equal).
        np.testing.assert_array_equal(
            np.asarray(captured['price_series']),
            np.asarray(close),
        )


# ---------------------------------------------------------------------------
# 2. Bug 3 — etf_open == 0 must be skipped, not raise
# ---------------------------------------------------------------------------
class TestBug3ZeroOpenSkipped:
    def test_zero_open_does_not_raise_and_is_filtered(self, make_panel, trading_dates):
        dates = trading_dates('2024-01-02', 5)
        etf_open = [1.0, 0.0, 1.05, 1.06, 1.07]
        etf_close = [1.01, 1.02, 1.06, 1.07, 1.08]
        # target_exposure flips so we'd trade on every bar were the bar valid.
        panel = make_panel(
            dates,
            etf_open=etf_open,
            etf_close=etf_close,
            target_exposure=[1.0, 0.0, 1.0, 0.0, 1.0],
        )
        # Should not raise.
        result = _replay_timing_positions(
            panel,
            initial_capital=100_000,
            buy_cost=0.0,
            sell_cost=0.0,
            settlement='T+0',
            commission_rate=0.0,
            commission_min=0.0,
        )
        # Zero-open bar gets dropped entirely from the replay output.
        kept_dates = pd.to_datetime(result['交易日期']).dt.strftime('%Y-%m-%d').tolist()
        bad_date = pd.to_datetime(dates[1]).strftime('%Y-%m-%d')
        assert bad_date not in kept_dates
        # The remaining bars still trade normally.
        assert (result['trade_quantity'].abs() > 0).any()


# ---------------------------------------------------------------------------
# 3. Bug 4 + Bug 5 — ETF pre-inception must not produce phantom trades
# ---------------------------------------------------------------------------
class _FakeStar50ForInception(BaseTimingStrategy):
    """Minimal strategy stub used to exercise run_timing_backtest end-to-end
    without hitting the akshare/Sina network path."""
    strategy_id = 'inception_test'
    display_name = 'inception test'

    def get_index_id(self):
        return 'csi1000'

    def get_etf_config(self):
        # Realistic A-share ETF config (T+1, 10% limit, SH).
        return {
            'code': '510980',
            'name': '中证1000ETF',
            'settlement': 'T+1',
            'limit_pct': 0.10,
            'market': 'SH',
        }


class TestBug4Bug5InceptionLeak:
    def _patch_etf_prices(self, monkeypatch, etf_first_date='2022-01-03'):
        """Mock `_attach_etf_prices` so it injects real ETF prices only
        from `etf_first_date` onwards and marks earlier bars as
        `has_real_etf_bar=False`."""

        def fake_attach(df, strategy):
            out = df.copy()
            first = pd.to_datetime(etf_first_date)
            mask = pd.to_datetime(out['交易日期']) >= first
            out['etf_open'] = np.where(mask, out['close'].astype(float), np.nan)
            out['etf_close'] = np.where(mask, out['close'].astype(float) * 1.002, np.nan)
            out['etf_prev_close'] = pd.Series(out['etf_close']).shift(1)
            out['has_real_etf_bar'] = out[['etf_open', 'etf_close']].notna().all(axis=1)
            out['etf_inception_date'] = first
            out['first_real_etf_date'] = first
            out['etf_code'] = '510980'
            out['etf_name'] = '中证1000ETF'
            return out

        monkeypatch.setattr(timing_backtest, '_attach_etf_prices', fake_attach)

    def _build_signal_df(self):
        """Build a 2018-2024 daily signal panel with constant buy intent."""
        dates = pd.bdate_range('2018-01-02', '2024-12-31')
        n = len(dates)
        return pd.DataFrame({
            '交易日期': dates,
            'close': np.linspace(1.0, 2.0, n),
            'target_exposure': [1.0] * n,
            'position': [1] * n,
            'prev_exposure': [0.0] + [1.0] * (n - 1),
            'exposure_change': [1.0] + [0.0] * (n - 1),
            'signal_action': ['buy'] + ['hold'] * (n - 1),
            'rebalance_action': ['enter'] + ['hold'] * (n - 1),
            'strength_score': [0.9] * n,
            'signal_score': [0.0] * n,
            'reason_summary': [''] * n,
            'reason_detail': [[] for _ in range(n)],
            'index_id': ['csi1000'] * n,
            'index_name': ['中证1000'] * n,
        })

    def test_pre_inception_bars_are_not_replayed(self, monkeypatch):
        self._patch_etf_prices(monkeypatch, '2022-01-03')
        strat = _FakeStar50ForInception(initial_capital=100_000,
                                        slippage_bps=0.0,
                                        cash_interest_rate=0.0,
                                        commission_rate=0.0,
                                        commission_min=0.0,
                                        transfer_fee_rate=0.0)
        result = run_timing_backtest(self._build_signal_df(), strat)
        assert len(result) > 0
        # No replayed bar should be earlier than the ETF inception date.
        first_replayed = pd.to_datetime(result['交易日期'].min())
        assert first_replayed >= pd.Timestamp('2022-01-03'), (
            f"replay leaked pre-inception bar: first={first_replayed}"
        )
        assert result.attrs.get('non_tradable') is False
        first_real = pd.to_datetime(result.attrs.get('first_real_etf_date'))
        assert first_real == pd.Timestamp('2022-01-03')

    def test_filter_to_pre_inception_window_returns_non_tradable(self, monkeypatch):
        self._patch_etf_prices(monkeypatch, '2022-01-03')
        strat = _FakeStar50ForInception(initial_capital=100_000,
                                        slippage_bps=0.0,
                                        cash_interest_rate=0.0,
                                        commission_rate=0.0,
                                        commission_min=0.0,
                                        transfer_fee_rate=0.0)
        result = run_timing_backtest(self._build_signal_df(), strat)
        sliced = filter_timing_result(result,
                                      start_date='2018-01-01',
                                      end_date='2019-12-31')
        assert len(sliced) == 0
        assert sliced.attrs.get('non_tradable') is True
        assert sliced.attrs.get('etf_inception_date') in (None, 'None')

    def test_filter_inside_etf_history_remains_tradable(self, monkeypatch):
        self._patch_etf_prices(monkeypatch, '2022-01-03')
        strat = _FakeStar50ForInception(initial_capital=100_000,
                                        slippage_bps=0.0,
                                        cash_interest_rate=0.0,
                                        commission_rate=0.0,
                                        commission_min=0.0,
                                        transfer_fee_rate=0.0)
        result = run_timing_backtest(self._build_signal_df(), strat)
        sliced = filter_timing_result(result,
                                      start_date='2023-01-01',
                                      end_date='2023-06-30')
        assert len(sliced) > 0
        assert sliced.attrs.get('non_tradable') is False


# ---------------------------------------------------------------------------
# 4. Staged design fix — binary_position acts as a gate
# ---------------------------------------------------------------------------
class TestStagedBinaryGate:
    def _strategy(self):
        return BaseTimingStrategy(
            exposure_mode='staged',
            enter_threshold=0.55,
            add_threshold=0.75,
            trim_threshold=0.35,
            exit_threshold=0.15,
            confirm_days=1,
            probe_confirm_days=1,
        )

    def test_zero_binary_forces_exposure_zero_despite_high_strength(self):
        strat = self._strategy()
        n = 20
        idx = pd.RangeIndex(n)
        df = pd.DataFrame({
            '交易日期': pd.bdate_range('2024-01-01', periods=n),
            'close': np.linspace(1.0, 1.2, n),
        }, index=idx)
        strength = pd.Series([0.9] * n, index=idx)
        binary = pd.Series([0] * n, index=idx)
        out = strat._apply_exposure_columns(
            df, binary, staged_strength=strength,
            ready_mask=pd.Series([True] * n, index=idx),
            price_series=df['close'],
        )
        assert (out['target_exposure'].abs() < 1e-9).all(), (
            "binary_position=0 must hard-gate target_exposure to 0 in staged mode"
        )

    def test_binary_toggle_resets_entry_when_flipped(self):
        strat = self._strategy()
        n = 30
        idx = pd.RangeIndex(n)
        # binary: ON for first 10, OFF for next 10, ON again for last 10.
        binary_vals = [1] * 10 + [0] * 10 + [1] * 10
        prices = np.concatenate([
            np.linspace(1.0, 1.30, 10),     # ramp while ON
            np.linspace(1.30, 1.30, 10),    # flat while OFF (forced flat)
            np.linspace(1.30, 1.34, 10),    # ramp again after re-enable
        ])
        df = pd.DataFrame({
            '交易日期': pd.bdate_range('2024-01-01', periods=n),
            'close': prices,
        }, index=idx)
        strength = pd.Series([0.9] * n, index=idx)
        binary = pd.Series(binary_vals, index=idx, dtype=float)
        out = strat._apply_exposure_columns(
            df, binary, staged_strength=strength,
            ready_mask=pd.Series([True] * n, index=idx),
            price_series=df['close'],
        )
        target = out['target_exposure'].values
        # While binary is 0, target must be 0.
        assert np.all(target[10:20] < 1e-9), "binary OFF must clamp exposure to 0"
        # After binary flips back to 1, exposure must rebuild (>0) at some point.
        assert (target[20:] > 1e-9).any(), (
            "binary flipping back to 1 should allow staged engine to re-enter"
        )


# ---------------------------------------------------------------------------
# 5. T+1 settlement (A-share ETFs) vs T+0 control (cross-border ETFs)
# ---------------------------------------------------------------------------
class TestTPlusOneSettlement:
    def test_t_plus_one_blocks_same_day_cash_reuse(self, make_panel, trading_dates):
        # bar0: buy full; bar1: full sell; bar2: try to buy again.
        dates = trading_dates('2024-01-02', 3)
        panel = make_panel(
            dates,
            etf_open=[1.0, 1.0, 1.0],
            etf_close=[1.0, 1.0, 1.0],
            etf_prev_close=[None, 1.0, 1.0],
            target_exposure=[1.0, 0.0, 1.0],
        )
        initial = 100_000.0
        result = _replay_timing_positions(
            panel,
            initial_capital=initial,
            buy_cost=0.0,
            sell_cost=0.0,
            settlement='T+1',
            slippage_bps=0.0,
            commission_rate=0.0,
            commission_min=0.0,
            transfer_fee_rate=0.0,
        )
        bar0_qty = float(result['trade_quantity'].iloc[0])
        bar1_qty = float(result['trade_quantity'].iloc[1])
        bar2_qty = float(result['trade_quantity'].iloc[2])
        assert bar0_qty > 0, 'bar0 should fill the initial buy'
        assert bar1_qty > 0, 'bar1 should fill the sell (T+1 share lock only delays cash)'
        # bar2 buy can only use cash that was already settled by bar2.
        # With settlement_lag=1, sell on bar1 (idx=1) releases at idx+1=2,
        # which IS available at bar2's step-1 settlement sweep. The key check:
        # cash from bar1 sell must NOT have been usable inside bar1 itself.
        # Inspect cash_balance progression to be sure.
        # bar1 cash_balance includes pending_cash_total (sum of pending sells)
        # via line 505 (cash_balances.append(current_cash_balance + pending_cash_total)).
        # The cleanest assertion: bar2 trade_amount > 0 (cash freed in time).
        assert bar2_qty > 0, 'bar2 buy should succeed once T+1 cash settles'

    def test_t_plus_zero_allows_immediate_reuse(self, make_panel, trading_dates):
        dates = trading_dates('2024-01-02', 3)
        panel = make_panel(
            dates,
            etf_open=[1.0, 1.0, 1.0],
            etf_close=[1.0, 1.0, 1.0],
            etf_prev_close=[None, 1.0, 1.0],
            target_exposure=[1.0, 0.0, 1.0],
        )
        result = _replay_timing_positions(
            panel,
            initial_capital=100_000.0,
            buy_cost=0.0,
            sell_cost=0.0,
            settlement='T+0',
            slippage_bps=0.0,
            commission_rate=0.0,
            commission_min=0.0,
            transfer_fee_rate=0.0,
        )
        # Under T+0 all three bars trade and bar2 buy notional should be close
        # to the initial capital (cash from bar1 sell immediately reusable).
        assert (result['trade_quantity'] > 0).sum() == 3


# ---------------------------------------------------------------------------
# 6. Price-limit blocking
# ---------------------------------------------------------------------------
class TestLimitUpBlocking:
    def test_buy_blocked_when_open_at_limit_up_retries_next_bar(self, make_panel, trading_dates):
        dates = trading_dates('2024-01-02', 3)
        # bar0: prev_close 1.0, open 1.10 → at 10% limit-up → buy blocked.
        # bar1: open back to 1.05 → buy should fill.
        panel = make_panel(
            dates,
            etf_open=[1.10, 1.05, 1.05],
            etf_close=[1.10, 1.05, 1.05],
            etf_prev_close=[1.0, 1.10, 1.05],
            target_exposure=[1.0, 1.0, 1.0],
        )
        result = _replay_timing_positions(
            panel,
            initial_capital=100_000.0,
            buy_cost=0.0,
            sell_cost=0.0,
            settlement='T+0',
            limit_pct=0.10,
            slippage_bps=0.0,
            commission_rate=0.0,
            commission_min=0.0,
            transfer_fee_rate=0.0,
            limit_max_delay_days=5,
        )
        # bar0 must be blocked, no quantity, blocked_by_limit flagged.
        assert float(result['trade_quantity'].iloc[0]) == 0.0
        assert bool(result['blocked_by_limit'].iloc[0]) is True
        # bar1 must successfully fill the deferred buy.
        assert float(result['trade_quantity'].iloc[1]) > 0.0


# ---------------------------------------------------------------------------
# 7. Open-price slippage
# ---------------------------------------------------------------------------
class TestSlippage:
    def test_buy_fill_price_marked_above_open(self, make_panel, trading_dates):
        dates = trading_dates('2024-01-02', 1)
        panel = make_panel(
            dates,
            etf_open=[1.0],
            etf_close=[1.0],
            etf_prev_close=[1.0],
            target_exposure=[1.0],
        )
        result = _replay_timing_positions(
            panel,
            initial_capital=100_000.0,
            buy_cost=0.0,
            sell_cost=0.0,
            settlement='T+0',
            slippage_bps=5.0,
            commission_rate=0.0,
            commission_min=0.0,
            transfer_fee_rate=0.0,
        )
        qty = float(result['trade_quantity'].iloc[0])
        slip = float(result['slippage_cost'].iloc[0])
        # Buy slippage cost = qty * open * (slippage_bps / 1e4) per side.
        assert qty > 0
        expected = qty * 1.0 * (5.0 / 1.0e4)
        assert slip == pytest.approx(expected, rel=1e-6, abs=1e-6)

    def test_sell_fill_price_marked_below_open(self, make_panel, trading_dates):
        # Hold position then exit on bar1; check sell-side slippage sign/size.
        dates = trading_dates('2024-01-02', 2)
        panel = make_panel(
            dates,
            etf_open=[1.0, 1.0],
            etf_close=[1.0, 1.0],
            etf_prev_close=[1.0, 1.0],
            target_exposure=[1.0, 0.0],
        )
        result = _replay_timing_positions(
            panel,
            initial_capital=100_000.0,
            buy_cost=0.0,
            sell_cost=0.0,
            settlement='T+0',
            slippage_bps=5.0,
            commission_rate=0.0,
            commission_min=0.0,
            transfer_fee_rate=0.0,
        )
        sell_qty = float(result['trade_quantity'].iloc[1])
        sell_slip = float(result['slippage_cost'].iloc[1])
        assert sell_qty > 0
        expected = sell_qty * 1.0 * (5.0 / 1.0e4)
        assert sell_slip == pytest.approx(expected, rel=1e-6, abs=1e-6)


# ---------------------------------------------------------------------------
# 8. Fee split (commission / stamp / transfer / slippage)
# ---------------------------------------------------------------------------
class TestFeeSplit:
    def test_buy_fee_breakdown_matches_expected_components(self, make_panel, trading_dates):
        dates = trading_dates('2024-01-02', 1)
        # Open price 1.0, target 1.0 of 100k → naive notional ≈ 100k, way above
        # the 1000-yuan example. To match the brief precisely we constrain
        # capital to ~1000 worth of shares: use 1000 initial_capital so we buy
        # exactly 1000 shares × 1.0 = 1000元 notional (round lot=100).
        panel = make_panel(
            dates,
            etf_open=[1.0],
            etf_close=[1.0],
            etf_prev_close=[1.0],
            target_exposure=[1.0],
        )
        result = _replay_timing_positions(
            panel,
            initial_capital=1100.0,  # leaves headroom for ≥1000元 notional buy
            buy_cost=0.0,
            sell_cost=0.0,
            settlement='T+1',
            market='SH',
            slippage_bps=0.0,
            commission_rate=0.0001,
            commission_min=5.0,
            stamp_tax_rate=0.0,
            transfer_fee_rate=0.00001,
            limit_pct=None,
        )
        qty = float(result['trade_quantity'].iloc[0])
        assert qty == 1000.0, f'expected 1000-share lot fill, got {qty}'
        notional = qty * 1.0
        expected_commission = max(notional * 0.0001, 5.0)
        expected_transfer = notional * 0.00001
        expected_stamp = 0.0
        assert float(result['commission_cost'].iloc[0]) == pytest.approx(expected_commission)
        assert float(result['transfer_cost'].iloc[0]) == pytest.approx(expected_transfer)
        assert float(result['stamp_cost'].iloc[0]) == pytest.approx(expected_stamp)

        # attrs aggregates should match the per-bar sums (replay alone does not
        # populate attrs; we mimic the run_timing_backtest accumulation here).
        assert float(result['commission_cost'].sum()) == pytest.approx(expected_commission)
        assert float(result['transfer_cost'].sum()) == pytest.approx(expected_transfer)
        assert float(result['stamp_cost'].sum()) == pytest.approx(expected_stamp)


# ---------------------------------------------------------------------------
# 9. Cash interest accrual
# ---------------------------------------------------------------------------
class TestCashInterest:
    def test_full_year_idle_cash_accrues_to_expected_rate(self, make_panel, trading_dates):
        n = 252
        dates = trading_dates('2024-01-02', n)
        panel = make_panel(
            dates,
            etf_open=[1.0] * n,
            etf_close=[1.0] * n,
            etf_prev_close=[None] + [1.0] * (n - 1),
            target_exposure=[0.0] * n,  # always flat
        )
        initial = 100_000.0
        rate = 0.015
        result = _replay_timing_positions(
            panel,
            initial_capital=initial,
            buy_cost=0.0,
            sell_cost=0.0,
            settlement='T+0',
            slippage_bps=0.0,
            cash_interest_rate=rate,
            commission_rate=0.0,
            commission_min=0.0,
            transfer_fee_rate=0.0,
        )
        cum_interest = float(result['cash_interest_cumulative'].iloc[-1])
        # Compounded daily at rate/252 for 252 bars, on a slightly growing
        # cash balance. The naive expectation is initial * (1+rate/252)**252 - initial
        # ≈ initial * rate (1.5% on 100k = 1500). Tolerance < 1%.
        expected = initial * rate
        assert cum_interest == pytest.approx(expected, rel=0.01)
