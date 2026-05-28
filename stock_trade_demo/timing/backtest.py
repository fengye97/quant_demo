import json
import logging

import numpy as np
import pandas as pd

from backtest import compute_alpha_beta
from index_data import INDEX_CONFIGS, build_period_lookup, get_index_return_for_date, get_timing_etf_daily


INTERVAL_WINDOW_MONTHS = {
    'recent_1m': 1,
    'recent_1q': 3,
    'recent_6m': 6,
}

INTERVAL_WINDOW_LABELS = {
    'pre_6m_history': '半年前历史',
    'recent_6m': '近半年',
    'recent_1q': '近一季',
    'recent_1m': '近一月',
}


def _safe_float(value, default=0.0):
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    return float(value)


def _attach_etf_prices(df, strategy):
    out = df.copy()
    etf_cfg = strategy.get_etf_config() if hasattr(strategy, 'get_etf_config') else {}
    etf_code = etf_cfg.get('code')
    etf_name = etf_cfg.get('name')
    etf_price_col = 'etf_close'
    first_real_etf_date = None
    try:
        etf_daily = get_timing_etf_daily(strategy.get_index_id()) if etf_code else pd.DataFrame()
    except Exception:
        etf_daily = pd.DataFrame()
    if len(etf_daily) > 0:
        etf_daily = etf_daily[['date', 'open', 'close']].rename(columns={'open': 'etf_open', 'close': etf_price_col}).copy()
        etf_daily['交易日期'] = pd.to_datetime(etf_daily['date'])
        etf_daily = etf_daily.sort_values('交易日期').reset_index(drop=True)
        first_real_etf_date = pd.to_datetime(etf_daily['交易日期'].min())
        # prev_close：撮合日实际撮合的 ETF 前一交易日真实 close，用于涨跌停板上下限。
        # 注：t 日信号在 t+1 日成交，撮合 bar 是 t+1，撮合 bar 的 prev_close 就是 t 日 ETF 的 close。
        etf_daily['etf_prev_close'] = etf_daily[etf_price_col].shift(1)
        # Signal generated at close(t) executes at next trading day's open/close(t+1)
        etf_daily['signal_date'] = etf_daily['交易日期'].shift(1)
        exec_lookup = (
            etf_daily.dropna(subset=['signal_date'])
            [['signal_date', 'etf_open', etf_price_col, 'etf_prev_close']]
            .rename(columns={'signal_date': '交易日期'})
        )
        out = out.merge(exec_lookup, on='交易日期', how='left')
    if 'etf_open' not in out.columns:
        out['etf_open'] = np.nan
    if etf_price_col not in out.columns:
        out[etf_price_col] = np.nan
    if 'etf_prev_close' not in out.columns:
        out['etf_prev_close'] = np.nan
    out['has_real_etf_bar'] = out[['etf_open', etf_price_col]].notna().all(axis=1)
    out['etf_inception_date'] = first_real_etf_date
    out['first_real_etf_date'] = first_real_etf_date
    out['etf_code'] = etf_code
    out['etf_name'] = etf_name
    return out


def _rebuild_timing_actions(result_df):
    result = result_df.copy().reset_index(drop=True)
    if 'target_exposure' not in result.columns:
        result['target_exposure'] = result.get('position', 0)
    target_exposure = result['target_exposure'].fillna(0.0).astype(float).clip(lower=0.0, upper=1.0)
    prev_exposure = target_exposure.shift(1).fillna(0.0)
    exposure_change = (target_exposure - prev_exposure).fillna(0.0)
    position = (target_exposure > 1e-8).astype(int)

    signal_actions = []
    rebalance_actions = []
    for prev_val, target_val in zip(prev_exposure.tolist(), target_exposure.tolist()):
        prev_on = prev_val > 1e-8
        target_on = target_val > 1e-8
        if not prev_on and target_on:
            signal_actions.append('buy')
            rebalance_actions.append('enter')
        elif prev_on and not target_on:
            signal_actions.append('sell')
            rebalance_actions.append('exit')
        elif target_val > prev_val + 1e-8:
            signal_actions.append('hold')
            rebalance_actions.append('add')
        elif target_val + 1e-8 < prev_val:
            signal_actions.append('hold')
            rebalance_actions.append('trim')
        elif target_on:
            signal_actions.append('hold')
            rebalance_actions.append('hold')
        else:
            signal_actions.append('flat')
            rebalance_actions.append('flat')

    result['target_exposure'] = target_exposure
    result['prev_exposure'] = prev_exposure.astype(float)
    result['exposure_change'] = exposure_change.astype(float)
    result['position'] = position
    result['signal_action'] = signal_actions
    result['rebalance_action'] = rebalance_actions
    return result


def _replay_timing_positions(result_df, initial_capital, buy_cost, sell_cost,
                              settlement='T+1', limit_pct=None, market='SH',
                              slippage_bps=0.0, cash_interest_rate=0.0,
                              commission_rate=0.0, commission_min=0.0,
                              stamp_tax_rate=0.0, transfer_fee_rate=0.0,
                              limit_max_delay_days=5):
    """完整 replay：含 T+1 结算 / 涨跌停 / 滑点 / 现金计息 / 费用拆分。

    binary 与 staged 共用本函数，规则层完全相同，仅 target_exposure 来源不同。
    """
    result = result_df.copy().reset_index(drop=True)
    if len(result) == 0:
        return result
    # 兜底：丢弃任何缺失或非正值的 ETF 开/收盘价（理论上 has_real_etf_bar 过滤已确保不会触发）
    invalid_mask = ~(
        result['etf_open'].notna()
        & (result['etf_open'].astype(float) > 0)
        & result['etf_close'].notna()
        & (result['etf_close'].astype(float) > 0)
    )
    if bool(invalid_mask.any()):
        for _, bad_row in result.loc[invalid_mask].iterrows():
            logging.warning(
                "drop timing replay bar: invalid etf price date=%s etf=%s open=%r close=%r",
                pd.to_datetime(bad_row['交易日期']).strftime('%Y-%m-%d'),
                bad_row.get('etf_code'),
                bad_row.get('etf_open'),
                bad_row.get('etf_close'),
            )
        result = result.loc[~invalid_mask].copy().reset_index(drop=True)
        if len(result) == 0:
            return result

    # === Invariants：数据完整性（drop 完无效行后必须严格满足） ===
    _etf_open_arr = result['etf_open'].astype(float)
    _etf_close_arr = result['etf_close'].astype(float)
    assert (_etf_open_arr > 0).all(), (
        f"ETF 包含 etf_open<=0 的行，无法撮合: "
        f"{result.loc[~(_etf_open_arr > 0), ['交易日期', 'etf_open']].head().to_dict('records')}"
    )
    assert (_etf_close_arr > 0).all(), (
        f"ETF 包含 etf_close<=0 的行: "
        f"{result.loc[~(_etf_close_arr > 0), ['交易日期', 'etf_close']].head().to_dict('records')}"
    )
    _dates_series = pd.to_datetime(result['交易日期'])
    assert _dates_series.is_monotonic_increasing, "ETF 日期未升序"
    first_real_date = _dates_series.iloc[0]

    settlement_mode = str(settlement or 'T+1').upper()
    market_code = str(market or 'SH').upper()
    # 2025 起深交所统一对深市 ETF 收过户费（0.001‰），沪深双市统一
    transfer_active = transfer_fee_rate
    slippage_factor_buy = 1.0 + (slippage_bps / 1.0e4)
    slippage_factor_sell = 1.0 - (slippage_bps / 1.0e4)
    daily_interest = float(cash_interest_rate) / 252.0 if cash_interest_rate else 0.0
    # 结算 lag：T+0 即时可用，T+1 隔交易日。settlement_lag 单位"index 内的下 N 个 bar"。
    settlement_lag = 0 if settlement_mode == 'T+0' else 1

    capitals = []
    pnls = []
    strategy_returns = []
    navs = []
    cum_capitals = []
    holding_units = []
    holding_values = []
    cash_balances = []
    trade_quantities = []
    trade_amounts = []
    trade_fee_amounts = []
    trade_costs = []
    realized_pnls = []
    realized_pnl_pcts = []
    unrealized_pnls = []
    entry_prices = []
    entry_dates = []
    holding_days = []
    entry_capitals = []
    invested_amounts = []
    trade_entry_prices = []
    trade_entry_dates = []
    trade_holding_days = []
    trade_entry_capitals = []
    trade_invested_amounts = []
    trade_cost_basis_solds = []
    position_labels = []
    # 真实规则层新增列
    commission_costs = []
    stamp_costs = []
    transfer_costs = []
    slippage_costs = []
    blocked_by_limit_flags = []
    limit_delays_col = []
    cash_interest_daily = []
    cash_interest_cumulative_col = []

    prev_capital = float(initial_capital)
    current_units = 0.0           # 全部份额（含 pending 锁仓中的卖出未结算份额由 pending_sell_shares 记录）
    available_units = 0.0         # 可卖份额（T+1 下不含当日新买入）
    current_cash_balance = float(initial_capital)  # 可用现金（不含 pending 卖出未结算）
    pending_cash = []             # list of (release_bar_idx, amount)：卖出释放尚未到账的现金
    pending_shares = []           # list of (release_bar_idx, qty)：买入尚未解锁的份额
    current_entry_price = np.nan
    current_entry_date = None
    current_cost_basis = 0.0
    current_entry_capital = 0.0
    cash_interest_cum = 0.0

    # FIFO 顺延 pending 订单：当日 blocked 后塞回队列，下一根 bar 优先重试
    # 每项：dict(side='buy'/'sell', amount_or_qty, intent_value, attempts, original_idx, target_exposure_snapshot)
    pending_orders = []

    for idx, row in result.iterrows():
        trade_price_raw = float(row.get('etf_open'))
        mark_price = float(row.get('etf_close'))
        prev_close_val = row.get('etf_prev_close')
        try:
            prev_close = float(prev_close_val) if pd.notna(prev_close_val) else None
        except (TypeError, ValueError):
            prev_close = None
        current_date = pd.to_datetime(row['交易日期'])

        # === Step 1: 结算到期的 pending cash / shares（队列头出队） ===
        if pending_cash:
            still = []
            for release_idx, amount in pending_cash:
                if release_idx <= idx:
                    current_cash_balance += amount
                else:
                    still.append((release_idx, amount))
            pending_cash = still
        if pending_shares:
            still = []
            for release_idx, qty in pending_shares:
                if release_idx <= idx:
                    available_units += qty
                else:
                    still.append((release_idx, qty))
            pending_shares = still

        # === Step 2: 现金日计息（按当前可用现金） ===
        if daily_interest > 0 and current_cash_balance > 0:
            interest_amt = current_cash_balance * daily_interest
            current_cash_balance += interest_amt
            cash_interest_cum += interest_amt
            day_interest = interest_amt
        else:
            day_interest = 0.0

        # 涨跌停 上下限（基于 ETF 前一交易日真实 close）
        upper_limit = None
        lower_limit = None
        if limit_pct is not None and prev_close is not None and prev_close > 0:
            upper_limit = prev_close * (1.0 + float(limit_pct))
            lower_limit = prev_close * (1.0 - float(limit_pct))

        current_holding_value_before_trade = current_units * trade_price_raw
        row_capital_before_trade = current_cash_balance + current_holding_value_before_trade + sum(a for _, a in pending_cash)
        capitals.append(row_capital_before_trade)

        target_exposure = _safe_float(row.get('target_exposure', row.get('position', 0.0)), 0.0)
        target_exposure = min(max(target_exposure, 0.0), 1.0)
        prev_target_exposure = _safe_float(row.get('prev_exposure', 0.0), 0.0)
        exposure_change = _safe_float(row.get('exposure_change', target_exposure - prev_target_exposure), 0.0)
        new_intent = (idx == 0) or (abs(exposure_change) > 1e-8)

        # === Step 3: 把当日新意图入队 ===
        # 注：当日全口径资本（含 holding mark）作为目标参考，沿用原口径用 row_capital_before_trade
        if new_intent and trade_price_raw > 0:
            desired_holding_value = row_capital_before_trade * target_exposure
            trade_delta = desired_holding_value - current_holding_value_before_trade
            if trade_delta > 1e-8:
                pending_orders.append({
                    'side': 'buy',
                    'desired_value': desired_holding_value,
                    'delta_value': trade_delta,
                    'attempts': 0,
                    'origin_idx': idx,
                    'origin_date': current_date,
                })
            elif trade_delta < -1e-8:
                pending_orders.append({
                    'side': 'sell',
                    'desired_value': desired_holding_value,
                    'delta_value': trade_delta,
                    'attempts': 0,
                    'origin_idx': idx,
                    'origin_date': current_date,
                })

        # === Step 4: 尝试撮合 pending_orders（FIFO） ===
        trade_quantity = 0.0
        trade_amount = 0.0       # 名义成交额（按开盘价 × 数量）
        trade_fee_amount = 0.0   # 全部费用合计（comm+stamp+transfer），不含滑点
        trade_cost_ratio = 0.0
        realized_pnl = np.nan
        realized_pnl_pct = np.nan
        commission_today = 0.0
        stamp_today = 0.0
        transfer_today = 0.0
        slippage_today = 0.0
        blocked_today = False
        delays_today = 0
        position_label = '空仓'
        trade_entry_price = current_entry_price if pd.notna(current_entry_price) else np.nan
        trade_entry_date = current_entry_date
        trade_entry_capital = current_entry_capital if current_entry_capital else np.nan
        trade_invested_amount = current_cost_basis if current_cost_basis else np.nan
        trade_cost_basis_sold = np.nan

        if trade_price_raw > 0 and pending_orders:
            remaining_orders = []
            for order in pending_orders:
                side = order['side']
                if side == 'buy':
                    blocked = (upper_limit is not None) and (trade_price_raw >= upper_limit - 1e-9)
                    if blocked:
                        order['attempts'] += 1
                        if limit_max_delay_days > 0 and order['attempts'] <= limit_max_delay_days:
                            remaining_orders.append(order)
                            blocked_today = True
                            delays_today = max(delays_today, order['attempts'])
                        else:
                            logging.warning(
                                "timing buy order dropped after %d limit-up blocks: origin_date=%s",
                                order['attempts'],
                                pd.to_datetime(order['origin_date']).strftime('%Y-%m-%d'),
                            )
                        continue
                    # 重新基于当前实际持仓估算成交目标
                    desired_value = order['desired_value']
                    holding_value_now = current_units * trade_price_raw
                    delta_value = desired_value - holding_value_now
                    if delta_value <= 1e-8 or current_cash_balance <= 1e-8:
                        continue
                    fill_price = trade_price_raw * slippage_factor_buy
                    # 留费用余地：先按 commission_rate + transfer_rate 之和给出上限
                    fee_pad = float(commission_rate) + float(transfer_active)
                    max_buy_amount = min(delta_value, current_cash_balance / (1 + fee_pad + 1e-6))
                    raw_quantity = max_buy_amount / fill_price if fill_price else 0.0
                    lot_quantity = int(raw_quantity / 100) * 100
                    if lot_quantity <= 0:
                        continue
                    qty = float(lot_quantity)
                    notional = qty * fill_price  # 实际成交金额（含滑点）
                    open_notional = qty * trade_price_raw
                    commission = max(notional * float(commission_rate), float(commission_min)) if commission_rate or commission_min else 0.0
                    transfer = notional * float(transfer_active)
                    stamp = 0.0  # 买入不收印花税
                    fee_total = commission + transfer + stamp
                    slippage_cost = notional - open_notional  # 正值

                    if current_cash_balance < notional + fee_total:
                        # 现金不足，回退一档（按 100 股最小单位）
                        affordable = max(current_cash_balance - fee_total, 0.0)
                        qty2 = int((affordable / fill_price) / 100) * 100 if fill_price > 0 else 0
                        if qty2 <= 0:
                            continue
                        qty = float(qty2)
                        notional = qty * fill_price
                        open_notional = qty * trade_price_raw
                        commission = max(notional * float(commission_rate), float(commission_min)) if commission_rate or commission_min else 0.0
                        transfer = notional * float(transfer_active)
                        fee_total = commission + transfer
                        slippage_cost = notional - open_notional

                    # Invariant：trade 价格必须为正，日期不得早于 ETF 首日
                    assert fill_price > 0, (
                        f"buy trade 价格非正: date={current_date} fill_price={fill_price}"
                    )
                    assert current_date >= first_real_date, (
                        f"inception leak: buy trade date={current_date} 早于 replay window 首日 {first_real_date}"
                    )
                    current_units += qty
                    # T+1：买入份额到下个 bar 才能卖
                    if settlement_lag <= 0:
                        available_units += qty
                    else:
                        pending_shares.append((idx + settlement_lag, qty))
                    current_cash_balance = max(current_cash_balance - notional - fee_total, 0.0)
                    current_cost_basis += notional  # 含滑点的"真实买入成本"
                    current_entry_capital += notional
                    if current_units > 1e-8:
                        current_entry_price = current_cost_basis / current_units
                    if current_entry_date is None:
                        current_entry_date = current_date

                    trade_quantity += qty
                    trade_amount += open_notional  # 用 etf_open 计的名义额，保持下游解读一致
                    trade_fee_amount += fee_total
                    commission_today += commission
                    stamp_today += stamp
                    transfer_today += transfer
                    slippage_today += slippage_cost
                    trade_entry_price = current_entry_price if pd.notna(current_entry_price) else np.nan
                    trade_entry_date = current_entry_date
                    trade_entry_capital = current_entry_capital if current_entry_capital else np.nan
                    trade_invested_amount = current_cost_basis if current_cost_basis else np.nan
                elif side == 'sell':
                    blocked = (lower_limit is not None) and (trade_price_raw <= lower_limit + 1e-9)
                    if blocked:
                        order['attempts'] += 1
                        if limit_max_delay_days > 0 and order['attempts'] <= limit_max_delay_days:
                            remaining_orders.append(order)
                            blocked_today = True
                            delays_today = max(delays_today, order['attempts'])
                        else:
                            logging.warning(
                                "timing sell order dropped after %d limit-down blocks: origin_date=%s",
                                order['attempts'],
                                pd.to_datetime(order['origin_date']).strftime('%Y-%m-%d'),
                            )
                        continue
                    if available_units <= 1e-8:
                        # T+1 下当日新买无法卖；保留意图等下根 bar
                        order['attempts'] += 1
                        if order['attempts'] <= max(limit_max_delay_days, 1):
                            remaining_orders.append(order)
                        continue
                    desired_value = order['desired_value']
                    holding_value_now = current_units * trade_price_raw
                    delta_value = desired_value - holding_value_now  # 应为负
                    if delta_value >= -1e-8:
                        continue
                    sell_value = min(abs(delta_value), holding_value_now)
                    fill_price = trade_price_raw * slippage_factor_sell
                    raw_qty = sell_value / trade_price_raw if trade_price_raw > 0 else 0.0
                    # 不强制 100 股向下取整：staged 减仓有可能减 25%，保留小数支持；
                    # 若希望严格整手卖，可改 int(raw_qty/100)*100。
                    qty = min(raw_qty, available_units)
                    if qty <= 1e-8:
                        continue
                    open_notional = qty * trade_price_raw   # 名义额
                    notional = qty * fill_price             # 实际成交（含滑点）
                    commission = max(notional * float(commission_rate), float(commission_min)) if commission_rate or commission_min else 0.0
                    transfer = notional * float(transfer_active)
                    stamp = notional * float(stamp_tax_rate)
                    fee_total = commission + transfer + stamp
                    slippage_cost = open_notional - notional  # 滑点造成的损失（卖出方向）

                    avg_cost = current_cost_basis / current_units if current_units > 1e-8 else 0.0
                    cost_basis_sold = avg_cost * qty
                    proceeds = notional - fee_total
                    realized_today = proceeds - cost_basis_sold

                    # Invariant：trade 价格必须为正，日期不得早于 ETF 首日
                    assert fill_price > 0, (
                        f"sell trade 价格非正: date={current_date} fill_price={fill_price}"
                    )
                    assert current_date >= first_real_date, (
                        f"inception leak: sell trade date={current_date} 早于 replay window 首日 {first_real_date}"
                    )
                    current_units = max(current_units - qty, 0.0)
                    available_units = max(available_units - qty, 0.0)
                    # 卖出现金 T+1 才可用
                    if settlement_lag <= 0:
                        current_cash_balance += max(proceeds, 0.0)
                    else:
                        pending_cash.append((idx + settlement_lag, max(proceeds, 0.0)))
                    current_cost_basis = max(current_cost_basis - cost_basis_sold, 0.0)
                    current_entry_capital = max(current_entry_capital - cost_basis_sold, 0.0)
                    if current_units <= 1e-8:
                        current_units = 0.0
                        current_cost_basis = 0.0
                        current_entry_capital = 0.0
                        current_entry_price = np.nan
                        current_entry_date = None
                    else:
                        current_entry_price = current_cost_basis / current_units

                    trade_quantity += qty
                    trade_amount += open_notional
                    trade_fee_amount += fee_total
                    commission_today += commission
                    stamp_today += stamp
                    transfer_today += transfer
                    slippage_today += slippage_cost
                    trade_entry_price = avg_cost if avg_cost > 0 else np.nan
                    trade_cost_basis_sold = cost_basis_sold if cost_basis_sold > 0 else np.nan
                    # 累计已实现盈亏（同日多笔卖，realized_pnl 取合计）
                    realized_pnl = (0.0 if pd.isna(realized_pnl) else realized_pnl) + realized_today
                    if cost_basis_sold:
                        realized_pnl_pct = realized_pnl / current_entry_capital if current_entry_capital else (realized_pnl / cost_basis_sold)
            pending_orders = remaining_orders

        trade_cost_ratio = (trade_fee_amount / prev_capital) if prev_capital else 0.0

        # === Step 5: 收盘按 etf_close mark ===
        holding_value = current_units * mark_price
        pending_cash_total = sum(a for _, a in pending_cash)
        capital_after_trade = current_cash_balance + holding_value + pending_cash_total
        pnl = capital_after_trade - prev_capital
        strategy_return = (pnl / prev_capital) if prev_capital else 0.0
        prev_capital = capital_after_trade

        if current_units > 1e-8:
            position_label = '持仓中'
            unrealized_pnl = holding_value - current_cost_basis
            if current_entry_date is not None:
                holding_day_count = max((current_date - current_entry_date).days, 0)
            else:
                holding_day_count = 0
        else:
            unrealized_pnl = np.nan
            holding_day_count = 0
            if trade_quantity > 0 and exposure_change < 0:
                position_label = '已卖出'

        trade_holding_day_count = 0
        if trade_entry_date is not None:
            trade_holding_day_count = max((current_date - trade_entry_date).days, 0)

        pnls.append(pnl)
        strategy_returns.append(strategy_return)
        navs.append(capital_after_trade / float(initial_capital) if initial_capital else 1.0)
        cum_capitals.append(capital_after_trade)
        holding_units.append(round(current_units, 10))
        holding_values.append(holding_value)
        cash_balances.append(current_cash_balance + pending_cash_total)
        trade_quantities.append(trade_quantity)
        trade_amounts.append(trade_amount)
        trade_fee_amounts.append(trade_fee_amount)
        trade_costs.append(trade_cost_ratio)
        realized_pnls.append(realized_pnl)
        realized_pnl_pcts.append(realized_pnl_pct)
        unrealized_pnls.append(unrealized_pnl)
        entry_prices.append(current_entry_price if pd.notna(current_entry_price) else np.nan)
        entry_dates.append(current_entry_date.strftime('%Y-%m-%d') if current_entry_date is not None else None)
        holding_days.append(holding_day_count)
        entry_capitals.append(current_entry_capital if current_entry_capital else np.nan)
        invested_amounts.append(current_cost_basis if current_cost_basis else np.nan)
        trade_entry_prices.append(trade_entry_price if pd.notna(trade_entry_price) else np.nan)
        trade_entry_dates.append(trade_entry_date.strftime('%Y-%m-%d') if trade_entry_date is not None else None)
        trade_holding_days.append(trade_holding_day_count)
        trade_entry_capitals.append(trade_entry_capital if pd.notna(trade_entry_capital) else np.nan)
        trade_invested_amounts.append(trade_invested_amount if pd.notna(trade_invested_amount) else np.nan)
        trade_cost_basis_solds.append(trade_cost_basis_sold if pd.notna(trade_cost_basis_sold) else np.nan)
        position_labels.append(position_label)
        commission_costs.append(commission_today)
        stamp_costs.append(stamp_today)
        transfer_costs.append(transfer_today)
        slippage_costs.append(slippage_today)
        blocked_by_limit_flags.append(bool(blocked_today))
        limit_delays_col.append(int(delays_today))
        cash_interest_daily.append(day_interest)
        cash_interest_cumulative_col.append(cash_interest_cum)

    result['当期本金'] = capitals
    result['当期盈亏'] = pnls
    result['strategy_return'] = strategy_returns
    result['累积净值'] = navs
    result['累计资金'] = cum_capitals
    result['holding_units'] = holding_units
    result['holding_value'] = holding_values
    result['cash_balance'] = cash_balances
    result['trade_quantity'] = trade_quantities
    result['trade_amount'] = trade_amounts
    result['trade_fee_amount'] = trade_fee_amounts
    result['trade_cost'] = trade_costs
    result['realized_pnl'] = realized_pnls
    result['realized_pnl_pct'] = realized_pnl_pcts
    result['unrealized_pnl'] = unrealized_pnls
    result['entry_price'] = entry_prices
    result['entry_date'] = entry_dates
    result['holding_days'] = holding_days
    result['entry_capital'] = entry_capitals
    result['invested_amount'] = invested_amounts
    result['trade_entry_price'] = trade_entry_prices
    result['trade_entry_date'] = trade_entry_dates
    result['trade_holding_days'] = trade_holding_days
    result['trade_entry_capital'] = trade_entry_capitals
    result['trade_invested_amount'] = trade_invested_amounts
    result['trade_cost_basis_sold'] = trade_cost_basis_solds
    result['position_label'] = position_labels
    # 新增真实规则相关列
    result['commission_cost'] = commission_costs
    result['stamp_cost'] = stamp_costs
    result['transfer_cost'] = transfer_costs
    result['slippage_cost'] = slippage_costs
    result['blocked_by_limit'] = blocked_by_limit_flags
    result['limit_delays'] = limit_delays_col
    result['cash_interest'] = cash_interest_daily
    result['cash_interest_cumulative'] = cash_interest_cumulative_col
    return result


def run_timing_backtest(signal_df, strategy, benchmark_returns=None):
    df = signal_df.copy().sort_values('交易日期').reset_index(drop=True)
    df = _rebuild_timing_actions(df)
    df['index_return'] = df['close'].pct_change().fillna(0.0)
    df = _attach_etf_prices(df, strategy)

    # Bug 5: first_real_etf_date 必须基于真实 ETF bar 派生，不能落到 panel 全量起点
    if 'has_real_etf_bar' in df.columns and bool(df['has_real_etf_bar'].any()):
        real_min = df.loc[df['has_real_etf_bar'], '交易日期'].min()
        first_real_etf_date = pd.to_datetime(real_min) if pd.notna(real_min) else None
        df = df[df['has_real_etf_bar']].copy().reset_index(drop=True)
    else:
        first_real_etf_date = None
        df = df.iloc[0:0].copy()

    df = _rebuild_timing_actions(df)
    etf_cfg = strategy.get_etf_config() if hasattr(strategy, 'get_etf_config') else {}
    settlement_mode = (etf_cfg.get('settlement') or 'T+1').upper()
    limit_pct = etf_cfg.get('limit_pct', None)
    market_code = (etf_cfg.get('market') or 'SH').upper()
    df = _replay_timing_positions(
        df,
        initial_capital=strategy.initial_capital,
        buy_cost=strategy.buy_cost,
        sell_cost=strategy.sell_cost,
        settlement=settlement_mode,
        limit_pct=limit_pct,
        market=market_code,
        slippage_bps=getattr(strategy, 'slippage_bps', 0.0),
        cash_interest_rate=getattr(strategy, 'cash_interest_rate', 0.0),
        commission_rate=getattr(strategy, 'commission_rate', 0.0),
        commission_min=getattr(strategy, 'commission_min', 0.0),
        stamp_tax_rate=getattr(strategy, 'stamp_tax_rate', 0.0),
        transfer_fee_rate=getattr(strategy, 'transfer_fee_rate', 0.0),
        limit_max_delay_days=getattr(strategy, 'limit_max_delay_days', 5),
    )
    df.attrs['initial_capital'] = strategy.initial_capital
    df.attrs['buy_cost'] = strategy.buy_cost
    df.attrs['sell_cost'] = strategy.sell_cost
    df.attrs['benchmark_returns'] = benchmark_returns
    df.attrs['etf_code'] = df['etf_code'].iloc[-1] if len(df) else etf_cfg.get('code')
    df.attrs['etf_name'] = df['etf_name'].iloc[-1] if len(df) else etf_cfg.get('name')
    df.attrs['exposure_mode'] = getattr(strategy, 'exposure_mode', 'binary')
    # 真实规则层元数据 + 汇总
    df.attrs['settlement_mode'] = settlement_mode
    df.attrs['limit_pct'] = limit_pct
    df.attrs['market'] = market_code
    df.attrs['slippage_bps'] = getattr(strategy, 'slippage_bps', 0.0)
    df.attrs['cash_interest_rate'] = getattr(strategy, 'cash_interest_rate', 0.0)
    df.attrs['commission_rate'] = getattr(strategy, 'commission_rate', 0.0)
    df.attrs['commission_min'] = getattr(strategy, 'commission_min', 0.0)
    df.attrs['stamp_tax_rate'] = getattr(strategy, 'stamp_tax_rate', 0.0)
    df.attrs['transfer_fee_rate'] = getattr(strategy, 'transfer_fee_rate', 0.0)
    df.attrs['limit_max_delay_days'] = getattr(strategy, 'limit_max_delay_days', 5)
    df.attrs['profit_lock_enabled'] = bool(getattr(strategy, 'profit_lock_enabled', False))
    df.attrs['profit_lock_drawdown'] = float(getattr(strategy, 'profit_lock_drawdown', 0.0))
    df.attrs['profit_lock_level_1'] = float(getattr(strategy, 'profit_lock_level_1', 0.0))
    df.attrs['profit_lock_level_2'] = float(getattr(strategy, 'profit_lock_level_2', 0.0))
    df.attrs['profit_lock_level_3'] = float(getattr(strategy, 'profit_lock_level_3', 0.0))
    if len(df):
        df.attrs['commission_total'] = float(df.get('commission_cost', pd.Series(dtype=float)).sum())
        df.attrs['stamp_total'] = float(df.get('stamp_cost', pd.Series(dtype=float)).sum())
        df.attrs['transfer_total'] = float(df.get('transfer_cost', pd.Series(dtype=float)).sum())
        df.attrs['slippage_total'] = float(df.get('slippage_cost', pd.Series(dtype=float)).sum())
        df.attrs['cash_interest_total'] = float(df.get('cash_interest', pd.Series(dtype=float)).sum())
        df.attrs['limit_block_count'] = int(df.get('blocked_by_limit', pd.Series(dtype=bool)).sum())
    else:
        df.attrs['commission_total'] = 0.0
        df.attrs['stamp_total'] = 0.0
        df.attrs['transfer_total'] = 0.0
        df.attrs['slippage_total'] = 0.0
        df.attrs['cash_interest_total'] = 0.0
        df.attrs['limit_block_count'] = 0
    if first_real_etf_date is not None and not pd.isna(first_real_etf_date):
        df.attrs['etf_inception_date'] = first_real_etf_date.strftime('%Y-%m-%d')
        df.attrs['non_tradable'] = False
    else:
        df.attrs['etf_inception_date'] = None
        df.attrs['non_tradable'] = True
    df.attrs['first_real_etf_date'] = df.attrs['etf_inception_date']
    return df


def _build_period_local_nav(result_df):
    """窗口本地 NAV：从窗口首日 close 起算，cumprod 不含 row[0] 的当日收益。

    背景（2026-05-28 修复）：row[0] 的 strategy_return 实际上是「窗口前一交易日 close →
    窗口首日 close」的当日变化（mark-to-market 口径），这一天不在窗口区间内。
    而同窗口的 ETF baseline (_build_etf_summary) 用 etf_close.iloc[0] 作起点，是窗口
    内首日 close。两者起点错位会导致："策略一直 100% 满仓持有时，窗口超额本应是 0
    (仅手续费)，但 UI 会把 row[0] 那天的 ETF 涨幅虚假地记成超额"。

    修复：把 row[0] 的 return 置 0 再 cumprod，让 strategy NAV 与 ETF baseline 都
    严格从窗口首日 close 起算。这与「reset_capital」语义一致——窗口起点资金重置后，
    才开始累计盈亏。
    """
    result = result_df.copy().reset_index(drop=True)
    if len(result) == 0:
        return pd.Series(dtype=float)
    returns = result['strategy_return'].fillna(0.0).astype(float).copy()
    returns.iloc[0] = 0.0
    return (1.0 + returns).cumprod()


def _build_etf_benchmark_returns(result_df):
    result = result_df.copy().reset_index(drop=True)
    if len(result) == 0 or 'etf_close' not in result.columns:
        return None
    etf = result[['交易日期', 'etf_close']].copy()
    etf['交易日期'] = pd.to_datetime(etf['交易日期'])
    etf['etf_close'] = pd.to_numeric(etf['etf_close'], errors='coerce')
    etf = etf.dropna(subset=['交易日期', 'etf_close'])
    etf = etf[etf['etf_close'] > 0].drop_duplicates(subset=['交易日期']).sort_values('交易日期')
    if len(etf) < 2:
        return None
    daily_returns = etf.set_index('交易日期')['etf_close'].pct_change().dropna()
    if len(daily_returns) == 0:
        return None
    monthly_returns = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1).dropna()
    return monthly_returns if len(monthly_returns) else None


def _build_etf_summary(result_df):
    result = result_df.copy().reset_index(drop=True)
    if len(result) == 0 or 'etf_close' not in result.columns:
        return {}
    etf = result[['交易日期', 'etf_close']].copy()
    etf['交易日期'] = pd.to_datetime(etf['交易日期'])
    etf['etf_close'] = pd.to_numeric(etf['etf_close'], errors='coerce')
    etf = etf.dropna(subset=['交易日期', 'etf_close'])
    etf = etf[etf['etf_close'] > 0].drop_duplicates(subset=['交易日期']).sort_values('交易日期')
    if len(etf) == 0:
        return {}

    start_price = float(etf['etf_close'].iloc[0])
    end_price = float(etf['etf_close'].iloc[-1])
    etf_return_pct = ((end_price / start_price) - 1.0) * 100.0 if start_price > 0 else None
    strategy_nav = _build_period_local_nav(result)
    strategy_return_pct = (float(strategy_nav.iloc[-1]) - 1.0) * 100.0 if len(strategy_nav) else None
    strategy_excess_pct = None
    if strategy_return_pct is not None and etf_return_pct is not None:
        strategy_excess_pct = strategy_return_pct - etf_return_pct

    latest = result.iloc[-1]
    return {
        'etf_code': latest.get('etf_code') or result.attrs.get('etf_code'),
        'etf_name': latest.get('etf_name') or result.attrs.get('etf_name'),
        'start_price': round(start_price, 4),
        'end_price': round(end_price, 4),
        'return_pct': round(etf_return_pct, 2) if etf_return_pct is not None else None,
        'strategy_return_pct': round(strategy_return_pct, 2) if strategy_return_pct is not None else None,
        'strategy_excess_pct': round(strategy_excess_pct, 2) if strategy_excess_pct is not None else None,
    }


def evaluate_timing_result(result_df, benchmark_returns=None, reset_capital=False):
    result = result_df.copy().reset_index(drop=True)
    metrics = {}
    if len(result) == 0:
        return metrics

    initial_capital = float(result.attrs.get('initial_capital', 50000))
    nav_series = _build_period_local_nav(result) if reset_capital else result['累积净值'].astype(float)
    final_nav = float(nav_series.iloc[-1])
    final_capital = initial_capital * final_nav

    metrics['累积净值'] = round(final_nav, 4)
    date_delta = pd.to_datetime(result['交易日期'].iloc[-1]) - pd.to_datetime(result['交易日期'].iloc[0])
    days = max(getattr(date_delta, 'days', 0), 1)
    annual_return = final_nav ** (365.0 / days) - 1 if days > 0 else 0.0
    metrics['年化收益'] = f"{round(annual_return * 100, 2)}%"

    peak = nav_series.cummax()
    dd = nav_series / peak - 1
    max_drawdown = float(dd.min()) if len(dd) else 0.0
    end_idx = int(dd.idxmin()) if len(dd) else 0
    end_date = pd.to_datetime(result.iloc[end_idx]['交易日期'])
    start_subset = result[result['交易日期'] <= end_date].copy()
    if len(start_subset):
        start_nav_subset = nav_series.loc[start_subset.index]
        start_idx = int(start_nav_subset.idxmax())
    else:
        start_idx = 0
    start_date = pd.to_datetime(result.loc[start_idx, '交易日期']) if len(result) else end_date

    metrics['最大回撤'] = format(max_drawdown, '.2%')
    metrics['最大回撤开始'] = start_date.strftime('%Y-%m-%d')
    metrics['最大回撤结束'] = end_date.strftime('%Y-%m-%d')
    metrics['年化收益/回撤比'] = round(annual_return / abs(max_drawdown), 2) if max_drawdown != 0 else 0
    metrics['最终资金'] = round(final_capital, 2)
    metrics['总收益率'] = f"{round((final_capital / initial_capital - 1) * 100, 2)}%"
    metrics['总盈亏'] = round(final_capital - initial_capital, 2)
    metrics['平均仓位'] = round(float(result.get('target_exposure', pd.Series([result['position'].mean()])).mean()), 4)
    metrics['调仓次数'] = int((result.get('trade_quantity', pd.Series(dtype=float)).abs() > 1e-8).sum()) if 'trade_quantity' in result.columns else 0
    metrics['手续费占比'] = round(float(result.get('trade_cost', pd.Series(dtype=float)).sum()), 6) if 'trade_cost' in result.columns else 0.0
    metrics['资金重置口径'] = bool(reset_capital)

    if benchmark_returns is not None:
        strategy_rets = result.set_index('交易日期')['strategy_return'].resample('M').apply(lambda x: (1 + x).prod() - 1)
        attr = compute_alpha_beta(strategy_rets, benchmark_returns)
        if 'error' not in attr:
            metrics['Beta'] = attr['beta']
            metrics['月度Alpha'] = f"{round(attr['alpha_monthly'] * 100, 4)}%"
            metrics['年化Alpha'] = f"{round(attr['alpha_annualized'] * 100, 2)}%"
            metrics['信息比率'] = attr['information_ratio']
            metrics['R-squared'] = attr['r_squared']
            metrics['上行捕获率'] = f"{round(attr['up_capture'] * 100, 1)}%" if attr['up_capture'] is not None else 'N/A'
            metrics['下行捕获率'] = f"{round(attr['down_capture'] * 100, 1)}%" if attr['down_capture'] is not None else 'N/A'
    return metrics


def _cold_start_window_replay(sliced_df, source_attrs):
    """窗口起点冷启动重算：cash=initial_capital、units=0、NAV=1.0。

    保留 sliced_df 的 target_exposure 信号路径（由全历史 warmup 算出），
    但把 holding_value/cash_balance/equity_curve/drawdown/累积净值/各类 pnl
    全部按"窗口内独立持仓"重新模拟。CLAUDE.md Rule 13 的最终落地形态。
    """
    if len(sliced_df) == 0:
        return sliced_df
    initial_capital = float(source_attrs.get('initial_capital', 50000))
    # 先重算 prev_exposure / exposure_change / signal_action: 窗口起点 prev=0
    # 否则 _replay_timing_positions 会以为 "之前就持有 X%" 而跳过首单。
    sliced_df = _rebuild_timing_actions(sliced_df)
    replayed = _replay_timing_positions(
        sliced_df,
        initial_capital=initial_capital,
        buy_cost=source_attrs.get('buy_cost', 0.001),
        sell_cost=source_attrs.get('sell_cost', 0.001),
        settlement=source_attrs.get('settlement_mode', 'T+1'),
        limit_pct=source_attrs.get('limit_pct', None),
        market=source_attrs.get('market', 'SH'),
        slippage_bps=source_attrs.get('slippage_bps', 0.0),
        cash_interest_rate=source_attrs.get('cash_interest_rate', 0.0),
        commission_rate=source_attrs.get('commission_rate', 0.0),
        commission_min=source_attrs.get('commission_min', 0.0),
        stamp_tax_rate=source_attrs.get('stamp_tax_rate', 0.0),
        transfer_fee_rate=source_attrs.get('transfer_fee_rate', 0.0),
        limit_max_delay_days=source_attrs.get('limit_max_delay_days', 5),
    )
    replayed.attrs['cold_start_window'] = True
    replayed.attrs['cold_start_initial_capital'] = initial_capital
    return replayed


def filter_timing_result(result_df, start_date=None, end_date=None):
    result = result_df.copy()
    if start_date:
        result = result[result['交易日期'] >= pd.to_datetime(start_date)].copy()
    if end_date:
        result = result[result['交易日期'] <= pd.to_datetime(end_date)].copy()

    # Bug 4: attrs 必须基于过滤后区间重新派生 etf_inception_date / non_tradable，
    # 否则会把全量回测起点泄漏到子区间。
    result = result.reset_index(drop=True)
    result.attrs.update(result_df.attrs)

    # 窗口 cold-start：仅当指定 start_date 且切片严格小于源 df 时触发。
    # （没指定 start_date 时认为是全历史展示，不重算）
    if start_date is not None and len(result) > 0 and len(result) < len(result_df):
        source_attrs = dict(result_df.attrs)
        replayed = _cold_start_window_replay(result, source_attrs)
        if len(replayed) > 0:
            preserved_attrs = dict(result.attrs)
            result = replayed
            for k, v in preserved_attrs.items():
                result.attrs.setdefault(k, v)

    if len(result) == 0:
        result.attrs['etf_inception_date'] = None
        result.attrs['first_real_etf_date'] = None
        result.attrs['non_tradable'] = True
        return result

    if 'has_real_etf_bar' in result.columns and bool(result['has_real_etf_bar'].any()):
        sub_min = result.loc[result['has_real_etf_bar'], '交易日期'].min()
        if pd.notna(sub_min):
            inception_str = pd.to_datetime(sub_min).strftime('%Y-%m-%d')
            result.attrs['etf_inception_date'] = inception_str
            result.attrs['first_real_etf_date'] = inception_str
            result.attrs['non_tradable'] = False
        else:
            result.attrs['etf_inception_date'] = None
            result.attrs['first_real_etf_date'] = None
            result.attrs['non_tradable'] = True
    else:
        result.attrs['etf_inception_date'] = None
        result.attrs['first_real_etf_date'] = None
        result.attrs['non_tradable'] = True

    # 区间汇总：覆盖从全量 attrs 拷过来的 commission_total 等，使其反映子区间。
    if 'commission_cost' in result.columns:
        result.attrs['commission_total'] = float(result['commission_cost'].sum())
    if 'stamp_cost' in result.columns:
        result.attrs['stamp_total'] = float(result['stamp_cost'].sum())
    if 'transfer_cost' in result.columns:
        result.attrs['transfer_total'] = float(result['transfer_cost'].sum())
    if 'slippage_cost' in result.columns:
        result.attrs['slippage_total'] = float(result['slippage_cost'].sum())
    if 'cash_interest' in result.columns:
        result.attrs['cash_interest_total'] = float(result['cash_interest'].sum())
    if 'blocked_by_limit' in result.columns:
        result.attrs['limit_block_count'] = int(result['blocked_by_limit'].sum())

    return result


def _month_start_from_end(end_date, months):
    end_ts = pd.to_datetime(end_date)
    start_ts = (end_ts - pd.DateOffset(months=months)) + pd.Timedelta(days=1)
    return start_ts.normalize()


def summarize_timing_windows(result_df, benchmark_returns=None, full_history_start=None):
    """
    full_history_start: 真实的策略预热起点（即 filter_timing_result 之前 result
    的最早交易日）。当用户在前端选了一个区间时，result_df 已经被截断到该区间，
    但策略本身是在更早的全历史上跑出来的——training_range 必须反映这段真实预热
    历史，否则前端会显示"无更早历史"，与 CLAUDE.md Rule 13 的语义冲突。
    """
    if len(result_df) == 0:
        return {}

    result = result_df.copy().reset_index(drop=True)
    sliced_full_start = pd.to_datetime(result['交易日期'].min())
    if full_history_start is not None:
        full_start = pd.to_datetime(full_history_start)
        if full_start > sliced_full_start:
            full_start = sliced_full_start
    else:
        full_start = sliced_full_start
    full_end = pd.to_datetime(result['交易日期'].max())
    recent_6m_start = _month_start_from_end(full_end, INTERVAL_WINDOW_MONTHS['recent_6m'])

    windows = {
        'pre_6m_history': (full_start, recent_6m_start - pd.Timedelta(days=1)),
        'recent_6m': (recent_6m_start, full_end),
        'recent_1q': (_month_start_from_end(full_end, INTERVAL_WINDOW_MONTHS['recent_1q']), full_end),
        'recent_1m': (_month_start_from_end(full_end, INTERVAL_WINDOW_MONTHS['recent_1m']), full_end),
    }
    summary = {}
    for name, (start_date, end_date) in windows.items():
        sliced = filter_timing_result(result, start_date=start_date, end_date=end_date)
        if len(sliced) == 0:
            summary[name] = {
                'label': INTERVAL_WINDOW_LABELS.get(name, name),
                'rows': 0,
                'is_tradable': False,
                'start': None,
                'end': None,
                'training_range': {'start': None, 'end': None},
                'test_range': {'start': None, 'end': None},
                'metrics': {},
                'etf_summary': {},
            }
            continue
        metrics = evaluate_timing_result(
            sliced,
            benchmark_returns=benchmark_returns,
            reset_capital=(name != 'pre_6m_history'),
        )
        sliced_start = pd.to_datetime(sliced['交易日期'].min())
        sliced_end = pd.to_datetime(sliced['交易日期'].max())
        training_end = sliced_start - pd.Timedelta(days=1)
        has_training_history = bool(sliced_start > full_start)
        summary[name] = {
            'label': INTERVAL_WINDOW_LABELS.get(name, name),
            'rows': len(sliced),
            'is_tradable': bool(len(sliced) > 0),
            'start': sliced_start.strftime('%Y-%m-%d'),
            'end': sliced_end.strftime('%Y-%m-%d'),
            'training_range': {
                'start': full_start.strftime('%Y-%m-%d') if has_training_history else None,
                'end': training_end.strftime('%Y-%m-%d') if has_training_history else None,
            },
            'test_range': {
                'start': sliced_start.strftime('%Y-%m-%d'),
                'end': sliced_end.strftime('%Y-%m-%d'),
            },
            'metrics': {
                'cumulative_return': metrics.get('累积净值'),
                'annual_return': metrics.get('年化收益'),
                'max_drawdown': metrics.get('最大回撤'),
                'calmar_ratio': metrics.get('年化收益/回撤比'),
                'final_capital': metrics.get('最终资金'),
                'total_return_pct': metrics.get('总收益率'),
                'total_pnl': metrics.get('总盈亏'),
                'avg_exposure': metrics.get('平均仓位'),
                'rebalance_count': metrics.get('调仓次数'),
                'fee_ratio': metrics.get('手续费占比'),
                'reset_capital': metrics.get('资金重置口径', False),
            },
            'etf_summary': _build_etf_summary(sliced),
        }
    return summary


def _compute_single_benchmark_curve(result_df, index_returns):
    if index_returns is None:
        return []
    lookup = build_period_lookup(index_returns)
    curve = []
    cum = 1.0
    monthly_dates = result_df['交易日期'].drop_duplicates().sort_values()
    for date in monthly_dates:
        ret = get_index_return_for_date(date, lookup)
        cum *= (1 + ret)
        curve.append({'date': pd.to_datetime(date).strftime('%Y-%m-%d'), 'value': round(float(cum), 4)})
    return curve


def _compute_benchmark_curves(result_df, index_returns_map):
    curves = []
    for index_id, cfg in INDEX_CONFIGS.items():
        series = (index_returns_map or {}).get(index_id)
        if series is None:
            continue
        curves.append({
            'id': index_id,
            'name': cfg['name'],
            'curve': _compute_single_benchmark_curve(result_df, series),
        })
    return curves


def _compress_curve_points(curve, max_points=None):
    if not max_points or len(curve) <= max_points:
        return curve
    if max_points <= 2:
        return [curve[0], curve[-1]]
    last_index = len(curve) - 1
    indexes = np.linspace(0, last_index, num=max_points, dtype=int)
    unique_indexes = sorted(set(int(idx) for idx in indexes))
    if unique_indexes[-1] != last_index:
        unique_indexes.append(last_index)
    return [curve[idx] for idx in unique_indexes]


def _rounded_or_none(value, digits=4):
    if pd.isna(value):
        return None
    return round(float(value), digits)


def timing_result_to_json(result_df, metrics, benchmark_meta=None, benchmark_curve=None, benchmark_curves=None, compact=False):
    if len(result_df) == 0:
        raise ValueError('无真实 ETF 数据，无法生成择时回测结果')

    dates = pd.to_datetime(result_df['交易日期']).dt.strftime('%Y-%m-%d').tolist()
    nav_vals = result_df['累积净值'].round(4).tolist()
    ret_vals = result_df['strategy_return'].round(6).tolist()
    close_vals = result_df['close'].round(2).tolist()
    etf_close_vals = result_df['etf_close'].tolist() if 'etf_close' in result_df.columns else [None] * len(result_df)

    equity_curve = [
        {'date': d, 'value': v, 'return': r, 'close': c, 'etf_close': _rounded_or_none(ec, 4)}
        for d, v, r, c, ec in zip(dates, nav_vals, ret_vals, close_vals, etf_close_vals)
    ]
    if compact:
        equity_curve = _compress_curve_points(equity_curve, max_points=480)
    daily_equity_curve = list(equity_curve) if not compact else []

    cum = result_df['累积净值'].values
    peak = np.maximum.accumulate(cum)
    dd = (cum / peak - 1).round(6).tolist()
    drawdown = [{'date': d, 'value': v} for d, v in zip(dates, dd)] if not compact else []

    monthly_returns = [{'date': d, 'value': r} for d, r in zip(dates, ret_vals)] if not compact else []

    if not compact:
        years = pd.to_datetime(result_df['交易日期']).dt.year
        yearly_returns = result_df.groupby(years)['strategy_return'].apply(lambda x: (1 + x).prod() - 1)
        yearly_payload = [{'year': int(y), 'value': round(float(v), 6)} for y, v in yearly_returns.items()]
    else:
        yearly_payload = []

    signal_rows = result_df.tail(20) if compact else result_df
    sig_dates = pd.to_datetime(signal_rows['交易日期']).dt.strftime('%Y-%m-%d').tolist()
    signals = []
    trades = []
    for i, (_, row) in enumerate(signal_rows.iterrows()):
        item = {
            'date': sig_dates[i],
            'action': row['signal_action'],
            'position': int(row['position']),
            'reason_summary': row['reason_summary'],
            'reason_detail': list(row['reason_detail']) if isinstance(row['reason_detail'], (list, tuple)) else [str(row['reason_detail'])],
            'score': round(float(row.get('signal_score', 0) or 0), 6),
            'strength_score': round(float(row.get('strength_score', 0) or 0), 6),
            'target_exposure': round(float(row.get('target_exposure', row.get('position', 0)) or 0), 4),
            'prev_exposure': round(float(row.get('prev_exposure', 0) or 0), 4),
            'exposure_change': round(float(row.get('exposure_change', 0) or 0), 4),
            'rebalance_action': row.get('rebalance_action', row['signal_action']),
            'close': round(float(row['close']), 2),
            'etf_close': _rounded_or_none(row.get('etf_close'), 4),
        }
        signals.append(item)
        if not compact and row['signal_action'] in {'buy', 'sell'}:
            trades.append(item)

    trade_rows = result_df[result_df['trade_quantity'].abs() > 1e-8].copy() if 'trade_quantity' in result_df.columns else result_df[result_df['signal_action'].isin(['buy', 'sell'])].copy()
    all_trades = []
    trade_details = []
    realized_series = trade_rows['realized_pnl'].dropna() if 'realized_pnl' in trade_rows.columns else pd.Series(dtype=float)
    total_realized_pnl = float(realized_series.sum()) if len(realized_series) else 0.0
    last_realized_pnl = float(realized_series.iloc[-1]) if len(realized_series) else None

    for _, row in trade_rows.iterrows():
        normalized_action = row.get('signal_action')
        if normalized_action not in {'buy', 'sell'}:
            normalized_action = 'buy' if float(row.get('exposure_change', 0) or 0) > 0 else 'sell'
        trade_item = {
            'date': pd.to_datetime(row['交易日期']).strftime('%Y-%m-%d'),
            'action': normalized_action,
            'rebalance_action': row.get('rebalance_action', normalized_action),
            'target_exposure': round(float(row.get('target_exposure', row.get('position', 0)) or 0), 4),
            'close': round(float(row['close']), 2),
            'etf_code': row.get('etf_code') or result_df.attrs.get('etf_code'),
            'etf_name': row.get('etf_name') or result_df.attrs.get('etf_name'),
            'trade_price': _rounded_or_none(row.get('etf_open'), 4),
            'etf_close': _rounded_or_none(row.get('etf_close'), 4),
            'latest_price': _rounded_or_none(row.get('etf_close'), 4),
            'quantity': round(float(row.get('trade_quantity', 0) or 0), 4),
            'trade_amount': round(float(row.get('trade_amount', 0) or 0), 2),
            'fee_amount': round(float(row.get('trade_fee_amount', 0) or 0), 2),
            'holding_value': round(float(row.get('holding_value', 0) or 0), 2),
            'cash_balance': round(float(row.get('cash_balance', 0) or 0), 2),
            'realized_pnl': round(float(row.get('realized_pnl', 0) or 0), 2) if pd.notna(row.get('realized_pnl')) else None,
            'realized_pnl_pct': round(float(row.get('realized_pnl_pct', 0) or 0) * 100, 2) if pd.notna(row.get('realized_pnl_pct')) else None,
            'unrealized_pnl': round(float(row.get('unrealized_pnl', 0) or 0), 2) if pd.notna(row.get('unrealized_pnl')) else None,
            'entry_price': round(float(row.get('trade_entry_price', row.get('entry_price', 0)) or 0), 4) if pd.notna(row.get('trade_entry_price', row.get('entry_price'))) else None,
            'cost_price': round(float(row.get('trade_entry_price', row.get('entry_price', 0)) or 0), 4) if pd.notna(row.get('trade_entry_price', row.get('entry_price'))) else None,
            'entry_date': row.get('trade_entry_date', row.get('entry_date')),
            'holding_days': int(row.get('trade_holding_days', row.get('holding_days', 0)) or 0),
            'entry_capital': round(float(row.get('trade_entry_capital', row.get('entry_capital', 0)) or 0), 2) if pd.notna(row.get('trade_entry_capital', row.get('entry_capital'))) else None,
            'invested_amount': round(float(row.get('trade_invested_amount', row.get('invested_amount', 0)) or 0), 2) if pd.notna(row.get('trade_invested_amount', row.get('invested_amount'))) else None,
            'nav': round(float(row['累积净值']), 4),
            'position_label': row.get('position_label') or ('持仓中' if int(row['position']) == 1 else '空仓'),
            'commission': round(float(row.get('commission_cost', 0) or 0), 2),
            'stamp': round(float(row.get('stamp_cost', 0) or 0), 2),
            'transfer': round(float(row.get('transfer_cost', 0) or 0), 2),
            'slippage_cost': round(float(row.get('slippage_cost', 0) or 0), 2),
            'blocked_by_limit': bool(row.get('blocked_by_limit', False)),
            'limit_delays': int(row.get('limit_delays', 0) or 0),
        }
        all_trades.append({
            'date': trade_item['date'],
            'action': trade_item['action'],
            'rebalance_action': trade_item['rebalance_action'],
            'target_exposure': trade_item['target_exposure'],
            'prev_exposure': round(float(row.get('prev_exposure', 0) or 0), 4),
            'close': trade_item['close'],
            'trade_price': trade_item['trade_price'],
            'etf_close': trade_item['etf_close'],
            'latest_price': trade_item['latest_price'],
            'cost_price': trade_item['cost_price'],
            'quantity': trade_item['quantity'],
            'trade_amount': trade_item['trade_amount'],
            'fee_amount': trade_item['fee_amount'],
            'nav': trade_item['nav'],
        })
        trade_details.append(trade_item)

    latest = result_df.iloc[-1]
    snapshot_fields = {}
    for col in ['close', 'ma_fast', 'ma_slow', 'momentum_long', 'momentum_short', 'trend_ma', 'breakout_high', 'exit_low', 'strength_score', 'target_exposure']:
        if col in result_df.columns and pd.notna(latest.get(col)):
            snapshot_fields[col] = round(float(latest[col]), 6)

    def g(key):
        return metrics.get(key, 'N/A')

    selected_benchmark_curve = benchmark_curve or []
    if compact:
        selected_benchmark_curve = _compress_curve_points(selected_benchmark_curve, max_points=480)

    position_snapshot = {
        'etf_code': latest.get('etf_code') or result_df.attrs.get('etf_code'),
        'etf_name': latest.get('etf_name') or result_df.attrs.get('etf_name'),
        'etf_close': _rounded_or_none(latest.get('etf_close'), 4),
        'holding_units': round(float(latest.get('holding_units', 0) or 0), 4),
        'holding_value': round(float(latest.get('holding_value', 0) or 0), 2),
        'cash_balance': round(float(latest.get('cash_balance', 0) or 0), 2),
        'entry_price': round(float(latest.get('entry_price', 0) or 0), 4) if pd.notna(latest.get('entry_price')) else None,
        'entry_date': latest.get('entry_date'),
        'unrealized_pnl': round(float(latest.get('unrealized_pnl', 0) or 0), 2) if pd.notna(latest.get('unrealized_pnl')) else None,
        'unrealized_pnl_pct': round((float(latest.get('unrealized_pnl', 0) or 0) / float(latest.get('invested_amount', 0) or 0)) * 100, 2) if pd.notna(latest.get('unrealized_pnl')) and float(latest.get('invested_amount', 0) or 0) else None,
        'position_label': latest.get('position_label') or ('持仓中' if int(latest['position']) == 1 else '空仓'),
        'target_exposure': round(float(latest.get('target_exposure', latest.get('position', 0)) or 0), 4),
        'prev_exposure': round(float(latest.get('prev_exposure', 0) or 0), 4),
        'rebalance_action': latest.get('rebalance_action', latest['signal_action']),
    }

    trade_summary = {
        'trade_count': len(trade_details),
        'completed_trade_count': int(sum(1 for item in trade_details if item['action'] == 'sell')),
        'total_realized_pnl': round(total_realized_pnl, 2),
        'last_realized_pnl': round(last_realized_pnl, 2) if last_realized_pnl is not None else None,
        'current_unrealized_pnl': position_snapshot['unrealized_pnl'],
        'current_holding_value': position_snapshot['holding_value'],
        'avg_exposure': g('平均仓位'),
        'rebalance_count': g('调仓次数'),
    }
    etf_summary = _build_etf_summary(result_df)

    return {
        'equity_curve': equity_curve,
        'daily_equity_curve': daily_equity_curve,
        'drawdown': drawdown,
        'yearly_returns': yearly_payload,
        'monthly_returns': monthly_returns,
        'metrics': {
            'cumulative_return': g('累积净值'),
            'annual_return': g('年化收益'),
            'max_drawdown': g('最大回撤'),
            'max_dd_start': g('最大回撤开始'),
            'max_dd_end': g('最大回撤结束'),
            'calmar_ratio': g('年化收益/回撤比'),
            'final_capital': g('最终资金'),
            'total_return_pct': g('总收益率'),
            'total_pnl': g('总盈亏'),
            'avg_exposure': g('平均仓位'),
            'rebalance_count': g('调仓次数'),
            'fee_ratio': g('手续费占比'),
            'beta': g('Beta'),
            'annual_alpha': g('年化Alpha'),
            'information_ratio': g('信息比率'),
            'r_squared': g('R-squared'),
            'up_capture': g('上行捕获率'),
            'down_capture': g('下行捕获率'),
        },
        'initial_capital': float(result_df.attrs.get('initial_capital', 50000)),
        'win_rate': round(float((result_df['strategy_return'] > 0).mean()), 4),
        'date_range': {
            'start': pd.to_datetime(result_df['交易日期'].min()).strftime('%Y-%m-%d'),
            'end': pd.to_datetime(result_df['交易日期'].max()).strftime('%Y-%m-%d'),
            'etf_inception_date': result_df.attrs.get('etf_inception_date'),
        },
        'total_months': len(result_df),
        'signals': signals,
        'trades': trades,
        'all_trades': all_trades,
        'trade_details': trade_details,
        'trade_summary': trade_summary,
        'position_snapshot': position_snapshot,
        'etf_summary': etf_summary,
        'signal_summary': {
            'current_action': latest['signal_action'],
            'current_position': int(latest['position']),
            'current_reason': latest['reason_summary'],
            'target_exposure': round(float(latest.get('target_exposure', latest.get('position', 0)) or 0), 4),
            'prev_exposure': round(float(latest.get('prev_exposure', 0) or 0), 4),
            'exposure_change': round(float(latest.get('exposure_change', 0) or 0), 4),
            'rebalance_action': latest.get('rebalance_action', latest['signal_action']),
        },
        'active_index': {
            'id': latest['index_id'],
            'name': latest['index_name'],
        },
        'indicator_snapshots': snapshot_fields,
        'active_benchmark': benchmark_meta,
        'benchmark_curve': selected_benchmark_curve,
        'benchmark_curves': [] if compact else (benchmark_curves or []),
        'fee_info': {
            'buy_cost': result_df.attrs.get('buy_cost', 0.001),
            'sell_cost': result_df.attrs.get('sell_cost', 0.001),
            'total_trade_cost': round(float(result_df.get('trade_fee_amount', pd.Series(dtype=float)).sum()), 2),
            'commission_total': round(float(result_df.attrs.get('commission_total', 0.0) or 0.0), 2),
            'stamp_total': round(float(result_df.attrs.get('stamp_total', 0.0) or 0.0), 2),
            'transfer_total': round(float(result_df.attrs.get('transfer_total', 0.0) or 0.0), 2),
            'slippage_total': round(float(result_df.attrs.get('slippage_total', 0.0) or 0.0), 2),
            'cash_interest_total': round(float(result_df.attrs.get('cash_interest_total', 0.0) or 0.0), 2),
            'limit_block_count': int(result_df.attrs.get('limit_block_count', 0) or 0),
        },
        'realism_meta': {
            'settlement_mode': result_df.attrs.get('settlement_mode'),
            'limit_pct': result_df.attrs.get('limit_pct'),
            'market': result_df.attrs.get('market'),
            'slippage_bps': result_df.attrs.get('slippage_bps'),
            'cash_interest_rate': result_df.attrs.get('cash_interest_rate'),
            'commission_rate': result_df.attrs.get('commission_rate'),
            'commission_min': result_df.attrs.get('commission_min'),
            'stamp_tax_rate': result_df.attrs.get('stamp_tax_rate'),
            'transfer_fee_rate': result_df.attrs.get('transfer_fee_rate'),
            'limit_max_delay_days': result_df.attrs.get('limit_max_delay_days'),
            'profit_lock_enabled': result_df.attrs.get('profit_lock_enabled'),
            'profit_lock_drawdown': result_df.attrs.get('profit_lock_drawdown'),
            'profit_lock_level_1': result_df.attrs.get('profit_lock_level_1'),
            'profit_lock_level_2': result_df.attrs.get('profit_lock_level_2'),
            'profit_lock_level_3': result_df.attrs.get('profit_lock_level_3'),
        },
        'exposure_mode': result_df.attrs.get('exposure_mode', 'binary'),
    }
