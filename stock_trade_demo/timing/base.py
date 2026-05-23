import pandas as pd

from index_data import INDEX_CONFIGS, TIMING_ETF_CONFIGS


class BaseTimingStrategy:
    strategy_id = ''
    display_name = ''
    strategy_description = ''

    def __init__(self, initial_capital=50000, buy_cost=0.001, sell_cost=0.001,
                 exposure_mode='binary', enter_threshold=0.55, add_threshold=0.75,
                 trim_threshold=0.35, exit_threshold=0.15, confirm_days=1,
                 max_entry_exposure=0.5, probe_entry_exposure=0.25,
                 probe_confirm_days=1, profit_lock_enabled=False,
                 profit_lock_drawdown=0.04, profit_lock_level_1=0.10,
                 profit_lock_level_2=0.18, profit_lock_level_3=0.28,
                 slippage_bps=5.0, cash_interest_rate=0.015,
                 commission_rate=0.0001, commission_min=5.0,
                 stamp_tax_rate=0.0, transfer_fee_rate=0.00001,
                 limit_max_delay_days=5, **params):
        self.initial_capital = initial_capital
        self.buy_cost = buy_cost
        self.sell_cost = sell_cost
        self.exposure_mode = exposure_mode or 'binary'
        self.enter_threshold = float(enter_threshold)
        self.add_threshold = float(add_threshold)
        self.trim_threshold = float(trim_threshold)
        self.exit_threshold = float(exit_threshold)
        self.confirm_days = max(int(confirm_days or 1), 1)
        self.max_entry_exposure = float(max_entry_exposure)
        self.probe_entry_exposure = float(probe_entry_exposure)
        self.probe_confirm_days = max(int(probe_confirm_days or 1), 1)
        self.profit_lock_enabled = bool(profit_lock_enabled)
        self.profit_lock_drawdown = max(float(profit_lock_drawdown or 0.0), 0.0)
        self.profit_lock_level_1 = float(profit_lock_level_1)
        self.profit_lock_level_2 = float(profit_lock_level_2)
        self.profit_lock_level_3 = float(profit_lock_level_3)
        # 真实交易规则层共享参数（binary 与 staged 共用）
        self.slippage_bps = max(float(slippage_bps or 0.0), 0.0)
        self.cash_interest_rate = max(float(cash_interest_rate or 0.0), 0.0)
        self.commission_rate = max(float(commission_rate or 0.0), 0.0)
        self.commission_min = max(float(commission_min or 0.0), 0.0)
        self.stamp_tax_rate = max(float(stamp_tax_rate or 0.0), 0.0)
        self.transfer_fee_rate = max(float(transfer_fee_rate or 0.0), 0.0)
        self.limit_max_delay_days = max(int(limit_max_delay_days or 0), 0)
        self._extra_params = params

    def get_display_name(self):
        return self.display_name or self.__class__.__name__

    def get_strategy_description(self):
        return self.strategy_description or self.__doc__ or ''

    def get_parameter_definitions(self):
        return []

    def get_shared_parameter_definitions(self):
        return [
            {'key': 'exposure_mode', 'label': '仓位模式', 'type': 'timing_select', 'options': [
                {'value': 'binary', 'label': 'binary'},
                {'value': 'staged', 'label': 'staged'},
            ], 'default': self.exposure_mode, 'description': 'binary 为满仓/空仓切换，staged 为分档加减仓。'},
            {'key': 'confirm_days', 'label': '确认天数', 'type': 'timing', 'min': 1, 'max': 5, 'step': 1, 'default': self.confirm_days, 'unit': '日', 'description': '进入或退出信号需要连续满足的天数，用于降低来回打脸。'},
            {'key': 'max_entry_exposure', 'label': '首次建仓上限', 'type': 'timing', 'min': 0.25, 'max': 1.0, 'step': 0.25, 'default': self.max_entry_exposure, 'unit': '', 'description': 'staged 模式首次建仓允许达到的最大仓位。'},
            {'key': 'probe_entry_exposure', 'label': '试探仓位', 'type': 'timing', 'min': 0.1, 'max': 0.5, 'step': 0.05, 'default': self.probe_entry_exposure, 'unit': '', 'description': '从空仓转入时先建立的小仓位，用于试探性跟随新趋势。'},
            {'key': 'probe_confirm_days', 'label': '试探确认', 'type': 'timing', 'min': 1, 'max': 5, 'step': 1, 'default': self.probe_confirm_days, 'unit': '日', 'description': '首次试探建仓前所需的连续确认天数。'},
            {'key': 'enter_threshold', 'label': '建仓阈值', 'type': 'timing', 'min': 0.4, 'max': 0.8, 'step': 0.05, 'default': self.enter_threshold, 'unit': '', 'description': '强度分数达到该阈值后，才允许开始建仓。'},
            {'key': 'add_threshold', 'label': '加仓阈值', 'type': 'timing', 'min': 0.55, 'max': 0.95, 'step': 0.05, 'default': self.add_threshold, 'unit': '', 'description': '强度继续上行并超过该阈值时，允许逐步加仓。'},
            {'key': 'trim_threshold', 'label': '减仓阈值', 'type': 'timing', 'min': 0.1, 'max': 0.6, 'step': 0.05, 'default': self.trim_threshold, 'unit': '', 'description': '强度回落到该阈值下方时，执行逐步减仓。'},
            {'key': 'exit_threshold', 'label': '清仓阈值', 'type': 'timing', 'min': 0.05, 'max': 0.4, 'step': 0.05, 'default': self.exit_threshold, 'unit': '', 'description': '强度跌破该阈值并确认后，执行清仓。'},
            {'key': 'profit_lock_enabled', 'label': '盈利锁定', 'type': 'timing_select', 'options': [
                {'value': False, 'label': 'off'},
                {'value': True, 'label': 'on'},
            ], 'default': self.profit_lock_enabled, 'description': '对 staged 仓位启用共享阶梯止盈/回撤锁盈覆盖层。'},
            {'key': 'profit_lock_drawdown', 'label': '锁盈回撤阈值', 'type': 'timing', 'min': 0.01, 'max': 0.1, 'step': 0.01, 'default': self.profit_lock_drawdown, 'unit': '', 'description': '持仓浮盈达到目标后，若从峰值回撤超过该比例，则按阶梯锁定后的仓位回落。'},
            {'key': 'profit_lock_level_1', 'label': '锁盈一级阈值', 'type': 'timing', 'min': 0.04, 'max': 0.2, 'step': 0.01, 'default': self.profit_lock_level_1, 'unit': '', 'description': '单笔持仓浮盈达到该比例后，至少锁定到 75% 仓位。'},
            {'key': 'profit_lock_level_2', 'label': '锁盈二级阈值', 'type': 'timing', 'min': 0.08, 'max': 0.3, 'step': 0.01, 'default': self.profit_lock_level_2, 'unit': '', 'description': '单笔持仓浮盈达到该比例后，至少锁定到 50% 仓位。'},
            {'key': 'profit_lock_level_3', 'label': '锁盈三级阈值', 'type': 'timing', 'min': 0.12, 'max': 0.5, 'step': 0.01, 'default': self.profit_lock_level_3, 'unit': '', 'description': '单笔持仓浮盈达到该比例后，至少锁定到 25% 仓位。'},
            # 真实交易规则层（binary / staged 共用）
            {'key': 'slippage_bps', 'label': '开盘滑点', 'type': 'timing', 'min': 0.0, 'max': 30.0, 'step': 0.5, 'default': self.slippage_bps, 'unit': 'bp', 'description': '买入按开盘价上浮、卖出按开盘价下浮该 bp 数模拟集合竞价滑点。'},
            {'key': 'cash_interest_rate', 'label': '现金计息', 'type': 'timing', 'min': 0.0, 'max': 0.05, 'step': 0.0025, 'default': self.cash_interest_rate, 'unit': '/年', 'description': '空仓 / 半仓现金按该年化利率（÷252 计入每日）累计利息。'},
            {'key': 'commission_rate', 'label': '佣金费率', 'type': 'timing', 'min': 0.0, 'max': 0.001, 'step': 0.00005, 'default': self.commission_rate, 'unit': '', 'description': 'ETF 佣金费率（双边），按成交金额计提，最低 commission_min 元/笔。'},
            {'key': 'commission_min', 'label': '佣金最低', 'type': 'timing', 'min': 0.0, 'max': 20.0, 'step': 1.0, 'default': self.commission_min, 'unit': '元', 'description': '单笔最低佣金，不足按最低值收取。'},
            {'key': 'stamp_tax_rate', 'label': '印花税', 'type': 'timing', 'min': 0.0, 'max': 0.002, 'step': 0.0001, 'default': self.stamp_tax_rate, 'unit': '', 'description': 'ETF 默认免征印花税（0），股票卖出方向才适用。'},
            {'key': 'transfer_fee_rate', 'label': '过户费', 'type': 'timing', 'min': 0.0, 'max': 0.0001, 'step': 0.000005, 'default': self.transfer_fee_rate, 'unit': '', 'description': '仅沪市 ETF 收取过户费（双边）。'},
            {'key': 'limit_max_delay_days', 'label': '涨跌停顺延天数', 'type': 'timing', 'min': 0, 'max': 10, 'step': 1, 'default': self.limit_max_delay_days, 'unit': '日', 'description': '买卖单遇涨跌停封板未成交时，最多顺延的交易日数；超过则丢弃。0 表示不顺延。'},
        ]

    def get_principle_summary(self):
        return self.get_strategy_description()

    def get_formula_blocks(self):
        return []

    def get_shared_exposure_blocks(self):
        return [
            {
                'title': '强度分数归一化',
                'expression': 's_t = 1 / (1 + e^{-raw_t})',
                'explanation': '将各策略的 raw score 通过 sigmoid 压缩到 0~1 区间，作为统一仓位控制输入。',
            },
            {
                'title': '分档目标仓位',
                'expression': 'E_t = 0, 0.25, 0.5, 0.75, 1.0',
                'explanation': '当 s_t 依次跌破 exit / trim / enter / add 阈值时，目标仓位在五个离散档位之间切换。',
            },
            {
                'title': '确认与试探建仓规则',
                'expression': 'up_streak >= probe_confirm_days,  E_probe = min(probe_entry_exposure, max_entry_exposure)',
                'explanation': '首次从空仓转入时，先以小仓位试探建仓；确认趋势后再按 staged 规则逐步加仓。',
            },
            {
                'title': '加减仓规则',
                'expression': '若 s_t >= add_threshold 则每次 +0.25；若 s_t <= trim_threshold 则每次 -0.25；若连续 confirm_days 天 s_t <= exit_threshold 则清仓。',
                'explanation': 'staged 模式通过分段加减仓降低 binary 满进满出带来的频繁交易。',
            },
            {
                'title': '共享阶梯锁盈覆盖层',
                'expression': '若浮盈先后达到 p1/p2/p3，锁仓下限依次提升为 0.75/0.5/0.25；若从持仓峰值回撤超过 d，则目标仓位不得高于该锁仓下限。',
                'explanation': '仅在启用 profit_lock 时生效，用于把已获得的盈利以网格式逐步兑现，避免强趋势后大幅回吐。',
            },
        ]

    def get_signal_metadata(self):
        return {
            'id': self.strategy_id,
            'name': self.get_display_name(),
            'description': self.get_strategy_description(),
            'principle_summary': self.get_principle_summary(),
            'formula_blocks': self.get_formula_blocks(),
            'shared_exposure_blocks': self.get_shared_exposure_blocks(),
            'parameters': self.get_parameter_definitions() + self.get_shared_parameter_definitions(),
        }

    def get_index_id(self):
        raise NotImplementedError

    def get_index_name(self):
        index_id = self.get_index_id()
        return INDEX_CONFIGS.get(index_id, {}).get('name', index_id)

    def get_etf_config(self):
        return TIMING_ETF_CONFIGS.get(self.get_index_id(), {})

    def compute_indicators(self, df):
        raise NotImplementedError

    def build_signal_reason(self, row):
        raise NotImplementedError

    def generate_signals(self, df):
        raise NotImplementedError

    def _bucket_exposure(self, strength):
        strength = float(min(max(strength, 0.0), 1.0))
        if strength <= self.exit_threshold:
            return 0.0
        if strength <= self.trim_threshold:
            return 0.25
        if strength <= self.enter_threshold:
            return 0.5
        if strength <= self.add_threshold:
            return 0.75
        return 1.0

    def _normalize_profit_lock_levels(self):
        levels = sorted([
            max(float(self.profit_lock_level_1), 0.0),
            max(float(self.profit_lock_level_2), 0.0),
            max(float(self.profit_lock_level_3), 0.0),
        ])
        return levels

    def _build_staged_target_exposure(self, strength_series, ready_mask=None, price_series=None):
        strength = pd.Series(strength_series, copy=True).astype(float).clip(lower=0.0, upper=1.0).fillna(0.0)
        if ready_mask is None:
            ready = pd.Series(True, index=strength.index)
        else:
            ready = pd.Series(ready_mask, index=strength.index).fillna(False).astype(bool)
        if price_series is None:
            prices = pd.Series(float('nan'), index=strength.index, dtype='float64')
        else:
            prices = pd.Series(price_series, index=strength.index).astype(float)

        lock_levels = self._normalize_profit_lock_levels()
        lock_targets = [0.75, 0.5, 0.25]

        exposures = []
        current_exposure = 0.0
        up_streak = 0
        down_streak = 0
        entry_price = None
        peak_price = None

        for idx, score in strength.items():
            if not bool(ready.loc[idx]):
                current_exposure = 0.0
                up_streak = 0
                down_streak = 0
                entry_price = None
                peak_price = None
                exposures.append(0.0)
                continue

            price = prices.loc[idx]
            if score >= self.enter_threshold:
                up_streak += 1
            else:
                up_streak = 0

            if score <= self.exit_threshold:
                down_streak += 1
            else:
                down_streak = 0

            desired_exposure = self._bucket_exposure(score)
            probe_entry = min(max(self.probe_entry_exposure, 0.1), self.max_entry_exposure)

            if current_exposure <= 0.0:
                entry_price = None
                peak_price = None
                if up_streak >= self.probe_confirm_days and desired_exposure > 0.0:
                    current_exposure = min(max(probe_entry, 0.1), 1.0)
                    if pd.notna(price) and float(price) > 0:
                        entry_price = float(price)
                        peak_price = float(price)
            else:
                if pd.notna(price) and float(price) > 0:
                    if entry_price is None:
                        entry_price = float(price)
                    peak_price = max(float(peak_price if peak_price is not None else price), float(price))

                if down_streak >= self.confirm_days:
                    current_exposure = 0.0
                    entry_price = None
                    peak_price = None
                elif desired_exposure > current_exposure and score >= self.add_threshold:
                    step = 0.25
                    next_exposure = current_exposure + step
                    current_exposure = min(desired_exposure, next_exposure, 1.0)
                elif desired_exposure < current_exposure and score <= self.trim_threshold:
                    current_exposure = max(desired_exposure, current_exposure - 0.25)

                if self.profit_lock_enabled and entry_price and peak_price and self.profit_lock_drawdown > 0:
                    locked_floor = None
                    profit_from_entry = peak_price / entry_price - 1 if entry_price > 0 else 0.0
                    for profit_level, exposure_floor in zip(lock_levels, lock_targets):
                        if profit_from_entry >= profit_level:
                            locked_floor = exposure_floor
                    if locked_floor is not None and peak_price > 0 and pd.notna(price) and float(price) > 0:
                        drawdown_from_peak = 1 - float(price) / peak_price
                        if drawdown_from_peak >= self.profit_lock_drawdown:
                            current_exposure = min(current_exposure, locked_floor)

                if current_exposure <= 0.0:
                    entry_price = None
                    peak_price = None

            exposures.append(round(float(min(max(current_exposure, 0.0), 1.0)), 4))

        return pd.Series(exposures, index=strength.index, dtype=float)

    def _apply_exposure_columns(self, df, binary_position, staged_strength=None, ready_mask=None, price_series=None):
        df = df.copy()
        binary_position = pd.Series(binary_position, index=df.index).fillna(0).astype(float).clip(lower=0.0, upper=1.0)

        if self.exposure_mode == 'staged' and staged_strength is not None:
            # binary 作为门控叠加：把 binary_position > 0 与 ready_mask 做 AND，
            # 让 staged 状态机内部 entry_price / peak_price 在 binary 清仓时同步重置。
            binary_gate = (binary_position > 0)
            if ready_mask is None:
                combined_ready = binary_gate
            else:
                combined_ready = pd.Series(ready_mask, index=df.index).fillna(False).astype(bool) & binary_gate
            target_exposure = self._build_staged_target_exposure(staged_strength, ready_mask=combined_ready, price_series=price_series)
            # 兜底：再次以 binary 门控覆盖，确保任何路径都不会绕开 binary 清仓信号
            target_exposure = target_exposure.where(binary_gate, 0.0)
        else:
            target_exposure = binary_position.astype(float)

        target_exposure = target_exposure.fillna(0.0).clip(lower=0.0, upper=1.0).round(4)
        prev_exposure = target_exposure.shift(1).fillna(0.0).round(4)
        exposure_change = (target_exposure - prev_exposure).round(4)

        signal_action = pd.Series('flat', index=df.index, dtype=object)
        signal_action.loc[(prev_exposure <= 0) & (target_exposure > 0)] = 'buy'
        signal_action.loc[(prev_exposure > 0) & (target_exposure <= 0)] = 'sell'
        signal_action.loc[(prev_exposure > 0) & (target_exposure > 0)] = 'hold'

        rebalance_action = pd.Series('flat', index=df.index, dtype=object)
        rebalance_action.loc[(prev_exposure <= 0) & (target_exposure > 0)] = 'enter'
        rebalance_action.loc[(prev_exposure > 0) & (target_exposure <= 0)] = 'exit'
        rebalance_action.loc[(prev_exposure > 0) & (target_exposure > prev_exposure)] = 'add'
        rebalance_action.loc[(prev_exposure > 0) & (target_exposure < prev_exposure) & (target_exposure > 0)] = 'trim'
        rebalance_action.loc[(prev_exposure > 0) & (target_exposure == prev_exposure)] = 'hold'

        df['target_exposure'] = target_exposure
        df['prev_exposure'] = prev_exposure
        df['exposure_change'] = exposure_change
        df['position'] = (target_exposure > 0).astype(int)
        df['signal_action'] = signal_action
        df['rebalance_action'] = rebalance_action
        return df

    def run(self, panel_df):
        df = panel_df.copy()
        df = self.compute_indicators(df)
        df = self.generate_signals(df)
        if '交易日期' not in df.columns and 'date' in df.columns:
            df['交易日期'] = pd.to_datetime(df['date'])
        return df
