import pandas as pd

from timing.base import BaseTimingStrategy


class CSI1000TimingStrategy(BaseTimingStrategy):
    strategy_id = 'csi1000_timing'
    display_name = 'CSI1000 择时策略'
    strategy_description = '基于均线趋势与中期动量判断 CSI1000 的买卖时机。'

    def __init__(self, fast_window=20, slow_window=60, momentum_window=60,
                 initial_capital=100000, buy_cost=0.001, sell_cost=0.001):
        super().__init__(initial_capital=initial_capital, buy_cost=buy_cost, sell_cost=sell_cost)
        self.fast_window = int(fast_window)
        self.slow_window = int(slow_window)
        self.momentum_window = int(momentum_window)

    def get_index_id(self):
        return 'csi1000'

    def get_parameter_definitions(self):
        return [
            {'key': 'fast_window', 'label': '快均线窗口', 'type': 'timing', 'min': 5, 'max': 60, 'step': 1, 'default': self.fast_window, 'unit': '日', 'description': '用于判断短期趋势的均线窗口。'},
            {'key': 'slow_window', 'label': '慢均线窗口', 'type': 'timing', 'min': 20, 'max': 180, 'step': 1, 'default': self.slow_window, 'unit': '日', 'description': '用于判断中期趋势的均线窗口。'},
            {'key': 'momentum_window', 'label': '动量窗口', 'type': 'timing', 'min': 10, 'max': 180, 'step': 1, 'default': self.momentum_window, 'unit': '日', 'description': '用于确认中期动量是否为正。'},
        ]

    def compute_indicators(self, df):
        close_col = 'csi1000_close'
        out = df[['交易日期', close_col]].dropna().copy()
        out['close'] = out[close_col]
        out['ma_fast'] = out['close'].rolling(self.fast_window).mean()
        out['ma_slow'] = out['close'].rolling(self.slow_window).mean()
        out['momentum_long'] = out['close'].pct_change(self.momentum_window)
        return out

    def build_signal_reason(self, row):
        details = [
            f"收盘价 {row['close']:.2f}",
            f"快线 {row['ma_fast']:.2f}" if pd.notna(row['ma_fast']) else '快线未就绪',
            f"慢线 {row['ma_slow']:.2f}" if pd.notna(row['ma_slow']) else '慢线未就绪',
            f"中期动量 {row['momentum_long'] * 100:.2f}%" if pd.notna(row['momentum_long']) else '动量未就绪',
        ]
        if row['position'] == 1:
            summary = 'CSI1000 满足趋势与动量条件，保持持有。'
        else:
            summary = 'CSI1000 未满足趋势与动量条件，保持空仓。'
        return summary, details

    def generate_signals(self, df):
        df = df.copy()
        buy_cond = (
            (df['close'] > df['ma_fast']) &
            (df['ma_fast'] > df['ma_slow']) &
            (df['momentum_long'] > 0)
        )
        df['position'] = buy_cond.fillna(False).astype(int)
        prev_pos = df['position'].shift(1).fillna(0).astype(int)
        df['signal_action'] = 'flat'
        df.loc[(prev_pos == 0) & (df['position'] == 1), 'signal_action'] = 'buy'
        df.loc[(prev_pos == 1) & (df['position'] == 0), 'signal_action'] = 'sell'
        df.loc[(prev_pos == 1) & (df['position'] == 1), 'signal_action'] = 'hold'
        df['signal_score'] = (
            ((df['close'] / df['ma_fast']) - 1).fillna(0) +
            ((df['ma_fast'] / df['ma_slow']) - 1).fillna(0) +
            df['momentum_long'].fillna(0)
        ) / 3
        df['index_id'] = self.get_index_id()
        df['index_name'] = self.get_index_name()
        reasons = df.apply(self.build_signal_reason, axis=1)
        df['reason_summary'] = reasons.map(lambda x: x[0])
        df['reason_detail'] = reasons.map(lambda x: x[1])
        return df


class Star50TimingStrategy(BaseTimingStrategy):
    strategy_id = 'star50_timing'
    display_name = '科创50 择时策略'
    strategy_description = '基于突破确认与趋势过滤判断科创50的买卖时机。'

    def __init__(self, breakout_window=20, exit_window=10, trend_window=60,
                 initial_capital=100000, buy_cost=0.001, sell_cost=0.001):
        super().__init__(initial_capital=initial_capital, buy_cost=buy_cost, sell_cost=sell_cost)
        self.breakout_window = int(breakout_window)
        self.exit_window = int(exit_window)
        self.trend_window = int(trend_window)

    def get_index_id(self):
        return 'star50'

    def get_parameter_definitions(self):
        return [
            {'key': 'breakout_window', 'label': '突破窗口', 'type': 'timing', 'min': 5, 'max': 80, 'step': 1, 'default': self.breakout_window, 'unit': '日', 'description': '用于判定价格向上突破的历史窗口。'},
            {'key': 'exit_window', 'label': '退出窗口', 'type': 'timing', 'min': 5, 'max': 40, 'step': 1, 'default': self.exit_window, 'unit': '日', 'description': '用于判定跌破防线的历史窗口。'},
            {'key': 'trend_window', 'label': '趋势均线', 'type': 'timing', 'min': 20, 'max': 180, 'step': 1, 'default': self.trend_window, 'unit': '日', 'description': '用于确认趋势方向的均线窗口。'},
        ]

    def compute_indicators(self, df):
        close_col = 'star50_close'
        high_col = 'star50_high'
        low_col = 'star50_low'
        out = df[['交易日期', close_col, high_col, low_col]].dropna().copy()
        out['close'] = out[close_col]
        out['high'] = out[high_col]
        out['low'] = out[low_col]
        out['trend_ma'] = out['close'].rolling(self.trend_window).mean()
        out['breakout_high'] = out['high'].rolling(self.breakout_window).max().shift(1)
        out['exit_low'] = out['low'].rolling(self.exit_window).min().shift(1)
        return out

    def build_signal_reason(self, row):
        details = [
            f"收盘价 {row['close']:.2f}",
            f"突破高点 {row['breakout_high']:.2f}" if pd.notna(row['breakout_high']) else '突破位未就绪',
            f"退出低点 {row['exit_low']:.2f}" if pd.notna(row['exit_low']) else '退出位未就绪',
            f"趋势均线 {row['trend_ma']:.2f}" if pd.notna(row['trend_ma']) else '趋势均线未就绪',
        ]
        if row['position'] == 1:
            summary = '科创50 突破确认且站上趋势线，保持持有。'
        else:
            summary = '科创50 未满足突破确认条件，保持空仓。'
        return summary, details

    def generate_signals(self, df):
        df = df.copy()
        buy_cond = (df['close'] > df['breakout_high']) & (df['close'] > df['trend_ma'])
        sell_cond = (df['close'] < df['exit_low']) | (df['close'] < df['trend_ma'])
        positions = []
        current = 0
        for _, row in df.iterrows():
            if current == 0 and bool(buy_cond.loc[row.name]) if pd.notna(buy_cond.loc[row.name]) else False:
                current = 1
            elif current == 1 and bool(sell_cond.loc[row.name]) if pd.notna(sell_cond.loc[row.name]) else False:
                current = 0
            positions.append(current)
        df['position'] = positions
        prev_pos = df['position'].shift(1).fillna(0).astype(int)
        df['signal_action'] = 'flat'
        df.loc[(prev_pos == 0) & (df['position'] == 1), 'signal_action'] = 'buy'
        df.loc[(prev_pos == 1) & (df['position'] == 0), 'signal_action'] = 'sell'
        df.loc[(prev_pos == 1) & (df['position'] == 1), 'signal_action'] = 'hold'
        df['signal_score'] = (
            ((df['close'] / df['breakout_high']) - 1).replace([pd.NA], 0).fillna(0) +
            ((df['close'] / df['trend_ma']) - 1).replace([pd.NA], 0).fillna(0)
        ) / 2
        df['index_id'] = self.get_index_id()
        df['index_name'] = self.get_index_name()
        reasons = df.apply(self.build_signal_reason, axis=1)
        df['reason_summary'] = reasons.map(lambda x: x[0])
        df['reason_detail'] = reasons.map(lambda x: x[1])
        return df


class ChiNextTimingStrategy(BaseTimingStrategy):
    strategy_id = 'chinext_timing'
    display_name = '创业板 择时策略'
    strategy_description = '基于短中期动量与趋势线确认创业板的买卖时机。'

    def __init__(self, momentum_short_window=20, momentum_long_window=60, trend_window=60,
                 momentum_threshold=0.0, initial_capital=100000, buy_cost=0.001, sell_cost=0.001):
        super().__init__(initial_capital=initial_capital, buy_cost=buy_cost, sell_cost=sell_cost)
        self.momentum_short_window = int(momentum_short_window)
        self.momentum_long_window = int(momentum_long_window)
        self.trend_window = int(trend_window)
        self.momentum_threshold = float(momentum_threshold)

    def get_index_id(self):
        return 'chinext'

    def get_parameter_definitions(self):
        return [
            {'key': 'momentum_short_window', 'label': '短动量窗口', 'type': 'timing', 'min': 5, 'max': 60, 'step': 1, 'default': self.momentum_short_window, 'unit': '日', 'description': '用于判定短期动量的窗口。'},
            {'key': 'momentum_long_window', 'label': '长动量窗口', 'type': 'timing', 'min': 20, 'max': 180, 'step': 1, 'default': self.momentum_long_window, 'unit': '日', 'description': '用于判定中期动量的窗口。'},
            {'key': 'trend_window', 'label': '趋势均线', 'type': 'timing', 'min': 20, 'max': 180, 'step': 1, 'default': self.trend_window, 'unit': '日', 'description': '用于确认趋势方向的均线窗口。'},
            {'key': 'momentum_threshold', 'label': '动量阈值', 'type': 'timing', 'min': -0.1, 'max': 0.2, 'step': 0.005, 'default': self.momentum_threshold, 'unit': '', 'description': '短期动量需要超过该阈值才视为有效买入。'},
        ]

    def compute_indicators(self, df):
        close_col = 'chinext_close'
        out = df[['交易日期', close_col]].dropna().copy()
        out['close'] = out[close_col]
        out['trend_ma'] = out['close'].rolling(self.trend_window).mean()
        out['momentum_short'] = out['close'].pct_change(self.momentum_short_window)
        out['momentum_long'] = out['close'].pct_change(self.momentum_long_window)
        return out

    def build_signal_reason(self, row):
        details = [
            f"收盘价 {row['close']:.2f}",
            f"短期动量 {row['momentum_short'] * 100:.2f}%" if pd.notna(row['momentum_short']) else '短动量未就绪',
            f"中期动量 {row['momentum_long'] * 100:.2f}%" if pd.notna(row['momentum_long']) else '长动量未就绪',
            f"趋势均线 {row['trend_ma']:.2f}" if pd.notna(row['trend_ma']) else '趋势均线未就绪',
        ]
        if row['position'] == 1:
            summary = '创业板动量与趋势同步转强，保持持有。'
        else:
            summary = '创业板动量或趋势不足，保持空仓。'
        return summary, details

    def generate_signals(self, df):
        df = df.copy()
        buy_cond = (
            (df['momentum_short'] > self.momentum_threshold) &
            (df['momentum_long'] > 0) &
            (df['close'] > df['trend_ma'])
        )
        df['position'] = buy_cond.fillna(False).astype(int)
        prev_pos = df['position'].shift(1).fillna(0).astype(int)
        df['signal_action'] = 'flat'
        df.loc[(prev_pos == 0) & (df['position'] == 1), 'signal_action'] = 'buy'
        df.loc[(prev_pos == 1) & (df['position'] == 0), 'signal_action'] = 'sell'
        df.loc[(prev_pos == 1) & (df['position'] == 1), 'signal_action'] = 'hold'
        df['signal_score'] = (
            df['momentum_short'].fillna(0) +
            df['momentum_long'].fillna(0) +
            ((df['close'] / df['trend_ma']) - 1).fillna(0)
        ) / 3
        df['index_id'] = self.get_index_id()
        df['index_name'] = self.get_index_name()
        reasons = df.apply(self.build_signal_reason, axis=1)
        df['reason_summary'] = reasons.map(lambda x: x[0])
        df['reason_detail'] = reasons.map(lambda x: x[1])
        return df
