import os
import numpy as np
import pandas as pd

from timing.base import BaseTimingStrategy

_MACRO_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')


def _load_fred_series(name, col):
    fp = os.path.join(_MACRO_DATA_DIR, f'fred_{name}.csv')
    if not os.path.exists(fp):
        return pd.Series(dtype=float, name=col)
    df = pd.read_csv(fp, parse_dates=[0], index_col=0).dropna()
    s = df.iloc[:, 0]
    s.name = col
    return s


def _zscore_rolling(s, window=252*5, min_periods=252):
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std()
    return ((s - mu) / (sd + 1e-9)).clip(-3, 3)


def _calc_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def _normalize_score(series):
    s = pd.Series(series, copy=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return 1.0 / (1.0 + np.exp(-s))


class CSI1000TimingStrategy(BaseTimingStrategy):
    strategy_id = 'csi1000_timing'
    display_name = 'CSI1000 择时策略'
    strategy_description = '基于突破确认、趋势过滤与 MACD 辅助信号判断 CSI1000 的买卖时机。'

    def __init__(self, breakout_window=15, exit_window=7, trend_window=50,
                 initial_capital=50000, buy_cost=0.001, sell_cost=0.001,
                 exposure_mode='staged', enter_threshold=0.55, add_threshold=0.75,
                 trim_threshold=0.38, exit_threshold=0.18, confirm_days=1,
                 max_entry_exposure=0.5, probe_entry_exposure=0.25,
                 probe_confirm_days=1, profit_lock_enabled=False,
                 profit_lock_drawdown=0.04, profit_lock_level_1=0.10,
                 profit_lock_level_2=0.18, profit_lock_level_3=0.28):
        super().__init__(
            initial_capital=initial_capital,
            buy_cost=buy_cost,
            sell_cost=sell_cost,
            exposure_mode=exposure_mode,
            enter_threshold=enter_threshold,
            add_threshold=add_threshold,
            trim_threshold=trim_threshold,
            exit_threshold=exit_threshold,
            confirm_days=confirm_days,
            max_entry_exposure=max_entry_exposure,
            probe_entry_exposure=probe_entry_exposure,
            probe_confirm_days=probe_confirm_days,
            profit_lock_enabled=profit_lock_enabled,
            profit_lock_drawdown=profit_lock_drawdown,
            profit_lock_level_1=profit_lock_level_1,
            profit_lock_level_2=profit_lock_level_2,
            profit_lock_level_3=profit_lock_level_3,
        )
        self.breakout_window = int(breakout_window)
        self.exit_window = int(exit_window)
        self.trend_window = int(trend_window)

    def get_index_id(self):
        return 'csi1000'

    def get_parameter_definitions(self):
        return [
            {'key': 'breakout_window', 'label': '突破窗口', 'type': 'timing', 'min': 5, 'max': 60, 'step': 1, 'default': self.breakout_window, 'unit': '日', 'description': '用于判定价格向上突破的历史窗口。'},
            {'key': 'exit_window', 'label': '退出窗口', 'type': 'timing', 'min': 3, 'max': 30, 'step': 1, 'default': self.exit_window, 'unit': '日', 'description': '用于判定跌破防线的历史窗口。'},
            {'key': 'trend_window', 'label': '趋势均线', 'type': 'timing', 'min': 20, 'max': 120, 'step': 1, 'default': self.trend_window, 'unit': '日', 'description': '用于确认中期趋势方向的均线窗口。'},
        ]

    def get_principle_summary(self):
        return 'CSI1000 先看价格是否重新突破近期高点并站上趋势线，再用 MACD 金叉辅助补充入场；一旦跌破退出位或趋势线，则快速收缩风险敞口。'

    def get_formula_blocks(self):
        return [
            {
                'title': '突破入场',
                'expression': 'buyBreakout_t = (close_t > breakoutHigh_t) \land (close_t > trendMA_t)',
                'explanation': '价格重新突破过去 breakout_window 的高点，且仍位于趋势线上方时，允许开始建仓。',
            },
            {
                'title': 'MACD 辅助入场',
                'expression': 'macdCross_t = (MACD_t > Signal_t) \land (MACD_{t-1} \le Signal_{t-1}) \land (close_t > trendMA_t)',
                'explanation': '即使尚未创出新高，只要 MACD 金叉且趋势未坏，也允许试探跟随反弹。',
            },
            {
                'title': '退出条件',
                'expression': 'sell_t = (close_t < exitLow_t) \lor (close_t < trendMA_t)',
                'explanation': '一旦跌破短期退出位或中期趋势线，状态机立即切回空仓，避免继续硬扛回撤。',
            },
            {
                'title': '强度评分',
                'expression': 'raw_t = 10(close_t/breakoutHigh_t-1) + 8(close_t/trendMA_t-1) + 4(MACD_t-Signal_t)',
                'explanation': '突破幅度、相对趋势线距离与 MACD 强弱共同决定 staged 模式下的加减仓速度。',
            },
        ]

    def compute_indicators(self, df):
        close_col = 'csi1000_close'
        high_col = 'csi1000_high'
        low_col = 'csi1000_low'
        out = df[['交易日期', close_col, high_col, low_col]].dropna().copy()
        out['close'] = out[close_col]
        out['high'] = out[high_col]
        out['low'] = out[low_col]
        out['trend_ma'] = out['close'].rolling(self.trend_window).mean()
        out['breakout_high'] = out['high'].rolling(self.breakout_window).max().shift(1)
        out['exit_low'] = out['low'].rolling(self.exit_window).min().shift(1)
        macd_line, signal_line = _calc_macd(out['close'])
        out['macd_line'] = macd_line
        out['macd_signal'] = signal_line
        return out

    def build_signal_reason(self, row):
        details = [
            f"收盘价 {row['close']:.2f}",
            f"突破高点 {row['breakout_high']:.2f}" if pd.notna(row['breakout_high']) else '突破位未就绪',
            f"退出低点 {row['exit_low']:.2f}" if pd.notna(row['exit_low']) else '退出位未就绪',
            f"趋势均线 {row['trend_ma']:.2f}" if pd.notna(row['trend_ma']) else '趋势均线未就绪',
            f"MACD {row['macd_line']:.4f} / 信号线 {row['macd_signal']:.4f}" if pd.notna(row.get('macd_line')) else '',
            f"目标仓位 {float(row.get('target_exposure', 0) or 0) * 100:.0f}%",
        ]
        details = [d for d in details if d]
        rebalance_action = row.get('rebalance_action', 'flat')
        if rebalance_action == 'add':
            summary = 'CSI1000 突破延续且趋势增强，执行加仓。'
        elif rebalance_action == 'trim':
            summary = 'CSI1000 强度回落但趋势未完全破坏，执行减仓。'
        elif row['position'] == 1:
            summary = 'CSI1000 重新站上趋势线并保持突破结构，继续持有。'
        else:
            summary = 'CSI1000 未满足突破或趋势条件，保持空仓。'
        return summary, details

    def generate_signals(self, df):
        df = df.copy()
        close = df['close']
        buy_breakout = (close > df['breakout_high']) & (close > df['trend_ma'])
        macd_cross = (
            (df['macd_line'] > df['macd_signal']) &
            (df['macd_line'].shift(1) <= df['macd_signal'].shift(1)) &
            (close > df['trend_ma'])
        )
        buy_cond = buy_breakout | macd_cross
        sell_cond = (close < df['exit_low']) | (close < df['trend_ma'])

        pos = np.zeros(len(df), dtype=int)
        cur = 0
        for i in range(len(df)):
            bc = bool(buy_cond.iloc[i]) if pd.notna(buy_cond.iloc[i]) else False
            sc = bool(sell_cond.iloc[i]) if pd.notna(sell_cond.iloc[i]) else False
            if cur == 0 and bc:
                cur = 1
            elif cur == 1 and sc:
                cur = 0
            pos[i] = cur

        raw_score = (
            ((close / df['breakout_high']) - 1).replace([np.inf, -np.inf], 0).fillna(0) * 10 +
            ((close / df['trend_ma']) - 1).fillna(0) * 8 +
            (df['macd_line'] - df['macd_signal']).fillna(0) * 4
        )
        df['signal_score'] = raw_score / 3
        df['strength_score'] = _normalize_score(raw_score)
        ready_mask = df[['breakout_high', 'exit_low', 'trend_ma', 'macd_line', 'macd_signal']].notna().all(axis=1)
        df = self._apply_exposure_columns(df, pd.Series(pos, index=df.index), staged_strength=df['strength_score'], ready_mask=ready_mask, price_series=df['close'])
        df['index_id'] = self.get_index_id()
        df['index_name'] = self.get_index_name()
        reasons = df.apply(self.build_signal_reason, axis=1)
        df['reason_summary'] = reasons.map(lambda x: x[0])
        df['reason_detail'] = reasons.map(lambda x: x[1])
        return df


class Star50TimingStrategy(BaseTimingStrategy):
    strategy_id = 'star50_timing'
    display_name = '科创50 择时策略'
    strategy_description = '基于突破确认、趋势过滤与 MACD 辅助信号判断科创50的买卖时机。'

    def __init__(self, breakout_window=10, exit_window=5, trend_window=40,
                 initial_capital=50000, buy_cost=0.001, sell_cost=0.001,
                 exposure_mode='staged', enter_threshold=0.55, add_threshold=0.75,
                 trim_threshold=0.35, exit_threshold=0.15, confirm_days=1,
                 max_entry_exposure=0.5, probe_entry_exposure=0.25,
                 probe_confirm_days=1, profit_lock_enabled=False,
                 profit_lock_drawdown=0.04, profit_lock_level_1=0.10,
                 profit_lock_level_2=0.18, profit_lock_level_3=0.28):
        super().__init__(
            initial_capital=initial_capital,
            buy_cost=buy_cost,
            sell_cost=sell_cost,
            exposure_mode=exposure_mode,
            enter_threshold=enter_threshold,
            add_threshold=add_threshold,
            trim_threshold=trim_threshold,
            exit_threshold=exit_threshold,
            confirm_days=confirm_days,
            max_entry_exposure=max_entry_exposure,
            probe_entry_exposure=probe_entry_exposure,
            probe_confirm_days=probe_confirm_days,
            profit_lock_enabled=profit_lock_enabled,
            profit_lock_drawdown=profit_lock_drawdown,
            profit_lock_level_1=profit_lock_level_1,
            profit_lock_level_2=profit_lock_level_2,
            profit_lock_level_3=profit_lock_level_3,
        )
        self.breakout_window = int(breakout_window)
        self.exit_window = int(exit_window)
        self.trend_window = int(trend_window)

    def get_index_id(self):
        return 'star50'

    def get_parameter_definitions(self):
        return [
            {'key': 'breakout_window', 'label': '突破窗口', 'type': 'timing', 'min': 5, 'max': 80, 'step': 1, 'default': self.breakout_window, 'unit': '日', 'description': '用于判定价格向上突破的历史窗口。'},
            {'key': 'exit_window', 'label': '退出窗口', 'type': 'timing', 'min': 3, 'max': 40, 'step': 1, 'default': self.exit_window, 'unit': '日', 'description': '用于判定跌破防线的历史窗口。'},
            {'key': 'trend_window', 'label': '趋势均线', 'type': 'timing', 'min': 20, 'max': 180, 'step': 1, 'default': self.trend_window, 'unit': '日', 'description': '用于确认趋势方向的均线窗口。'},
        ]

    def get_principle_summary(self):
        return '科创50采用更接近交易系统的状态机：先看趋势与突破，再用 MACD 金叉辅助触发入场；一旦跌破退出位或趋势线则离场。'

    def get_formula_blocks(self):
        return [
            {
                'title': '突破入场',
                'expression': 'buy_breakout_t = (close_t > breakoutHigh_t) \land (close_t > trendMA_t)',
                'explanation': '价格向上突破过去 breakout_window 的高点，且仍位于趋势线上方。',
            },
            {
                'title': 'MACD 辅助入场',
                'expression': 'macdCross_t = (MACD_t > Signal_t) \land (MACD_{t-1} \le Signal_{t-1}) \land (close_t > trendMA_t)',
                'explanation': '即使没有创突破新高，只要 MACD 金叉且趋势未坏，也允许入场。',
            },
            {
                'title': '退出条件',
                'expression': 'sell_t = (close_t < exitLow_t) \lor (close_t < trendMA_t)',
                'explanation': '跌破短期退出位或中期趋势线时，状态机从持仓切回空仓。',
            },
            {
                'title': '强度评分',
                'expression': 'raw_t = 10(close_t/breakoutHigh_t-1) + 8(close_t/trendMA_t-1) + 4(MACD_t-Signal_t)',
                'explanation': '突破幅度、趋势距离与 MACD 强弱共同决定 staged 仓位提升速度。',
            },
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
        macd_line, signal_line = _calc_macd(out['close'])
        out['macd_line'] = macd_line
        out['macd_signal'] = signal_line
        return out

    def build_signal_reason(self, row):
        details = [
            f"收盘价 {row['close']:.2f}",
            f"突破高点 {row['breakout_high']:.2f}" if pd.notna(row['breakout_high']) else '突破位未就绪',
            f"退出低点 {row['exit_low']:.2f}" if pd.notna(row['exit_low']) else '退出位未就绪',
            f"趋势均线 {row['trend_ma']:.2f}" if pd.notna(row['trend_ma']) else '趋势均线未就绪',
            f"MACD {row['macd_line']:.4f} / 信号线 {row['macd_signal']:.4f}" if pd.notna(row.get('macd_line')) else '',
            f"目标仓位 {float(row.get('target_exposure', 0) or 0) * 100:.0f}%",
        ]
        details = [d for d in details if d]
        rebalance_action = row.get('rebalance_action', 'flat')
        if rebalance_action == 'add':
            summary = '科创50 趋势增强，执行加仓。'
        elif rebalance_action == 'trim':
            summary = '科创50 仍在趋势中但强度回落，执行减仓。'
        elif row['position'] == 1:
            summary = '科创50 突破确认且站上趋势线，保持持有。'
        else:
            summary = '科创50 未满足突破确认条件，保持空仓。'
        return summary, details

    def generate_signals(self, df):
        df = df.copy()
        close = df['close']
        buy_breakout = (close > df['breakout_high']) & (close > df['trend_ma'])
        macd_cross = (
            (df['macd_line'] > df['macd_signal']) &
            (df['macd_line'].shift(1) <= df['macd_signal'].shift(1)) &
            (close > df['trend_ma'])
        )
        buy_cond = buy_breakout | macd_cross
        sell_cond = (close < df['exit_low']) | (close < df['trend_ma'])

        pos = np.zeros(len(df), dtype=int)
        cur = 0
        for i in range(len(df)):
            bc = bool(buy_cond.iloc[i]) if pd.notna(buy_cond.iloc[i]) else False
            sc = bool(sell_cond.iloc[i]) if pd.notna(sell_cond.iloc[i]) else False
            if cur == 0 and bc:
                cur = 1
            elif cur == 1 and sc:
                cur = 0
            pos[i] = cur

        raw_score = (
            ((close / df['breakout_high']) - 1).replace([np.inf, -np.inf], 0).fillna(0) * 10 +
            ((close / df['trend_ma']) - 1).fillna(0) * 8 +
            (df['macd_line'] - df['macd_signal']).fillna(0) * 4
        )
        df['signal_score'] = raw_score / 3
        df['strength_score'] = _normalize_score(raw_score)
        ready_mask = df[['breakout_high', 'exit_low', 'trend_ma', 'macd_line', 'macd_signal']].notna().all(axis=1)
        df = self._apply_exposure_columns(df, pd.Series(pos, index=df.index), staged_strength=df['strength_score'], ready_mask=ready_mask, price_series=df['close'])
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

    def __init__(self, momentum_short_window=15, momentum_long_window=40, trend_window=40,
                 momentum_threshold=0.0, initial_capital=50000, buy_cost=0.001, sell_cost=0.001,
                 exposure_mode='binary', enter_threshold=0.55, add_threshold=0.75,
                 trim_threshold=0.35, exit_threshold=0.15, confirm_days=1,
                 max_entry_exposure=0.5, probe_entry_exposure=0.25,
                 probe_confirm_days=1, profit_lock_enabled=False,
                 profit_lock_drawdown=0.04, profit_lock_level_1=0.10,
                 profit_lock_level_2=0.18, profit_lock_level_3=0.28):
        super().__init__(
            initial_capital=initial_capital,
            buy_cost=buy_cost,
            sell_cost=sell_cost,
            exposure_mode=exposure_mode,
            enter_threshold=enter_threshold,
            add_threshold=add_threshold,
            trim_threshold=trim_threshold,
            exit_threshold=exit_threshold,
            confirm_days=confirm_days,
            max_entry_exposure=max_entry_exposure,
            probe_entry_exposure=probe_entry_exposure,
            probe_confirm_days=probe_confirm_days,
            profit_lock_enabled=profit_lock_enabled,
            profit_lock_drawdown=profit_lock_drawdown,
            profit_lock_level_1=profit_lock_level_1,
            profit_lock_level_2=profit_lock_level_2,
            profit_lock_level_3=profit_lock_level_3,
        )
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

    def get_principle_summary(self):
        return '创业板策略关注短中期动量是否同步为正，并要求价格位于趋势均线上方；在 staged 模式下，再由强度分数控制分批建仓和减仓。'

    def get_formula_blocks(self):
        return [
            {
                'title': '基础入场条件',
                'expression': 'momShort_t > \theta,  momLong_t > 0,  close_t > trendMA_t',
                'explanation': '短动量超过阈值、中动量为正且价格仍在趋势线上方时，认为创业板进入可做多区间。',
            },
            {
                'title': '动量与趋势定义',
                'expression': 'momShort_t = close_t / close_{t-s} - 1,  momLong_t = close_t / close_{t-l} - 1,  trendMA_t = mean(close_{t-w+1:t})',
                'explanation': '分别刻画短周期、中周期收益率以及趋势均线位置。',
            },
            {
                'title': '强度评分',
                'expression': 'raw_t = 6 momShort_t + 4 momLong_t + 8(close_t/trendMA_t-1)',
                'explanation': '短动量权重最高，其次是长动量和价格相对趋势线的偏离度。',
            },
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
            f"目标仓位 {float(row.get('target_exposure', 0) or 0) * 100:.0f}%",
        ]
        rebalance_action = row.get('rebalance_action', 'flat')
        if rebalance_action == 'add':
            summary = '创业板动量继续增强，执行加仓。'
        elif rebalance_action == 'trim':
            summary = '创业板动量回落，执行减仓。'
        elif row['position'] == 1:
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
        binary_position = buy_cond.fillna(False).astype(int)
        raw_score = (
            df['momentum_short'].fillna(0) * 6 +
            df['momentum_long'].fillna(0) * 4 +
            ((df['close'] / df['trend_ma']) - 1).fillna(0) * 8
        )
        df['signal_score'] = raw_score / 3
        df['strength_score'] = _normalize_score(raw_score)
        ready_mask = df[['momentum_short', 'momentum_long', 'trend_ma']].notna().all(axis=1)
        df = self._apply_exposure_columns(df, binary_position, staged_strength=df['strength_score'], ready_mask=ready_mask, price_series=df['close'])
        df['index_id'] = self.get_index_id()
        df['index_name'] = self.get_index_name()
        reasons = df.apply(self.build_signal_reason, axis=1)
        df['reason_summary'] = reasons.map(lambda x: x[0])
        df['reason_detail'] = reasons.map(lambda x: x[1])
        return df


class NasdaqTimingStrategy(BaseTimingStrategy):
    strategy_id = 'nasdaq_timing'
    display_name = '纳指ETF 择时策略'
    strategy_description = '基于均线趋势与中期动量判断纳指ETF的买卖时机，适配美股趋势更持久的特点。'

    def __init__(self, fast_window=20, slow_window=120, momentum_window=120,
                 initial_capital=50000, buy_cost=0.001, sell_cost=0.001,
                 exposure_mode='binary', enter_threshold=0.55, add_threshold=0.75,
                 trim_threshold=0.35, exit_threshold=0.15, confirm_days=2,
                 max_entry_exposure=0.5):
        super().__init__(
            initial_capital=initial_capital,
            buy_cost=buy_cost,
            sell_cost=sell_cost,
            exposure_mode=exposure_mode,
            enter_threshold=enter_threshold,
            add_threshold=add_threshold,
            trim_threshold=trim_threshold,
            exit_threshold=exit_threshold,
            confirm_days=confirm_days,
            max_entry_exposure=max_entry_exposure,
        )
        self.fast_window = int(fast_window)
        self.slow_window = int(slow_window)
        self.momentum_window = int(momentum_window)

    def get_index_id(self):
        return 'nasdaq'

    def get_parameter_definitions(self):
        return [
            {'key': 'fast_window', 'label': '快均线窗口', 'type': 'timing', 'min': 5, 'max': 60, 'step': 1, 'default': self.fast_window, 'unit': '日', 'description': '短期趋势均线，美股趋势持久，建议 ≥ 20 日。'},
            {'key': 'slow_window', 'label': '慢均线窗口', 'type': 'timing', 'min': 60, 'max': 250, 'step': 5, 'default': self.slow_window, 'unit': '日', 'description': '中长期趋势均线，纳指可用 120 日（约半年）。'},
            {'key': 'momentum_window', 'label': '动量窗口', 'type': 'timing', 'min': 60, 'max': 250, 'step': 5, 'default': self.momentum_window, 'unit': '日', 'description': '中长期动量确认窗口，过短容易被噪音误导。'},
        ]

    def get_principle_summary(self):
        return '纳指ETF采用均线多头排列 + 中长期正动量作为持仓信号，美股趋势通常比A股更持久，使用更长的均线和动量窗口以减少频繁交易。'

    def get_formula_blocks(self):
        return [
            {'title': '均线多头入场', 'expression': 'close_t > MA_f(t),  MA_f(t) > MA_s(t),  mom_t > 0', 'explanation': '价格站上快线、快线站上慢线、中长期动量为正，三者同时满足才入场。'},
            {'title': '均线与动量定义', 'expression': 'MA_f = mean(close_{t-f+1:t}),  MA_s = mean(close_{t-s+1:t}),  mom_t = close_t/close_{t-m} - 1', 'explanation': '默认使用 20 日快线、120 日慢线、120 日动量，适合美股中长线持有。'},
            {'title': '强度评分', 'expression': 'raw_t = 8(close_t/MA_f-1) + 10(MA_f/MA_s-1) + 4 mom_t', 'explanation': '趋势越强、均线斜率越大、动量越高，staged 模式仓位信号越高。'},
        ]

    def compute_indicators(self, df):
        close_col = 'nasdaq_close'
        if close_col not in df.columns:
            return pd.DataFrame(columns=['交易日期', 'close', 'ma_fast', 'ma_slow', 'momentum_long'])
        out = df[['交易日期', close_col]].dropna().copy()
        out['close'] = out[close_col]
        out['ma_fast'] = out['close'].rolling(self.fast_window).mean()
        out['ma_slow'] = out['close'].rolling(self.slow_window).mean()
        out['momentum_long'] = out['close'].pct_change(self.momentum_window)
        return out

    def build_signal_reason(self, row):
        details = [
            f"收盘价 {row['close']:.4f}",
            f"快线 {row['ma_fast']:.4f}" if pd.notna(row['ma_fast']) else '快线未就绪',
            f"慢线 {row['ma_slow']:.4f}" if pd.notna(row['ma_slow']) else '慢线未就绪',
            f"中期动量 {row['momentum_long'] * 100:.2f}%" if pd.notna(row['momentum_long']) else '动量未就绪',
            f"目标仓位 {float(row.get('target_exposure', 0) or 0) * 100:.0f}%",
        ]
        rebalance_action = row.get('rebalance_action', 'flat')
        if rebalance_action == 'add':
            summary = '纳指ETF趋势持续增强，执行加仓。'
        elif rebalance_action == 'trim':
            summary = '纳指ETF趋势转弱但未破坏，执行减仓。'
        elif row['position'] == 1:
            summary = '纳指ETF满足均线多头与动量条件，保持持有。'
        else:
            summary = '纳指ETF未满足入场条件，保持空仓。'
        return summary, details

    def generate_signals(self, df):
        df = df.copy()
        buy_cond = (
            (df['close'] > df['ma_fast']) &
            (df['ma_fast'] > df['ma_slow']) &
            (df['momentum_long'] > 0)
        )
        binary_position = buy_cond.fillna(False).astype(int)
        raw_score = (
            ((df['close'] / df['ma_fast']) - 1).fillna(0) * 8 +
            ((df['ma_fast'] / df['ma_slow']) - 1).fillna(0) * 10 +
            df['momentum_long'].fillna(0) * 4
        )
        df['signal_score'] = raw_score / 3
        df['strength_score'] = _normalize_score(raw_score)
        ready_mask = df[['ma_fast', 'ma_slow', 'momentum_long']].notna().all(axis=1)
        df = self._apply_exposure_columns(df, binary_position, staged_strength=df['strength_score'], ready_mask=ready_mask, price_series=df['close'])
        df['index_id'] = self.get_index_id()
        df['index_name'] = self.get_index_name()
        reasons = df.apply(self.build_signal_reason, axis=1)
        df['reason_summary'] = reasons.map(lambda x: x[0])
        df['reason_detail'] = reasons.map(lambda x: x[1])
        return df


class SP500TimingStrategy(BaseTimingStrategy):
    strategy_id = 'sp500_timing'
    display_name = '标普500ETF 择时策略'
    strategy_description = '基于均线趋势与中期动量判断标普500ETF的买卖时机，标普500波动率低于纳指，适合更长持有周期。'

    def __init__(self, fast_window=20, slow_window=125, momentum_window=100,
                 initial_capital=50000, buy_cost=0.001, sell_cost=0.001,
                 exposure_mode='staged', enter_threshold=0.5, add_threshold=0.72,
                 trim_threshold=0.32, exit_threshold=0.14, confirm_days=2,
                 max_entry_exposure=0.5):
        super().__init__(
            initial_capital=initial_capital,
            buy_cost=buy_cost,
            sell_cost=sell_cost,
            exposure_mode=exposure_mode,
            enter_threshold=enter_threshold,
            add_threshold=add_threshold,
            trim_threshold=trim_threshold,
            exit_threshold=exit_threshold,
            confirm_days=confirm_days,
            max_entry_exposure=max_entry_exposure,
        )
        self.fast_window = int(fast_window)
        self.slow_window = int(slow_window)
        self.momentum_window = int(momentum_window)

    def get_index_id(self):
        return 'sp500'

    def get_parameter_definitions(self):
        return [
            {'key': 'fast_window', 'label': '快均线窗口', 'type': 'timing', 'min': 10, 'max': 80, 'step': 5, 'default': self.fast_window, 'unit': '日', 'description': '标普500趋势恢复通常比纳指更平缓，快线下调到 20 日以更早识别反弹。'},
            {'key': 'slow_window', 'label': '慢均线窗口', 'type': 'timing', 'min': 80, 'max': 300, 'step': 5, 'default': self.slow_window, 'unit': '日', 'description': '慢线从 150 日缩短到 125 日，减少反弹后重新进场过晚的问题。'},
            {'key': 'momentum_window', 'label': '动量窗口', 'type': 'timing', 'min': 80, 'max': 300, 'step': 5, 'default': self.momentum_window, 'unit': '日', 'description': '动量窗口缩短到 100 日，保留中期趋势判断但提升近期修复阶段的响应速度。'},
        ]

    def get_principle_summary(self):
        return '标普500ETF保留均线 + 中期动量主框架，但把默认仓位模式切换为 staged，并适度缩短窗口，以减少反弹后入场过慢和 binary 满进满出的噪音。'

    def get_formula_blocks(self):
        return [
            {'title': '均线多头入场', 'expression': 'close_t > MA_f(t),  MA_f(t) > MA_s(t),  mom_t > 0', 'explanation': '价格、快慢均线与中期动量三重确认；保留原逻辑，但配合 staged 仓位减少一次性满仓切入。'},
            {'title': '均线与动量定义', 'expression': 'MA_f = mean(close_{t-f+1:t}),  MA_s = mean(close_{t-s+1:t}),  mom_t = close_t/close_{t-m} - 1', 'explanation': '默认调整为 20 日快线、125 日慢线、100 日动量，在不脱离长趋势框架的前提下提升修复阶段响应速度。'},
            {'title': '强度评分', 'expression': 'raw_t = 8(close_t/MA_f-1) + 10(MA_f/MA_s-1) + 4 mom_t', 'explanation': '信号强度继续用于 staged 渐进建仓/减仓，目标是降低 late-2025 到 early-2026 一类来回打脸。'},
        ]

    def compute_indicators(self, df):
        close_col = 'sp500_close'
        if close_col not in df.columns:
            return pd.DataFrame(columns=['交易日期', 'close', 'ma_fast', 'ma_slow', 'momentum_long'])
        out = df[['交易日期', close_col]].dropna().copy()
        out['close'] = out[close_col]
        out['ma_fast'] = out['close'].rolling(self.fast_window).mean()
        out['ma_slow'] = out['close'].rolling(self.slow_window).mean()
        out['momentum_long'] = out['close'].pct_change(self.momentum_window)
        return out

    def build_signal_reason(self, row):
        details = [
            f"收盘价 {row['close']:.4f}",
            f"快线 {row['ma_fast']:.4f}" if pd.notna(row['ma_fast']) else '快线未就绪',
            f"慢线 {row['ma_slow']:.4f}" if pd.notna(row['ma_slow']) else '慢线未就绪',
            f"中期动量 {row['momentum_long'] * 100:.2f}%" if pd.notna(row['momentum_long']) else '动量未就绪',
            f"目标仓位 {float(row.get('target_exposure', 0) or 0) * 100:.0f}%",
        ]
        rebalance_action = row.get('rebalance_action', 'flat')
        if rebalance_action == 'add':
            summary = '标普500ETF趋势持续增强，执行加仓。'
        elif rebalance_action == 'trim':
            summary = '标普500ETF趋势转弱但未破坏，执行减仓。'
        elif row['position'] == 1:
            summary = '标普500ETF满足均线多头与动量条件，保持持有。'
        else:
            summary = '标普500ETF未满足入场条件，保持空仓。'
        return summary, details

    def generate_signals(self, df):
        df = df.copy()
        buy_cond = (
            (df['close'] > df['ma_fast']) &
            (df['ma_fast'] > df['ma_slow']) &
            (df['momentum_long'] > 0)
        )
        binary_position = buy_cond.fillna(False).astype(int)
        raw_score = (
            ((df['close'] / df['ma_fast']) - 1).fillna(0) * 8 +
            ((df['ma_fast'] / df['ma_slow']) - 1).fillna(0) * 10 +
            df['momentum_long'].fillna(0) * 4
        )
        df['signal_score'] = raw_score / 3
        df['strength_score'] = _normalize_score(raw_score)
        ready_mask = df[['ma_fast', 'ma_slow', 'momentum_long']].notna().all(axis=1)
        df = self._apply_exposure_columns(df, binary_position, staged_strength=df['strength_score'], ready_mask=ready_mask, price_series=df['close'])
        df['index_id'] = self.get_index_id()
        df['index_name'] = self.get_index_name()
        reasons = df.apply(self.build_signal_reason, axis=1)
        df['reason_summary'] = reasons.map(lambda x: x[0])
        df['reason_detail'] = reasons.map(lambda x: x[1])
        return df


class MacroV32TimingStrategy(BaseTimingStrategy):
    """v3.2 美股宏观多因子择时 — sigmoid 网格最优 (k=1.5, lev=1.4, base=0.5)。"""

    strategy_id = 'macro_v32_timing'
    display_name = '纳指宏观多因子 v3.2 (Macro Sigmoid)'
    strategy_description = (
        '使用 8 个 FRED 宏观因子 (Fed Funds / 收益率曲线 / 核心CPI / 失业率 / 趋势 / 动量 / VIX / 高收益利差) '
        '构建 ContScore，经 sigmoid 平滑得到目标仓位；网格搜索鲁棒最优参数 (k=1.5, lev=1.4, base=0.5)，'
        'OOS Test 2019-2026 Sharpe 0.848 vs BH 0.794，MaxDD -19.8% vs BH -35.6%。'
    )

    def __init__(self, initial_capital=50000, buy_cost=0.001, sell_cost=0.001,
                 exposure_mode='staged', enter_threshold=0.55, add_threshold=0.75,
                 trim_threshold=0.35, exit_threshold=0.15, confirm_days=1,
                 max_entry_exposure=1.0,
                 sigmoid_k=1.5, max_leverage=1.4, base_position=0.5,
                 inertia=0.03, crisis_vix=35.0):
        super().__init__(
            initial_capital=initial_capital,
            buy_cost=buy_cost,
            sell_cost=sell_cost,
            exposure_mode=exposure_mode,
            enter_threshold=enter_threshold,
            add_threshold=add_threshold,
            trim_threshold=trim_threshold,
            exit_threshold=exit_threshold,
            confirm_days=confirm_days,
            max_entry_exposure=max_entry_exposure,
        )
        self.sigmoid_k = float(sigmoid_k)
        self.max_leverage = float(max_leverage)
        self.base_position = float(base_position)
        self.inertia = float(inertia)
        self.crisis_vix = float(crisis_vix)

    def get_index_id(self):
        return 'nasdaq'

    def get_parameter_definitions(self):
        return [
            {'key': 'sigmoid_k', 'label': 'Sigmoid 斜率 k', 'type': 'timing', 'min': 1.0, 'max': 3.0, 'step': 0.1, 'default': self.sigmoid_k, 'unit': '', 'description': '越大越接近阶梯映射；v3.2 网格最优 k=1.5（较缓）。'},
            {'key': 'max_leverage', 'label': '杠杆上限', 'type': 'timing', 'min': 1.0, 'max': 2.0, 'step': 0.1, 'default': self.max_leverage, 'unit': '倍', 'description': 'ContScore 极强时允许的最大仓位（前端展示截断至 100%）。'},
            {'key': 'base_position', 'label': '中性仓位中心', 'type': 'timing', 'min': 0.3, 'max': 0.7, 'step': 0.05, 'default': self.base_position, 'unit': '', 'description': 'ContScore=0 时对应的中性仓位水平。'},
            {'key': 'inertia', 'label': '仓位惯性阈值', 'type': 'timing', 'min': 0.0, 'max': 0.1, 'step': 0.01, 'default': self.inertia, 'unit': '', 'description': '月度调仓变化幅度低于该阈值时不动作，降低噪音换手。'},
            {'key': 'crisis_vix', 'label': '危机 VIX 阈值', 'type': 'timing', 'min': 25.0, 'max': 50.0, 'step': 1.0, 'default': self.crisis_vix, 'unit': '', 'description': 'VIX 超过该值时强制 ContScore = -2 触发减仓。'},
        ]

    def get_principle_summary(self):
        return ('以 4 类宏观因子 (货币/流动性/通胀/经济) + 4 类市场因子 (趋势/动量/波动率/信用) 滚动 5 年 z-score '
                '平均得 ContScore，sigmoid 映射成 0-100% 仓位；VIX > 35 危机覆盖，月度调仓 + 3% 惯性。')

    def get_formula_blocks(self):
        return [
            {
                'title': '8 因子构成',
                'expression': 'Z1=Money, Z2=Liquidity, Z3=Inflation, Z4=Economy, Z5a=Trend, Z5b=Mom, Z5c=VIX, Z5d=Credit',
                'explanation': 'Fed Funds 3M-12M、收益率曲线斜率/变化、核心CPI 同比/3M MoM、Sahm 失业率、200MA 价比、20/60MA 比、VIX、高收益利差。',
            },
            {
                'title': 'ContScore 综合分',
                'expression': 'ContScore = mean(Z1..Z5d),  if VIX > crisis_vix: ContScore = -2',
                'explanation': '8 因子等权平均，每个因子滚动 5 年 z-score 标准化并截断在 [-3, +3]，VIX 危机区强制覆盖。',
            },
            {
                'title': 'Sigmoid 仓位映射',
                'expression': 'Position = clip( sigmoid( k · (ContScore - (base - 0.5)) ) · max_lev , 0, 1)',
                'explanation': 'ContScore = 0 对应仓位 ≈ base × max_lev；k=1.5 控制斜率平缓；前端展示截断到 100%。',
            },
            {
                'title': '月度调仓 + 惯性',
                'expression': 'pos_m = M-end last;  Δpos < inertia → 维持上月仓位',
                'explanation': '日频信号月末重采样，避免日内噪音；变化幅度 < 3% 不调仓以降低换手。',
            },
        ]

    def _build_macro_panel(self, date_index):
        codes = [
            ('FedFundsRate', 'FedFunds'),
            ('YieldCurve_10Y2Y', 'YC'),
            ('CPI_core', 'CPI'),
            ('Unemployment', 'Unemp'),
            ('VIX', 'VIX'),
            ('HighYieldSpread', 'HYS'),
            ('Treasury10Y', 'T10Y'),
        ]
        full_idx = pd.date_range(date_index.min(), date_index.max(), freq='D')
        panel = pd.DataFrame(index=full_idx)
        for code, col in codes:
            s = _load_fred_series(code, col)
            if len(s) == 0:
                panel[col] = np.nan
                continue
            panel[col] = s.reindex(full_idx, method='ffill')
        panel['HYS_proxy'] = panel['HYS'].fillna(panel['VIX'] / 5.0)
        return panel

    def _compute_factors(self, ndx, macro):
        f = pd.DataFrame(index=macro.index)
        ff_diff = macro['FedFunds'].rolling(90).mean() - macro['FedFunds'].rolling(365).mean()
        f['Z1_Money'] = -_zscore_rolling(ff_diff)
        yc_lvl = _zscore_rolling(macro['YC'])
        yc_chg = _zscore_rolling(macro['YC'].diff(60))
        f['Z2_Liquidity'] = (yc_lvl + yc_chg) / 2
        cpi_yoy = (macro['CPI'] / macro['CPI'].shift(365) - 1) * 100
        f['Z3_Inflation'] = (-_zscore_rolling(cpi_yoy.diff(90)) - _zscore_rolling(cpi_yoy.clip(lower=2.0))) / 2
        sahm = macro['Unemp'].rolling(90).mean() - macro['Unemp'].rolling(365).min()
        f['Z4_Economy'] = -_zscore_rolling(sahm)
        ndx_daily = ndx.reindex(macro.index).ffill()
        ma200 = ndx_daily.rolling(200).mean()
        f['Z5a_Trend'] = _zscore_rolling((ndx_daily / ma200 - 1) * 100)
        f['Z5b_Mom'] = _zscore_rolling((ndx_daily.rolling(20).mean() / ndx_daily.rolling(60).mean() - 1) * 100)
        f['Z5c_VIX'] = -_zscore_rolling(macro['VIX'])
        f['Z5d_Credit'] = -_zscore_rolling(macro['HYS_proxy'])
        z_cols = [c for c in f.columns if c.startswith('Z')]
        f['ContScore'] = f[z_cols].mean(axis=1)
        crisis = macro['VIX'] > self.crisis_vix
        f.loc[crisis, 'ContScore'] = -2.0
        return f

    def _sigmoid_position(self, score):
        if pd.isna(score):
            return np.nan
        sig = 1.0 / (1.0 + np.exp(-self.sigmoid_k * (score - (self.base_position - 0.5))))
        pos = sig * self.max_leverage
        return float(max(0.0, min(self.max_leverage, pos)))

    def _apply_inertia(self, series):
        out = series.copy()
        if out.empty:
            return out
        last = out.iloc[0]
        for i in range(1, len(out)):
            v = out.iloc[i]
            if pd.isna(v) or pd.isna(last):
                last = v
                continue
            if abs(v - last) < self.inertia:
                out.iloc[i] = last
            else:
                last = v
        return out

    def compute_indicators(self, df):
        close_col = 'nasdaq_close'
        if close_col not in df.columns:
            return pd.DataFrame(columns=['交易日期', 'close'])
        out = df[['交易日期', close_col]].dropna().copy()
        out['close'] = out[close_col]
        out = out.sort_values('交易日期').reset_index(drop=True)

        ndx_series = out.set_index('交易日期')['close']
        macro = self._build_macro_panel(out['交易日期'])
        factors = self._compute_factors(ndx_series, macro)

        raw_pos_daily = factors['ContScore'].apply(self._sigmoid_position)
        pos_monthly = raw_pos_daily.resample('M').last().ffill()
        pos_monthly = self._apply_inertia(pos_monthly)
        pos_daily = pos_monthly.reindex(factors.index, method='ffill').fillna(0.0)

        out['cont_score'] = factors['ContScore'].reindex(out['交易日期']).values
        out['z_money'] = factors['Z1_Money'].reindex(out['交易日期']).values
        out['z_liquidity'] = factors['Z2_Liquidity'].reindex(out['交易日期']).values
        out['z_inflation'] = factors['Z3_Inflation'].reindex(out['交易日期']).values
        out['z_economy'] = factors['Z4_Economy'].reindex(out['交易日期']).values
        out['z_trend'] = factors['Z5a_Trend'].reindex(out['交易日期']).values
        out['z_momentum'] = factors['Z5b_Mom'].reindex(out['交易日期']).values
        out['z_vix'] = factors['Z5c_VIX'].reindex(out['交易日期']).values
        out['z_credit'] = factors['Z5d_Credit'].reindex(out['交易日期']).values
        out['macro_position'] = pos_daily.reindex(out['交易日期']).values
        return out

    def build_signal_reason(self, row):
        cs = row.get('cont_score', np.nan)
        pos = float(row.get('target_exposure', 0) or 0)
        zs = {
            '货币': row.get('z_money'), '流动性': row.get('z_liquidity'),
            '通胀': row.get('z_inflation'), '经济': row.get('z_economy'),
            '趋势': row.get('z_trend'), '动量': row.get('z_momentum'),
            'VIX': row.get('z_vix'), '信用': row.get('z_credit'),
        }
        details = [f"{k}: {v:+.2f}" for k, v in zs.items() if pd.notna(v)]
        details.append(f"ContScore {cs:+.2f}" if pd.notna(cs) else 'ContScore 未就绪')
        details.append(f"目标仓位 {pos*100:.0f}%")
        action = row.get('rebalance_action', 'flat')
        if pd.isna(cs):
            summary = '宏观因子尚未就绪，保持空仓。'
        elif cs <= -1.5:
            summary = f'宏观分数 {cs:+.2f} 极弱，触发危机保护减至最低仓位。'
        elif cs <= -0.3:
            summary = f'宏观分数 {cs:+.2f} 偏弱，降低风险敞口。'
        elif cs >= 0.6:
            summary = f'宏观分数 {cs:+.2f} 偏强，趋势 + 流动性共振，扩大仓位。'
        else:
            summary = f'宏观分数 {cs:+.2f} 中性区间，维持基准仓位。'
        if action == 'add':
            summary += ' 本月加仓。'
        elif action == 'trim':
            summary += ' 本月减仓。'
        elif action == 'enter':
            summary += ' 本月建仓。'
        elif action == 'exit':
            summary += ' 本月清仓。'
        return summary, details

    def generate_signals(self, df):
        df = df.copy()
        if 'macro_position' not in df.columns or len(df) == 0:
            df['target_exposure'] = 0.0
            df['prev_exposure'] = 0.0
            df['exposure_change'] = 0.0
            df['position'] = 0
            df['signal_action'] = 'flat'
            df['rebalance_action'] = 'flat'
            df['signal_score'] = 0.0
            df['strength_score'] = 0.5
            df['index_id'] = self.get_index_id()
            df['index_name'] = self.get_index_name()
            df['reason_summary'] = '数据未就绪'
            df['reason_detail'] = [['等待 NDX 与 FRED 数据加载']] * len(df)
            return df

        target = pd.Series(df['macro_position'].values, index=df.index).fillna(0.0).clip(lower=0.0, upper=1.0).round(4)
        prev = target.shift(1).fillna(0.0).round(4)
        change = (target - prev).round(4)

        sig_action = pd.Series('flat', index=df.index, dtype=object)
        sig_action.loc[(prev <= 0) & (target > 0)] = 'buy'
        sig_action.loc[(prev > 0) & (target <= 0)] = 'sell'
        sig_action.loc[(prev > 0) & (target > 0)] = 'hold'

        reb = pd.Series('flat', index=df.index, dtype=object)
        reb.loc[(prev <= 0) & (target > 0)] = 'enter'
        reb.loc[(prev > 0) & (target <= 0)] = 'exit'
        reb.loc[(prev > 0) & (target > prev)] = 'add'
        reb.loc[(prev > 0) & (target < prev) & (target > 0)] = 'trim'
        reb.loc[(prev > 0) & (target == prev)] = 'hold'

        df['target_exposure'] = target
        df['prev_exposure'] = prev
        df['exposure_change'] = change
        df['position'] = (target > 0).astype(int)
        df['signal_action'] = sig_action
        df['rebalance_action'] = reb
        df['signal_score'] = df['cont_score'].fillna(0.0)
        df['strength_score'] = target
        df['index_id'] = self.get_index_id()
        df['index_name'] = self.get_index_name()
        reasons = df.apply(self.build_signal_reason, axis=1)
        df['reason_summary'] = reasons.map(lambda x: x[0])
        df['reason_detail'] = reasons.map(lambda x: x[1])
        return df
