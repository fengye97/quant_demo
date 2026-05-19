import pandas as pd

from index_data import INDEX_CONFIGS


class BaseTimingStrategy:
    strategy_id = ''
    display_name = ''
    strategy_description = ''

    def __init__(self, initial_capital=100000, buy_cost=0.001, sell_cost=0.001, **params):
        self.initial_capital = initial_capital
        self.buy_cost = buy_cost
        self.sell_cost = sell_cost
        self._extra_params = params

    def get_display_name(self):
        return self.display_name or self.__class__.__name__

    def get_strategy_description(self):
        return self.strategy_description or self.__doc__ or ''

    def get_parameter_definitions(self):
        return []

    def get_signal_metadata(self):
        return {
            'id': self.strategy_id,
            'name': self.get_display_name(),
            'description': self.get_strategy_description(),
            'parameters': self.get_parameter_definitions(),
        }

    def get_index_id(self):
        raise NotImplementedError

    def get_index_name(self):
        index_id = self.get_index_id()
        return INDEX_CONFIGS.get(index_id, {}).get('name', index_id)

    def compute_indicators(self, df):
        raise NotImplementedError

    def build_signal_reason(self, row):
        raise NotImplementedError

    def generate_signals(self, df):
        raise NotImplementedError

    def run(self, panel_df):
        df = panel_df.copy()
        df = self.compute_indicators(df)
        df = self.generate_signals(df)
        if '交易日期' not in df.columns and 'date' in df.columns:
            df['交易日期'] = pd.to_datetime(df['date'])
        return df
