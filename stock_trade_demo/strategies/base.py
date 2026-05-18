"""
BaseStrategy — 所有选股策略的抽象基类。

定义策略的标准流水线：
  prepare_data → compute_factors → apply_filters → rank_stocks → 产出因子列

子类只需要实现 compute_factors、apply_filters、rank_stocks 三个方法，
或直接覆盖 run() 完全自定义流水线。
"""

import pandas as pd


def safe_float(series, default=0.0):
    """将 Series 转为 float，非法值填 default。用于清洗原始数据中的脏值。"""
    return pd.to_numeric(series, errors='coerce').fillna(default)


class BaseStrategy:
    """
    选股策略基类。

    流水线：
      1. prepare_data(df)   — 基础过滤（上市天数、北交所排除）+ 数值化关键列
      2. compute_factors(df) — 计算选股因子（子类实现）
      3. apply_filters(df)   — 策略专属过滤（子类实现）
      4. rank_stocks(df)     — 对股票排名，产出"因子"列用于选股（子类实现）
      5. run(df)             — 按顺序执行 1-4，返回处理后的 DataFrame

    参数:
      select_stock_num - 每期选股数量，默认 6
      c_rate           - 买入手续费率，默认万分之一（1.0/10000）
      t_rate           - 卖出印花税率，默认千分之一（1/1000）
      bull_tp          - 牛市止盈阈值，默认 30%
      bear_tp          - 熊市止盈阈值，默认 22%
      bull_n           - 牛市持仓上限，默认 6
      bear_n           - 熊市持仓上限，默认 4
      initial_capital  - 初始本金，默认 100,000
    """

    strategy_id = ''
    display_name = ''
    strategy_description = ''

    def __init__(self, select_stock_num=6, c_rate=1.0 / 10000,
                 t_rate=1 / 1000, bull_tp=0.30, bear_tp=0.22,
                 bull_n=6, bear_n=4, initial_capital=100000):
        self.select_stock_num = select_stock_num
        self.c_rate = c_rate
        self.t_rate = t_rate
        self.sell_cost = c_rate + t_rate
        self.bull_tp = bull_tp
        self.bear_tp = bear_tp
        self.bull_n = bull_n
        self.bear_n = bear_n
        self.initial_capital = initial_capital

    def get_display_name(self):
        return self.display_name or self.__class__.__name__

    def get_strategy_description(self):
        return self.strategy_description or self.__doc__ or ''

    def get_parameter_definitions(self):
        return []

    def get_filter_descriptions(self):
        return []

    def get_ranking_metadata(self):
        return {
            'name': '综合得分',
            'formula': '因子值越小越好',
            'direction': '升序（越小越好）',
            'description': '按因子升序选股。',
            'combination_method': 'custom',
            'normalization_method': '未声明',
            'components': [],
        }

    def get_factor_metadata(self):
        ranking = self.get_ranking_metadata() or {}
        ranking_factor = {
            'name': ranking.get('name', ''),
            'direction': ranking.get('direction', ''),
            'description': ranking.get('description', ''),
        }
        if ranking.get('weight_details'):
            ranking_factor['weight_details'] = ranking['weight_details']
        return {
            'id': self.strategy_id or self.__class__.__name__.lower(),
            'name': self.get_display_name(),
            'description': self.get_strategy_description(),
            'parameters': self.get_parameter_definitions(),
            'filters': self.get_filter_descriptions(),
            'ranking': ranking,
            'ranking_factor': ranking_factor,
        }

    def _format_rank(self, rank, total):
        if rank is None or total in (None, 0):
            return '排名信息缺失'
        return f'当期排名第 {int(rank)} / {int(total)}'

    def _market_cap_yi(self, row):
        market_cap = row.get('总市值')
        if pd.isna(market_cap):
            return None
        return round(float(market_cap) / 1e8, 2)

    def _build_fundamentals(self, row):
        fundamentals = []
        industry = str(row.get('新版申万二级行业名称', '') or '')
        if industry:
            fundamentals.append(f'细分行业: {industry}')

        market_cap_yi = self._market_cap_yi(row)
        if market_cap_yi is not None:
            fundamentals.append(f'总市值: {market_cap_yi} 亿')

        pe_inv = row.get('市盈率倒数')
        if pe_inv is not None and pd.notna(pe_inv) and float(pe_inv) != 0:
            fundamentals.append(f'PE: {round(1.0 / float(pe_inv), 2)}')

        pb_inv = row.get('市净率倒数')
        if pb_inv is not None and pd.notna(pb_inv) and float(pb_inv) != 0:
            fundamentals.append(f'PB: {round(1.0 / float(pb_inv), 2)}')

        return fundamentals

    def _factor_item(self, label, value, role='', note=''):
        return {
            'label': label,
            'value': value,
            'role': role,
            'note': note,
        }

    def build_selection_reason(self, row, rank, total):
        ranking = self.get_ranking_metadata()
        summary = f"按{ranking.get('name', '主排序因子')}入选，{self._format_rank(rank, total)}"
        details = [
            ranking.get('description', ''),
            f"排序公式: {ranking.get('formula', '因子值越小越好')}",
            f"标准化方式: {ranking.get('normalization_method', '未声明')}",
        ]
        details = [item for item in details if item]
        return {
            'summary': summary,
            'details': details,
            'fundamentals': self._build_fundamentals(row),
            'factor_breakdown': [],
        }

    # ── 基础数据准备 ───────────────────────────────────────────────

    def prepare_data(self, df):
        """
        基础数据清洗与过滤。

        两件事：
          1. 过滤：排除上市不满一年（交易天数 ≤ 250）的股票、排除北交所（代码含 'bj'）
          2. 数值化：将关键列转为 float，防止后续计算因脏数据报错
        """
        df = df[df['上市至今交易天数'] > 250]
        df = df[~df['股票代码'].str.contains('bj')]

        numeric_cols = [
            '总市值', 'bias_20', '成交额std_10', '市盈率倒数', '市净率倒数',
            '最高价', '最低价', '收盘价', 'MACD', 'DIF', 'DEA',
            '涨跌幅_20', '涨跌幅std_20', '成交额'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = safe_float(df[col])
        return df

    def compute_factors(self, df):
        raise NotImplementedError

    def apply_filters(self, df):
        raise NotImplementedError

    def rank_stocks(self, df):
        raise NotImplementedError

    def run(self, df):
        df = df.copy()
        df = self.prepare_data(df)
        df = self.compute_factors(df)
        df = self.apply_filters(df)
        df = self.rank_stocks(df)
        return df
