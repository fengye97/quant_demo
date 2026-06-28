"""
QualityValueStrategy — 质量价值小盘策略 v3.1

策略设计思路：
  - 不能放弃小市值（A股最强alpha来源）
  - 增加质量筛选，过滤跌破最惨的垃圾微盘
  - 增加价值倾斜，提供下行保护
  - 增加流动性过滤，保证实盘可行性

v3.0 → v3.1 调参记录（目标：控制最大回撤<40%，同时保留小市值alpha）：
  - size_weight:       0.50 → 0.65（强化小市值主效应，与原版策略方向一致）
  - bm_weight:         0.25 → 0.20（BM 依赖 carry-forward 净资产，可靠性有限）
  - roe_weight:        0.15 → 0.10（同上）
  - turnover_weight:   0.10 → 0.05
  - min_market_cap:    20亿 → 10亿（原20亿切断了大量小市值 alpha 来源）
  - select_stock_num:  3 → 5（增加分散度，最直接有效的降回撤手段）

流水线：
  1. prepare_data   — 基础过滤（IPO>1年、北交所排除）+ 数值化关键列
  2. compute_factors — SIZE(总市值)、BM(净资产/总市值)、ROE(归母净利润_ttm/净资产)、TURNOVER_PROXY(成交额std_10/成交额)
  3. apply_filters   — 总市值下限、成交额下限、ROE>0、净资产>0、bias_20过滤、成交额波动过滤
  4. rank_stocks     — Z-score截面标准化 → 加权复合得分 → 升序排名（越小越好）
"""

import pandas as pd
import numpy as np
from strategies.base import BaseStrategy, safe_float


class QualityValueStrategy(BaseStrategy):
    """质量价值小盘策略 v3.1 — Size主导(65%) + Value下行保护(20%) + Quality筛选(10%) + 反操纵过滤(5%)；10亿市值下限；5只持仓"""

    strategy_id = 'quality_value'
    display_name = '质量价值小盘 v3.1'
    strategy_description = '多因子线性组合策略，先按交易日对各因子做 z-score，再按权重求和。v3.1调参：size权重65%+10亿下限过滤+5只持仓，强化小市值主效应，目标将最大回撤从约-55%降至<40%。'

    def __init__(self, size_weight=0.65, bm_weight=0.20, roe_weight=0.10,
                 turnover_weight=0.05, min_market_cap=10e8, min_turnover=5e7,
                 bias_pct=0.52, vol_pct=0.78, select_stock_num=5, **kwargs):
        """
        参数:
          size_weight      — 规模因子权重，v3.1调整为 0.65（强化小市值主效应）
          bm_weight        — 价值因子权重(BM)，v3.1调整为 0.20（减少BM权重，降低对carry-forward数据的依赖）
          roe_weight       — 质量因子权重(ROE)，v3.1调整为 0.10（减少ROE权重，因ROE在carry-forward数据下可靠性有限）
          turnover_weight  — 反操纵因子权重，v3.1调整为 0.05（保留少量流动性过滤）
          min_market_cap   — 最低总市值过滤（元），v3.1调整为 10亿（放开更多小市值空间，避免削弱小市值alpha）
          min_turnover     — 最低成交额过滤（元），默认 5000万
          bias_pct         — bias_20 截断分位数，默认 0.52
          vol_pct          — 成交额波动截断分位数，默认 0.78
          select_stock_num — 每期持仓数量，v3.1调整为 5（分散降低回撤，目标将最大回撤从约-55%控制到<40%）

        v3.0 → v3.1 调参记录：
          - size_weight: 0.50 → 0.65（强化小市值 alpha，与原版策略 100% size 方向一致）
          - bm_weight: 0.25 → 0.20（减少价值因子；BM 依赖 carry-forward 净资产数据，可靠性有限）
          - roe_weight: 0.15 → 0.10（同上；ROE 用 归母净利润_ttm/净资产 计算，carry-forward 偏差较大）
          - turnover_weight: 0.10 → 0.05（保留少量流动性约束即可）
          - min_market_cap: 20亿 → 10亿（原20亿过滤削弱小市值 alpha；A股5-10亿市值区间是小市值溢价主要来源）
          - select_stock_num: 3 → 5（增加持仓分散度，是降低最大回撤最直接有效的方式）
        """
        super().__init__(select_stock_num=select_stock_num, **kwargs)
        self.size_weight = size_weight
        self.bm_weight = bm_weight
        self.roe_weight = roe_weight
        self.turnover_weight = turnover_weight
        self.min_market_cap = min_market_cap
        self.min_turnover = min_turnover
        self.bias_pct = bias_pct
        self.vol_pct = vol_pct

    def get_parameter_definitions(self):
        return [
            {'key': 'size_weight', 'label': '规模因子权重',
             'description': '总市值在复合得分中的权重。',
             'default': self.size_weight, 'min': 0.0, 'max': 1.0, 'step': 0.05,
             'unit': '', 'type': 'weight'},
            {'key': 'bm_weight', 'label': '价值因子权重 (BM)',
             'description': '净资产/总市值在复合得分中的权重。',
             'default': self.bm_weight, 'min': 0.0, 'max': 1.0, 'step': 0.05,
             'unit': '', 'type': 'weight'},
            {'key': 'roe_weight', 'label': '质量因子权重 (ROE)',
             'description': '归母净利润TTM/净资产在复合得分中的权重。',
             'default': self.roe_weight, 'min': 0.0, 'max': 1.0, 'step': 0.05,
             'unit': '', 'type': 'weight'},
            {'key': 'turnover_weight', 'label': '反操纵因子权重',
             'description': '成交额波动代理因子在复合得分中的权重。',
             'default': self.turnover_weight, 'min': 0.0, 'max': 1.0, 'step': 0.05,
             'unit': '', 'type': 'weight'},
            {'key': 'select_stock_num', 'label': '每期持仓数量',
             'description': '每期持有的股票数量。',
             'default': self.select_stock_num, 'min': 1, 'max': 10, 'step': 1,
             'unit': '只', 'type': 'trading'},
            {'key': 'min_market_cap', 'label': '最低总市值',
             'description': '剔除总市值低于该值的股票。',
             'default': round(self.min_market_cap / 1e8, 2), 'min': 0, 'max': 500, 'step': 5,
             'unit': '亿', 'type': 'filter'},
            {'key': 'min_turnover', 'label': '最低成交额',
             'description': '剔除月成交额低于该值的股票。',
             'default': round(self.min_turnover / 1e8, 2), 'min': 0.1, 'max': 10, 'step': 0.1,
             'unit': '亿', 'type': 'filter'},
            {'key': 'bias_pct', 'label': 'bias_20 截断分位数',
             'description': '剔除偏离20日均线幅度过大的股票。',
             'default': self.bias_pct, 'min': 0.3, 'max': 1.0, 'step': 0.01,
             'unit': '', 'type': 'filter'},
            {'key': 'vol_pct', 'label': '成交额波动截断分位数',
             'description': '剔除成交额波动异常股票。',
             'default': self.vol_pct, 'min': 0.3, 'max': 1.0, 'step': 0.01,
             'unit': '', 'type': 'filter'},
        ]

    def get_filter_descriptions(self):
        return [
            {'name': '最低总市值过滤', 'description': '先排除过小微盘，避免极端操纵风险。'},
            {'name': '最低成交额过滤', 'description': '保证月度流动性。'},
            {'name': 'ROE > 0', 'description': '排除亏损公司。'},
            {'name': '净资产 > 0', 'description': '排除资不抵债公司。'},
            {'name': 'bias_20 过滤', 'description': '剔除短期涨幅过大的股票。'},
            {'name': '成交额波动过滤', 'description': '剔除异常放量股票。'},
        ]

    def get_ranking_metadata(self):
        return {
            'name': 'Z-score 复合得分',
            'formula': '因子 = size_weight×SIZE_z - bm_weight×BM_z - roe_weight×ROE_z + turnover_weight×TURNOVER_PROXY_z',
            'direction': '升序（越小越好）',
            'description': '这是标准的多因子线性组合：先按交易日把各因子 z-score 标准化，再按权重线性相加。',
            'combination_method': 'zscore_weighted_sum',
            'normalization_method': '按交易日期做截面 z-score 标准化',
            'components': [
                {
                    'key': 'SIZE_z',
                    'label': '规模因子 (SIZE_z)',
                    'source_column': 'SIZE_z',
                    'role': '小市值暴露',
                    'orientation': '越小越好',
                    'transformation': 'zscore',
                    'weight': round(self.size_weight, 4),
                    'notes': '总市值先 z-score，负值代表相对更小市值。',
                },
                {
                    'key': 'BM_z',
                    'label': '价值因子 (BM_z)',
                    'source_column': 'BM_z',
                    'role': '低估值暴露',
                    'orientation': '越大越好',
                    'transformation': 'zscore',
                    'weight': round(self.bm_weight, 4),
                    'notes': '线性组合时以负号进入最终因子，高 BM 会降低最终得分。',
                },
                {
                    'key': 'ROE_z',
                    'label': '质量因子 (ROE_z)',
                    'source_column': 'ROE_z',
                    'role': '盈利质量',
                    'orientation': '越大越好',
                    'transformation': 'zscore',
                    'weight': round(self.roe_weight, 4),
                    'notes': 'ROE 越高，线性组合中越能压低最终因子。',
                },
                {
                    'key': 'TURNOVER_PROXY_z',
                    'label': '反操纵因子 (TURNOVER_PROXY_z)',
                    'source_column': 'TURNOVER_PROXY_z',
                    'role': '交易稳定性',
                    'orientation': '越小越好',
                    'transformation': 'zscore',
                    'weight': round(self.turnover_weight, 4),
                    'notes': '成交额波动越小越好，因此以正号进入最终因子。',
                },
            ],
            'weight_details': {
                '规模(Size)': round(self.size_weight, 4),
                '价值(BM)': round(self.bm_weight, 4),
                '质量(ROE)': round(self.roe_weight, 4),
                '反操纵': round(self.turnover_weight, 4),
            },
        }

    def build_selection_reason(self, row, rank, total):
        size_z = round(float(row.get('SIZE_z', 0)), 2)
        bm_z = round(float(row.get('BM_z', 0)), 2)
        roe_z = round(float(row.get('ROE_z', 0)), 2)
        turnover_z = round(float(row.get('TURNOVER_PROXY_z', 0)), 2)
        breakdown = [
            self._factor_item('规模因子 SIZE_z', size_z, f'{round(self.size_weight * 100, 1)}% 权重', '总市值按交易日做 z-score；值越小越好'),
            self._factor_item('价值因子 BM_z', bm_z, f'{round(self.bm_weight * 100, 1)}% 权重', '净资产/总市值按交易日做 z-score；值越大越好，在线性组合中以负号进入'),
            self._factor_item('质量因子 ROE_z', roe_z, f'{round(self.roe_weight * 100, 1)}% 权重', 'ROE 按交易日做 z-score；值越大越好，在线性组合中以负号进入'),
            self._factor_item('反操纵因子 TURNOVER_PROXY_z', turnover_z, f'{round(self.turnover_weight * 100, 1)}% 权重', '成交额波动代理因子按交易日做 z-score；值越小越好'),
        ]
        return {
            'summary': f'综合质量价值得分靠前，{self._format_rank(rank, total)}',
            'details': [
                f'规模因子(Size): z={size_z}，越小越好，小市值更优。',
                f'价值因子(BM): z={bm_z}，越高越好，低估值更优。',
                f'质量因子(ROE): z={roe_z}，越高越好，盈利能力更强。',
                f'反操纵因子: z={turnover_z}，越低越好，交易行为更稳定。',
                '这里各因子已先做按交易日的 z-score 标准化，再线性加权，因此不会出现原始市值量纲吞噬其他因子的情况。',
            ],
            'fundamentals': self._build_fundamentals(row),
            'factor_breakdown': breakdown,
        }

    # ── 数据准备 ──────────────────────────────────────────────────

    def prepare_data(self, df):
        """
        基础数据准备 + 此策略需要的额外列数值化。
        净资产、归母净利润_ttm 用于计算 BM 和 ROE 因子。
        """
        df = super().prepare_data(df)
        for col in ['净资产', '归母净利润_ttm']:
            if col in df.columns:
                df[col] = safe_float(df[col])
        return df

    # ── 因子计算 ──────────────────────────────────────────────────

    def compute_factors(self, df):
        """
        计算四个选股因子：

          SIZE          = 总市值（越小越好，A股最强alpha）
          BM            = 净资产 / 总市值（book-to-market，越高越便宜）
          ROE           = 归母净利润_ttm / 净资产（越高盈利能力越强）
          TURNOVER_PROXY = 成交额std_10 / 成交额
                           （成交量波动率代理变量，越低表示交易行为越稳定，
                            替代 quant_factor.md §9 中需要日换手率数据的
                            异常换手率因子）

        所有因子中的 inf/-inf 用 0 填充，NaN 用 0 填充。
        """
        # SIZE: 总市值，直接使用已有列
        df['SIZE'] = df['总市值']

        # BM: 净资产 / 总市值（价值因子）
        df['BM'] = df['净资产'] / df['总市值']
        df['BM'] = df['BM'].replace([np.inf, -np.inf], np.nan).fillna(0)

        # ROE: 归母净利润_ttm / 净资产（质量因子）
        df['ROE'] = df['归母净利润_ttm'] / df['净资产']
        df['ROE'] = df['ROE'].replace([np.inf, -np.inf], np.nan).fillna(0)

        # TURNOVER_PROXY: 成交额标准差 / 成交额（反操纵代理因子）
        df['TURNOVER_PROXY'] = df['成交额std_10'] / df['成交额']
        df['TURNOVER_PROXY'] = df['TURNOVER_PROXY'].replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0)

        return df

    # ── 过滤层 ────────────────────────────────────────────────────

    def apply_filters(self, df):
        """
        六步过滤，按顺序执行：

        Step 1 — 总市值下限：剔除微盘股（v3.1默认 > 10亿，原v3.0默认 > 20亿），避免操纵风险
        Step 2 — 成交额下限：确保流动性可交易（默认 > 5000万/月）
        Step 3 — ROE > 0：排除亏损公司
        Step 4 — 净资产 > 0：排除资不抵债公司
        Step 5 — bias_20 过滤（继承自原版策略）：剔除短期涨幅过大的股票
        Step 6 — 成交额波动过滤（继承自原版策略）：剔除异常放量股票
        """
        # Step 1: 总市值下限
        df = df[df['SIZE'] >= self.min_market_cap]

        # Step 2: 成交额下限
        df = df[df['成交额'] >= self.min_turnover]

        # Step 3: ROE > 0（盈利公司）
        df = df[df['ROE'] > 0]

        # Step 4: 净资产 > 0（正权益）
        df = df[df['净资产'] > 0]

        # Step 5: bias_20 过滤
        cutoff = df.groupby('交易日期')['bias_20'].transform(
            lambda x: x.quantile(self.bias_pct)
        )
        df = df[df['bias_20'] < cutoff]

        # Step 6: 成交额波动过滤
        vol_cutoff = df.groupby('交易日期')['成交额std_10'].transform(
            lambda x: x.quantile(self.vol_pct)
        )
        df = df[df['成交额std_10'] < vol_cutoff]

        return df

    # ── 排名 ──────────────────────────────────────────────────────

    def rank_stocks(self, df):
        """
        Z-score 截面标准化 + 加权复合排名。

        各因子按"交易日期"分组，在每个截面上做 Z-score 标准化，
        然后按权重线性组合为复合得分。

        方向处理（复合得分越小 → 排名越靠前）:
          SIZE:          市值越小越好 → +权重 × zscore(SIZE)
                         （小市值 → 负z值 → 负得分 → 排名靠前）
          BM:            越高越好   → -权重 × zscore(BM)
                         （高BM → 正z值 → 负得分 → 排名靠前）
          ROE:           越高越好   → -权重 × zscore(ROE)
                         （高ROE → 正z值 → 负得分 → 排名靠前）
          TURNOVER_PROXY: 越低越好  → +权重 × zscore(TURNOVER_PROXY)
                         （低波动 → 负z值 → 负得分 → 排名靠前）

        权重默认：规模 50%、价值 25%、质量 15%、反操纵 10%
        """
        def zscore(series):
            """截面Z-score标准化，NaN 填充为 0。"""
            mean = series.mean()
            std = series.std()
            if std == 0 or pd.isna(std):
                return pd.Series(0.0, index=series.index)
            return (series - mean) / std

        factor_names = ['SIZE', 'BM', 'ROE', 'TURNOVER_PROXY']
        for f in factor_names:
            df[f + '_z'] = df.groupby('交易日期')[f].transform(zscore)
            df[f + '_z'] = df[f + '_z'].fillna(0)

        df['因子'] = (
            self.size_weight      * df['SIZE_z']
            - self.bm_weight       * df['BM_z']
            - self.roe_weight      * df['ROE_z']
            + self.turnover_weight * df['TURNOVER_PROXY_z']
        )

        return df
