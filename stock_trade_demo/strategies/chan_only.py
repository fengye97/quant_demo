"""
ChanOnlyStrategy — 纯缠论策略 v1.2。

策略逻辑：
  不使用任何行业估值、bias、成交额过滤。
  仅使用缠论因子进行过滤和排名，用于**单独评估缠论因子的有效性**。

过滤规则（缠论视角的"不买"条件）：
  - 有顶背驰 → 动能衰竭，不买
  - 有顶分型 → 局部见顶，不买
  - 中枢无效 → 无参考区间，不买
  - 在中枢上方 → 价格过高，不买

排名规则：
  缠论综合得分排名（70%）+ 市值排名（30%）。

注意：A 股小市值效应极强，纯缠论策略如完全不考虑市值会大幅跑输。
30% 的市值权重是为了防止选到缠论结构好但市值巨大（年化收益低）的股票。

此策略主要用于：
  - 验证缠论因子的独立有效性
  - 理解缠论信号在小市值效应面前的相对贡献
  - 作为基准对比其他策略中缠论因子的边际价值
"""

import pandas as pd
import numpy as np
from strategies.base import BaseStrategy
from chan_factors import compute_chan_factors


class ChanOnlyStrategy(BaseStrategy):
    """纯缠论策略：仅缠论因子过滤+排名，无行业/bias/成交额过滤"""

    strategy_id = 'chan_only'
    display_name = '纯缠论策略 v1.2'
    strategy_description = '只保留缠论过滤，最终由缠论排名与市值排名做同量纲线性组合。'

    def __init__(self, chan_weight=0.70, **kwargs):
        """
        参数:
          chan_weight — 缠论排名权重（剩余为市值权重），默认 0.70
        """
        super().__init__(**kwargs)
        self.chan_weight = chan_weight

    def get_parameter_definitions(self):
        return [
            {'key': 'chan_weight', 'label': '缠论排名权重',
             'description': '缠论排名在线性组合中的权重；剩余部分自动分配给市值排名。',
             'default': self.chan_weight, 'min': 0.0, 'max': 1.0, 'step': 0.05,
             'unit': '', 'type': 'weight'},
        ]

    def get_filter_descriptions(self):
        return [
            {'name': '顶背驰排除', 'description': 'MACD 动能与价格方向背离时不买入。'},
            {'name': '顶分型排除', 'description': '局部顶部形态出现时不买入。'},
            {'name': '中枢有效性过滤', 'description': '无有效中枢时不参与。'},
            {'name': '中枢上方排除', 'description': '价格位于中枢上方时视为买入成本偏高。'},
        ]

    def get_ranking_metadata(self):
        return {
            'name': '缠论排名 × 权重 + 市值排名 × 权重',
            'formula': '因子 = chan_weight × rank_chan + (1 - chan_weight) × rank_size',
            'direction': '升序（越小越好）',
            'description': '两个输入都是 rank，因此已经处于同一尺度，可直接做线性组合。',
            'combination_method': 'weighted_rank_sum',
            'normalization_method': '缠论信号与总市值都先转换为截面 rank',
            'components': [
                {
                    'key': 'rank_chan',
                    'label': '缠论信号排名',
                    'source_column': 'rank_chan',
                    'role': '主排序因子',
                    'orientation': '越小越好',
                    'transformation': 'rank',
                    'weight': round(self.chan_weight, 4),
                    'notes': '缠论分数越高，rank 越靠前。',
                },
                {
                    'key': 'rank_size',
                    'label': '市值排名',
                    'source_column': 'rank_size',
                    'role': '约束因子',
                    'orientation': '越小越好',
                    'transformation': 'rank',
                    'weight': round(1 - self.chan_weight, 4),
                    'notes': '给小市值一定权重，避免纯缠论选到过大市值股票。',
                },
            ],
            'weight_details': {'缠论排名': round(self.chan_weight, 4), '市值排名': round(1 - self.chan_weight, 4)},
        }

    def build_selection_reason(self, row, rank, total):
        rank_chan = row.get('rank_chan', rank or 0)
        rank_size = row.get('rank_size', rank or 0)
        rank_chan = int(rank_chan) if pd.notna(rank_chan) else rank
        rank_size = int(rank_size) if pd.notna(rank_size) else rank
        breakdown = [
            self._factor_item('缠论信号排名', f'第 {rank_chan}', f'{round(self.chan_weight * 100, 1)}% 权重', '由 chan_signal_score 降序排名得到'),
            self._factor_item('市值排名', f'第 {rank_size}', f'{round((1 - self.chan_weight) * 100, 1)}% 权重', '由总市值升序排名得到'),
            self._factor_item('最终排序公式', 'chan_weight × rank_chan + (1 - chan_weight) × rank_size', '线性组合', '两个输入都已转成 rank，同量纲可直接相加'),
        ]
        return {
            'summary': f'缠论信号主导，辅以小市值约束，{self._format_rank(rank, total)}',
            'details': [
                f'缠论信号排名第 {rank_chan}，得分越高越优先。',
                f'市值排名第 {rank_size}，用于控制同信号下的小市值暴露。',
                '两个输入都先变成 rank，再线性组合，因此不存在原始总市值量纲吞噬其他因子的问题。',
                '最终排序 = 缠论排名权重 × rank_chan + 市值排名权重 × rank_size。',
            ],
            'fundamentals': self._build_fundamentals(row),
            'factor_breakdown': breakdown,
        }

    def compute_factors(self, df):
        """计算缠论因子集合"""
        return compute_chan_factors(df)

    def apply_filters(self, df):
        """
        缠论专属过滤（不包含行业/bias/成交额过滤）。

        排除规则（每一条都是缠论中的"不应持有"信号）：
          1. 顶背驰 → MACD动能与价格方向背离，上涨趋势衰竭
          2. 顶分型 → 三K线结构显示局部顶部
          3. 中枢无效 → ZG≤ZD，三K线无重叠，缺乏参考区间
          4. 中枢上方 → 当前价格已超出中枢上轨，买入成本过高
        """
        df = df[df['chan_bearish_div'] == 0]    # 无顶背驰
        df = df[df['chan_top_fractal'] == 0]    # 无顶分型
        df = df[df['chan_zs_valid'] == 1]       # 中枢有效
        df = df[df['chan_above_zs'] == 0]       # 不在中枢上方
        return df

    def rank_stocks(self, df):
        """
        复合排名：缠论得分（70%）+ 市值（30%）。

        rank_chan: chan_signal_score 降序排名（得分越高排越前）
        rank_size: 总市值升序排名（越小排越前）
        最终因子 = 70% * rank_chan + 30% * rank_size

        注意：这里两个排名方向不同——
        chan_signal_score 越大越好（descending rank），
        总市值 越小越好（ascending rank），
        所以在合成因子时统一为"越小越好"的语义。
        """
        df['rank_chan'] = df.groupby('交易日期')['chan_signal_score'].rank(
            ascending=False  # 得分越高排名越前（rank 值越小）
        )
        df['rank_size'] = df.groupby('交易日期')['总市值'].rank(
            ascending=True   # 市值越小排名越前
        )
        df['因子'] = (
            self.chan_weight * df['rank_chan']
            + (1 - self.chan_weight) * df['rank_size']
        )
        return df
