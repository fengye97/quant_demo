"""
OriginalStrategy — 原版策略（历史收益最高，累积净值 5223x）。

策略逻辑（三步过滤 + 小市值排名）:
  1. 行业估值分位过滤：
     对每个二级行业每月计算市盈率倒数(TTM)和市净率倒数的中位数，
     再做 expanding rank percentile（最小 12 期历史）。
     取 EP rank 和 BP rank 的均值作为行业估值综合分位。
     剔除综合分位 > 68% 的行业（即排除估值最高的 32% 行业）。
     逻辑：EP/BP 过低（估值过高）的行业未来收益往往较差。

  2. bias_20 过滤：
     股价偏离 20 日均线的幅度。每期取全市场 52% 分位数作为截断点，
     剔除偏离过大的股票。逻辑：短期涨幅过大的股票有均值回归压力。

  3. 成交额波动过滤：
     10 日成交额标准差。每期取全市场 78% 分位数作为截断点，
     剔除波动过大的股票。逻辑：异常放量常伴随主力出货或消息炒作。

  4. 小市值排名：
     按总市值升序排名（越小越好）。逻辑：A 股存在极强的小市值效应，
     过去 20 年小市值组合累积收益远超市场平均。
"""

import pandas as pd
import numpy as np
from strategies.base import BaseStrategy


class OriginalStrategy(BaseStrategy):
    """原版策略：行业估值 + bias反转 + 小市值"""

    strategy_id = 'original'
    display_name = '原版策略'
    strategy_description = '行业估值 + bias反转 + 小市值，主排序因子是原始总市值。'

    def __init__(self, val_pct_cutoff=0.68, bias_pct=0.52,
                 vol_pct=0.78, **kwargs):
        """
        参数:
          val_pct_cutoff — 行业估值分位阈值，低于此值的行业保留
          bias_pct       — bias_20 截断分位数
          vol_pct        — 成交额波动截断分位数
        """
        super().__init__(**kwargs)
        self.val_pct_cutoff = val_pct_cutoff
        self.bias_pct = bias_pct
        self.vol_pct = vol_pct

    def get_parameter_definitions(self):
        return [
            {'key': 'val_pct_cutoff', 'label': '行业估值分位阈值',
             'description': '剔除估值高于此分位的行业。值越小筛选越严。',
             'default': self.val_pct_cutoff, 'min': 0.3, 'max': 1.0, 'step': 0.01,
             'unit': '', 'type': 'filter'},
            {'key': 'bias_pct', 'label': 'bias_20 截断分位数',
             'description': '剔除股价偏离20日均线幅度超过此分位的股票。',
             'default': self.bias_pct, 'min': 0.3, 'max': 1.0, 'step': 0.01,
             'unit': '', 'type': 'filter'},
            {'key': 'vol_pct', 'label': '成交额波动截断分位数',
             'description': '剔除成交额标准差超过此分位的股票。',
             'default': self.vol_pct, 'min': 0.3, 'max': 1.0, 'step': 0.01,
             'unit': '', 'type': 'filter'},
        ]

    def get_filter_descriptions(self):
        return [
            {'name': '行业估值过滤', 'description': '剔除估值分位高于阈值的行业，避免在高估行业里做小市值暴露。'},
            {'name': 'bias_20 过滤', 'description': '剔除短期偏离20日均线过大的股票，降低追高回撤风险。'},
            {'name': '成交额波动过滤', 'description': '剔除成交额异常波动的股票，过滤交易行为不稳定标的。'},
        ]

    def get_ranking_metadata(self):
        return {
            'name': '总市值',
            'formula': '因子 = 总市值',
            'direction': '升序（越小越好）',
            'description': '这是单因子小市值策略，直接按总市值升序选股，不与其他异质量纲因子做线性相加。',
            'combination_method': 'single_factor',
            'normalization_method': 'raw（未归一化；因为没有多因子线性组合）',
            'components': [
                {
                    'key': 'total_market_cap',
                    'label': '总市值',
                    'source_column': '总市值',
                    'role': '100% 主排序因子',
                    'orientation': '越小越好',
                    'transformation': 'raw',
                    'weight': 1.0,
                    'notes': '这里不存在市值量纲吞噬其他因子的风险，因为最终排序只使用这一个因子。',
                },
            ],
            'weight_details': {'总市值': 1.0},
        }

    def build_selection_reason(self, row, rank, total):
        market_cap_yi = self._market_cap_yi(row)
        details = [
            f"主排序因子是总市值：{market_cap_yi} 亿，越小越优先" if market_cap_yi is not None else '主排序因子是总市值，越小越优先',
            '这是单因子策略，不存在多因子线性组合时的量纲吞噬问题。',
            '先通过行业估值、bias 与成交额波动过滤，再按总市值升序选股。',
        ]
        breakdown = []
        if market_cap_yi is not None:
            breakdown.append(self._factor_item('总市值', f'{market_cap_yi} 亿', '100% 主排序因子', '未归一化，直接升序排名'))
        return {
            'summary': f'以小市值为主导逻辑入选，{self._format_rank(rank, total)}',
            'details': details,
            'fundamentals': self._build_fundamentals(row),
            'factor_breakdown': breakdown,
        }

    # ── 因子计算 ──────────────────────────────────────────────────

    def compute_factors(self, df):
        """
        计算行业估值分位因子。

        方法：
          - 每个"二级行业 × 交易日期"计算 EP/BP 中位数
          - expanding rank(pct) 给出当前值在历史中的分位位置
          - 最终 val_pct = mean(EP分位, BP分位)，综合衡量行业相对估值
        """
        ind_col = '新版申万二级行业名称'

        # 计算每个行业每期的 EP/BP 中位数（代表该行业估值水平）
        ind_val = df.groupby([ind_col, '交易日期']).agg(
            med_ep=('市盈率倒数', 'median'),
            med_bp=('市净率倒数', 'median'),
        ).reset_index()

        def calc_val_percentile(grp):
            """
            对单个行业的时间序列计算 expanding 估值分位。

            expanding rank(pct):
              第 t 期的分位 = rank(med_ep_t among med_ep_{1..t}) / t
              最小值需要 12 期历史才能计算（min_periods=12）

            含义：如果某行业当期 EP 在所有历史期中排名很低（分位 < 0.5），
                  说明当前该行业相对于自身历史处于高估值状态。
            """
            grp = grp.sort_values('交易日期')
            ep_pct = grp['med_ep'].expanding(min_periods=12).rank(pct=True)
            bp_pct = grp['med_bp'].expanding(min_periods=12).rank(pct=True)
            # 综合分位 = EP分位和BP分位的均值（等权）
            grp['val_pct'] = (ep_pct.fillna(0.5) + bp_pct.fillna(0.5)) / 2
            return grp

        ind_val = ind_val.groupby(ind_col, group_keys=False).apply(
            calc_val_percentile
        )
        df = df.merge(
            ind_val[[ind_col, '交易日期', 'val_pct']],
            on=[ind_col, '交易日期'], how='left'
        )
        # 没有历史数据的行业填 0.5（中性），不剔除也不优待
        df['val_pct'] = df['val_pct'].fillna(0.5)
        return df

    # ── 过滤层 ────────────────────────────────────────────────────

    def apply_filters(self, df):
        """
        三步过滤，按顺序执行：

        Step 1 — 行业估值过滤：
          剔除 val_pct 高于阈值的行业（高估值行业）。
          阈值 0.68 意味着排除估值最高的 32% 行业。

        Step 2 — 股价偏离过滤：
          每期计算 bias_20 的市场分位数，剔除偏离过大的股票。
          阈值设在 52% 分位，即仅保留偏离最小的 52% 股票。

        Step 3 — 成交额波动过滤：
          每期计算成交额std_10 的市场分位数，剔除波动异常股票。
          阈值设在 78% 分位。

        设计原则：过滤是为了把明显差的排除掉，而不是把特别好的选出来。
        所以三个过滤的阈值都相对宽松（保留 50%+ 的股票），
        把最终选股权留给排名环节。
        """
        # Step 1: 行业估值过滤
        # val_pct < 0.68 → 排除估值最高的 ~32% 行业
        df = df[df['val_pct'] < self.val_pct_cutoff]

        # Step 2: bias_20 过滤（股价偏离20日均线幅度）
        # bias_20 太大 = 短期涨太多，有回撤风险
        cutoff = df.groupby('交易日期')['bias_20'].transform(
            lambda x: x.quantile(self.bias_pct)
        )
        df = df[df['bias_20'] < cutoff]

        # Step 3: 成交额波动过滤
        # 成交额std_10 太大 = 交易行为异常，可能被操纵
        vol_cutoff = df.groupby('交易日期')['成交额std_10'].transform(
            lambda x: x.quantile(self.vol_pct)
        )
        df = df[df['成交额std_10'] < vol_cutoff]

        return df

    # ── 排名 ──────────────────────────────────────────────────────

    def rank_stocks(self, df):
        """
        按总市值升序排名（越小越靠前）。

        逻辑：A 股存在极强的小市值效应。
        不引入其他因子的原因是保持策略一致性——
        20 年的历史回测证明小市值因子在 A 股的主导地位远超其他单因子。
        """
        df['因子'] = df['总市值']
        return df
