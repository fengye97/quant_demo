"""
ChanEnhancedStrategy — 缠论增强策略 v1.1。

策略逻辑：
  在保留原版三步过滤的基础上，融入缠论因子作为边际优化层：
    - 负向排除：剔除"中枢上方 + 顶背驰 + 顶分型"三重确认的最差信号股票
    - 正向加成：在市值排名中，给缠论积极信号最多 3% 的排名倾斜

设计原则（PM 建议）：
  缠论因子与小市值因子协同而非互斥。A 股小市值效应极强（20年5000x+），
  缠论因子不应替代市值排名，而应在同等市值水平下充当 tiebreaker。
  3% 的权重确保缠论仅在几乎同市值的股票之间发挥作用。

对应 quant_factor.md Sections 10-14。
"""

import numpy as np
import pandas as pd
from strategies.base import BaseStrategy
from chan_factors import compute_chan_factors


class ChanEnhancedStrategy(BaseStrategy):
    """
    缠论增强策略。

    继承原版的行业估值 + bias + 成交额三步过滤，
    新增缠论负向排除和排名边际加成。
    """

    strategy_id = 'chan_enhanced'
    display_name = '缠论增强策略 v1.1'
    strategy_description = '原版过滤 + 缠论代理因子负向排除，最终由市值排名主导并叠加缠论边际倾斜。'

    def __init__(self, val_pct_cutoff=0.68, bias_pct=0.52,
                 vol_pct=0.78, chan_tilt=0.03, **kwargs):
        """
        参数:
          val_pct_cutoff — 行业估值分位阈值
          bias_pct       — bias_20 截断分位数
          vol_pct        — 成交额波动截断分位数
          chan_tilt      — 缠论因子在排名中的边际权重（0-1），默认 3%
        """
        super().__init__(**kwargs)
        self.val_pct_cutoff = val_pct_cutoff
        self.bias_pct = bias_pct
        self.vol_pct = vol_pct
        self.chan_tilt = chan_tilt

    def get_parameter_definitions(self):
        return [
            {'key': 'val_pct_cutoff', 'label': '行业估值分位阈值',
             'description': '同原版策略。', 'default': self.val_pct_cutoff,
             'min': 0.3, 'max': 1.0, 'step': 0.01, 'unit': '', 'type': 'filter'},
            {'key': 'bias_pct', 'label': 'bias_20 截断分位数',
             'description': '同原版策略。', 'default': self.bias_pct,
             'min': 0.3, 'max': 1.0, 'step': 0.01, 'unit': '', 'type': 'filter'},
            {'key': 'vol_pct', 'label': '成交额波动截断分位数',
             'description': '同原版策略。', 'default': self.vol_pct,
             'min': 0.3, 'max': 1.0, 'step': 0.01, 'unit': '', 'type': 'filter'},
            {'key': 'chan_tilt', 'label': '缠论边际权重',
             'description': '缠论信号对市值排名的边际影响。数值越大，缠论信号的加分/减分越明显。',
             'default': self.chan_tilt, 'min': 0.0, 'max': 0.2, 'step': 0.005,
             'unit': '', 'type': 'weight'},
        ]

    def get_filter_descriptions(self):
        return [
            {'name': '行业估值过滤', 'description': '沿用原版行业估值过滤，先避开相对高估行业。'},
            {'name': 'bias_20 过滤', 'description': '剔除短期偏离20日均线过大的股票。'},
            {'name': '成交额波动过滤', 'description': '剔除成交额波动异常股票。'},
            {'name': '缠论强卖点排除', 'description': '中枢上方 + 顶背驰 + 顶分型三重共振时直接排除。'},
        ]

    def get_ranking_metadata(self):
        return {
            'name': '市值排名 × (1 - chan_tilt × 缠论信号)',
            'formula': '因子 = rank_size × (1 - chan_tilt × chan_signal_norm)',
            'direction': '升序（越小越好）',
            'description': '最终排序不是简单线性加权和，而是先按市值排名，再让归一化缠论信号做小幅乘法倾斜。',
            'combination_method': 'multiplicative_rank_tilt',
            'normalization_method': '总市值先转为截面 rank；缠论得分 clip 到 [-8, 8] 后缩放到 [-1, 1]',
            'components': [
                {
                    'key': 'rank_size',
                    'label': '市值排名',
                    'source_column': 'rank_size',
                    'role': '主排序因子',
                    'orientation': '越小越好',
                    'transformation': 'rank',
                    'weight': round(1 - self.chan_tilt, 4),
                    'notes': '主体逻辑仍然是小市值优先，市值量纲已经通过排名消除。',
                },
                {
                    'key': 'chan_signal_norm',
                    'label': '缠论归一化信号',
                    'source_column': 'chan_signal_norm',
                    'role': '边际倾斜',
                    'orientation': '越大越好',
                    'transformation': 'clip_and_scale',
                    'weight': round(self.chan_tilt, 4),
                    'notes': '正值会把最终因子压低，帮助同市值桶内更好的缠论结构靠前。',
                },
            ],
            'weight_details': {'市值主导': round(1 - self.chan_tilt, 4), '缠论边际倾斜': round(self.chan_tilt, 4)},
        }

    def build_selection_reason(self, row, rank, total):
        chan_norm = round(float(row.get('chan_signal_norm', 0)), 2)
        rank_size = row.get('rank_size', rank or 0)
        rank_size = int(rank_size) if pd.notna(rank_size) else rank
        breakdown = [
            self._factor_item('市值排名', f'第 {rank_size}', '主排序因子', '先将总市值转为截面排名，避免原始量纲直接参与组合'),
            self._factor_item('缠论归一化信号', chan_norm, f'{round(self.chan_tilt * 100, 1)}% 边际倾斜', '原始缠论得分先 clip 到 [-8, 8]，再缩放到 [-1, 1]'),
            self._factor_item('最终排序公式', 'rank_size × (1 - chan_tilt × chan_signal_norm)', '乘法 tilt', '不是简单线性加权和'),
        ]
        return {
            'summary': f'小市值优先，并由缠论信号做边际加分，{self._format_rank(rank, total)}',
            'details': [
                f'主体逻辑是小市值优先：市值排名第 {rank_size}',
                f'缠论信号归一化得分为 {chan_norm}，正值会压低最终因子并提升排序。',
                '总市值先转成 rank，缠论信号再做归一化，因此不存在原始市值量纲吞噬缠论信号的问题。',
                '最终排序 = 市值排名 × (1 - chan_tilt × 缠论信号)',
            ],
            'fundamentals': self._build_fundamentals(row),
            'factor_breakdown': breakdown,
        }

    # ── 因子计算 ──────────────────────────────────────────────────

    def compute_factors(self, df):
        """
        计算缠论因子 + 行业估值分位。

        缠论因子通过 chan_factors.compute_chan_factors 计算，
        行业估值分位复制原版逻辑。
        """
        # 缠论因子（来自 chan_factors 模块）
        df = compute_chan_factors(df)

        # 行业估值分位（与原版策略相同）
        ind_col = '新版申万二级行业名称'
        ind_val = df.groupby([ind_col, '交易日期']).agg(
            med_ep=('市盈率倒数', 'median'),
            med_bp=('市净率倒数', 'median'),
        ).reset_index()

        def calc_val_percentile(grp):
            grp = grp.sort_values('交易日期')
            ep_pct = grp['med_ep'].expanding(min_periods=12).rank(pct=True)
            bp_pct = grp['med_bp'].expanding(min_periods=12).rank(pct=True)
            grp['val_pct'] = (ep_pct.fillna(0.5) + bp_pct.fillna(0.5)) / 2
            return grp

        ind_val = ind_val.groupby(ind_col, group_keys=False).apply(
            calc_val_percentile
        )
        df = df.merge(
            ind_val[[ind_col, '交易日期', 'val_pct']],
            on=[ind_col, '交易日期'], how='left'
        )
        df['val_pct'] = df['val_pct'].fillna(0.5)
        return df

    # ── 过滤层 ────────────────────────────────────────────────────

    def apply_filters(self, df):
        """
        四步过滤：原版三步 + 缠论负向排除。

        Steps 1-3 与原版策略完全一致。
        Step 4 (缠论负向排除):
          同时满足以下三个条件的股票被认为是缠论中的"强卖点"：
            - 中枢上方（价格已到阻力区）
            - 顶背驰（MACD动能衰竭）
            - 顶分型（局部见顶形态）
          三重确认后排除。在月线数据中出现极少（<1%），排除成本极低。
          缠论理论依据：此三重信号共振时是缠论中最明确的离场/不买入信号。
        """
        # Step 1: 行业估值过滤（同原版）
        df = df[df['val_pct'] < self.val_pct_cutoff]

        # Step 2: bias_20 过滤（同原版）
        cutoff = df.groupby('交易日期')['bias_20'].transform(
            lambda x: x.quantile(self.bias_pct)
        )
        df = df[df['bias_20'] < cutoff]

        # Step 3: 成交额波动过滤（同原版）
        vol_cutoff = df.groupby('交易日期')['成交额std_10'].transform(
            lambda x: x.quantile(self.vol_pct)
        )
        df = df[df['成交额std_10'] < vol_cutoff]

        # Step 4: 缠论负向排除
        # 中枢上方 + 顶背驰 + 顶分型 = 三重确认强卖点信号
        # 三者同时出现时才排除，避免单因子误判
        strong_sell_triple = (
            (df['chan_above_zs'] == 1) &      # 价格在阻力区上方
            (df['chan_bearish_div'] == 1) &   # 动能衰竭
            (df['chan_top_fractal'] == 1)     # 局部顶部形态
        )
        df = df[~strong_sell_triple]

        return df

    # ── 排名 ──────────────────────────────────────────────────────

    def rank_stocks(self, df):
        """
        复合排名：市值排名为主 (97%) + 缠论信号边际加成 (3%)。

        公式: 因子 = rank_size * (1 - chan_tilt * chan_signal_norm)

        其中：
          rank_size        — 总市值升序排名（1 = 最小）
          chan_signal_norm — 缠论综合得分归一化到 [-1, 1]
          1 - tilt*norm     — 最好信号得 0.97 折（前进 3%）
                              最差信号得 1.03 折（后退 3%）

        3% 的权重确保：只有在两只股票市值几乎相同时，
        缠论信号才会改变选股结果。这避免了缠论因子与
        小市值效应正面冲突。
        """
        # 市值排名（主体）
        df['rank_size'] = df.groupby('交易日期')['总市值'].rank(
            ascending=True
        )

        # 缠论得分归一化到 [-1, 1]
        df['chan_signal_norm'] = df['chan_signal_score'].clip(-8, 8) / 8.0

        # 最终因子 = 市值排名 * (1 - tilt * 信号)
        # chan_signal_norm > 0（好信号）→ 因子变小 → 排名提升
        # chan_signal_norm < 0（坏信号）→ 因子变大 → 排名下降
        df['因子'] = df['rank_size'] * (
            1.0 - self.chan_tilt * df['chan_signal_norm']
        )

        return df
