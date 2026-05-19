"""
MethodAStrategy — 日线缠论流水线 v2.0。

与 v1.1 (ChanEnhancedStrategy) 的核心区别：
  v1.1 使用月度频率的代理因子：
       - 3-K线分型（同频率三根月K线）
       - MACD bar 对比
       - 3-bar 中枢
      这些是日线缠论概念的月度近似，精度有限。

  v2.0 使用 chan_monthly_factor_builder.py 从日线数据跑完整流水线后
       聚合到月度。流水线步骤：
       日线含包含处理 → 分型识别 → 笔构建 → 线段划分 → 中枢识别
       → 背驰判断 → 买卖点标记 → 月度聚合统计
       这才是缠论的标准分析框架。

因子来源：
  .cache/chan_factors_v2/chan_factors_500.csv
  如果文件不存在，自动回退到代理因子（等价于 ChanEnhancedStrategy）。

策略逻辑（PM 建议）：
  缠论因子与小市值因子协同而非互斥。
  在小市值桶内用缠论因子排序（5% tilt vs v1.1 的 3%，
  因为日线流水线因子信号更可靠，权重略有提升）。
"""

import os
import pandas as pd
import numpy as np
from strategies.base import BaseStrategy
from chan_factors import compute_chan_factors

# 默认 Method A 因子文件路径
DEFAULT_FACTOR_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    '.cache', 'chan_factors_v2', 'chan_factors_500.csv'
)


class MethodAStrategy(BaseStrategy):
    """
    Method A — 日线缠论流水线聚合策略。

    使用 chan_monthly_factor_builder.py 产出的日线→月度缠论因子。
    """

    strategy_id = 'method_a'
    display_name = 'Method A v2.0'
    strategy_description = '日线缠论流水线聚合到月度后，作为对市值排名的边际乘法倾斜。'

    def __init__(self, val_pct_cutoff=0.68, bias_pct=0.52,
                 vol_pct=0.78, chan_tilt=0.05,
                 factor_path=DEFAULT_FACTOR_PATH, **kwargs):
        """
        参数:
          val_pct_cutoff — 行业估值分位阈值
          bias_pct       — bias_20 截断分位数
          vol_pct        — 成交额波动截断分位数
          chan_tilt      — 缠论因子在排名中的边际权重（默认 5%）
          factor_path    — Method A 因子 CSV 路径
        """
        super().__init__(**kwargs)
        self.val_pct_cutoff = val_pct_cutoff
        self.bias_pct = bias_pct
        self.vol_pct = vol_pct
        self.chan_tilt = chan_tilt
        self.factor_path = factor_path
        self.require_positive_pe = True
        self.require_positive_net_assets = True
        self.require_positive_profit = True

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
            {'key': 'chan_tilt', 'label': 'Method A 边际权重',
             'description': 'Method A 结构得分对市值排名的边际影响。',
             'default': self.chan_tilt, 'min': 0.0, 'max': 0.2, 'step': 0.005,
             'unit': '', 'type': 'weight'},
        ]

    def get_filter_descriptions(self):
        return self.get_quality_filter_descriptions() + [
            {'name': '行业估值过滤', 'description': '沿用原版行业估值过滤。'},
            {'name': 'bias_20 过滤', 'description': '剔除短期偏离均线过大的股票。'},
            {'name': '成交额波动过滤', 'description': '剔除成交额波动异常股票。'},
            {'name': 'Method A 强卖点排除', 'description': '顶背驰确认 + 中枢位置偏上 + 卖点多于买点时排除。'},
        ]

    def get_factor_overview_tags(self):
        return ['规模因子', '缠论背驰因子', '缠论中枢位置因子', '缠论分型因子', '缠论笔强度因子', '缠论买卖点信号因子']

    def get_ranking_metadata(self):
        return {
            'name': '市值排名 × (1 - chan_tilt × Method A 结构分数)',
            'formula': '因子 = rank_size × (1 - chan_tilt × ma_score_norm)',
            'direction': '升序（越小越好）',
            'description': 'Method A 不是直接和总市值做线性加权，而是让归一化结构分数对市值排名做边际乘法调整。',
            'combination_method': 'multiplicative_rank_tilt',
            'normalization_method': '总市值先转为截面 rank；ma_score clip 到 [-10, 10] 后缩放到 [-1, 1]',
            'components': [
                {
                    'key': 'rank_size',
                    'label': '市值排名',
                    'source_column': 'rank_size',
                    'role': '主排序因子',
                    'orientation': '越小越好',
                    'transformation': 'rank',
                    'weight': round(1 - self.chan_tilt, 4),
                    'notes': '主体逻辑仍然是小市值优先。',
                },
                {
                    'key': 'ma_score_norm',
                    'label': 'Method A 归一化结构分数',
                    'source_column': 'ma_score_norm',
                    'role': '边际倾斜',
                    'orientation': '越大越好',
                    'transformation': 'clip_and_scale',
                    'weight': round(self.chan_tilt, 4),
                    'notes': '来自日线缠论流水线聚合分数，正值会提升最终排序。',
                },
            ],
            'weight_details': {'市值主导': round(1 - self.chan_tilt, 4), 'Method A 边际倾斜': round(self.chan_tilt, 4)},
        }

    def build_selection_reason(self, row, rank, total):
        ma_score = round(float(row.get('ma_score', 0)), 2)
        ma_norm = round(float(row.get('ma_score_norm', 0)), 2)
        rank_size = row.get('rank_size', rank or 0)
        rank_size = int(rank_size) if pd.notna(rank_size) else rank
        breakdown = [
            self._factor_item('市值排名', f'第 {rank_size}', '主排序因子', '先转成截面 rank，避免原始市值量纲直接进入组合'),
            self._factor_item('Method A 结构分数', ma_score, '原始技术分数', '由底分型、背驰、买卖点、中枢位置等日线流水线特征聚合得到'),
            self._factor_item('Method A 归一化分数', ma_norm, f'{round(self.chan_tilt * 100, 1)}% 边际倾斜', 'clip 到 [-10, 10] 后缩放到 [-1, 1]'),
            self._factor_item('最终排序公式', 'rank_size × (1 - chan_tilt × ma_score_norm)', '乘法 tilt', '不是简单线性加权和'),
        ]
        return {
            'summary': f'小市值为主，Method A 结构信号做加减分，{self._format_rank(rank, total)}',
            'details': [
                f'主体逻辑是小市值优先：市值排名第 {rank_size}。',
                f'Method A 原始结构得分为 {ma_score}，归一化后为 {ma_norm}。',
                '总市值先转 rank，结构分数再归一化后做边际乘法调整，因此不会被原始市值量纲直接吞噬。',
                '最终排序 = 市值排名 × (1 - chan_tilt × Method A 结构分数)。',
            ],
            'fundamentals': self._build_fundamentals(row),
            'factor_breakdown': breakdown,
        }

    # ── 因子计算 ──────────────────────────────────────────────────

    def compute_factors(self, df):
        """
        加载 Method A 日线流水线因子并合并。

        先计算行业估值分位，再尝试加载外部日线缠论因子文件。
        如果文件不存在，回退到代理因子。
        """
        # 行业估值分位（同原版策略）
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

        # 加载 Method A 因子
        df = self._load_method_a_factors(df)
        return df

    def _load_method_a_factors(self, df):
        """
        从缓存文件加载日线流水线聚合因子并合并到 df。

        重命名规则：chan_xxx → ma_xxx（避免与代理因子同名冲突）。

        缺失值填充策略：
          - 分型比例 → 0.5（中性：无信号）
          - 计数类因子 → 0（无该事件发生）
          - 方向/位置类 → 0（无趋势/中性位置）
        """
        if not os.path.exists(self.factor_path):
            print(f"[Method A] 因子文件不存在: {self.factor_path}，回退到代理因子")
            return compute_chan_factors(df)

        chan_df = pd.read_csv(self.factor_path, encoding='gbk')
        chan_df['交易日期'] = pd.to_datetime(chan_df['交易日期'])

        # 列名映射: chan_xxx → ma_xxx
        ma_cols = {
            'chan_top_fractal': 'ma_top_fractal',
            'chan_bottom_fractal': 'ma_bottom_fractal',
            'chan_fractal_ratio': 'ma_fractal_ratio',
            'chan_stroke_dir': 'ma_stroke_dir',
            'chan_stroke_count': 'ma_stroke_count',
            'chan_stroke_strength': 'ma_stroke_strength',
            'chan_zhongshu_count': 'ma_zhongshu_count',
            'chan_zhongshu_position': 'ma_zhongshu_position',
            'chan_zhongshu_width': 'ma_zhongshu_width',
            'chan_top_div': 'ma_top_div',
            'chan_bottom_div': 'ma_bottom_div',
            'chan_div_signal': 'ma_div_signal',
            'chan_buy_signals': 'ma_buy_signals',
            'chan_sell_signals': 'ma_sell_signals',
            'chan_segment_count': 'ma_segment_count',
        }
        chan_df = chan_df.rename(columns=ma_cols)

        # 只保留合并需要的列
        merge_cols = ['交易日期', '股票代码'] + list(ma_cols.values())
        chan_df = chan_df[[c for c in merge_cols if c in chan_df.columns]]

        df = df.merge(chan_df, on=['交易日期', '股票代码'], how='left')

        # 缺失因子 → 中性值
        for col in ma_cols.values():
            if col not in df.columns:
                continue
            if col == 'ma_fractal_ratio':
                df[col] = df[col].fillna(0.5)   # 分型比例中性值
            elif col in ('ma_stroke_dir', 'ma_div_signal',
                         'ma_zhongshu_position', 'ma_stroke_strength',
                         'ma_zhongshu_width'):
                df[col] = df[col].fillna(0.0)   # 方向/位置中性值
            else:
                df[col] = df[col].fillna(0)      # 计数类因子缺省为 0

        return df

    # ── 过滤层 ────────────────────────────────────────────────────

    def apply_filters(self, df):
        """
        四步过滤：原版三步 + Method A 缠论负向排除。

        Method A 的负向排除使用日线流水线因子：
          顶背驰信号 + 中枢上方位置 + 卖点信号多于买点信号
          三者同时满足 → 排除。

        相比 v1.1 的三重确认（中枢上方+顶背驰+顶分型），
        v2.0 的条件略有不同因为在日线流水线中买卖点比简单分型更精确。
        """
        # Step 0: 经营质量底线过滤
        df = self.apply_quality_filters(df)

        # Step 1: 行业估值过滤
        df = df[df['val_pct'] < self.val_pct_cutoff]

        # Step 2: bias_20 过滤
        cutoff = df.groupby('交易日期')['bias_20'].transform(
            lambda x: x.quantile(self.bias_pct)
        )
        df = df[df['bias_20'] < cutoff]

        # Step 3: 成交额波动过滤
        vol_cutoff = df.groupby('交易日期')['成交额std_10'].transform(
            lambda x: x.quantile(self.vol_pct)
        )
        df = df[df['成交额std_10'] < vol_cutoff]

        # Step 4: Method A 缠论负向排除
        # 使用日线流水线真实因子进行排除
        # 条件: div_signal=-1（顶背驰确认）且中枢位置偏上且卖点>买点
        if all(c in df.columns for c in ['ma_div_signal',
                                           'ma_zhongshu_position',
                                           'ma_sell_signals',
                                           'ma_buy_signals']):
            strong_sell = (
                (df['ma_div_signal'] == -1) &
                (df['ma_zhongshu_position'] > 0.5) &
                (df['ma_sell_signals'] > df['ma_buy_signals'])
            )
            df = df[~strong_sell]

        return df

    # ── 排名 ──────────────────────────────────────────────────────

    def rank_stocks(self, df):
        """
        复合排名：市值为主 (95%) + Method A 缠论信号边际加成 (5%)。

        先构建 Method A 综合得分 (ma_score):
          正向: 底分型 + 底背驰 + 买点信号 + 中枢下方
          负向: 顶分型 + 顶背驰 + 卖点信号 + 中枢上方

        最终因子 = rank_size * (1 - chan_tilt * ma_score_norm)
        tilt=5%（vs v1.1 的 3%），因为日线流水线因子更可靠。
        """
        # 构建 Method A 综合得分
        df['ma_score'] = 0.0

        # 正向信号
        if 'ma_bottom_fractal' in df.columns:
            df['ma_score'] += df['ma_bottom_fractal'] * 1.0    # 底分型+1
        if 'ma_bottom_div' in df.columns:
            df['ma_score'] += df['ma_bottom_div'] * 3.0        # 底背驰+3（可信度更高）
        if 'ma_buy_signals' in df.columns:
            df['ma_score'] += df['ma_buy_signals'] * 2.0       # 买入信号+2
        if 'ma_zhongshu_position' in df.columns:
            df['ma_score'] += (df['ma_zhongshu_position'] < 0).astype(int) * 1.5

        # 负向信号
        if 'ma_top_fractal' in df.columns:
            df['ma_score'] -= df['ma_top_fractal'] * 1.0       # 顶分型-1
        if 'ma_top_div' in df.columns:
            df['ma_score'] -= df['ma_top_div'] * 3.0           # 顶背驰-3
        if 'ma_sell_signals' in df.columns:
            df['ma_score'] -= df['ma_sell_signals'] * 2.0      # 卖出信号-2
        if 'ma_zhongshu_position' in df.columns:
            df['ma_score'] -= (df['ma_zhongshu_position'] > 0.5).astype(int) * 1.5

        # 市值排名（主体）
        df['rank_size'] = df.groupby('交易日期')['总市值'].rank(
            ascending=True
        )

        # 得分归一化后加权
        df['ma_score_norm'] = df['ma_score'].clip(-10, 10) / 10.0
        df['因子'] = df['rank_size'] * (
            1.0 - self.chan_tilt * df['ma_score_norm']
        )

        return df
