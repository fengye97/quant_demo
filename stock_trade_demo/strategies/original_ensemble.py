"""
OriginalEnsembleStrategy — 多时间窗口投票版原策略。

思路：
  1. 用 3 年 / 5 年 / 全量训练窗口，分别在一组受控候选参数中选出最优 profile
  2. 三个 profile 独立跑原版策略逻辑（行业估值 + bias + 成交额波动 + 小市值）
  3. 每个 profile 对当期前若干名股票给排序分数，再按近期更高权重做加权投票
  4. 最终在有投票支持的股票中，按“全市场小市值 rank / (1 + 投票分)”排序

这不是逐月 walk-forward 重训练，而是固定训练截止日后的轻量多窗口融合版，
重点验证 2026-04 / 2026-05 是否比单一全历史参数更贴近最新市场风格。
"""

import math
from copy import deepcopy

import numpy as np
import pandas as pd

from backtest import select_and_backtest
from index_data import build_index_panel
from strategies.original import OriginalStrategy

PROFILE_WINDOWS = {
    '3y': 3,
    '5y': 5,
    'full': None,
}

WINDOW_LABELS = {
    '3y': '近3年子策略',
    '5y': '近5年子策略',
    'full': '全量子策略',
}

WINDOW_SCORING_CONFIG = {
    '3y': {'tail_months': 6, 'recent_weight': 0.65},
    '5y': {'tail_months': 12, 'recent_weight': 0.45},
    'full': {'tail_months': 18, 'recent_weight': 0.25},
}

CANDIDATE_LIBRARY = [
    {
        'name': 'baseline',
        'val_pct_cutoff': 0.68,
        'bias_pct': 0.52,
        'vol_pct': 0.78,
        'bull_tp': 0.30,
        'bear_tp': 0.22,
        'bull_n': 6,
        'bear_n': 4,
    },
    {
        'name': 'tight_defensive',
        'val_pct_cutoff': 0.58,
        'bias_pct': 0.44,
        'vol_pct': 0.72,
        'bull_tp': 0.26,
        'bear_tp': 0.16,
        'bull_n': 4,
        'bear_n': 2,
    },
    {
        'name': 'defensive',
        'val_pct_cutoff': 0.62,
        'bias_pct': 0.48,
        'vol_pct': 0.74,
        'bull_tp': 0.28,
        'bear_tp': 0.18,
        'bull_n': 5,
        'bear_n': 3,
    },
    {
        'name': 'balanced',
        'val_pct_cutoff': 0.66,
        'bias_pct': 0.50,
        'vol_pct': 0.78,
        'bull_tp': 0.30,
        'bear_tp': 0.20,
        'bull_n': 5,
        'bear_n': 3,
    },
    {
        'name': 'recent_tilt',
        'val_pct_cutoff': 0.64,
        'bias_pct': 0.50,
        'vol_pct': 0.76,
        'bull_tp': 0.32,
        'bear_tp': 0.20,
        'bull_n': 6,
        'bear_n': 3,
    },
    {
        'name': 'offensive',
        'val_pct_cutoff': 0.72,
        'bias_pct': 0.56,
        'vol_pct': 0.82,
        'bull_tp': 0.34,
        'bear_tp': 0.24,
        'bull_n': 7,
        'bear_n': 4,
    },
    {
        'name': 'high_beta_expansion',
        'val_pct_cutoff': 0.74,
        'bias_pct': 0.56,
        'vol_pct': 0.84,
        'bull_tp': 0.36,
        'bear_tp': 0.22,
        'bull_n': 7,
        'bear_n': 3,
    },
    {
        'name': 'concentrated_reversal',
        'val_pct_cutoff': 0.60,
        'bias_pct': 0.44,
        'vol_pct': 0.76,
        'bull_tp': 0.30,
        'bear_tp': 0.18,
        'bull_n': 4,
        'bear_n': 2,
    },
]

_PROFILE_CACHE = {}
_INDEX_PANEL_CACHE = None
_BOARD_STRENGTH_CACHE = {}


class OriginalEnsembleStrategy(OriginalStrategy):
    strategy_id = 'original_ensemble'
    display_name = '多窗口投票版原策略'
    strategy_description = '近3年/近5年/全量三个原版子策略先独立拟合，再按 0.5/0.3/0.2 加权投票，近期窗口权重更高。'

    def __init__(self, weight_3y=0.5, weight_5y=0.3, weight_full=0.2,
                 vote_top_k=12, profile_end_date='2026-03-31',
                 board_tilt_strength=0.4, board_recent_weight=0.65,
                 board_short_window=20, board_long_window=60, **kwargs):
        super().__init__(**kwargs)
        self.weight_3y = weight_3y
        self.weight_5y = weight_5y
        self.weight_full = weight_full
        self.vote_top_k = int(vote_top_k)
        self.profile_end_date = profile_end_date
        self.board_tilt_strength = float(board_tilt_strength)
        self.board_recent_weight = float(board_recent_weight)
        self.board_short_window = int(board_short_window)
        self.board_long_window = int(board_long_window)
        self._profiles = None
        self._profile_summary = []

    def get_parameter_definitions(self):
        return [
            {'key': 'weight_3y', 'label': '近3年权重',
             'description': '近 3 年子策略在投票中的权重。',
             'default': self.weight_3y, 'min': 0.0, 'max': 1.0, 'step': 0.05,
             'unit': '', 'type': 'weight'},
            {'key': 'weight_5y', 'label': '近5年权重',
             'description': '近 5 年子策略在投票中的权重。',
             'default': self.weight_5y, 'min': 0.0, 'max': 1.0, 'step': 0.05,
             'unit': '', 'type': 'weight'},
            {'key': 'weight_full', 'label': '全量权重',
             'description': '全量历史子策略在投票中的权重。',
             'default': self.weight_full, 'min': 0.0, 'max': 1.0, 'step': 0.05,
             'unit': '', 'type': 'weight'},
            {'key': 'vote_top_k', 'label': '投票候选数',
             'description': '每个子策略对当期前多少名股票给分。',
             'default': self.vote_top_k, 'min': 6, 'max': 24, 'step': 1,
             'unit': '只', 'type': 'filter'},
            {'key': 'board_tilt_strength', 'label': '板块倾斜强度',
             'description': '根据创业板/科创板/主板相对强弱，对最终投票结果做轻量倾斜；默认保持保守，避免压过主投票逻辑。',
             'default': self.board_tilt_strength, 'min': 0.0, 'max': 3.0, 'step': 0.1,
             'unit': '', 'type': 'weight'},
        ]

    def get_filter_descriptions(self):
        return self.get_quality_filter_descriptions() + [
            {'name': '多窗口训练', 'description': '固定训练截止日之前，分别基于近3年、近5年和全量历史选择最优原版参数 profile。'},
            {'name': '子策略候选过滤', 'description': '每个子策略内部沿用原版的行业估值、bias_20 和成交额波动过滤。'},
            {'name': '投票支持过滤', 'description': '最终只在至少获得一个子策略投票支持的股票中排名。'},
            {'name': '板块强弱倾斜', 'description': '根据创业板/科创板相对 CSI1000 的近期强弱，对最终票分做轻量倾斜，帮助组合跟随最新主线风格。'},
        ]

    def get_factor_overview_tags(self):
        return ['规模因子']

    def get_ranking_metadata(self):
        weights = self._normalized_weights()
        return {
            'name': '全市场小市值 rank / (1 + 多窗口加权投票分 + 板块强弱分)',
            'formula': '因子 = size_rank / (1 + 3y票分 + 5y票分 + full票分 + board_tilt)',
            'direction': '升序（越小越好）',
            'description': '三个时间窗口子策略先投票，再根据创业板/科创板/主板相对强弱做轻量倾斜，在保留小市值锚的同时跟随最新主线风格。',
            'combination_method': 'weighted_vote_then_size_tiebreak',
            'normalization_method': '各窗口先转 rank score，再按窗口权重加权求和',
            'components': [
                {
                    'key': 'size_rank',
                    'label': '全市场小市值排名',
                    'source_column': 'vote_size_rank',
                    'role': '主排序锚点',
                    'orientation': '越小越好',
                    'transformation': 'rank',
                    'weight': 'tie-break',
                    'notes': '投票相同或接近时，仍由小市值优先。',
                },
                {
                    'key': 'vote_score_3y',
                    'label': '近3年票分',
                    'source_column': 'vote_score_3y',
                    'role': '加权投票',
                    'orientation': '越大越好',
                    'transformation': f'rank_score × {weights["3y"]}',
                    'weight': round(weights['3y'], 4),
                    'notes': '更强调最近风格，默认权重最高。',
                },
                {
                    'key': 'vote_score_5y',
                    'label': '近5年票分',
                    'source_column': 'vote_score_5y',
                    'role': '加权投票',
                    'orientation': '越大越好',
                    'transformation': f'rank_score × {weights["5y"]}',
                    'weight': round(weights['5y'], 4),
                    'notes': '兼顾中期稳定性。',
                },
                {
                    'key': 'vote_score_full',
                    'label': '全量票分',
                    'source_column': 'vote_score_full',
                    'role': '加权投票',
                    'orientation': '越大越好',
                    'transformation': f'rank_score × {weights["full"]}',
                    'weight': round(weights['full'], 4),
                    'notes': '保留长期小市值经验作为低权重锚。',
                },
                {
                    'key': 'board_tilt_score',
                    'label': '板块强弱分',
                    'source_column': 'board_tilt_score',
                    'role': '风格倾斜',
                    'orientation': '越大越好',
                    'transformation': f'board_strength × {round(self.board_tilt_strength, 2)}',
                    'weight': round(self.board_tilt_strength, 4),
                    'notes': '创业板/科创板近期更强时，对相应板块股票做轻量加分。',
                },
            ],
            'weight_details': {
                '近3年': round(weights['3y'], 4),
                '近5年': round(weights['5y'], 4),
                '全量': round(weights['full'], 4),
            },
        }

    def build_selection_reason(self, row, rank, total):
        weights = self._normalized_weights()
        score_3y = round(float(row.get('vote_score_3y', 0.0)), 2)
        score_5y = round(float(row.get('vote_score_5y', 0.0)), 2)
        score_full = round(float(row.get('vote_score_full', 0.0)), 2)
        total_score = round(float(row.get('vote_total_score', 0.0)), 2)
        board_tilt_score = round(float(row.get('board_tilt_score', 0.0)), 2)
        board_raw_score = round(float(row.get('board_raw_score', 0.0)), 2)
        size_rank = int(row.get('vote_size_rank', rank or 0)) if pd.notna(row.get('vote_size_rank', np.nan)) else (rank or 0)
        support_count = int(row.get('vote_support_count', 0) or 0)
        primary_window = str(row.get('vote_primary_window', '') or '')
        primary_label = WINDOW_LABELS.get(primary_window, '多窗口')
        board_label = self._board_label(str(row.get('board_key', '') or 'main_board'))

        breakdown = [
            self._factor_item('全市场小市值排名', f'第 {size_rank}', '排序锚点', '投票接近时仍优先小市值'),
            self._factor_item('近3年票分', score_3y, f'权重 {round(weights["3y"] * 100, 1)}%', '来自近3年最优 profile 的前排加分'),
            self._factor_item('近5年票分', score_5y, f'权重 {round(weights["5y"] * 100, 1)}%', '来自近5年最优 profile 的前排加分'),
            self._factor_item('全量票分', score_full, f'权重 {round(weights["full"] * 100, 1)}%', '来自全量最优 profile 的前排加分'),
            self._factor_item('板块归属', board_label, '风格识别', '按股票代码前缀识别创业板/科创板/主板'),
            self._factor_item('板块强弱分', board_tilt_score, f'原始强弱 {board_raw_score}', '最近更强的板块会获得轻量加分'),
            self._factor_item('最终排序公式', 'size_rank / (1 + vote_total + board_tilt)', '投票+板块+小市值', '票分和板块强弱分越高，最终因子越小'),
        ]

        profile_summaries = []
        if self._profiles:
            for key in ['3y', '5y', 'full']:
                profile = self._profiles.get(key)
                if not profile:
                    continue
                params = profile.get('params', {})
                profile_summaries.append(
                    f"{WINDOW_LABELS[key]}: {profile.get('candidate_name', 'baseline')}，"
                    f"val<{params.get('val_pct_cutoff', self.val_pct_cutoff)} / "
                    f"bias<{params.get('bias_pct', self.bias_pct)} / "
                    f"bull_tp={params.get('bull_tp', self.bull_tp)} / bear_tp={params.get('bear_tp', self.bear_tp)} / "
                    f"bull_n={params.get('bull_n', self.bull_n)} / bear_n={params.get('bear_n', self.bear_n)}"
                )

        details = [
            f'该股票获得 {support_count} 个时间窗口子策略支持，主导来源是 {primary_label}。',
            f'三路票分分别为 3Y={score_3y}、5Y={score_5y}、Full={score_full}，合计 {total_score}。',
            f'该股属于{board_label}，当前板块原始强弱分为 {board_raw_score}，倾斜后贡献 {board_tilt_score} 分。',
            f'最终不是简单按总票分排序，而是用全市场小市值排名第 {size_rank} 名作为锚，再叠加板块强弱倾斜。',
        ]
        details.extend(profile_summaries)

        return {
            'summary': f'多窗口投票后入选，主导窗口为{primary_label}，{self._format_rank(rank, total)}',
            'details': details,
            'fundamentals': self._build_fundamentals(row),
            'factor_breakdown': breakdown,
        }

    def run(self, df):
        weights = self._normalized_weights()
        profiles = self._resolve_profiles(df)
        base_df = self.prepare_data(df.copy())
        base_df['vote_size_rank'] = base_df.groupby('交易日期')['总市值'].rank(ascending=True, method='first')
        base_df['board_key'] = base_df['股票代码'].map(self._infer_board_key)
        board_strength_lookup = self._build_board_strength_lookup(base_df['交易日期'])

        vote_frames = []
        for key in ['3y', '5y', 'full']:
            profile = profiles.get(key)
            if not profile:
                continue
            strategy = self._build_original_strategy(profile['params'])
            ranked = strategy.run(df.copy())
            if len(ranked) == 0:
                continue
            ranked = ranked.copy()
            ranked['window_rank'] = ranked.groupby('交易日期')['因子'].rank(ascending=True, method='first')
            ranked = ranked[ranked['window_rank'] <= self.vote_top_k].copy()
            if len(ranked) == 0:
                continue
            ranked[f'vote_rank_{key}'] = ranked['window_rank']
            ranked[f'vote_raw_score_{key}'] = (self.vote_top_k + 1 - ranked['window_rank']).astype(float)
            ranked[f'vote_score_{key}'] = ranked[f'vote_raw_score_{key}'] * weights[key]
            keep_cols = [
                '交易日期', '股票代码',
                f'vote_rank_{key}', f'vote_raw_score_{key}', f'vote_score_{key}',
            ]
            vote_frames.append(ranked[keep_cols])

        if not vote_frames:
            fallback = super().run(df.copy())
            fallback['vote_score_3y'] = 0.0
            fallback['vote_score_5y'] = 0.0
            fallback['vote_score_full'] = 0.0
            fallback['vote_total_score'] = 0.0
            fallback['vote_support_count'] = 0
            fallback['vote_primary_window'] = 'full'
            fallback['vote_size_rank'] = fallback.groupby('交易日期')['总市值'].rank(ascending=True, method='first')
            fallback['board_key'] = fallback['股票代码'].map(self._infer_board_key)
            fallback['board_raw_score'] = 0.0
            fallback['board_tilt_score'] = 0.0
            fallback.attrs['strategy_meta'] = {
                'profile_summary': self._profile_summary,
            }
            return fallback

        votes = vote_frames[0]
        for extra in vote_frames[1:]:
            votes = votes.merge(extra, on=['交易日期', '股票代码'], how='outer')

        merged = base_df.merge(votes, on=['交易日期', '股票代码'], how='left')
        for key in ['3y', '5y', 'full']:
            for col in [f'vote_rank_{key}', f'vote_raw_score_{key}', f'vote_score_{key}']:
                if col in merged.columns:
                    merged[col] = pd.to_numeric(merged[col], errors='coerce')

        merged['vote_score_3y'] = merged.get('vote_score_3y', 0.0).fillna(0.0)
        merged['vote_score_5y'] = merged.get('vote_score_5y', 0.0).fillna(0.0)
        merged['vote_score_full'] = merged.get('vote_score_full', 0.0).fillna(0.0)
        merged['vote_total_score'] = merged['vote_score_3y'] + merged['vote_score_5y'] + merged['vote_score_full']
        merged['vote_support_count'] = (
            (merged['vote_score_3y'] > 0).astype(int) +
            (merged['vote_score_5y'] > 0).astype(int) +
            (merged['vote_score_full'] > 0).astype(int)
        )

        merged = merged[merged['vote_total_score'] > 0].copy()
        merged['board_raw_score'] = merged.apply(
            lambda row: self._get_board_strength_for_date(row['交易日期'], str(row.get('board_key', 'main_board') or 'main_board'), board_strength_lookup),
            axis=1,
        )
        merged['board_tilt_score'] = merged['board_raw_score'] * self.board_tilt_strength
        merged['vote_primary_window'] = merged[['vote_score_3y', 'vote_score_5y', 'vote_score_full']].idxmax(axis=1)
        merged['vote_primary_window'] = merged['vote_primary_window'].map({
            'vote_score_3y': '3y',
            'vote_score_5y': '5y',
            'vote_score_full': 'full',
        }).fillna('full')
        merged['因子'] = merged['vote_size_rank'] / (1.0 + merged['vote_total_score'] + merged['board_tilt_score'])
        merged.attrs['strategy_meta'] = {
            'profile_summary': self._profile_summary,
        }
        return merged

    def _normalized_weights(self):
        weights = {
            '3y': max(float(self.weight_3y), 0.0),
            '5y': max(float(self.weight_5y), 0.0),
            'full': max(float(self.weight_full), 0.0),
        }
        total = sum(weights.values())
        if total <= 0:
            return {'3y': 0.5, '5y': 0.3, 'full': 0.2}
        return {k: v / total for k, v in weights.items()}

    def _infer_board_key(self, code):
        digits = ''.join(ch for ch in str(code or '') if ch.isdigit())
        if digits.startswith(('688', '689')):
            return 'star50'
        if digits.startswith(('300', '301')):
            return 'chinext'
        return 'main_board'

    def _board_label(self, board_key):
        return {
            'star50': '科创板',
            'chinext': '创业板',
            'main_board': '主板',
        }.get(board_key, '主板')

    def _get_index_panel(self):
        global _INDEX_PANEL_CACHE
        if _INDEX_PANEL_CACHE is None:
            panel = build_index_panel(index_ids=['csi1000', 'chinext', 'star50'])
            panel = panel.sort_values('交易日期').reset_index(drop=True)
            _INDEX_PANEL_CACHE = panel
        return _INDEX_PANEL_CACHE.copy()

    def _build_board_strength_lookup(self, trade_dates):
        trade_dates = pd.to_datetime(pd.Series(trade_dates).dropna().unique())
        if len(trade_dates) == 0:
            return {}
        max_trade_date = pd.to_datetime(max(trade_dates)).strftime('%Y-%m-%d')
        cache_key = (max_trade_date, self.board_short_window, self.board_long_window, round(self.board_recent_weight, 4))
        if cache_key in _BOARD_STRENGTH_CACHE:
            return deepcopy(_BOARD_STRENGTH_CACHE[cache_key])

        panel = self._get_index_panel()
        panel = panel[panel['交易日期'] <= pd.to_datetime(max_trade_date)].copy()
        if len(panel) == 0:
            return {}

        lookup = {}
        for idx in panel.index:
            row = panel.loc[idx]
            scores = self._board_scores_for_index(panel, idx)
            lookup[pd.to_datetime(row['交易日期']).strftime('%Y-%m-%d')] = scores
        _BOARD_STRENGTH_CACHE[cache_key] = deepcopy(lookup)
        return lookup

    def _board_scores_for_index(self, panel, idx):
        star_rel_short = self._relative_return(panel, 'star50_close', 'csi1000_close', idx, self.board_short_window)
        star_rel_long = self._relative_return(panel, 'star50_close', 'csi1000_close', idx, self.board_long_window)
        chinext_rel_short = self._relative_return(panel, 'chinext_close', 'csi1000_close', idx, self.board_short_window)
        chinext_rel_long = self._relative_return(panel, 'chinext_close', 'csi1000_close', idx, self.board_long_window)

        star_score = self._blend_board_strength(star_rel_short, star_rel_long)
        chinext_score = self._blend_board_strength(chinext_rel_short, chinext_rel_long)
        main_score = float(np.clip(-0.5 * (star_score + chinext_score), -1.0, 1.0))
        return {
            'star50': star_score,
            'chinext': chinext_score,
            'main_board': main_score,
        }

    def _relative_return(self, panel, asset_col, benchmark_col, idx, lookback):
        if asset_col not in panel.columns or benchmark_col not in panel.columns:
            return 0.0
        if idx <= 0:
            return 0.0
        start_idx = max(0, idx - int(lookback))
        asset_slice = panel.loc[start_idx:idx, asset_col].astype(float)
        benchmark_slice = panel.loc[start_idx:idx, benchmark_col].astype(float)
        if len(asset_slice) < 2 or len(benchmark_slice) < 2:
            return 0.0
        asset_start, asset_end = float(asset_slice.iloc[0]), float(asset_slice.iloc[-1])
        benchmark_start, benchmark_end = float(benchmark_slice.iloc[0]), float(benchmark_slice.iloc[-1])
        if asset_start <= 0 or benchmark_start <= 0:
            return 0.0
        asset_ret = asset_end / asset_start - 1.0
        benchmark_ret = benchmark_end / benchmark_start - 1.0
        return float(asset_ret - benchmark_ret)

    def _blend_board_strength(self, short_rel, long_rel):
        raw = self.board_recent_weight * short_rel + (1 - self.board_recent_weight) * long_rel
        return float(np.clip(raw / 0.12, -1.0, 1.0))

    def _get_board_strength_for_date(self, trade_date, board_key, lookup):
        key = pd.to_datetime(trade_date).strftime('%Y-%m-%d')
        scores = lookup.get(key, {})
        return float(scores.get(board_key, 0.0))

    def _resolve_profile_end_date(self, df):
        max_date = pd.to_datetime(df['交易日期']).max()
        if self.profile_end_date:
            target = pd.to_datetime(self.profile_end_date)
            return min(target, max_date)
        return max_date

    def _profile_cache_key(self, df, profile_end_date):
        min_date = pd.to_datetime(df['交易日期']).min().strftime('%Y-%m-%d')
        max_date = pd.to_datetime(df['交易日期']).max().strftime('%Y-%m-%d')
        return (min_date, max_date, len(df), profile_end_date.strftime('%Y-%m-%d'), self.select_stock_num)

    def _resolve_profiles(self, df):
        profile_end_date = self._resolve_profile_end_date(df)
        cache_key = self._profile_cache_key(df, profile_end_date)
        if cache_key in _PROFILE_CACHE:
            cached = deepcopy(_PROFILE_CACHE[cache_key])
            self._profiles = cached['profiles']
            self._profile_summary = cached['summary']
            return self._profiles

        profiles = {}
        summary = []
        for key, years in PROFILE_WINDOWS.items():
            window_df, start_date = self._slice_training_window(df, profile_end_date, years)
            best_profile = self._select_best_profile(window_df, key)
            best_profile['window_key'] = key
            best_profile['window_start'] = start_date.strftime('%Y-%m-%d')
            best_profile['window_end'] = profile_end_date.strftime('%Y-%m-%d')
            profiles[key] = best_profile
            summary.append({
                'window': key,
                'label': WINDOW_LABELS[key],
                'candidate_name': best_profile.get('candidate_name', ''),
                'score': round(float(best_profile.get('score', 0.0)), 4),
                'overall_score': round(float(best_profile.get('overall_score', 0.0)), 4),
                'recent_score': round(float(best_profile.get('recent_score', 0.0)), 4),
                'months': int(best_profile.get('months', 0)),
                'window_start': best_profile['window_start'],
                'window_end': best_profile['window_end'],
                'params': best_profile.get('params', {}),
            })

        self._profiles = profiles
        self._profile_summary = summary
        _PROFILE_CACHE[cache_key] = {
            'profiles': deepcopy(profiles),
            'summary': deepcopy(summary),
        }
        return profiles

    def _slice_training_window(self, df, profile_end_date, years):
        train = df[pd.to_datetime(df['交易日期']) <= profile_end_date].copy()
        if years is None:
            start_date = pd.to_datetime(train['交易日期']).min()
            return train, start_date
        start_date = profile_end_date - pd.DateOffset(years=years)
        window_df = train[pd.to_datetime(train['交易日期']) >= start_date].copy()
        if len(window_df) == 0:
            return train, pd.to_datetime(train['交易日期']).min()
        return window_df, pd.to_datetime(window_df['交易日期']).min()

    def _select_best_profile(self, train_df, window_key):
        fallback = {
            'candidate_name': 'baseline',
            'params': deepcopy(CANDIDATE_LIBRARY[0]),
            'score': -1e9,
            'overall_score': -1e9,
            'recent_score': -1e9,
            'months': 0,
        }
        if len(train_df) == 0:
            return fallback

        best = None
        for candidate in CANDIDATE_LIBRARY:
            params = {k: v for k, v in candidate.items() if k != 'name'}
            try:
                strategy = self._build_original_strategy(params)
                ranked = strategy.run(train_df.copy())
                if len(ranked) == 0:
                    continue
                result = select_and_backtest(
                    ranked, strategy,
                    select_stock_num=self.select_stock_num,
                    c_rate=strategy.c_rate,
                    t_rate=strategy.t_rate,
                    bull_tp=strategy.bull_tp,
                    bear_tp=strategy.bear_tp,
                    bull_n=strategy.bull_n,
                    bear_n=strategy.bear_n,
                    initial_capital=strategy.initial_capital,
                )
                score_payload = self._score_result(result, window_key)
                candidate_result = {
                    'candidate_name': candidate['name'],
                    'params': deepcopy(params),
                    **score_payload,
                }
                if best is None or candidate_result['score'] > best['score']:
                    best = candidate_result
            except Exception:
                continue

        return best or fallback

    def _build_original_strategy(self, params):
        return OriginalStrategy(
            select_stock_num=self.select_stock_num,
            c_rate=self.c_rate,
            t_rate=self.t_rate,
            initial_capital=self.initial_capital,
            **params,
        )

    def _score_result(self, result, window_key):
        if result is None or len(result) == 0:
            return {'score': -1e9, 'overall_score': -1e9, 'recent_score': -1e9, 'months': 0}

        months = len(result)
        if months < 6:
            return {'score': -1e9, 'overall_score': -1e9, 'recent_score': -1e9, 'months': months}

        overall = self._calc_period_score(result, dd_floor=0.15, annualize=True)
        cfg = WINDOW_SCORING_CONFIG.get(window_key, {'tail_months': 12, 'recent_weight': 0.35})
        tail_months = min(int(cfg.get('tail_months', 12)), months)
        recent_slice = result.tail(tail_months).copy()
        recent = self._calc_period_score(recent_slice, dd_floor=0.10, annualize=False)
        recent_weight = float(cfg.get('recent_weight', 0.35))
        score = overall['score'] * (1 - recent_weight) + recent['score'] * recent_weight
        return {
            'score': float(score),
            'overall_score': float(overall['score']),
            'recent_score': float(recent['score']),
            'months': months,
            'annual_return': float(overall['annual_return']),
            'max_drawdown': float(overall['max_drawdown']),
            'win_rate': float(overall['win_rate']),
            'cumulative_return': float(overall['cumulative_return']),
            'recent_cumulative_return': float(recent['cumulative_return']),
        }

    def _calc_period_score(self, result, dd_floor, annualize):
        cum = float(result['累积净值'].iloc[-1])
        total_return = cum - 1.0
        date_delta = result['交易日期'].iloc[-1] - result['交易日期'].iloc[0]
        days = max(int(getattr(date_delta, 'days', 0)), 1)
        annual_return = cum ** (365.0 / days) - 1 if cum > 0 else -1.0
        drawdown = result['累积净值'] / result['累积净值'].cummax() - 1
        max_dd = abs(float(drawdown.min())) if len(drawdown) > 0 else 0.0
        win_rate = float((result['选股下周期涨跌幅'] > 0).mean())
        base_return = annual_return if annualize else math.log(max(cum, 1e-9))
        return_over_dd = base_return / max(max_dd, dd_floor)
        return_over_dd = float(np.clip(return_over_dd, -5.0, 5.0))
        score = return_over_dd + 0.15 * math.log(max(cum, 1e-9)) + 0.05 * win_rate
        return {
            'score': float(score),
            'annual_return': float(annual_return),
            'total_return': float(total_return),
            'max_drawdown': float(max_dd),
            'win_rate': float(win_rate),
            'cumulative_return': float(cum),
        }
