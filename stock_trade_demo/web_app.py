"""
量化策略 Web 可视化 — Flask 后端。

启动:
  python3 web_app.py
  然后访问 http://localhost:8080

架构:
  - 启动时预加载数据并预运行回测（缓存全量结果）
  - API 请求时按日期范围过滤缓存的回测结果
  - 因子参数变化时重新运行回测
  - timing 策略探索拆分: 训练集 ≤ 2026-03-31, 验证集为 2026-04-01 ~ 2026-05-31
"""

import os
import sys
import json
import time
import pickle
import threading
import warnings
import inspect
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv as _csv
from datetime import datetime as _datetime
from get_stock_info import (
    supplement_csv as _supplement_csv,
    supplement_csv_incremental as _supplement_csv_incremental,
    fetch_realtime_quotes_batch as _fetch_realtime_quotes_batch,
)

from strategies.original import OriginalStrategy
from strategies.original_ensemble import OriginalEnsembleStrategy
from strategies.chan_enhanced import ChanEnhancedStrategy
from strategies.chan_only import ChanOnlyStrategy
from strategies.method_a import MethodAStrategy
from strategies.quality_value import QualityValueStrategy
from strategies.sector_heat import SectorHeatStrategy
from backtest import load_data, select_and_backtest, strategy_evaluate, compute_alpha_beta
from index_data import INDEX_CONFIGS, TIMING_ETF_CONFIGS, A_SHARE_INDEX_IDS, get_index_daily, get_timing_etf_daily, get_a_share_trading_calendar, get_index_returns, build_index_panel, build_us_index_panel, build_period_lookup, get_index_return_for_date, refresh_all_timing_etf_daily
from timing import (
    CSI1000TimingStrategy,
    Star50TimingStrategy,
    ChiNextTimingStrategy,
    SP500TimingStrategy,
    MacroV32TimingStrategy,
    run_timing_backtest,
    evaluate_timing_result,
    timing_result_to_json,
    filter_timing_result,
    summarize_timing_windows,
)

warnings.filterwarnings('ignore')
pd.set_option('expand_frame_repr', False)

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), 'web', 'templates'))

# 全局缓存
DATA_DF = None
INDEX_RETURNS = None  # CSI 1000 月度收益 Series（用于归因主基准）
INDEX_RETURNS_MAP = {}  # key: index_id -> monthly returns series
BACKTEST_CACHE = {}  # key: strategy_name → (result_df, eval_df)
TIMING_PANEL = None
TIMING_CACHE = {}  # key: strategy_name -> result_df
US_TIMING_PANEL = None
US_TIMING_CACHE = {}  # key: strategy_name -> result_df
_PROFILE_SUMMARY_CACHE = {}  # key: strategy_name -> profile_summary list (避免每次 API 调用重新计算)
FACTOR_BACKTEST_CACHE = {}   # key: "top_k=N" -> factor backtest result payload (persisted to disk)
CSI1000_SIGNAL_SERIES = None  # CSI1000 择时策略日线仓位信号，用于选股月度门控

# ========== 训练 cutoff / Holdout 边界（全局唯一来源） ==========
# 训练区结束日：所有参数选择 / profile 选优 / 候选库筛选都只能用 <= TRAINING_CUTOFF 的数据
# Holdout 起点：TRAINING_CUTOFF 之后的所有日期都是 holdout，仅展示 metrics，禁止反向调参
# 如需推进 cutoff（每季度/每半年），同时把 _CACHE_VERSION 递增以触发整库重算
TRAINING_CUTOFF = '2025-11-30'
HOLDOUT_START = '2025-12-01'

# Phase 2 walk-forward 输出：strategy/best_profile_{name}.json
# build_timing_strategy / build_us_timing_strategy 优先加载该文件覆盖默认参数
_BEST_PROFILE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'strategy'))
_BEST_PROFILE_CACHE = {}  # key: strategy_name -> profile dict (None 表示文件不存在)

# 实时风险因子离线产物：scripts/build_risk_signals.py 生成
_RISK_SIGNALS_FILE = os.path.join(_BEST_PROFILE_DIR, 'risk_signals.json')
_RISK_SIGNALS_CACHE = {'mtime': 0, 'data': None}

# Holdout 结构化报告离线产物：scripts/build_holdout_reports.py 生成的 sibling json
_HOLDOUT_REPORT_CACHE = {}  # strategy_name -> {'data': dict|None, 'mtime': float}

# 实盘交易记录持久化路径（独立于策略缓存；用户手动 / 半自动录入实际下单）
_LIVE_DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data'))
_LIVE_TRADES_FILE = os.path.join(_LIVE_DATA_DIR, 'live_trades.csv')
_LIVE_TRADES_LOCK = threading.Lock()
_LIVE_TRADES_COLUMNS = [
    'record_id', 'date', 'strategy', 'signal_target', 'actual_position',
    'exec_price', 'capital', 'notes', 'created_at',
    # 追加列（向后兼容，缺失视为空字符串）：成交股数；用于按 价格×股数/初始资金 反算 actual_position
    'shares',
]

# 每个策略的实盘初始资金（人民币 / 美元统一按数值，前端展示用 ¥/$）。
# 默认 5w，可在录入表单里覆盖（capital 字段）。
_LIVE_INITIAL_CAPITAL = 50000.0
# A 股一手 = 100 股；美股按 1 股粒度。
_LIVE_LOT_SIZE = {
    'csi1000_timing': 100, 'star50_timing': 100, 'chinext_timing': 100,
    'sp500_timing': 1, 'macro_v32_timing': 1,
}
_LIVE_CURRENCY = {
    'csi1000_timing': 'CNY', 'star50_timing': 'CNY', 'chinext_timing': 'CNY',
    'sp500_timing': 'USD', 'macro_v32_timing': 'USD',
}

# CLAUDE.md Rule 14: 把「收益+回撤都已跑赢 ETF（任意默认验证窗口）」标记为 production，
# 其余仍标记为 research。用户在前端看到信号时，能立刻识别哪个是经过验证的、哪个仅供参考。
# macro_v32: holdout (+3.87% / -11.86%) 显著优于纯技术 nasdaq_timing，已下线后者作为唯一纳指方向策略。
_STRATEGY_RULE14_STATUS = {
    'csi1000_timing': 'research',
    'star50_timing': 'production',
    'chinext_timing': 'research',
    'sp500_timing': 'research',
    'macro_v32_timing': 'production',
}

_REBALANCE_ACTION_LABELS = {
    'hold': '继续持有',
    'enter': '建仓买入',
    'add': '加仓',
    'trim': '减仓',
    'exit': '清仓',
    'flat': '空仓观望',
}

# 看多场景下的风险候选：通用部分 + 按策略附加
_BULLISH_RISKS_GENERAL = [
    "ETF 跟踪误差与滑点：实盘开盘成交价与策略基准 ETF 收盘价偏离（盘中波动 + 申赎冲击）会带来 0.1~0.3%/月级别的成本，不应被回测净值完全覆盖。",
    "策略历史样本有限：择时模型在极端 regime（熔断 / 政策急转 / 黑天鹅）下未必有可靠表现，遇到首次出现的环境建议主动降仓避险。",
]
_BULLISH_RISKS_BY_STRATEGY = {
    'csi1000_timing': [
        "中证1000 是小盘股代表，流动性弱于沪深300，缩量阶段反向时回撤会被放大。",
        "突破型信号易出现「假突破」：若 T+1 高开低走，建仓首日就可能回吐 1~2%。",
    ],
    'star50_timing': [
        "科创板个股集中度高（前 10 大权重接近一半），单只权重股波动会直接影响整体净值。",
        "科创板 IPO / 解禁 / 减持节奏对市场情绪冲击显著，需关注公告日临近的回撤风险。",
    ],
    'chinext_timing': [
        "创业板成长股估值对利率和流动性变化敏感，央行行长讲话 / 社融数据日易出现急跌。",
        "若指数已从底部反弹 ≥15%，继续追涨会面临短期获利盘抛压。",
    ],
    'sp500_timing': [
        "标普500 在牛市末期常出现「最后的拉升」后快速回落，注意 VIX 持续走低后的反转风险。",
        "美股财报季个股波动放大，板块轮动可能令 ETF 表现与个别龙头股脱钩。",
    ],
    'macro_v32_timing': [
        "宏观多因子模型依赖月频数据（PMI / 失业率 / CPI），对突发事件（地缘 / 银行流动性）反应滞后。",
        "Fed regime 切换时因子权重重排，策略短期信号可能反复，需容忍 1~2 周方向噪声。",
    ],
}
# 看空场景下的「其他市场机会」候选（与跨策略 target ≥ 0.5 的实时列表合并）
_BEARISH_OPPS_BY_REGION = {
    'cn': [
        "防御资产：10 年期国债 ETF（511260）或黄金 ETF（518880），历史上与小盘成长 ETF 呈负相关。",
        "现金管理：货币 ETF（511990）/ 银行 T+0 理财可作为等待信号期间的资金停车位。",
    ],
    'us': [
        "防御资产：长债 ETF（TLT）或黄金 ETF（GLD），在风险偏好下行时通常受益。",
        "现金管理：短端美债 ETF（SGOV / BIL），当前美债短端收益率仍处于历史较高水平。",
    ],
}
_REGION_OF_STRATEGY = {
    'csi1000_timing': 'cn', 'star50_timing': 'cn', 'chinext_timing': 'cn',
    'sp500_timing': 'us', 'macro_v32_timing': 'us',
}
_STRATEGY_DISPLAY = {
    'csi1000_timing': '中证1000',
    'star50_timing': '科创50',
    'chinext_timing': '创业板',
    'sp500_timing': '标普500',
    'macro_v32_timing': '纳指宏观 v3.3',
}
_SIGNAL_ACTION_LABELS = {
    'buy': '买入信号',
    'sell': '卖出信号',
    'hold': '维持',
    'flat': '空仓',
}

# 磁盘缓存配置：避免每次重启都重新计算回测缓存
_CACHE_VERSION = 12  # v12: Phase2 walk-forward best_profile 接入；切换 cutoff 或 profile 时递增
_CACHE_DIR = os.path.join(os.path.dirname(__file__), '.cache')
_CACHE_FILE = os.path.join(_CACHE_DIR, 'web_cache.pkl')
# 单因子回测必须由离线脚本预生成（参见 build_single_factor_cache.py），
# web_app 启动时只 load 这个文件，绝不在线计算。
FACTOR_BACKTEST_CACHE_FILE = os.path.join(_CACHE_DIR, 'single_factor_results.pkl')
FACTOR_BACKTEST_BUILD_SCRIPT = 'stock_trade_demo/build_single_factor_cache.py'


def _get_data_mtime():
    """获取数据文件的最新修改时间，用于判断缓存是否过期。"""
    max_mtime = 0
    for fname in ['stock_data.parquet', 'stock_data.csv']:
        fpath = os.path.join(os.path.dirname(__file__), fname)
        if os.path.exists(fpath):
            max_mtime = max(max_mtime, os.path.getmtime(fpath))
    return max_mtime


def _save_disk_cache():
    """将 BACKTEST_CACHE 和 TIMING_CACHE 序列化到磁盘。"""
    cache_dir = os.path.dirname(_CACHE_FILE)
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    payload = {
        'version': _CACHE_VERSION,
        'data_mtime': _get_data_mtime(),
        'backtest': BACKTEST_CACHE.copy(),
        'timing': TIMING_CACHE.copy(),
        'profile_summary': _PROFILE_SUMMARY_CACHE.copy(),
        'saved_at': time.time(),
    }
    try:
        with open(_CACHE_FILE, 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = os.path.getsize(_CACHE_FILE) / (1024 * 1024)
        print(f'[cache] 磁盘缓存已保存 ({size_mb:.1f}MB)')
    except Exception as e:
        print(f'[cache] 保存失败: {e}')


def _load_disk_cache():
    """从磁盘加载缓存。返回 True 表示加载成功且有效。"""
    if not os.path.exists(_CACHE_FILE):
        return False
    try:
        with open(_CACHE_FILE, 'rb') as f:
            payload = pickle.load(f)
    except Exception as e:
        print(f'[cache] 读取磁盘缓存失败: {e}')
        return False

    # 有效性检查
    if payload.get('version') != _CACHE_VERSION:
        print('[cache] 版本不匹配，需要重新计算')
        return False
    if payload.get('data_mtime', 0) < _get_data_mtime():
        print('[cache] 数据文件已更新，需要重新计算')
        return False

    BACKTEST_CACHE.update(payload.get('backtest', {}))
    TIMING_CACHE.update(payload.get('timing', {}))
    _PROFILE_SUMMARY_CACHE.update(payload.get('profile_summary', {}))
    age = time.time() - payload.get('saved_at', 0)
    print(f'[cache] 磁盘缓存加载成功 (缓存保存于 {age:.0f}s 前)')
    return True


def _load_factor_backtest_cache():
    """从独立的离线产物文件加载单因子回测结果（只读，不在线计算）。

    返回 True 表示加载成功；False 表示文件缺失或损坏 —— 此时应由
    `build_single_factor_cache.py` 离线脚本重新生成。
    """
    if not os.path.exists(FACTOR_BACKTEST_CACHE_FILE):
        return False
    try:
        with open(FACTOR_BACKTEST_CACHE_FILE, 'rb') as f:
            payload = pickle.load(f)
    except Exception as e:
        print(f'[cache] 读取单因子回测缓存失败: {e}')
        return False

    factors = payload.get('factors')
    top_k = payload.get('top_k', 5)
    if not isinstance(factors, list):
        print('[cache] 单因子回测缓存格式异常，已忽略')
        return False
    FACTOR_BACKTEST_CACHE[f'top_k={top_k}'] = {'factors': factors, 'top_k': top_k}
    age = time.time() - payload.get('saved_at', 0)
    print(f'[cache] 单因子回测缓存加载成功 ({len(factors)} 个因子，保存于 {age:.0f}s 前)')
    return True

STRATEGY_MAP = {
    'original': OriginalStrategy,
    'original_ensemble': OriginalEnsembleStrategy,
    'chan_enhanced': ChanEnhancedStrategy,
    'chan_only': ChanOnlyStrategy,
    'method_a': MethodAStrategy,
    'quality_value': QualityValueStrategy,
    'sector_heat': SectorHeatStrategy,
}

TIMING_STRATEGY_MAP = {
    'csi1000_timing': CSI1000TimingStrategy,
    'star50_timing': Star50TimingStrategy,
    'chinext_timing': ChiNextTimingStrategy,
}

US_TIMING_STRATEGY_MAP = {
    'macro_v32_timing': MacroV32TimingStrategy,
    'sp500_timing': SP500TimingStrategy,
}

US_TIMING_PAGE_STRATEGY_IDS = [
    'macro_v32_timing',
    'sp500_timing',
]

US_TIMING_CHANGELOG_META = {
    'macro_v32_timing': {
        'market_group': 'nasdaq',
        'supersedes': 'nasdaq_timing',
        'changelog_title': '纳指ETF 策略升级至宏观多因子 v3.2',
        'changelog_summary': '从价格趋势/动量二元择时升级为宏观多因子 + Sigmoid 连续仓位模型，默认前台只保留当前收益更高的生产策略。',
        'changelog_bullets': [
            '从均线 + 动量二元开关，升级为 8 个宏观/市场因子等权聚合的 ContScore。',
            '新增 Sigmoid 仓位映射、VIX 危机覆盖和月度惯性阈值，减少高波动区间的大幅回撤。',
            '继续沿用离线缓存产物，前端只读结果，不在默认页面请求上重算策略。',
        ],
    },
    'sp500_timing': {
        'market_group': 'sp500',
        'supersedes': None,
        'changelog_title': '标普500ETF 生产参数已做小步提速优化',
        'changelog_summary': '保留原有均线 + 动量主框架，但把默认生产参数切换为更温和的 staged 仓位与更快的趋势窗口，优先修复近期反弹阶段入场过慢的问题。',
        'changelog_bullets': [
            '默认仓位模式从 binary 调整为 staged，降低满进满出带来的来回打脸。',
            '快/慢均线与动量窗口同步缩短，争取在标普500恢复阶段更早恢复部分仓位。',
            '仍然通过离线 cache 产物供前端展示，默认页面不在请求路径上实时重算。',
        ],
    },
}

TIMING_CHANGELOG_META = {
    'csi1000_timing': {
        'market_group': 'csi1000',
        'supersedes': '旧版均线趋势 + 中期动量',
        'changelog_title': 'CSI1000 生产策略升级为突破确认 + 趋势过滤 + MACD 辅助',
        'changelog_summary': '当前默认版本不再依赖单一的均线多头排列硬扛回撤，而是改为突破确认入场、跌破防线快速退出，更贴近 CSI1000 的高波动切换节奏。',
        'changelog_bullets': [
            '默认核心窗口切换为 breakout / exit / trend = 15 / 7 / 50，用状态机替代旧版趋势+动量开关。',
            '继续保留 staged 分批建仓，但把风险收缩逻辑前移，重点修复 2026-03 这类高仓位撤退过慢的问题。',
            '交易语义、ETF next-open 成交、next-close 估值、全历史 replay 后切片等共享规则保持不变。',
        ],
        'performance_delta': {
            'cumulative_return_diff': 0.1951,
            'annual_return_diff': 7.28,
            'max_drawdown_improvement': -2.79,
            'final_capital_diff': 9753.02,
        },
    },
}

FACTOR_OVERVIEW = [
    {
        'name': '市场因子',
        'core_fields': '市场组合收益率、无风险收益率',
        'sort_direction': '不适用',
        'long_short': '不适用',
        'double_sort': '不适用',
        'book_recommended': '是',
        'category': '风险归因',
    },
    {
        'name': '规模因子',
        'core_fields': '总市值',
        'sort_direction': '从低到高',
        'long_short': 'Small - Big',
        'double_sort': '否',
        'book_recommended': '是',
        'category': '核心选股',
    },
    {
        'name': '价值因子',
        'core_fields': 'BM',
        'sort_direction': '从低到高',
        'long_short': 'High - Low',
        'double_sort': '是',
        'book_recommended': '是',
        'category': '核心选股',
    },
    {
        'name': '动量因子',
        'core_fields': '过去 11 个月累计收益',
        'sort_direction': '从低到高',
        'long_short': 'High - Low',
        'double_sort': '是',
        'book_recommended': '否，A 股不稳',
        'category': '核心选股',
    },
    {
        'name': '盈利因子',
        'core_fields': 'ROE(TTM)',
        'sort_direction': '从低到高',
        'long_short': 'High - Low',
        'double_sort': '是',
        'book_recommended': '是',
        'category': '核心选股',
    },
    {
        'name': '投资因子',
        'core_fields': '总资产同比增长率',
        'sort_direction': '从低到高',
        'long_short': 'Low - High',
        'double_sort': '是，但仍受污染',
        'book_recommended': '否，证据较弱',
        'category': '核心选股',
    },
    {
        'name': '换手率因子',
        'core_fields': '异常换手率',
        'sort_direction': '从低到高',
        'long_short': 'Low - High',
        'double_sort': '是',
        'book_recommended': '是，A 股很强',
        'category': '交易行为',
    },
    {
        'name': '缠论背驰因子',
        'core_fields': '收盘价、MACD柱',
        'sort_direction': '从低到高',
        'long_short': 'High - Low',
        'double_sort': '是',
        'book_recommended': '否，缠论扩展',
        'category': '缠论扩展',
    },
    {
        'name': '缠论中枢位置因子',
        'core_fields': '最高价、最低价、收盘价',
        'sort_direction': '从低到高',
        'long_short': 'High - Low',
        'double_sort': '是',
        'book_recommended': '否，缠论扩展',
        'category': '缠论扩展',
    },
    {
        'name': '缠论分型因子',
        'core_fields': '最高价、最低价',
        'sort_direction': '从低到高',
        'long_short': 'High - Low',
        'double_sort': '是',
        'book_recommended': '否，缠论扩展',
        'category': '缠论扩展',
    },
    {
        'name': '缠论笔强度因子',
        'core_fields': '收盘价、涨跌幅_20',
        'sort_direction': '从低到高',
        'long_short': 'High - Low',
        'double_sort': '是',
        'book_recommended': '否，缠论扩展',
        'category': '缠论扩展',
    },
    {
        'name': '缠论买卖点信号因子',
        'core_fields': '收盘价、最高价、最低价、MACD',
        'sort_direction': '从低到高',
        'long_short': 'High - Low',
        'double_sort': '是',
        'book_recommended': '否，缠论扩展',
        'category': '缠论扩展',
    },
    {
        'name': 'BIAS偏离因子',
        'core_fields': 'bias_20 (20日偏离率)',
        'sort_direction': '从低到高（超跌优先）',
        'long_short': 'Low - High',
        'double_sort': '是',
        'book_recommended': '否，技术指标',
        'category': '技术指标',
        'single_factor_id': 'bias',
    },
    {
        'name': 'KDJ超卖因子',
        'core_fields': 'J值 (KDJ随机指标)',
        'sort_direction': '从低到高（超卖优先）',
        'long_short': 'Low - High',
        'double_sort': '否',
        'book_recommended': '否，技术指标',
        'category': '技术指标',
        'single_factor_id': 'kdj_j',
    },
    {
        'name': '市盈率倒数因子',
        'core_fields': '市盈率倒数 (EP)',
        'sort_direction': '从高到低（高EP优先）',
        'long_short': 'High - Low',
        'double_sort': '是',
        'book_recommended': '是，与BM有一定互补',
        'category': '核心选股',
        'single_factor_id': 'pe_inv',
    },
    {
        'name': '成交额波动因子',
        'core_fields': '成交额std_10 (10日成交额标准差)',
        'sort_direction': '从低到高（低波动优先）',
        'long_short': 'Low - High',
        'double_sort': '否',
        'book_recommended': '否，流动性衍生',
        'category': '交易行为',
        'single_factor_id': 'vol_stab',
    },
]

# Mapping from single_factor_id to FACTOR_OVERVIEW name (for frontend enrichment)
SINGLE_FACTOR_ID_TO_OVERVIEW_NAME = {
    'size':     '规模因子',
    'pb':       '价值因子',
    'profit':   '盈利因子',
    'momentum': '动量因子',
    'bias':     'BIAS偏离因子',
    'kdj_j':    'KDJ超卖因子',
    'pe_inv':   '市盈率倒数因子',
    'vol_stab': '成交额波动因子',
    'turn_abn': '换手率因子',
    'chan_div': '缠论背驰因子',
    'chan_zs':  '缠论中枢位置因子',
    'chan_fr':  '缠论分型因子',
    'chan_str': '缠论笔强度因子',
    'chan_sig': '缠论买卖点信号因子',
}

FOCUSED_STRATEGY_ID = 'original_ensemble'

# 训练/测试集拆分日期
SPLIT_DATE = pd.to_datetime('2026-03-31')
DEFAULT_BENCHMARK_ID = 'csi1000'


def _normalize_benchmark_id(benchmark_id):
    if benchmark_id in INDEX_CONFIGS:
        return benchmark_id
    if DEFAULT_BENCHMARK_ID in INDEX_CONFIGS:
        return DEFAULT_BENCHMARK_ID
    return next(iter(INDEX_CONFIGS.keys()), None)


def _get_benchmark_series(benchmark_id):
    normalized_id = _normalize_benchmark_id(benchmark_id)
    if normalized_id is None:
        return None, None
    series = INDEX_RETURNS_MAP.get(normalized_id)
    if series is not None:
        return normalized_id, series
    fallback_id = DEFAULT_BENCHMARK_ID if DEFAULT_BENCHMARK_ID in INDEX_RETURNS_MAP else None
    if fallback_id is not None:
        return fallback_id, INDEX_RETURNS_MAP[fallback_id]
    if INDEX_RETURNS_MAP:
        first_id = next(iter(INDEX_RETURNS_MAP.keys()))
        return first_id, INDEX_RETURNS_MAP[first_id]
    return normalized_id, None


def _get_benchmark_meta(benchmark_id):
    normalized_id = _normalize_benchmark_id(benchmark_id)
    if normalized_id is None:
        return None
    cfg = INDEX_CONFIGS.get(normalized_id, {})
    return {
        'id': normalized_id,
        'name': cfg.get('name', normalized_id),
    }


def _infer_market_label(code):
    code = str(code or '').strip()
    digits = ''.join(ch for ch in code if ch.isdigit())
    if digits.startswith(('688', '689')):
        return '科创板'
    if digits.startswith(('300', '301')):
        return '创业板'
    if digits.startswith(('600', '601', '603', '605')):
        return '上证主板'
    if digits.startswith(('000', '001', '002', '003')):
        return '深证主板'
    return '其他'


# Eager-load status: set by background thread on startup
_LOAD_STATUS = {
    'loading': False,
    'start_time': None,
    'end_time': None,
    'message': '等待启动',
    'stage': 'idle',
}
_DATA_READY = threading.Event()

# 数据更新状态追踪
_UPDATE_DATA_STATUS = {
    'running': False,
    'stage': 'idle',
    'message': '',
    'progress_pct': 0,
    'error': None,
}

# 指数数据更新状态追踪
_INDEX_UPDATE_STATUS = {'stage': 'idle', 'message': '', 'progress': 0, 'warning': None, 'details': None}


def ensure_index_returns_loaded():
    global INDEX_RETURNS, INDEX_RETURNS_MAP
    if INDEX_RETURNS_MAP:
        return
    index_returns_map = {}
    for index_id, cfg in INDEX_CONFIGS.items():
        try:
            series = get_index_returns(index_id=index_id)
            index_returns_map[index_id] = series
            print(f"[init] {cfg['name']} 指数收益加载完成，{len(series)} 个月")
        except Exception as e:
            print(f"[WARN] 无法加载 {cfg['name']} 指数收益: {e}")
    INDEX_RETURNS_MAP = index_returns_map
    INDEX_RETURNS = INDEX_RETURNS_MAP.get('csi1000')
    if INDEX_RETURNS is None:
        print('[WARN] CSI 1000 不可用，业绩归因功能将不可用')


def ensure_stock_data_loaded():
    global DATA_DF
    if DATA_DF is not None:
        return
    # If eager loading is in progress, wait for it to finish
    if _LOAD_STATUS.get('loading'):
        _DATA_READY.wait()
        if DATA_DF is not None:
            return
    # Fallback: load synchronously if eager loading didn't happen
    csv_path = os.path.join(os.path.dirname(__file__), 'stock_data.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'数据文件不存在: {csv_path}')
    print('[init] 加载数据中 (823MB)...')
    DATA_DF = load_data(csv_path)
    print('[init] 数据加载完成')


def init_cache():
    """按需预热选股缓存（优先从磁盘加载，仅预热当前使用的策略）。"""
    global CSI1000_SIGNAL_SERIES
    ensure_index_returns_loaded()
    ensure_stock_data_loaded()
    # 确保 CSI1000 择时信号已加载（用于选股门控）
    if CSI1000_SIGNAL_SERIES is None:
        try:
            ensure_timing_panel_loaded()
            csi_strategy = build_timing_strategy('csi1000_timing')
            signal_df = csi_strategy.run(TIMING_PANEL.copy())
            CSI1000_SIGNAL_SERIES = pd.Series(
                pd.to_numeric(signal_df['target_exposure'], errors='coerce').fillna(0.0).values,
                index=pd.to_datetime(signal_df['交易日期']),
            ).sort_index()
            print('[init] CSI1000 择时信号预加载完成，共 {} 条'.format(len(CSI1000_SIGNAL_SERIES)))
        except Exception as e:
            print(f'[WARN] CSI1000 择时信号加载失败: {e}')

    # 尝试磁盘缓存
    if not BACKTEST_CACHE and _load_disk_cache():
        return  # 磁盘缓存命中，无需重算

    if BACKTEST_CACHE:
        return

    sid = get_focused_strategy_id()
    cls = STRATEGY_MAP.get(sid)
    if cls is None:
        return
    try:
        print(f"[init] 预运行 {sid} 策略...")
        s = cls()
        df = s.run(DATA_DF.copy())
        result = select_and_backtest(df, s,
                                     c_rate=s.c_rate,
                                     t_rate=s.t_rate,
                                     bull_tp=s.bull_tp,
                                     bear_tp=s.bear_tp,
                                     bull_n=s.bull_n,
                                     bear_n=s.bear_n,
                                     initial_capital=s.initial_capital)
        if hasattr(s, '_profile_summary'):
            result.attrs['strategy_meta'] = {
                'profile_summary': getattr(s, '_profile_summary', []),
            }
            _PROFILE_SUMMARY_CACHE[sid] = getattr(s, '_profile_summary', [])
        ev = strategy_evaluate(result, index_returns=INDEX_RETURNS)
        BACKTEST_CACHE[sid] = (result, ev)
        print(f"[init] {sid} 完成, 累积净值: {result['累积净值'].iloc[-1]:.2f}")
        _save_disk_cache()
    except Exception as e:
        print(f"[init] {sid} 失败: {e}")


def ensure_timing_panel_loaded():
    global TIMING_PANEL
    ensure_index_returns_loaded()
    if TIMING_PANEL is not None:
        return
    try:
        TIMING_PANEL = build_index_panel()
    except Exception as e:
        print(f"[WARN] 无法加载指数日线面板: {e}")
        TIMING_PANEL = None
        raise


def ensure_us_timing_panel_loaded():
    global US_TIMING_PANEL
    if US_TIMING_PANEL is not None:
        return
    try:
        US_TIMING_PANEL = build_us_index_panel()
        print(f"[init] 美股ETF面板加载完成: {len(US_TIMING_PANEL)} 行")
    except Exception as e:
        print(f"[WARN] 无法加载美股ETF面板: {e}")
        US_TIMING_PANEL = None
        raise


def build_us_timing_strategy(strategy_name='macro_v32_timing', **params):
    strat_cls = US_TIMING_STRATEGY_MAP.get(strategy_name, MacroV32TimingStrategy)
    sig = inspect.signature(strat_cls.__init__)
    init_keys = set(sig.parameters.keys())
    merged_params = dict(_US_TIMING_CACHE_DEFAULTS.get(strategy_name, {}))
    # Phase 2: best_profile 覆盖出厂默认（walk-forward 选出的最优参数）
    best_profile = _load_best_profile(strategy_name)
    if best_profile is not None:
        merged_params.update(best_profile.get('all_params', {}))
    merged_params.update({k: v for k, v in params.items() if v is not None})

    init_params = {k: v for k, v in merged_params.items()
                   if k in init_keys and k != 'self' and v is not None}
    deferred_params = {k: v for k, v in merged_params.items()
                       if k in _REALISM_ALL_KEYS and k not in init_keys and v is not None}

    instance = strat_cls(**init_params)
    for k, v in deferred_params.items():
        if k == 'profit_lock_enabled':
            v = bool(v)
        elif k == 'limit_max_delay_days':
            v = max(int(v or 0), 0)
        elif k in {'slippage_bps', 'cash_interest_rate', 'commission_rate',
                   'commission_min', 'stamp_tax_rate', 'transfer_fee_rate',
                   'profit_lock_drawdown'}:
            v = max(float(v or 0.0), 0.0)
        else:
            v = float(v)
        setattr(instance, k, v)
    return instance


def init_us_timing_cache():
    """从离线缓存加载美股择时回测结果。

    缓存由 `scripts/build_us_timing_cache.py` 离线生成，写入
    `stock_trade_demo/.cache/us_timing/<strategy_id>.pkl`。
    web 层只读盘，不再在请求路径上重新跑回测。
    """
    global US_TIMING_CACHE
    if US_TIMING_CACHE:
        return
    cache_dir = os.path.join(_CACHE_DIR, 'us_timing')
    US_TIMING_CACHE = {}
    missing = []
    for sid in US_TIMING_STRATEGY_MAP.keys():
        pkl_path = os.path.join(cache_dir, f"{sid}.pkl")
        if not os.path.exists(pkl_path):
            missing.append(sid)
            print(f"[init] {sid} 美股择时缓存缺失: {pkl_path}")
            continue
        try:
            with open(pkl_path, 'rb') as f:
                US_TIMING_CACHE[sid] = pickle.load(f)
            print(f"[init] {sid} 美股择时缓存载入, 累积净值: {US_TIMING_CACHE[sid]['累积净值'].iloc[-1]:.4f}")
        except Exception as e:
            missing.append(sid)
            print(f"[init] {sid} 美股择时缓存载入失败: {e}")
    if missing:
        print(f"[init] 请运行: python scripts/build_us_timing_cache.py 重建缓存 (缺失: {missing})")


def init_timing_cache():
    ensure_timing_panel_loaded()
    global TIMING_CACHE

    if not TIMING_CACHE:
        _load_disk_cache()  # 尝试从磁盘恢复（init_cache 可能已加载，此处做兜底）

    if TIMING_CACHE:
        return

    TIMING_CACHE = {}
    for sid, cls in TIMING_STRATEGY_MAP.items():
        try:
            strategy = build_timing_strategy(sid)
            signal_df = strategy.run(TIMING_PANEL.copy())
            # 捕获 CSI1000 原始择时信号（用于选股门控）
            if sid == 'csi1000_timing':
                global CSI1000_SIGNAL_SERIES
                CSI1000_SIGNAL_SERIES = pd.Series(
                    pd.to_numeric(signal_df['target_exposure'], errors='coerce').fillna(0.0).values,
                    index=pd.to_datetime(signal_df['交易日期']),
                ).sort_index()
            result = run_timing_backtest(signal_df, strategy, benchmark_returns=INDEX_RETURNS_MAP.get(strategy.get_index_id()))
            TIMING_CACHE[sid] = result
            print(f"[init] {sid} 择时策略完成, 累积净值: {result['累积净值'].iloc[-1]:.2f}, exposure_mode={getattr(strategy, 'exposure_mode', 'binary')}")
        except Exception as e:
            print(f"[init] {sid} 择时策略失败: {e}")
    _save_disk_cache()


def _get_csi1000_timing_gate():
    """返回 CSI1000 择时信号 Series（日线索引），用于选股月度门控。None 表示未加载。"""
    return CSI1000_SIGNAL_SERIES if CSI1000_SIGNAL_SERIES is not None and len(CSI1000_SIGNAL_SERIES) > 0 else None


def build_strategy(strategy_name='original', **params):
    strat_cls = STRATEGY_MAP.get(strategy_name, OriginalStrategy)
    sig = inspect.signature(strat_cls.__init__)
    valid_params = {k: v for k, v in params.items()
                    if k in sig.parameters and v is not None}
    valid_params.pop('self', None)
    return strat_cls(**valid_params)


def _load_best_profile(strategy_name):
    """读取 strategy/best_profile_{name}.json。缺失返回 None。内存缓存。"""
    if strategy_name in _BEST_PROFILE_CACHE:
        return _BEST_PROFILE_CACHE[strategy_name]
    fp = os.path.join(_BEST_PROFILE_DIR, f'best_profile_{strategy_name}.json')
    if not os.path.exists(fp):
        _BEST_PROFILE_CACHE[strategy_name] = None
        return None
    try:
        with open(fp) as f:
            profile = json.load(f)
        _BEST_PROFILE_CACHE[strategy_name] = profile
        return profile
    except Exception as exc:
        print(f"[WARN] 读取 {fp} 失败: {exc}")
        _BEST_PROFILE_CACHE[strategy_name] = None
        return None


def _load_holdout_report(strategy_name):
    """读取 strategy/holdout_report_{name}.json（build_holdout_reports.py 产出）。
    mtime 失效；缺失返回 None。前端「策略验证概览」面板用此数据。"""
    fp = os.path.join(_BEST_PROFILE_DIR, f'holdout_report_{strategy_name}.json')
    if not os.path.exists(fp):
        _HOLDOUT_REPORT_CACHE[strategy_name] = {'data': None, 'mtime': 0}
        return None
    mtime = os.path.getmtime(fp)
    cache = _HOLDOUT_REPORT_CACHE.get(strategy_name)
    if cache and cache.get('mtime') == mtime and cache.get('data') is not None:
        return cache['data']
    try:
        with open(fp, encoding='utf-8') as f:
            data = json.load(f)
        _HOLDOUT_REPORT_CACHE[strategy_name] = {'data': data, 'mtime': mtime}
        return data
    except Exception as exc:
        print(f"[WARN] 读取 {fp} 失败: {exc}")
        _HOLDOUT_REPORT_CACHE[strategy_name] = {'data': None, 'mtime': 0}
        return None


def _load_risk_signals():
    """读取 strategy/risk_signals.json。按 mtime 失效；缺失返回 None。"""
    if not os.path.exists(_RISK_SIGNALS_FILE):
        _RISK_SIGNALS_CACHE['data'] = None
        _RISK_SIGNALS_CACHE['mtime'] = 0
        return None
    mtime = os.path.getmtime(_RISK_SIGNALS_FILE)
    if mtime == _RISK_SIGNALS_CACHE['mtime'] and _RISK_SIGNALS_CACHE['data'] is not None:
        return _RISK_SIGNALS_CACHE['data']
    try:
        with open(_RISK_SIGNALS_FILE, encoding='utf-8') as f:
            data = json.load(f)
        _RISK_SIGNALS_CACHE['data'] = data
        _RISK_SIGNALS_CACHE['mtime'] = mtime
        return data
    except Exception as exc:
        print(f"[WARN] 读取 {_RISK_SIGNALS_FILE} 失败: {exc}")
        _RISK_SIGNALS_CACHE['data'] = None
        return None


def get_best_profile_view(strategy_name):
    """前端展示用的精简 profile（只读）。"""
    profile = _load_best_profile(strategy_name)
    if profile is None:
        return None
    return {
        'strategy_id': strategy_name,
        'training_cutoff': profile.get('training_cutoff'),
        'generated_at': profile.get('generated_at'),
        'score': profile.get('score'),
        'score_formula': profile.get('score_formula'),
        'maxdd_threshold': profile.get('maxdd_threshold'),
        'tuned_params': profile.get('tuned_params', {}),
        'window_metrics': profile.get('window_metrics', {}),
    }


def build_timing_strategy(strategy_name='csi1000_timing', **params):
    strat_cls = TIMING_STRATEGY_MAP.get(strategy_name, CSI1000TimingStrategy)
    sig = inspect.signature(strat_cls.__init__)
    init_keys = set(sig.parameters.keys())
    merged_params = _get_timing_default_params(strategy_name)
    # Phase 2: best_profile 覆盖出厂默认（walk-forward 选出的最优参数）
    best_profile = _load_best_profile(strategy_name)
    if best_profile is not None:
        merged_params.update(best_profile.get('all_params', {}))
    merged_params.update({k: v for k, v in params.items() if v is not None})

    # 子类 __init__ 显式声明的参数走构造函数；
    # 子类未显式声明但属于 12 个真实交易规则参数的（slippage_bps / cash_interest_rate / ...）
    # 在实例化之后通过 setattr 注入到实例上——回测引擎读取这些值都用 getattr(strategy, key, default)，
    # 因此不需要进 __init__ 也能生效。
    init_params = {k: v for k, v in merged_params.items()
                   if k in init_keys and k != 'self' and v is not None}
    deferred_params = {k: v for k, v in merged_params.items()
                       if k in _REALISM_ALL_KEYS and k not in init_keys and v is not None}

    instance = strat_cls(**init_params)
    for k, v in deferred_params.items():
        # 与 BaseTimingStrategy.__init__ 的归一化逻辑保持一致
        if k == 'profit_lock_enabled':
            v = bool(v)
        elif k == 'limit_max_delay_days':
            v = max(int(v or 0), 0)
        elif k in {'slippage_bps', 'cash_interest_rate', 'commission_rate',
                   'commission_min', 'stamp_tax_rate', 'transfer_fee_rate',
                   'profit_lock_drawdown'}:
            v = max(float(v or 0.0), 0.0)
        else:
            v = float(v)
        setattr(instance, k, v)
    return instance


def run_timing_backtest_fresh(strategy_name='csi1000_timing', benchmark_id=None, **params):
    ensure_timing_panel_loaded()
    strategy = build_timing_strategy(strategy_name, **params)
    _, benchmark_series = _get_benchmark_series(benchmark_id)
    signal_df = strategy.run(TIMING_PANEL.copy())
    result = run_timing_backtest(signal_df, strategy, benchmark_returns=benchmark_series)
    metrics = evaluate_timing_result(result, benchmark_returns=benchmark_series)
    return result, metrics, strategy


def _build_timing_compare_payload(result_df, metrics, benchmark_series, params=None, start_date=None, end_date=None):
    payload = {
        'metrics': metrics,
        'interval_windows': summarize_timing_windows(result_df, benchmark_returns=benchmark_series),
    }
    if params is not None:
        payload['params'] = params
    if start_date or end_date:
        sliced = filter_timing_result(result_df, start_date=start_date, end_date=end_date)
        payload['current_range'] = {
            'start': start_date,
            'end': end_date,
            'rows': len(sliced),
            'metrics': evaluate_timing_result(sliced, benchmark_returns=benchmark_series) if len(sliced) else {},
        }
    return payload


def get_focused_strategy_id():
    return FOCUSED_STRATEGY_ID if FOCUSED_STRATEGY_ID in STRATEGY_MAP else 'original'


def build_factor_overview_payload(strategy_name=None):
    strategy_id = strategy_name or get_focused_strategy_id()
    strategy = build_strategy(strategy_id)

    # 优先使用 init_cache 预热时已算好的 profile_summary
    if strategy_id in _PROFILE_SUMMARY_CACHE:
        setattr(strategy, '_profile_summary', _PROFILE_SUMMARY_CACHE[strategy_id])
    elif hasattr(strategy, '_resolve_profiles') and DATA_DF is not None and not getattr(strategy, '_profile_summary', None):
        try:
            strategy._resolve_profiles(DATA_DF)
        except Exception:
            pass
    active_tags = set(strategy.get_factor_overview_tags() or [])
    items = []
    for row in FACTOR_OVERVIEW:
        item = dict(row)
        item['active'] = item['name'] in active_tags
        items.append(item)
    payload = {
        'strategy_id': strategy_id,
        'strategy_name': strategy.get_display_name(),
        'active_factor_names': sorted(active_tags),
        'items': items,
        'single_factor_id_map': SINGLE_FACTOR_ID_TO_OVERVIEW_NAME,
    }
    if hasattr(strategy, '_profile_summary'):
        payload['profile_summary'] = getattr(strategy, '_profile_summary', [])
    return payload


def run_backtest_fresh(strategy_name='original', benchmark_id=None, **params):
    """重新运行回测（参数不同于默认值时使用）。返回: (result_df, eval_df)"""
    if DATA_DF is None:
        raise RuntimeError("数据未加载，请确认 stock_data.csv 存在")

    strategy = build_strategy(strategy_name, **params)
    _, benchmark_series = _get_benchmark_series(benchmark_id)

    df = strategy.run(DATA_DF.copy())
    result = select_and_backtest(df, strategy,
                                 c_rate=strategy.c_rate,
                                 t_rate=strategy.t_rate,
                                 bull_tp=strategy.bull_tp,
                                 bear_tp=strategy.bear_tp,
                                 bull_n=strategy.bull_n,
                                 bear_n=strategy.bear_n,
                                 initial_capital=strategy.initial_capital)
    if hasattr(strategy, '_profile_summary'):
        result.attrs['strategy_meta'] = {
            'profile_summary': getattr(strategy, '_profile_summary', []),
        }
    ev = strategy_evaluate(result, index_returns=benchmark_series)
    return result, ev



def filter_by_date(result, start_date, end_date, benchmark_id=None):
    """按日期范围过滤回测结果并重新计算累积净值和评估指标。
    初始资金始终重置为 100,000——自定义日期范围视为独立回测区间。"""
    original = result.copy()
    original_dates = list(original['交易日期'])
    _, benchmark_series = _get_benchmark_series(benchmark_id)
    if start_date:
        result = result[result['交易日期'] >= pd.to_datetime(start_date)].copy()
    if end_date:
        result = result[result['交易日期'] <= pd.to_datetime(end_date)].copy()
    if len(result) == 0:
        return None, None

    # 重新计算累积净值
    result['累积净值'] = (1 + result['选股下周期涨跌幅']).cumprod()
    result['资金曲线'] = result['累积净值']

    # 自定义日期范围：初始资金重置为 100,000
    start_capital = float(result.attrs.get('initial_capital', 100000))

    capital = float(start_capital)
    capitals = []
    pnls = []
    cum_caps = []
    for _, row in result.iterrows():
        capitals.append(capital)
        pnl = capital * row['选股下周期涨跌幅']
        pnls.append(pnl)
        capital += pnl
        cum_caps.append(capital)

    result['当期本金'] = capitals
    result['当期盈亏'] = pnls
    result['累计资金'] = cum_caps
    result.attrs['initial_capital'] = start_capital

    if 'period_daily_curves' in original.attrs:
        curve_lookup = {
            pd.to_datetime(dt).strftime('%Y-%m-%d'): curve
            for dt, curve in zip(original_dates, original.attrs.get('period_daily_curves', []))
        }
        result.attrs['period_daily_curves'] = [
            curve_lookup.get(pd.to_datetime(dt).strftime('%Y-%m-%d'), [])
            for dt in result['交易日期']
        ]
    if 'daily_equity_curve' in original.attrs:
        result.attrs['daily_equity_curve'] = _build_daily_curve_slice(result, original.attrs.get('daily_equity_curve', []))

    ev = strategy_evaluate(result, initial_capital=start_capital,
                          index_returns=benchmark_series)
    return result, ev


def _resample_curve(curve, resolution):
    """将月度曲线重采样为季线/年线，返回重采样后的曲线列表。"""
    if not curve or resolution == 'month':
        return curve
    groups = {}
    for d in curve:
        dt = pd.to_datetime(d['date'])
        if resolution == 'quarter':
            key = f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
        else:
            key = str(dt.year)
        if key not in groups:
            groups[key] = {'returns': [], 'last_date': d['date']}
        groups[key]['returns'].append(d.get('return', 0))
        groups[key]['last_date'] = d['date']
    result = []
    cum = 1.0
    for g in groups.values():
        period_ret = float(np.prod([1 + r for r in g['returns']]) - 1)
        cum *= (1 + period_ret)
        result.append({
            'date': g['last_date'],
            'value': round(cum, 4),
            'return': round(period_ret, 6),
        })
    return result


def _load_trading_calendar(benchmark_id=None):
    try:
        calendar_df = get_a_share_trading_calendar()
        if calendar_df is not None and len(calendar_df) > 0 and 'date' in calendar_df.columns:
            dates = pd.to_datetime(calendar_df['date'], errors='coerce').dropna().sort_values().unique()
            if len(dates) > 0:
                return pd.DatetimeIndex(dates)
    except Exception:
        pass

    normalized_id = _normalize_benchmark_id(benchmark_id)
    candidate_ids = []
    if normalized_id in A_SHARE_INDEX_IDS:
        candidate_ids.append(normalized_id)
    if DEFAULT_BENCHMARK_ID not in candidate_ids:
        candidate_ids.append(DEFAULT_BENCHMARK_ID)
    for index_id in A_SHARE_INDEX_IDS:
        if index_id not in candidate_ids:
            candidate_ids.append(index_id)

    best_dates = pd.DatetimeIndex([])
    for index_id in candidate_ids:
        try:
            df = get_index_daily(index_id)
        except Exception:
            continue
        if df is None or len(df) == 0 or 'date' not in df.columns:
            continue
        dates = pd.to_datetime(df['date'], errors='coerce').dropna().sort_values().unique()
        if len(dates) > len(best_dates):
            best_dates = pd.DatetimeIndex(dates)
        elif len(dates) == len(best_dates) and len(dates) > 0 and pd.to_datetime(dates[-1]) > pd.to_datetime(best_dates[-1]):
            best_dates = pd.DatetimeIndex(dates)
    return best_dates



def _resolve_period_trading_dates(trade_date, period_curve, trading_calendar):
    trade_ts = pd.to_datetime(trade_date)
    period_len = len(period_curve or [])
    if period_len <= 0:
        return []
    if trading_calendar is None or len(trading_calendar) == 0:
        return []

    future_dates = trading_calendar[trading_calendar > trade_ts]
    if len(future_dates) == 0:
        return []
    return [pd.to_datetime(d) for d in future_dates[:period_len]]



def _build_holding_date_range(trade_date, period_curve, trading_calendar):
    trade_label = pd.to_datetime(trade_date).strftime('%Y-%m-%d')
    trading_dates = _resolve_period_trading_dates(trade_date, period_curve, trading_calendar)
    if trading_dates:
        start_label = trading_dates[0].strftime('%Y-%m-%d')
        end_label = trading_dates[-1].strftime('%Y-%m-%d')
    else:
        start_label = trade_label
        end_label = trade_label
    return {
        'holding_start_date': start_label,
        'holding_end_date': end_label,
        'holding_date_range_label': f'{start_label} → {end_label}',
    }



def _is_open_snapshot_period(raw_stocks):
    return bool(raw_stocks) and all(stock.get('sell_price') is None for stock in raw_stocks)



def _build_daily_curve_slice(result_df, full_daily_curve, base_value=1.0, trading_calendar=None):
    """按结果区间切分并重置日线净值基准。"""
    if not full_daily_curve or len(result_df) == 0:
        return []

    period_curves = result_df.attrs.get('period_daily_curves', [])
    if period_curves:
        daily_curve = []
        running_value = float(base_value)
        for period_idx, period_curve in enumerate(period_curves):
            trade_date = pd.to_datetime(result_df.iloc[period_idx]['交易日期'])
            trading_dates = _resolve_period_trading_dates(trade_date, period_curve, trading_calendar)
            if not period_curve:
                daily_curve.append({
                    'date': trade_date.strftime('%Y-%m-%d'),
                    'value': round(running_value, 6),
                    'return': 0.0,
                })
                continue
            prev = 1.0
            for day_idx, period_value in enumerate(period_curve, start=1):
                day_ret = 0.0 if prev == 0 else float(period_value / prev - 1)
                running_value *= (1 + day_ret)
                curve_date = trading_dates[day_idx - 1].strftime('%Y-%m-%d') if day_idx - 1 < len(trading_dates) else (trade_date + pd.Timedelta(days=day_idx)).strftime('%Y-%m-%d')
                daily_curve.append({
                    'date': curve_date,
                    'value': round(running_value, 6),
                    'return': round(day_ret, 6),
                })
                prev = period_value
        return daily_curve

    first_value = float(full_daily_curve[0].get('value', base_value))
    if first_value == 0:
        first_value = base_value

    return [{
        'date': p.get('date'),
        'value': round(float(p.get('value', 1.0)) / first_value * base_value, 6),
        'return': round(float(p.get('return', 0.0)), 6),
    } for p in full_daily_curve]


def _safe_float_or_none(value):
    if value in (None, ''):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None



def _normalize_stock_code(code):
    text = str(code or '').strip().lower()
    if not text:
        return ''
    if text.startswith(('sh', 'sz', 'bj')):
        return text[2:]
    return text



def _extract_open_stock_codes(result):
    if result is None or len(result) == 0 or '买入个股收益' not in result.columns:
        return []
    codes = set()
    for raw in result['买入个股收益']:
        try:
            stocks = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        for stock in stocks or []:
            code = str(stock.get('code', '')).strip()
            if code and stock.get('sell_price') is None:
                codes.add(code)
    return sorted(codes)



def _fetch_open_stock_quotes(result):
    codes = _extract_open_stock_codes(result)
    if not codes:
        return {}
    try:
        return _fetch_realtime_quotes_batch(codes)
    except Exception:
        return {}



def _build_stock_payload(raw_stock, cap_per_stock, period_capital, stock_count, quote_map=None, allow_open_position=False):
    quote_map = quote_map or {}
    ret = float(raw_stock.get('return', 0) or 0)
    buy_price = _safe_float_or_none(raw_stock.get('buy_price')) or 0.0
    sell_price = _safe_float_or_none(raw_stock.get('sell_price'))
    code = raw_stock.get('code', '')
    if buy_price > 0:
        shares = int(cap_per_stock / buy_price / 100) * 100
        actual_invested = shares * buy_price
    else:
        shares = 0
        actual_invested = cap_per_stock

    normalized_code = _normalize_stock_code(code)
    quote = quote_map.get(code) or quote_map.get(normalized_code) or {}
    realtime_price = _safe_float_or_none(quote.get('latest'))
    is_open = bool(allow_open_position and sell_price is None)
    if is_open and realtime_price is not None:
        latest_price = realtime_price
        price_source = 'realtime'
    elif sell_price is not None:
        latest_price = sell_price
        price_source = 'exit'
    elif is_open and buy_price > 0:
        latest_price = buy_price
        price_source = 'buy_fallback'
    else:
        latest_price = None
        price_source = None

    position_market_value = round(float(shares * latest_price), 2) if shares > 0 and latest_price is not None else None
    position_weight = round(position_market_value / period_capital, 6) if position_market_value is not None and period_capital > 0 else raw_stock.get('weight', round(1.0 / stock_count, 4))

    if actual_invested > 0:
        backtest_pnl = round(float(actual_invested * ret), 2)
    else:
        backtest_pnl = round(float(cap_per_stock * ret), 2)

    display_pnl = backtest_pnl
    if is_open and position_market_value is not None and actual_invested > 0:
        display_pnl = round(float(position_market_value - actual_invested), 2)

    return {
        'code': code,
        'name': raw_stock.get('name', ''),
        'weight': raw_stock.get('weight', round(1.0 / stock_count, 4)),
        'position_weight': position_weight,
        'return': ret,
        'pnl': backtest_pnl,
        'display_pnl': display_pnl,
        'buy_price': buy_price,
        'sell_price': sell_price,
        'exit_price': sell_price,
        'latest_price': round(float(latest_price), 2) if latest_price is not None else None,
        'is_open': is_open,
        'price_source': price_source,
        'shares': shares,
        'position_market_value': position_market_value,
        'factor_score': raw_stock.get('factor_score'),
        'rank': raw_stock.get('rank'),
        'industry_l2': raw_stock.get('industry_l2', ''),
        'market_label': raw_stock.get('market_label') or _infer_market_label(code),
        'pe': raw_stock.get('pe'),
        'pb': raw_stock.get('pb'),
        'market_cap': raw_stock.get('market_cap'),
        'selection_reason_summary': raw_stock.get('selection_reason_summary', ''),
        'selection_reason_detail': raw_stock.get('selection_reason_detail', []),
        'selection_fundamentals': raw_stock.get('selection_fundamentals', []),
        'selection_factor_breakdown': raw_stock.get('selection_factor_breakdown', []),
    }



def _build_holdings_payload(df, default_capital, quote_map=None, trading_calendar=None):
    holdings = []
    if df is None or len(df) == 0:
        return holdings
    quote_map = quote_map or {}
    period_curves = df.attrs.get('period_daily_curves', [])
    last_open_snapshot_idx = None
    for row_idx, (_, row) in enumerate(df.iterrows()):
        raw_stocks = []
        if '买入个股收益' in df.columns:
            try:
                raw_stocks = json.loads(row['买入个股收益'])
            except (json.JSONDecodeError, TypeError):
                pass
        if _is_open_snapshot_period(raw_stocks):
            last_open_snapshot_idx = row_idx

    for row_idx, (_, row) in enumerate(df.iterrows()):
        raw_stocks = []
        if '买入个股收益' in df.columns:
            try:
                raw_stocks = json.loads(row['买入个股收益'])
            except (json.JSONDecodeError, TypeError):
                pass
        is_open_snapshot = _is_open_snapshot_period(raw_stocks)
        if is_open_snapshot and row_idx != last_open_snapshot_idx:
            continue
        allow_open_position = is_open_snapshot and row_idx == last_open_snapshot_idx
        period_capital = float(row.get('当期本金', default_capital))
        n = len(raw_stocks) if raw_stocks else 1
        stocks = []
        for s in raw_stocks:
            target_weight = float(s.get('weight', round(1.0 / n, 4))) if n > 0 else 0
            cap_per_stock = period_capital * target_weight
            stocks.append(_build_stock_payload(s, cap_per_stock, period_capital, n, quote_map=quote_map, allow_open_position=allow_open_position))
        if not stocks:
            codes = str(row.get('买入股票代码', '')).strip().split()
            names = str(row.get('买入股票名称', '')).strip().split()
            stocks = [{
                'code': c,
                'name': names[i] if i < len(names) else '',
                'weight': round(1.0 / len(codes), 4) if codes else 0,
                'position_weight': round(1.0 / len(codes), 4) if codes else 0,
                'return': 0,
                'pnl': 0,
                'display_pnl': 0,
                'buy_price': 0,
                'sell_price': None,
                'exit_price': None,
                'latest_price': None,
                'is_open': False,
                'price_source': None,
                'shares': 0,
                'position_market_value': None,
                'factor_score': None,
                'rank': None,
                'industry_l2': '',
                'pe': None,
                'pb': None,
                'market_cap': None,
            } for i, c in enumerate(codes)]
        has_open_position = any(bool(s.get('is_open')) for s in stocks)
        display_period_pnl = round(sum(float(s.get('display_pnl', 0) or 0) for s in stocks), 2) if stocks else round(float(row.get('当期盈亏', 0)), 2)
        period_curve = period_curves[row_idx] if row_idx < len(period_curves) else []
        holding_range = _build_holding_date_range(row['交易日期'], period_curve, trading_calendar)
        holdings.append({
            'date': row['交易日期'].strftime('%Y-%m-%d'),
            'period_return': round(float(row['选股下周期涨跌幅']), 6),
            'period_pnl': round(float(row.get('当期盈亏', 0)), 2),
            'display_period_pnl': display_period_pnl,
            'period_pnl_label': '当前浮盈亏' if has_open_position else '持仓盈亏',
            'capital': round(period_capital, 2),
            'stocks': stocks,
            'stock_count': len(stocks),
            'benchmark_returns': _build_period_benchmark_returns(row['交易日期'], INDEX_RETURNS_MAP),
            **holding_range,
        })
    holdings.reverse()
    return holdings


def _compute_single_benchmark_curve(result, index_returns):
    if index_returns is None:
        return []
    lookup = build_period_lookup(index_returns)
    bm = []
    cum = 1.0
    for _, row in result.iterrows():
        idx_ret = get_index_return_for_date(row['交易日期'], lookup)
        cum *= (1 + idx_ret)
        bm.append({
            'date': row['交易日期'].strftime('%Y-%m-%d'),
            'value': round(cum, 4),
        })
    return bm


def _compute_benchmark_curves(result, index_returns_map):
    curves = []
    for index_id, cfg in INDEX_CONFIGS.items():
        series = (index_returns_map or {}).get(index_id)
        if series is None:
            continue
        curves.append({
            'id': index_id,
            'name': cfg['name'],
            'curve': _compute_single_benchmark_curve(result, series),
        })
    return curves


def _build_period_benchmark_returns(trade_date, index_returns_map=None):
    benchmark_returns = []
    for index_id, cfg in INDEX_CONFIGS.items():
        series = (index_returns_map or {}).get(index_id)
        if series is None:
            continue
        period_lookup = build_period_lookup(series)
        benchmark_returns.append({
            'id': index_id,
            'name': cfg['name'],
            'return': round(float(get_index_return_for_date(trade_date, period_lookup)), 6),
        })
    return benchmark_returns


def _build_etf_monthly_returns(result_df):
    if result_df is None or len(result_df) == 0 or 'etf_close' not in result_df.columns:
        return None
    etf = result_df[['交易日期', 'etf_close']].copy()
    etf['交易日期'] = pd.to_datetime(etf['交易日期'])
    etf['etf_close'] = pd.to_numeric(etf['etf_close'], errors='coerce')
    etf = etf.dropna(subset=['交易日期', 'etf_close'])
    etf = etf[etf['etf_close'] > 0].drop_duplicates(subset=['交易日期']).sort_values('交易日期')
    if len(etf) < 2:
        return None
    daily_returns = etf.set_index('交易日期')['etf_close'].pct_change().dropna()
    if len(daily_returns) == 0:
        return None
    monthly_returns = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1).dropna()
    return monthly_returns if len(monthly_returns) else None


def _month_start_from_end(end_date, months):
    end_ts = pd.to_datetime(end_date)
    start_ts = (end_ts - pd.DateOffset(months=months)) + pd.Timedelta(days=1)
    return start_ts.normalize()


def build_selection_interval_windows(result, index_returns=None, benchmark_id=None, quote_map=None):
    if len(result) == 0:
        return {}

    trading_calendar = _load_trading_calendar(benchmark_id)
    full_result = result.copy().reset_index(drop=True)
    full_start = pd.to_datetime(full_result['交易日期'].min())
    full_end = pd.to_datetime(full_result['交易日期'].max())
    recent_6m_start = _month_start_from_end(full_end, 6)

    windows = {
        'pre_6m_history': (full_start, recent_6m_start - pd.Timedelta(days=1), False),
        'recent_6m': (_month_start_from_end(full_end, 6), full_end, True),
        'recent_1q': (_month_start_from_end(full_end, 3), full_end, True),
        'recent_1m': (_month_start_from_end(full_end, 1), full_end, True),
    }

    initial_capital = float(full_result.attrs.get('initial_capital', 100000))
    period_curves = full_result.attrs.get('period_daily_curves', [])
    curve_lookup = {
        pd.to_datetime(dt).strftime('%Y-%m-%d'): curve
        for dt, curve in zip(full_result['交易日期'], period_curves)
    } if period_curves else {}

    summary = {}
    for name, (start_date, end_date, reset_capital) in windows.items():
        df = full_result[(full_result['交易日期'] >= pd.to_datetime(start_date)) & (full_result['交易日期'] <= pd.to_datetime(end_date))].copy()
        if len(df) == 0:
            summary[name] = {
                'label': {
                    'pre_6m_history': '半年前历史',
                    'recent_6m': '近半年',
                    'recent_1q': '近一季',
                    'recent_1m': '近一月',
                }.get(name, name),
                'months': 0,
                'reset_capital': reset_capital,
                'date_range': {'start': None, 'end': None},
                'metrics': {},
                'holdings': [],
                'benchmark_curves': [],
                'daily_equity_curve': [],
            }
            continue

        if curve_lookup:
            df.attrs['period_daily_curves'] = [
                curve_lookup.get(pd.to_datetime(dt).strftime('%Y-%m-%d'), [])
                for dt in df['交易日期']
            ]

        df['累积净值'] = (1 + df['选股下周期涨跌幅']).cumprod()
        capital = float(initial_capital)
        capitals, pnls, cum_caps = [], [], []
        for _, row in df.iterrows():
            capitals.append(capital)
            pnl = capital * row['选股下周期涨跌幅']
            pnls.append(pnl)
            capital += pnl
            cum_caps.append(capital)
        df['当期本金'] = capitals
        df['当期盈亏'] = pnls
        df['累计资金'] = cum_caps

        ev = strategy_evaluate(df, initial_capital=initial_capital, index_returns=index_returns)

        def g(metric_name):
            if metric_name not in ev.index:
                return 'N/A'
            value = ev.loc[metric_name].values[0]
            if value is None:
                return 'N/A'
            text = str(value).strip()
            return 'N/A' if text.lower() in {'nan', 'nan%', 'none', 'undefined'} else text

        holdings = []
        if reset_capital:
            holdings = _build_holdings_payload(
                df,
                initial_capital,
                quote_map=quote_map or _fetch_open_stock_quotes(df),
                trading_calendar=trading_calendar,
            )

        bm_curves_raw = _compute_benchmark_curves(df, INDEX_RETURNS_MAP)
        df_final_val = float(df['累积净值'].iloc[-1]) if len(df) > 0 else 1.0
        benchmark_curves = []
        for item in bm_curves_raw:
            bm_c = item.get('curve', [])
            bm_final = bm_c[-1]['value'] if bm_c else 1.0
            excess_pct = round((df_final_val / bm_final - 1) * 100, 2) if bm_final != 0 else 0
            benchmark_curves.append({
                'id': item['id'],
                'name': item['name'],
                'curve': bm_c,
                'curve_quarterly': _resample_curve(bm_c, 'quarter'),
                'curve_yearly': _resample_curve(bm_c, 'year'),
                'excess_return_pct': excess_pct,
            })

        summary[name] = {
            'label': {
                'pre_6m_history': '半年前历史',
                'recent_6m': '近半年',
                'recent_1q': '近一季',
                'recent_1m': '近一月',
            }.get(name, name),
            'months': len(df),
            'win_rate': round(float((df['选股下周期涨跌幅'] > 0).mean()), 4),
            'reset_capital': reset_capital,
            'initial_capital': initial_capital,
            'final_capital': float(df['累计资金'].iloc[-1]) if len(df) else initial_capital,
            'date_range': {
                'start': df['交易日期'].min().strftime('%Y-%m-%d'),
                'end': df['交易日期'].max().strftime('%Y-%m-%d'),
            },
            'metrics': {
                'cumulative_return': g('累积净值'),
                'annual_return': g('年化收益'),
                'max_drawdown': g('最大回撤'),
                'max_dd_start': g('最大回撤开始'),
                'max_dd_end': g('最大回撤结束'),
                'calmar_ratio': g('年化收益/回撤比'),
                'final_capital': g('最终资金'),
                'total_return_pct': g('总收益率'),
                'total_pnl': g('总盈亏'),
                'beta': g('Beta'),
                'annual_alpha': g('年化Alpha'),
                'information_ratio': g('信息比率'),
                'r_squared': g('R-squared'),
                'up_capture': g('上行捕获率'),
                'down_capture': g('下行捕获率'),
            },
            'equity_curve': [
                {
                    'date': r['交易日期'].strftime('%Y-%m-%d'),
                    'value': round(float(r['累积净值']), 4),
                    'return': round(float(r['选股下周期涨跌幅']), 6),
                }
                for _, r in df.iterrows()
            ],
            'equity_curve_quarterly': _resample_curve([
                {
                    'date': r['交易日期'].strftime('%Y-%m-%d'),
                    'value': round(float(r['累积净值']), 4),
                    'return': round(float(r['选股下周期涨跌幅']), 6),
                }
                for _, r in df.iterrows()
            ], 'quarter'),
            'equity_curve_yearly': _resample_curve([
                {
                    'date': r['交易日期'].strftime('%Y-%m-%d'),
                    'value': round(float(r['累积净值']), 4),
                    'return': round(float(r['选股下周期涨跌幅']), 6),
                }
                for _, r in df.iterrows()
            ], 'year'),
            'daily_equity_curve': _build_daily_curve_slice(df, result.attrs.get('daily_equity_curve', []), trading_calendar=trading_calendar),
            'holdings': holdings,
            'benchmark_curves': benchmark_curves,
        }

    return summary


def compute_split_metrics(result, split_date=SPLIT_DATE, index_returns=None, benchmark_id=None, quote_map=None):
    """
    将回测结果拆分为训练集和测试集，分别计算指标。

    训练集: 交易日 ≤ split_date
    测试集: 交易日 > split_date

    如果提供 index_returns，还会计算每段的 alpha/beta 归因指标和基准曲线。
    如果 split_date 为 None，返回空的 train/test（表示不拆分）。

    返回 dict:
      train: {metrics, months, win_rate, monthly_returns, benchmark_curve, attribution}
      test:  {metrics, months, win_rate, monthly_returns, benchmark_curve, attribution}
      split_date: str
    """
    if split_date is None or pd.isna(split_date):
        return {'train': None, 'test': None, 'split_date': None}

    trading_calendar = _load_trading_calendar(benchmark_id)
    train = result[result['交易日期'] <= split_date].copy()
    test = result[result['交易日期'] > split_date].copy()

    full_period_curves = result.attrs.get('period_daily_curves', [])
    if full_period_curves:
        period_curve_lookup = {
            pd.to_datetime(dt).strftime('%Y-%m-%d'): curve
            for dt, curve in zip(result['交易日期'], full_period_curves)
        }
        train.attrs['period_daily_curves'] = [
            period_curve_lookup.get(pd.to_datetime(dt).strftime('%Y-%m-%d'), [])
            for dt in train['交易日期']
        ]
        test.attrs['period_daily_curves'] = [
            period_curve_lookup.get(pd.to_datetime(dt).strftime('%Y-%m-%d'), [])
            for dt in test['交易日期']
        ]

    # 确定初始本金
    initial_capital = result.attrs.get('initial_capital', 100000)
    if '当期本金' in result.columns and len(result) > 0:
        initial_capital = result['当期本金'].iloc[0]

    def compute_period(df, start_capital, include_holdings=False):
        if len(df) == 0:
            return None, start_capital
        df = df.copy()
        df['累积净值'] = (1 + df['选股下周期涨跌幅']).cumprod()

        # 重新计算绝对资金
        capital = float(start_capital)
        capitals = []
        pnls = []
        cum_caps = []
        for _, row in df.iterrows():
            capitals.append(capital)
            pnl = capital * row['选股下周期涨跌幅']
            pnls.append(pnl)
            capital += pnl
            cum_caps.append(capital)
        df['当期本金'] = capitals
        df['当期盈亏'] = pnls
        df['累计资金'] = cum_caps

        ev = strategy_evaluate(df, initial_capital=start_capital,
                              index_returns=index_returns)
        win_rate = round(float((df['选股下周期涨跌幅'] > 0).mean()), 4)
        monthly = [{'date': r['交易日期'].strftime('%Y-%m-%d'),
                     'value': round(float(r['选股下周期涨跌幅']), 6)}
                   for _, r in df.iterrows()]
        final_capital = float(df['累计资金'].iloc[-1]) if len(df) > 0 else start_capital

        period_result = {
            'metrics': {
                'cumulative_return': str(ev.loc['累积净值'].values[0]) if '累积净值' in ev.index else 'N/A',
                'annual_return': str(ev.loc['年化收益'].values[0]) if '年化收益' in ev.index else 'N/A',
                'max_drawdown': str(ev.loc['最大回撤'].values[0]) if '最大回撤' in ev.index else 'N/A',
                'max_dd_start': str(ev.loc['最大回撤开始'].values[0]) if '最大回撤开始' in ev.index else 'N/A',
                'max_dd_end': str(ev.loc['最大回撤结束'].values[0]) if '最大回撤结束' in ev.index else 'N/A',
                'calmar_ratio': str(ev.loc['年化收益/回撤比'].values[0]) if '年化收益/回撤比' in ev.index else 'N/A',
                'final_capital': str(ev.loc['最终资金'].values[0]) if '最终资金' in ev.index else 'N/A',
                'total_return_pct': str(ev.loc['总收益率'].values[0]) if '总收益率' in ev.index else 'N/A',
                'total_pnl': str(ev.loc['总盈亏'].values[0]) if '总盈亏' in ev.index else 'N/A',
            },
            'win_rate': win_rate,
            'months': len(df),
            'monthly_returns': monthly,
            'daily_equity_curve': _build_daily_curve_slice(df, result.attrs.get('daily_equity_curve', []), trading_calendar=trading_calendar),
            'initial_capital': start_capital,
            'final_capital': final_capital,
            'date_range': {
                'start': df['交易日期'].min().strftime('%Y-%m-%d'),
                'end': df['交易日期'].max().strftime('%Y-%m-%d'),
            },
        }
        # 持股明细
        if include_holdings:
            period_result['holdings'] = _build_holdings_payload(
                df,
                start_capital,
                quote_map=quote_map or _fetch_open_stock_quotes(df),
                trading_calendar=trading_calendar,
            )

        # ── 归因指标 ──
        if index_returns is not None:
            attr_keys_map = {
                'Beta': 'beta', '年化Alpha': 'annual_alpha',
                '信息比率': 'information_ratio', 'R-squared': 'r_squared',
                '上行捕获率': 'up_capture', '下行捕获率': 'down_capture',
            }
            for ev_key, json_key in attr_keys_map.items():
                if ev_key in ev.index:
                    period_result['metrics'][json_key] = str(ev.loc[ev_key].values[0])

        bm_curve = _compute_single_benchmark_curve(df, index_returns)
        period_result['benchmark_curve'] = bm_curve
        period_result['benchmark_curve_quarterly'] = _resample_curve(bm_curve, 'quarter')
        period_result['benchmark_curve_yearly'] = _resample_curve(bm_curve, 'year')

        bm_curves_raw = _compute_benchmark_curves(df, INDEX_RETURNS_MAP)
        df_final_val = float(df['累积净值'].iloc[-1]) if len(df) > 0 else 1.0
        bm_curves = []
        for item in bm_curves_raw:
            bm_c = item.get('curve', [])
            bm_final = bm_c[-1]['value'] if bm_c else 1.0
            excess_pct = round((df_final_val / bm_final - 1) * 100, 2) if bm_final != 0 else 0
            bm_curves.append({
                'id': item['id'],
                'name': item['name'],
                'curve': bm_c,
                'curve_quarterly': _resample_curve(bm_c, 'quarter'),
                'curve_yearly': _resample_curve(bm_c, 'year'),
                'excess_return_pct': excess_pct,
            })
        period_result['benchmark_curves'] = bm_curves
        period_result['active_benchmark'] = _get_benchmark_meta(benchmark_id)

        return period_result, final_capital

    train_result, train_final_cap = compute_period(train, initial_capital)
    test_result, _ = compute_period(test,
                                    train_final_cap if train_final_cap is not None else initial_capital,
                                    include_holdings=True)

    return {
        'train': train_result,
        'test': test_result,
        'split_date': split_date.strftime('%Y-%m-%d'),
        'initial_capital': initial_capital,
    }


def result_to_json(result, ev, split_date=SPLIT_DATE, benchmark_id=None):
    """将回测结果 DataFrame 转为前端 JSON，包含训练/测试集拆分"""
    # 资金曲线（倍数）
    equity_curve = [{'date': r['交易日期'].strftime('%Y-%m-%d'),
                     'value': round(float(r['累积净值']), 4),
                     'return': round(float(r['选股下周期涨跌幅']), 6)}
                    for _, r in result.iterrows()]

    # 绝对资金曲线
    capital_curve = []
    if '累计资金' in result.columns:
        capital_curve = [{'date': r['交易日期'].strftime('%Y-%m-%d'),
                          'value': round(float(r['累计资金']), 2),
                          'capital_start': round(float(r.get('当期本金', 0)), 2),
                          'pnl': round(float(r.get('当期盈亏', 0)), 2),
                          'return': round(float(r['选股下周期涨跌幅']), 6)}
                         for _, r in result.iterrows()]

    # 回撤
    cum = result['累积净值'].values
    peak = np.maximum.accumulate(cum)
    dd = cum / peak - 1
    drawdown = [{'date': result.iloc[i]['交易日期'].strftime('%Y-%m-%d'),
                 'value': round(float(dd[i]), 6)}
                for i in range(len(result))]

    # 年度收益
    yr = result.copy()
    yr['年份'] = yr['交易日期'].dt.year
    yearly = yr.groupby('年份')['选股下周期涨跌幅'].apply(
        lambda x: (1 + x).prod() - 1)
    yearly_returns = [{'year': int(y), 'value': round(float(v), 6)}
                      for y, v in yearly.items()]

    # 月度收益
    monthly = [{'date': r['交易日期'].strftime('%Y-%m-%d'),
                'value': round(float(r['选股下周期涨跌幅']), 6)}
               for _, r in result.iterrows()]
    trading_calendar = _load_trading_calendar(benchmark_id)
    daily_equity_curve = _build_daily_curve_slice(result, result.attrs.get('daily_equity_curve', []), trading_calendar=trading_calendar)

    # 预计算各分辨率曲线（月线为原始数据，季线/年线由后端重采样）
    equity_curve_quarterly = _resample_curve(equity_curve, 'quarter')
    equity_curve_yearly = _resample_curve(equity_curve, 'year')

    # 持仓明细（含个股仓位占比和盈亏）
    holdings_quote_map = _fetch_open_stock_quotes(result)
    holdings = _build_holdings_payload(
        result,
        float(result.attrs.get('initial_capital', 100000)),
        quote_map=holdings_quote_map,
        trading_calendar=trading_calendar,
    ) if '买入个股收益' in result.columns else []

    def g(m):
        if m not in ev.index:
            return 'N/A'
        value = ev.loc[m].values[0]
        if value is None:
            return 'N/A'
        text = str(value).strip()
        if text.lower() in {'nan', 'nan%', 'none', 'undefined'}:
            return 'N/A'
        return text

    # 初始本金和费率信息
    initial_capital = float(result.attrs.get('initial_capital', 100000))

    active_benchmark_id, active_benchmark_series = _get_benchmark_series(benchmark_id)

    # 统一区间窗口摘要
    interval_windows = build_selection_interval_windows(
        result,
        index_returns=active_benchmark_series,
        benchmark_id=active_benchmark_id,
        quote_map=holdings_quote_map,
    )

    # 兼容旧结构的临时拆分摘要
    split = compute_split_metrics(result, split_date,
                                  index_returns=active_benchmark_series,
                                  benchmark_id=active_benchmark_id,
                                  quote_map=holdings_quote_map)

    # 分别构建训练集和测试集的资金曲线（各自从 1 开始）
    train_curve = []
    test_curve = []
    train_capital_curve = []
    test_capital_curve = []
    if split_date and split and split.get('train') and split.get('test'):
        train_df = result[result['交易日期'] <= split_date].copy()
        train_df['累积净值'] = (1 + train_df['选股下周期涨跌幅']).cumprod()
        train_curve = [{'date': r['交易日期'].strftime('%Y-%m-%d'),
                        'value': round(float(r['累积净值']), 4),
                        'return': round(float(r['选股下周期涨跌幅']), 6)}
                       for _, r in train_df.iterrows()]

        test_df = result[result['交易日期'] > split_date].copy()
        test_df['累积净值'] = (1 + test_df['选股下周期涨跌幅']).cumprod()
        test_curve = [{'date': r['交易日期'].strftime('%Y-%m-%d'),
                       'value': round(float(r['累积净值']), 4),
                       'return': round(float(r['选股下周期涨跌幅']), 6)}
                      for _, r in test_df.iterrows()]

        # 训练/测试集的绝对资金曲线
        if split['train'] and 'final_capital' in split['train']:
            train_initial = split['train'].get('initial_capital', initial_capital)
            cap = float(train_initial)
            for _, r in train_df.iterrows():
                pnl = cap * r['选股下周期涨跌幅']
                cap_before = cap
                cap += pnl
                train_capital_curve.append({
                    'date': r['交易日期'].strftime('%Y-%m-%d'),
                    'value': round(cap, 2),
                    'pnl': round(pnl, 2),
                })

        if split['test'] and 'initial_capital' in split['test']:
            test_initial = split['test'].get('initial_capital', initial_capital)
            cap = float(test_initial)
            for _, r in test_df.iterrows():
                pnl = cap * r['选股下周期涨跌幅']
                cap_before = cap
                cap += pnl
                test_capital_curve.append({
                    'date': r['交易日期'].strftime('%Y-%m-%d'),
                    'value': round(cap, 2),
                    'pnl': round(pnl, 2),
                })

    # 预计算训练/测试集各分辨率曲线
    train_curve_quarterly = _resample_curve(train_curve, 'quarter')
    train_curve_yearly = _resample_curve(train_curve, 'year')
    test_curve_quarterly = _resample_curve(test_curve, 'quarter')
    test_curve_yearly = _resample_curve(test_curve, 'year')

    profile_summary = []
    strategy_meta = result.attrs.get('strategy_meta', {}) if hasattr(result, 'attrs') else {}
    if isinstance(strategy_meta, dict):
        profile_summary = strategy_meta.get('profile_summary', []) or []

    # 基准曲线及超额收益
    benchmark_curve = _compute_single_benchmark_curve(result, active_benchmark_series)
    benchmark_curves_raw = _compute_benchmark_curves(result, INDEX_RETURNS_MAP)
    strategy_final = equity_curve[-1]['value'] if equity_curve else 1.0
    benchmark_curves = []
    for item in benchmark_curves_raw:
        bm_curve = item.get('curve', [])
        bm_final = bm_curve[-1]['value'] if bm_curve else 1.0
        excess_pct = round((strategy_final / bm_final - 1) * 100, 2) if bm_final != 0 else 0
        benchmark_curves.append({
            'id': item['id'],
            'name': item['name'],
            'curve': bm_curve,
            'curve_quarterly': _resample_curve(bm_curve, 'quarter'),
            'curve_yearly': _resample_curve(bm_curve, 'year'),
            'excess_return_pct': excess_pct,
        })

    start_label = result['交易日期'].min().strftime('%Y-%m-%d') if len(result) > 0 else None
    end_label = result['交易日期'].max().strftime('%Y-%m-%d') if len(result) > 0 else None
    if len(result) <= 1:
        holdings_label = '近一月持股明细 & 仓位'
    else:
        holdings_label = '当前区间持股明细 & 仓位'
    holdings_context = {
        'label': holdings_label,
        'date_range': {
            'start': start_label,
            'end': end_label,
        },
        'holdings': holdings,
    } if len(result) > 0 else None

    return {
        'equity_curve': equity_curve,
        'equity_curve_quarterly': equity_curve_quarterly,
        'equity_curve_yearly': equity_curve_yearly,
        'daily_equity_curve': daily_equity_curve,
        'capital_curve': capital_curve,
        'train_equity_curve': train_curve,
        'train_equity_curve_quarterly': train_curve_quarterly,
        'train_equity_curve_yearly': train_curve_yearly,
        'test_equity_curve': test_curve,
        'test_equity_curve_quarterly': test_curve_quarterly,
        'test_equity_curve_yearly': test_curve_yearly,
        'train_daily_equity_curve': split.get('train', {}).get('daily_equity_curve', []) if split and split.get('train') else [],
        'test_daily_equity_curve': split.get('test', {}).get('daily_equity_curve', []) if split and split.get('test') else [],
        'train_capital_curve': train_capital_curve,
        'test_capital_curve': test_capital_curve,
        'drawdown': drawdown,
        'yearly_returns': yearly_returns,
        'monthly_returns': monthly,
        'holdings': holdings,
        'holdings_context': holdings_context,
        'metrics': {
            'cumulative_return': g('累积净值'),
            'annual_return': g('年化收益'),
            'max_drawdown': g('最大回撤'),
            'max_dd_start': g('最大回撤开始'),
            'max_dd_end': g('最大回撤结束'),
            'calmar_ratio': g('年化收益/回撤比'),
            'final_capital': g('最终资金'),
            'total_return_pct': g('总收益率'),
            'total_pnl': g('总盈亏'),
            'beta': g('Beta'),
            'annual_alpha': g('年化Alpha'),
            'information_ratio': g('信息比率'),
            'r_squared': g('R-squared'),
            'up_capture': g('上行捕获率'),
            'down_capture': g('下行捕获率'),
        },
        'initial_capital': initial_capital,
        'fee_info': {
            'c_rate': result.attrs.get('c_rate', 1.0 / 10000),
            't_rate': result.attrs.get('t_rate', 1 / 1000),
            'sell_cost': result.attrs.get('sell_cost', 1.0 / 10000 + 1 / 1000),
            'total_buy_fees': round(result.attrs.get('total_buy_fees', 0), 2),
            'total_sell_fees': round(result.attrs.get('total_sell_fees', 0), 2),
            'total_fees': round(result.attrs.get('total_fees', 0), 2),
        },
        'win_rate': round(float((result['选股下周期涨跌幅'] > 0).mean()), 4),
        'date_range': {
            'start': result['交易日期'].min().strftime('%Y-%m-%d'),
            'end': result['交易日期'].max().strftime('%Y-%m-%d'),
        },
        'total_months': len(result),
        'split': split,
        'interval_windows': interval_windows,
        'profile_summary': profile_summary,
        'active_benchmark': _get_benchmark_meta(active_benchmark_id),
        'benchmark_curve': benchmark_curve,
        'benchmark_curve_quarterly': _resample_curve(benchmark_curve, 'quarter'),
        'benchmark_curve_yearly': _resample_curve(benchmark_curve, 'year'),
        'benchmark_curves': benchmark_curves,
    }


# ═══════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/timing')
def timing_page():
    return render_template('timing.html')


@app.route('/us_timing')
def us_timing_page():
    return render_template('us_timing.html')


@app.route('/api/us_timing/info')
def api_us_timing_info():
    try:
        ensure_us_timing_panel_loaded()
    except Exception as e:
        return jsonify({'error': f'美股指数数据加载失败: {e}'}), 500
    if US_TIMING_PANEL is None or len(US_TIMING_PANEL) == 0:
        return jsonify({'error': '美股指数数据未加载'}), 500
    max_date = pd.to_datetime(US_TIMING_PANEL['交易日期'].max())
    min_date = pd.to_datetime(US_TIMING_PANEL['交易日期'].min())
    return jsonify({
        'data_min_date': min_date.strftime('%Y-%m-%d'),
        'data_max_date': max_date.strftime('%Y-%m-%d'),
        'indexes': [
            {'id': strategy_id, 'name': strategy_cls().get_display_name(), 'index_name': strategy_cls().get_index_name()}
            for strategy_id, strategy_cls in US_TIMING_STRATEGY_MAP.items()
        ],
    })


@app.route('/api/us_timing/strategy_list')
def api_us_timing_strategy_list():
    try:
        ensure_us_timing_panel_loaded()
    except Exception as e:
        return jsonify({'error': f'美股指数数据加载失败: {e}'}), 500
    data_max_date = None
    if US_TIMING_PANEL is not None and len(US_TIMING_PANEL) > 0:
        data_max_date = pd.to_datetime(US_TIMING_PANEL['交易日期'].max()).strftime('%Y-%m-%d')

    def _build_perf_delta(current_id, baseline_id=None):
        if not baseline_id:
            return None
        current = US_TIMING_CACHE.get(current_id)
        baseline = US_TIMING_CACHE.get(baseline_id)
        if current is None or baseline is None or len(current) == 0 or len(baseline) == 0:
            return None

        def _last(df, col):
            if col not in df.columns or len(df[col]) == 0:
                return None
            val = df[col].iloc[-1]
            return float(val) if pd.notna(val) else None

        current_nav = _last(current, '累积净值')
        baseline_nav = _last(baseline, '累积净值')
        current_capital = _last(current, '总资金')
        baseline_capital = _last(baseline, '总资金')
        current_mdd = current.attrs.get('metrics', {}).get('最大回撤')
        baseline_mdd = baseline.attrs.get('metrics', {}).get('最大回撤')
        current_annual = current.attrs.get('metrics', {}).get('年化收益')
        baseline_annual = baseline.attrs.get('metrics', {}).get('年化收益')

        payload = {}
        if current_nav is not None and baseline_nav is not None:
            payload['cumulative_return_diff'] = round(current_nav - baseline_nav, 4)
        if current_capital is not None and baseline_capital is not None:
            payload['final_capital_diff'] = round(current_capital - baseline_capital, 2)
        if current_mdd is not None and baseline_mdd is not None:
            payload['max_drawdown_improvement'] = round(float(current_mdd) - float(baseline_mdd), 2)
        if current_annual is not None and baseline_annual is not None:
            payload['annual_return_diff'] = round(float(current_annual) - float(baseline_annual), 2)
        payload['baseline_strategy'] = baseline_id
        return payload or None

    try:
        init_us_timing_cache()
    except Exception as e:
        return jsonify({'error': f'美股择时缓存初始化失败: {e}'}), 500
    payload = []
    for strategy_id in US_TIMING_PAGE_STRATEGY_IDS:
        strategy_cls = US_TIMING_STRATEGY_MAP[strategy_id]
        strategy = strategy_cls()
        cached = US_TIMING_CACHE.get(strategy_id)
        cumulative_return = None
        current_action = None
        total_return_pct = None
        annual_return = None
        max_drawdown = None
        if cached is not None and len(cached) > 0:
            cumulative_return = round(float(cached['累积净值'].iloc[-1]), 4)
            current_action = str(cached['signal_action'].iloc[-1])
            metrics = cached.attrs.get('metrics', {})
            total_return_pct = metrics.get('总收益率')
            annual_return = metrics.get('年化收益')
            max_drawdown = metrics.get('最大回撤')

        changelog_meta = dict(US_TIMING_CHANGELOG_META.get(strategy_id, {}))
        perf_delta = _build_perf_delta(strategy_id, changelog_meta.get('supersedes'))
        payload.append({
            'id': strategy_id,
            'name': strategy.get_display_name(),
            'description': strategy.get_strategy_description(),
            'index_name': strategy.get_index_name(),
            'cumulative_return': cumulative_return,
            'current_action': current_action,
            'total_return_pct': total_return_pct,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'data_max_date': data_max_date,
            'is_page_winner': True,
            **changelog_meta,
            'performance_delta': perf_delta,
        })
    return jsonify(payload)


@app.route('/api/us_timing/params')
def api_us_timing_params():
    strategy_name = request.args.get('strategy', 'macro_v32_timing')
    strategy = build_us_timing_strategy(strategy_name)
    payload = strategy.get_signal_metadata()
    profile_view = get_best_profile_view(strategy_name)
    if profile_view is not None:
        payload['best_profile'] = profile_view
    return jsonify(payload)


@app.route('/api/us_timing/backtest')
def api_us_timing_backtest():
    strategy_name = request.args.get('strategy', 'macro_v32_timing')
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    compact = request.args.get('compact', '0') in {'1', 'true', 'yes'}

    use_cache = strategy_name in US_TIMING_CACHE
    if use_cache and strategy_name in _US_TIMING_CACHE_DEFAULTS:
        for key, default_val in _US_TIMING_CACHE_DEFAULTS[strategy_name].items():
            raw_val = request.args.get(key)
            if raw_val is None:
                continue
            try:
                if isinstance(default_val, bool):
                    val = _parse_realism_bool(raw_val)
                    if val is None:
                        raise ValueError(f'cannot parse bool: {raw_val}')
                elif isinstance(default_val, str):
                    val = raw_val
                elif isinstance(default_val, int):
                    val = int(raw_val)
                else:
                    val = float(raw_val)
            except (TypeError, ValueError):
                use_cache = False
                break
            if val != default_val:
                use_cache = False
                break

    _t0 = time.time()
    try:
        params = {}
        int_keys = {'fast_window', 'slow_window', 'momentum_window'}
        float_keys = {'enter_threshold', 'add_threshold', 'trim_threshold', 'exit_threshold', 'max_entry_exposure',
                      'probe_entry_exposure', 'sigmoid_k', 'max_leverage', 'base_position', 'inertia', 'crisis_vix',
                      'fed_block_weight', 'restrictive_threshold', 'pivot_relief', 'base_floor'}
        for key in int_keys | float_keys:
            val = request.args.get(key, type=float)
            if val is not None:
                params[key] = int(val) if key in int_keys else float(val)
        confirm_days = request.args.get('confirm_days', type=int)
        if confirm_days is not None:
            params['confirm_days'] = int(confirm_days)
        probe_confirm_days = request.args.get('probe_confirm_days', type=int)
        if probe_confirm_days is not None:
            params['probe_confirm_days'] = int(probe_confirm_days)
        exposure_mode = request.args.get('exposure_mode')
        if exposure_mode:
            params['exposure_mode'] = exposure_mode

        # 12 个真实交易规则参数（5 个 profit_lock_* + 7 个费用/滑点/涨跌停）
        params.update(_collect_realism_params(request.args))

        if use_cache:
            result = US_TIMING_CACHE[strategy_name].copy()
            strategy = build_us_timing_strategy(strategy_name)
        else:
            ensure_us_timing_panel_loaded()
            strategy = build_us_timing_strategy(strategy_name, **params)
            signal_df = strategy.run(US_TIMING_PANEL.copy())
            result = run_timing_backtest(signal_df, strategy)

        full_history_start = pd.to_datetime(result['交易日期'].min()) if len(result) else None
        result = filter_timing_result(result, start_date=start_date, end_date=end_date)
        if len(result) == 0:
            return jsonify({'error': '所选日期范围内无数据'}), 400

        etf_benchmark_returns = _build_etf_monthly_returns(result)
        metrics = evaluate_timing_result(result, benchmark_returns=etf_benchmark_returns, reset_capital=True)
        bm_curve = []
        payload = timing_result_to_json(result, metrics, benchmark_curve=bm_curve, compact=compact)
        payload['interval_windows'] = summarize_timing_windows(
            result,
            benchmark_returns=etf_benchmark_returns,
            full_history_start=full_history_start,
        )
        print(f'[us_timing/backtest] strategy={strategy_name} cache_hit={use_cache} total={(time.time()-_t0)*1000:.0f}ms')
        return jsonify(payload)
    except Exception as e:
        print(f'[us_timing/backtest] ERROR: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/status')
def api_status():
    elapsed = 0
    if _LOAD_STATUS['start_time'] is not None:
        elapsed = time.time() - _LOAD_STATUS['start_time']
    return jsonify({
        'stage': _LOAD_STATUS['stage'],
        'message': _LOAD_STATUS['message'],
        'elapsed_sec': round(elapsed, 1),
        'loading': _LOAD_STATUS['loading'],
        'ready': _DATA_READY.is_set(),
    })


# 各策略默认参数值（用于缓存命中判断，值对应前端 slider 默认值）
_CACHE_DEFAULTS = {
    'original': {'val_pct_cutoff': 0.68, 'bias_pct': 0.52, 'vol_pct': 0.78},
    'original_ensemble': {
        'weight_3y': 0.5,
        'weight_5y': 0.3,
        'weight_full': 0.2,
        'vote_top_k': 12,
        'board_tilt_strength': 0.4,
        'growth_timing_mode': 'off',
        'growth_hold_days': 4,
        'growth_top_n': 2,
    },
    'chan_enhanced': {'val_pct_cutoff': 0.68, 'bias_pct': 0.52, 'vol_pct': 0.78, 'chan_tilt': 0.03},
    'chan_only': {'chan_weight': 0.70},
    'method_a': {'val_pct_cutoff': 0.68, 'bias_pct': 0.52, 'vol_pct': 0.78, 'chan_tilt': 0.05},
    'quality_value': {
        'size_weight': 0.50, 'bm_weight': 0.25, 'roe_weight': 0.15,
        'turnover_weight': 0.10, 'min_market_cap': 20, 'min_turnover': 0.5,
        'select_stock_num': 3, 'bias_pct': 0.52, 'vol_pct': 0.78,
    },
}

# 共享的真实交易规则参数默认值（与 BaseTimingStrategy.__init__ 一致）
# 12 项 = 5 项 profit_lock_* + 7 项费用 / 滑点 / 涨跌停
_SHARED_REALISM_DEFAULTS = {
    'profit_lock_enabled': False,
    'profit_lock_drawdown': 0.04,
    'profit_lock_level_1': 0.10,
    'profit_lock_level_2': 0.18,
    'profit_lock_level_3': 0.28,
    'slippage_bps': 5.0,
    'cash_interest_rate': 0.015,
    'commission_rate': 0.0001,
    'commission_min': 5.0,
    'stamp_tax_rate': 0.0,
    'transfer_fee_rate': 0.00001,
    'limit_max_delay_days': 5,
    # CLAUDE.md Rule 14: 择时策略支持最低底仓 base_floor，避免长牛中过度避险。
    # 共享默认为 0（无地板）；只有通过 walk-forward 达成「收益+回撤都跑赢 ETF」
    # 的策略，才在自己的 best_profile.json 中显式启用 base_floor（如 star50=0.7）。
    'base_floor': 0.0,
}

_TIMING_CACHE_DEFAULTS = {
    'csi1000_timing': {
        'breakout_window': 15, 'exit_window': 7, 'trend_window': 50,
        'exposure_mode': 'staged', 'enter_threshold': 0.55, 'add_threshold': 0.75,
        'trim_threshold': 0.38, 'exit_threshold': 0.18, 'confirm_days': 1,
        'max_entry_exposure': 0.5, 'probe_entry_exposure': 0.25, 'probe_confirm_days': 1,
        **_SHARED_REALISM_DEFAULTS,
    },
    'star50_timing': {
        'breakout_window': 10, 'exit_window': 5, 'trend_window': 40,
        'exposure_mode': 'staged', 'enter_threshold': 0.55, 'add_threshold': 0.75,
        'trim_threshold': 0.35, 'exit_threshold': 0.15, 'confirm_days': 1,
        'max_entry_exposure': 0.5, 'probe_entry_exposure': 0.25, 'probe_confirm_days': 1,
        **_SHARED_REALISM_DEFAULTS,
    },
    'chinext_timing': {
        'momentum_short_window': 20, 'momentum_long_window': 60, 'trend_window': 60, 'momentum_threshold': 0.02,
        'exposure_mode': 'staged', 'enter_threshold': 0.6, 'add_threshold': 0.8,
        'trim_threshold': 0.35, 'exit_threshold': 0.15, 'confirm_days': 2,
        'max_entry_exposure': 0.5, 'probe_entry_exposure': 0.25, 'probe_confirm_days': 1,
        **_SHARED_REALISM_DEFAULTS,
    },
}

_US_TIMING_CACHE_DEFAULTS = {
    'macro_v32_timing': {
        'sigmoid_k': 1.2, 'max_leverage': 1.4, 'base_position': 0.45,
        'inertia': 0.05, 'crisis_vix': 40.0,
        'fed_block_weight': 0.25, 'restrictive_threshold': 0.40, 'pivot_relief': 0.60,
        'exposure_mode': 'staged', 'enter_threshold': 0.55, 'add_threshold': 0.75,
        'trim_threshold': 0.35, 'exit_threshold': 0.15, 'confirm_days': 1,
        'max_entry_exposure': 1.0,
        **_SHARED_REALISM_DEFAULTS,
    },
    'sp500_timing': {
        'fast_window': 20, 'slow_window': 125, 'momentum_window': 100,
        'exposure_mode': 'staged', 'enter_threshold': 0.5, 'add_threshold': 0.72,
        'trim_threshold': 0.32, 'exit_threshold': 0.14, 'confirm_days': 2,
        'max_entry_exposure': 0.5, 'probe_entry_exposure': 0.25, 'probe_confirm_days': 1,
        **_SHARED_REALISM_DEFAULTS,
    },
}


# 真实交易规则参数的类型定义；用于 API 层把 request.args 转成对应 Python 类型
# bool: profit_lock_enabled (前端可能传 'on'/'true'/'1' 或 'off'/'false'/'0')
# int : limit_max_delay_days
# 其余: float
_REALISM_BOOL_KEYS = {'profit_lock_enabled'}
_REALISM_INT_KEYS = {'limit_max_delay_days'}
_REALISM_FLOAT_KEYS = {
    'profit_lock_drawdown', 'profit_lock_level_1', 'profit_lock_level_2', 'profit_lock_level_3',
    'slippage_bps', 'cash_interest_rate',
    'commission_rate', 'commission_min',
    'stamp_tax_rate', 'transfer_fee_rate',
    'base_floor',
}
_REALISM_ALL_KEYS = _REALISM_BOOL_KEYS | _REALISM_INT_KEYS | _REALISM_FLOAT_KEYS


def _parse_realism_bool(raw):
    """把前端传来的布尔值字符串解析成 Python bool。"""
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in {'on', 'true', '1', 'yes', 'y'}:
        return True
    if s in {'off', 'false', '0', 'no', 'n', ''}:
        return False
    return None


def _collect_realism_params(args):
    """从 request.args 抽取 12 个真实交易规则参数；不存在的 key 不写入。"""
    params = {}
    for key in _REALISM_BOOL_KEYS:
        raw = args.get(key)
        if raw is None:
            continue
        val = _parse_realism_bool(raw)
        if val is not None:
            params[key] = val
    for key in _REALISM_INT_KEYS:
        raw = args.get(key)
        if raw is None:
            continue
        try:
            params[key] = int(float(raw))
        except (TypeError, ValueError):
            continue
    for key in _REALISM_FLOAT_KEYS:
        raw = args.get(key)
        if raw is None:
            continue
        try:
            params[key] = float(raw)
        except (TypeError, ValueError):
            continue
    return params


def _get_timing_default_params(strategy_name):
    return dict(_TIMING_CACHE_DEFAULTS.get(strategy_name, {}))


@app.route('/api/backtest')
def api_backtest():
    init_cache()
    strategy = request.args.get('strategy', 'original')
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    benchmark_id = _normalize_benchmark_id(request.args.get('benchmark', DEFAULT_BENCHMARK_ID))

    # ── 缓存命中判断 ──
    use_cache = strategy in BACKTEST_CACHE
    if use_cache and strategy in _CACHE_DEFAULTS:
        for key, default_val in _CACHE_DEFAULTS[strategy].items():
            raw_val = request.args.get(key)
            if raw_val is None:
                continue
            try:
                if isinstance(default_val, str):
                    val = raw_val
                elif isinstance(default_val, int) and not isinstance(default_val, bool):
                    val = int(raw_val)
                else:
                    val = float(raw_val)
            except (TypeError, ValueError):
                use_cache = False
                break
            if val != default_val:
                use_cache = False
                break

    try:
        if use_cache:
            result, _ = BACKTEST_CACHE[strategy]
            result = result.copy()
        else:
            # 收集前端 slider 传来的所有参数
            params = {}

            # 通用参数（key 与策略 __init__ 参数名一致）
            for key in ['val_pct_cutoff', 'bias_pct', 'vol_pct', 'chan_tilt',
                        'chan_weight', 'size_weight', 'bm_weight', 'roe_weight', 'turnover_weight',
                        'weight_3y', 'weight_5y', 'weight_full', 'vote_top_k', 'board_tilt_strength']:
                val = request.args.get(key, type=float)
                if val is not None:
                    params[key] = val

            growth_timing_mode = request.args.get('growth_timing_mode')
            if growth_timing_mode:
                params['growth_timing_mode'] = growth_timing_mode

            for key in ['select_stock_num', 'growth_hold_days', 'growth_top_n']:
                val = request.args.get(key, type=int)
                if val is not None:
                    params[key] = val

            # min_market_cap / min_turnover：前端以"亿"为单位，转换为元
            min_market_cap_raw = request.args.get('min_market_cap', type=float)
            if min_market_cap_raw is not None:
                params['min_market_cap'] = min_market_cap_raw * 1e8

            min_turnover_raw = request.args.get('min_turnover', type=float)
            if min_turnover_raw is not None:
                params['min_turnover'] = min_turnover_raw * 1e8

            result, _ = run_backtest_fresh(strategy, benchmark_id=benchmark_id, **params)

        active_benchmark_id, active_benchmark_series = _get_benchmark_series(benchmark_id)

        # 日期过滤（如果用户选了特定日期范围）
        if start_date or end_date:
            result, ev = filter_by_date(result, start_date, end_date, benchmark_id=active_benchmark_id)
            if result is None:
                return jsonify({'error': '所选日期范围内无数据'}), 400
            # 用户自定义日期范围时不显示训练/测试拆分
            return jsonify(result_to_json(result, ev, split_date=None, benchmark_id=active_benchmark_id))
        else:
            # 全量数据：包含训练/测试集拆分
            if use_cache:
                ev = strategy_evaluate(result, index_returns=active_benchmark_series)
            else:
                ev = strategy_evaluate(result, index_returns=active_benchmark_series)
            return jsonify(result_to_json(result, ev, split_date=SPLIT_DATE, benchmark_id=active_benchmark_id))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/factors')
def api_factors():
    ensure_stock_data_loaded()
    strategy_name = request.args.get('strategy', get_focused_strategy_id())
    strategy = build_strategy(strategy_name)
    return jsonify(strategy.get_factor_metadata())


@app.route('/api/factor_overview')
def api_factor_overview():
    ensure_stock_data_loaded()
    strategy_name = request.args.get('strategy', get_focused_strategy_id())
    return jsonify(build_factor_overview_payload(strategy_name))


@app.route('/api/info')
def api_info():
    """返回数据库基本信息，包括最新日期范围"""
    ensure_stock_data_loaded()
    if DATA_DF is None:
        return jsonify({'error': '数据未加载'}), 500
    max_date = pd.to_datetime(DATA_DF['交易日期'].max())
    min_date = pd.to_datetime(DATA_DF['交易日期'].min())
    return jsonify({
        'data_min_date': min_date.strftime('%Y-%m-%d'),
        'data_max_date': max_date.strftime('%Y-%m-%d'),
        'training_cutoff': TRAINING_CUTOFF,
        'holdout_start': HOLDOUT_START,
    })


@app.route('/api/timing/info')
def api_timing_info():
    try:
        ensure_timing_panel_loaded()
    except Exception as e:
        return jsonify({'error': f'指数数据加载失败: {e}'}), 500
    if TIMING_PANEL is None or len(TIMING_PANEL) == 0:
        return jsonify({'error': '指数数据未加载'}), 500
    max_date = pd.to_datetime(TIMING_PANEL['交易日期'].max())
    min_date = pd.to_datetime(TIMING_PANEL['交易日期'].min())
    return jsonify({
        'data_min_date': min_date.strftime('%Y-%m-%d'),
        'data_max_date': max_date.strftime('%Y-%m-%d'),
        'indexes': [
            {'id': strategy_id, 'name': strategy_cls().get_display_name(), 'index_name': strategy_cls().get_index_name()}
            for strategy_id, strategy_cls in TIMING_STRATEGY_MAP.items()
        ],
    })


@app.route('/api/timing/strategy_list')
def api_timing_strategy_list():
    try:
        init_timing_cache()
    except Exception as e:
        return jsonify({'error': f'择时缓存初始化失败: {e}'}), 500
    payload = []
    for strategy_id, strategy_cls in TIMING_STRATEGY_MAP.items():
        strategy = strategy_cls()
        cached = TIMING_CACHE.get(strategy_id)
        cumulative_return = None
        current_action = None
        if cached is not None and len(cached) > 0:
            cumulative_return = round(float(cached['累积净值'].iloc[-1]), 2)
            current_action = str(cached['signal_action'].iloc[-1])
        payload.append({
            'id': strategy_id,
            'name': strategy.get_display_name(),
            'description': strategy.get_strategy_description(),
            'index_name': strategy.get_index_name(),
            'cumulative_return': cumulative_return,
            'current_action': current_action,
            **TIMING_CHANGELOG_META.get(strategy_id, {}),
        })
    return jsonify(payload)


@app.route('/api/timing/params')
def api_timing_params():
    strategy_name = request.args.get('strategy', 'csi1000_timing')
    strategy = build_timing_strategy(strategy_name)
    payload = strategy.get_signal_metadata()
    # Phase 2: 把 best_profile 信息一并返回，前端用于显示只读 profile 卡片
    profile_view = get_best_profile_view(strategy_name)
    if profile_view is not None:
        payload['best_profile'] = profile_view
    return jsonify(payload)


def _latest_live_position(strategy_id):
    """读 live_trades.csv，返回该策略最近一条记录的 actual_position；没有记录则返回 0.0。

    用于在「最新入市信号卡」上把「操作建议」从策略视角（昨仓 vs 今仓）改成实盘视角（实盘当前仓 vs 今仓目标）。
    例如：策略前后两日都是 hold@100%，但实盘仓位还是 0% → 建议应当是「建仓买入」而非「继续持有」。
    """
    try:
        with _LIVE_TRADES_LOCK:
            rows = _read_live_trades()
    except Exception:
        return 0.0
    rows = [r for r in rows if r.get('strategy') == strategy_id and r.get('date')]
    if not rows:
        return 0.0
    rows.sort(key=lambda r: r['date'])
    try:
        return float(rows[-1].get('actual_position') or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _derive_action_from_delta(live_pos, target, tol=0.005):
    """基于「实盘当前仓位 vs 目标仓位」推导 rebalance_action。"""
    delta = target - live_pos
    if abs(delta) < tol:
        return 'flat' if target <= tol else 'hold'
    if delta > 0:
        return 'enter' if live_pos <= tol else 'add'
    return 'exit' if target <= tol else 'trim'


def _format_money(v, currency):
    sym = '$' if currency == 'USD' else '¥'
    return f'{sym}{v:,.2f}'


def _build_action_rationale(action, live_pos, target, ref_open, ref_close, strategy_id):
    """按 action 类型生成详细的「为什么是这个动作 + 怎么执行」说明文本。"""
    capital = _LIVE_INITIAL_CAPITAL
    currency = _LIVE_CURRENCY.get(strategy_id, 'CNY')
    lot = _LIVE_LOT_SIZE.get(strategy_id, 1)
    price = ref_open or ref_close  # 优先用 T+1 开盘价（执行价）；缺失时用收盘
    delta = target - live_pos
    cap_str = _format_money(capital, currency)

    def _shares_for(pos):
        if not price or price <= 0:
            return None
        return int((capital * pos) / price / lot) * lot

    target_shares = _shares_for(target)
    live_shares = _shares_for(live_pos)
    delta_shares = (target_shares - live_shares) if (target_shares is not None and live_shares is not None) else None
    px_txt = f'{price:.3f}' if price else '—'

    if action == 'enter':
        return (
            f'首次建仓。策略目标 {target*100:.1f}%，实盘当前 0%（{cap_str} 全部空仓）。'
            f'建议 T+1 开盘按参考价 {px_txt} 一次性买入约 {target_shares or "—"} 股，'
            f'完成后实盘仓位 ≈ {target*100:.1f}%。每手 {lot} 股已按整手取整。'
        )
    if action == 'add':
        return (
            f'加仓补齐。实盘当前 {live_pos*100:.1f}%（约 {live_shares or "—"} 股），策略目标 {target*100:.1f}%，'
            f'差 +{delta*100:.1f}%。建议 T+1 开盘按 {px_txt} 补买约 {delta_shares if delta_shares is not None else "—"} 股。'
        )
    if action == 'trim':
        return (
            f'减仓回落。实盘当前 {live_pos*100:.1f}%（约 {live_shares or "—"} 股），策略目标降到 {target*100:.1f}%，'
            f'差 {delta*100:.1f}%。建议 T+1 开盘按 {px_txt} 卖出约 {abs(delta_shares) if delta_shares is not None else "—"} 股。'
        )
    if action == 'exit':
        return (
            f'清仓离场。策略已切换为空仓信号（目标 0%），实盘当前 {live_pos*100:.1f}%。'
            f'建议 T+1 开盘按 {px_txt} 全部卖出，共约 {live_shares or "—"} 股，资金回到现金。'
        )
    if action == 'hold':
        return (
            f'与目标一致。实盘 {live_pos*100:.1f}% 与策略目标 {target*100:.1f}% 基本对齐（差 {abs(delta)*100:.2f}%），'
            f'无需操作。等待下一次信号变化（每日收盘后由 T+1 信号决定）。'
        )
    if action == 'flat':
        return (
            f'空仓观望。策略目标 0%，实盘已为 0%，无持仓需要管理。'
            f'等待 ETF 给出新的入市信号（如均线突破 + 成交量放量配合）再建仓。'
        )
    return ''


def _build_decision_context(strategy_id, target, live_pos, action):
    """看多 → 列出风险；看空 → 列出其他市场的看多策略 + 防御资产。返回 (view_bias, risks, opportunities)。"""
    if target >= 0.5 or action in ('enter', 'add'):
        view_bias = 'bullish'
    elif target <= 0.3 or action in ('exit', 'trim', 'flat'):
        view_bias = 'bearish'
    else:
        view_bias = 'neutral'

    risks = []
    opportunities = []

    # 实时风险因子（VIX / 利率 / 200dma 偏离等）：动态条目排在静态文案之前
    dynamic = _load_risk_signals()
    dyn_by_strat = (dynamic or {}).get('by_strategy', {}).get(strategy_id, {})
    dyn_risks = list(dyn_by_strat.get('bullish_risks_dynamic', []))
    dyn_opps = list(dyn_by_strat.get('bearish_opportunities_dynamic', []))

    if view_bias == 'bullish':
        risks = dyn_risks \
              + list(_BULLISH_RISKS_BY_STRATEGY.get(strategy_id, [])) \
              + list(_BULLISH_RISKS_GENERAL)
    elif view_bias == 'bearish':
        # 扫描其他策略的最新 target_exposure，挑出 target >= 0.5 的列入「其他市场看多」
        cross = []
        for other_id, other_cache in (
            list(TIMING_STRATEGY_MAP.items()) + list(US_TIMING_STRATEGY_MAP.items())
        ):
            if other_id == strategy_id:
                continue
            cache = TIMING_CACHE if other_id in TIMING_STRATEGY_MAP else US_TIMING_CACHE
            df = cache.get(other_id)
            if df is None or len(df) == 0:
                continue
            try:
                other_target = float(df.iloc[-1].get('target_exposure', 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if other_target >= 0.5:
                cross.append(
                    f'{_STRATEGY_DISPLAY.get(other_id, other_id)} 当前目标 {other_target*100:.0f}%，'
                    f'可考虑把部分资金分散到该方向。'
                )
        # 顺序：动态触发（VIX 极端 / 利率拐点 / 深度超卖）→ 跨市场看多 → 静态防御资产
        if dyn_opps:
            opportunities.extend(dyn_opps)
        if cross:
            opportunities.extend(cross)
        region = _REGION_OF_STRATEGY.get(strategy_id, 'cn')
        opportunities.extend(_BEARISH_OPPS_BY_REGION.get(region, []))

    return view_bias, risks, opportunities


def _build_latest_signal(strategy_id, strategy, result_df, profile=None):
    """从 result_df 末尾构造「最新入市信号卡」的统一 dict。

    返回字段对前端友好：as_of_date 是最新已收盘交易日；target_exposure 是 t+1 应持仓比例；
    action_label 给出可读操作；exec_basis 描述执行规则；status 标记 production / research。
    同时返回 live_* 字段，让 live 页基于「实盘当前仓位 vs 目标」给出准确的建仓/加仓/减仓/清仓建议。
    """
    if result_df is None or len(result_df) == 0:
        return {
            'strategy_id': strategy_id,
            'name': strategy.get_display_name() if strategy else strategy_id,
            'status': _STRATEGY_RULE14_STATUS.get(strategy_id, 'research'),
            'as_of_date': None,
            'error': '缓存未加载',
        }
    latest = result_df.iloc[-1]
    prev = result_df.iloc[-2] if len(result_df) >= 2 else latest
    target = float(latest.get('target_exposure', 0.0) or 0.0)
    prev_exp = float(prev.get('target_exposure', 0.0) or 0.0)
    rebalance = str(latest.get('rebalance_action') or 'hold')
    signal = str(latest.get('signal_action') or 'hold')
    exposure_delta = round(target - prev_exp, 4)
    ref_close = float(latest.get('etf_close', float('nan')))
    if pd.isna(ref_close):
        ref_close = None
    ref_open = float(latest.get('etf_open', float('nan')))
    if pd.isna(ref_open):
        ref_open = None
    etf_code = latest.get('etf_code') or None
    etf_name = latest.get('etf_name') or (strategy.get_index_name() if strategy else None)
    status = _STRATEGY_RULE14_STATUS.get(strategy_id, 'research')
    # 实盘视角：基于当前实盘仓位推导真实建议（修复「实盘空仓但显示继续持有」之类的错配）
    live_position = _latest_live_position(strategy_id)
    live_action = _derive_action_from_delta(live_position, target)
    live_delta = round(target - live_position, 4)
    # 行动详解 + 看多风险 / 看空机会
    action_rationale = _build_action_rationale(live_action, live_position, target, ref_open, ref_close, strategy_id)
    view_bias, risks, opportunities = _build_decision_context(strategy_id, target, live_position, live_action)
    profile_window = (profile or {}).get('window_metrics', {}) if isinstance(profile, dict) else {}
    if not isinstance(profile_window, dict):
        profile_window = {}
    win = profile_window.get('recent_6m', {}) if isinstance(profile_window, dict) else {}
    # 策略验证概览：训练 cutoff / holdout 区间 / 训练区核心指标 / OOS 真实表现
    holdout_payload = _load_holdout_report(strategy_id) or {}
    holdout_metrics = holdout_payload.get('holdout_metrics') if isinstance(holdout_payload, dict) else None
    training_metrics = holdout_payload.get('training_window_metrics') or profile_window or {}
    if not isinstance(training_metrics, dict):
        training_metrics = {}
    experiment_meta = {
        'training_cutoff': (profile or {}).get('training_cutoff') if isinstance(profile, dict) else None,
        'holdout_start': holdout_payload.get('holdout_start'),
        'holdout_end': holdout_payload.get('holdout_end'),
        'holdout_bars': holdout_payload.get('holdout_bars'),
        'training_recent_6m': training_metrics.get('recent_6m'),
        'training_full_pre_cutoff': training_metrics.get('full_pre_cutoff'),
        'holdout_metrics': holdout_metrics,
    } if (profile or holdout_payload) else None
    # macro_v32 fallback profile 场景：window_metrics 为 None 时，从 holdout_metrics 推一份 6m 展示数据
    if (not win) and isinstance(holdout_metrics, dict):
        win = {
            'total_return': holdout_metrics.get('final_nav', 1.0) - 1.0,
            'max_drawdown': holdout_metrics.get('max_drawdown', 0.0),
        }
    return {
        'strategy_id': strategy_id,
        'name': strategy.get_display_name() if strategy else strategy_id,
        'index_name': strategy.get_index_name() if strategy else etf_name,
        'etf_code': etf_code,
        'etf_name': etf_name,
        'status': status,
        'passes_rule14': status == 'production',
        'as_of_date': pd.to_datetime(latest['交易日期']).strftime('%Y-%m-%d'),
        'target_exposure': round(target, 4),
        'prev_exposure': round(prev_exp, 4),
        'exposure_delta': exposure_delta,
        'rebalance_action': rebalance,
        'rebalance_label': _REBALANCE_ACTION_LABELS.get(rebalance, rebalance),
        # 相对当前实盘仓位的建议（优先在 live 页展示）
        'live_position': round(live_position, 4),
        'live_exposure_delta': live_delta,
        'live_rebalance_action': live_action,
        'live_rebalance_label': _REBALANCE_ACTION_LABELS.get(live_action, live_action),
        # 决策详解：行动原因、风险（看多）、跨市场机会（看空）
        'action_rationale': action_rationale,
        'view_bias': view_bias,
        'risks': risks,
        'opportunities': opportunities,
        'signal_action': signal,
        'signal_label': _SIGNAL_ACTION_LABELS.get(signal, signal),
        'reason_summary': latest.get('reason_summary') or '',
        'ref_close': ref_close,
        'ref_open': ref_open,
        'exec_basis': '信号基于 T 日收盘生成；T+1 交易日按开盘价执行；当日盈亏按 T+1 收盘价标记。',
        'nav': round(float(latest.get('累积净值', 1.0)), 4) if pd.notna(latest.get('累积净值', None)) else None,
        'profile_recent_6m': {
            'strategy_total_return_pct': round(float(win.get('total_return', 0.0)) * 100, 2) if isinstance(win, dict) and 'total_return' in win else None,
            'etf_total_return_pct': round(float(win.get('etf_total_return', 0.0)) * 100, 2) if isinstance(win, dict) and 'etf_total_return' in win else None,
            'excess_return_pct': round(float(win.get('excess_return', 0.0)) * 100, 2) if isinstance(win, dict) and 'excess_return' in win else None,
            'max_drawdown_pct': round(float(win.get('max_drawdown', 0.0)) * 100, 2) if isinstance(win, dict) and 'max_drawdown' in win else None,
            'etf_max_drawdown_pct': round(float(win.get('etf_max_drawdown', 0.0)) * 100, 2) if isinstance(win, dict) and 'etf_max_drawdown' in win else None,
        } if isinstance(win, dict) and win else None,
        'experiment_meta': experiment_meta,
    }


@app.route('/api/timing/latest_signal')
def api_timing_latest_signal():
    """A股某个择时策略的最新入市信号（单策略详情）。"""
    strategy_name = request.args.get('strategy', 'csi1000_timing')
    if strategy_name not in TIMING_STRATEGY_MAP:
        return jsonify({'error': f'未知策略: {strategy_name}'}), 404
    try:
        init_timing_cache()
    except Exception as e:
        return jsonify({'error': f'择时缓存初始化失败: {e}'}), 500
    strategy = TIMING_STRATEGY_MAP[strategy_name]()
    result = TIMING_CACHE.get(strategy_name)
    profile = _load_best_profile(strategy_name)
    return jsonify(_build_latest_signal(strategy_name, strategy, result, profile))


@app.route('/api/us_timing/latest_signal')
def api_us_timing_latest_signal():
    """美股某个择时策略的最新入市信号。"""
    strategy_name = request.args.get('strategy', 'macro_v32_timing')
    if strategy_name not in US_TIMING_STRATEGY_MAP:
        return jsonify({'error': f'未知策略: {strategy_name}'}), 404
    try:
        init_us_timing_cache()
    except Exception as e:
        return jsonify({'error': f'美股择时缓存初始化失败: {e}'}), 500
    strategy = US_TIMING_STRATEGY_MAP[strategy_name]()
    result = US_TIMING_CACHE.get(strategy_name)
    profile = _load_best_profile(strategy_name)
    return jsonify(_build_latest_signal(strategy_name, strategy, result, profile))


# ====== 实盘记录持久化 ======
def _ensure_live_trades_file():
    os.makedirs(_LIVE_DATA_DIR, exist_ok=True)
    if not os.path.exists(_LIVE_TRADES_FILE):
        with open(_LIVE_TRADES_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = _csv.DictWriter(f, fieldnames=_LIVE_TRADES_COLUMNS)
            writer.writeheader()


def _read_live_trades():
    _ensure_live_trades_file()
    rows = []
    with open(_LIVE_TRADES_FILE, 'r', newline='', encoding='utf-8') as f:
        reader = _csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def _write_live_trades(rows):
    tmp_path = _LIVE_TRADES_FILE + '.tmp'
    with open(tmp_path, 'w', newline='', encoding='utf-8') as f:
        writer = _csv.DictWriter(f, fieldnames=_LIVE_TRADES_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in _LIVE_TRADES_COLUMNS})
    os.replace(tmp_path, _LIVE_TRADES_FILE)


def _next_record_id(rows):
    max_id = 0
    for r in rows:
        try:
            max_id = max(max_id, int(r.get('record_id') or 0))
        except (TypeError, ValueError):
            continue
    return max_id + 1


@app.route('/api/live/records', methods=['GET'])
def api_live_records():
    strategy = request.args.get('strategy')
    with _LIVE_TRADES_LOCK:
        rows = _read_live_trades()
    if strategy:
        rows = [r for r in rows if r.get('strategy') == strategy]
    rows.sort(key=lambda r: (r.get('date') or '', r.get('record_id') or ''))
    return jsonify({'records': rows})


@app.route('/api/live/record', methods=['POST'])
def api_live_record_create():
    payload = request.get_json(silent=True) or {}
    required = ['date', 'strategy']
    missing = [k for k in required if payload.get(k) in (None, '')]
    if missing:
        return jsonify({'error': f'缺少字段: {missing}'}), 400
    known_strategies = set(TIMING_STRATEGY_MAP.keys()) | set(US_TIMING_STRATEGY_MAP.keys())
    if payload['strategy'] not in known_strategies:
        return jsonify({'error': f'未知策略: {payload["strategy"]}'}), 400

    strategy = payload['strategy']
    # 资金口径：用户可在表单覆盖，否则用默认 5w
    try:
        capital = float(payload.get('capital') or _LIVE_INITIAL_CAPITAL)
    except (TypeError, ValueError):
        return jsonify({'error': 'capital 必须是数值'}), 400
    if capital <= 0:
        return jsonify({'error': 'capital 必须 > 0'}), 400

    # 新口径：用户提交 成交价 + 持仓股数，actual_position 由后端算 = price * shares / capital
    # 旧口径：直接传 actual_position（保留兼容，便于已有调用方/测试脚本）
    exec_price = payload.get('exec_price')
    shares = payload.get('shares')
    actual_position = payload.get('actual_position')

    if exec_price not in (None, '') and shares not in (None, ''):
        try:
            exec_price_f = float(exec_price)
            shares_f = float(shares)
        except (TypeError, ValueError):
            return jsonify({'error': 'exec_price / shares 必须是数值'}), 400
        if exec_price_f <= 0 or shares_f < 0:
            return jsonify({'error': 'exec_price 必须 > 0，shares 必须 >= 0'}), 400
        holding_value = exec_price_f * shares_f
        actual_position_f = holding_value / capital
        if actual_position_f > 1.0:
            # clamp，避免超过 100% 仓位（这里仅限制记账口径，不阻止用户加杠杆，但要给出提示）
            actual_position_f = 1.0
        exec_price_str = f'{exec_price_f:.4f}'
        shares_str = f'{shares_f:.4f}' if shares_f != int(shares_f) else str(int(shares_f))
    elif actual_position not in (None, ''):
        try:
            actual_position_f = float(actual_position)
        except (TypeError, ValueError):
            return jsonify({'error': 'actual_position 必须是 0~1 的浮点数'}), 400
        if not (0.0 <= actual_position_f <= 1.0):
            return jsonify({'error': 'actual_position 必须在 [0, 1] 之间'}), 400
        exec_price_str = str(exec_price or '')
        shares_str = str(shares or '')
    else:
        return jsonify({'error': '请提供 exec_price + shares，或直接提供 actual_position'}), 400

    with _LIVE_TRADES_LOCK:
        rows = _read_live_trades()
        new_id = _next_record_id(rows)
        new_row = {
            'record_id': str(new_id),
            'date': str(payload['date']),
            'strategy': str(strategy),
            'signal_target': str(payload.get('signal_target') or ''),
            'actual_position': f'{actual_position_f:.4f}',
            'exec_price': exec_price_str,
            'capital': f'{capital:.2f}',
            'notes': str(payload.get('notes') or ''),
            'created_at': _datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'shares': shares_str,
        }
        rows.append(new_row)
        _write_live_trades(rows)
    return jsonify({'ok': True, 'record': new_row})


@app.route('/api/live/record/<int:record_id>', methods=['DELETE'])
def api_live_record_delete(record_id):
    with _LIVE_TRADES_LOCK:
        rows = _read_live_trades()
        kept = [r for r in rows if str(r.get('record_id')) != str(record_id)]
        if len(kept) == len(rows):
            return jsonify({'error': f'未找到 record_id={record_id}'}), 404
        _write_live_trades(kept)
    return jsonify({'ok': True, 'deleted': record_id})


@app.route('/api/live/reconcile')
def api_live_reconcile():
    """单策略对账：返回 策略 NAV 序列 vs 实盘 NAV 序列（按实盘起点重置策略基准）。

    实盘 NAV 由 actual_position 走「与策略相同的 ETF 收盘日收益」逻辑近似拟合：
    每天 nav_t+1 = nav_t * (1 + actual_position * etf_daily_return_t+1) - 手续费忽略。
    用户的 exec_price/capital 字段仅供记录，不参与对账（实盘真实成交太碎，过度建模反失真）。
    """
    strategy_name = request.args.get('strategy')
    if not strategy_name:
        return jsonify({'error': '缺少 strategy 参数'}), 400
    cache = TIMING_CACHE if strategy_name in TIMING_STRATEGY_MAP else US_TIMING_CACHE
    if strategy_name not in TIMING_STRATEGY_MAP and strategy_name not in US_TIMING_STRATEGY_MAP:
        return jsonify({'error': f'未知策略: {strategy_name}'}), 404
    if strategy_name in TIMING_STRATEGY_MAP:
        try:
            init_timing_cache()
        except Exception as e:
            return jsonify({'error': f'策略缓存初始化失败: {e}'}), 500
    else:
        try:
            init_us_timing_cache()
        except Exception as e:
            return jsonify({'error': f'美股策略缓存初始化失败: {e}'}), 500
    result = cache.get(strategy_name)
    if result is None or len(result) == 0:
        return jsonify({'error': '策略缓存为空'}), 500
    with _LIVE_TRADES_LOCK:
        all_rows = _read_live_trades()
    live_rows = sorted(
        [r for r in all_rows if r.get('strategy') == strategy_name and r.get('date')],
        key=lambda r: r['date'],
    )
    if not live_rows:
        # 默认空仓初始状态（CLAUDE.md 规则 15）：
        # 没有任何实盘记录时，实盘 NAV = 1.0，actual_position = 0；
        # 不回放策略历史，等待用户录入第一笔实盘交易。
        return jsonify({
            'strategy_id': strategy_name,
            'live_records': 0,
            'empty_state': True,
            'initial_nav': 1.0,
            'initial_position': 0.0,
            'initial_capital': _LIVE_INITIAL_CAPITAL,
            'currency': _LIVE_CURRENCY.get(strategy_name, 'CNY'),
            'lot_size': _LIVE_LOT_SIZE.get(strategy_name, 1),
            'message': '默认空仓状态：实盘 NAV = 1.0，当前持仓 = 0%。录入第一笔实盘交易后开始对账。',
            'series': [],
        })
    start_date = pd.to_datetime(live_rows[0]['date'])
    df = result.copy()
    df['交易日期'] = pd.to_datetime(df['交易日期'])
    cache_max = df['交易日期'].max()
    df = df[df['交易日期'] >= start_date].reset_index(drop=True)
    if len(df) == 0:
        # 实盘录入日期晚于策略缓存的最后一根日线（例如刚刚录入「今天」但 ETF 数据还没刷新）。
        # 返回 200 + 提示语，而不是 500——避免前端把它和「默认空仓」混在一起。
        return jsonify({
            'strategy_id': strategy_name,
            'live_records': len(live_rows),
            'empty_state': False,
            'pending_cache': True,
            'cache_max_date': cache_max.strftime('%Y-%m-%d') if pd.notna(cache_max) else None,
            'first_record_date': start_date.strftime('%Y-%m-%d'),
            'initial_capital': _LIVE_INITIAL_CAPITAL,
            'currency': _LIVE_CURRENCY.get(strategy_name, 'CNY'),
            'lot_size': _LIVE_LOT_SIZE.get(strategy_name, 1),
            'message': f'已录入 {len(live_rows)} 条实盘记录，但首条日期 {start_date.strftime("%Y-%m-%d")} 已超过策略缓存的最后交易日 {cache_max.strftime("%Y-%m-%d") if pd.notna(cache_max) else "—"}。等下次数据刷新后将自动纳入对账。',
            'series': [],
        })

    # 策略 NAV：在 start_date 重置为 1.0
    base_nav = float(df['累积净值'].iloc[0]) or 1.0
    df['strategy_nav'] = df['累积净值'].astype(float) / base_nav

    # 实盘 NAV：用 actual_position 顺序回填，每日 close→close
    pos_by_date = {}
    for r in live_rows:
        try:
            pos_by_date[pd.to_datetime(r['date']).strftime('%Y-%m-%d')] = float(r['actual_position'] or 0.0)
        except ValueError:
            continue
    live_nav = 1.0
    current_pos = 0.0
    prev_close = None
    series = []
    for _, row in df.iterrows():
        d = row['交易日期'].strftime('%Y-%m-%d')
        etf_close = row.get('etf_close')
        if d in pos_by_date:
            current_pos = pos_by_date[d]
        if prev_close is not None and pd.notna(etf_close) and pd.notna(prev_close) and prev_close > 0:
            etf_ret = float(etf_close) / float(prev_close) - 1.0
            live_nav *= (1.0 + current_pos * etf_ret)
        prev_close = etf_close if pd.notna(etf_close) else prev_close
        series.append({
            'date': d,
            'strategy_nav': round(float(row['strategy_nav']), 4),
            'live_nav': round(float(live_nav), 4),
            'actual_position': round(float(current_pos), 4),
            'strategy_target': round(float(row.get('target_exposure', 0.0) or 0.0), 4),
        })
    return jsonify({
        'strategy_id': strategy_name,
        'live_records': len(live_rows),
        'start_date': start_date.strftime('%Y-%m-%d'),
        'initial_capital': _LIVE_INITIAL_CAPITAL,
        'currency': _LIVE_CURRENCY.get(strategy_name, 'CNY'),
        'lot_size': _LIVE_LOT_SIZE.get(strategy_name, 1),
        'series': series,
        'final_strategy_nav': series[-1]['strategy_nav'] if series else None,
        'final_live_nav': series[-1]['live_nav'] if series else None,
    })


@app.route('/live')
def page_live():
    return render_template('live.html')


@app.route('/api/timing/signals')
def api_timing_signals():
    payload = []
    for strategy_id, strategy_cls in TIMING_STRATEGY_MAP.items():
        strategy = strategy_cls()
        result = TIMING_CACHE.get(strategy_id)
        if result is None or len(result) == 0:
            payload.append({
                'id': strategy_id,
                'name': strategy.get_display_name(),
                'index_name': strategy.get_index_name(),
                'date': None,
                'action': None,
                'position': 0,
                'reason_summary': '加载中',
                'nav': None,
            })
            continue
        latest = result.iloc[-1]
        payload.append({
            'id': strategy_id,
            'name': strategy.get_display_name(),
            'index_name': strategy.get_index_name(),
            'date': pd.to_datetime(latest['交易日期']).strftime('%Y-%m-%d'),
            'action': latest['signal_action'],
            'position': int(latest['position']),
            'reason_summary': latest['reason_summary'],
            'nav': round(float(latest['累积净值']), 4),
        })
    return jsonify(payload)


@app.route('/api/timing/backtest')
def api_timing_backtest():
    strategy_name = request.args.get('strategy', 'csi1000_timing')
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    compact = request.args.get('compact', '0') in {'1', 'true', 'yes'}
    benchmark_id = _normalize_benchmark_id(request.args.get('benchmark', DEFAULT_BENCHMARK_ID))

    use_cache = strategy_name in TIMING_CACHE
    if use_cache and strategy_name in _TIMING_CACHE_DEFAULTS:
        for key, default_val in _TIMING_CACHE_DEFAULTS[strategy_name].items():
            raw_val = request.args.get(key)
            if raw_val is None:
                continue
            try:
                if isinstance(default_val, bool):
                    val = _parse_realism_bool(raw_val)
                    if val is None:
                        raise ValueError(f'cannot parse bool: {raw_val}')
                elif isinstance(default_val, str):
                    val = raw_val
                elif isinstance(default_val, int):
                    val = int(raw_val)
                else:
                    val = float(raw_val)
            except (TypeError, ValueError):
                use_cache = False
                break
            if val != default_val:
                use_cache = False
                break

    _t0 = time.time()
    try:
        params = {}
        for key in ['fast_window', 'slow_window', 'momentum_window', 'breakout_window', 'exit_window', 'trend_window', 'momentum_short_window', 'momentum_long_window', 'momentum_threshold', 'enter_threshold', 'add_threshold', 'trim_threshold', 'exit_threshold', 'max_entry_exposure', 'probe_entry_exposure', 'sigmoid_k', 'max_leverage', 'base_position', 'inertia', 'crisis_vix', 'fed_block_weight', 'restrictive_threshold', 'pivot_relief', 'base_floor']:
            val = request.args.get(key, type=float)
            if val is not None:
                params[key] = int(val) if key not in {'momentum_threshold', 'enter_threshold', 'add_threshold', 'trim_threshold', 'exit_threshold', 'max_entry_exposure', 'probe_entry_exposure', 'sigmoid_k', 'max_leverage', 'base_position', 'inertia', 'crisis_vix', 'fed_block_weight', 'restrictive_threshold', 'pivot_relief', 'base_floor'} else float(val)
        confirm_days = request.args.get('confirm_days', type=int)
        if confirm_days is not None:
            params['confirm_days'] = int(confirm_days)
        probe_confirm_days = request.args.get('probe_confirm_days', type=int)
        if probe_confirm_days is not None:
            params['probe_confirm_days'] = int(probe_confirm_days)
        exposure_mode = request.args.get('exposure_mode')
        if exposure_mode:
            params['exposure_mode'] = exposure_mode

        # 12 个真实交易规则参数（5 个 profit_lock_* + 7 个费用/滑点/涨跌停）
        params.update(_collect_realism_params(request.args))

        print(f'[timing/backtest] strategy={strategy_name} cache_hit={use_cache} compact={compact} params={params}')

        if use_cache:
            _t = time.time()
            result = TIMING_CACHE[strategy_name].copy()
            strategy = build_timing_strategy(strategy_name)
            print(f'[timing/backtest] cache copy: {(time.time()-_t)*1000:.0f}ms  rows={len(result)}')
        else:
            _t = time.time()
            result, _, strategy = run_timing_backtest_fresh(strategy_name, benchmark_id=benchmark_id, **params)
            print(f'[timing/backtest] fresh run: {(time.time()-_t)*1000:.0f}ms  rows={len(result)}')

        active_benchmark_id, active_benchmark_series = _get_benchmark_series(benchmark_id)
        # 在 filter 前 capture 真实预热历史起点，传给 summarize_timing_windows
        # 这样 training_range 不会被用户选的窗口截断（CLAUDE.md Rule 13）
        full_history_start = pd.to_datetime(result['交易日期'].min()) if len(result) else None
        result = filter_timing_result(result, start_date=start_date, end_date=end_date)
        if len(result) == 0:
            return jsonify({'error': '所选日期范围内无数据'}), 400

        _t = time.time()
        metrics = evaluate_timing_result(result, benchmark_returns=active_benchmark_series, reset_capital=True)
        print(f'[timing/backtest] evaluate: {(time.time()-_t)*1000:.0f}ms')

        _t = time.time()
        bm_curve = _compute_single_benchmark_curve(result, active_benchmark_series)
        # compact 模式下 benchmark_curves 在 timing_result_to_json 里直接返回 []，跳过计算
        bm_curves = [] if compact else _compute_benchmark_curves(result, INDEX_RETURNS_MAP)
        print(f'[timing/backtest] benchmark curves: {(time.time()-_t)*1000:.0f}ms (compact={compact})')

        _t = time.time()
        payload = timing_result_to_json(
            result,
            metrics,
            benchmark_meta=_get_benchmark_meta(active_benchmark_id),
            benchmark_curve=bm_curve,
            benchmark_curves=bm_curves,
            compact=compact,
        )
        payload['interval_windows'] = summarize_timing_windows(
            result,
            benchmark_returns=active_benchmark_series,
            full_history_start=full_history_start,
        )
        print(f'[timing/backtest] to_json: {(time.time()-_t)*1000:.0f}ms')
        print(f'[timing/backtest] total: {(time.time()-_t0)*1000:.0f}ms')
        return jsonify(payload)
    except Exception as e:
        print(f'[timing/backtest] ERROR after {(time.time()-_t0)*1000:.0f}ms: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/timing/explore_compare')
def api_timing_explore_compare():
    strategy_name = request.args.get('strategy', 'csi1000_timing')
    benchmark_id = _normalize_benchmark_id(request.args.get('benchmark', DEFAULT_BENCHMARK_ID))
    start_date = request.args.get('start')
    end_date = request.args.get('end')

    staged_defaults = {
        'exposure_mode': request.args.get('exposure_mode', 'staged'),
        'enter_threshold': request.args.get('enter_threshold', type=float) or 0.55,
        'add_threshold': request.args.get('add_threshold', type=float) or 0.75,
        'trim_threshold': request.args.get('trim_threshold', type=float) or 0.35,
        'exit_threshold': request.args.get('exit_threshold', type=float) or 0.15,
        'confirm_days': request.args.get('confirm_days', type=int) or 1,
        'max_entry_exposure': request.args.get('max_entry_exposure', type=float) or 0.5,
    }
    for key in ['fast_window', 'slow_window', 'momentum_window', 'breakout_window', 'exit_window', 'trend_window', 'momentum_short_window', 'momentum_long_window', 'momentum_threshold']:
        val = request.args.get(key, type=float)
        if val is not None:
            staged_defaults[key] = int(val) if key != 'momentum_threshold' else float(val)

    strategy_defaults = dict(_TIMING_CACHE_DEFAULTS.get(strategy_name, {}))
    strategy_defaults.update({k: v for k, v in staged_defaults.items() if v is not None})

    try:
        _, benchmark_series = _get_benchmark_series(benchmark_id)
        binary_result, binary_metrics, _ = run_timing_backtest_fresh(strategy_name, benchmark_id=benchmark_id)
        staged_result, staged_metrics, _ = run_timing_backtest_fresh(strategy_name, benchmark_id=benchmark_id, **strategy_defaults)
        return jsonify({
            'strategy': strategy_name,
            'interval_policy': {
                'windows': ['recent_1m', 'recent_1q', 'recent_6m'],
                'history_bucket': 'pre_6m_history',
                'shared_params': True,
                'reset_capital': True,
            },
            'baseline_binary': {
                'mode': 'binary',
                **_build_timing_compare_payload(binary_result, binary_metrics, benchmark_series, start_date=start_date, end_date=end_date),
            },
            'candidate_staged': {
                'mode': strategy_defaults.get('exposure_mode', 'staged'),
                **_build_timing_compare_payload(staged_result, staged_metrics, benchmark_series, params=strategy_defaults, start_date=start_date, end_date=end_date),
            },
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/strategy_list')
def api_strategy_list():
    strategy_id = get_focused_strategy_id()
    strategy = build_strategy(strategy_id)
    cumulative_return = None
    cached = BACKTEST_CACHE.get(strategy_id)
    if cached is not None:
        result, _ = cached
        if len(result) > 0:
            cumulative_return = f"{float(result['累积净值'].iloc[-1]):.2f}x"
    return jsonify([
        {
            'id': strategy_id,
            'name': strategy.get_display_name(),
            'description': strategy.get_strategy_description(),
            'cumulative_return': cumulative_return,
            'best': True,
            'focus_only': True,
        }
    ])


def _check_a_share_index_etf_alignment():
    mismatches = []
    for index_id in A_SHARE_INDEX_IDS:
        try:
            idx_df = get_index_daily(index_id)
            etf_df = get_timing_etf_daily(index_id)
        except Exception as exc:
            mismatches.append({
                'index_id': index_id,
                'error': repr(exc),
            })
            continue
        idx_max = pd.to_datetime(idx_df['date']).max() if idx_df is not None and len(idx_df) > 0 else None
        etf_max = pd.to_datetime(etf_df['date']).max() if etf_df is not None and len(etf_df) > 0 else None
        if idx_max is None or etf_max is None:
            continue
        if idx_max < etf_max:
            mismatches.append({
                'index_id': index_id,
                'index_max_date': idx_max.strftime('%Y-%m-%d'),
                'etf_max_date': etf_max.strftime('%Y-%m-%d'),
            })
    return mismatches



def _run_index_data_update():
    """在后台线程中强制重新拉取指数与 timing ETF 数据，并重建择时缓存。"""
    global INDEX_RETURNS, INDEX_RETURNS_MAP, TIMING_PANEL, TIMING_CACHE
    status = _INDEX_UPDATE_STATUS
    status['stage'] = 'running'
    status['progress'] = 0
    status['warning'] = None
    status['details'] = None

    index_ids = list(INDEX_CONFIGS.keys())
    total = len(index_ids)

    try:
        for i, index_id in enumerate(index_ids):
            cfg = INDEX_CONFIGS[index_id]
            name = cfg.get('name', index_id)

            status['message'] = f'正在拉取日K：{name}...'
            status['progress'] = int(i / total * 45)
            get_index_daily(index_id, force_refetch=True)

            status['message'] = f'正在计算月度收益：{name}...'
            status['progress'] = int((i + 0.5) / total * 45)
            series = get_index_returns(index_id, force_refetch=True)
            INDEX_RETURNS_MAP[index_id] = series

        status['message'] = '正在刷新择时 ETF 日线缓存...'
        status['progress'] = 55
        refresh_all_timing_etf_daily()

        INDEX_RETURNS = INDEX_RETURNS_MAP.get('csi1000')

        mismatches = _check_a_share_index_etf_alignment()
        if mismatches:
            status['warning'] = 'A股指数日线与对应ETF日线最新日期不一致'
            status['details'] = mismatches
            raise RuntimeError(f'A股指数/ETF日线不同步: {mismatches}')

        status['message'] = '正在重建指数日线面板...'
        status['progress'] = 70
        TIMING_PANEL = None
        ensure_timing_panel_loaded()

        status['message'] = '正在重建择时回测缓存...'
        status['progress'] = 85
        TIMING_CACHE.clear()
        init_timing_cache()
        _save_disk_cache()

        status['stage'] = 'done'
        status['message'] = '指数与ETF数据刷新完成'
        status['progress'] = 100
        print('[index_update] 指数与ETF数据更新完成', file=sys.stderr)

    except Exception as e:
        status['stage'] = 'error'
        status['message'] = f'更新失败: {e}'
        status['progress'] = 0
        if status.get('details') is None:
            status['details'] = {'error': repr(e)}
        print(f'[index_update ERROR] {e}', file=sys.stderr)


@app.route('/api/update_index_data', methods=['POST'])
def api_update_index_data():
    if _INDEX_UPDATE_STATUS.get('stage') == 'running':
        return jsonify({'error': '指数数据更新正在进行中，请勿重复触发'}), 409
    _INDEX_UPDATE_STATUS['stage'] = 'idle'
    _INDEX_UPDATE_STATUS['message'] = ''
    _INDEX_UPDATE_STATUS['progress'] = 0
    _INDEX_UPDATE_STATUS['warning'] = None
    _INDEX_UPDATE_STATUS['details'] = None
    threading.Thread(target=_run_index_data_update, daemon=True).start()
    return jsonify({'status': 'started'})


@app.route('/api/update_index_data/status')
def api_update_index_data_status():
    return jsonify(_INDEX_UPDATE_STATUS)


def _run_data_update():
    """在后台线程中执行数据更新全流程。"""
    global DATA_DF, INDEX_RETURNS, INDEX_RETURNS_MAP, TIMING_PANEL
    status = _UPDATE_DATA_STATUS
    status['running'] = True
    status['error'] = None

    try:
        now = _datetime.now()
        target_year = now.year
        target_month = now.month
        target_prefix = f"{target_year}-{target_month:02d}"
        prev_month = target_month - 1
        prev_year = target_year
        if prev_month == 0:
            prev_month = 12
            prev_year -= 1
        prev_prefix = f"{prev_year}-{prev_month:02d}"

        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stock_data.csv')
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.path.join(repo_root, '.cache')

        # Stage 1: 增量拉取（每只股票本地日线缓存 + 仅补差值）
        # 不再做破坏性清理：增量逻辑会以 upsert 方式重写本月数据行，
        # 历史月份保持原样，prev_month 的 "下周期每天涨跌幅" 列由 supplement_csv_incremental 内部负责回填。
        status['stage'] = 'fetching'
        status['message'] = '正在增量获取最新行情数据（首次约3-5分钟，之后只补差值）...'
        status['progress_pct'] = 10
        print(f'[update] 开始增量拉取 {target_prefix} 数据', file=sys.stderr)

        count = _supplement_csv_incremental(
            csv_path,
            target_year=target_year,
            target_month=target_month,
            cache_dir=cache_dir,
        )
        print(f'[update] 增量拉取完成，本月 upsert {count} 行', file=sys.stderr)
        status['progress_pct'] = 70

        # Stage 2: 重建 parquet（load_data 优先读 parquet，若不刷新会读到陈旧数据）
        parquet_path = csv_path.replace('.csv', '.parquet')
        try:
            status['message'] = '正在重建 Parquet 缓存...'
            status['progress_pct'] = 72
            print('[update] 重建 stock_data.parquet', file=sys.stderr)
            _df_for_parquet = pd.read_csv(csv_path, encoding='gbk', low_memory=False)
            _df_for_parquet.to_parquet(parquet_path, engine='pyarrow', compression='snappy', index=False)
            del _df_for_parquet
            print('[update] parquet 重建完成', file=sys.stderr)
        except Exception as parquet_err:
            # parquet 失败不致命：删掉旧 parquet，让 load_data fallback 到 CSV
            print(f'[update] parquet 重建失败 ({parquet_err})，删除旧 parquet 强制 CSV fallback', file=sys.stderr)
            if os.path.exists(parquet_path):
                os.remove(parquet_path)

        # Stage 3: 清除所有内存缓存并重新加载
        status['stage'] = 'rebuilding_cache'
        status['message'] = '正在重建回测缓存...'
        status['progress_pct'] = 75

        # 清除磁盘缓存
        if os.path.exists(_CACHE_FILE):
            os.remove(_CACHE_FILE)

        # 清除内存缓存
        BACKTEST_CACHE.clear()
        TIMING_CACHE.clear()
        _PROFILE_SUMMARY_CACHE.clear()
        INDEX_RETURNS_MAP.clear()
        DATA_DF = None
        INDEX_RETURNS = None
        TIMING_PANEL = None

        # 重新加载数据
        status['message'] = '正在重新加载股票数据...'
        status['progress_pct'] = 80
        DATA_DF = load_data(csv_path)

        status['message'] = '正在加载指数数据...'
        status['progress_pct'] = 85
        ensure_index_returns_loaded()

        status['message'] = '正在运行选股策略回测...'
        status['progress_pct'] = 90
        init_cache()

        status['message'] = '正在运行择时策略回测...'
        status['progress_pct'] = 95
        init_timing_cache()

        status['stage'] = 'done'
        status['message'] = f'更新完成！新增 {count} 行数据'
        status['progress_pct'] = 100
        print(f'[update] 全部完成', file=sys.stderr)

    except Exception as e:
        status['stage'] = 'error'
        status['message'] = f'更新失败: {e}'
        status['error'] = str(e)
        print(f'[update ERROR] {e}', file=sys.stderr)
    finally:
        status['running'] = False


@app.route('/api/update_data', methods=['POST'])
def api_update_data():
    if _UPDATE_DATA_STATUS['running']:
        return jsonify({'error': '数据更新正在进行中，请勿重复触发'}), 409
    threading.Thread(target=_run_data_update, daemon=True).start()
    return jsonify({'status': 'started', 'message': '数据更新已启动'})


@app.route('/api/update_data/status')
def api_update_data_status():
    return jsonify({
        'running': _UPDATE_DATA_STATUS['running'],
        'stage': _UPDATE_DATA_STATUS['stage'],
        'message': _UPDATE_DATA_STATUS['message'],
        'progress_pct': _UPDATE_DATA_STATUS['progress_pct'],
        'error': _UPDATE_DATA_STATUS['error'],
    })


# ─────────────────────────────────────────────────────────────────────────────
# 因子单独收益曲线分析
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: column name, display name, ascending (True = smaller is better)
_SINGLE_FACTOR_CONFIGS = [
    {'id': 'size',      'name': '总市值(小市值)',    'column': '总市值',           'ascending': True},
    {'id': 'pb',        'name': '市净率倒数(高BM)',  'column': '市净率倒数',       'ascending': False},
    {'id': 'profit',    'name': '净利润TTM(高盈利)', 'column': '归母净利润_ttm',   'ascending': False},
    {'id': 'vol_stab',  'name': '成交额波动(低波动)','column': '成交额std_10',     'ascending': True},
    {'id': 'bias',      'name': 'BIAS20(超跌反弹)',  'column': 'bias_20',          'ascending': True},
    {'id': 'kdj_j',     'name': 'KDJ-J(超卖)',       'column': 'J',                'ascending': True},
    {'id': 'momentum',  'name': '动量(20日涨幅)',     'column': '涨跌幅_20',        'ascending': False},
    {'id': 'pe_inv',    'name': '市盈率倒数(低PE)',   'column': '市盈率倒数',       'ascending': False},
    # ── 换手率: 异常换手率 = 当日换手率 / 20日均换手率 - 1，越低代表越冷门 ──
    {'id': 'turn_abn',  'name': '异常换手率(冷门)',   'column': '异常换手率',       'ascending': True},
    # ── 缠论 5 因子（由 compute_chan_factors 计算，详见 chan_factors.py） ──
    {'id': 'chan_div',  'name': '缠论背驰强度',       'column': 'chan_div_strength','ascending': False},
    {'id': 'chan_zs',   'name': '缠论中枢位置(下方)','column': 'chan_zs_position', 'ascending': True},
    {'id': 'chan_fr',   'name': '缠论底分型',         'column': 'chan_bottom_fractal','ascending': False},
    {'id': 'chan_str',  'name': '缠论笔强度',         'column': 'chan_stroke_strength','ascending': False},
    {'id': 'chan_sig',  'name': '缠论买卖点综合得分',  'column': 'chan_signal_score','ascending': False},
]


class _MockStrategy:
    """Minimal strategy-like object for single-factor backtest."""
    name = 'single_factor'
    c_rate = 1.0 / 10000
    t_rate = 1.0 / 1000
    bull_tp = 0.30
    bear_tp = 0.22
    bull_n = 6
    bear_n = 4
    initial_capital = 100000

    def build_position_weights(self, selected_df):
        if selected_df is None or len(selected_df) == 0:
            return []
        n = len(selected_df)
        return [round(1.0 / n, 6)] * n

    def build_selection_reason(self, row, rank, total):
        return {'summary': f'单因子排名第{rank}/{total}', 'details': [], 'fundamentals': {}, 'factor_breakdown': []}


_EXTRA_FACTOR_COLS_READY = False


def _ensure_extra_factor_columns():
    """一次性在 DATA_DF 上计算 缠论 5 因子 + 异常换手率，供单因子回测使用。"""
    global DATA_DF, _EXTRA_FACTOR_COLS_READY
    if _EXTRA_FACTOR_COLS_READY:
        return
    if DATA_DF is None:
        return

    # 1) 异常换手率 = 当日成交额/流通市值 / 该股票20日均值 - 1
    if '异常换手率' not in DATA_DF.columns:
        if {'成交额', '流通市值'}.issubset(DATA_DF.columns):
            turn = DATA_DF['成交额'] / DATA_DF['流通市值'].replace(0, np.nan)
            DATA_DF['_turn_raw'] = turn
            g = DATA_DF.sort_values(['股票代码', '交易日期']).groupby('股票代码')['_turn_raw']
            DATA_DF['_turn_ma20'] = g.transform(lambda s: s.rolling(20, min_periods=5).mean())
            DATA_DF['异常换手率'] = DATA_DF['_turn_raw'] / DATA_DF['_turn_ma20'].replace(0, np.nan) - 1
            DATA_DF.drop(columns=['_turn_raw', '_turn_ma20'], inplace=True)
            print('[init] 已计算 异常换手率 列')

    # 2) 缠论 5 因子
    needed_chan = ['chan_div_strength', 'chan_zs_position', 'chan_bottom_fractal',
                   'chan_stroke_strength', 'chan_signal_score']
    if not all(c in DATA_DF.columns for c in needed_chan):
        try:
            from chan_factors import compute_chan_factors
            DATA_DF = compute_chan_factors(DATA_DF)
            print(f"[init] 缠论因子列已生成: {[c for c in needed_chan if c in DATA_DF.columns]}")
        except Exception as e:
            print(f'[init] 缠论因子计算失败: {e}', file=sys.stderr)

    _EXTRA_FACTOR_COLS_READY = True


def _run_single_factor_backtest(top_k=5):
    """Run all single-factor backtests. Returns list of per-factor result dicts."""
    ensure_stock_data_loaded()
    if DATA_DF is None:
        raise RuntimeError("数据未加载")

    _ensure_extra_factor_columns()

    required_cols = ['交易日期', '股票代码', '股票名称', '市场状态', '下周期每天涨跌幅']
    regime_map = DATA_DF.groupby('交易日期')['市场状态'].first()

    results = []
    for fc in _SINGLE_FACTOR_CONFIGS:
        col = fc['column']
        if col not in DATA_DF.columns:
            continue
        df = DATA_DF[required_cols + [col]].dropna(subset=[col]).copy()
        df['因子'] = df[col] if fc['ascending'] else -df[col]

        try:
            result = select_and_backtest(df, _MockStrategy(), select_stock_num=top_k)
        except Exception as e:
            print(f"[factor backtest] {fc['id']} failed: {e}", file=sys.stderr)
            continue

        ev = strategy_evaluate(result)

        # per-regime average monthly return
        result2 = result.copy()
        result2['regime'] = result2['交易日期'].map(regime_map)
        regime_metrics = {}
        for regime_key, regime_label in [('bull', '牛市'), ('bear', '熊市')]:
            sub = result2[result2['regime'] == regime_key]
            if len(sub) < 2:
                regime_metrics[regime_label] = {'avg_monthly_return': 'N/A', 'n_periods': 0}
            else:
                avg_ret = float(sub['选股下周期涨跌幅'].mean())
                regime_metrics[regime_label] = {
                    'avg_monthly_return': f"{round(avg_ret * 100, 2)}%",
                    'n_periods': len(sub),
                }

        results.append({
            'id': fc['id'],
            'name': fc['name'],
            'column': col,
            'dates': [str(d)[:10] for d in result['交易日期'].tolist()],
            'nav': [round(float(v), 4) for v in result['累积净值'].tolist()],
            'annual_return': str(ev.loc['年化收益', 0]),
            'max_drawdown': str(ev.loc['最大回撤', 0]),
            'calmar': float(ev.loc['年化收益/回撤比', 0]),
            'regime_metrics': regime_metrics,
        })

    return results


# ── 行业热度 ──────────────────────────────────────────────────────────────────
_SECTOR_HEAT_FILE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', 'strategy', 'sector_weekly_heat.csv')
)
_SECTOR_HEAT_CACHE = {'mtime': 0, 'data': None}


def _load_sector_heat():
    """懒加载并缓存 sector_weekly_heat.csv（文件变更时自动刷新）。"""
    try:
        mtime = os.path.getmtime(_SECTOR_HEAT_FILE)
    except FileNotFoundError:
        return None
    if _SECTOR_HEAT_CACHE['data'] is None or mtime != _SECTOR_HEAT_CACHE['mtime']:
        df = pd.read_csv(_SECTOR_HEAT_FILE, encoding='utf-8-sig')
        _SECTOR_HEAT_CACHE['data'] = df
        _SECTOR_HEAT_CACHE['mtime'] = mtime
    return _SECTOR_HEAT_CACHE['data']


@app.route('/api/sector_heat')
def api_sector_heat():
    """
    返回最近 N 周（默认 8 周）各行业涨跌幅热度。

    Query params:
      weeks   — 返回的周数（默认 8）
      level   — 'l1'（默认，申万一级）| 'l2'（未支持，留扩展口）

    Response JSON:
      {
        weeks:      ["2026-04 W1", ...],
        industries: ["电子", ...],
        data: [[row_idx, col_idx, pct_value], ...],   // ECharts heatmap series
        latest_ranking: [{industry, avg_ret, rank}, ...]
      }
    """
    n_weeks = request.args.get('weeks', 8, type=int)
    df = _load_sector_heat()
    if df is None:
        return jsonify({'error': f'找不到 {_SECTOR_HEAT_FILE}，请先运行 scripts/compute_sector_weekly_heat.py'}), 503

    latest_weeks = sorted(df['week_label'].unique())[-n_weeks:]
    sub = df[df['week_label'].isin(latest_weeks)].copy()

    industries = sorted(sub['industry'].unique(), key=lambda x: (
        sub[sub['industry'] == x]['weekly_ret_pct'].mean()
    ), reverse=True)
    week_list = sorted(latest_weeks)
    ind_idx = {ind: i for i, ind in enumerate(industries)}
    week_idx = {w: j for j, w in enumerate(week_list)}

    heat_data = []
    for _, row in sub.iterrows():
        i = ind_idx.get(row['industry'])
        j = week_idx.get(row['week_label'])
        if i is not None and j is not None:
            heat_data.append([j, i, round(float(row['weekly_ret_pct']), 2)])

    recent_4w = week_list[-4:]
    ranking = (
        sub[sub['week_label'].isin(recent_4w)]
        .groupby('industry')['weekly_ret_pct']
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    ranking.columns = ['industry', 'avg_ret_4w']
    ranking['rank'] = range(1, len(ranking) + 1)

    return jsonify({
        'weeks': week_list,
        'industries': industries,
        'data': heat_data,
        'latest_ranking': ranking.to_dict(orient='records'),
    })


@app.route('/api/factor_single_backtest')
def api_factor_single_backtest():
    """单因子回测结果只读 API：仅返回离线脚本预生成的缓存，绝不在线计算。"""
    top_k = request.args.get('top_k', 5, type=int)
    cache_key = f'top_k={top_k}'

    # 内存缓存命中
    if cache_key in FACTOR_BACKTEST_CACHE:
        return jsonify(FACTOR_BACKTEST_CACHE[cache_key])

    # 内存里没有，尝试从离线产物文件懒加载一次
    if _load_factor_backtest_cache() and cache_key in FACTOR_BACKTEST_CACHE:
        return jsonify(FACTOR_BACKTEST_CACHE[cache_key])

    # 仍然没有 → 显式报错，指引用户运行离线脚本
    return jsonify({
        'error': '单因子回测缓存缺失',
        'detail': (
            f'未找到 {FACTOR_BACKTEST_CACHE_FILE}（或请求的 top_k={top_k} 不在缓存里）。\n'
            f'web_app 不会在线计算这份回测，请先运行离线脚本生成：\n'
            f'  python3 {FACTOR_BACKTEST_BUILD_SCRIPT}\n'
            f'生成后无需重启 Flask，再次刷新即可。'
        ),
        'cache_file': FACTOR_BACKTEST_CACHE_FILE,
        'build_script': FACTOR_BACKTEST_BUILD_SCRIPT,
    }), 503


if __name__ == '__main__':
    print("=" * 60)
    print("  量化策略 Web 可视化")
    print("  访问 http://localhost:8080")
    print("=" * 60)

    def _eager_load():
        """在后台线程中预加载数据，这样前端可以第一时间显示加载进度。"""
        global _LOAD_STATUS, DATA_DF
        _LOAD_STATUS['loading'] = True
        _LOAD_STATUS['start_time'] = time.time()
        try:
            # Stage 1: load stock data directly (not via ensure_stock_data_loaded, to avoid deadlock)
            _LOAD_STATUS['stage'] = 'stock_data'
            _LOAD_STATUS['message'] = '正在加载股票数据 (823MB CSV)...'
            csv_path = os.path.join(os.path.dirname(__file__), 'stock_data.csv')
            if os.path.exists(csv_path) and DATA_DF is None:
                DATA_DF = load_data(csv_path)
            # Stage 2: load index returns
            _LOAD_STATUS['stage'] = 'index_data'
            _LOAD_STATUS['message'] = '正在加载指数收益数据...'
            ensure_index_returns_loaded()
            # Stage 3: pre-run stock selection strategy cache (1 focused strategy)
            _LOAD_STATUS['stage'] = 'strategy_cache'
            _LOAD_STATUS['message'] = '正在预热选股策略回测缓存...'
            init_cache()
            # Stage 4: pre-run timing strategy cache (3 timing strategies)
            _LOAD_STATUS['stage'] = 'timing_cache'
            _LOAD_STATUS['message'] = '正在预热择时策略回测缓存...'
            init_timing_cache()
            # Stage 5: 仅从离线产物加载单因子回测结果（绝不在线计算）。
            # 生成步骤见 build_single_factor_cache.py
            _LOAD_STATUS['stage'] = 'factor_cache'
            _LOAD_STATUS['message'] = '正在加载单因子回测缓存...'
            if _load_factor_backtest_cache():
                print('[init] 单因子回测缓存加载成功')
            else:
                print(
                    f'[init] 未找到 {FACTOR_BACKTEST_CACHE_FILE}，'
                    f'/api/factor_single_backtest 将返回 503。\n'
                    f'  请离线运行: python3 {FACTOR_BACKTEST_BUILD_SCRIPT}'
                )
            _LOAD_STATUS['end_time'] = time.time()
            _LOAD_STATUS['loading'] = False
            _LOAD_STATUS['message'] = '数据加载完成'
            _LOAD_STATUS['stage'] = 'ready'
            print(f'[init] 全部数据加载完成，耗时 {_LOAD_STATUS["end_time"] - _LOAD_STATUS["start_time"]:.1f}s')
        except Exception as e:
            _LOAD_STATUS['loading'] = False
            _LOAD_STATUS['message'] = f'加载失败: {e}'
            _LOAD_STATUS['stage'] = 'error'
            print(f'[init ERROR] {e}')
        finally:
            _DATA_READY.set()

    threading.Thread(target=_eager_load, daemon=True).start()
    app.run(debug=False, host='0.0.0.0', port=8080, threaded=True)
