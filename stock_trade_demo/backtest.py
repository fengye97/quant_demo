"""
回测引擎 — 选股→止盈→评估。

核心流程：
  1. select_and_backtest(df, strategy)  — 从因子排名到资金曲线
  2. strategy_evaluate(result)          — 评估回测结果

市场状态判断（牛市/熊市）：
  使用全市场等权涨跌幅的累积收益曲线与 MA12 比较。
  cumret > MA12 → 牛市（使用更宽松的止盈参数）
  cumret < MA12 → 熊市（使用更保守的止盈参数）

止盈机制：
  牛市：30% 止盈，持有最多 6 只
  熊市：22% 止盈，持有最多 4 只
  触发止盈后平仓（扣除卖出成本），剩余时间资金闲置（0涨幅）

数据加载：
  - 优先使用 stock_data.parquet（Snappy 压缩，读取更快）
  - 如 Parquet 不存在则回退到 stock_data.csv（GBK 编码）
  - 支持 columns 参数按需加载列以减少内存占用
"""

import ast
import json
import math
import os
import pandas as pd
import numpy as np

def compute_alpha_beta(portfolio_returns, index_returns):
    """
    Compute alpha/beta attribution metrics via OLS regression.

    Regresses strategy period returns on aligned benchmark period returns:
        strategy_ret = alpha + beta * index_ret + epsilon

    Alignment rule:
      1. Prefer exact timestamp match (works for weekly/date-aligned series)
      2. Fallback to (year, month) match for legacy monthly benchmark series

    Parameters:
        portfolio_returns: pd.Series of strategy period returns,
                           indexed by trading date.
        index_returns:     pd.Series of benchmark period returns,
                           indexed by period-end date.

    Returns:
        dict with keys: beta, alpha_period, alpha_monthly, alpha_annualized,
        tracking_error, information_ratio, up_capture, down_capture,
        r_squared, n_periods, avg_period_days. Returns {'error': ...} on failure.
    """
    exact_lookup = {}
    month_lookup = {}
    for idx, val in index_returns.items():
        ts = pd.to_datetime(idx)
        exact_lookup[pd.Timestamp(ts).normalize()] = float(val)
        month_lookup[(ts.year, ts.month)] = float(val)

    aligned_dates = []
    x_vals = []
    y_vals = []
    for date, ret in portfolio_returns.items():
        ts = pd.to_datetime(date)
        idx_ret = exact_lookup.get(pd.Timestamp(ts).normalize())
        if idx_ret is None:
            idx_ret = month_lookup.get((ts.year, ts.month))
        if idx_ret is not None:
            aligned_dates.append(ts)
            x_vals.append(idx_ret)
            y_vals.append(float(ret))

    n_periods = len(x_vals)
    if n_periods < 2:
        return {'error': f'Insufficient overlapping data: {n_periods} periods'}

    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)

    if n_periods == 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        beta = 1.0
        alpha_period = float(np.mean(y - x))
        r_squared = 0.0
        residuals = y - x
    else:
        slope, intercept = np.polyfit(x, y, 1)
        beta = float(slope)
        alpha_period = float(intercept)
        residuals = y - (intercept + slope * x)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    if n_periods >= 2:
        span_days = max((aligned_dates[-1] - aligned_dates[0]).days, 1)
        avg_period_days = max(span_days / max(n_periods - 1, 1), 1.0)
    else:
        avg_period_days = 30.0
    periods_per_year = max(365.0 / avg_period_days, 1.0)
    monthly_scale = 30.0 / avg_period_days

    if alpha_period <= -0.999999:
        annualized_alpha = -1.0
    else:
        annualized_alpha = (1 + alpha_period) ** periods_per_year - 1
    alpha_monthly = (1 + alpha_period) ** monthly_scale - 1 if alpha_period > -0.999999 else -1.0

    excess = y - x
    tracking_error = float(np.std(excess, ddof=1)) if n_periods >= 2 else 0.0

    if tracking_error > 0:
        information_ratio = annualized_alpha / (tracking_error * np.sqrt(periods_per_year))
    else:
        information_ratio = 0.0

    up_mask = x > 0
    down_mask = x < 0

    if up_mask.any() and np.mean(x[up_mask]) != 0:
        up_capture = float(np.mean(y[up_mask]) / np.mean(x[up_mask]))
    else:
        up_capture = None

    if down_mask.any() and np.mean(x[down_mask]) != 0:
        down_capture = float(np.mean(y[down_mask]) / np.mean(x[down_mask]))
    else:
        down_capture = None

    return {
        'beta': round(beta, 4),
        'alpha_period': round(alpha_period, 6),
        'alpha_monthly': round(alpha_monthly, 6),
        'alpha_annualized': round(annualized_alpha, 6),
        'tracking_error': round(tracking_error, 6),
        'information_ratio': round(information_ratio, 4),
        'r_squared': round(r_squared, 4),
        'up_capture': round(up_capture, 4) if up_capture is not None else None,
        'down_capture': round(down_capture, 4) if down_capture is not None else None,
        'n_periods': n_periods,
        'n_months': n_periods,
        'avg_period_days': round(avg_period_days, 2),
    }


def safe_float(series, default=0.0):
    """将 Series 转为 float，非法值填 default。"""
    return pd.to_numeric(series, errors='coerce').fillna(default)


def load_data(path=None, columns=None):
    """
    加载股票数据并计算市场状态。

    数据源自动检测：
      - 如果 path 以 .parquet 结尾 → 直接读取 Parquet 文件
      - 如果 path 以 .csv 结尾 → 优先查找同名 .parquet 文件，不存在则读 CSV
      - 如果 path 为 None → 依次查找 stock_data.parquet / stock_data.csv

    参数:
      path    — 数据文件路径，None 时自动检测
      columns — 可选，只加载指定列（减少内存占用）。Parquet 模式下仅读取
                需要的列（列裁剪），CSV 模式会先读全部再筛选。
                传入 None 表示读取全部列。

    市场状态判定：
      - 计算每日全市场等权平均涨跌幅
      - 累积收益曲线 vs MA12：cum > ma12 → bull，否则 bear
      - 牛市通常对应更大止盈阈值和更多持仓

    返回带"市场状态"列的 DataFrame。
    """
    # ── 自动检测数据源 ──
    if path is None:
        # 优先 Parquet，其次 CSV
        for candidate in ['stock_data.parquet', 'stock_data.csv']:
            if os.path.exists(candidate):
                path = candidate
                break
        else:
            raise FileNotFoundError(
                "未找到 stock_data.parquet 或 stock_data.csv，"
                "请先运行 convert_data.py 生成 Parquet 文件，"
                "或确保 stock_data.csv 存在于当前目录"
            )

    # ── 根据扩展名确定格式 ──
    if path.endswith('.parquet'):
        use_parquet = True
    elif path.endswith('.csv'):
        # 尝试查找同名 Parquet 文件
        parquet_path = path.replace('.csv', '.parquet')
        if os.path.exists(parquet_path):
            path = parquet_path
            use_parquet = True
        else:
            use_parquet = False
    else:
        raise ValueError(f"不支持的文件格式: {path}，仅支持 .csv 和 .parquet")

    # ── 确定实际需要加载的列 ──
    # 交易日期 和 涨跌幅 是市场状态计算的必要列，始终需要
    required_cols = {'交易日期', '涨跌幅'}
    if columns is not None:
        columns = list(columns)
        load_columns = list(set(columns) | required_cols)
    else:
        load_columns = None  # 加载全部列

    # ── 读取数据 ──
    if use_parquet:
        if load_columns is not None:
            # Parquet 原生支持列裁剪，只读取需要的列
            df = pd.read_parquet(path, columns=load_columns)
        else:
            df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, encoding='gbk',
                         parse_dates=['交易日期'], low_memory=False)
        if load_columns is not None:
            # CSV 不支持列裁剪，先全量读再筛选
            keep_cols = [c for c in load_columns if c in df.columns]
            df = df[keep_cols]

    # 确保 交易日期 是 datetime 类型（Parquet 原生保留，CSV 由 parse_dates 处理）
    if not pd.api.types.is_datetime64_any_dtype(df['交易日期']):
        df['交易日期'] = pd.to_datetime(df['交易日期'])

    # 同月多行归一：每个 YYYY-MM 只保留该月最晚的那个交易日的所有行。
    # 每次月中 supplement_csv_incremental 会产生不同日期的行（如 5-11 / 5-12 / 5-22 / 5-25），
    # 不同股票的最新日期可能不同。若直接让回测按「交易日期」分组，这些中间日期会被视为
    # 独立换仓期，导致同一个自然月内出现多次换仓。
    # 修复策略：对每个 YYYY-MM 计算「最晚交易日（canonical_date）」，
    # 只保留该日期的行——其他日期的行（数据落后于最晚快照）一律丢弃，
    # 与月度换仓一次的语义对齐。注：数据缺失则该月只用有数据的股票，不伪造。
    _ym = df['交易日期'].dt.to_period('M')
    _canonical_date = df.groupby(_ym)['交易日期'].transform('max')
    df = df[df['交易日期'] == _canonical_date].reset_index(drop=True)

    # Invariant：每个 YYYY-MM 只能映射到 1 个 canonical 交易日；期数必须 == 月数。
    # 这两条 assert 是 eb4d369（同月多行 silent 多次换仓）的回归防线，绝不能去掉。
    _ym_to_dates = df.groupby(df['交易日期'].dt.to_period('M'))['交易日期'].nunique()
    if not (_ym_to_dates == 1).all():
        _bad = _ym_to_dates[_ym_to_dates > 1].to_dict()
        raise AssertionError(
            f"load_data() invariant 失败：以下月份有多个 canonical 交易日 {_bad}。"
            f"dedup 逻辑可能被绕过——禁止继续回测。"
        )
    _n_periods = df['交易日期'].nunique()
    _ym_span = df['交易日期'].dt.to_period('M').nunique()
    if _n_periods != _ym_span:
        raise AssertionError(
            f"load_data() invariant 失败：期数 {_n_periods} 与月数 {_ym_span} 不一致。"
        )

    # ── 全市场等权累积收益 → 市场牛熊划分 ──
    mkt_ret = df.groupby('交易日期')['涨跌幅'].mean()
    mkt_cum = (1 + mkt_ret).cumprod()
    mkt_ma12 = mkt_cum.rolling(12).mean()
    df['市场状态'] = df['交易日期'].map(
        (mkt_cum > mkt_ma12).map({True: 'bull', False: 'bear'})
    )

    # ── 数值化关键列 ──
    numeric_cols = [
        '总市值', 'bias_20', '成交额std_10', '市盈率倒数', '市净率倒数',
        '最高价', '最低价', '收盘价', 'MACD', 'DIF', 'DEA',
        '涨跌幅_20', '涨跌幅std_20', '成交额'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = safe_float(df[col])
    return df


def get_data_info(path=None):
    """
    获取数据文件元信息，不加载全量数据。

    用于 Web 应用显示数据摘要而无需加载整个文件。

    参数:
      path — 数据文件路径，None 时自动检测（同 load_data）

    返回 dict:
      file_path    — 实际使用的文件路径
      file_size_mb — 文件大小 (MB)
      num_rows     — 总行数
      num_columns  — 总列数
      columns      — 列名列表
      date_start   — 最早日期 (str, YYYY-MM-DD)
      date_end     — 最晚日期 (str, YYYY-MM-DD)
      stock_count  — 唯一股票代码数量
      format       — 'parquet' 或 'csv'
    """
    # ── 自动检测数据源 ──
    if path is None:
        for candidate in ['stock_data.parquet', 'stock_data.csv']:
            if os.path.exists(candidate):
                path = candidate
                break
        else:
            raise FileNotFoundError(
                "未找到 stock_data.parquet 或 stock_data.csv"
            )

    # ── CSV → Parquet 自动选择 ──
    use_parquet = path.endswith('.parquet')
    if not use_parquet and path.endswith('.csv'):
        parquet_path = path.replace('.csv', '.parquet')
        if os.path.exists(parquet_path):
            path = parquet_path
            use_parquet = True

    file_size_mb = os.path.getsize(path) / (1024 * 1024)

    if use_parquet:
        import pyarrow.parquet as pq

        metadata = pq.read_metadata(path)
        schema = pq.read_schema(path)
        num_rows = metadata.num_rows
        columns = schema.names
        num_columns = len(columns)

        # 读取日期范围：只读 交易日期 列的最小值和最大值
        # Parquet 支持列裁剪，所以这个操作很轻量
        dates = pd.read_parquet(path, columns=['交易日期'])
        date_start = dates['交易日期'].min()
        date_end = dates['交易日期'].max()

        # 股票数量：只读 股票代码 列的唯一值
        codes = pd.read_parquet(path, columns=['股票代码'])
        stock_count = codes['股票代码'].nunique()
    else:
        # CSV 模式：只读列名和日期范围
        df_head = pd.read_csv(path, encoding='gbk', nrows=0)
        columns = list(df_head.columns)
        num_columns = len(columns)

        # 读取 交易日期 列来获取范围和行数
        df_dates = pd.read_csv(path, encoding='gbk',
                               usecols=['交易日期'],
                               parse_dates=['交易日期'])
        num_rows = len(df_dates)
        date_start = df_dates['交易日期'].min()
        date_end = df_dates['交易日期'].max()

        # 股票数量
        df_codes = pd.read_csv(path, encoding='gbk', usecols=['股票代码'])
        stock_count = df_codes['股票代码'].nunique()

    return {
        'file_path': path,
        'file_size_mb': round(file_size_mb, 1),
        'num_rows': num_rows,
        'num_columns': num_columns,
        'columns': columns,
        'date_start': date_start.strftime('%Y-%m-%d') if hasattr(date_start, 'strftime') else str(date_start)[:10],
        'date_end': date_end.strftime('%Y-%m-%d') if hasattr(date_end, 'strftime') else str(date_end)[:10],
        'stock_count': stock_count,
        'format': 'parquet' if use_parquet else 'csv',
    }


def parse_returns(x):
    """
    解析"下周期每天涨跌幅"列。

    该列在 CSV 中存储为字符串形式的 list 或已经是 list。
    返回 list[float] 或空 list。
    """
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []
    return x if isinstance(x, list) else []


# Pillar 1 Step 7: these were extracted verbatim into ``engine/take_profit.py``
# so the math is reusable without dragging the whole backtest module along.
# They are re-exported here for backwards compatibility with every existing
# call site (compare_strategies / choose_stock / scripts / web_app).
from engine.take_profit import (  # noqa: E402,F401  (re-export)
    apply_take_profit,
    build_period_daily_curve,
)
from engine.costs import CommissionModel  # noqa: E402,F401  (re-export)


def select_and_backtest(df, strategy, select_stock_num=6,
                        c_rate=1.0 / 10000, t_rate=1 / 1000,
                        bull_tp=0.30, bear_tp=0.22,
                        bull_n=6, bear_n=4,
                        initial_capital=100000,
                        timing_signal=None,
                        stop_loss_pct=None):
    """
    执行选股和回测。

    流程：
      1. 解析"下周期每天涨跌幅"列
      2. 按因子排名选股（取因子值最小的 select_stock_num 只）
      3. 根据市场状态（牛市/熊市）选择止盈参数
      4. 对每只选中股票应用止盈规则
      5. 组合等权平均，扣买入手续费
      6. 计算资金曲线、累积净值和绝对资金曲线

    参数:
      df              — 策略已打好"因子"列的 DataFrame
      strategy        — 策略实例（用于日志标识）
      select_stock_num — 每期选股数量
      c_rate, t_rate   — 手续费率（买入万1，卖出千1印花税）
      bull_tp, bear_tp — 牛/熊市止盈阈值
      bull_n, bear_n   — 牛/熊市最大持仓数
      initial_capital  — 初始本金，默认 100,000

    返回 DataFrame，列为:
      交易日期, 买入股票代码, 买入股票名称, 选股下周期涨跌幅,
      资金曲线, 累积净值, 当期本金, 当期盈亏, 累计资金, 买入个股收益
    """
    df = df.copy()
    sell_cost = c_rate + t_rate

    _timing_series = None
    if timing_signal is not None:
        if isinstance(timing_signal, pd.Series):
            _timing_series = timing_signal.sort_index()
        elif isinstance(timing_signal, dict):
            _timing_series = pd.Series(
                list(timing_signal.values()),
                index=pd.to_datetime(list(timing_signal.keys()))
            ).sort_index()

    # 解析下周期涨跌幅
    df['下周期每天涨跌幅'] = df['下周期每天涨跌幅'].apply(parse_returns)

    # 排名选股：因子值越小越好，取前 select_stock_num 只
    df['排名'] = df.groupby('交易日期')['因子'].rank(ascending=True)
    df = df[df['排名'] <= select_stock_num]

    # 月中数据更新时（supplement 写入本月最新日但下月尚未开始），
    # 当前月 canonical row 的 下周期每天涨跌幅 全为空 —— 该月还未结束，不构成完整持股周期。
    # 在 groupby 之前统一过滤掉这些 0 期的 canonical date，避免 select_stock 与
    # period_returns 等 list 出现长度不一致（Length mismatch）错误。
    _periods_with_returns = df.groupby('交易日期')['下周期每天涨跌幅'].apply(
        lambda s: any(len(x) > 0 for x in s if isinstance(x, list))
    )
    _valid_dates = _periods_with_returns[_periods_with_returns].index
    df = df[df['交易日期'].isin(_valid_dates)]

    # 格式化代码/名称为空格分隔
    df['股票代码'] = df['股票代码'].astype(str) + ' '
    df['股票名称'] = df['股票名称'].astype(str) + ' '

    group = df.groupby('交易日期')
    select_stock = pd.DataFrame()
    select_stock['买入股票代码'] = group['股票代码'].sum()
    select_stock['买入股票名称'] = group['股票名称'].sum()

    # 逐期计算组合收益，同时跟踪绝对资金
    period_returns = []
    capitals = []       # 当期本金
    pnls = []           # 当期盈亏
    cum_capitals = []   # 累计资金
    holdings_detail = []  # 每期个股明细 (JSON string)
    period_daily_curves = []
    total_buy_fees = 0.0
    total_sell_fees = 0.0

    capital = float(initial_capital)

    for date, grp in group:
        if _timing_series is not None:
            timing_pos = _timing_series.asof(pd.to_datetime(date))
            if pd.isna(timing_pos):
                timing_pos = 1.0
            if float(timing_pos) <= 0.0:
                period_returns.append(0.0)
                capitals.append(capital)
                pnls.append(0.0)
                cum_capitals.append(capital)
                holdings_detail.append(json.dumps([], ensure_ascii=False))
                period_daily_curves.append([1.0])
                continue

        # 根据市场状态选择止盈参数
        regime = grp['市场状态'].iloc[0]
        if regime == 'bull':
            tp, n_stocks = bull_tp, bull_n
        else:
            tp, n_stocks = bear_tp, bear_n

        # 取前 n_stocks 只（保持原有顺序，仅取前 n_stocks 行）
        grp_top = grp.head(n_stocks)
        actual_n_stocks = len(grp_top)
        pool_size = len(grp)
        strategy_name = strategy.__class__.__name__ if strategy is not None else ''
        daily_lists = list(grp_top['下周期每天涨跌幅'])
        stock_codes = list(grp_top['股票代码'].astype(str).str.strip())
        stock_names = list(grp_top['股票名称'].astype(str).str.strip())
        buy_prices = list(grp_top['收盘价']) if '收盘价' in grp_top.columns else [0] * actual_n_stocks
        # 选股因子及财务数据 —— 用于前端展示选股逻辑可解释性
        factor_scores = list(grp_top['因子']) if '因子' in grp_top.columns else [0] * actual_n_stocks
        rankings = list(grp_top['排名']) if '排名' in grp_top.columns else [0] * actual_n_stocks
        industry_l2s = list(grp_top['新版申万二级行业名称']) if '新版申万二级行业名称' in grp_top.columns else [''] * actual_n_stocks
        pe_invs = list(grp_top['市盈率倒数']) if '市盈率倒数' in grp_top.columns else [0] * actual_n_stocks
        pb_invs = list(grp_top['市净率倒数']) if '市净率倒数' in grp_top.columns else [0] * actual_n_stocks
        market_caps = list(grp_top['总市值']) if '总市值' in grp_top.columns else [0] * actual_n_stocks

        capital_start = capital
        target_weights = strategy.build_position_weights(grp_top) if strategy is not None else []
        if actual_n_stocks == 0:
            continue
        if len(target_weights) != actual_n_stocks or sum(target_weights) <= 0:
            target_weights = [1.0 / actual_n_stocks] * actual_n_stocks
        total_weight = sum(target_weights)
        target_weights = [float(w) / total_weight for w in target_weights]
        capital_allocations = [capital_start * w for w in target_weights]

        period_daily_curve = build_period_daily_curve(daily_lists, tp, sell_cost, c_rate, sl_pct=stop_loss_pct)
        period_daily_curves.append(period_daily_curve)

        # 本期买入手续费
        period_buy_fees = capital_start * c_rate
        total_buy_fees += period_buy_fees

        final_rets = []
        stock_details = []
        period_sell_fees = 0.0
        for i, daily_ret in enumerate(daily_lists):
            code = stock_codes[i] if i < len(stock_codes) else ''
            name = stock_names[i] if i < len(stock_names) else ''
            buy_price = float(buy_prices[i]) if i < len(buy_prices) and buy_prices[i] > 0 else 0
            row = grp_top.iloc[i] if i < len(grp_top) else {}
            rank_value = int(rankings[i]) if i < len(rankings) else 0
            reason = strategy.build_selection_reason(row, rank_value, pool_size)

            target_weight = target_weights[i] if i < len(target_weights) else (1.0 / n_stocks)
            capital_per_stock = capital_allocations[i] if i < len(capital_allocations) else (capital_start / n_stocks)

            # 计算持股数量（A股100股整数倍）
            if buy_price > 0:
                shares = int(capital_per_stock / buy_price / 100) * 100
                actual_invested = shares * buy_price
            else:
                shares = 0
                actual_invested = 0

            if not isinstance(daily_ret, list) or len(daily_ret) == 0:
                # 无下期数据：显示买入价，卖出价暂无
                final_rets.append(1.0)
                stock_details.append({
                    'code': code, 'name': name,
                    'weight': round(float(target_weight), 6),
                    'return': 0.0, 'pnl': 0.0,
                    'buy_price': round(buy_price, 2),
                    'sell_price': None,
                    'shares': shares,
                    'factor_score': round(float(factor_scores[i]) if i < len(factor_scores) else 0, 4),
                    'rank': int(rankings[i]) if i < len(rankings) else 0,
                    'industry_l2': str(industry_l2s[i]) if i < len(industry_l2s) else '',
                    'pe': round(1.0 / float(pe_invs[i]), 2) if i < len(pe_invs) and float(pe_invs[i]) != 0 else None,
                    'pb': round(1.0 / float(pb_invs[i]), 2) if i < len(pb_invs) and float(pb_invs[i]) != 0 else None,
                    'market_cap': round(float(market_caps[i]) / 1e8, 2) if i < len(market_caps) else None,
                    'selection_reason_summary': reason['summary'],
                    'selection_reason_detail': reason['details'],
                    'selection_fundamentals': reason['fundamentals'],
                    'selection_factor_breakdown': reason.get('factor_breakdown', []),
                })
                continue

            modified, triggered = apply_take_profit(daily_ret, tp, sell_cost, sl_pct=stop_loss_pct)
            cumret = np.prod([1 + r for r in modified])

            # 计算卖出价（不含交易成本的毛价格）
            if triggered:
                # 找到触发日，计算到触发日为止的累积收益
                gross_cum = 1.0
                for r in daily_ret:
                    gross_cum *= (1 + r)
                    if gross_cum - 1 > tp:
                        break
                sell_price = round(buy_price * gross_cum, 2)
            else:
                gross_cum = np.prod([1 + r for r in daily_ret])
                sell_price = round(buy_price * gross_cum, 2)

            if not triggered:
                cumret *= (1 - sell_cost)
            final_rets.append(cumret)

            # 基于实际投入资金计算净盈亏
            if actual_invested > 0:
                stock_pnl = actual_invested * (cumret - 1)
            else:
                stock_pnl = capital_per_stock * (cumret - 1)

            stock_details.append({
                'code': code, 'name': name,
                'weight': round(float(target_weight), 6),
                'return': round(float(cumret - 1), 6),
                'pnl': round(float(stock_pnl), 2),
                'buy_price': round(buy_price, 2),
                'sell_price': sell_price,
                'shares': shares,
                'factor_score': round(float(factor_scores[i]) if i < len(factor_scores) else 0, 4),
                'rank': int(rankings[i]) if i < len(rankings) else 0,
                'industry_l2': str(industry_l2s[i]) if i < len(industry_l2s) else '',
                'pe': round(1.0 / float(pe_invs[i]), 2) if i < len(pe_invs) and float(pe_invs[i]) != 0 else None,
                'pb': round(1.0 / float(pb_invs[i]), 2) if i < len(pb_invs) and float(pb_invs[i]) != 0 else None,
                'market_cap': round(float(market_caps[i]) / 1e8, 2) if i < len(market_caps) else None,
                'selection_reason_summary': reason['summary'],
                'selection_reason_detail': reason['details'],
                'selection_fundamentals': reason['fundamentals'],
                'selection_factor_breakdown': reason.get('factor_breakdown', []),
            })
            if cumret > 0:
                period_sell_fees += capital_per_stock * cumret * sell_cost

        total_sell_fees += period_sell_fees

        # 组合收益 = 各持仓按目标权重加权后的组合收益 * (1 - 买入手续费)
        portfolio_ret = float(np.dot(final_rets, target_weights))
        portfolio_ret *= (1 - c_rate)
        period_return = portfolio_ret - 1
        period_returns.append(period_return)
        holdings_detail.append(json.dumps(stock_details, ensure_ascii=False))

        # 绝对资金计算
        period_pnl = capital_start * period_return
        capital = capital_start + period_pnl

        capitals.append(capital_start)
        pnls.append(period_pnl)
        cum_capitals.append(capital)

    select_stock['选股下周期涨跌幅'] = period_returns
    select_stock.reset_index(inplace=True)
    select_stock['资金曲线'] = (select_stock['选股下周期涨跌幅'] + 1).cumprod()
    select_stock['累积净值'] = (select_stock['选股下周期涨跌幅'] + 1).cumprod()
    select_stock['当期本金'] = capitals
    select_stock['当期盈亏'] = pnls
    select_stock['累计资金'] = cum_capitals
    select_stock['买入个股收益'] = holdings_detail

    daily_equity_curve = []
    daily_value = 1.0
    for period_idx, period_curve in enumerate(period_daily_curves):
        trade_date = pd.to_datetime(select_stock.iloc[period_idx]['交易日期'])
        if not period_curve:
            daily_equity_curve.append({
                'date': trade_date.strftime('%Y-%m-%d'),
                'value': round(float(daily_value), 6),
                'return': 0.0,
            })
            continue
        prev = 1.0
        for day_idx, period_value in enumerate(period_curve, start=1):
            day_ret = 0.0 if prev == 0 else float(period_value / prev - 1)
            daily_value *= (1 + day_ret)
            daily_equity_curve.append({
                'date': (trade_date + pd.Timedelta(days=day_idx)).strftime('%Y-%m-%d'),
                'value': round(float(daily_value), 6),
                'return': round(day_ret, 6),
            })
            prev = period_value

    # 存储元信息
    select_stock.attrs['initial_capital'] = initial_capital
    select_stock.attrs['c_rate'] = c_rate
    select_stock.attrs['t_rate'] = t_rate
    select_stock.attrs['sell_cost'] = sell_cost
    select_stock.attrs['total_buy_fees'] = total_buy_fees
    select_stock.attrs['total_sell_fees'] = total_sell_fees
    select_stock.attrs['total_fees'] = total_buy_fees + total_sell_fees
    select_stock.attrs['period_daily_curves'] = period_daily_curves
    select_stock.attrs['daily_equity_curve'] = daily_equity_curve

    return select_stock


def strategy_evaluate(select_stock, initial_capital=None, index_returns=None):
    """
    评估策略表现。

    指标：
      - 累积净值：最终累积收益倍数
      - 年化收益：(最终净值)^(365/总天数) - 1
      - 最大回撤：从历史峰值到谷底的最大跌幅
      - 最大回撤起止时间
      - 年化收益/回撤比 (Calmar ratio)
      - 最终资金：回测结束时的绝对资金量
      - 总收益率：(最终资金 / 初始本金 - 1) * 100%
      - 总盈亏：最终资金 - 初始本金

    如果提供 index_returns，还会计算 alpha/beta 归因指标：
      - Beta：策略相对基准的贝塔系数
      - Alpha（年化）：年化超额收益
      - 信息比率：年化Alpha / 年化跟踪误差
      - 上行捕获率 / 下行捕获率：非对称市场表现
      - R-squared：基准收益对策略收益的解释力

    参数:
      select_stock    — select_and_backtest 返回的 DataFrame
      initial_capital — 初始本金，为 None 时从 DataFrame.attrs 或 当期本金 列读取
      index_returns   — 可选，基准指数月度收益 Series（index 为月末日期），
                        传入后会自动计算 alpha/beta 归因指标

    返回 DataFrame(index=指标名, columns=[0]=值)。
    """
    results = pd.DataFrame()

    # 确定初始本金
    if initial_capital is None:
        initial_capital = select_stock.attrs.get('initial_capital')
    if initial_capital is None:
        # 从 当期本金 列的第一行获取
        if '当期本金' in select_stock.columns and len(select_stock) > 0:
            initial_capital = select_stock['当期本金'].iloc[0]
        else:
            initial_capital = 100000

    # 累积净值
    results.loc[0, '累积净值'] = round(select_stock['累积净值'].iloc[-1], 2)

    # 年化收益
    date_delta = select_stock['交易日期'].iloc[-1] - select_stock['交易日期'].iloc[0]
    days = date_delta.days if hasattr(date_delta, 'days') else 365
    if days > 0:
        annual_return = (select_stock['累积净值'].iloc[-1]) ** (365.0 / days) - 1
    else:
        annual_return = 0
    results.loc[0, '年化收益'] = f"{round(annual_return * 100, 2)}%"

    # 最大回撤
    select_stock['max2here'] = select_stock['累积净值'].expanding().max()
    select_stock['dd2here'] = select_stock['累积净值'] / select_stock['max2here'] - 1
    end_date, max_draw_down = tuple(
        select_stock.sort_values(by=['dd2here']).iloc[0][['交易日期', 'dd2here']]
    )
    # 回撤开始时间 = 在结束日期之前累积净值最高的日期
    start_date = (
        select_stock[select_stock['交易日期'] <= end_date]
        .sort_values(by='累积净值', ascending=False)
        .iloc[0]['交易日期']
    )
    select_stock.drop(['max2here', 'dd2here'], axis=1, inplace=True)

    results.loc[0, '最大回撤'] = format(max_draw_down, '.2%')
    results.loc[0, '最大回撤开始'] = str(start_date)[:10]
    results.loc[0, '最大回撤结束'] = str(end_date)[:10]

    # Calmar 比率
    results.loc[0, '年化收益/回撤比'] = (
        round(annual_return / abs(max_draw_down), 2) if max_draw_down != 0 else 0
    )

    # ── 绝对资金指标 ──
    if '累计资金' in select_stock.columns and len(select_stock) > 0:
        final_capital = select_stock['累计资金'].iloc[-1]
    else:
        final_capital = initial_capital * select_stock['累积净值'].iloc[-1]

    results.loc[0, '最终资金'] = round(final_capital, 2)
    results.loc[0, '总收益率'] = f"{round((final_capital / initial_capital - 1) * 100, 2)}%"
    results.loc[0, '总盈亏'] = round(final_capital - initial_capital, 2)

    # ── Alpha / Beta 归因分析 ──
    if index_returns is not None:
        # Build strategy monthly returns series indexed by trading date
        strategy_rets = select_stock.set_index('交易日期')['选股下周期涨跌幅']
        attr = compute_alpha_beta(strategy_rets, index_returns)

        if 'error' not in attr:
            def _finite_number(val):
                return val is not None and isinstance(val, (int, float, np.integer, np.floating)) and math.isfinite(float(val))

            results.loc[0, 'Beta'] = attr['beta'] if _finite_number(attr.get('beta')) else 'N/A'
            results.loc[0, '月度Alpha'] = f"{round(attr['alpha_monthly'] * 100, 4)}%" if _finite_number(attr.get('alpha_monthly')) else 'N/A'
            results.loc[0, '年化Alpha'] = f"{round(attr['alpha_annualized'] * 100, 2)}%" if _finite_number(attr.get('alpha_annualized')) else 'N/A'
            results.loc[0, '信息比率'] = attr['information_ratio'] if _finite_number(attr.get('information_ratio')) else 'N/A'
            results.loc[0, 'R-squared'] = attr['r_squared'] if _finite_number(attr.get('r_squared')) else 'N/A'
            results.loc[0, '跟踪误差（月）'] = f"{round(attr['tracking_error'] * 100, 2)}%" if _finite_number(attr.get('tracking_error')) else 'N/A'
            if _finite_number(attr.get('up_capture')):
                results.loc[0, '上行捕获率'] = f"{round(attr['up_capture'] * 100, 1)}%"
            else:
                results.loc[0, '上行捕获率'] = 'N/A'
            if _finite_number(attr.get('down_capture')):
                results.loc[0, '下行捕获率'] = f"{round(attr['down_capture'] * 100, 1)}%"
            else:
                results.loc[0, '下行捕获率'] = 'N/A'
            results.loc[0, '归因月份数'] = attr.get('n_months', attr.get('n_periods'))
        else:
            results.loc[0, 'Beta'] = 'N/A'
            results.loc[0, '月度Alpha'] = 'N/A'
            results.loc[0, '年化Alpha'] = 'N/A'
            results.loc[0, '信息比率'] = 'N/A'
            results.loc[0, 'R-squared'] = 'N/A'
            results.loc[0, '跟踪误差（月）'] = 'N/A'
            results.loc[0, '上行捕获率'] = 'N/A'
            results.loc[0, '下行捕获率'] = 'N/A'

    return results.T
