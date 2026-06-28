"""
缠论 (Chan Theory) 因子计算模块。

基于 OHLC + MACD 日线数据，计算 5 大类缠论量化因子：
  1. 分型检测 (Fractal)        — 顶分型/底分型
  2. 背驰检测 (Divergence)     — 底背驰/顶背驰 + 背驰强度
  3. 中枢位置 (Zhongshu)       — 中枢上下界/位置/有效性
  4. 笔强度   (Stroke)         — 笔方向/笔强度/笔加速度
  5. 买卖点信号 (Signal)       — 一/二/三类买点 + 综合得分

对应 quant_factor.md Sections 10-14。

使用方式:
    from chan_factors import compute_chan_factors
    df = compute_chan_factors(df)
"""

import numpy as np
import pandas as pd


def compute_chan_factors(df):
    """
    计算 5 大缠论量化因子，返回带因子列的 DataFrame。

    依赖列（必须在 df 中存在）：
      最高价, 最低价, 收盘价, MACD, 涨跌幅_20, 涨跌幅std_20, 成交额

    新增因子列：
      chan_bottom_fractal   — 底分型标记（1=有，0=无）       [Section 12]
      chan_top_fractal      — 顶分型标记（1=有，0=无）       [Section 12]
      chan_bullish_div      — 底背驰标记（1=有，0=无）       [Section 10]
      chan_bearish_div      — 顶背驰标记（1=有，0=无）       [Section 10]
      chan_div_strength     — 背驰强度（0-100）              [Section 10]
      chan_zs_position      — 中枢相对位置（<0下方, 0~1内部, >1上方）[Section 11]
      chan_zs_valid         — 中枢是否有效（ZG > ZD）        [Section 11]
      chan_below_zs         — 是否在中枢下方                  [Section 11]
      chan_above_zs         — 是否在中枢上方                  [Section 11]
      chan_inside_zs        — 是否在中枢内部                  [Section 11]
      chan_stroke_dir       — 笔方向（+1向上, -1向下）       [Section 13]
      chan_stroke_strength  — 笔强度（0-10）                  [Section 13]
      chan_stroke_accel     — 笔加速度                        [Section 13]
      chan_buy_type1        — 一类买点（底分型+底背驰）      [Section 14]
      chan_buy_type2        — 二类买点（底分型或中枢下方）    [Section 14]
      chan_buy_type3        — 三类买点（中枢下方+笔向上）     [Section 14]
      chan_signal_score     — 综合买卖点得分                  [Section 14]
    """
    df = df.sort_values(['股票代码', '交易日期']).copy()

    # ── 滞后值（按股票分组 shift） ──
    # 缠论分析需要比较当前K线与前两根K线的关系
    g = df.groupby('股票代码')
    df['high_l1'] = g['最高价'].shift(1)    # 前一根K线最高价
    df['high_l2'] = g['最高价'].shift(2)    # 前两根K线最高价
    df['low_l1'] = g['最低价'].shift(1)     # 前一根K线最低价
    df['low_l2'] = g['最低价'].shift(2)     # 前两根K线最低价
    df['close_l1'] = g['收盘价'].shift(1)   # 前一根收盘价
    df['close_l2'] = g['收盘价'].shift(2)   # 前两根收盘价
    df['close_l3'] = g['收盘价'].shift(3)   # 前三根收盘价
    df['macd_l2'] = g['MACD'].shift(2)      # 前两根MACD
    df['volume_l1'] = g['成交额'].shift(1)   # 前一根成交额

    # ══════════════════════════════════════════════════════════════
    # 因子 1: 分型检测 (Fractal, Section 12)
    #
    # 缠论定义：
    #   底分型 = 中间K线低点在三者中最低，且中间K线高点在三者中最低
    #           含义：空方力量衰竭，可能反转向上
    #   顶分型 = 中间K线高点在三者中最高，且中间K线低点在三者中最高
    #           含义：多方力量衰竭，可能反转向下
    #
    # K线序列: K1(t-2), K2(t-1), K3(t)，检测 K2 是否为分型点
    # ══════════════════════════════════════════════════════════════
    df['chan_bottom_fractal'] = (
        (df['high_l1'] < df['high_l2']) &     # K2高点 < K1高点
        (df['high_l1'] < df['最高价']) &       # K2高点 < K3高点
        (df['low_l1'] < df['low_l2']) &       # K2低点 < K1低点
        (df['low_l1'] < df['最低价'])          # K2低点 < K3低点
    ).astype(int)

    df['chan_top_fractal'] = (
        (df['high_l1'] > df['high_l2']) &     # K2高点 > K1高点
        (df['high_l1'] > df['最高价']) &       # K2高点 > K3高点
        (df['low_l1'] > df['low_l2']) &       # K2低点 > K1低点
        (df['low_l1'] > df['最低价'])          # K2低点 > K3低点
    ).astype(int)

    # ══════════════════════════════════════════════════════════════
    # 因子 2: 背驰检测 (Divergence, Section 10)
    #
    # 缠论定义：
    #   底背驰 = 价格创新低但MACD动能不再创新低（价格下跌、动能改善）
    #           含义：下跌趋势衰竭，是重要买点信号
    #   顶背驰 = 价格创新高但MACD动能不再创新高（价格上涨、动能衰竭）
    #           含义：上涨趋势衰竭，是重要卖点信号
    #
    # 简化实现：比较当前与 t-2 的价格方向和 MACD 方向是否相反
    #   close < close_{t-2} AND MACD > MACD_{t-2} → 底背驰
    #   close > close_{t-2} AND MACD < MACD_{t-2} → 顶背驰
    # ══════════════════════════════════════════════════════════════
    df['chan_bullish_div'] = (
        (df['收盘价'] < df['close_l2']) &      # 价格创新低
        (df['MACD'] > df['macd_l2'])           # MACD动能改善
    ).astype(int)

    df['chan_bearish_div'] = (
        (df['收盘价'] > df['close_l2']) &      # 价格创新高
        (df['MACD'] < df['macd_l2'])           # MACD动能衰竭
    ).astype(int)

    # 背驰强度: |ΔMACD| / |Δprice%|, 值越大说明价格与动能的背离越显著
    denom = np.abs(df['收盘价'] / (df['close_l2'] + 1e-8) - 1) + 1e-8
    df['chan_div_strength'] = (
        np.abs(df['MACD'] - df['macd_l2']) / denom
    )
    df['chan_div_strength'] = df['chan_div_strength'].clip(0, 100)

    # ══════════════════════════════════════════════════════════════
    # 因子 3: 中枢位置 (Zhongshu, Section 11)
    #
    # 中枢是缠论核心概念，由连续三段次级别走势重叠区间构成。
    # 此处用三根K线的重叠区间作为日线级别的近似中枢：
    #   ZG (中枢上沿) = min(三根K线的高点)
    #   ZD (中枢下沿) = max(三根K线的低点)
    #   有效中枢条件: ZG > ZD（即三根K线确实有重叠）
    #
    # 中枢位置 = (close - ZD) / (ZG - ZD)
    #   < 0 → 中枢下方（买点区域，价格被低估）
    #   0~1 → 中枢内部（震荡区域）
    #   > 1 → 中枢上方（卖点区域，价格已充分反映）
    # ══════════════════════════════════════════════════════════════
    df['chan_zg'] = df[['最高价', 'high_l1', 'high_l2']].min(axis=1)  # 中枢上沿
    df['chan_zd'] = df[['最低价', 'low_l1', 'low_l2']].max(axis=1)   # 中枢下沿
    df['chan_zs_valid'] = (df['chan_zg'] > df['chan_zd']).astype(int)  # ZG>ZD 才算有效中枢

    df['chan_zs_position'] = (
        (df['收盘价'] - df['chan_zd']) / (df['chan_zg'] - df['chan_zd'] + 1e-8)
    )
    df['chan_zs_position'] = df['chan_zs_position'].clip(-1, 2)

    df['chan_below_zs'] = (df['chan_zs_position'] < 0).astype(int)     # 中枢下方
    df['chan_above_zs'] = (df['chan_zs_position'] > 1).astype(int)     # 中枢上方
    df['chan_inside_zs'] = (
        (df['chan_zs_position'] >= 0) & (df['chan_zs_position'] <= 1)  # 中枢内部
    ).astype(int)

    # ══════════════════════════════════════════════════════════════
    # 因子 4: 笔强度 (Stroke, Section 13)
    #
    # 笔是缠论的基本走势单元，由顶底分型交替连接而成。
    # 笔方向 = sign(涨跌幅_20)            — 使用20日涨跌作为笔方向代理
    # 笔强度 = |涨跌幅_20| / 涨跌幅std_20  — 单位风险下的收益（类夏普比率）
    # 笔加速度 = 近期动量变化              — 判断笔是否在加速/减速
    # ══════════════════════════════════════════════════════════════
    df['chan_stroke_dir'] = np.sign(df['涨跌幅_20'])          # +1=向上笔，-1=向下笔
    df['chan_stroke_strength'] = (
        np.abs(df['涨跌幅_20']) / (df['涨跌幅std_20'] + 1e-8)  # |收益|/波动 = 趋势质量
    )
    df['chan_stroke_strength'] = df['chan_stroke_strength'].clip(0, 10)

    # 笔加速度: 近期两段涨跌幅的差值，正值表示动量加速
    df['chan_stroke_accel'] = (
        (df['收盘价'] - df['close_l1']) / (df['close_l1'] + 1e-8)
        - (df['close_l1'] - df['close_l3']) / (df['close_l3'] + 1e-8)
    )

    # ══════════════════════════════════════════════════════════════
    # 因子 5: 综合买卖点信号 (Signal, Section 14)
    #
    # 缠论三类买点定义：
    #   一类买点 = 底分型 + 底背驰 → 最强的反转信号
    #             （趋势末端，结构+动能双重确认）
    #   二类买点 = 底分型 或 中枢下方 → 中等强度的入场信号
    #             （有个别信号但未形成双重确认）
    #   三类买点 = 中枢下方 + 笔向上 → 趋势中继信号
    #             （回调到支撑位后重新向上）
    #
    # 综合得分 = 正向信号加权 - 负向信号惩罚
    #   买点信号加分，卖点信号（顶分型+顶背驰+中枢上方）减分
    # ══════════════════════════════════════════════════════════════

    # 三类买点
    df['chan_buy_type1'] = (
        (df['chan_bottom_fractal'] == 1) & (df['chan_bullish_div'] == 1)
    ).astype(int)

    df['chan_buy_type2'] = (
        (df['chan_bottom_fractal'] == 1) | (df['chan_below_zs'] == 1)
    ).astype(int)

    df['chan_buy_type3'] = (
        (df['chan_below_zs'] == 1) & (df['chan_stroke_dir'] > 0)
    ).astype(int)

    # 综合得分
    # 权重设计: 一类买点 3分（最强信号）> 二类 2分 > 三类 1分
    #          顶分型 -2分、顶背驰 -2分 = 双重顶部确认，同等权重惩罚
    #          中枢上方 -1分 = 轻微负向（防止高买）
    df['chan_signal_score'] = (
        3.0 * df['chan_buy_type1'] +      # 一类买点: 结构+动能双确认，最高权重
        2.0 * df['chan_buy_type2'] +      # 二类买点: 有结构或位置优势
        1.0 * df['chan_buy_type3'] -      # 三类买点: 趋势中继
        2.0 * df['chan_top_fractal'] -    # 顶分型惩罚: 局部见顶信号
        2.0 * df['chan_bearish_div'] -    # 顶背驰惩罚: 动能衰竭信号
        1.0 * df['chan_above_zs']         # 中枢上方惩罚: 价格已到阻力区
    )

    # ── 清理中间列（只保留最终因子，去掉 shift 产生的临时列） ──
    drop_cols = ['high_l1', 'high_l2', 'low_l1', 'low_l2',
                 'close_l1', 'close_l2', 'close_l3', 'macd_l2',
                 'volume_l1', 'chan_zg', 'chan_zd']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    return df
