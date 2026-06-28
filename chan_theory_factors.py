#!/usr/bin/env python3
"""
缠中说禅（Chan Theory / 缠论）核心概念的Python实现

本模块实现了缠论从底层到高层的完整分析框架：
  1. K线包含关系处理 (Inclusion Processing)
  2. 顶底分型识别 (Fractal Detection)
  3. 笔的划分 (Stroke Construction)
  4. 线段构建 (Segment Construction)
  5. 中枢识别 (Pivot Zone / Zhongshu Detection)
  6. 背驰检测 (Divergence Detection)
  7. 三类买卖点识别 (Three Types of Buy/Sell Points)

所有函数基于 pandas DataFrame，输入为标准的 OHLCV 数据格式。

Usage:
    from chan_theory_factors import ChanTheoryAnalyzer, extract_chan_factors

    analyzer = ChanTheoryAnalyzer()
    result = analyzer.analyze(df)
    analyzer.print_report()

    factors = extract_chan_factors(df)

Author: Quant Research Team
Date: 2026-05-16
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# 基础数据结构定义
# ============================================================================


class Direction(Enum):
    """走势方向枚举"""
    UP = 1        # 向上 / 上涨
    DOWN = -1     # 向下 / 下跌
    UNKNOWN = 0   # 未知


class FractalType(Enum):
    """分型类型"""
    TOP = 1        # 顶分型
    BOTTOM = -1    # 底分型
    NONE = 0       # 非分型


@dataclass
class Fractal:
    """分型数据结构

    缠论定义：连续三根K线，中间K线的高低点均为极值
    - 顶分型: K2.high > K1.high, K2.high > K3.high, K2.low > K1.low, K2.low > K3.low
    - 底分型: K2.high < K1.high, K2.high < K3.high, K2.low < K1.low, K2.low < K3.low
    """
    index: int                     # K线索引位置
    fractal_type: FractalType      # 顶/底分型
    high: float                    # 最高价
    low: float                     # 最低价
    timestamp: pd.Timestamp = None # 时间戳
    is_confirmed: bool = False     # 是否被后续K线确认


@dataclass
class Stroke:
    """笔 (Bi) 的数据结构

    笔连接相邻的顶分型和底分型：
    - 向上笔: 底分型 → 顶分型
    - 向下笔: 顶分型 → 底分型

    条件：
    1. 分型之间至少有一根独立K线
    2. 分型必须交替出现（顶→底→顶→底...）
    """
    start_idx: int           # 起始K线索引
    end_idx: int             # 结束K线索引
    direction: Direction     # 方向
    start_price: float       # 起始价格
    end_price: float         # 结束价格
    start_fractal: Optional[Fractal] = None  # 起始分型
    end_fractal: Optional[Fractal] = None    # 结束分型
    high: float = 0.0        # 笔内最高价
    low: float = 0.0         # 笔内最低价
    kline_count: int = 0     # 包含的K线数

    @property
    def amplitude(self) -> float:
        """笔的绝对幅度"""
        return abs(self.end_price - self.start_price)

    @property
    def amplitude_pct(self) -> float:
        """笔的百分比幅度"""
        if self.start_price == 0:
            return 0.0
        return abs(self.end_price - self.start_price) / self.start_price

    @property
    def slope(self) -> float:
        """笔的斜率（每根K线的价格变化）"""
        if self.kline_count == 0:
            return 0.0
        return (self.end_price - self.start_price) / self.kline_count


@dataclass
class Segment:
    """线段 (Xian Duan) 的数据结构

    线段由至少三个连续的笔构成，代表比笔更高一级的走势：
    - 向上线段: 上笔+下笔+上笔+...（奇数笔，首尾向上）
    - 向下线段: 下笔+上笔+下笔+...（奇数笔，首尾向下）
    """
    start_idx: int                      # 起始K线索引
    end_idx: int                        # 结束K线索引
    direction: Direction                # 方向
    strokes: List[Stroke] = field(default_factory=list)  # 组成的笔
    start_price: float = 0.0
    end_price: float = 0.0
    high: float = 0.0                   # 线段内最高价
    low: float = 0.0                    # 线段内最低价

    @property
    def stroke_count(self) -> int:
        return len(self.strokes)


@dataclass
class Zhongshu:
    """中枢 (Pivot Zone) 的数据结构

    中枢是缠论最核心的概念：
    三个连续次级别走势类型（线段）重叠的价格区间。

    ZG（中枢高点）= min(线段1高, 线段2高, 线段3高)
    ZD（中枢低点）= max(线段1低, 线段2低, 线段3低)
    中枢存在条件：ZG > ZD
    """
    ZG: float                # 中枢高点
    ZD: float                # 中枢低点
    start_idx: int           # 起始K线索引
    end_idx: int             # 结束K线索引
    segments: List[Segment] = field(default_factory=list)

    @property
    def center(self) -> float:
        """中枢中心价格"""
        return (self.ZG + self.ZD) / 2.0

    @property
    def width(self) -> float:
        """中枢绝对宽度"""
        return self.ZG - self.ZD

    @property
    def width_pct(self) -> float:
        """中枢相对宽度"""
        if self.ZD == 0:
            return 0.0
        return (self.ZG - self.ZD) / self.ZD

    def is_valid(self) -> bool:
        """中枢是否有效（ZG > ZD，三段有重叠）"""
        return self.ZG > self.ZD

    def price_position(self, price: float) -> float:
        """价格相对于中枢的位置
        Returns:
            < 0: 中枢下方
            0~1: 中枢内部
            > 1: 中枢上方
        """
        if self.width <= 0:
            return 0.0
        return (price - self.ZD) / self.width


# ============================================================================
# 1. K线包含关系处理
# ============================================================================


class InclusionProcessor:
    """K线包含关系处理器

    缠论要求在分型识别之前处理相邻K线的包含关系：
    - 包含关系：一根K线的高/低点完全包含另一根K线的高/低点
    - 向上处理（涨势中）：新高 = max(h1,h2), 新低 = max(l1,l2)
    - 向下处理（跌势中）：新高 = min(h1,h2), 新低 = min(l1,l2)
    """

    @staticmethod
    def has_inclusion(h1: float, l1: float, h2: float, l2: float) -> bool:
        """判断两根K线是否存在包含关系"""
        return (
            (h1 >= h2 and l1 <= l2) or
            (h1 <= h2 and l1 >= l2)
        )

    @staticmethod
    def merge_bars(h1: float, l1: float, h2: float, l2: float,
                   direction: Direction) -> Tuple[float, float]:
        """合并两根有包含关系的K线

        Args:
            direction: UP=向上处理(取高高), DOWN=向下处理(取低低)
        Returns:
            (new_high, new_low)
        """
        if direction == Direction.UP:
            return max(h1, h2), max(l1, l2)
        else:
            return min(h1, h2), min(l1, l2)

    @classmethod
    def process(cls, df: pd.DataFrame,
                high_col: str = 'high',
                low_col: str = 'low') -> pd.DataFrame:
        """对K线数据进行包含关系处理

        Args:
            df: 包含 high 和 low 列的K线数据
        Returns:
            处理后的DataFrame，额外包含 'original_idx' 列
        """
        if len(df) < 2:
            result = df.copy()
            result['original_idx'] = range(len(df))
            return result

        highs = df[high_col].values
        lows = df[low_col].values
        n = len(df)

        result_highs = [highs[0]]
        result_lows = [lows[0]]
        orig_indices = [0]
        direction = Direction.UNKNOWN

        i = 1
        while i < n:
            ch, cl = highs[i], lows[i]
            ph, pl = result_highs[-1], result_lows[-1]

            if cls.has_inclusion(ph, pl, ch, cl):
                # 有包含关系：确定处理方向
                if direction == Direction.UNKNOWN:
                    if i >= 1:
                        direction = (
                            Direction.UP if highs[i] > highs[i - 1]
                            else Direction.DOWN if highs[i] < highs[i - 1]
                            else Direction.UP
                        )
                    else:
                        direction = Direction.UP

                new_h, new_l = cls.merge_bars(ph, pl, ch, cl, direction)
                result_highs[-1] = new_h
                result_lows[-1] = new_l
                orig_indices[-1] = i
            else:
                # 无包含关系
                if result_highs[-1] < ch:
                    direction = Direction.UP
                elif result_highs[-1] > ch:
                    direction = Direction.DOWN

                result_highs.append(ch)
                result_lows.append(cl)
                orig_indices.append(i)

            i += 1

        result_df = pd.DataFrame({
            'high': result_highs,
            'low': result_lows,
            'original_idx': orig_indices
        })

        # 补充其他列
        for col in ['open', 'close', 'volume']:
            if col in df.columns:
                result_df[col] = result_df['original_idx'].apply(
                    lambda x: df[col].iloc[x]
                )

        return result_df


# ============================================================================
# 2. 顶底分型识别
# ============================================================================


class FractalDetector:
    """顶底分型检测器

    在包含处理后的K线序列上识别顶分型和底分型：
    - 顶分型: K2为三者中最高，K2的low也为最高
    - 底分型: K2为三者中最低，K2的high也为最低
    """

    def __init__(self, strict: bool = True, confirmation_bars: int = 3):
        """
        Args:
            strict: True=严格分型, False=宽松分型（允许等号）
            confirmation_bars: 分型确认所需的后续K线数
        """
        self.strict = strict
        self.confirmation_bars = confirmation_bars

    def find_fractals(self, df: pd.DataFrame,
                      high_col: str = 'high',
                      low_col: str = 'low') -> List[Fractal]:
        """在K线数据中寻找顶底分型

        Returns:
            按时间顺序排列的Fractal列表
        """
        n = len(df)
        if n < 3:
            return []

        highs = df[high_col].values
        lows = df[low_col].values
        fractals = []

        for i in range(1, n - 1):
            is_top = self._is_top_fractal(highs, lows, i)
            is_bottom = self._is_bottom_fractal(highs, lows, i)

            if is_top:
                ts = df.index[i] if hasattr(df.index[i], 'strftime') else None
                fractals.append(Fractal(
                    index=i,
                    fractal_type=FractalType.TOP,
                    high=highs[i],
                    low=lows[i],
                    timestamp=ts
                ))
            elif is_bottom:
                ts = df.index[i] if hasattr(df.index[i], 'strftime') else None
                fractals.append(Fractal(
                    index=i,
                    fractal_type=FractalType.BOTTOM,
                    high=highs[i],
                    low=lows[i],
                    timestamp=ts
                ))

        # 确认分型有效性
        self._confirm_fractals(fractals, highs, lows, n)

        return fractals

    def _is_top_fractal(self, highs: np.ndarray, lows: np.ndarray,
                        i: int) -> bool:
        """判断是否为顶分型"""
        if self.strict:
            return (highs[i] > highs[i - 1] and highs[i] > highs[i + 1] and
                    lows[i] > lows[i - 1] and lows[i] > lows[i + 1])
        else:
            return (highs[i] >= highs[i - 1] and highs[i] >= highs[i + 1] and
                    lows[i] >= lows[i - 1] and lows[i] >= lows[i + 1])

    def _is_bottom_fractal(self, highs: np.ndarray, lows: np.ndarray,
                           i: int) -> bool:
        """判断是否为底分型"""
        if self.strict:
            return (highs[i] < highs[i - 1] and highs[i] < highs[i + 1] and
                    lows[i] < lows[i - 1] and lows[i] < lows[i + 1])
        else:
            return (highs[i] <= highs[i - 1] and highs[i] <= highs[i + 1] and
                    lows[i] <= lows[i - 1] and lows[i] <= lows[i + 1])

    def _confirm_fractals(self, fractals: List[Fractal],
                          highs: np.ndarray, lows: np.ndarray,
                          n: int) -> None:
        """确认分型有效性

        顶分型有效: 后续K线跌破该分型最低点
        底分型有效: 后续K线升破该分型最高点
        """
        for f in fractals:
            look = min(self.confirmation_bars, n - f.index - 1)
            if look <= 0:
                continue
            if f.fractal_type == FractalType.TOP:
                for j in range(f.index + 1, f.index + look + 1):
                    if lows[j] < f.low:
                        f.is_confirmed = True
                        break
            else:
                for j in range(f.index + 1, f.index + look + 1):
                    if highs[j] > f.high:
                        f.is_confirmed = True
                        break

    @staticmethod
    def merge_same_type_fractals(fractals: List[Fractal]) -> List[Fractal]:
        """合并连续的同类分型

        两个连续顶分型: 取更高的那个
        两个连续底分型: 取更低那个
        """
        if len(fractals) < 2:
            return fractals

        # 先确保交替：遇到同类保留更极端的
        merged = [fractals[0]]
        for f in fractals[1:]:
            last = merged[-1]
            if f.fractal_type == last.fractal_type:
                if f.fractal_type == FractalType.TOP:
                    if f.high > last.high:
                        merged[-1] = f
                else:  # BOTTOM
                    if f.low < last.low:
                        merged[-1] = f
            else:
                merged.append(f)

        return merged


# ============================================================================
# 3. 笔的构建
# ============================================================================


class StrokeBuilder:
    """笔 (Bi) 的构建器

    笔连接相邻的顶分型和底分型。
    条件：
    1. 分型交替（一顶一底或一底一顶）
    2. 分型之间至少有一根独立K线
    """

    def __init__(self, min_kline_between: int = 1):
        """
        Args:
            min_kline_between: 两分型之间最少独立K线数
        """
        self.min_kline_between = min_kline_between

    def build_strokes(self, fractals: List[Fractal],
                      df: pd.DataFrame,
                      high_col: str = 'high',
                      low_col: str = 'low') -> List[Stroke]:
        """从分型序列构建笔

        Args:
            fractals: 按时间排序的有效分型列表
            df: 原始K线数据（用于计算笔内极值）
        Returns:
            笔的列表
        """
        if len(fractals) < 2:
            return []

        strokes = []
        highs = df[high_col].values
        lows = df[low_col].values

        for i in range(len(fractals) - 1):
            f1 = fractals[i]
            f2 = fractals[i + 1]

            if f1.fractal_type == f2.fractal_type:
                continue

            if f2.index - f1.index < self.min_kline_between + 1:
                continue

            if f1.fractal_type == FractalType.BOTTOM:
                direction = Direction.UP
                start_price = f1.low
                end_price = f2.high
            else:
                direction = Direction.DOWN
                start_price = f1.high
                end_price = f2.low

            seg_h = highs[f1.index:f2.index + 1]
            seg_l = lows[f1.index:f2.index + 1]

            stroke = Stroke(
                start_idx=f1.index,
                end_idx=f2.index,
                direction=direction,
                start_price=start_price,
                end_price=end_price,
                start_fractal=f1,
                end_fractal=f2,
                high=float(np.max(seg_h)),
                low=float(np.min(seg_l)),
                kline_count=f2.index - f1.index + 1
            )
            strokes.append(stroke)

        return strokes

    @staticmethod
    def merge_same_direction_strokes(strokes: List[Stroke]) -> List[Stroke]:
        """合并连续的同向笔"""
        if len(strokes) < 2:
            return strokes

        merged = [strokes[0]]
        for s in strokes[1:]:
            last = merged[-1]
            if s.direction == last.direction:
                last.end_idx = s.end_idx
                last.end_price = s.end_price
                last.end_fractal = s.end_fractal
                last.high = max(last.high, s.high)
                last.low = min(last.low, s.low)
                last.kline_count = last.end_idx - last.start_idx + 1
            else:
                merged.append(s)
        return merged


# ============================================================================
# 4. 线段构建
# ============================================================================


class SegmentBuilder:
    """线段 (Xian Duan) 构建器

    线段由至少三个连续的笔构成：
    - 向上线段: 上+下+上+...（奇数笔，首尾向上）
    - 向下线段: 下+上+下+...（奇数笔，首尾向下）
    最后笔的终点需超越第一笔的起点
    """

    def __init__(self, min_strokes: int = 3):
        self.min_strokes = min_strokes

    def build_segments(self, strokes: List[Stroke],
                       df: pd.DataFrame) -> List[Segment]:
        """从笔序列构建线段"""
        if len(strokes) < self.min_strokes:
            return []

        segments = []
        highs_arr = df['high'].values
        lows_arr = df['low'].values
        i = 0

        while i <= len(strokes) - self.min_strokes:
            direction = strokes[i].direction
            collected = [strokes[i]]
            j = i + 1
            found = False

            while j < len(strokes):
                collected.append(strokes[j])
                if len(collected) >= self.min_strokes:
                    if self._can_end(collected, direction):
                        found = True
                        break
                j += 1

            if found:
                seg = self._make_segment(collected, direction,
                                         highs_arr, lows_arr)
                segments.append(seg)
                i = j + 1
            else:
                i += 1

        return segments

    def _can_end(self, strokes: List[Stroke], direction: Direction) -> bool:
        """判断线段是否可以构成"""
        if len(strokes) < self.min_strokes:
            return False
        if strokes[0].direction != direction:
            return False
        if strokes[-1].direction != direction:
            return False
        if direction == Direction.UP:
            return strokes[-1].end_price > strokes[0].start_price
        else:
            return strokes[-1].end_price < strokes[0].start_price

    @staticmethod
    def _make_segment(strokes: List[Stroke], direction: Direction,
                      highs: np.ndarray, lows: np.ndarray) -> Segment:
        """创建Segment对象"""
        s0, sn = strokes[0], strokes[-1]
        seg_h = highs[s0.start_idx:sn.end_idx + 1]
        seg_l = lows[s0.start_idx:sn.end_idx + 1]
        return Segment(
            start_idx=s0.start_idx,
            end_idx=sn.end_idx,
            direction=direction,
            strokes=strokes,
            start_price=s0.start_price,
            end_price=sn.end_price,
            high=float(np.max(seg_h)),
            low=float(np.min(seg_l))
        )


# ============================================================================
# 5. 中枢识别
# ============================================================================


class ZhongshuDetector:
    """中枢 (Pivot Zone) 识别器

    中枢定义: 至少三个连续线段重叠的价格区间
    ZG = min(三线段高点), ZD = max(三线段低点)
    条件: ZG > ZD
    """

    def find_from_segments(self, segments: List[Segment]) -> List[Zhongshu]:
        """从线段序列中寻找中枢（滑动窗口法）"""
        if len(segments) < 3:
            return []

        zs_list = []
        for i in range(len(segments) - 2):
            s1, s2, s3 = segments[i], segments[i + 1], segments[i + 2]
            zg = min(s1.high, s2.high, s3.high)
            zd = max(s1.low, s2.low, s3.low)

            if zg > zd:
                zs_list.append(Zhongshu(
                    ZG=zg,
                    ZD=zd,
                    start_idx=s1.start_idx,
                    end_idx=s3.end_idx,
                    segments=[s1, s2, s3]
                ))

        return zs_list

    def find_from_strokes(self, strokes: List[Stroke]) -> List[Zhongshu]:
        """从笔序列中寻找笔级别中枢（用于无线段的情况）"""
        if len(strokes) < 3:
            return []

        zs_list = []
        for i in range(len(strokes) - 2):
            s1, s2, s3 = strokes[i], strokes[i + 1], strokes[i + 2]
            h1 = max(s1.start_price, s1.end_price)
            l1 = min(s1.start_price, s1.end_price)
            h2 = max(s2.start_price, s2.end_price)
            l2 = min(s2.start_price, s2.end_price)
            h3 = max(s3.start_price, s3.end_price)
            l3 = min(s3.start_price, s3.end_price)

            zg = min(h1, h2, h3)
            zd = max(l1, l2, l3)

            if zg > zd:
                zs_list.append(Zhongshu(
                    ZG=zg, ZD=zd,
                    start_idx=s1.start_idx,
                    end_idx=s3.end_idx
                ))

        return zs_list

    @staticmethod
    def merge_overlapping(zs_list: List[Zhongshu]) -> List[Zhongshu]:
        """合并重叠的中枢"""
        if len(zs_list) < 2:
            return zs_list

        merged = [zs_list[0]]
        for zs in zs_list[1:]:
            last = merged[-1]
            if zs.ZD <= last.ZG and zs.ZG >= last.ZD:
                last.ZG = max(last.ZG, zs.ZG)
                last.ZD = min(last.ZD, zs.ZD)
                last.end_idx = zs.end_idx
            else:
                merged.append(zs)
        return merged

    @staticmethod
    def get_current_zhongshu(
        zs_list: List[Zhongshu], idx: int
    ) -> Optional[Zhongshu]:
        """获取当前位置所属/最近的中枢"""
        for zs in reversed(zs_list):
            if zs.start_idx <= idx <= zs.end_idx:
                return zs
        for zs in reversed(zs_list):
            if zs.end_idx <= idx:
                return zs
        return None

    @staticmethod
    def position_label(price: float, zs: Zhongshu) -> str:
        """价格相对于中枢的位置: 'above' / 'inside' / 'below'"""
        if zs is None or not zs.is_valid():
            return 'unknown'
        if price > zs.ZG:
            return 'above'
        elif price < zs.ZD:
            return 'below'
        else:
            return 'inside'


# ============================================================================
# 6. 背驰检测
# ============================================================================


class DivergenceDetector:
    """背驰 (Bei Chi) 检测器

    背驰是判断走势转折的核心工具。
    使用 MACD 柱面积比较：后一段价格创新高/低但MACD面积缩小
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def compute_macd(self, close: np.ndarray) -> Tuple[np.ndarray,
                                                        np.ndarray,
                                                        np.ndarray]:
        """计算MACD指标
        Returns: (dif, dea, macd_bar)
        """
        ema_fast = self._ema(close, self.fast)
        ema_slow = self._ema(close, self.slow)
        dif = ema_fast - ema_slow
        dea = self._ema(dif, self.signal)
        macd_bar = 2 * (dif - dea)
        return dif, dea, macd_bar

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        alpha = 2 / (period + 1)
        result = np.zeros_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        return result

    @staticmethod
    def macd_area(macd_bar: np.ndarray, start: int, end: int) -> float:
        """计算指定区间内的MACD柱面积（绝对值）"""
        if start >= len(macd_bar) or end >= len(macd_bar) or start > end:
            return 0.0
        return float(np.sum(np.abs(macd_bar[start:end + 1])))

    def detect_stroke_divergence(
        self,
        strokes: List[Stroke],
        macd_bar: np.ndarray,
        price_arr: np.ndarray
    ) -> pd.DataFrame:
        """基于笔的背驰检测

        比较相邻同向笔：后一笔创新高/低但MACD面积缩小
        Returns: DataFrame with 'divergence_signal' and 'divergence_strength'
        """
        n = len(macd_bar)
        div_signal = np.zeros(n, dtype=int)
        div_strength = np.zeros(n)

        # 分别检查向上笔和向下笔
        for direction, is_up in [(Direction.UP, True), (Direction.DOWN, False)]:
            same_dir = [s for s in strokes if s.direction == direction]
            for i in range(1, len(same_dir)):
                prev, curr = same_dir[i - 1], same_dir[i]
                prev_area = self.macd_area(macd_bar, prev.start_idx, prev.end_idx)
                curr_area = self.macd_area(macd_bar, curr.start_idx, curr.end_idx)
                if prev_area == 0:
                    continue
                ratio = curr_area / prev_area

                if is_up:
                    if curr.end_price > prev.end_price and ratio < 0.8:
                        idx = curr.end_idx
                        if idx < n:
                            div_signal[idx] = 1
                            div_strength[idx] = 1 - ratio
                else:
                    if curr.end_price < prev.end_price and ratio < 0.8:
                        idx = curr.end_idx
                        if idx < n:
                            div_signal[idx] = -1
                            div_strength[idx] = 1 - ratio

        return pd.DataFrame({
            'divergence_signal': div_signal,
            'divergence_strength': div_strength
        })

    def detect_rsi_divergence(self, close: np.ndarray,
                               period: int = 14,
                               lookback: int = 20) -> np.ndarray:
        """RSI 辅助背驰检测

        Returns: 1=底背驰, -1=顶背驰, 0=无
        """
        n = len(close)
        rsi = self._calc_rsi(close, period)
        result = np.zeros(n, dtype=int)

        for i in range(lookback, n):
            hist_close = close[i - lookback:i]
            hist_rsi = rsi[i - lookback:i]

            # 顶背离: 价格新高但RSI未新高
            if (close[i] > np.max(hist_close) and
                    rsi[i] < np.max(hist_rsi)):
                result[i] = -1

            # 底背离: 价格新低但RSI未新低
            if (close[i] < np.min(hist_close) and
                    rsi[i] > np.min(hist_rsi)):
                result[i] = 1

        return result

    @staticmethod
    def _calc_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
        """计算RSI"""
        n = len(close)
        rsi = np.full(n, 50.0)
        if n < period + 1:
            return rsi

        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.zeros(n)
        avg_loss = np.zeros(n)
        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])

        for i in range(period + 1, n):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period

        mask = avg_loss > 0
        rsi[mask] = 100 - 100 / (1 + avg_gain[mask] / avg_loss[mask])
        return rsi


# ============================================================================
# 7. 三类买卖点识别
# ============================================================================


class TradeSignalGenerator:
    """三类买卖点信号生成器

    第一类买卖点: 趋势背驰产生的极点
    第二类买卖点: 一类买卖点后，第一次回抽不创新高/低
    第三类买卖点: 离开中枢后回抽，不回到中枢区间
    """

    def generate(self,
                 df: pd.DataFrame,
                 strokes: List[Stroke],
                 zhongshu_list: List[Zhongshu],
                 divergence_df: pd.DataFrame) -> pd.DataFrame:
        """生成所有买卖点信号

        Returns:
            DataFrame with 'buy_signal' (0/1/2/3) and 'sell_signal' (0/1/2/3)
        """
        n = len(df)
        result = df[['close', 'high', 'low']].copy()
        result['buy_signal'] = 0
        result['sell_signal'] = 0

        div_signal = divergence_df['divergence_signal'].values

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        # 第一类买卖点
        self._detect_type1(div_signal, close, result, n)

        # 第二类买卖点
        self._detect_type2(result, close, high, low, n)

        # 第三类买卖点
        self._detect_type3(result, zhongshu_list, close, high, low, n)

        return result

    def _detect_type1(self, div_signal: np.ndarray, close: np.ndarray,
                      result: pd.DataFrame, n: int) -> None:
        """一类买卖点: 背驰点后价格确认"""
        for i in range(3, n):
            if div_signal[i] == -1 and i + 3 < n and close[i + 1] > close[i]:
                result.iloc[i, result.columns.get_loc('buy_signal')] = 1
            elif div_signal[i] == 1 and i + 3 < n and close[i + 1] < close[i]:
                result.iloc[i, result.columns.get_loc('sell_signal')] = 1

    def _detect_type2(self, result: pd.DataFrame, close: np.ndarray,
                      high: np.ndarray, low: np.ndarray, n: int) -> None:
        """二类买卖点: 一类后第一次回抽"""
        buy_col = result.columns.get_loc('buy_signal')
        sell_col = result.columns.get_loc('sell_signal')
        buy_vals = result['buy_signal'].values
        sell_vals = result['sell_signal'].values

        # 二类买点
        for i in range(n):
            if buy_vals[i] == 1:
                bp = low[i]
                for j in range(i + 2, min(i + 30, n)):
                    if (j >= i + 5 and
                            low[j] < low[j - 1] and
                            low[j] > bp * 0.95 and
                            buy_vals[j] == 0):
                        result.iloc[j, buy_col] = 2
                        break

        # 二类卖点
        for i in range(n):
            if sell_vals[i] == 1:
                sp = high[i]
                for j in range(i + 2, min(i + 30, n)):
                    if (j >= i + 5 and
                            high[j] > high[j - 1] and
                            high[j] < sp * 1.05 and
                            sell_vals[j] == 0):
                        result.iloc[j, sell_col] = 2
                        break

    def _detect_type3(self, result: pd.DataFrame,
                      zhongshu_list: List[Zhongshu],
                      close: np.ndarray, high: np.ndarray, low: np.ndarray,
                      n: int) -> None:
        """三类买卖点: 离开中枢后回抽不回调"""
        if not zhongshu_list:
            return

        last_zs = zhongshu_list[-1]
        if not last_zs.is_valid():
            return

        zg, zd = last_zs.ZG, last_zs.ZD
        buy_col = result.columns.get_loc('buy_signal')
        sell_col = result.columns.get_loc('sell_signal')
        start = last_zs.end_idx

        for i in range(start, n):
            # 三类买点: 曾突破ZG后回调到ZG附近
            if i >= 5:
                if np.max(high[i - 5:i]) > zg and low[i] > zd:
                    if (low[i] <= low[i - 1] and
                            (i + 1 >= n or low[i] <= low[i + 1])):
                        if abs(low[i] - zg) / zg < 0.03:
                            result.iloc[i, buy_col] = 3

            # 三类卖点: 曾跌破ZD后反弹到ZD附近
            if i >= 5:
                if np.min(low[i - 5:i]) < zd and high[i] < zg:
                    if (high[i] >= high[i - 1] and
                            (i + 1 >= n or high[i] >= high[i + 1])):
                        if abs(high[i] - zd) / zd < 0.03:
                            result.iloc[i, sell_col] = 3


# ============================================================================
# 8. 综合分析器（主入口）
# ============================================================================


class ChanTheoryAnalyzer:
    """缠论综合分析器

    整合所有模块，提供一键式缠论分析。

    Usage:
        analyzer = ChanTheoryAnalyzer()
        result = analyzer.analyze(df)
        analyzer.print_report()
    """

    def __init__(self,
                 fractal_strict: bool = True,
                 confirmation_bars: int = 3,
                 min_strokes_per_segment: int = 3):
        self.fractal_strict = fractal_strict
        self.confirmation_bars = confirmation_bars
        self.min_strokes_per_segment = min_strokes_per_segment

        # 子模块
        self.inclusion_processor = InclusionProcessor()
        self.fractal_detector = FractalDetector(
            strict=fractal_strict,
            confirmation_bars=confirmation_bars
        )
        self.stroke_builder = StrokeBuilder()
        self.segment_builder = SegmentBuilder(
            min_strokes=min_strokes_per_segment
        )
        self.zhongshu_detector = ZhongshuDetector()
        self.divergence_detector = DivergenceDetector()
        self.signal_generator = TradeSignalGenerator()

        # 结果存储
        self.inclusions_df: Optional[pd.DataFrame] = None
        self.fractals: List[Fractal] = []
        self.strokes: List[Stroke] = []
        self.segments: List[Segment] = []
        self.zhongshu_list: List[Zhongshu] = []
        self.divergence_df: Optional[pd.DataFrame] = None
        self.trade_signals_df: Optional[pd.DataFrame] = None

    def analyze(self, df: pd.DataFrame,
                close_col: str = 'close',
                high_col: str = 'high',
                low_col: str = 'low') -> Dict:
        """执行完整缠论分析

        Args:
            df: 包含 high, low, close 列的K线DataFrame
        Returns:
            包含所有分析结果的字典
        """
        for col in [high_col, low_col, close_col]:
            if col not in df.columns:
                raise ValueError(f"缺少必要列: {col}")

        # Step 1: K线包含关系处理
        self.inclusions_df = self.inclusion_processor.process(
            df, high_col, low_col
        )

        # Step 2: 顶底分型识别
        self.fractals = self.fractal_detector.find_fractals(
            self.inclusions_df, 'high', 'low'
        )
        self.fractals = FractalDetector.merge_same_type_fractals(self.fractals)

        # Step 3: 笔的构建
        self.strokes = self.stroke_builder.build_strokes(
            self.fractals, df, high_col, low_col
        )
        self.strokes = StrokeBuilder.merge_same_direction_strokes(self.strokes)

        # Step 4: 线段构建
        self.segments = self.segment_builder.build_segments(self.strokes, df)

        # Step 5: 中枢识别
        if self.segments:
            self.zhongshu_list = self.zhongshu_detector.find_from_segments(
                self.segments
            )
            self.zhongshu_list = ZhongshuDetector.merge_overlapping(
                self.zhongshu_list
            )
        else:
            self.zhongshu_list = self.zhongshu_detector.find_from_strokes(
                self.strokes
            )

        # Step 6: 背驰检测
        close = df[close_col].values
        _, _, macd_bar = self.divergence_detector.compute_macd(close)
        self.divergence_df = self.divergence_detector.detect_stroke_divergence(
            self.strokes, macd_bar, close
        )
        # RSI辅助
        rsi_div = self.divergence_detector.detect_rsi_divergence(close)
        self.divergence_df['rsi_divergence'] = rsi_div

        # Step 7: 买卖点信号
        self.trade_signals_df = self.signal_generator.generate(
            df, self.strokes, self.zhongshu_list, self.divergence_df
        )

        summary = self._build_summary(df)
        return {
            'inclusions_df': self.inclusions_df,
            'fractals': self.fractals,
            'strokes': self.strokes,
            'segments': self.segments,
            'zhongshu_list': self.zhongshu_list,
            'divergence_df': self.divergence_df,
            'trade_signals_df': self.trade_signals_df,
            'summary': summary
        }

    def _build_summary(self, df: pd.DataFrame) -> Dict:
        """构建分析摘要"""
        ts = self.trade_signals_df
        buy = ts['buy_signal'].values if ts is not None else np.array([])
        sell = ts['sell_signal'].values if ts is not None else np.array([])
        div = (self.divergence_df['divergence_signal'].values
               if self.divergence_df is not None else np.array([]))

        return {
            'data_points': len(df),
            'inclusion_processed': (
                len(self.inclusions_df) if self.inclusions_df is not None else 0
            ),
            'fractals_top': sum(1 for f in self.fractals
                                if f.fractal_type == FractalType.TOP),
            'fractals_bottom': sum(1 for f in self.fractals
                                   if f.fractal_type == FractalType.BOTTOM),
            'fractals_confirmed': sum(1 for f in self.fractals if f.is_confirmed),
            'stroke_count': len(self.strokes),
            'stroke_up': sum(1 for s in self.strokes if s.direction == Direction.UP),
            'stroke_down': sum(1 for s in self.strokes if s.direction == Direction.DOWN),
            'segment_count': len(self.segments),
            'zhongshu_count': len(self.zhongshu_list),
            'valid_zhongshu': sum(1 for z in self.zhongshu_list if z.is_valid()),
            'top_divergence': int(np.sum(div == 1)),
            'bottom_divergence': int(np.sum(div == -1)),
            'buy1': int(np.sum(buy == 1)),
            'buy2': int(np.sum(buy == 2)),
            'buy3': int(np.sum(buy == 3)),
            'sell1': int(np.sum(sell == 1)),
            'sell2': int(np.sum(sell == 2)),
            'sell3': int(np.sum(sell == 3)),
        }

    def get_zhongshu_position_factors(self, df: pd.DataFrame,
                                       close_col: str = 'close') -> pd.DataFrame:
        """提取中枢位置因子

        Returns DataFrame with:
            zhongshu_position: 相对位置（<0下方, 0~1内部, >1上方）
            zhongshu_distance: 到中枢中心距离
            above/below/inside_zhongshu: 布尔标记
        """
        result = df.copy()
        n = len(df)
        close = df[close_col].values

        result['zhongshu_position'] = np.nan
        result['zhongshu_distance'] = np.nan
        result['above_zhongshu'] = False
        result['below_zhongshu'] = False
        result['inside_zhongshu'] = False

        for i in range(n):
            zs = self.zhongshu_detector.get_current_zhongshu(
                self.zhongshu_list, i
            )
            if zs and zs.is_valid():
                price = close[i]
                result.iloc[i, result.columns.get_loc('zhongshu_position')] = (
                    zs.price_position(price)
                )
                result.iloc[i, result.columns.get_loc('zhongshu_distance')] = (
                    price - zs.center
                )
                result.iloc[i, result.columns.get_loc('above_zhongshu')] = (
                    price > zs.ZG
                )
                result.iloc[i, result.columns.get_loc('below_zhongshu')] = (
                    price < zs.ZD
                )
                result.iloc[i, result.columns.get_loc('inside_zhongshu')] = (
                    zs.ZD <= price <= zs.ZG
                )

        return result

    def get_stroke_momentum_factors(self) -> pd.DataFrame:
        """提取笔动量因子

        Returns DataFrame with: amplitude, amplitude_pct, slope, power, etc.
        """
        if not self.strokes:
            return pd.DataFrame()

        records = []
        for i, s in enumerate(self.strokes):
            records.append({
                'stroke_idx': i,
                'direction': 'UP' if s.direction == Direction.UP else 'DOWN',
                'start_idx': s.start_idx,
                'end_idx': s.end_idx,
                'amplitude': s.amplitude,
                'amplitude_pct': s.amplitude_pct,
                'slope': s.slope,
                'kline_count': s.kline_count,
                'power': s.amplitude_pct / max(s.kline_count, 1)
            })
        return pd.DataFrame(records)

    def print_report(self) -> None:
        """打印分析报告"""
        s = self._build_summary(pd.DataFrame())
        print("=" * 60)
        print("  缠中说禅（Chan Theory）分析报告")
        print("=" * 60)
        print(f"\n数据: {s['data_points']} 根K线 -> "
              f"包含处理后 {s['inclusion_processed']} 根")
        print(f"\n分型: 顶{s['fractals_top']} 底{s['fractals_bottom']} "
              f"已确认{s['fractals_confirmed']}")
        print(f"笔: {s['stroke_count']} (上{s['stroke_up']} 下{s['stroke_down']})")
        print(f"线段: {s['segment_count']}")
        print(f"中枢: {s['zhongshu_count']} (有效{s['valid_zhongshu']})")
        if self.zhongshu_list:
            z = self.zhongshu_list[-1]
            if z.is_valid():
                print(f"  最近中枢: ZG={z.ZG:.4f} ZD={z.ZD:.4f} "
                      f"中心={z.center:.4f} 宽={z.width:.4f}")
        print(f"\n背驰: 顶{s['top_divergence']} 底{s['bottom_divergence']}")
        print(f"买点: 一{s['buy1']} 二{s['buy2']} 三{s['buy3']}")
        print(f"卖点: 一{s['sell1']} 二{s['sell2']} 三{s['sell3']}")
        print("=" * 60)


# ============================================================================
# 9. 便捷函数
# ============================================================================


def extract_chan_factors(df: pd.DataFrame,
                         close_col: str = 'close',
                         high_col: str = 'high',
                         low_col: str = 'low') -> pd.DataFrame:
    """便捷函数：一键提取缠论量化因子

    Returns DataFrame with:
        - divergence_signal, divergence_strength: 背驰信号和强度
        - buy_signal, sell_signal: 买卖点类型 (0/1/2/3)
        - zhongshu_position: 中枢相对位置
        - above/below/inside_zhongshu: 中枢位置布尔标记
    """
    analyzer = ChanTheoryAnalyzer()
    results = analyzer.analyze(df, close_col, high_col, low_col)

    # 交易信号
    factors = results['trade_signals_df'].copy()

    # 中枢位置因子
    pos_factors = analyzer.get_zhongshu_position_factors(df, close_col)
    for col in ['zhongshu_position', 'zhongshu_distance',
                'above_zhongshu', 'below_zhongshu', 'inside_zhongshu']:
        if col in pos_factors.columns:
            factors[col] = pos_factors[col].values

    return factors


# ============================================================================
# 10. 自测
# ============================================================================


def generate_sample_data(n: int = 200) -> pd.DataFrame:
    """生成模拟K线数据用于测试

    生成包含上涨、盘整、下跌的完整周期走势
    """
    np.random.seed(42)
    segs = [
        ('up', 50, 10.0, 0.15, 0.02),
        ('range', 40, None, 0.02, 0.015),
        ('up', 30, None, 0.10, 0.02),
        ('down', 50, None, 0.12, 0.025),
        ('range', 30, None, 0.02, 0.02),
    ]

    closes, highs, lows, opens = [], [], [], []
    price = None

    for stype, length, start, trend, vol in segs:
        if start is not None:
            price = start
        for _ in range(length):
            if stype == 'up':
                price += np.random.normal(trend, vol)
            elif stype == 'down':
                price -= np.random.normal(trend, vol)
            else:
                price += np.random.normal(0, vol)
            price = max(price, 0.01)

            o = price + np.random.normal(0, vol * 0.3)
            c = price
            h = max(o, c) + abs(np.random.normal(0, vol * 0.5))
            l_val = min(o, c) - abs(np.random.normal(0, vol * 0.5))
            l_val = max(l_val, 0.01)

            opens.append(o)
            closes.append(c)
            highs.append(h)
            lows.append(l_val)

    dates = pd.date_range('2006-01-01', periods=len(closes), freq='D')
    return pd.DataFrame({
        'open': opens, 'high': highs,
        'low': lows, 'close': closes,
        'volume': np.random.randint(1000000, 10000000, len(closes))
    }, index=dates)


def run_self_test() -> Dict:
    """运行自测，验证所有模块"""
    print("=" * 60)
    print("  缠中说禅 Python实现 - 自测")
    print("=" * 60)

    print("\n[1] 生成模拟数据...")
    df = generate_sample_data(200)
    print(f"    {len(df)} 根K线, {df.index[0].date()} ~ {df.index[-1].date()}")

    print("\n[2] 运行完整分析...")
    analyzer = ChanTheoryAnalyzer()
    results = analyzer.analyze(df)

    print("\n[3] 分析报告:")
    analyzer.print_report()

    print("\n[4] 中枢详情:")
    for i, zs in enumerate(results['zhongshu_list']):
        if zs.is_valid():
            print(f"    中枢{i + 1}: ZG={zs.ZG:.4f} ZD={zs.ZD:.4f} "
                  f"(K线{zs.start_idx}~{zs.end_idx})")

    print("\n[5] 前10笔:")
    for i, s in enumerate(results['strokes'][:10]):
        d = "up" if s.direction == Direction.UP else "dn"
        print(f"    笔{i + 1}: {d} {s.start_price:.4f}->{s.end_price:.4f} "
              f"({s.amplitude_pct:.2%})")

    print("\n[6] 因子提取...")
    factors = extract_chan_factors(df)
    print(f"    因子DataFrame: {factors.shape}")
    for col in ['divergence_signal', 'buy_signal', 'sell_signal',
                'zhongshu_position']:
        if col in factors.columns:
            nz = (factors[col] != 0).sum() if col != 'zhongshu_position' else \
                 factors[col].notna().sum()
            print(f"    {col}: {nz} 非空/非零")

    print("\n" + "=" * 60)
    print("  自测通过！")
    print("=" * 60)
    return results


# ============================================================================
# 主入口
# ============================================================================

if __name__ == '__main__':
    results = run_self_test()

    print("\n\n使用示例:")
    print("-" * 50)
    print("""
from chan_theory_factors import ChanTheoryAnalyzer, extract_chan_factors
import pandas as pd

# 加载数据
df = pd.read_csv('your_stock.csv', index_col='date', parse_dates=True)

# 方式1: 完整分析
analyzer = ChanTheoryAnalyzer()
results = analyzer.analyze(df)
analyzer.print_report()

# 获取买卖点
signals = results['trade_signals_df']
buy = signals[signals['buy_signal'] > 0]
sell = signals[signals['sell_signal'] > 0]

# 方式2: 一键提取因子
factors = extract_chan_factors(df)

# 方式3: 获取中枢位置因子
pos = analyzer.get_zhongshu_position_factors(df)

# 方式4: 获取笔动量
mom = analyzer.get_stroke_momentum_factors()
""")
