"""
Stock Selection Environment — 强化学习选股环境

基于 OpenAI Gym 接口的自定义环境，模拟月度选股-调仓-评估流程。

状态空间 (State Space):
    - 量价特征 (30维): OHLCV, bias_5/10/20, 振幅_5/10/20, 涨跌幅std, 成交额std
    - 技术指标 (7维): K, D, J, DIF, DEA, MACD, 换手率
    - 基本面特征 (4维): 市盈率倒数, 市净率倒数, ROE(TTM)代理, 市值
    - 市场特征 (5维): 市场状态, 指数收益, 行业收益, 资金流, 波动率
    - 持仓特征 (K维): 当前持仓权重
    → 拼接后通过 LLM Encoder 映射为 256 维统一状态

动作空间 (Action Space):
    离散: 从股票池中选择 top-K 只等权持有
    或者连续: 输出每只股票的持仓权重

奖励函数 (Reward Function):
    r_t = portfolio_return_t - 0.5 * turnover_penalty_t - 0.1 * drawdown_penalty_t

约束:
    - 股票池: 排除 ST/科创/次新股
    - 调仓频率: 每月末
    - 交易成本: 0.12% 单边
    - 最大持仓: K 只
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import deque
import warnings

warnings.filterwarnings("ignore")


class StockSelectionEnv:
    """
    月度选股强化学习环境

    Parameters
    ----------
    df : pd.DataFrame
        月度股票数据 (需包含 stock_data.csv 全部列)
    feature_cols : List[str]
        特征列名列表
    lookback : int
        回看窗口 (月数)，默认 12
    top_k : int
        每期选股数量，默认 6
    max_position : float
        单票最大仓位，默认 0.3
    transaction_cost : float
        单边交易成本，默认 0.0012
    risk_free_rate : float
        无风险利率 (年化)，默认 0.03
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        lookback: int = 12,
        top_k: int = 6,
        max_position: float = 0.3,
        transaction_cost: float = 0.0012,
        risk_free_rate: float = 0.03,
    ):
        self.df = df.copy()
        self.lookback = lookback
        self.top_k = top_k
        self.max_position = max_position
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate

        # 特征列
        if feature_cols is None:
            self.feature_cols = self._default_features()
        else:
            self.feature_cols = feature_cols

        # 预处理数据
        self._preprocess()

        # 环境状态
        self.current_step = 0          # 当前月份索引
        self.portfolio_value = 1.0     # 组合净值
        self.positions = {}            # 当前持仓 {stock_code: weight}
        self.prev_positions = {}       # 上期持仓
        self.returns_history = deque(maxlen=lookback)  # 收益历史
        self.drawdown = 0.0            # 当前回撤
        self.peak_value = 1.0          # 历史最高净值

    def _default_features(self) -> List[str]:
        """默认特征列"""
        return [
            # 量价特征 (14维)
            "bias_5", "bias_10", "bias_20",
            "振幅_5", "振幅_10", "振幅_20",
            "涨跌幅std_5", "涨跌幅std_10", "涨跌幅std_20",
            "成交额std_5", "成交额std_10", "成交额std_20",
            "涨跌幅_10", "涨跌幅_20",
            # 技术指标 (6维)
            "K", "D", "J",
            "DIF", "DEA", "MACD",
            # 基本面 (4维)
            "市盈率倒数", "市净率倒数",
            "总市值_log", "成交额_log",
            # 市场特征 (从单只股票不好构建，在 state 中单独处理)
        ]

    def _preprocess(self):
        """数据预处理"""
        df = self.df

        # 数值化
        for col in self.feature_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 添加衍生特征
        df["总市值_log"] = np.log(pd.to_numeric(df["总市值"], errors="coerce").clip(lower=1e8))
        df["成交额_log"] = np.log(pd.to_numeric(df["成交额"], errors="coerce").clip(lower=1e4))

        # 填充缺失值
        for col in self.feature_cols + ["总市值_log", "成交额_log"]:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)

        # 按日期和股票排序
        df = df.sort_values(["交易日期", "股票代码"]).reset_index(drop=True)

        # 提取唯一日期
        self.dates = sorted(df["交易日期"].unique())
        self.n_dates = len(self.dates)
        self.df = df

    def _get_month_data(self, date) -> pd.DataFrame:
        """获取某月的数据截面"""
        month_df = self.df[self.df["交易日期"] == date].copy()
        return month_df

    def _extract_state(self, month_df: pd.DataFrame) -> np.ndarray:
        """
        提取状态向量

        状态 = [市场特征(5) | 截面特征统计(10) | 组合特征(3+top_k)]

        Returns:
            state: (state_dim,) numpy array
        """
        states = []

        # ── 市场特征 (5维) ──
        if "涨跌幅" in month_df.columns:
            mkt_return = month_df["涨跌幅"].mean()
            mkt_std = month_df["涨跌幅"].std()
        else:
            mkt_return = 0.0
            mkt_std = 0.01

        # 市场状态 (bull/bear)
        if "市场状态" in month_df.columns:
            bull_frac = (month_df["市场状态"] == "bull").mean()
        else:
            bull_frac = 0.5

        # 截面分散度
        cross_dispersion = 0.0
        if "涨跌幅_10" in month_df.columns:
            cross_dispersion = month_df["涨跌幅_10"].std() if month_df["涨跌幅_10"].notna().any() else 0.0

        states.extend([
            mkt_return, mkt_std, bull_frac, cross_dispersion,
            self.drawdown,  # 当前回撤
        ])

        # ── 截面特征统计 (10维) ──
        stat_features = []
        for feat in self.feature_cols[:10]:  # 取前10个关键特征
            if feat in month_df.columns:
                vals = month_df[feat].dropna()
                if len(vals) > 0:
                    stat_features.extend([
                        vals.mean(), vals.std(), vals.skew() if len(vals) > 2 else 0.0,
                    ])
                else:
                    stat_features.extend([0, 0, 0])
            else:
                stat_features.extend([0, 0, 0])

        # 截断到10维 (取前10个)
        states.extend(stat_features[:10])

        # ── 组合特征 (3 + top_k 维) ──
        portfolio_return = self.returns_history[-1] if self.returns_history else 0.0
        portfolio_vol = np.std(list(self.returns_history)) if len(self.returns_history) > 1 else 0.01
        n_positions = len(self.positions)

        states.extend([portfolio_return, portfolio_vol, float(n_positions) / self.top_k])

        # 当前持仓权重表示 (top_k 维)
        for i in range(self.top_k):
            states.append(1.0 if i < n_positions else 0.0)

        return np.array(states, dtype=np.float32)

    def _calculate_reward(
        self,
        month_df: pd.DataFrame,
        selected_codes: List[str],
    ) -> float:
        """
        计算奖励

        r_t = portfolio_return - cost_penalty - risk_penalty

        使用下一周期 (下周期每天涨跌幅) 的真实收益
        """
        if len(selected_codes) == 0:
            return -0.01  # 空仓惩罚

        # 获取每只选中股票的下月收益
        rets = []
        for code in selected_codes:
            stock_row = month_df[month_df["股票代码"] == code]
            if len(stock_row) == 0:
                continue
            row = stock_row.iloc[0]
            # 下周期每天涨跌幅
            daily_rets = row.get("下周期每天涨跌幅", [])
            if isinstance(daily_rets, str):
                try:
                    import ast
                    daily_rets = ast.literal_eval(daily_rets)
                except (ValueError, SyntaxError):
                    daily_rets = []
            if isinstance(daily_rets, list) and len(daily_rets) > 0:
                # 等权组合收益 = 每日收益的累积
                cum_ret = np.prod([1 + r for r in daily_rets])
                rets.append(cum_ret - 1)
            else:
                rets.append(0.0)

        if len(rets) == 0:
            return -0.01

        # 等权组合收益
        portfolio_return = np.mean(rets)

        # 换手率惩罚
        prev_codes = set(self.prev_positions.keys())
        curr_codes = set(selected_codes)
        turnover = 1 - len(prev_codes & curr_codes) / max(len(curr_codes), 1)
        cost_penalty = turnover * self.transaction_cost

        # 回撤惩罚
        drawdown_penalty = max(0, self.drawdown - 0.2) * 0.1

        reward = portfolio_return - cost_penalty - drawdown_penalty

        return float(reward)

    def reset(self) -> np.ndarray:
        """重置环境到初始状态"""
        self.current_step = self.lookback  # 从第 lookback 月开始 (确保有足够历史)
        self.portfolio_value = 1.0
        self.positions = {}
        self.prev_positions = {}
        self.returns_history = deque(maxlen=self.lookback)
        self.drawdown = 0.0
        self.peak_value = 1.0

        date = self.dates[self.current_step]
        month_df = self._get_month_data(date)

        return self._extract_state(month_df)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步

        Parameters
        ----------
        action : np.ndarray (n_stocks,) 或 (top_k,)
            离散动作: top_k 个选中的股票索引
            连续动作: 每只股票的权重

        Returns
        -------
        state, reward, done, info
        """
        date = self.dates[self.current_step]
        month_df = self._get_month_data(date)

        if len(month_df) == 0:
            # 无数据，跳过
            self.current_step += 1
            done = self.current_step >= self.n_dates - 1
            if not done:
                next_date = self.dates[self.current_step]
                next_df = self._get_month_data(next_date)
                return self._extract_state(next_df), 0.0, done, {}
            return np.zeros(24), 0.0, done, {}

        # 解析动作
        n_stocks_available = len(month_df)
        stock_codes = month_df["股票代码"].tolist()

        if isinstance(action, np.ndarray) and action.ndim == 1:
            if len(action) == n_stocks_available:
                # 连续动作: 权重向量
                weights = np.abs(action)
                weights = weights / (weights.sum() + 1e-8)
                # 选 top_k
                top_indices = np.argsort(weights)[-self.top_k:]
                selected_codes = [stock_codes[i] for i in top_indices]
                self.positions = {stock_codes[i]: float(weights[i]) for i in top_indices}
            elif len(action) == self.top_k:
                # 离散动作: top_k 个股票索引
                indices = np.clip(action.astype(int), 0, n_stocks_available - 1)
                selected_codes = [stock_codes[i] for i in indices]
                self.positions = {code: 1.0 / self.top_k for code in selected_codes}
            else:
                # fallback: 随机选
                selected_codes = np.random.choice(
                    stock_codes, size=min(self.top_k, n_stocks_available), replace=False
                ).tolist()
                self.positions = {code: 1.0 / len(selected_codes) for code in selected_codes}
        else:
            selected_codes = []
            self.positions = {}

        # 计算奖励
        reward = self._calculate_reward(month_df, selected_codes)

        # 更新状态
        self.prev_positions = self.positions.copy()
        self.returns_history.append(reward)
        self.portfolio_value *= (1 + reward)
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
        self.drawdown = 1 - self.portfolio_value / self.peak_value

        # 下一步
        self.current_step += 1
        done = self.current_step >= self.n_dates - 1

        if not done:
            next_date = self.dates[self.current_step]
            next_df = self._get_month_data(next_date)
            next_state = self._extract_state(next_df)
        else:
            next_state = np.zeros(24, dtype=np.float32)

        info = {
            "date": str(date)[:10],
            "selected_codes": selected_codes,
            "reward": reward,
            "portfolio_value": self.portfolio_value,
            "drawdown": self.drawdown,
            "n_selected": len(selected_codes),
        }

        return next_state, reward, done, info

    @property
    def state_dim(self) -> int:
        """状态空间维度"""
        return 24  # 5 market + 10 cross-section + 3 portfolio + 6 position

    @property
    def action_dim(self) -> int:
        """动作空间维度 (top_k 个选中的股票索引)"""
        return self.top_k

    def get_available_stocks(self) -> List[str]:
        """获取当前可用的股票列表"""
        date = self.dates[self.current_step]
        month_df = self._get_month_data(date)
        return month_df["股票代码"].tolist()

    def seed(self, seed: int = None):
        """设置随机种子"""
        np.random.seed(seed)
        return [seed]
