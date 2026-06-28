"""
Custom Gym Environment for Monthly Stock Selection

This environment simulates a monthly stock selection process:
  1. At each step (month), the agent observes a factor matrix of N stocks
  2. The agent selects K stocks (or outputs allocation weights)
  3. The environment returns the next-month portfolio return as reward

Key design decisions:
  - Universe: Top M stocks by market cap each month (fixed observation dim)
  - Observation: Stock factor matrix + LLM embeddings + market features
  - Action: Continuous vector (weights per stock), top-K selected for equal-weight
  - Reward: Portfolio return - benchmark (可配置)

Market constraints modeled:
  - T+1 settlement: Buy at month-end close, hold for full next month
  - Transaction costs: 1.2‱ commission + 1‰ stamp duty (sell only)
  - No short selling (long-only)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
from collections import deque
import warnings

warnings.filterwarnings("ignore")


class StockSelectionEnv(gym.Env):
    """Monthly stock selection environment.

    Observation Space:
      Box(low=-inf, high=inf, shape=(obs_dim,)):
        obs_dim = n_stocks * n_features_per_stock + n_llm_features + n_market_features

    Action Space:
      Box(low=-1, high=1, shape=(n_stocks,)):
        Continuous scores for each stock. Top K scores are selected.

    Attributes
    ----------
    dates : list
        Sorted list of valid rebalance dates.
    current_idx : int
        Current position in dates list.
    portfolio_value : float
        Track cumulative portfolio value.
    history : deque
        Rolling window of past observations.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        market_features_df: pd.DataFrame = None,
        llm_encoder=None,
        select_stock_num: int = 5,
        universe_size: int = 100,
        max_lookback: int = 12,
        transaction_cost_buy: float = 1.2 / 10000,   # 万分之1.2
        transaction_cost_sell: float = 1.2 / 10000 + 1 / 1000,  # 佣金+印花税
        benchmark_col: str = "mkt_mean_return",
        reward_type: str = "excess_return",  # excess_return, raw_return, sharpe
        render_mode: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        df : pd.DataFrame
            Processed stock data with normalized features.
        feature_cols : list
            Normalized feature column names.
        market_features_df : pd.DataFrame, optional
            Market-level features per date.
        llm_encoder : BaseEncoder, optional
            LLM encoder for text-to-embedding conversion.
        select_stock_num : int
            Number of stocks to select (K).
        universe_size : int
            Number of stocks in the universe (M).
        max_lookback : int
            Number of past observations to include (not fully used yet).
        transaction_cost_buy : float
            Buy cost as fraction of trade value.
        transaction_cost_sell : float
            Sell cost as fraction of trade value.
        benchmark_col : str
            Column name for benchmark return in market_features_df.
        reward_type : str
            Type of reward function.
        render_mode : str, optional
            Gym render mode.
        """
        super().__init__()

        self.df = df.copy()
        self.feature_cols = [c for c in feature_cols if c in df.columns]
        self.market_features_df = market_features_df
        self.llm_encoder = llm_encoder
        self.select_stock_num = select_stock_num
        self.universe_size = universe_size
        self.max_lookback = max_lookback
        self.transaction_cost_buy = transaction_cost_buy
        self.transaction_cost_sell = transaction_cost_sell
        self.benchmark_col = benchmark_col
        self.reward_type = reward_type
        self.render_mode = render_mode

        # Get sorted unique dates
        self.all_dates = sorted(self.df["交易日期"].unique())
        self.dates = self.all_dates[max_lookback:]  # skip first lookback for warm-up

        # Pre-compute per-date stock data for efficiency
        self._date_cache = {}
        self._precompute_date_data()

        # Build observation and action spaces
        self._setup_spaces()

        # State tracking
        self.current_idx = 0
        self.portfolio_value = 1.0
        self.prev_selected = None  # previously selected stock codes
        self.portfolio_history = []  # track portfolio values
        self.trade_history = []  # track individual trade results

        # Observation history
        self.obs_history = deque(maxlen=max_lookback)

    def _precompute_date_data(self):
        """Pre-compute stock data per date for fast access."""
        for date in self.all_dates:
            date_df = self.df[self.df["交易日期"] == date].copy()
            if date_df.empty:
                continue

            # Sort by market cap
            if "总市值" in date_df.columns:
                date_df = date_df.sort_values("总市值", ascending=False)
            date_df = date_df.head(self.universe_size)

            # Ensure feature columns exist
            available_features = [c for c in self.feature_cols if c in date_df.columns]

            self._date_cache[date] = {
                "df": date_df,
                "features": date_df[available_features].values.astype(np.float32),
                "stock_codes": date_df["股票代码"].values,
                "n_stocks": len(date_df),
            }

    def _setup_spaces(self):
        """Configure observation and action spaces."""
        # Get a sample observation to determine dimensions
        sample_date = self.all_dates[0]
        sample = self._date_cache.get(sample_date)
        if sample is None:
            raise ValueError("No data available for setup")

        n_stocks = sample["n_stocks"]
        n_stock_features = sample["features"].shape[1]

        # LLM embedding dimension
        if self.llm_encoder is not None:
            n_llm_features = self.llm_encoder.embedding_dim
        else:
            n_llm_features = 0

        # Market features dimension
        if self.market_features_df is not None:
            mkt_cols = [c for c in self.market_features_df.columns
                        if c != "交易日期"]
            n_market_features = len(mkt_cols)
        else:
            n_market_features = 0

        # Total observation dimension
        # Stock features: (n_stocks, n_features_per_stock) flattened
        # LLM features: (n_stocks, n_llm_features) flattened
        # Market features: (n_market_features,)
        self.n_stocks = n_stocks
        self.n_stock_features = n_stock_features
        self.n_llm_features = n_llm_features
        self.n_market_features = n_market_features

        obs_dim = (n_stocks * n_stock_features +
                   n_stocks * n_llm_features +
                   n_market_features)

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )

        # Action space: one score per stock in [-1, 1]
        # Score > 0 means "prefer", score < 0 means "avoid"
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_stocks,), dtype=np.float32
        )

        self.obs_dim = obs_dim

    def _get_llm_embeddings(self, date_df: pd.DataFrame) -> np.ndarray:
        """Get LLM embeddings for stocks on a given date."""
        if self.llm_encoder is None:
            return np.zeros((len(date_df), 0), dtype=np.float32)

        try:
            embeddings = self.llm_encoder.encode(date_df)
        except Exception:
            embeddings = np.zeros((len(date_df), self.llm_encoder.embedding_dim),
                                  dtype=np.float32)

        # Ensure correct shape
        if len(embeddings) < self.n_stocks:
            pad = np.zeros((self.n_stocks - len(embeddings), embeddings.shape[1]),
                           dtype=np.float32)
            embeddings = np.concatenate([embeddings, pad], axis=0)
        elif len(embeddings) > self.n_stocks:
            embeddings = embeddings[:self.n_stocks]

        return embeddings

    def _get_market_features(self, date: pd.Timestamp) -> np.ndarray:
        """Get market features for a given date."""
        if self.market_features_df is None:
            return np.zeros(self.n_market_features, dtype=np.float32)

        mkt = self.market_features_df[
            self.market_features_df["交易日期"] == date
        ]
        if mkt.empty:
            return np.zeros(self.n_market_features, dtype=np.float32)

        mkt_cols = [c for c in self.market_features_df.columns if c != "交易日期"]
        return mkt[mkt_cols].values.astype(np.float32).flatten()

    def _build_observation(self, date: pd.Timestamp) -> np.ndarray:
        """Build the full observation vector for a given date."""
        cached = self._date_cache.get(date)
        if cached is None:
            return np.zeros(self.obs_dim, dtype=np.float32)

        date_df = cached["df"]
        stock_features = cached["features"]

        # Pad stock features if needed
        n_actual = stock_features.shape[0]
        if n_actual < self.n_stocks:
            pad = np.zeros((self.n_stocks - n_actual, stock_features.shape[1]),
                           dtype=np.float32)
            stock_features = np.concatenate([stock_features, pad], axis=0)
        elif n_actual > self.n_stocks:
            stock_features = stock_features[:self.n_stocks]

        # Get LLM embeddings
        llm_emb = self._get_llm_embeddings(date_df)

        # Get market features
        mkt_feat = self._get_market_features(date)

        # Concatenate into flat observation
        obs = np.concatenate([
            stock_features.flatten(),
            llm_emb.flatten(),
            mkt_feat.flatten(),
        ])

        return obs.astype(np.float32)

    def _get_next_period_return(self, date: pd.Timestamp,
                                 selected_codes: List[str]) -> np.ndarray:
        """Get forward (next-period) return for selected stocks.

        IMPORTANT: Returns the "涨跌幅" from the NEXT date, representing
        the return from 'date' to 'date+1'. This avoids lookahead bias.

        Parameters
        ----------
        date : pd.Timestamp
            Current rebalance date (entry point).
        selected_codes : list of str
            Selected stock codes.

        Returns
        -------
        np.ndarray
            Array of per-stock next-period returns.
        """
        # Find the next date in the schedule
        try:
            date_idx = list(self.all_dates).index(date)
            next_date = self.all_dates[date_idx + 1]
        except (ValueError, IndexError):
            return np.zeros(len(selected_codes))

        cached = self._date_cache.get(next_date)
        if cached is None:
            return np.zeros(len(selected_codes))

        next_df = cached["df"]
        returns = []

        for code in selected_codes:
            stock_data = next_df[next_df["股票代码"] == code]
            if stock_data.empty:
                returns.append(0.0)
                continue

            # Use 涨跌幅 from the NEXT date (forward return)
            if "涨跌幅" in stock_data.columns:
                ret = stock_data["涨跌幅"].values[0]
                if pd.isna(ret):
                    ret = 0.0
            else:
                ret = 0.0

            returns.append(float(ret))

        return np.array(returns)

    def reset(self, seed: Optional[int] = None,
              options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment to the beginning of the date range.

        Returns
        -------
        observation : np.ndarray
        info : dict
        """
        super().reset(seed=seed)

        # Start with some randomness in offset for training variation
        if options and "start_idx" in options:
            start_idx = options["start_idx"]
        else:
            max_start = max(0, len(self.dates) - self.max_lookback - 1)
            start_idx = 0
            if max_start > 0 and self.np_random is not None:
                start_idx = self.np_random.integers(0, min(max_start, 20))

        self.current_idx = start_idx
        self.portfolio_value = 1.0
        self.prev_selected = None
        self.portfolio_history = [1.0]
        self.trade_history = []
        self.obs_history.clear()

        obs = self._build_observation(self.dates[self.current_idx])

        # Fill observation history with initial observation
        for _ in range(self.max_lookback):
            self.obs_history.append(obs.copy())

        info = {"date": str(self.dates[self.current_idx]),
                "portfolio_value": self.portfolio_value}

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step (month) of stock selection.

        Parameters
        ----------
        action : np.ndarray of shape (n_stocks,)
            Continuous scores for each stock. Top K are selected.

        Returns
        -------
        observation : np.ndarray
        reward : float
        terminated : bool
        truncated : bool
        info : dict
        """
        current_date = self.dates[self.current_idx]
        cached = self._date_cache.get(current_date)
        if cached is None:
            # No data for this date, skip
            self.current_idx += 1
            terminated = self.current_idx >= len(self.dates) - 1
            obs = self._build_observation(self.dates[self.current_idx]) if not terminated else np.zeros(self.obs_dim, dtype=np.float32)
            return obs, 0.0, terminated, False, {}

        stock_codes = cached["stock_codes"]
        n_available = len(stock_codes)

        # Clip action to match available stocks
        action_scores = action[:n_available]

        # Select top K stocks by action score
        k = min(self.select_stock_num, n_available)
        top_k_indices = np.argsort(action_scores)[-k:]
        selected_codes = [stock_codes[i] for i in top_k_indices]

        # Get forward-looking next-period return
        # From date[t] to date[t+1], use date[t+1]'s 涨跌幅
        stock_returns = self._get_next_period_return(current_date, selected_codes)

        # Calculate portfolio return (equal weight)
        portfolio_return = np.mean(stock_returns)

        # Apply transaction costs
        if self.prev_selected is not None:
            sold = set(self.prev_selected) - set(selected_codes)
            bought = set(selected_codes) - set(self.prev_selected)
            turnover_rate = (len(sold) + len(bought)) / max(len(selected_codes), 1)
        else:
            turnover_rate = 1.0

        total_cost = turnover_rate * (self.transaction_cost_buy + self.transaction_cost_sell)
        portfolio_return_after_cost = portfolio_return - total_cost

        self.prev_selected = selected_codes
        self.portfolio_value *= (1 + portfolio_return_after_cost)
        self.portfolio_history.append(self.portfolio_value)

        # Advance to next date BEFORE computing reward
        # (reward uses forward date's market data for benchmarking)
        self.current_idx += 1
        terminated = self.current_idx >= len(self.dates) - 1
        truncated = False

        # Compute reward using forward date's market features
        if not terminated:
            forward_date = self.dates[self.current_idx]
            reward = self._compute_reward(portfolio_return_after_cost, forward_date)
            obs = self._build_observation(forward_date)
            self.obs_history.append(obs.copy())
        else:
            reward = self._compute_reward(portfolio_return_after_cost, current_date)
            obs = np.zeros(self.obs_dim, dtype=np.float32)

        info = {
            "date": str(current_date),
            "portfolio_value": self.portfolio_value,
            "portfolio_return": portfolio_return_after_cost,
            "n_selected": k,
            "selected_codes": selected_codes,
            "turnover": turnover_rate,
        }

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, portfolio_return: float,
                        date: pd.Timestamp) -> float:
        """Compute the reward for the current step.

        Parameters
        ----------
        portfolio_return : float
            Portfolio return after costs.
        date : pd.Timestamp
            Current date.

        Returns
        -------
        float
        """
        if self.reward_type == "raw_return":
            return float(portfolio_return)

        elif self.reward_type == "excess_return":
            # Excess return over market average
            mkt_feat = self._get_market_features(date)
            # mkt_mean_return is the first market feature
            if len(mkt_feat) > 0:
                benchmark_return = float(mkt_feat[0])
            else:
                benchmark_return = 0.0
            return float(portfolio_return - benchmark_return)

        elif self.reward_type == "sharpe":
            # Rolling Sharpe-like reward
            self.trade_history.append(portfolio_return)
            if len(self.trade_history) < 3:
                return float(portfolio_return)
            recent = self.trade_history[-12:]
            mean_ret = np.mean(recent)
            std_ret = np.std(recent) + 1e-8
            # Reward: recent Sharpe * scaling factor
            return float(mean_ret / std_ret * 0.1)

        elif self.reward_type == "sortino":
            # Downside-risk adjusted
            self.trade_history.append(portfolio_return)
            if len(self.trade_history) < 3:
                return float(portfolio_return)
            recent = self.trade_history[-12:]
            mean_ret = np.mean(recent)
            downside = [r for r in recent if r < 0]
            if len(downside) == 0:
                return float(mean_ret)
            downside_std = np.std(downside) + 1e-8
            return float(mean_ret / downside_std * 0.1)

        else:
            return float(portfolio_return)

    def get_portfolio_history(self) -> List[float]:
        """Get portfolio value history."""
        return self.portfolio_history

    def render(self):
        """Simple text-based rendering."""
        if self.render_mode == "human":
            idx = self.current_idx
            if idx < len(self.dates):
                date = self.dates[idx]
                print(f"[{date}] Portfolio: {self.portfolio_value:.4f}")
