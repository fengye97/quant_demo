"""
Feature Engineering for RL Stock Selection

Extracts and normalizes features from stock_data.csv for use as
the state representation in the RL environment.

Features extracted:
  - Price/Technical: bias_5, bias_10, bias_20, 振幅_5/10/20, K/D/J, DIF/DEA/MACD
  - Volatility: 涨跌幅std_5/10/20, 成交额std_5/10/20
  - Fundamental: 市盈率倒数, 市净率倒数
  - Momentum: 涨跌幅_10, 涨跌幅_20, 涨跌幅
  - Size/Liquidity: log(总市值), log(成交额)
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings

warnings.filterwarnings("ignore")

# ── Feature column definitions ─────────────────────────────────────────────

# Primary technical / price features
TECHNICAL_FEATURES = [
    "bias_5", "bias_10", "bias_20",           # 乖离率: 收盘价偏离均线百分比
    "振幅_5", "振幅_10", "振幅_20",              # 振幅: (高-低)/前收，过去N日均值
    "K", "D", "J",                              # KDJ 指标
    "DIF", "DEA", "MACD",                       # MACD 指标
]

# Volatility & volume features
VOLATILITY_FEATURES = [
    "涨跌幅std_5", "涨跌幅std_10", "涨跌幅std_20",   # 涨跌幅波动率
    "成交额std_5", "成交额std_10", "成交额std_20",   # 成交额波动率
]

# Fundamental features
FUNDAMENTAL_FEATURES = [
    "市盈率倒数",                                  # E/P: 盈利收益率
    "市净率倒数",                                  # B/P: 净资产收益率倒数
]

# Momentum features
MOMENTUM_FEATURES = [
    "涨跌幅",                                      # 当月涨跌幅
    "涨跌幅_10",                                   # 过去10日涨跌幅
    "涨跌幅_20",                                   # 过去20日涨跌幅
]

# Size / liquidity features (derived)
DERIVED_FEATURES = [
    "log_市值",                                    # log(总市值)
    "log_成交额",                                  # log(成交额)
    "log_流通市值",                                # log(流通市值)
]

# ── Selected feature set (configurable) ────────────────────────────────────

DEFAULT_FEATURES = (
    TECHNICAL_FEATURES +
    VOLATILITY_FEATURES +
    FUNDAMENTAL_FEATURES +
    MOMENTUM_FEATURES
)

# ── Market features (global, same for all stocks at a given date) ──────────

MARKET_FEATURES = [
    "mkt_mean_return",        # 全市场等权平均涨跌幅
    "mkt_std_return",         # 全市场涨跌幅标准差
    "mkt_pct_positive",       # 上涨股票占比
    "mkt_mean_turnover",      # 平均换手率(用成交额/流通市值近似)
    "mkt_ma12_signal",        # 市场状态: 牛/熊 (1=cumulative > MA12, 0 otherwise)
]

# ── All features (stock-level + market) ────────────────────────────────────

ALL_FEATURES = DEFAULT_FEATURES + DERIVED_FEATURES


def load_stock_data(csv_path: str = None) -> pd.DataFrame:
    """Load stock data from CSV with proper encoding.

    Parameters
    ----------
    csv_path : str, optional
        Path to stock_data.csv. Defaults to stock_trade_demo/stock_data.csv.

    Returns
    -------
    pd.DataFrame
        Raw stock data with parsed dates.
    """
    if csv_path is None:
        import os
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base, "stock_trade_demo", "stock_data.csv")

    df = pd.read_csv(csv_path, encoding="gbk", parse_dates=["交易日期"], low_memory=False)

    # Remove daily-level rows from the last month — keep only month-end
    # Dates with <= 5 stocks are treated as incomplete daily snapshots
    date_counts = df.groupby("交易日期").size()
    valid_dates = date_counts[date_counts > 100].index  # real months have >100 stocks
    df = df[df["交易日期"].isin(valid_dates)].copy()

    df.sort_values(["交易日期", "股票代码"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def filter_stock_universe(df: pd.DataFrame, min_listing_days: int = 250,
                          exclude_bj: bool = True,
                          min_market_cap_rank: Optional[int] = None) -> pd.DataFrame:
    """Filter stocks to create the investment universe.

    Parameters
    ----------
    df : pd.DataFrame
        Raw stock data.
    min_listing_days : int
        Exclude stocks listed less than this many days.
    exclude_bj : bool
        Exclude Beijing Stock Exchange (bj) stocks.
    min_market_cap_rank : int, optional
        If set, keep only top N stocks by market cap each month.

    Returns
    -------
    pd.DataFrame
        Filtered stock data.
    """
    df = df.copy()

    if "上市至今交易天数" in df.columns:
        df = df[df["上市至今交易天数"] > min_listing_days]

    if exclude_bj:
        df = df[~df["股票代码"].str.contains("bj")]

    if min_market_cap_rank is not None:
        df["cap_rank"] = df.groupby("交易日期")["总市值"].rank(ascending=False)
        df = df[df["cap_rank"] <= min_market_cap_rank]
        df.drop(columns=["cap_rank"], inplace=True)

    return df


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features from raw data.

    Parameters
    ----------
    df : pd.DataFrame
        Stock data with raw columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with added derived feature columns.
    """
    df = df.copy()

    # Log market cap (handle zero/negative)
    for col, new_col in [("总市值", "log_市值"),
                         ("成交额", "log_成交额"),
                         ("流通市值", "log_流通市值")]:
        if col in df.columns:
            vals = df[col].values.astype(float)
            vals = np.where(vals > 0, vals, np.nan)
            df[new_col] = np.log(vals)

    return df


def compute_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute market-level features for each date.

    Parameters
    ----------
    df : pd.DataFrame
        Stock data with '涨跌幅' and other columns.

    Returns
    -------
    pd.DataFrame
        Date-indexed DataFrame with market features.
    """
    mkt = df.groupby("交易日期").agg(
        mkt_mean_return=("涨跌幅", "mean"),
        mkt_std_return=("涨跌幅", "std"),
        mkt_pct_positive=("涨跌幅", lambda x: (x > 0).mean()),
    ).reset_index()

    # Approximate turnover: 成交额 / 流通市值
    if "成交额" in df.columns and "流通市值" in df.columns:
        df_temp = df.copy()
        df_temp["turnover_proxy"] = np.where(
            df_temp["流通市值"] > 0,
            df_temp["成交额"] / df_temp["流通市值"],
            np.nan
        )
        mkt_turn = df_temp.groupby("交易日期")["turnover_proxy"].mean().reset_index()
        mkt_turn.rename(columns={"turnover_proxy": "mkt_mean_turnover"}, inplace=True)
        mkt = mkt.merge(mkt_turn, on="交易日期", how="left")

    # Market regime signal: cumulative return vs 12-month MA
    mkt = mkt.sort_values("交易日期")
    mkt["mkt_cum"] = (1 + mkt["mkt_mean_return"]).cumprod()
    mkt["mkt_ma12"] = mkt["mkt_cum"].rolling(12, min_periods=1).mean()
    mkt["mkt_ma12_signal"] = (mkt["mkt_cum"] > mkt["mkt_ma12"]).astype(float)

    mkt.drop(columns=["mkt_cum", "mkt_ma12"], inplace=True)
    return mkt


def handle_missing_values(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Handle NaN values in feature columns.

    Strategy:
      1. Per stock: forward-fill only (bfill is look-ahead bias in time series)
      2. Cross-section: fill remaining NaN with date median
      3. Any remaining: fill with 0

    Parameters
    ----------
    df : pd.DataFrame
        Stock data.
    feature_cols : list
        Feature columns to process.

    Returns
    -------
    pd.DataFrame
        DataFrame with NaN handled.
    """
    df = df.copy()

    for col in feature_cols:
        if col not in df.columns:
            continue

        # Per-stock time-series fill (ffill only — bfill would leak future values)
        df[col] = df.groupby("股票代码")[col].transform(
            lambda x: x.ffill()
        )

        # Cross-section median fill
        medians = df.groupby("交易日期")[col].transform("median")
        df[col] = df[col].fillna(medians)

        # Final fallback
        df[col] = df[col].fillna(0.0)

    return df


def normalize_features(df: pd.DataFrame, feature_cols: List[str],
                       scaler_type: str = "robust",
                       scaler_dict: Optional[Dict[str, object]] = None
                       ) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Normalize features cross-sectionally (per date) or globally.

    Cross-sectional normalization is preferred for financial data
    because it removes time-varying market-wide effects.

    Parameters
    ----------
    df : pd.DataFrame
        Stock data with feature columns.
    feature_cols : list
        Feature columns to normalize.
    scaler_type : str
        "robust" (RobustScaler, recommended) or "standard" (StandardScaler).
    scaler_dict : dict, optional
        Pre-fitted scalers for each feature (used during test/inference).

    Returns
    -------
    df : pd.DataFrame
        DataFrame with normalized feature columns.
    scaler_dict : dict
        Fitted scalers (keyed by feature name).
    """
    df = df.copy()
    fit_scalers = scaler_dict is None
    if fit_scalers:
        scaler_dict = {}

    for col in feature_cols:
        if col not in df.columns:
            continue

        new_col = f"{col}_norm"

        if fit_scalers:
            if scaler_type == "robust":
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
            # Fit on all non-NaN training data
            valid = df[col].dropna().values.reshape(-1, 1)
            scaler.fit(valid)
            scaler_dict[col] = scaler
        else:
            if col not in scaler_dict:
                continue
            scaler = scaler_dict[col]

        # Transform
        values = df[col].values.reshape(-1, 1)
        mask = ~np.isnan(values.flatten())
        if mask.any():
            transformed = scaler.transform(values[mask].reshape(-1, 1))
            df.loc[df.index[mask], new_col] = transformed.flatten()
        else:
            df[new_col] = 0.0

        df[new_col] = df[new_col].fillna(0.0)

        # Clip extreme values
        df[new_col] = df[new_col].clip(-5, 5)

    return df, scaler_dict


def prepare_features(df: pd.DataFrame,
                     feature_cols: Optional[List[str]] = None,
                     scaler_type: str = "robust",
                     scaler_dict: Optional[Dict] = None,
                     top_n_stocks: int = 100,
                     market_features_df: Optional[pd.DataFrame] = None,
                     ) -> Tuple[pd.DataFrame, List[str], Dict, pd.DataFrame]:
    """Complete feature preparation pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw stock data.
    feature_cols : list, optional
        Feature columns to use. Defaults to DEFAULT_FEATURES.
    scaler_type : str
        "robust" or "standard".
    scaler_dict : dict, optional
        Pre-fitted scalers for inference.
    top_n_stocks : int
        Number of top market-cap stocks to keep per date.
    market_features_df : pd.DataFrame, optional
        Pre-computed global market features (computed on full dataset before
        train/val/test split so mkt_cum is historically continuous). When
        provided, filtered to this split's dates instead of recomputing.

    Returns
    -------
    df_out : pd.DataFrame
        Processed data with normalized features.
    all_feature_cols : list
        List of normalized feature column names.
    scaler_dict : dict
        Fitted scalers.
    market_features : pd.DataFrame
        Market-level features per date.
    """
    if feature_cols is None:
        feature_cols = list(DEFAULT_FEATURES)

    df = df.copy()

    # Derived features
    df = compute_derived_features(df)

    # Market features — use pre-computed global version when provided so that
    # mkt_cum (and thus mkt_ma12_signal) is continuous across splits.
    if market_features_df is not None:
        split_dates = df["交易日期"].unique()
        mkt_feat = market_features_df[market_features_df["交易日期"].isin(split_dates)].copy()
    else:
        mkt_feat = compute_market_features(df)

    # Handle missing values
    all_cols = feature_cols + DERIVED_FEATURES
    all_cols = [c for c in all_cols if c in df.columns]
    df = handle_missing_values(df, all_cols)

    # Filter universe by market cap
    df["_cap_rank"] = df.groupby("交易日期")["总市值"].rank(ascending=False)
    df = df[df["_cap_rank"] <= top_n_stocks]
    df.drop(columns=["_cap_rank"], inplace=True)

    # Normalize
    df, scaler_dict = normalize_features(df, all_cols, scaler_type, scaler_dict)

    # Normalized feature names
    all_norm_cols = [f"{c}_norm" for c in all_cols]

    return df, all_norm_cols, scaler_dict, mkt_feat


def build_observation_matrix(df: pd.DataFrame,
                             norm_feature_cols: List[str],
                             market_features: pd.DataFrame,
                             date: pd.Timestamp,
                             ) -> np.ndarray:
    """Build the observation matrix for a given rebalance date.

    Returns a flat vector: stock features concatenated for all stocks,
    plus market features at the end.

    Parameters
    ----------
    df : pd.DataFrame
        Processed data.
    norm_feature_cols : list
        Normalized feature column names.
    market_features : pd.DataFrame
        Market features per date.
    date : pd.Timestamp
        The rebalance date.

    Returns
    -------
    np.ndarray
        Flattened observation vector.
    """
    date_data = df[df["交易日期"] == date].copy()

    if date_data.empty:
        raise ValueError(f"No data for date {date}")

    # Sort by market cap descending (consistent ordering)
    date_data = date_data.sort_values("总市值", ascending=False)

    # Stock features: (n_stocks, n_features)
    stock_feat = date_data[norm_feature_cols].values.astype(np.float32)

    # Market features
    mkt = market_features[market_features["交易日期"] == date]
    if not mkt.empty:
        mkt_vals = mkt[MARKET_FEATURES].values.astype(np.float32).flatten()
    else:
        mkt_vals = np.zeros(len(MARKET_FEATURES), dtype=np.float32)

    # Flatten and concatenate
    obs = np.concatenate([stock_feat.flatten(), mkt_vals])

    return obs


def get_date_splits(dates: np.ndarray,
                    train_end: str = "2021-12-31",
                    val_end: str = "2024-12-31",
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split dates into train / validation / test sets.

    Parameters
    ----------
    dates : np.ndarray
        Sorted array of unique dates.
    train_end : str
        Last training date.
    val_end : str
        Last validation date.

    Returns
    -------
    train_dates, val_dates, test_dates : np.ndarray
    """
    train_end = pd.Timestamp(train_end)
    val_end = pd.Timestamp(val_end)

    train_dates = dates[dates <= train_end]
    val_dates = dates[(dates > train_end) & (dates <= val_end)]
    test_dates = dates[dates > val_end]

    return train_dates, val_dates, test_dates
