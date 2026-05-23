#!/usr/bin/env python
"""
Main Training Script for RL + LLM Stock Selection (Plan A)

Architecture:
  [Financial Data] -> [LLM Encoder (Mock/FinBERT)] -> [State Vector] --+
  [Price/Technical] -> [Feature Engineering] -> [State Vector] ---------+-> [PPO Actor-Critic] -> [Stock Selection]

Training schedule:
  - Train:    2006-2021 (~180 months)
  - Validate: 2022-2024 (~36 months)
  - Test:     2025-2026 (~17 months, out-of-sample)

Principles (为什么选择PPO + LLM编码):

  1. 为什么 PPO？
     - PPO 使用 Clipped Surrogate Objective, 限制每次策略更新的幅度
     - 相比 DQN，PPO 天然支持连续动作空间（股票评分/权重）
     - 相比 A2C/A3C，PPO 更稳定，超参数敏感度低
     - On-policy 算法在金融数据中过拟合风险较低（每轮用新数据）

  2. 为什么 LLM 编码？
     - 市场不仅是数字，金融文本（新闻、财报、公告）包含丰富语义信息
     - LLM 可以将非结构化文本转化为稠密向量，与量价特征融合
     - FinBERT 等金融预训练模型能捕捉情感、风险、增长等维度
     - 即使使用 Mock 编码器（PCA on industry + fundamentals），也有增量信息

  3. 两者如何结合？
     - LLM 作为"感知模块"：将文本转化为语义向量（冻结，不做RL更新）
     - PPO 作为"决策模块"：接收融合后的状态，输出选股动作
     - 状态 = [量价特征 | LLM嵌入 | 市场特征] 拼接
     - 这种分离设计使得LLM可以独立升级，不影响RL训练

Runtime Requirements:
  - Python 3.8+
  - pandas, numpy, matplotlib, scikit-learn
  - torch >= 1.10 (installed as sb3 dependency)
  - stable-baselines3 >= 2.0
  - gymnasium >= 0.29
  - sentence-transformers (optional, for real LLM mode)
  - RAM: 8GB+ (16GB recommended for full dataset)
  - GPU: Optional (CPU training ~2-4 hours for 100k steps)

Usage:
  python train.py                           # Full pipeline with mock LLM
  python train.py --mode mock               # Mock LLM encoder (no GPU)
  python train.py --mode real               # Real sentence-transformers encoder
  python train.py --total-timesteps 200000  # Train longer
  python train.py --reward-type sharpe      # Sharpe ratio reward
  python train.py --no-train --model-path ./output/models/best_model  # Eval only
"""

import os
import sys
import argparse
import json
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_stock_selector.features import (
    load_stock_data,
    filter_stock_universe,
    compute_market_features,
    prepare_features,
    get_date_splits,
    DEFAULT_FEATURES,
    DERIVED_FEATURES,
)
from rl_stock_selector.llm_encoder import get_encoder
from rl_stock_selector.env import StockSelectionEnv
from rl_stock_selector.agent import (
    PPOTrainer,
    run_episode,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="RL + LLM Stock Selection Training"
    )
    parser.add_argument("--mode", type=str, default="mock",
                        choices=["mock", "real"],
                        help="LLM encoder mode")
    parser.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence-transformers model (real mode)")
    parser.add_argument("--total-timesteps", type=int, default=100_000,
                        help="Total training timesteps")
    parser.add_argument("--reward-type", type=str, default="excess_return",
                        choices=["raw_return", "excess_return", "sharpe", "sortino"])
    parser.add_argument("--select-stock-num", type=int, default=5, help="K stocks to select")
    parser.add_argument("--universe-size", type=int, default=100, help="Stocks in universe")
    parser.add_argument("--embedding-dim", type=int, default=64, help="Mock LLM embedding dim")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--csv-path", type=str, default=None, help="Path to stock_data.csv")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-train", action="store_true", help="Skip training, eval only")
    parser.add_argument("--model-path", type=str, default=None, help="Path to pre-trained model")
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds."""
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_backtest_results(env, model, split_name: str) -> pd.DataFrame:
    """Run backtest and return results DataFrame."""
    portfolio_values, step_infos = run_episode(model, env, deterministic=True)

    records = []
    for info in step_infos:
        records.append({
            "date": info.get("date", ""),
            "portfolio_value": info.get("portfolio_value", 1.0),
            "portfolio_return": info.get("portfolio_return", 0.0),
            "n_selected": info.get("n_selected", 0),
            "turnover": info.get("turnover", 0.0),
        })

    results = pd.DataFrame(records)
    results["split"] = split_name
    return results


def compute_metrics(results: pd.DataFrame) -> dict:
    """Compute performance metrics from backtest results."""
    if results.empty:
        return {}

    returns = results["portfolio_return"].values
    cumulative = results["portfolio_value"].values
    n_periods = len(returns)

    if n_periods == 0:
        return {}

    total_return = cumulative[-1] - 1.0
    annual_return = (cumulative[-1]) ** (12.0 / max(n_periods, 1)) - 1.0
    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))
    sharpe = mean_ret / (std_ret + 1e-8) * np.sqrt(12)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / (peak + 1e-8)
    max_dd = float(np.min(drawdown))
    win_rate = float(np.mean(returns > 0))
    calmar = annual_return / (abs(max_dd) + 1e-8)

    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "calmar_ratio": float(calmar),
        "n_periods": n_periods,
        "mean_monthly_return": mean_ret,
        "std_monthly_return": std_ret,
    }


def plot_results(train_results, val_results, test_results, all_metrics, output_dir):
    """Plot training curves and backtest results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Portfolio value curves
    ax = axes[0, 0]
    offset = 0
    if not train_results.empty:
        vals = train_results["portfolio_value"].values
        ax.plot(range(offset, offset + len(vals)), vals, label="Train", lw=1, alpha=0.7)
        offset += len(vals)
    if not val_results.empty:
        vals = val_results["portfolio_value"].values
        ax.plot(range(offset, offset + len(vals)), vals, label="Validation", lw=1.5, alpha=0.9)
        offset += len(vals)
    if not test_results.empty:
        vals = test_results["portfolio_value"].values
        ax.plot(range(offset, offset + len(vals)), vals, label="Test", lw=2)
    ax.axhline(y=1.0, color="gray", ls="--", lw=0.5)
    ax.set_title("Portfolio Value (Cumulative)")
    ax.set_xlabel("Months"); ax.set_ylabel("Value")
    ax.legend(); ax.grid(True, alpha=0.3)

    # 2. Monthly returns distribution
    ax = axes[0, 1]
    all_returns = pd.concat([train_results, val_results, test_results])
    if "portfolio_return" in all_returns.columns:
        rets = all_returns["portfolio_return"].dropna().values
        ax.hist(rets, bins=50, edgecolor="white", alpha=0.7, density=True)
        ax.axvline(x=0, color="red", ls="--", lw=0.8)
        ax.axvline(x=np.mean(rets), color="blue", ls="-", lw=1.5,
                   label=f"Mean={np.mean(rets):.4f}")
        ax.set_title("Monthly Return Distribution")
        ax.set_xlabel("Return"); ax.set_ylabel("Density")
        ax.legend(); ax.grid(True, alpha=0.3)

    # 3. Drawdown curve (test set)
    ax = axes[0, 2]
    if not test_results.empty:
        cum = test_results["portfolio_value"].values
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / (peak + 1e-8) * 100
        ax.fill_between(range(len(dd)), dd, 0, alpha=0.3, color="red")
        ax.plot(range(len(dd)), dd, color="darkred", lw=1)
        ax.set_title(f"Test Drawdown (Max DD: {abs(dd.min()):.1f}%)")
        ax.set_xlabel("Months"); ax.set_ylabel("Drawdown %")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="gray", ls="--", lw=0.5)

    # 4. Metrics bar chart
    ax = axes[1, 0]
    splits = ["Train", "Validation", "Test"]
    met_keys = ["annual_return", "sharpe_ratio", "max_drawdown"]
    met_labels = ["Annual Return", "Sharpe Ratio", "Max Drawdown"]
    colors = ["green", "blue", "red"]
    x = np.arange(len(splits)); width = 0.2
    for i, (key, label, color) in enumerate(zip(met_keys, met_labels, colors)):
        vals = [all_metrics.get(s, {}).get(key, 0) for s in splits]
        if key in ("annual_return", "max_drawdown"):
            vals = [v * 100 for v in vals]
        ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.7)
    ax.set_title("Metrics by Split")
    ax.set_xticks(x + width); ax.set_xticklabels(splits)
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    # 5. Rolling Sharpe (12-month)
    ax = axes[1, 1]
    combined = pd.concat([train_results, val_results, test_results])
    if "portfolio_return" in combined.columns:
        def rolling_sharpe(x):
            return np.mean(x) / (np.std(x) + 1e-8) * np.sqrt(12)
        roll_sharpe = combined["portfolio_return"].rolling(12, min_periods=6).apply(rolling_sharpe)
        ax.plot(roll_sharpe.values, color="darkblue", lw=1)
        ax.axhline(y=0, color="gray", ls="--", lw=0.5)
        ax.set_title("Rolling 12-Month Sharpe")
        ax.set_xlabel("Months"); ax.set_ylabel("Sharpe")
        ax.grid(True, alpha=0.3)

    # 6. Cumulative returns (log scale)
    ax = axes[1, 2]
    if "portfolio_return" in combined.columns:
        cum_ret = (1 + combined["portfolio_return"].fillna(0)).cumprod().values
        ax.semilogy(cum_ret, label="RL Strategy", lw=1.5)
        ax.axhline(y=1.0, color="gray", ls="--", lw=0.5, label="Baseline (1.0)")
        ax.set_title("Cumulative Returns (Log Scale)")
        ax.set_xlabel("Months"); ax.set_ylabel("Cumulative Return")
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "backtest_results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Results plot saved to {plot_path}")
    return plot_path


def main():
    """Main training and evaluation pipeline."""
    args = parse_args()

    # Setup
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  RL + LLM Stock Selection (Plan A)")
    print("  LLM as State Encoder + PPO RL Agent")
    print("=" * 70)
    print(f"  Mode:           {args.mode}")
    print(f"  Reward type:    {args.reward_type}")
    print(f"  Universe size:  {args.universe_size}")
    print(f"  Select K:       {args.select_stock_num}")
    print(f"  Timesteps:      {args.total_timesteps}")
    print(f"  Output dir:     {args.output_dir}")
    print("=" * 70)

    # ── Step 1: Load and prepare data ──────────────────────────────────────
    print("\n[1/5] Loading data...")
    df = load_stock_data(args.csv_path)
    print(f"  Loaded {len(df):,} rows, {df['交易日期'].nunique()} dates, "
          f"{df['股票代码'].nunique()} stocks")

    # Split dates
    all_dates = sorted(df["交易日期"].unique())
    train_dates, val_dates, test_dates = get_date_splits(
        np.array(all_dates),
        train_end="2021-12-31",
        val_end="2024-12-31",
    )
    print(f"  Train: {len(train_dates)} months ({train_dates[0]} to {train_dates[-1]})")
    print(f"  Val:   {len(val_dates)} months ({val_dates[0]} to {val_dates[-1]})")
    print(f"  Test:  {len(test_dates)} months ({test_dates[0]} to {test_dates[-1]})")

    # ── Step 2: Feature engineering ────────────────────────────────────────
    print("\n[2/5] Engineering features...")
    df = filter_stock_universe(df, min_listing_days=250, exclude_bj=True)

    # Compute market features once on the full dataset so mkt_cum is
    # historically continuous across train/val/test splits.
    mkt_feat_all = compute_market_features(df)

    # Fit on training data only
    train_df = df[df["交易日期"].isin(train_dates)].copy()
    train_df, norm_cols, scaler_dict, mkt_feat_train = prepare_features(
        train_df, scaler_type="robust", top_n_stocks=args.universe_size,
        market_features_df=mkt_feat_all,
    )
    print(f"  Train features: {len(norm_cols)} normalized, shape: {train_df.shape}")

    # Apply same transforms to val
    val_df = df[df["交易日期"].isin(val_dates)].copy()
    val_df = val_df[val_df["股票代码"].isin(train_df["股票代码"].unique())]
    val_df, _, _, mkt_feat_val = prepare_features(
        val_df, feature_cols=list(DEFAULT_FEATURES),
        scaler_type="robust", scaler_dict=scaler_dict,
        top_n_stocks=args.universe_size,
        market_features_df=mkt_feat_all,
    )

    # Apply same transforms to test
    test_df = df[df["交易日期"].isin(test_dates)].copy()
    test_df = test_df[test_df["股票代码"].isin(train_df["股票代码"].unique())]
    test_df, _, _, mkt_feat_test = prepare_features(
        test_df, feature_cols=list(DEFAULT_FEATURES),
        scaler_type="robust", scaler_dict=scaler_dict,
        top_n_stocks=args.universe_size,
        market_features_df=mkt_feat_all,
    )

    print(f"  Train samples: {len(train_df):,}")
    print(f"  Val samples:   {len(val_df):,}")
    print(f"  Test samples:  {len(test_df):,}")

    # ── Step 3: LLM Encoder ────────────────────────────────────────────────
    print("\n[3/5] Setting up LLM encoder...")
    llm_encoder = get_encoder(
        mode=args.mode,
        embedding_dim=args.embedding_dim,
        model_name=args.model_name,
        device="cpu",
    )
    if args.mode == "mock":
        llm_encoder.fit(train_df)
        print(f"  Mock encoder: {llm_encoder.embedding_dim}-dim embeddings (fitted)")
    else:
        print(f"  Real encoder: {args.model_name}")

    # ── Step 4: Create environments ────────────────────────────────────────
    print("\n[4/5] Creating RL environments...")

    # Build normalized feature column names
    all_raw_features = list(DEFAULT_FEATURES) + DERIVED_FEATURES
    feature_cols = [f"{c}_norm" for c in all_raw_features
                    if f"{c}_norm" in train_df.columns]
    print(f"  Using {len(feature_cols)} normalized features")

    env_train = StockSelectionEnv(
        df=train_df,
        feature_cols=feature_cols,
        market_features_df=mkt_feat_train,
        llm_encoder=llm_encoder,
        select_stock_num=args.select_stock_num,
        universe_size=args.universe_size,
        reward_type=args.reward_type,
    )

    env_val = StockSelectionEnv(
        df=val_df,
        feature_cols=feature_cols,
        market_features_df=mkt_feat_val,
        llm_encoder=llm_encoder,
        select_stock_num=args.select_stock_num,
        universe_size=args.universe_size,
        reward_type=args.reward_type,
    )

    env_test = StockSelectionEnv(
        df=test_df,
        feature_cols=feature_cols,
        market_features_df=mkt_feat_test,
        llm_encoder=llm_encoder,
        select_stock_num=args.select_stock_num,
        universe_size=args.universe_size,
        reward_type=args.reward_type,
    )

    print(f"  Observation dim: {env_train.obs_dim}")
    print(f"  Action dim:      {env_train.action_space.shape[0]}")

    # ── Step 5: Train or load model ────────────────────────────────────────
    print("\n[5/5] Training / Evaluation...")

    if args.no_train and args.model_path:
        from stable_baselines3 import PPO
        model = PPO.load(args.model_path, env=env_train)
        print(f"  Loaded model from {args.model_path}")
    else:
        model_dir = os.path.join(args.output_dir, "models")
        log_dir = os.path.join(args.output_dir, "logs")

        trainer = PPOTrainer(
            env=env_train,
            model_config={
                "seed": args.seed,
                "verbose": 1,
            },
            model_dir=model_dir,
            log_dir=log_dir,
        )

        model = trainer.train(
            total_timesteps=args.total_timesteps,
            eval_env=env_val,
            eval_freq=10_000,
            progress_bar=True,
        )
        print("  Training complete!")

    # ── Backtest evaluation ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Backtest Results")
    print("=" * 70)

    train_results = build_backtest_results(env_train, model, "Train")
    val_results = build_backtest_results(env_val, model, "Validation")
    test_results = build_backtest_results(env_test, model, "Test")

    all_metrics = {
        "Train": compute_metrics(train_results),
        "Validation": compute_metrics(val_results),
        "Test": compute_metrics(test_results),
    }

    for split, metrics in all_metrics.items():
        print(f"\n  {split}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.4f}")
            else:
                print(f"    {key}: {value}")

    # ── Save results ───────────────────────────────────────────────────────
    all_results = pd.concat([train_results, val_results, test_results],
                            ignore_index=True)
    results_path = os.path.join(args.output_dir, "backtest_results.csv")
    all_results.to_csv(results_path, index=False, encoding="gbk")
    print(f"\n  Results saved to {results_path}")

    metrics_serializable = {}
    for split, metrics in all_metrics.items():
        metrics_serializable[split] = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in metrics.items()
        }
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_serializable, f, indent=2, default=str)
    print(f"  Metrics saved to {metrics_path}")

    # ── Plot ───────────────────────────────────────────────────────────────
    plot_path = plot_results(train_results, val_results, test_results,
                             all_metrics, args.output_dir)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    test_m = all_metrics.get("Test", {})
    print(f"  Test Annual Return:  {test_m.get('annual_return', 0):.2%}")
    print(f"  Test Sharpe Ratio:   {test_m.get('sharpe_ratio', 0):.2f}")
    print(f"  Test Max Drawdown:   {test_m.get('max_drawdown', 0):.2%}")
    print(f"  Test Win Rate:       {test_m.get('win_rate', 0):.2%}")
    print(f"  Test Calmar Ratio:   {test_m.get('calmar_ratio', 0):.2f}")
    print(f"\n  Output directory: {args.output_dir}")
    print("  Done!")


if __name__ == "__main__":
    main()
