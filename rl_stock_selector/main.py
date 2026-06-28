"""
RL Stock Selector — 命令行入口

Usage:
    python3 -m rl_stock_selector --mode train
    python3 -m rl_stock_selector --mode backtest --model ppo_model.pt
    python3 -m rl_stock_selector --mode full

以模块方式运行:
    cd /path/to/quant
    python3 -m rl_stock_selector.main --mode train
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    import torch
    HAS_TORCH = True
except ImportError:
    print("ERROR: torch is required. Install with: pip install torch")
    sys.exit(1)

from rl_stock_selector.environment import StockSelectionEnv
from rl_stock_selector.models import PPOModel
from rl_stock_selector.train import PPOTrainer
from rl_stock_selector.backtest import run_backtest


def get_data_path() -> str:
    """获取数据文件路径"""
    # 尝试多个可能路径
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "stock_trade_demo", "stock_data.csv"),
        os.path.join(os.getcwd(), "stock_trade_demo", "stock_data.csv"),
        os.path.join(os.getcwd(), "stock_data.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    raise FileNotFoundError(
        "stock_data.csv not found. Please ensure the file exists in stock_trade_demo/ "
        "or current directory."
    )


def load_data(data_path: str) -> pd.DataFrame:
    """加载股票数据"""
    print(f"[Data] Loading {data_path} ...")
    df = pd.read_csv(data_path, encoding="gbk", parse_dates=["交易日期"], low_memory=False)
    print(f"[Data] Loaded {len(df)} rows, {df['股票代码'].nunique()} stocks, "
          f"{df['交易日期'].nunique()} dates")
    print(f"[Data] Date range: {df['交易日期'].min()} ~ {df['交易日期'].max()}")
    return df


def train_mode(args, df: pd.DataFrame):
    """训练模式"""
    print("\n" + "=" * 60)
    print("  RL Stock Selector — Training Mode")
    print("=" * 60)

    # 创建环境
    env = StockSelectionEnv(
        df,
        lookback=args.lookback,
        top_k=args.top_k,
    )
    print(f"[Env] State dim: {env.state_dim}, Action dim: {env.action_dim}")
    print(f"[Env] Date range: {env.dates[0]} ~ {env.dates[-1]} ({len(env.dates)} months)")

    # 创建模型
    n_stocks_per_month = df.groupby("交易日期")["股票代码"].nunique().max()
    print(f"[Model] Max stocks per month: {n_stocks_per_month}")

    model = PPOModel(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_stocks=n_stocks_per_month,
        hidden_dim=args.hidden_dim,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # 创建训练器
    trainer = PPOTrainer(
        env=env,
        model=model,
        lr=args.lr,
        clip_range=args.clip_range,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        device=args.device,
    )

    # 训练
    save_path = args.save or "ppo_stock_selector.pt"
    history = trainer.train(
        total_timesteps=args.steps,
        steps_per_iteration=args.steps_per_iter,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_path=save_path,
    )

    print(f"\n[Train] Model saved to {save_path}")
    return model, env


def backtest_mode(args, df: pd.DataFrame):
    """回测模式"""
    print("\n" + "=" * 60)
    print("  RL Stock Selector — Backtest Mode")
    print("=" * 60)

    # 创建环境
    env = StockSelectionEnv(df, lookback=args.lookback, top_k=args.top_k)

    # 创建模型
    n_stocks_per_month = df.groupby("交易日期")["股票代码"].nunique().max()
    model = PPOModel(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_stocks=n_stocks_per_month,
        hidden_dim=args.hidden_dim,
    )

    # 加载权重
    model_path = args.model or "ppo_stock_selector.pt"
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        print("Please train first: python3 -m rl_stock_selector.main --mode train")
        return

    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[Model] Loaded from {model_path} (iter {checkpoint.get('iteration', '?')}, "
              f"reward {checkpoint.get('mean_reward', '?'):.4f})")
    else:
        model.load_state_dict(checkpoint)
        print(f"[Model] Loaded from {model_path}")

    # 运行回测
    result = run_backtest(model, env, df, deterministic=not args.stochastic)

    # 打印结果
    print(result.summary())

    # 保存结果
    if args.output:
        result_df = result.to_dataframe()
        result_df.to_csv(args.output, index=False, encoding="gbk")
        print(f"\n[Output] Backtest results saved to {args.output}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="RL Stock Selector — LLM + PPO for A-share Stock Selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 -m rl_stock_selector.main --mode train --steps 50000
  python3 -m rl_stock_selector.main --mode backtest --model ppo_stock_selector.pt
  python3 -m rl_stock_selector.main --mode full
        """,
    )

    parser.add_argument(
        "--mode", type=str, default="train",
        choices=["train", "backtest", "full"],
        help="Operation mode"
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to stock_data.csv"
    )

    # Environment
    parser.add_argument("--lookback", type=int, default=12, help="Lookback window (months)")
    parser.add_argument("--top-k", type=int, default=6, help="Number of stocks to select")

    # Model
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")

    # Training
    parser.add_argument("--steps", type=int, default=100000, help="Total training steps")
    parser.add_argument("--steps-per-iter", type=int, default=512, help="Steps per iteration")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--value-coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max gradient norm")
    parser.add_argument("--n-epochs", type=int, default=10, help="PPO epochs per update")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--log-interval", type=int, default=10, help="Log interval")
    parser.add_argument("--eval-interval", type=int, default=50, help="Eval interval")

    # Device
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"], help="Training device")

    # Save/Load
    parser.add_argument("--save", type=str, default=None, help="Model save path")
    parser.add_argument("--model", type=str, default=None, help="Model load path")
    parser.add_argument("--output", type=str, default=None, help="Backtest output CSV path")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy in backtest")

    args = parser.parse_args()

    # 加载数据
    data_path = args.data or get_data_path()
    df = load_data(data_path)

    if args.mode == "train":
        train_mode(args, df)
    elif args.mode == "backtest":
        backtest_mode(args, df)
    elif args.mode == "full":
        model, env = train_mode(args, df)
        print("\n" + "=" * 60)
        print("  Running backtest on trained model...")
        print("=" * 60)
        result = run_backtest(model, env, df, deterministic=not args.stochastic)
        print(result.summary())
        if args.output:
            result.to_dataframe().to_csv(args.output, index=False, encoding="gbk")


if __name__ == "__main__":
    main()
