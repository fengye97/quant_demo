# RL + LLM Stock Selection (Plan A)

基于强化学习（PPO）+ 大语言模型（LLM）的 A 股月度选股系统。

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                        Plan A Architecture                         │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────────┐     ┌──────────────────┐                    │
│  │ Financial Text    │     │ Price/Technical  │                    │
│  │ (行业, 基本面,    │     │ Data (bias, KDJ, │                    │
│  │  估值描述)        │     │  MACD, 波动率)   │                    │
│  └────────┬─────────┘     └────────┬─────────┘                    │
│           │                        │                              │
│           ▼                        ▼                              │
│  ┌──────────────────┐     ┌──────────────────┐                    │
│  │ LLM Encoder       │     │ Feature Engineering│                 │
│  │ (Mock/FinBERT)    │     │ (Normalize, Handle │                 │
│  │ → embedding vector│     │  NaN, Scale)       │                 │
│  └────────┬─────────┘     └────────┬─────────┘                    │
│           │                        │                              │
│           └────────┬───────────────┘                              │
│                    ▼                                              │
│           ┌──────────────────┐                                    │
│           │ Combined State    │  ← Market Features                │
│           │ [Stock Factors |  │                                    │
│           │  LLM Embed | Mkt] │                                    │
│           └────────┬─────────┘                                    │
│                    ▼                                              │
│           ┌──────────────────┐                                    │
│           │ PPO Actor-Critic │                                    │
│           │  Actor: stock     │                                    │
│           │  scores → top-K   │                                    │
│           │  Critic: V(state) │                                    │
│           └────────┬─────────┘                                    │
│                    ▼                                              │
│           ┌──────────────────┐                                    │
│           │ Action: Select K  │                                    │
│           │ stocks from N     │                                    │
│           └────────┬─────────┘                                    │
│                    ▼                                              │
│           ┌──────────────────┐                                    │
│           │ Reward: Next-month│                                   │
│           │ portfolio return  │                                    │
│           └──────────────────┘                                    │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

## Principles (原理说明)

### 为什么选择 PPO？

1. **Clipped Surrogate Objective（核心稳定机制）**

   ```
   L(θ) = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]
   ```

   其中 `r_t = π_new(a|s) / π_old(a|s)` 是新旧策略的概率比。clip 操作将更新限制在 [1-ε, 1+ε] 区间，防止策略崩溃。

   金融数据信噪比极低（~1-5%），如果不限制更新幅度，RL 极易过拟合噪音。

2. **连续动作空间支持**: 股票选择需要连续评分/权重，PPO 使用 Gaussian Policy 天然支持 Box 动作空间。对比：DQN 仅支持离散动作，SAC 超参数敏感。

3. **On-policy 在金融中的优势**: On-policy（PPO）用当前策略生成的数据更新，样本反映最新市场状态。Off-policy（DQN/SAC）的经验回放可能包含过时市场 regime。

4. **调参友好**: PPO 默认超参数（clip_range=0.2, gae_lambda=0.95）在广泛任务中表现稳定。

### 为什么使用 LLM 编码？

1. **非结构化信息提取**: 金融文本（新闻、财报、公告）包含情感、风险事件、增长预期等量价数据无法捕捉的信息。
2. **预训练知识**: FinBERT/Qwen-Fin 在海量金融文本上预训练，已学到金融语义关系和情感-市场关联。
3. **增量信息维度**: 即使使用 Mock 编码器（行业 + 基本面 PCA），也提供了量价因子之外的决策维度。

### 两者如何结合？

**分离式架构（方案 A）**:
- LLM 作为"感知模块"：参数冻结，将文本转化为语义向量
- PPO 作为"决策模块"：接收融合状态，输出选股动作
- 状态 = [Stock Factors | LLM Embed | Market Features]
- 优势：LLM 可独立升级，不影响 RL 训练；计算高效，适合月度调仓频率

## Runtime Requirements

```
Python >= 3.8
pandas >= 1.5, numpy >= 1.23
matplotlib >= 3.3, scikit-learn >= 1.2
torch >= 1.10 (auto-installed with sb3)
stable-baselines3 >= 2.0
gymnasium >= 0.29
sentence-transformers >= 3.0 (optional, real LLM mode only)
```

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| GPU | None (CPU training) | NVIDIA GPU (LLM inference) |
| Disk | 1 GB | 5 GB |
| CPU Training Time | ~2-4 hours (100k steps) | ~1 hour with GPU |

## Installation

```bash
# Install core dependencies
pip install stable-baselines3 gymnasium torch

# Optional: for real LLM encoder
pip install sentence-transformers
```

## Usage

### Quick Start (Mock LLM, no GPU needed)

```bash
cd /Users/fatcat/Desktop/quant
python rl_stock_selector/train.py
```

This will:
1. Load `stock_trade_demo/stock_data.csv` (703K rows, 55 columns)
2. Extract and normalize ~20 features per stock
3. Use mock LLM encoder (PCA on industry + fundamentals)
4. Train PPO for 100,000 steps
5. Evaluate on train (2006-2021), val (2022-2024), test (2025-2026) splits
6. Save results plots and metrics to `output/`

### Advanced Usage

```bash
# Train longer
python rl_stock_selector/train.py --total-timesteps 200000

# Use Sharpe ratio reward (risk-adjusted)
python rl_stock_selector/train.py --reward-type sharpe

# Select more stocks per period (top 10 instead of 5)
python rl_stock_selector/train.py --select-stock-num 10

# Larger universe (top 200 by market cap)
python rl_stock_selector/train.py --universe-size 200 --select-stock-num 10

# Real sentence-transformers encoder (English)
python rl_stock_selector/train.py --mode real

# Chinese BGE model for financial text
python rl_stock_selector/train.py --mode real --model-name BAAI/bge-large-zh-v1.5

# Evaluate pre-trained model only
python rl_stock_selector/train.py --no-train --model-path ./output/models/ppo_stock_selector_final.zip

# Custom data path
python rl_stock_selector/train.py --csv-path /path/to/stock_data.csv
```

### Programmatic Usage

```python
from rl_stock_selector.features import load_stock_data, prepare_features, get_date_splits
from rl_stock_selector.llm_encoder import get_encoder
from rl_stock_selector.env import StockSelectionEnv
from rl_stock_selector.agent import PPOTrainer, run_episode

# 1. Load and prepare data
df = load_stock_data()
all_dates = sorted(df['交易日期'].unique())
train_dates, val_dates, test_dates = get_date_splits(np.array(all_dates))

# 2. Feature engineering
train_df = df[df['交易日期'].isin(train_dates)]
train_df, norm_cols, scaler_dict, mkt_feat = prepare_features(train_df)

# 3. LLM Encoder
encoder = get_encoder(mode='mock', embedding_dim=64)
encoder.fit(train_df)

# 4. Create environment
env = StockSelectionEnv(
    df=train_df, feature_cols=norm_cols,
    market_features_df=mkt_feat, llm_encoder=encoder,
    select_stock_num=5, universe_size=100
)

# 5. Train
trainer = PPOTrainer(env)
model = trainer.train(total_timesteps=100_000)

# 6. Evaluate
values, infos = run_episode(model, env)
```

### Output Structure

```
output/
├── models/
│   ├── ppo_stock_selector_final.zip   # Final trained model
│   └── best_model.zip                 # Best model during training
├── logs/                              # Tensorboard logs
├── backtest_results.csv               # Per-month portfolio returns
├── backtest_results.png               # 6-panel performance plot
└── metrics.json                       # Summary metrics (JSON)
```

## Module Structure

```
rl_stock_selector/
├── __init__.py          # Package init, version info
├── features.py          # Data loading, feature engineering, normalization
├── llm_encoder.py       # LLM text encoder (Mock/FinBERT) + usage docs
├── env.py               # Custom Gym environment for stock selection
├── agent.py             # PPO agent (stable-baselines3 wrapper)
├── train.py             # Main training entry point (CLI + pipeline)
└── README.md            # This documentation
```

### Module Details

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `features.py` | Load CSV, filter universe, impute NaN, normalize features, split dates | `load_stock_data()`, `prepare_features()`, `get_date_splits()` |
| `llm_encoder.py` | Mock encoder (PCA on industry + fundamentals), real FinBERT encoder | `MockLLMEncoder`, `FinBERTEncoder`, `get_encoder()` |
| `env.py` | Gymnasium environment: monthly stock selection with observation/action/reward | `StockSelectionEnv(gym.Env)` |
| `agent.py` | PPO agent wrapper, training loop, evaluation utils | `PPOTrainer`, `run_episode()`, `create_ppo_model()` |
| `train.py` | CLI entry point, full pipeline orchestration, plotting, metrics | `main()` |

## Feature Engineering

### Features Used (20+ factors)

**Technical (13 factors)**:
- `bias_5`, `bias_10`, `bias_20` -- 乖离率：收盘价偏离均线百分比
- `振幅_5`, `振幅_10`, `振幅_20` -- 振幅：(高-低)/前收，N日均值
- `K`, `D`, `J` -- KDJ 随机指标
- `DIF`, `DEA`, `MACD` -- MACD 指标

**Volatility (6 factors)**:
- `涨跌幅std_5`, `涨跌幅std_10`, `涨跌幅std_20` -- 价格波动率
- `成交额std_5`, `成交额std_10`, `成交额std_20` -- 成交量波动率

**Fundamental (2 factors)**:
- `市盈率倒数` -- E/P 盈利收益率
- `市净率倒数` -- B/P 净资产收益率倒数

**Momentum (3 factors)**:
- `涨跌幅` -- 当月涨跌幅
- `涨跌幅_10`, `涨跌幅_20` -- 10/20日动量

**Derived (3 factors)**:
- `log_市值`, `log_成交额`, `log_流通市值`

### Normalization Pipeline

1. Handle NaN: per-stock forward/backward fill, then cross-section median
2. Scale: RobustScaler (median/IQR), clipped to [-5, 5]
3. Market features: equal-weight market mean, std, bullish %, MA12 regime signal

## PPO Training Configuration

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| learning_rate | 3e-4 | Adam learning rate |
| n_steps | 2048 | Steps collected per rollout |
| batch_size | 64 | Mini-batch size for PPO update |
| n_epochs | 10 | Update epochs per rollout |
| gamma | 0.99 | Discount factor |
| gae_lambda | 0.95 | GAE λ (bias-variance tradeoff) |
| clip_range | 0.2 | PPO clipping ε |
| ent_coef | 0.01 | Entropy bonus for exploration |

### Reward Functions

| Type | Formula | Use Case |
|------|---------|----------|
| `raw_return` | R = portfolio_return | Absolute return maximization |
| `excess_return` | R = portfolio_return - market_mean | Beat the benchmark |
| `sharpe` | R = mean(ret) / std(ret) * 0.1 | Rolling 12-month risk-adjusted |
| `sortino` | R = mean(ret) / std(neg_ret) * 0.1 | Downside-risk adjusted |

## LLM Encoder Modes

| Mode | Encoder | Embedding Dim | GPU | Quality |
|------|---------|--------------|-----|---------|
| `mock` | PCA on industry + fundamentals | 64 | No | Baseline (structured data only) |
| `real` (all-MiniLM-L6-v2) | SentenceTransformer | 384 | Optional | Good (English BERT) |
| `real` (bge-large-zh-v1.5) | BGE Chinese | 1024 | Recommended | Best (Chinese financial) |

### Mock Encoder Details

The mock encoder creates embeddings by:
1. One-hot encoding the Shenwan Level-1 industry classification
2. Concatenating with fundamental features (PE, PB, momentum, bias)
3. Applying PCA to reduce to target dimension (default: 64)
4. Normalizing to unit vectors

This provides meaningful structured embeddings without requiring GPU or LLM downloads.

### Real LLM Integration Guide

To use a real financial LLM encoder:

```python
from rl_stock_selector.llm_encoder import FinBERTEncoder, build_stock_descriptions

# Create text descriptions for each stock
descriptions = build_stock_descriptions(df, date='2024-12-31')
# Example: "股票浦发银行(sh600000)，所属银行行业，细分股份制商业银行，
#           盈利收益率0.0456，净资产收益率倒数0.2356，近10日上涨3.21%。"

# Encode with FinBERT
encoder = FinBERTEncoder(model_name="BAAI/bge-large-zh-v1.5", device="cuda")
embeddings = encoder.encode(descriptions)  # shape: (n_stocks, 1024)
```

For production use:
- Pre-compute embeddings for all stocks at each rebalance date
- Cache embeddings to avoid redundant LLM inference
- Batch encode for GPU efficiency (batch_size=32-128)

## Training Schedule

```
2006 ────────────────── 2021 ───────── 2024 ───── 2026
│                        │              │          │
└── Train (180 months) ──┘              │          │
                         └── Val (36m) ─┘          │
                                      └── Test (17m)┘
```

- **Train**: 2006-12 to 2021-12 (~180 months, 15 years)
- **Validation**: 2022-01 to 2024-12 (~36 months, 3 years)
- **Test**: 2025-01 to 2026-05 (~17 months, true out-of-sample)

## Expected Results and Limitations

### Expected Performance (backtest)

| Metric | Conservative | Optimistic |
|--------|-------------|------------|
| Annual Excess Return (vs equal-weight) | 3-8% | 8-15% |
| Sharpe Ratio (monthly) | 0.8-1.2 | 1.2-2.0 |
| Max Drawdown | 20-35% | 15-25% |
| Monthly Win Rate | 52-58% | 58-65% |
| Monthly Turnover | 30-60% | 20-40% |

### Limitations

1. **幸存者偏差 (Survivorship Bias)**: CSV 数据中缺少已退市股票记录，会高估收益
2. **过拟合风险 (Overfitting)**: PPO 有大量参数，可能在训练集发现虚假模式
3. **市场结构变化 (Regime Shift)**: 2006-2021 市场环境与 2025-2026 显著不同
4. **无真实文本数据**: Mock 编码器基于行业 + 基本面，非真实新闻/财报
5. **交易成本简化**: 仅考虑佣金 (1.2‱) + 印花税 (1‰)，未建模滑点、冲击成本
6. **A 股约束不完全**: 环境未完整建模 T+1 结算、涨跌停板、做空限制

### Improvement Roadmap

1. **Phase 1 (current)**: Mock LLM + PPO baseline on technical factors
2. **Phase 2**: Real FinBERT news encoding, add fundamental factors (ROE, cash flow)
3. **Phase 3**: Model A-share constraints (T+1, price limits), add risk budget
4. **Phase 4**: Multi-agent ensemble, meta-learning for regime adaptation
5. **Phase 5**: Live paper trading, incremental online learning

## References

- [FinRL: Deep RL Framework for Automated Stock Trading](https://arxiv.org/abs/2011.09607)
- [FinGPT: Open-Source Financial LLM](https://arxiv.org/abs/2306.06031)
- [PPO: Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [GAE: High-Dimensional Continuous Control](https://arxiv.org/abs/1506.02438)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [rl_llm_stock_selection_research.md](../rl_llm_stock_selection_research.md) -- Project research report
- [quant_factor.md](../quant_factor.md) -- Factor definitions and documentation
