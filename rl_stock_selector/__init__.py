"""
RL Stock Selector -- LLM + RL Stock Selection System (Plan A)

Architecture: LLM as State Encoder + PPO Reinforcement Learning Agent

Components:
    - features:     Data loading, feature engineering, normalization
    - llm_encoder:  Stock text encoding (Mock PCA / FinBERT sentence-transformers)
    - env:          Custom Gymnasium environment for monthly stock selection
    - agent:        PPO training wrapper (stable-baselines3)
    - train:        Main CLI entry point and pipeline orchestration

Usage:
    from rl_stock_selector.features import load_stock_data, prepare_features
    from rl_stock_selector.llm_encoder import get_encoder
    from rl_stock_selector.env import StockSelectionEnv
    from rl_stock_selector.agent import PPOTrainer, run_episode

    df = load_stock_data()
    train_df, norm_cols, scaler_dict, mkt_feat = prepare_features(train_df)
    encoder = get_encoder(mode='mock', embedding_dim=64); encoder.fit(train_df)
    env = StockSelectionEnv(train_df, norm_cols, mkt_feat, encoder)
    trainer = PPOTrainer(env); model = trainer.train(total_timesteps=100_000)
    values, infos = run_episode(model, env)

Author: Quant Research Team
Date: 2026-05-16
Version: 0.1.0
"""

__version__ = "0.1.0"
