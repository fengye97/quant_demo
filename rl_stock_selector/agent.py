"""
PPO Agent for Stock Selection

Implements a PPO (Proximal Policy Optimization) agent for monthly stock
selection. Uses stable-baselines3's PPO implementation as the core RL algorithm.

Architecture:
  - Actor: Outputs stock scores (which stocks to select)
  - Critic: Estimates state value for advantage computation
  - PPO: Clipped surrogate objective for stable training

Key hyperparameters:
  - learning_rate: Policy network learning rate
  - n_steps: Steps per rollout before update
  - batch_size: Mini-batch size for PPO update
  - n_epochs: Number of epochs per PPO update
  - gamma: Discount factor
  - gae_lambda: GAE lambda for advantage estimation
  - clip_range: PPO clipping epsilon
  - ent_coef: Entropy coefficient for exploration

References:
  - Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
  - Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Callable, List, Tuple
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


# ── Default PPO hyperparameters ────────────────────────────────────────────

DEFAULT_PPO_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "normalize_advantage": True,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "use_sde": False,
    "sde_sample_freq": -1,
    "target_kl": None,
    "tensorboard_log": None,
    "policy_kwargs": None,
    "verbose": 1,
    "seed": 42,
    "device": "auto",
}


def create_ppo_model(
    env,
    config: Optional[Dict[str, Any]] = None,
    tensorboard_log: Optional[str] = None,
) -> "PPO":
    """Create a PPO model with default or custom configuration.

    Parameters
    ----------
    env : gym.Env
        The stock selection environment.
    config : dict, optional
        Override default PPO hyperparameters.
    tensorboard_log : str, optional
        Path for tensorboard logs.

    Returns
    -------
    stable_baselines3.PPO
        Configured PPO model.
    """
    from stable_baselines3 import PPO

    ppo_config = DEFAULT_PPO_CONFIG.copy()
    if config:
        ppo_config.update(config)

    if tensorboard_log:
        ppo_config["tensorboard_log"] = tensorboard_log

    # Build policy kwargs for custom network architecture
    # For stock selection, we want a wider network to handle large observations
    if ppo_config["policy_kwargs"] is None:
        import torch.nn as nn
        ppo_config["policy_kwargs"] = dict(
            net_arch=dict(
                pi=[256, 128],      # Policy network
                vf=[256, 128],      # Value network
            ),
            activation_fn=nn.ReLU,
            ortho_init=True,
        )

    model = PPO(
        policy=ppo_config["policy"],
        env=env,
        learning_rate=float(ppo_config["learning_rate"]),
        n_steps=int(ppo_config["n_steps"]),
        batch_size=int(ppo_config["batch_size"]),
        n_epochs=int(ppo_config["n_epochs"]),
        gamma=float(ppo_config["gamma"]),
        gae_lambda=float(ppo_config["gae_lambda"]),
        clip_range=float(ppo_config["clip_range"]),
        clip_range_vf=ppo_config["clip_range_vf"],
        normalize_advantage=ppo_config["normalize_advantage"],
        ent_coef=float(ppo_config["ent_coef"]),
        vf_coef=float(ppo_config["vf_coef"]),
        max_grad_norm=float(ppo_config["max_grad_norm"]),
        use_sde=ppo_config["use_sde"],
        sde_sample_freq=ppo_config["sde_sample_freq"],
        target_kl=ppo_config["target_kl"],
        tensorboard_log=ppo_config["tensorboard_log"],
        policy_kwargs=ppo_config["policy_kwargs"],
        verbose=int(ppo_config["verbose"]),
        seed=int(ppo_config["seed"]),
        device=ppo_config["device"],
    )

    return model


class PPOTrainer:
    """Training wrapper for PPO stock selection.

    Provides a clean training interface with:
      - Periodic evaluation callbacks
      - Model checkpointing
      - Training metrics logging
    """

    def __init__(
        self,
        env,
        model_config: Optional[Dict] = None,
        model_dir: str = "./models",
        log_dir: str = "./logs",
    ):
        """
        Parameters
        ----------
        env : gym.Env
            Training environment.
        model_config : dict, optional
            PPO hyperparameter overrides.
        model_dir : str
            Directory to save model checkpoints.
        log_dir : str
            Directory for tensorboard logs.
        """
        self.env = env
        self.model_config = model_config or {}
        self.model_dir = model_dir
        self.log_dir = log_dir

        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        self.model = None
        self.train_history = {
            "timesteps": [],
            "ep_rew_mean": [],
            "value_loss": [],
            "policy_gradient_loss": [],
            "explained_variance": [],
        }

    def train(
        self,
        total_timesteps: int = 100_000,
        eval_env=None,
        eval_freq: int = 10_000,
        save_freq: int = 50_000,
        progress_bar: bool = True,
    ) -> "PPO":
        """Train the PPO agent.

        Parameters
        ----------
        total_timesteps : int
            Total environment steps to train for.
        eval_env : gym.Env, optional
            Separate evaluation environment.
        eval_freq : int
            Evaluate every N timesteps.
        save_freq : int
            Save checkpoint every N timesteps.
        progress_bar : bool
            Show tqdm progress bar.

        Returns
        -------
        stable_baselines3.PPO
            Trained model.
        """
        from stable_baselines3.common.callbacks import (
            EvalCallback,
            StopTrainingOnNoModelImprovement,
        )
        from stable_baselines3.common.monitor import Monitor

        # Create model
        tensorboard_log = os.path.join(self.log_dir,
                                       datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.model = create_ppo_model(
            self.env,
            config=self.model_config,
            tensorboard_log=tensorboard_log,
        )

        # Setup callbacks
        callbacks = []

        if eval_env is not None:
            eval_env = Monitor(eval_env)
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=self.model_dir,
                log_path=self.log_dir,
                eval_freq=max(eval_freq, self.model.n_steps),
                deterministic=True,
                render=False,
            )
            callbacks.append(eval_callback)

        # Add custom metrics callback
        metrics_callback = _MetricsCallback(self)
        callbacks.append(metrics_callback)

        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks if callbacks else None,
            progress_bar=progress_bar,
        )

        # Save final model
        final_path = os.path.join(self.model_dir, "ppo_stock_selector_final")
        self.model.save(final_path)
        print(f"Model saved to {final_path}")

        return self.model

    def save(self, path: str):
        """Save model to path."""
        if self.model is not None:
            self.model.save(path)
            print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from path."""
        from stable_baselines3 import PPO
        self.model = PPO.load(path, env=self.env)
        print(f"Model loaded from {path}")

    def get_metrics(self) -> Dict[str, list]:
        """Get training metrics history."""
        return self.train_history


from stable_baselines3.common.callbacks import BaseCallback


class _MetricsCallback(BaseCallback):
    """Internal callback to track training metrics."""

    def __init__(self, trainer: "PPOTrainer"):
        super().__init__()
        self.trainer = trainer

    def _on_step(self) -> bool:
        """Called after each step."""
        if self.model is not None:
            try:
                self.trainer.train_history["timesteps"].append(
                    self.model.num_timesteps
                )
            except Exception:
                pass
        infos = self.locals.get("infos", [])
        for info in infos:
            if isinstance(info, dict) and "episode" in info:
                self.trainer.train_history["ep_rew_mean"].append(
                    info["episode"]["r"]
                )
        return True


def evaluate_model(
    model,
    env,
    n_episodes: int = 5,
    deterministic: bool = True,
    render: bool = False,
) -> Dict[str, Any]:
    """Evaluate a trained model over multiple episodes.

    Parameters
    ----------
    model : stable_baselines3.PPO
        Trained PPO model.
    env : gym.Env
        Evaluation environment.
    n_episodes : int
        Number of evaluation episodes.
    deterministic : bool
        Use deterministic actions if True.
    render : bool
        Render environment.

    Returns
    -------
    dict
        Evaluation metrics.
    """
    from stable_baselines3.common.evaluation import evaluate_policy

    mean_reward, std_reward = evaluate_policy(
        model, env,
        n_eval_episodes=n_episodes,
        deterministic=deterministic,
        render=render,
    )

    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "n_episodes": n_episodes,
    }


def run_episode(
    model,
    env,
    deterministic: bool = True,
) -> Tuple[List[float], List[Dict]]:
    """Run a single episode with detailed tracking.

    Returns portfolio values and per-step info.

    Parameters
    ----------
    model : stable_baselines3.PPO
        Trained model.
    env : gym.Env
        Environment.
    deterministic : bool
        Use deterministic actions.

    Returns
    -------
    portfolio_values : list of float
        Portfolio value at each step.
    step_infos : list of dict
        Info dict from each step.
    """
    obs, info = env.reset()
    done = False
    portfolio_values = [1.0]
    step_infos = []

    while not done:
        action, _states = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        portfolio_values.append(info.get("portfolio_value", portfolio_values[-1]))
        step_infos.append(info)

    return portfolio_values, step_infos
