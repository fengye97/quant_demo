"""
LLM + RL 模型定义 — Plan A: LLM 作为状态编码器 + PPO 策略网络

架构:
    ┌──────────────────────────────────────────────────┐
    │  输入层                                          │
    │  ├── 量价特征 [30维] ──► MLP Encoder ──► 128维   │
    │  ├── 技术指标 [7维]  ──► MLP Encoder ──► 64维    │
    │  ├── 基本面   [4维]  ──► MLP Encoder ──► 64维    │
    │  ├── 市场特征 [5维]  ──► MLP Encoder ──► 32维    │
    │  └── 文本特征 [768维] ──► LLM Encoder ──► 256维 (可选) │
    │                                                    │
    │  ┌─────────────────────────────────────────────┐  │
    │  │  State Fusion Layer                         │  │
    │  │  Cross-Attention → 256维 unified state      │  │
    │  └─────────────────────────────────────────────┘  │
    │                                                    │
    │  ┌──────────────┐    ┌──────────────────────┐    │
    │  │  PPO Actor    │    │  PPO Critic          │    │
    │  │  输出: 选股权重│    │  输出: V(s) 状态价值  │    │
    │  └──────────────┘    └──────────────────────┘    │
    └──────────────────────────────────────────────────┘

原理:
    1. LLM Encoder: 使用预训练的 FinBERT/Qwen-Fin 提取金融文本的语义特征
       将新闻/公告/研报文本转化为稠密向量，捕捉价格数据之外的信息维度
    2. State Fusion: 跨模态注意力融合，确保不同来源的特征能够交互
    3. PPO Actor: 学习最优选股策略 π(a|s)，最大化期望累积奖励
    4. PPO Critic: 估计状态价值 V(s)，用于优势函数计算

注意:
    - LLM Encoder 需要 torch 和 transformers 库
    - 若 LLM 不可用，使用 Pure MLP Encoder 代替 (降级方案)
    - 所有网络均支持 GPU 加速 (CUDA/MPS)
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
import warnings

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════
# Pure MLP Encoder (降级方案: 不依赖LLM)
# ═══════════════════════════════════════════════════════════════════════════

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:

    class MLPEncoder(nn.Module):
        """
        纯 MLP 状态编码器 (替代 LLM)

        当 LLM 不可用时使用，将原始特征映射为统一状态表示
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 256,
            output_dim: int = 256,
            n_layers: int = 3,
            dropout: float = 0.1,
        ):
            super().__init__()
            layers = []
            in_dim = input_dim
            for i in range(n_layers):
                out_dim = hidden_dim if i < n_layers - 1 else output_dim
                layers.append(nn.Linear(in_dim, out_dim))
                if i < n_layers - 1:
                    layers.append(nn.LayerNorm(out_dim))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout))
                in_dim = out_dim
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


    class LLMStateEncoder(nn.Module):
        """
        LLM 状态编码器 (Plan A 核心)

        设计思路:
        - 价格特征通过 MLP 编码为 price_embed
        - 文本特征通过 LLM 编码为 text_embed (或降级为 MLP)
        - Cross-Attention 融合: price_embed 查询 text_embed
        - 拼接后映射为统一状态

        注意:
        - LLM 部分的推理是 frozen (不参与梯度更新)，降低计算成本
        - 可在 Phase 2 中替换为 FinBERT/Qwen-Fin 的真实 embedding
        """

        def __init__(
            self,
            price_feature_dim: int = 30,
            tech_feature_dim: int = 7,
            fundamental_dim: int = 4,
            market_dim: int = 5,
            text_embed_dim: int = 768,
            unified_dim: int = 256,
            dropout: float = 0.1,
            use_llm: bool = False,
            llm_model_name: str = "ProsusAI/finBERT",
        ):
            super().__init__()
            self.use_llm = use_llm
            self.unified_dim = unified_dim

            # 子编码器
            self.price_encoder = MLPEncoder(price_feature_dim, 128, 128, n_layers=2)
            self.tech_encoder = MLPEncoder(tech_feature_dim, 64, 64, n_layers=2)
            self.fund_encoder = MLPEncoder(fundamental_dim, 64, 64, n_layers=2)
            self.market_encoder = MLPEncoder(market_dim, 32, 32, n_layers=1)

            if use_llm:
                # LLM 编码器 (frozen pretrained)
                try:
                    from transformers import AutoModel, AutoTokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
                    self.llm = AutoModel.from_pretrained(llm_model_name)
                    for param in self.llm.parameters():
                        param.requires_grad = False  # Freeze LLM
                    self.llm_output_dim = self.llm.config.hidden_size
                except ImportError:
                    print("Warning: transformers not available, falling back to MLP text encoder")
                    self.use_llm = False
                    self.text_encoder = MLPEncoder(text_embed_dim, 256, 256, n_layers=2)
            else:
                # 降级: MLP 文本编码器
                self.text_encoder = MLPEncoder(text_embed_dim, 256, 256, n_layers=2)

            # Cross-Attention Fusion
            concat_dim = 128 + 64 + 64 + 32 + 256  # = 544
            self.fusion = nn.Sequential(
                nn.Linear(concat_dim, unified_dim * 2),
                nn.LayerNorm(unified_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(unified_dim * 2, unified_dim),
                nn.LayerNorm(unified_dim),
                nn.ReLU(),
            )

        def forward(
            self,
            price_features: torch.Tensor,     # (batch, 30)
            tech_features: torch.Tensor,       # (batch, 7)
            fund_features: torch.Tensor,       # (batch, 4)
            market_features: torch.Tensor,     # (batch, 5)
            text_features: Optional[torch.Tensor] = None,  # (batch, 768) or None
        ) -> torch.Tensor:
            """
            编码状态

            Returns:
                unified_state: (batch, unified_dim)
            """
            p_embed = self.price_encoder(price_features)
            t_embed = self.tech_encoder(tech_features)
            f_embed = self.fund_encoder(fund_features)
            m_embed = self.market_encoder(market_features)

            if text_features is not None and self.use_llm:
                # LLM encoding
                with torch.no_grad():
                    txt_embed = self.llm(**text_features).last_hidden_state[:, 0, :]
            elif text_features is not None:
                txt_embed = self.text_encoder(text_features)
            else:
                # 无文本特征: 使用 zero embedding
                txt_embed = torch.zeros(
                    price_features.size(0), 256,
                    device=price_features.device
                )

            # 拼接所有 embedding
            unified = torch.cat([p_embed, t_embed, f_embed, m_embed, txt_embed], dim=-1)
            return self.fusion(unified)


    class PPOModel(nn.Module):
        """
        PPO Actor-Critic 模型

        Actor (Policy Network):
            输入: unified_state (256维)
            输出: 动作分布 (选股概率)

        Critic (Value Network):
            输入: unified_state (256维)
            输出: 状态价值 V(s)
        """

        def __init__(
            self,
            state_dim: int = 256,
            action_dim: int = 6,  # top_k
            n_stocks: int = 100,  # 股票池大小上限
            hidden_dim: int = 256,
        ):
            super().__init__()
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.n_stocks = n_stocks

            # 共享特征提取层
            self.shared = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )

            # Actor: 输出每只股票的"吸引力"分数
            self.actor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_stocks),
            )

            # Critic: 输出状态价值
            self.critic = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

            # 初始化权重
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.constant_(m.bias, 0.0)

        def forward(
            self, state: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Parameters
            ----------
            state: (batch, state_dim)

            Returns
            -------
            action_logits: (batch, n_stocks) — 每只股票的 logits
            value: (batch, 1) — 状态价值
            """
            shared_features = self.shared(state)
            action_logits = self.actor(shared_features)
            value = self.critic(shared_features)
            return action_logits, value

        def get_action(
            self,
            state: torch.Tensor,
            deterministic: bool = False,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            采样动作

            Returns
            -------
            action_indices: (batch, action_dim) top_k 股票索引
            log_probs: (batch, action_dim) 对应 log probability
            value: (batch, 1) 状态价值
            """
            logits, value = self.forward(state)

            # Mask: 将 -inf 的 logits 设置为极小值
            logits = torch.where(
                torch.isfinite(logits),
                logits,
                torch.tensor(-1e9, device=logits.device),
            )

            # Softmax 得到概率分布
            probs = F.softmax(logits, dim=-1)

            if deterministic:
                # 确定性: 选 top_k
                _, top_indices = torch.topk(logits, k=self.action_dim, dim=-1)
                # 获取对应 log_probs
                log_probs = F.log_softmax(logits, dim=-1)
                selected_log_probs = torch.gather(log_probs, dim=-1, index=top_indices)
                return top_indices, selected_log_probs, value
            else:
                # 随机采样: 从概率分布中不放回采样 top_k
                # 使用 Gumbel-Softmax 近似或逐次采样
                batch_size = state.size(0)
                action_indices = torch.zeros(
                    batch_size, self.action_dim, dtype=torch.long, device=state.device
                )
                log_probs = torch.zeros(
                    batch_size, self.action_dim, device=state.device
                )

                for i in range(batch_size):
                    remaining_probs = probs[i].clone()
                    for k in range(self.action_dim):
                        dist = Categorical(remaining_probs)
                        idx = dist.sample()
                        action_indices[i, k] = idx
                        log_probs[i, k] = dist.log_prob(idx)
                        remaining_probs[idx] = 0.0
                        remaining_probs = remaining_probs / (remaining_probs.sum() + 1e-8)

                return action_indices, log_probs, value

        def evaluate_action(
            self,
            state: torch.Tensor,
            action_indices: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            评估动作 (用于 PPO 更新)

            Parameters
            ----------
            state: (batch, state_dim)
            action_indices: (batch, action_dim) 已执行的动作

            Returns
            -------
            log_probs: (batch, action_dim)
            values: (batch, 1)
            entropy: (batch, 1)
            """
            logits, value = self.forward(state)
            logits = torch.where(
                torch.isfinite(logits),
                logits,
                torch.tensor(-1e9, device=logits.device),
            )

            probs = F.softmax(logits, dim=-1)
            log_probs_all = F.log_softmax(logits, dim=-1)

            # Gather log_probs for selected actions
            log_probs = torch.gather(log_probs_all, dim=-1, index=action_indices)

            # Entropy for exploration bonus
            entropy = -(probs * log_probs_all).sum(dim=-1, keepdim=True)

            return log_probs, value, entropy


else:
    # torch 不可用时的占位定义
    MLPEncoder = None
    LLMStateEncoder = None
    PPOModel = None
