# RL + LLM 强化学习选股方案调研报告

**日期**: 2026-05-16
**目标**: 调研基于强化学习进行选股的可行性，设计 LLM + RL 混合模型架构，评估开源方案

---

## 1. 执行摘要

基于强化学习（RL）进行股票选择是一个活跃且快速发展的研究方向。LLM + RL 的混合架构在技术上**完全可行**，且在多个维度上互补：LLM 擅长从非结构化金融文本中提取语义特征，RL 擅长在复杂、非平稳的市场环境中做序列决策。

**核心结论**: 建议采用 **LLM 作为状态编码器 + RL 作为策略网络** 的架构，分阶段实施。短期可基于 FinRL 等成熟框架快速搭建原型，中期引入 LLM 特征提取，长期构建端到端 LLM+RL 系统。

---

## 2. RL 选股技术现状

### 2.1 主流 RL 算法及适用性

| 算法 | 适用场景 | 优势 | 劣势 |
|------|---------|------|------|
| **DQN / Double DQN** | 离散动作空间（选 top-k 股票） | 样本效率高，训练稳定 | 仅支持离散动作 |
| **PPO** | 连续动作空间（组合权重分配） | 稳定性好，调参友好 | on-policy，样本效率低 |
| **A2C / A3C** | 连续动作空间 | 并行训练快 | 高方差 |
| **SAC** | 连续动作空间 | 探索能力强 | 超参数敏感 |
| **TD3** | 连续动作空间 | 减少 over-estimation | 调参复杂 |
| **Decision Transformer** | 离线 RL，基于历史数据 | 避免在线交互风险 | 分布偏移问题 |

### 2.2 关键挑战

1. **稀疏奖励（Sparse Rewards）**: 交易盈亏信号滞后且稀疏，需要精心设计奖励函数（Sharpe ratio, Calmar ratio, 信息比率）
2. **非平稳环境（Non-stationary）**: 市场分布持续变化（regime shift），需要在线适应或元学习
3. **高噪声信号比**: 金融数据信噪比极低（~1-5%），RL 容易过拟合噪声
4. **幸存者偏差**: 历史回测中退市股票被剔除，导致高估策略收益
5. **过拟合风险**: 参数空间大，数据量相对有限

### 2.3 动作空间设计

离散动作空间（适用于选股）:
- **Top-K 选择**: 从 N 只股票中选 K 只做多
- **分组选择**: 将股票分为买入/持有/卖出三组
- **排序选择**: 输出股票的相对排序

连续动作空间（适用于组合管理）:
- **权重向量**: [w1, w2, ..., wn] 满足 sum(wi) = 1
- **仓位管理**: 含现金仓位 [w1, w2, ..., wn, cash]

---

## 3. LLM + RL 混合架构设计

### 3.1 LLM 的角色定位

```
┌─────────────────────────────────────────────────────┐
│                   LLM + RL Pipeline                   │
├─────────────────────────────────────────────────────┤
│                                                       │
│  ┌──────────┐    ┌──────────────┐    ┌────────────┐ │
│  │ 金融文本  │───▶│ LLM Encoder   │───▶│  State     │ │
│  │ 新闻/公告 │    │ (特征提取)    │    │ Vector     │ │
│  │ 财报/研报 │    └──────────────┘    └─────┬──────┘ │
│  └──────────┘                               │        │
│                                              ▼        │
│  ┌──────────┐    ┌──────────────┐    ┌────────────┐ │
│  │ 行情数据  │───▶│ 特征工程     │───▶│  RL Policy  │ │
│  │ OHLCV    │    │ (价格+技术)  │    │  Network    │ │
│  │ 基本面   │    └──────────────┘    └─────┬──────┘ │
│  └──────────┘                               │        │
│                                              ▼        │
│                                       ┌────────────┐ │
│                                       │  Action     │ │
│                                       │  选股/权重  │ │
│                                       └────────────┘ │
│                                                       │
└─────────────────────────────────────────────────────┘
```

### 3.2 三种架构方案

#### 方案 A: LLM 作为状态编码器（推荐起步方案）
- **架构**: 金融文本 → LLM Embedding → 拼接价格特征 → RL Policy Network
- **LLM 选择**: FinBERT / BloombergGPT / Qwen-Fin / ChatGLM-Fin
- **RL 算法**: PPO（稳定性优先）
- **优点**: 轻量，LLM 仅做推理不需微调；可解释性强（embedding 可视化）
- **缺点**: LLM 不在优化回路中，可能提取无关特征

#### 方案 B: LLM 作为奖励模型
- **架构**: RL Agent 选股 → 模拟盘盈利 → LLM 评估选股理由质量 → 组合奖励
- **LLM 角色**: 评估选股逻辑是否合理（如"基于财报超预期买入"是否成立）
- **优点**: 引入常识判断，减少过拟合噪音
- **缺点**: LLM 评估本身可能不可靠，需要大量标注

#### 方案 C: 端到端 LLM Agent（前沿方案）
- **架构**: LLM 直接输出选股决策（text → action token），RL 微调 LLM 参数
- **技术路线**: RLHF / GRPO（类似 DeepSeek-R1 训练方式）
- **优点**: 统一模型，端到端优化
- **缺点**: 计算成本极高，目前无成功先例

### 3.3 推荐架构（方案 A+）

```
输入层:
├── 量价数据 (30+ 因子) ─► MLP Encoder ─► 128-dim vector
├── 基本面数据 (财报因子) ─► MLP Encoder ─► 64-dim vector
└── 文本数据 (新闻/公告) ─► LLM Embedding ─► 768-dim vector

融合层:
├── Cross-Attention Fusion
└── 256-dim unified state

RL 层:
├── PPO Actor (输出选股权重)
├── PPO Critic (估计状态价值)
└── 环境: Backtrader/Zipline/自定义回测
```

### 3.4 状态空间设计

| 特征类别 | 维度 | 说明 |
|---------|------|------|
| 价格特征 | ~30 | OHLCV, 均线, MACD, KDJ, Bollinger |
| 基本面特征 | ~15 | PE, PB, ROE, 营收增长, 现金流 |
| 文本特征 | ~768 | LLM embedding（新闻/研报摘要） |
| 市场特征 | ~10 | 指数涨跌, 行业表现, 资金流向 |
| 持仓特征 | ~N | 当前持仓权重向量 |

---

## 4. 可行性评估

### 4.1 技术可行性: ★★★★☆ (高)

- LLM 特征提取技术成熟（FinBERT 等已广泛验证）
- RL 选股有大量学术论文和开源实现支撑
- 组合方案的技术路径清晰

### 4.2 数据需求

| 数据类型 | 最低量级 | 推荐量级 | 获取难度 |
|---------|---------|---------|---------|
| 日线行情 | 5年 | 10年+ | 低（AKShare/TuShare） |
| 财务数据 | 3年季度 | 5年季度 | 中 |
| 新闻文本 | 1年 | 3年+ | 中高（需爬虫或采购） |
| 研报文本 | 6个月 | 2年+ | 高（通常付费） |

### 4.3 计算资源

| 阶段 | 最低配置 | 推荐配置 |
|------|---------|---------|
| 原型开发 | RTX 3090 (24GB) | RTX 4090 (24GB) |
| LLM 推理 (7B) | RTX 3090 (24GB) | A100 (40GB) |
| RL 训练 | CPU 32核 | RTX 4090 |
| 全量训练 | A100 x1 | A100 x4 |

### 4.4 主要风险

1. **过拟合历史数据**: RL 容易发现虚假模式，需要严格的样本外测试
2. **市场结构变化**: 训练环境与部署环境分布不同（regime shift）
3. **幸存者偏差**: 回测数据中退市股票被剔除
4. **交易成本估算**: 实际交易成本（滑点、手续费）难以精确建模
5. **LLM 幻觉**: LLM 生成的文本分析可能有事实错误
6. **监管风险**: A 股市场的交易限制（涨跌停、T+1）需在环境中建模

### 4.5 预期效果参考

基于已有文献和开源项目，LLM+RL 选股策略在 A 股市场的预期表现：
- 年化超额收益（vs 沪深300）: 5-15%（回测），实盘可能打 3-5 折
- Sharpe Ratio: 1.0-2.0（回测）
- 最大回撤: 15-25%
- 换手率: 月度 30-100%

---

## 5. 开源方案参考

### 5.1 RL 框架

| 项目 | GitHub | 特点 |
|------|--------|------|
| **FinRL** | [AI4Finance-Foundation/FinRL](https://github.com/AI4Finance-Foundation/FinRL) | 最完善的金融 RL 框架，支持多种环境（股票/期货/加密货币），内置多种 RL 算法，活跃维护 |
| **ElegantRL** | [AI4Finance-Foundation/ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL) | 轻量级 RL 库，支持多种算法的云端并行训练 |
| **FinRL-Meta** | [AI4Finance-Foundation/FinRL-Meta](https://github.com/AI4Finance-Foundation/FinRL-Meta) | 金融市场环境元库，提供 300+ 市场数据集和环境 |
| **Stable-Baselines3** | [DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3) | PyTorch 版 RL 算法库，PPO/A2C/SAC/TD3 开箱即用 |
| **RLlib** | [ray-project/ray](https://github.com/ray-project/ray) | 分布式 RL 库，支持大规模并行训练 |

### 5.2 LLM + 金融

| 项目 | GitHub | 特点 |
|------|--------|------|
| **FinGPT** | [AI4Finance-Foundation/FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) | 金融 LLM 开源框架，支持情感分析、市场预测、RL 集成 |
| **FinBERT** | [ProsusAI/finBERT](https://github.com/ProsusAI/finBERT) | 金融文本 BERT，情感分类 SOTA |
| **BB-FinBERT** | 多个实现 | Bloomberg 风格金融 BERT |
| **Qwen-Fin** | 通义千问金融版 | 中文金融 LLM（阿里云） |

### 5.3 关键论文

| 论文 | 链接 | 要点 |
|------|------|------|
| FinRL: Deep RL Framework for Automated Stock Trading | [arXiv:2011.09607](https://arxiv.org/abs/2011.09607) | FinRL 框架论文 |
| FinGPT: Open-Source Financial LLM | [arXiv:2306.06031](https://arxiv.org/abs/2306.06031) | FinGPT 论文，含 RL 集成 |
| Deep RL for Portfolio Management | [arXiv:1804.03755](https://arxiv.org/abs/1804.03755) | 经典组合管理 RL 论文 |
| Decision Transformer for Finance | [arXiv:2206.03938](https://arxiv.org/abs/2206.03938) | DT 在金融的应用 |
| Can ChatGPT Forecast Stock Price Movements? | [arXiv:2304.07619](https://arxiv.org/abs/2304.07619) | LLM 选股能力评估 |

### 5.4 中文/A 股相关资源

- **Qlib** (微软): [microsoft/qlib](https://github.com/microsoft/qlib) — 支持 A 股的 AI 量化平台，内置 RL 模块
- **BigQuant**: 国内 AI 量化平台，有 RL 选股教程
- **掘金量化**: 国内量化交易平台，支持 Python 策略
- **JoinQuant (聚宽)**: 在线回测平台，支持 RL 策略开发

---

## 6. 实施建议

### 6.1 三阶段路线图

**Phase 1: 基线验证（2-4周）**
- 搭建 FinRL 环境，实现 PPO 选股基线
- 使用日线数据 + 技术因子
- 目标：跑通训练→回测→评估的完整流程

**Phase 2: LLM 特征增强（4-8周）**
- 引入 FinBERT/Qwen-Fin 做新闻情感特征提取
- 特征融合（价格+文本）作为 RL 状态
- 对比消融实验（有/无 LLM 特征）

**Phase 3: 策略优化与部署（8-12周）**
- 引入更多数据源（财报、龙虎榜、资金流向）
- 超参数优化 + 稳健性测试
- 模拟盘/小资金实盘验证

### 6.2 技术选型建议

| 组件 | 推荐 | 备选 |
|------|------|------|
| RL 框架 | FinRL | Stable-Baselines3 |
| LLM | Qwen-7B-Fin / FinBERT | FinGPT |
| 回测引擎 | Backtrader | Zipline / 自定义 |
| 数据源 | AKShare + TuShare | Wind（付费）|
| 实验管理 | MLflow | Weights & Biases |

### 6.3 风险提示

1. 实盘前必须充分样本外测试（至少覆盖一个完整牛熊周期）
2. LLM 推理延迟可能影响日内策略，更适合日频/周频选股
3. A 股市场存在涨跌停、T+1 等独特约束，必须在环境中建模
4. 建议先从沪深300成分股池开始，逐步扩展到全市场
