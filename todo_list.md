# TODO List
## 5.18
[x] 1.1 优化股票数据管理 — convert_data.py (CSV→Parquet/Snappy压缩) + backtest.py load_data() Parquet自动检测 + get_data_info() 元信息查询
[x] 1.2 大盘指数Beta/Alpha归因 — index_data.py (Sina API获取CSI 1000日线→月度收益,缓存) + backtest.py compute_alpha_beta() OLS回归 + web_app.py集成归因指标
[x] 2.1 初始本金10w + 费率万1.0 — backtest.py select_and_backtest() initial_capital=100000 + c_rate=1.0/10000 + 当期本金/当期盈亏/累计资金列 + strategies/base.py同步更新
[] 2.2 基于quant_factor.md优化实盘策略 (已实现, 待调参) — QualityValueStrategy (quality_value.py 178行): Size(50%)+BM(25%)+ROE(15%)+Turnover(10%) Z-score复合排名, 市值>20亿+成交额>5000万+ROE>0 过滤, 默认持仓3只。2024-2026回测: 年化17.6%/回撤-54.9% (vs 原版55.6%/-71.5%)。⚠️ 关键发现: (1) 市值过滤天然削弱小市值alpha——A股小市值溢价是收益绝对主导因素; (2) 净资产/归母净利润_ttm为carry-forward数据, BM/ROE因子可靠性存疑; (3) 多持仓(3只)分散降低了波动但未达回撤<30%目标。📋 后续: 进一步调参(仓位/止损/牛熊切换)或引入择时信号
[x] 3.1 Beta/Alpha 前端展示补充 — benchmark曲线叠加 + 归因指标面板 (beta/alpha/IR/R²/上下行捕获已在metrics card展示, CSI 1000曲线叠加在图表中)


## 5.16
> **完成概览 (2026-05-16)**：8/8 全部完成 ✅
> 
> | # | 任务 | 状态 | 核心交付物 |
> |---|------|------|-----------|
> | 1 | 缠论PDF分析 + 量化实现 | [x] | ref_books/chan_theory_analysis.md (850行), chan_theory_factors.py (1399行) |
> | 1.1 | 缠论因子整合 + 策略实现 | [x] | quant_factor.md §17, choose_stock.py 缠论增强版, 策略对比 (3.42x vs 7.82x) |
> | 1.2 | **Method A 日线流水线 + 策略迭代** | [x] | chan_monthly_factor_builder.py, choose_stock.py v2.0, STRATEGY_CHANGELOG.md |
> | 2 | 数据补充 May 2026 | [x] | get_stock_info.py (1370行), stock_data.csv (703,177行) |
> | 2.1 | 数据交叉验证 + 第三方库 | [x] | data_crosscheck_report.md (256行), akshare 1.18.60, tushare 1.4.29 |
> | 3 | RL+LLM 选股调研 | [x] | rl_llm_stock_selection_research.md (249行) |
> | 3.1 | RL+LLM 方案A 代码实现 | [x] | rl_stock_selector/ (11文件), PPO + GAE 完整实现 |
> 
> **待办后续 (Backlog)**：① ~~缠论因子改日线流水线 → 月度聚合~~ ✅ ② akshare 完整数据替代 carry-forward ③ RL 在真实数据上训练对比原版策略


[x] 1. 把ref_books/缠中说禅(原文).pdf 分析完成，撰写详细的分析报告，同时确认是否可以提取量化交易的相关因子，或者把本书的思想用代码实现
    ✅ PM Review: 完成度优秀。ref_books/chan_theory_analysis.md (850行) 详尽覆盖缠论全体系——从包含关系处理、分型、笔、线段、中枢到背驰和三类买卖点，数学定义与量化规则清晰。chan_theory_factors.py (1399行) 实现了完整的7模块流水线（InclusionProcessor → FractalDetector → StrokeBuilder → SegmentBuilder → HubDetector → DivergenceDetector → TradeSignalGenerator），代码编译通过、自测运行正常。识别了10个可量化因子，具备直接对接A股数据的工程基础。建议后续：(1) 引入成交量辅助中枢判断，(2) 增加多级别联立（区间套）的自动检测，(3) 对笔/线段歧义情况补充更多边界处理。
[x] 1.1 把整理到的缠论中的新因子整合到quant_factor.md中，按照文档中的要求进行review。将quant_factor.md中的因子尝试在stock_trade_demo/choose_stock.py中实现，确认一下收益是否可以突破新高，同时尽量减小回撤幅度。
	    ✅ PM Review: 完成。quant_factor.md新增Section 17（缠论因子），按规范格式定义了10个因子（分型密度/分型确认率/笔斜率/笔动能/中枢位置/中枢偏离度/中枢宽度/背驰强度/买卖点状态/走势结构），含输入字段、计算公式、排序方向、单变量/双重排序规则、多空构造方式、代码注意事项。总览表（Section 2）已自动合并5个缠论扩展因子。choose_stock.py经两轮迭代：初版用月度代理变量（bias_20/MACD bar），终版（linter增强）用groupby时间序列操作精确计算——chan_bottom_fractal/top_fractal（三K线分型检测）、chan_bullish_div/bearish_div（价格-MACD背离）、chan_zs_position（中枢位置ZG/ZD归一化）、chan_stroke_dir/stroke_strength（笔方向与强度）、chan_signal_score（综合买卖点评分=3*一类买点+2*二类买点+1*三类买点-2*顶分型-2*顶背驰-1*中枢上方）。策略对比结果：原版累积净值7.82（年化11.19%，最大回撤-69.77%）vs 缠论增强累积净值3.42（年化6.54%，最大回撤-72.75%），缠论策略未超越原版。月度收益相关系数0.706，缠论跑赢原版50.0%月份。⚠️ 关键发现：(1) 缠论因子用月度数据计算时丢失了日线级别精细结构（真正的分型/笔/中枢需日K线流水线，月度近似可能产生假信号）；(2) "买低"偏好与A股小市值动量效应方向冲突，过滤掉大量小盘成长机会；(3) 缠论因子的IC方向可能与直觉相反（如高背驰强度在A股可能预示继续下跌而非反转）；(4) 因子权重未经IC优化。📋 建议后续：(a) 用chan_theory_factors.py对每只股票跑完整日线缠论流水线后月度聚合；(b) 逐个因子做IC测试确定有效方向（A股验证）；(c) 缠论因子应与小市值因子协同而非互斥（在小市值桶内用缠论因子排序）；(d) 引入成交量辅助中枢和背驰判断。
[x] 2. stock_trade_demo/stock_data.csv 这个文档里只有截止到2026.4.30的A股交易数据，看一下get_stock_info.py这个代码是否可用，按照之前同样的数据格式看看把2026年5月份的股票市场数据也完整补充一下，完善一下数据获取代码。
    ✅ PM Review: get_stock_info.py 验证可用（Sina/Tencent API正常响应），代码从267行扩展至1370行，新增：批量股票日线获取、55列技术指标全计算（bias/振幅/std/KDJ/MACD）、月度聚合、CSV补充模式、缓存机制、多线程并发（20 workers）、错误重试。已运行补充脚本，成功获取2026-05数据。注意：因API免费源限制，全市场5223只股票的数据获取需分批进行，补充脚本支持断点续传（缓存）。建议后续安装akshare/tushare以获取更全面的基本面数据。
[x] 2.1 用之前的数据cross-check一下你的结果是正确的，不要胡编乱造。看一下怎么安装akshare/tushare这些第三方库，以便于获取更全的数据。
	    ✅ PM Review: 完成度良好。数据交叉验证：(1) stock_data.csv当前703,178行/862MB，末次更新2026-05-16 12:22；(2) 策略回测成功跑通全周期（2007-2026），5月数据5,217只股票正常参与选股；(3) 月度截面统计通过基础一致性检验（bias_20均值0.0108/中位数-0.0083、MACD bar>0占比60.51%、J均值33.53合理）；(4) 2026-04与2026-05股票重叠度高（>95%），数据连续性正常；(5) 下周期每天涨跌幅字段在回测中被正确解析和使用。akshare/tushare安装：pip3 install akshare在沙箱环境中被拦截（安全限制），但安装命令已明确：`pip3 install akshare` 和 `pip3 install tushare`（tushare需注册token）。get_stock_info.py（1370行）已在Round 1验证可用（Sina/Tencent双源API，20线程并发，缓存+断点续传）。📋 建议后续：(a) 解除沙箱限制后执行pip3 install akshare tushare；(b) 用akshare获取完整的基本面数据（ROE/ROA/营收增长/现金流等）替代当前carry-forward模式；(c) 用tushare获取行业分类和指数成分股数据；(d) 增加数据质量自动检测脚本（缺失值/异常值/前后一致性）。
[x] 3. 调研一下是否可以基于强化学习进行选股，整个模型设计是LLM + RL，评估一下是否可行，看看是否有开源方案可以参考。
    ✅ PM Review: 完成度良好。rl_llm_stock_selection_research.md (249行) 系统调研了RL选股技术现状（DQN/PPO/A2C/SAC/Decision Transformer），提出三种LLM+RL架构方案（LLM作为状态编码器/奖励模型/端到端Agent），推荐方案A（LLM编码+PPO策略）作为起步。整理了FinRL/FinGPT/ElegantRL/Qlib等开源框架及关键论文。可行性评估客观（技术可行★★★★☆），明确列出了数据需求、计算资源、6大风险和预期效果。建议后续：Phase 1基于FinRL搭建PPO基线（2-4周），Phase 2引入FinBERT做LLM特征增强（4-8周）。
[x] 3.1 选择方案A先把codebase实现一下，并说明相关原理和需要的运行条件
	    ✅ PM Review: 完成度良好。rl_stock_selector/ 目录已创建，含以下模块：(1) environment.py (353行) — 自定义StockSelectionEnv，模拟月度选股-调仓-评估全流程，状态空间24维（市场特征5+截面统计10+组合特征3+持仓6），动作空间为top-K离散选股，奖励函数=组合收益-换手惩罚-回撤惩罚；(2) models.py (344行) — MLPEncoder+LLMStateEncoder+PPOModel（Actor-Critic），支持LLM（FinBERT）语义编码和纯MLP降级方案，Cross-Attention融合多模态特征，PPOModel含get_action/evaluate_action双接口；(3) train.py (345行) — 完整PPO训练器，实现Clipped Surrogate Objective、GAE优势估计（γ=0.99/λ=0.95）、多Epoch Mini-batch SGD、Value Clipping、Entropy Bonus、梯度裁剪，支持CPU/CUDA/MPS三端；(4) backtest.py (368行) — BacktestResult数据类+run_backtest回测引擎，输出累积净值/年化收益/夏普比率/最大回撤/Calmar比率/胜率/换手率/信息比率，含matplotlib可视化（资金曲线/回撤/月度收益）；(5) main.py (228行) — CLI入口支持train/backtest/full三模式，参数可配置（--steps/--lr/--top-k等）；(6) README.md — 详尽文档含架构图、PPO算法原理（数学公式）、状态/动作/奖励设计、运行条件、使用方式、预期效果与风险。(7) 专家并行新增文件：env.py（gymnasium环境）、agent.py（stable-baselines3 PPO封装）、llm_encoder.py（LLM编码器含mock/real双模式）、features.py（特征工程），与主实现互补。⚠️ 注意：(a) 训练需torch，LLM模式需transformers；(b) 纯CPU训练100k步预计2-4小时；(c) 文本特征（Phase 2）当前默认关闭，需安装FinBERT后启用；(d) 回测依赖下周期每天涨跌幅字段。📋 建议后续：(a) 在真实数据上跑通完整训练-回测流程；(b) 对比PPO vs 原版策略的样本外表现；(c) Phase 2引入FinBERT中文金融文本特征；(d) 增加多进程采样（VectorEnv）加速训练；(e) 超参数网格搜索（learning rate/clip range/entropy coef）。

[x] 1.2 Method A: 日线缠论流水线 → 月度聚合 + choose_stock.py v2.0 + 策略迭代变更日志
    ✅ PM Review: 三部分全部完成。
    (a) chan_monthly_factor_builder.py — Sina API 500日K线 → ChanTheoryAnalyzer完整流水线（7模块）→ 月度聚合16个因子。8 workers多线程，500只股票/13,051行/0错误。修复3个bug（fractal_type/Direction枚举比较、trade_signals_df无date/signal列、空DataFrame打印崩溃）。
    (b) choose_stock.py v2.0 — 新增method_a_strategy()，加载Method A因子CSV merge到主数据，日线流水线因子负向排除（顶背驰+中枢上方+卖>买）+ 综合评分（分型1x+背驰3x+买卖点2x+中枢1.5x），5%排名倾斜。PM建议"在小市值桶内用缠论因子排序"已实现。
    (c) STRATEGY_CHANGELOG.md — 四版本策略迭代日志含完整收益/回撤指标：v1.0原版（5223.19x, 55.51%年化, -71.50%DD）、v1.1缠论增强代理因子（5130.98x, 55.37%, -71.57%DD）、v1.2纯缠论代理因子（16.04x, 15.39%, -70.91%DD）、v2.0 Method A日线流水线（5223.19x, 55.51%, -71.50%DD）。含IC分析、相关性矩阵、牛熊子周期、滚动回撤。
    ⚠️ 关键发现：(1) v2.0与v1.0相关性1.000，因Method A因子仅覆盖500/5383=9.3%股票，5%倾斜对整体选股无实质影响；(2) 底分型IC=-0.0118(t=-2.86)、背驰强度IC=-0.0217(t=-3.46)显著为负——缠论"买入"信号在月度频率下是负向预测因子；(3) 中枢下方IC=+0.0037、底背驰IC=+0.0076为弱正向，远不足以克服小市值效应；(4) A股小市值溢价是收益的绝对主导因素，任何偏离纯市值排名的增强都稀释该效应。
    📋 建议后续：(a) 扩大Method A覆盖至1000+只股票；(b) IC为负的因子反向使用；(c) 缠论信号更适合风险管理（止损/止盈触发）而非横截面排序；(d) 尝试周度持仓频率。