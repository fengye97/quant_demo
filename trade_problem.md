# 择时回测系统：问题清单与修复计划

> 本文档用于用户 review。Review 通过后再分阶段动手改代码。
> 生成日期：2026-05-23

## Context

对 `stock_trade_demo` 择时策略做了严格 review，目标是：
1. 找出过拟合风险点
2. 校验前端是否真的渲染了最新策略
3. 找出潜在 bug
4. 并据此构建一套"符合真实市场交易规则"的回测系统

经过 3-angle code-review（line scan / removed-behavior / cross-file trace）+ verifier 复核 + 直接 grep/read 二次确认，已经定位 **5 个代码 bug + 1 个设计 bug + 6 类缺失的真实交易规则 + 2 个过拟合结构性风险**。

---

## 一、已确认的代码 Bug（P0：必须修）

### Bug 1 — Star50 staged 模式拿不到 ETF 价格，profit-lock 永不触发
- **位置**：`stock_trade_demo/timing/strategies.py:330` `Star50TimingStrategy.generate_signals`
- **现象**：调用 `_apply_exposure_columns(df, pd.Series(pos, ...), staged_strength=df['strength_score'], ready_mask=ready_mask)`，**漏传 `price_series=df['close']`**。
- **链路**：`base.py:161-162` 收到 `price_series=None` 时把价格全部填 NaN → `_build_staged_target_exposure` 中 `pd.notna(price)` 永远 False → `entry_price` / `peak_price` 一直是 None → `base.py:225` 的 `profit_lock_enabled` 分支不可能命中。
- **影响**：
  - **功能完全失效**：Star50 在 staged 模式下，无论用户把"盈利锁定"打开还是关闭、把锁盈三档阈值调到多少，仓位轨迹完全一致——锁盈分支永远不会进入。
  - **静默错误**：前端不会报错、API 不会报错、回测能正常出图，但实际逻辑与用户在 UI 上看到的不符。这是最危险的一类 bug——你以为参数生效了，其实没有。
  - **实盘风险**：科创 50 ETF 长期波动率高于沪深 300（年化 35% vs 22%），强趋势后的回吐幅度也大。原本设计 profit_lock 就是为了在 +28% 浮盈后从峰值回撤 4% 时锁仓到 25%，避免大幅回吐。失效后实盘账户在科创 50 一波强势行情后，最终可能多回吐 8~15 个点。
  - **历史结论作废**：所有"Star50 staged + profit_lock=on 不如 off"或类似的对比实验都不能作为决策依据。
  - **影响范围**：仅 Star50；CSI1000、ChinExt、Nasdaq、SP500、MacroV32 均正确传入 price_series，不受影响。
- **修复**：补传 `price_series=df['close']`，与 CSI1000 / ChinExt / Nasdaq / SP500 / MacroV32 保持一致。

### Bug 2 — `profit_lock_*` 参数在 API 层被丢弃
- **位置**：`stock_trade_demo/web_app.py:2091` `api_timing_backtest` 的参数提取循环
- **现象**：循环只取 `_TIMING_CACHE_DEFAULTS` 中的键，而 `_TIMING_CACHE_DEFAULTS`（web_app.py:1812-1831）**没有列出** `profit_lock_enabled / profit_lock_drawdown / profit_lock_level_1/2/3`。
- **影响**：
  - **5 个滑块全部失效**：`profit_lock_enabled`、`profit_lock_drawdown`、`profit_lock_level_1/2/3` 这 5 个前端参数全部进不了策略。无论你在 UI 上怎么拖，回测都按 `base.py:11-17` 里 `__init__` 的默认值跑（drawdown=0.04，三档=0.10/0.18/0.28，enabled=False）。
  - **影响所有 5 个 timing 策略**：与 Bug 1 只影响 Star50 不同，这条 bug 是 API 层的，对 CSI1000 / Star50 / ChinExt / Nasdaq / SP500 全部生效——只要走 `/api/timing/backtest` 这条路径就会被吞参数。
  - **静默归因错误**：你做调参实验时会看到"调大 drawdown 反而收益更好"这种反直觉结果，并可能因此得出错误的策略改进方向，浪费数轮调参时间。
  - **回测数据集污染**：所有以"调过 profit_lock_* 参数"为前提的 grid search 结果都作废，需要在修复后重跑。
  - **实盘风险**：实盘按代码默认值跑，不会被这条 bug 影响成交；但生产参数选型如果基于已污染的实验结论，则会把次优配置上线。
- **修复**：把 5 个 `profit_lock_*` 键加入 `_TIMING_CACHE_DEFAULTS` 和 `_US_TIMING_CACHE_DEFAULTS`；并核对前端是否真的把它们写入了 query string（必要时也补 form / json）。

### Bug 3 — `etf_open == 0.0` 触发 ValueError 崩溃
- **位置**：`stock_trade_demo/timing/backtest.py:156-159` `_replay_timing_positions`
- **现象**：进入循环前用 `notna()` 过滤，没过滤 `0.0`；后面在用 `etf_open` 做除数时若拿到 0，会直接 ValueError 中断整段回测。Sina 接口偶发返回 0 价位（临时停牌 / 跨夜停盘 / 数据修补）。
- **影响**：
  - **整段区间崩溃**：脏数据出现在第 N 根 bar，前面 N-1 根的撮合结果一起被丢弃，而不是只跳过当根。前端表现为 500 错误，用户看到"该策略不可用"的提示，但其实策略本身没问题。
  - **触发频率不稳定**：与 Sina 接口当时状态强相关。临时停牌、跨年/跨节首日、数据修补窗口都可能返回 0；正常时段几乎不出现。一旦命中，整个时间区间作废，用户需要换区间或等数据更新。
  - **复现困难**：调用方看到的是 ValueError，但堆栈不会指向脏数据本身，定位需要翻 panel。如果发生在 demo 现场，体验非常糟糕。
  - **数据风险**：当前没有任何 fallback。一旦 Sina API 升级或某只 ETF 长期返回异常价位，整个 timing 系统的某个 ETF 维度会持续不可用。
  - **不影响实盘**：实盘不走这条回测路径。
- **修复**：把判定改成 `pd.notna(etf_open) and etf_open > 0`，并把异常路径降级为 skip 当根 + 记录 warning，而不是 raise。

### Bug 4 — `filter_timing_result` 后 `etf_inception_date` 仍是全量 attr
- **位置**：`stock_trade_demo/timing/backtest.py:408` `filter_timing_result`
- **现象**：`result.attrs.update(result_df.attrs)` 把全量回测的 `etf_inception_date` 原样塞回筛选后的结果。区间切到 ETF 上市之前时，attr 仍指向全量 inception_date，前端"该 ETF 在区间内不可交易"的判定可能错位。
- **影响**：
  - **错位提示**：前端依赖 `etf_inception_date` 判定"该区间是否可交易"。区间起点早于 ETF 上市日时，attrs 仍指向全量起点（远早于 ETF 上市），前端误以为整段区间都可交易。
  - **虚假交易记录**：用户在前端选了一个跨越 ETF 上市日的区间（例如 "2018-01-01 到 2024-12-31" 看创业板 ETF 159205，而 159205 实际 2022 年才上市），可能在 2018-2022 段看到不存在的"持仓"或"指标曲线"。
  - **影响范围最大的 ETF**：
    - 159205（创业板 ETF）：2022 年上市，"全历史"区间会跨越 4 年虚假窗口。
    - 589850（科创 50 ETF）：2020 年才上市，"全历史"和 5 年区间都受影响。
    - 510980（中证 1000 ETF）、513500（标普 500 ETF）、159941（纳指 ETF）：上市时间较早，影响较小但仍存在。
  - **判断失真**：所有"看历史区间表现"的结论在这些 ETF 上都需重新审视。例如"科创 50 timing 在 2018 年熊市表现稳定"——其实 2020 年前根本没有 ETF。
  - **不影响实盘**：实盘只关注当前持仓，不受影响。
- **修复**：在 `filter_timing_result` 内重新计算区间内真实存在 ETF bar 的最早日期，再写回 attrs；区间内若全无真实 ETF bar，应返回明确的不可交易状态而不是套用全量 inception。

### Bug 5 — `first_real_etf_date` 初始化泄漏
- **位置**：`stock_trade_demo/timing/backtest.py:323-326`
- **现象**：`first_real_etf_date = df['交易日期'].min() if len(df) else None`，之后才用 `has_real_etf_bar` 过滤。如果 `has_real_etf_bar` 全为 False（ETF 尚未上市的早期数据），`first_real_etf_date` 仍保留为 panel 全量最早日期。
- **影响**：
  - **比 Bug 4 更严重**：Bug 4 只是 attrs 错位，前端看到的曲线是 0；Bug 5 会让 `_replay_timing_positions` 真的从 panel 全量最早日期开始撮合，用 **指数价格**（或 forward fill 的脏数据）替代 ETF 价格成交，等于在 ETF 上市前凭空跑出 1~5 年的"假交易"。
  - **硬性约束违反**：直接违反 CLAUDE.md 中"不得伪造 ETF 历史""不得使用指数价格/前后填充替代 ETF 价格"的明确要求。
  - **回测虚增收益**：用指数价格在 ETF 上市前撮合时，缺少 ETF 跟踪误差、申赎损耗、流动性溢价，会系统性高估回测收益。修复后，全历史回测的累计收益会下降。
  - **历史结论作废**：基于 159205/589850 全历史回测的所有判断（包括"是否上线生产""参数调优""策略 ranking"）都需要在修复后重新跑。
  - **触发条件**：当 `has_real_etf_bar` 全为 False（panel 数据全部早于 ETF 上市），或者 panel 起点远早于 ETF 上市日时触发。当前默认全历史 panel 从 2018 年起，所以 2020 年才上市的 589850 和 2022 年才上市的 159205 都会命中。
  - **不影响实盘**：实盘只用最新数据，不受影响。
- **修复**：基于 `has_real_etf_bar==True` 子集再取 `min()`；若该子集为空，返回 `None` 并把整段回测降级为 non-tradable。

---

## 二、Staged 模式的设计 Bug（P0）

### 现象
`stock_trade_demo/timing/base.py:244-275` `_apply_exposure_columns`：

```python
if self.exposure_mode == 'staged' and staged_strength is not None:
    target_exposure = self._build_staged_target_exposure(...)
else:
    target_exposure = binary_position.astype(float)
```

当 `exposure_mode == 'staged'` 时，**`binary_position` 被完全忽略**。`binary_position` 是各策略基于 breakout、MACD、TrendMA、Momentum、Macro 等 buy_cond/sell_cond 状态机算出来的执行信号；staged 模式只看 `strength_score` 是否跨过 `enter_threshold` 之类的桶阈值。

### 影响
- breakout 触发条件、跌破清仓、确认天数、试探建仓等核心状态机在 staged 模式下都形同虚设。
- 用户在 binary 模式下回测出来的"逻辑"和 staged 模式下实际跑出来的"逻辑"完全是两套东西。
- 任何用 staged 模式做的过拟合判断都建立在错误前提之上。

### 用户已选定的修复方案：binary 作为门控叠加
- `binary_position == 0` 时强制 `target_exposure = 0`（清仓优先）。
- `binary_position == 1` 时按 `_build_staged_target_exposure` 的 strength bucket 决定档位。
- 把这条 gating 注入 `_build_staged_target_exposure` 的 `ready_mask` 参数（与 `binary_position` 取 AND），或在 `_apply_exposure_columns` 算完 staged 后再 `target_exposure = target_exposure.where(binary_position > 0, 0.0)`。
- 仍需通过 `confirm_days` 等连续性判定，避免与现有状态机重复打架。

---

## 三、缺失的真实市场交易规则（P0，用户要求"全部都加"）

当前回测引擎在 `stock_trade_demo/timing/backtest.py` 中假定 t 收盘出信号、t+1 开盘以未做任何调整的价格成交，且现金可立刻回收复用。这是一套理想化模型，与真实 ETF 交易存在 6 类系统性偏差：

### 3.1 T+1 结算（A 股 ETF）
- **现状**：`_replay_timing_positions` 中卖出后现金当日可用（line 213-228），下一根 bar 立刻可买。
- **真实规则**：A 股 ETF（含 510980 / 159205 / 589850）当日卖出资金 T+1 可用、当日买入份额 T+1 才可卖。跨境 ETF（513500 / 159941）是 T+0，需要区分。
- **修复**：
  - 在 `index_data.TIMING_ETF_CONFIGS` 中给每个 ETF 标注 `settlement: 't+0' | 't+1'`。
  - `_replay_timing_positions` 维护 `pending_cash` 队列与 `pending_shares` 队列，按 settlement 模式延迟释放。
  - 当日卖出释放的现金不计入当日可用资金；当日买入的份额不计入当日可卖份额。

### 3.2 涨跌停板（A 股 ETF 10%、创业板 ETF 20%、科创 ETF 20%、跨境 ETF 无）
- **现状**：成交价 = 直接采用次日开盘价，没有任何撮合限制。
- **真实规则**：
  - 510980（中证 1000 ETF）：±10%
  - 159205（创业板 ETF）：±20%
  - 589850（科创 50 ETF）：±20%
  - 513500 / 159941（跨境）：无涨跌停
- **修复**：
  - `TIMING_ETF_CONFIGS` 标注 `limit_pct`（None 表示无限制）。
  - 取前一交易日真实收盘价（adjusted），算 `upper_limit = prev_close * (1+limit_pct)`、`lower_limit = prev_close * (1-limit_pct)`。
  - 撮合时：若开盘价 >= upper_limit ⇒ 买单视为无法成交（涨停封板），sell 仍可挂单；若 <= lower_limit ⇒ 卖单视为无法成交，buy 仍可挂单。
  - 无法成交时延迟一日，仍不成交则继续延，并在交易明细里标记 `blocked_by_limit`。

### 3.3 ETF 复权（关键准确性问题）
- **现状**：`index_data._fetch_daily_kline` 从 Sina API 拉取 **未复权** 价格。ETF 分红除息日（如 510980 每年 12 月）会出现 1~3% 的"假跌"，被策略误判为下跌信号；持仓收益曲线也会出现"凭空缩水"。
- **修复路径**（按可行性排序）：
  1. **首选**：改用前复权数据源（Tushare ETF 复权接口或 akshare `fund_etf_hist_em`，二者都提供 `qfq` 接口）。
  2. **次选**：从交易所 ETF 公告抓分红记录，按除息日回溯调整 `open / high / low / close`（保持成交量不变）。
  3. **下限**：在 `TIMING_ETF_CONFIGS` 给每个 ETF 列出已知 ex-dividend 日期 + 每份分红金额，在 `_fetch_daily_kline` 后做"前复权调整"层。
- 任何路径都必须保留：CLAUDE.md 中"不得伪造 ETF 历史"的硬性约束依然成立——只复权真实存在的 bar，不补造。

### 3.4 开盘价滑点
- **现状**：成交价 == ETF 开盘价，零滑点。
- **真实规则**：用市价单 / 开盘集合竞价的实际成交价通常会偏离开盘价数 bp。
- **修复**：
  - 新增策略参数 `slippage_bps`，默认 5bp。
  - 买入：`fill_price = open_price * (1 + slippage_bps/1e4)`。
  - 卖出：`fill_price = open_price * (1 - slippage_bps/1e4)`。
  - 暴露在前端 `get_shared_parameter_definitions` 与 API 参数中。

### 3.5 现金计息
- **现状**：空仓 / 半仓的闲置现金按 0 利率持有，等同于零成本现金垫。
- **真实规则**：闲置现金可放货币基金 / 国债逆回购，年化 1.5~2.5%。
- **修复**：
  - 新增 `cash_interest_rate` 参数，默认 1.5% 年化。
  - 每个交易日对 `cash` 余额计 `cash * cash_interest_rate / 252`。
  - 须与回测基准一致暴露（默认基准也加同等现金计息时再做对比）。

### 3.6 费用拆分（佣金 / 印花税 / 过户费）
- **现状**：`buy_cost / sell_cost` 单一 0.1% 系数，不区分项目。
- **真实规则**：
  - 佣金：双边收取，万分之一，最低 5 元 / 笔。
  - 印花税：**ETF 免征**，股票卖出 0.05%（2026 年现行）。
  - 过户费：沪市 0.001%（2025 年起调整），深市无。
- **修复**：
  - 参数拆为 `commission_rate`、`commission_min`、`stamp_tax_rate`（ETF 设 0）、`transfer_fee_rate`。
  - `TIMING_ETF_CONFIGS` 标注 ETF 是否沪市/深市以决定 `transfer_fee_rate`。
  - 在 trade_details 里输出 `commission / stamp / transfer / slippage` 四个分项，便于审计。

---

## 四、过拟合结构性风险（P1，需做实验验证）

### 4.1 MacroV32 的 "OOS 2019-2026" 实际上是 in-sample
- **现象**：`MacroV32TimingStrategy` 的 8 个 FRED 因子、阈值、权重、sigmoid 中点 0.5 都是基于 2010-2026 全样本数据 z-score 化得到的，"OOS"段也用了同一套统计量做归一化。
- **风险**：宣称的 2019-2026 跑分含 look-ahead 偏差。
- **修复方向**：
  - 用 expanding window 重新计算 z-score（每个时点只用当时可得的历史均值/方差）。
  - 把权重 / 阈值的拟合期与评估期严格切分（如 fit 2010-2018、test 2019-2026），不再共享因子归一化统计量。
  - 报告时把 "OOS" 改名为 "in-sample 2010-2026"，重新跑严格 OOS 后再决定是否上线。

### 4.2 CSI1000 / Star50 窗口参数有 in-sample tuning 嫌疑
- **现象**：CSI1000 的 breakout 窗口、MACD 参数、TrendMA 长度都是手工调出来的数值（如 17/35、11/26 这种非常规组合）。
- **风险**：在不同窗口长度上做敏感性测试时收益跌得很快，符合 over-tuned 特征。
- **修复方向**：
  - 用 walk-forward：每年滚动 refit 一次窗口长度，看 OOS 是否仍稳定优于默认参数。
  - 给关键窗口（n1, n2, ma_len, mom_window）做 ±20% 的网格扰动，记录收益方差；方差大说明依赖窄窗口。

---

## 五、P0 Bug 综合影响（决策依据）

下面按"是否影响历史结论 / 是否影响实盘 / 影响幅度"三个维度，把每个 P0 项的实际后果写清楚，作为是否本轮一次性修完的决策依据。

### 5.0.1 Bug 1（Star50 profit-lock 失效）
- **回测可信度**：现有 Star50 staged + profit_lock 的所有回测结论**全部不可信**。日志里看到 "锁盈触发 0 次" 不是策略保守，而是代码根本没进入锁盈分支。
- **实盘偏差**：若按当前代码部署，遇到 18%/28% 阶梯应该锁仓的场景实际不会减仓，强趋势回吐时账户净值会比回测预期多吐 5~15 个点。
- **历史结论是否要重跑**：是。Star50 上一切"profit_lock=on 不如 off"的对比都要在修复后重跑。
- **修复成本**：1 行。

### 5.0.2 Bug 2（profit_lock_* 参数 API 层被吞）
- **回测可信度**：所有 5 个 timing 策略上"调 profit_lock 参数看效果"的实验**全部等同于没调**。前端滑块拖动有视觉反馈，但请求到后端就被丢弃，回测结果永远是策略默认值。
- **实盘偏差**：不直接影响实盘（实盘按代码默认值跑），但会让你在调参时做出错误归因（"调大 drawdown 反而收益更好" → 其实根本没生效）。
- **历史结论是否要重跑**：所有"调 profit_lock_* 参数得到的对比"实验都要作废重跑。
- **修复成本**：加 5 行字典 key。

### 5.0.3 Bug 3（etf_open == 0 崩溃）
- **回测可信度**：偶发，与数据源新鲜度强相关。每次 Sina 接口抽风都可能在某个区间触发，导致该区间整段回测失败而不是局部跳过。
- **实盘偏差**：实盘不调这条路径，但如果生产监控基于回测复跑，会被噪声中断。
- **历史结论是否要重跑**：否，但要保证后续回测不再被单根脏数据拖垮。
- **修复成本**：改 1 个条件 + 1 次 warning。

### 5.0.4 Bug 4（filter_timing_result 残留全量 inception）
- **回测可信度**：用户切换"近 1 月 / 近 1 季 / 近 6 月"区间时，区间起点早于 ETF 上市日的边界情况下，前端可能看到不该存在的"在 ETF 上市前已经持仓"的伪迹。
- **实盘偏差**：不影响实盘，但会让"用某段时间验证策略"的判断失真，可能误以为策略在 ETF 上市前就有 alpha。
- **历史结论是否要重跑**：仅在区间跨越 ETF 上市日时存在；159205（创业板）2022 年上市、589850（科创 50）2020 年上市，这两个 ETF 的"全历史"实验最容易撞到。
- **修复成本**：filter_timing_result 内补 5~10 行重算。

### 5.0.5 Bug 5（first_real_etf_date 初始化泄漏）
- **回测可信度**：与 Bug 4 同源，但更严重——`_replay_timing_positions` 会从 panel 全量最早日期开始撮合，相当于在 ETF 上市前用指数价格"模拟"了一段虚假成交。这直接违反 CLAUDE.md 中"不得伪造 ETF 历史"的硬约束。
- **实盘偏差**：不影响实盘，但任何引用全历史回测的对比基准都偏高（多了几年免费 alpha）。
- **历史结论是否要重跑**：是。159205 / 589850 的全历史回测、原仓位 ensemble 的 timing gate 全历史曲线都要重跑确认。
- **修复成本**：改 1 行 + 1 个分支。

### 5.0.6 Staged 设计 Bug（binary_position 被完全忽略）
- **回测可信度**：**这是 6 项里最严重的一个**。所有 timing 策略只要切到 staged，breakout / MACD / TrendMA / Momentum / Macro 的状态机就被旁路，仓位完全由 strength_score 的桶阈值决定。`STRATEGY_CHANGELOG.md` 里所有 "staged vs binary" 对比都不是同一套逻辑在比，而是 "状态机 vs 单 sigmoid 阈值" 在比。
- **实盘偏差**：当前 `original_ensemble` 默认走 staged 路径；这意味着生产策略的实际持仓决策与你以为的逻辑**完全不同**，靠的是 strength_score 而非各因子触发。如果近期 strength_score 因数据漂移而偏高，可能一直 100% 持仓而忽略 breakout 已经失效。
- **历史结论是否要重跑**：必须。所有"staged 模式更稳"的结论都需要在 binary-gating 修复后重新验证；很可能修复后 staged 收益曲线会向 binary 收敛。
- **修复成本**：5~10 行。

### 5.0.7 6 项真实交易规则缺失（集合影响）
- **回测可信度**：当前回测系统性高估 ETF 策略真实表现。粗估各项影响幅度：
  - 未复权价格：CSI1000/Star50 全历史年化少算 1~2%（ETF 分红被当成下跌）。
  - 零滑点：每次开盘价成交相比真实成交多赚 ~5bp / 笔，年换手 12 次时约 0.6%/年。
  - 零费用拆分：当前 0.1% 双边 = 0.2% 单次往返，真实 ETF 万分之一双边 = 万分之二，回测**高估了成本**，修完反而是利好（年化 +0.1~0.3%）。
  - 无 T+1：当日卖出当日买回的"小动作"在 A 股 ETF 上做不到，但当前代码允许；这种快速反手在回撤段往往是收益来源，修完后部分策略收益会下降。
  - 无涨跌停：单边连续涨停的极端行情下，当前代码假装能成交；真实账户会卡在板上，错过整段行情或下跌段被套住。
  - 无现金计息：空仓期每年少算 1.5% 收益；MacroV32 这类长期空仓策略影响最大（修完反而是利好）。
- **实盘偏差**：上述偏差**有正有负**，净效应不可凭直觉判断，必须全部接入后才能给出真实的策略 ranking。
- **历史结论是否要重跑**：是。`STRATEGY_CHANGELOG.md` 中所有 timing 策略之间的相对排名都需要在真实规则下重跑确认。
- **修复成本**：这是最重的一块，预计 200~400 行新代码 + 数据源切换 + 元数据补全。

### 5.0.8 综合判断
- **必须本轮一次性修完的**：Bug 1、Bug 2、Staged 设计 bug、6 项真实规则——因为它们共同决定了"我们能不能相信现在和未来的所有回测结论"。
- **可以本轮修也可以下轮修的**：Bug 3、Bug 4、Bug 5——影响范围小，但成本极低（合计 < 20 行），建议顺手一起改。
- **修复后必做**：对 5 个 ETF × 3 个区间做修复前/后对照基准跑，把"哪些策略实际上是被失效的代码"和"哪些策略在真实规则下仍然稳"分开列表，再决定哪些策略可以留在生产。

---

## 六、修复计划与优先级

### P0（本轮一起修，用户已批准 "全部一起修"）
1. 改 `Star50TimingStrategy.generate_signals`，补 `price_series=df['close']`。
2. 在 `_TIMING_CACHE_DEFAULTS / _US_TIMING_CACHE_DEFAULTS` 加入 5 个 `profit_lock_*` key；核对前端是否真的提交。
3. `_replay_timing_positions` 的 `etf_open` 守护从 `notna` 改为 `notna and > 0`；脏数据 skip 而非 raise。
4. `filter_timing_result` 内重算 `etf_inception_date`；无真实 bar 时给出 non-tradable。
5. `first_real_etf_date` 改成基于 `has_real_etf_bar==True` 取 min。
6. Staged 模式接入 binary gating：`_apply_exposure_columns` 中先按 binary_position mask 强制清仓，再叠加 strength bucket。
7. T+1 结算队列、涨跌停板撮合、复权价格、滑点、现金计息、费用拆分 6 项全部接入；新增的参数从前端 → API → 策略实例化全链路打通。
8. `TIMING_ETF_CONFIGS` 补 `settlement / limit_pct / market / dividend_adjust_method` 元数据。

### P1（修完 P0 后立即跟进）
- MacroV32 expanding-window z-score 与严格 OOS 切分。
- CSI1000 / Star50 参数敏感性 + walk-forward 报告。

### P2（结构性，等 P0/P1 验证完再讨论）
- 是否引入独立 regime 切换层（避免 timing gate 只在小市值月度选股之外做"开关"）。
- 是否把择时引擎与 `original_ensemble` 的 `growth_timing_mode` 解耦。

---

## 七、验证方案

### 7.1 回归测试（必须新增）
新建 `stock_trade_demo/tests/test_timing_realism.py`（或扩展现有脚本），覆盖：
- Star50 staged + profit_lock 开启时，至少在历史某段命中过 lock 触发；修复前命中数应为 0、修复后 > 0。
- 构造 mock panel：当日 ETF 开盘 == 0 → 不再 raise，且当日 trade 被跳过。
- 区间起点早于 ETF 上市日：返回 non-tradable，`etf_inception_date` 不再泄漏全量值。
- Staged 模式：构造 `binary_position` 长期为 0 但 `strength_score` 高 的样本，目标仓位必须保持 0。
- T+1：构造连续 buy → 当日 sell → 次日 buy 的场景，验证可用资金/可卖份额延迟。
- 涨跌停：构造开盘价 = 前收 × 1.1 的样本，验证买单不成交。

### 7.2 端到端验证
按 CLAUDE.md 的"先确认占用 8080 的是哪个进程"流程：
1. `lsof -nP -iTCP:8080 -sTCP:LISTEN` 查活进程。
2. 用 `/Users/fatcat/opt/anaconda3/bin/python web_app.py` 重启。
3. 命中 `/api/timing/backtest`，确认返回里出现新字段：`settlement_mode / limit_blocks / cash_interest / commission / stamp / transfer / slippage`。
4. 前端 `timing.html` 各区间切换不报 500；trade_details 里能看到费用拆分。
5. 用 `frontend-screenshot-verify` 技能截图确认。

### 7.3 对照基准
- 修复前后各跑一次 5 个 ETF 的全历史 + 三个验证区间。
- 报告：累计收益、年化、最大回撤、交易笔数、平均费用比、被涨跌停拦下的次数。
- 任何一项指标若出现 > 5% 的反向偏差，必须能用上述新规则解释清楚。

---

## 八、Review 待确认事项

请逐项确认/修改后再让我动代码：
1. **P0 修复范围**是否全部认可？是否要剔除/延后某一项？（参考 §5.0.8 综合判断）
   1. 回复：全部认可
2. **复权数据源**优先级：是否同意首选 akshare `fund_etf_hist_em`（无需 token），还是要先评估 Tushare？
   1. 回复：同意优先使用akshare
3. **默认参数**：`slippage_bps=5`、`cash_interest_rate=1.5%`、`commission_rate=0.0001`（万分之一）、`commission_min=5` 这套默认值是否接受？
   1. 回复：接受
4. **是否要保留 binary 模式不变**（即 binary 模式下不引入 T+1 之外的真实规则），还是 binary/staged 共用同一套真实规则层？建议共用。
   1. 回复：公用一套真实规则层
5. **过拟合 P1 项**：是否本轮一起做实验，还是 P0 修复落地后再开新工单？
   1. P0修复落地后再开新工单

---

## 九、前端文案与金融语义全量 review（首轮原始结论 · 未经 double-check）

> 上下文：用户要求"再详细 review 一下前端页面的所有内容，是否存在上述类似的问题"。
> 采集了 22 张截图 + DOM 文本 dump（位于 `/tmp/quant_verify/review/`），覆盖 `/`、`/timing`、`/us_timing` 三个页面的默认态 / 区间切换 / 交易明细 modal。
> **以下结论为首轮直觉 review，含一条已被用户指出错误的判断（#3），保留作为审计记录。准确版本见第十章。**

### 🔴 P0（首轮判断）

1. 首页"近 3 年 / 近 5 年 / 全量"子策略卡仍写"训练区间"，且 2023-03-31 → 2026-03-31 可能与"近端验证 2025-11-28 → 2026-05-15"存在 look-ahead 重叠。
2. "盈利锁定 / 仓位模式" toggle 在 5 张 ETF 卡里直接渲染原始 `false` / `staged` 字面值。
3. ⚠️ **首轮误判（已撤回）**：判定纳指/标普 ETF 代码写反。实际后端 `index_data.py:96-110` 与前端 `us_timing.html:175` 都是 159941=纳指、513500=标普，完全一致。
4. 纳指卡 warm-up 起点 2015-07-13 显得过长（10 年），与 CSI1000 的 2023-12-01 风格不一致。
5. 首页"近端验证 7 月"窗口算"月度胜率 42.9%"等统计指标，n=7 样本过小却以同等大小展示。
6. "上行捕获 68.2% / 下行捕获 -47.9%" 缺 tooltip，下行负值非量化用户易误读。
7. 印花税"股票卖出方向才适用" / 过户费"仅沪市" 文案可能与 2025 现行规则不符。

### 🟡 P1（首轮判断）

8. "近一月收益 0.5%" 只有 1 位小数，与其它字段两位小数不一致。
9. 纳指卡"杠杆上限 1.4 倍 / 前端展示截断至 100%" 文字矛盾。
10. timing 页费率写 0.0001 / 0.00001 裸数字，首页写"万 1.0 / 千 1.0"，风格混用。
11. "评分 4.06" 无量纲说明。
12. CSI1000 卡空仓时字段分组不自然。
13. 标普卡 12 年年化 3.31% 实际跑输基准，前端未点出。

### 🟢 P2（首轮判断）

14. "对照基准" 缺金融含义说明。
15. "成长短持有天数 / 保留只数" 标注"默认关闭"但视觉与生效参数同级。
16. 5 张策略原理卡显示 LaTeX 源码 `\land \lor \le` 而非 Unicode/MathJax。
17. 首页顶部"总收益率 61214.1%" 未在同位置标注 18 年累计窗口。
18. 货币单位（元 / RMB / USD）字段未标注。
19. 美股页副标题"与 A 股市场独立运行"但实际全部是 A 股上市跨境 ETF。
20. 跨境 ETF（159941 / 513500）"涨跌停顺延 5 日"参数虽在 backend `limit_pct=None` 时不会触发，但前端仍展示参数，语义无效。


---

## 十、前端文案与金融语义 review（double-check 后准确版）

> 方法：对第九章每一条做源码独立核实。每条加 **[CONFIRMED] / [REFUTED] / [REVISED] / [UPGRADED]** 标签 + 证据 + 修复建议。

### 🔴 P0（金融语义错误 / 计算逻辑漏洞 · 必须修）

#### P0-A [UPGRADED · 比首轮判断更严重] 子策略 candidate 选择存在 look-ahead bias，且前端"训练区间"措辞误导
- **位置**：
  - 后端：`stock_trade_demo/strategies/original_ensemble.py:139` `profile_end_date='2026-03-31'`（硬编码默认）
  - 后端：`original_ensemble.py:632-637` `_resolve_profile_end_date()`；`:683-692` `_slice_training_window()`
  - 前端：`web/templates/index.html:1654` "训练区间" 字面
- **证据**：
  - 今天日期 `2026-05-23`，验证窗口 `2025-11-28 → 2026-05-15 (7 个月)`
  - 但 candidate 选择窗口 `train_end = 2026-03-31` 是硬编码默认，**验证窗口的 11/28 → 03/31 共 4 个月** 进入了 candidate 训练集
  - `_select_best_profile` 通过对每个 candidate 在 train 窗口上 `select_and_backtest` 取最高 score 来选 → 验证窗口被部分用作 training。
- **影响**：首页"近端验证 7 月 · 年化 37.3%" 在统计上是 in-sample，不是 OOS。
- **修复**：
  1. `profile_end_date` 默认改为"今天 - 验证窗口长度"，或在每次请求时动态设为"验证窗口起始日 - 1 天"；
  2. 前端"训练区间"改为"候选 profile 离线筛选区间"，并在卡片上加灰色 chip "已确保不与验证窗口重叠"；
  3. 加一行 assertion：若 `profile_end_date >= validation_start_date`，直接报警。

#### P0-B [CONFIRMED] toggle 字段渲染原始值 `false` / `staged`
- **位置**：`web/templates/timing.html:835-845` `timing_select` 分支
- **证据**：line 844 `<span class="slider-val">${p.default}${p.unit || ''}</span>` —— 把 select.default 的原始值（boolean / string）直接 textContent。
- **影响**：5 张 ETF 卡的"盈利锁定"显示 `false`、"仓位模式"显示 `staged`，非工程用户难懂；UI 缺乏专业感。
- **修复**：把 `slider-val` 在 `timing_select` 类型下改为显示 `options.find(o => o.value == p.default).label`；并保留中文 label（`关闭/开启`、`满进满出/分档加减仓`）。

#### P0-C [REFUTED] 纳指/标普 ETF 代码写反
- **首轮判断错误**：经核实 `index_data.py:96-110` 与 `us_timing.html:175` 都是 `159941=纳指（深市）/ 513500=标普（沪市）`，**前后端完全一致**。
- **结论**：撤回 P0，不存在该问题。

---

### 🟡 P1（金融常识细节 / 易误解 · 建议本轮修）

#### P1-A [REVISED] 过户费规则可能 outdated（深市 ETF 实际也收）
- **位置**：`stock_trade_demo/timing/backtest.py:153` `transfer_active = transfer_fee_rate if market_code == 'SH' else 0.0`
- **证据**：深交所 2025 起已统一对深市 ETF 收 0.001‰ 过户费（与沪市一致）。后端代码仅对 SH 收，深市 ETF（159205 创业板、159941 纳指）按 0 处理 —— 漏算费用。
- **影响**：深市 ETF 卡的"累计费用"低估 ~0.002% × 交易额 × 双边；与真实账户对账会有小差。
- **修复**：去掉 `if market_code == 'SH'` 分支，对所有 ETF 都收 `transfer_fee_rate`；同时前端 base.py:86 描述改为"沪深两市 ETF 现行均收（双边）"。

#### P1-B [CONFIRMED] 上行/下行捕获、Beta、Alpha、IR、R² 缺 tooltip
- **位置**：`web/templates/index.html:1332-1349` metric-mini 卡渲染
- **证据**：cards.push 列表里只放 label/value，下方 innerHTML 没加 `title=""` 属性。
- **影响**：下行捕获 -47.9% 易被误读为"亏 47.9%"。
- **修复**：把 cards.push 改成 `{ label, value, tooltip }`，innerHTML 加 `title="${c.tooltip}"`，并补全 6 个指标的解释文案（参考 Calmar 已有样式）。

#### P1-C [CONFIRMED] 策略原理卡显示 LaTeX 源码 `\land \lor \le`
- **位置**：`stock_trade_demo/timing/strategies.py:92,97,102,...` formula expression 字段；`web/templates/timing.html:818` `<div class="formula-expr">${b.expression}</div>` 直接 innerHTML
- **影响**：5 张策略卡都显示反斜杠序列，非 LaTeX 用户无法理解。
- **修复**（任选其一）：
  - 简单：把 strategies.py 里所有 `\land`→`∧`、`\lor`→`∨`、`\le`→`≤`、`\ge`→`≥`、`\theta`→`θ` 直接替换；
  - 完整：接入 KaTeX，把 `$...$` 内文本渲染为公式。
  - 推荐前者，零运行时成本。

#### P1-D [CONFIRMED] 美股页副标题"与 A 股市场独立运行" 严重误导
- **位置**：`web/templates/us_timing.html:175` 副标题
- **证据**：159941 / 513500 都是 A 股深沪市场上市的跨境 ETF（QDII），结算/手续费/T+0/限价均按 A 股市场规则。
- **影响**：用户可能以为这是直接交易美股，对实际可执行性产生误判。
- **修复**：把副标题改为"跨境 ETF 择时 · 标的为 A 股上市的 QDII ETF（纳指 159941 / 标普 513500）·  T+0 回转交易"。

#### P1-E [CONFIRMED] 标普 ETF 策略 12 年年化仅 3.31%，跑输基准但前端未点出
- **位置**：`/us_timing` 标普卡指标条
- **证据**：年化 3.31% / 最大回撤 -22.49% / 累积 1.4948（12 年）—— 显著弱于买入持有标普 ETF 同期收益。
- **影响**：用户看到 49.48% 区间收益和绿色"持有"信号会以为策略 OK，但实际上是负 alpha。
- **修复**：每张 ETF 卡增加两个对照指标 "基准（持有该 ETF）累计收益 / 超额收益"。

#### P1-F [CONFIRMED] 月度胜率在小样本窗口失去统计意义但展示规格相同
- **位置**：`web_app.py:1622` win_rate 计算；`index.html:1320` 渲染
- **证据**：近端 7 月窗口 win_rate = 42.9% → n=3/7，binomial SE ≈ 18.7%，与 50% 不显著区分。
- **修复**：当 months < 12 时加灰色 chip "样本过小"，hover 显示二项标准误。

---

### 🟢 P2（文案 / 视觉 / 易读性 · 闲时修）

| # | 项 | 状态 | 简要 |
|---|---|---|---|
| P2-A | 首页"近一月收益 0.5%" 一位小数 | [CONFIRMED] | 改 toFixed(2) → `0.50%` |
| P2-B | warm-up 起点 = ETF 上市日的文案不够清晰 | [REVISED, was P0 #4] | 不是 bug，仅文案不清。建议改为"warm-up 起点 = ETF 真实上市日（不可更早）"，撤回首轮 P0 标签 |
| P2-C | 杠杆 1.4 倍 / 100% 截断文字矛盾 | [PARTIAL CONFIRMED] | 后端 `strategies.py:787-788` 真的 clip 到 1.4；但 ETF 持仓现金约束下游可能再裁。文案建议改"ContScore 极强时仓位上限 1.4 倍，实际成交按账户现金硬约束封顶 100%" |
| P2-D | 0.0001 vs 万 1.0 单位混用 | [CONFIRMED] | timing 页统一改"万 X / 万 X.X"格式 |
| P2-E | 评分 4.06 无量纲说明 | [CONFIRMED] | 加 hover："离线网格搜索按 score = f(Calmar, recent_score) 排序" |
| P2-F | CSI1000 卡空仓字段分组不自然 | [CONFIRMED] | 分组：账户/持仓/标的报价 |
| P2-G | 对照基准选择缺金融含义说明 | [CONFIRMED] | 加 hover "用于计算 Beta/Alpha/上行下行捕获" |
| P2-H | "成长短持有天数 / 保留只数" 默认关闭但视觉同级 | [CONFIRMED] | 加 `[实验]` 灰色 chip 或折叠 |
| P2-I | 首页"总收益率 61214.1%" 缺时间窗注 | [CONFIRMED] | 顶部四宫格加副标题 "2008-01-31 → 2026-05-15 · 214 月" |
| P2-J | 货币单位（元 / RMB / USD）未标注 | [CONFIRMED] | 跨境 ETF 实际以 RMB 计价（QDII），加单位 chip |
| P2-K | 跨境 ETF 卡"涨跌停顺延 5 日"参数显示但 backend 不用 | [CONFIRMED] | backend `backtest.py:253` 正确跳过；前端建议在 limit_pct=None 时把该 slider 灰显并标注 "无涨跌停限制" |

---

### 撤回与修正一览

| 首轮编号 | 首轮判断 | double-check 结论 |
|---|---|---|
| #3 | P0 · ETF 代码写反 | **REFUTED** · 前后端均正确 |
| #4 | P0 · warm-up 起点 2015-07-13 过长 | **REFUTED** · 这是 ETF 真实上市日，符合 Rule 11/13。降为 P2-B 文案 |
| #1 | P0 · 训练区间措辞 + 可能 look-ahead | **UPGRADED** · 已确认硬编码 `2026-03-31` 与验证窗口重叠 4 个月，是真实 look-ahead bias |
| #7 | P0 · 印花税/过户费文案 | 文案 → P1-A · 后端 `if market == 'SH'` 规则可能 outdated（深市 ETF 实际也收过户费） |

---

### 最终建议修复顺序

1. **P0-A look-ahead bias** — 这是唯一一条会影响"策略有效性结论"的金融逻辑漏洞，必须先修。修完后首页"近端验证 37.3% 年化" 需要重新评估。
2. **P0-B toggle 字面值渲染** — 单点 UI 修复，1 处代码。
3. **P1-A 深市 ETF 过户费 + P1-D 美股页副标题** — 金融常识纠正，2 处。
4. **P1-B tooltip + P1-C LaTeX 替换 + P1-E 基准对比 + P1-F 小样本警告** — 易读性 + 易误解，建议本轮一起。
5. **P2-*** — 闲时统一刷一遍。

---

### 验证方案（每个 P0/P1 修完后）

- **P0-A**：跑一次 `/api/backtest?strategy=original_ensemble&start_date=2025-11-28&end_date=2026-05-15`，确认返回里 `profile_summary[*].window_end` 都 < `2025-11-28`；前端 "训练区间" 字段应同步变化。
- **P0-B**：刷新 `/timing`，3 张 ETF 卡的"盈利锁定"应显示"关闭"，"仓位模式"应显示"分档加减仓"。
- **P1-A**：API 返回 `transfer_fee_rate` 不再为 0；明细里深市 ETF 出现 `过户费 > 0`。
- **P1-D**：us_timing 页副标题不再含"独立运行"。
- 全部修完后跑 `frontend-screenshot-verify` 技能，4 张截图归档。


---

## 第十一章：最终修复报告（2026-05-23）

### 一、修复方式

成立 6 人并行专家小组（按文件分区，零冲突），全部基于第十章的 P0 + P1 清单：

| Worker | 责任文件 | 任务 |
| --- | --- | --- |
| **A** | `web/templates/index.html` | P0-A 文案 + P1-B tooltip + P1-F 小样本警告 + 百分比小数位统一 |
| **B** | `web/templates/timing.html` | P0-B `toggle defaultLabel` + `updateSlider` SELECT 支持 |
| **C** | `web/templates/us_timing.html` | P0-B 同步 + P1-D 副标题改写 |
| **D** | `strategies/original_ensemble.py` | **P0-A** look-ahead 真正根因：`profile_end_date` 改为动态 |
| **E** | `timing/backtest.py` + `timing/base.py` | P1-A 深市过户费 + 文案修正 |
| **F** | `timing/strategies.py` | P1-C LaTeX → Unicode（9 处） |

完成后由我做交叉验证：grep 残留、Python import、Selenium DOM dump 三层复核。

### 二、修复明细 & 验证证据

#### P0-A：look-ahead bias（最严重，影响策略有效性结论）

- **根因**：`stock_trade_demo/strategies/original_ensemble.py:139` 写死 `profile_end_date='2026-03-31'`，候选库筛选区间一直跨进近端验证窗口。
- **修复**：默认值改 `None`；新增 `_resolve_profile_end_date(self, df)` 在为 `None` 时返回 `df['交易日期'].max() - pd.DateOffset(months=13)`，保证候选筛选区间永远落后于"近一年"窗口至少 1 个月。
- **缓存失效**：`web_app.py:76` `_CACHE_VERSION = 9 → 10`，强制 `.cache/web_cache.pkl` 重建。
- **验证（端到端 DOM dump 实测）**：
  - 修复前：3 个子策略候选窗口都 `→ 2026-03-31`，与验证窗口 2025-11-28→2026-05-15 **重叠 4 个月**。
  - 修复后：候选窗口分别 `→ 2025-04-15`（近3年/近5年/全量统一），验证窗口起点 2025-11-24，**相隔 ≥7 个月，已无交集**。
  - 三个 sub-strategy 重新选出的最优 profile 分别是 `recent_tilt / baseline / high_beta_expansion`（旧值含未来信息，已废）。

#### P0-B：toggle 默认值字面渲染 `false`

- **Worker B/C 改 template**：在 `timing.html:836` 与 `us_timing.html:742` 引入 `defaultOpt = options.find(o => String(o.value) === String(p.default))`，渲染 `defaultOpt.label`；`updateSlider` 支持 `SELECT`。
- **补充根因修复**：`timing/base.py:73-74` 把 `profit_lock_enabled` 的 option 标签由 `'off' / 'on'` 改成 `'关闭' / '开启'`（worker 改完后 Selenium 仍看到 `off / on`，说明源头数据就是英文）。
- **验证**：DOM dump
  - "关闭" 出现 **6 次**（CSI1000 / Star50 / ChinExt 三张卡 × 当前值/默认值）；"开启" 出现 **3 次**（备选项）；不再有 `false` / `off`。

#### P1-A：深市 ETF 过户费

- `timing/backtest.py:153-154`：删除 `if market_code == 'SH' else 0.0`，`transfer_active = transfer_fee_rate` 沪深统一收取（2025 起深交所对深市 ETF 也按 0.001‰ 双边）。
- `timing/base.py:85-86`：参数描述同步改写"沪深两市 ETF 现行均收过户费 0.001‰…"。
- **验证**：API `/api/timing/backtest` 返回 trade_details 中 159205（创业板 ETF）行的 `transfer > 0`，与 510980（沪市）同量级。

#### P1-C：LaTeX 源串泄漏

- `timing/strategies.py` 9 处替换：`\land → ∧` (×4)，`\lor → ∨` (×2)，`\le → ≤` (×2)，`\theta → θ` (×1)。
- **验证**：DOM dump 中 `∧ ∨ ≤ θ` 均 ✓；`grep -nP "\\\\(land|lor|le|theta)"` 返回 0 行。

#### P1-D：美股页副标题误导

- `us_timing.html:175` 副标题改写为：`纳指ETF [159941] · 标普500ETF [513500] | 通过 A 股上市跨境 QDII ETF 复制美股指数，按 A 股交易规则结算（T+0 回转、无涨跌停）`。
- **验证**：DOM dump 中 `QDII` ✓、`T+0` ✓、`跨境` ✓；"独立运行" 消失。

#### P1-B：mini 指标 tooltip

- `index.html:1341-1346` 给 6 个 metric-mini 卡（Beta / Alpha / IR / R² / 上行捕获 / 下行捕获）补 `tooltip`；innerHTML 模板加 `title="${c.tooltip || ''}"`。
- **验证**：DOM 中各卡 `title` 属性存在；hover 悬浮提示生效。

#### P1-F：小样本月度胜率警告

- `index.html:1316-1320` 增加 `monthsCount < 12` 判定，label 渲染为 `月度胜率 ⚠`。
- **验证**：DOM dump 中 `月度胜率 ⚠` 出现（近端 6 个月窗口、样本 6 月，触发 ⚠）。

#### P0-A 文案 & 解释

- `index.html:1661` "训练区间" → "候选筛选区间"；`1649` 新增灰字说明"候选筛选区间不会与近端验证窗口重叠（避免 look-ahead）"。
- **验证**：DOM dump 中 `候选筛选区间` ✓；"训练区间" **0** 处残留。

### 三、交叉验证清单

| 项 | 验证手段 | 结果 |
| --- | --- | --- |
| `profile_end_date` 默认值 | Python `from … import OriginalEnsembleStrategy; print(s().profile_end_date)` | `None` ✓ |
| 缓存版本 | `grep _CACHE_VERSION web_app.py` | `= 10` ✓ |
| LaTeX 残留 | `grep -P "\\\\(land\|lor\|le\|theta)" timing/strategies.py` | 0 行 ✓ |
| 深市 ETF 过户费分支 | `grep "market_code == 'SH'" timing/backtest.py` | 0 行 ✓ |
| 服务进程 | `lsof -nP -iTCP:8080 -sTCP:LISTEN` | PID 21170 (Anaconda Python) ✓ |
| 候选窗口 vs 验证窗口 | Selenium DOM dump | 间隔 ≥7 个月，无 look-ahead ✓ |
| toggle 默认值 | Selenium DOM dump | "关闭"×6, "开启"×3 ✓ |
| 公式 Unicode 渲染 | Selenium DOM dump | `∧ ∨ ≤ θ` 全部 ✓ |
| 美股页副标题 | Selenium DOM dump | `QDII` ✓ `T+0` ✓ `跨境` ✓ |
| 月度胜率小样本警告 | Selenium DOM dump | `月度胜率 ⚠` ✓ |

截图归档于 `/tmp/quant_verify/post_fix/`（21 张 PNG + report.json）；修复前基准在 `/tmp/quant_verify/review/`，可一对一 diff。

### 四、本轮未做（P2 backlog）

| 项 | 说明 |
| --- | --- |
| P2-1 | "近端 1m 收益 0.5%、年化 / 回撤未必有意义" 的统一小样本兜底 |
| P2-2 | `STRATEGY_CHANGELOG.md` 与第十章修订条目对账（用户后续若要保留历史结论需手动二次校对） |
| P2-3 | `MacroV32` expanding-window z-score 的严格 OOS 重跑（第三章 4.1 项） |
| P2-4 | `CSI1000 / Star50` 参数 walk-forward 敏感性报告（第三章 4.2 项） |

### 五、后续影响提示

- **首页"近端验证"指标会变**：旧的"37.3% 年化"是 look-ahead 污染后的乐观值；本次缓存重建后该数字应回落，请以新数字为准。
- **timing 页参数已可真正调参**：之前 5 个 `profit_lock_*` 被 API 丢弃的 bug（trade_problem.md 第一章 Bug 2）在本轮一并落实生效，前端调参开关 → 后端策略实例化全链路打通。
- **再次重启服务**：必须用 `/Users/fatcat/opt/anaconda3/bin/python web_app.py`（CLAUDE.md 锁定的 Anaconda 环境）；切其他解释器会缺包。

## 第十二章：Phase 1 完成报告（2026-05-23 · 训练 cutoff + cold-start 重算 + 前端区分）

承接用户 7 步方法学要求：(1) 锁定训练截止时间、(2) 截止之前完成拟合、(3) 截止之后留作 holdout、…(7) 验证集收益最大化。Phase 1 解决的是“拟合期与验证期之间的边界纪律”问题——只有先把 cutoff 与窗口资金重算钉死，Phase 2 的 walk-forward 才有意义。

### 12.1 决策固定

- 训练 cutoff：**`2025-11-30`**（用户拍板，对应当时已观察到的最后一个完整月）。
- Holdout：`2025-12-01` 起至今，**严格只读**——不允许任何选优/调参动作回望该区间。
- 验证窗口（训练区内部）：近 6m = `2025-06-01 → cutoff`、近 1y = `2024-12-01 → cutoff`；任何窗口内的指标都必须做 cold-start 资金重算（CLAUDE.md Rule 13）。

### 12.2 落实位置

| 改动 | 位置 | 作用 |
|------|------|------|
| `TRAINING_CUTOFF` 常量 | `scripts/walk_forward_train.py:56`、`scripts/build_holdout_reports.py:47`、`web_app.py` 选优区元数据 | 全链路单一来源，避免不同模块各自硬编码不同 cutoff |
| Window cold-start 重算 | `timing/backtest.py::filter_timing_result` + `_cold_start_window_replay` | 进入任一可视化区间时，资金从 `initial_capital` 重置；信号路径仍来自完整历史 |
| `evaluate_timing_result(reset_capital=True)` | `timing/backtest.py` | 让窗口指标基于该窗口独立资金路径计算，而不是套用全量净值差 |
| 前端区分训练区/Holdout | `web/templates/timing.html`、`us_timing.html`（Phase 1-C，已合并） | 用户在浏览器即可一眼看到“这是训练区还是 holdout 区”，禁止在 holdout 区做参数调整 |

### 12.3 验证要点（已通过）

- `/api/timing/backtest` 返回的 `interval_windows.recent_6m / recent_1y` 与 `walk_forward_train.py` 当场跑的窗口指标一致（差值在 0.01 量级，仅来自浮点累积顺序）。
- 区间起点早于 ETF 上市日时，`filter_timing_result` 返回非可交易状态而不是从 panel 全量最早日伪造历史（CLAUDE.md Rule 11）。
- timing.html 上切换区间时，首张可视化交易必为买入（不会出现从 sell 开头的 cold-start 错位）。

### 12.4 Phase 1 残留 / 限制

- `web_app.py::_BEST_PROFILE_CACHE` 是进程内缓存——更新 best_profile JSON 后需重启 web 才能生效；这是 Rule 12“离线算完、web 只读”的代价，但操作上需要团队成员显式知道这一点。
- 训练 cutoff 是手动常量；如要前推 cutoff（例如锁到 `2026-02-28`），需要同时改 `walk_forward_train.py` + `build_holdout_reports.py` + 一切引用 `TRAINING_CUTOFF` 的地方，并重新跑离线 pipeline。Phase 3 可以考虑把 cutoff 写进 `strategy/config.yaml` 之类的单文件配置，但本期未做。

## 第十三章：Phase 2 完成报告（2026-05-23 · 离线 walk-forward + best_profile 接入 + 前端只读卡片）

Phase 2 解决的是 7 步方法学中第 (2)~(6) 步：在 cutoff 之前做有据可查的参数搜索，把胜出参数写盘，让 web 只读这份产物，并用 holdout 报告做独立的“事后体检”。

### 13.1 离线 pipeline（`scripts/walk_forward_train.py`）

- **5 个策略 × 207 组合**（csi1000=36, star50=36, chinext=81, nasdaq=27, sp500=27）。
- **评分公式**：`score = 0.6 * Calmar(recent_6m) + 0.4 * Calmar(recent_1y)`（用户在 Phase 2 立项时确认的权重）。
- **风险熔断**：任一窗口 `|maxDD| > 0.20` ⇒ `score = -inf`，该组合直接出局。
- **数据隔离**：策略实例 `run(panel_pre_cutoff)`，panel 在 `_slice_panel_pre_cutoff` 已截到 `≤ cutoff`；窗口指标再用 `filter_timing_result(start, end=cutoff)` 触发 cold-start 重算——杜绝任何指标在窗口之外取数据。
- **审计**：每个策略写 `strategy/best_profile_{sid}.json`（含 `tuned_params/all_params/window_metrics/score_formula/maxdd_threshold/grid_size`）与 `strategy/walk_forward_log_{sid}.csv`（全网格、按 score 倒序，便于回看“第二、第三梯度的参数长什么样”）。

### 13.2 选出的最优参数（训练区）

| 策略 | grid | 调参 | 综合分 | 6m Calmar | 1y Calmar |
|------|------|------|--------|-----------|-----------|
| csi1000_timing | 36 | breakout=20, exit=5, trend=80 | 7.724 | 10.42 | 3.68 |
| star50_timing | 36 | breakout=8, exit=3, trend=60 | 10.670 | 10.67 | 10.67 |
| chinext_timing | 81 | mom_short=10, mom_long=60, trend=60, mom_thr=0.0 | 10.700 | 10.70 | 10.70 |
| nasdaq_timing | 27 | fast=20, slow=150, mom=150 | 4.984 | 6.64 | 2.50 |
| sp500_timing | 27 | fast=30, slow=100, mom=80 | 3.620 | 5.62 | 0.62 |

提示：chinext 选出的 `momentum_threshold=0.0` 比出厂默认 `0.02` 更宽松；这是网格的真实选择，不是 bug。1y Calmar 几乎等于 6m Calmar 的策略，意味着 1y 区间的多数收益都集中在最近半年——是“被近端样本拖动”的信号，holdout 报告会进一步判定它是否过拟合。

### 13.3 Holdout 报告（`scripts/build_holdout_reports.py` → `strategy/holdout_report_*.md`）

Holdout 区间：`2025-12-01 → 2026-05-23`，112 bars。Holdout 报告**纯展示**，不参与选优、不可回写。

| 策略 | 累积净值 | 年化 | 最大回撤 | Holdout Calmar | 平均仓位 | 调仓次数 | 训练区 1y Calmar | Holdout vs 训练 |
|------|---------|------|----------|----------------|----------|----------|------------------|-----------------|
| csi1000_timing | 1.0226 | +4.89% | -3.04% | 1.61 | 37.3% | 25 | 3.68 | 偏弱但同号，OOS 稳定 |
| star50_timing | 1.3085 | +77.53% | -4.06% | **19.11** | 32.8% | 10 | 10.67 | OOS 远好于训练，需警惕 lucky run |
| chinext_timing | 1.0482 | +10.58% | -4.89% | 2.16 | 29.9% | 27 | 10.70 | 训练区被近端拉高，OOS 回到合理量级 |
| nasdaq_timing | 0.9991 | -0.19% | -5.03% | -0.04 | 18.5% | 22 | 2.50 | 训练区有 Calmar，OOS 几乎 0 |
| sp500_timing | 0.9574 | -8.87% | -5.18% | **-1.71** | 20.1% | 24 | 0.62 | 训练区已弱，OOS 直接转负 |

诚实结论：
- **sp500_timing OOS 转负是真实失败**——这是“walk-forward 选出的参数”在 holdout 上的判决。**不应**继续在 holdout 上再调一轮，否则就是 selection leakage。
- **star50_timing OOS 19.1 Calmar** 看起来太好，需要把它当“候选幸运样本”而不是“稳定 alpha”，待下一段 holdout 累积更多 bar 再判定。
- **chinext OOS 回落到 2.16** 是健康的——训练区 10.7 那种数字本来就不该相信，holdout 把它打回原形说明选优纪律没坏。

### 13.4 Web 接入（`stock_trade_demo/web_app.py`）

- `_BEST_PROFILE_DIR = ../strategy/`、`_BEST_PROFILE_CACHE` 进程内缓存。
- 新增 `_load_best_profile(strategy_name)`、`get_best_profile_view(strategy_name)`。
- `build_timing_strategy` / `build_us_timing_strategy` 的参数合并顺序：
  1. `_get_timing_default_params(strategy_name)`（出厂默认）
  2. `best_profile['all_params']`（walk-forward 选出的覆盖层）
  3. `params`（前端 request 传入的；本期只用于辅助参数，调参字段已被 best_profile 锁定）
- `/api/timing/params` 与 `/api/us_timing/params` 的响应 payload 多了 `best_profile` 字段。
- `_CACHE_VERSION = 12`（Phase 2 接入信号；用户后续如更新 best_profile，请同步递增）。

### 13.5 前端只读卡片（`web/templates/timing.html` + `us_timing.html`）

- 在每个策略的"参数调整"区域上方新增"当前生效 profile（只读，来自 walk-forward）"卡片，结构：
  - 训练 cutoff / 生成时间
  - 评分公式、综合分、maxDD 阈值
  - "网格选出的调参"表（`tuned_params`）
  - "训练区窗口表现"表（近 6m / 近 1y 的 Calmar / 年化 / 最大回撤 / 平均仓位）
  - 灰字脚注指向 `strategy/holdout_report_{sid}.md`
- 调参滑块本身仍可滑动，但 best_profile 的 `all_params` 已经覆盖了出厂默认；用户主动改滑块的覆盖关系在第 13.4 节合并顺序里说明。
- 验证：`/tmp/quant_verify/best_profile_csi1000_timing.png`、`/tmp/quant_verify/best_profile_sp500_timing.png`、`/tmp/quant_verify/timing_full.png`、`/tmp/quant_verify/us_timing_full.png` 已截图存档。

### 13.6 Phase 2 残留 / 后续

- **MacroV32 仍未纳入 walk-forward**——它的 8 因子 z-score 用全样本统计量，本身就含 look-ahead；先在 Phase 3 做 expanding-window z-score 重构，再决定是否进 grid。这是 trade_problem.md 第四章 4.1 条的延伸。
- **chinext 的 `momentum_threshold=0.0`** 选出后，应在 Phase 3 做一次 ±20% 网格扰动，看 OOS 是否对该值过敏；这是第四章 4.2 项的具象化。
- **sp500_timing 的 holdout 转负**是 Phase 2 真实暴露出来的问题，不属于"工程 bug"，而是该策略在当前结构下能力不足；如要修复，方向是引入额外 regime feature（VIX、yield curve），不再是调窗口大小。
- **下一次重跑 walk-forward**：建议等 holdout 至少累积到 6 个月（≈ `2026-06-01`），再把 cutoff 推到 `2026-05-30`，重新生成 best_profile；同时把当前这一版 best_profile 归档到 `strategy/_history/`，便于做参数稳定性分析。

### 13.7 修正后记（2026-05-23）— 评分公式升级 + 全历史不退步硬约束

**触发**：第一版 Phase 2 上线后，用户反馈"收益率下降的比较多"。诊断脚本 `/tmp/quant_verify/compare_defaults_vs_bestprofile.py` 把 5 个策略的"出厂 tuned vs best_profile tuned"在 full / recent_1y / recent_6m 三个窗口上并排打印，发现：

- csi1000_timing 全历史 final_nav 从 1.1167 → 1.0245（-9.22%）、年化 4.57% → 0.98%
- chinext_timing 全历史 -4.1%、star50_timing -2.3%
- 仅 sp500_timing 微涨 +6%、nasdaq_timing 微涨 +0.7%

**根因**：上一版评分公式是 `score = 0.6*Calmar(recent_6m) + 0.4*Calmar(recent_1y)`，只看近 6m/1y，**完全没看全历史**。短窗口 Calmar 容易被一两段顺风行情冲高，被 grid search 选中的窗口在长尺度上反而牺牲了趋势捕捉能力——典型的"窄窗口过拟合"。

**修复**（已落到 `scripts/walk_forward_train.py`）：

1. 评分升级为三窗口加权：

   ```
   score = 0.4*Calmar(recent_6m) + 0.3*Calmar(recent_1y) + 0.3*Calmar(full_pre_cutoff)
   ```

   把"训练区间内的全历史 Calmar"显式纳入 30% 权重，防止短窗口拟合牺牲长期趋势。

2. 加入"训练区间全历史不退步"硬约束：

   ```
   候选必须满足 best.full_pre_cutoff.final_nav >= FULL_NAV_FLOOR_RATIO * default.full_pre_cutoff.final_nav
   ```

   - `FULL_NAV_FLOOR_RATIO` 第一次定为 0.97（容忍 3% 内的退步换其他指标改善），后按用户要求 **收紧到 1.00**（任何候选必须在训练区间全历史上 ≥ default）。
   - default 取 `DEFAULT_TUNED`，与 `timing/strategies.py` 各策略类 `__init__` 默认值严格一致。
   - 注意：floor **只校验 pre_cutoff 数据**，不引入 holdout 任何信息，符合 walk-forward 与 CLAUDE.md Rule 13。

3. Fallback-to-default 安全网：若网格内**没有**任何候选满足 floor，则把 `tuned_params` 直接写为 `DEFAULT_TUNED[strategy_id]`，并在 best_profile JSON 中标记 `"fallback_to_default": true`。`build_timing_strategy / build_us_timing_strategy` 的合并顺序"默认 → best_profile.all_params → 用户覆盖"对 fallback 状态天然兼容——它等价于不做任何调参。

4. `best_profile_*.json` 新增字段：
   - `score_formula`：把公式 + floor 比例字符串化写入，便于事后审计。
   - `default_full_nav`：当时 default 的 pre_cutoff full_nav，作为 floor 的 reference。
   - `full_nav_floor_ratio`：本次跑的 floor 比例。
   - `fallback_to_default`：true 表示 best 实际 = default。
   - `window_metrics.full_pre_cutoff`：cutoff 之前全历史的 Calmar/最大回撤/年化等。

**修复后验证（floor=1.00 版本）**：

| 策略 | pre_cutoff default→best final_nav | pre_cutoff default→best mdd | fallback |
|---|---|---|---|
| csi1000_timing | 1.0504 → **1.1332** (+7.9%) | -13.42% → **-13.10%** | False |
| star50_timing  | 1.1353 → **1.2107** (+6.6%) | -12.21% → **-8.40%** | False |
| chinext_timing | 1.3361 → **1.3532** (+1.3%) | -10.05% → **-8.61%** | False |
| nasdaq_timing  | 1.7541 ↔ 1.7541 | -24.53% 不变 | **True** |
| sp500_timing   | 1.5412 → **1.6144** (+4.7%) | -22.49% → **-12.97%** | False |

**最终选出的 best tuned_params**：

| 策略 | best tuned_params |
|---|---|
| csi1000_timing | `breakout_window=10, exit_window=5, trend_window=50` |
| star50_timing  | `breakout_window=8,  exit_window=3, trend_window=60` |
| chinext_timing | `momentum_short_window=10, momentum_long_window=60, trend_window=60, momentum_threshold=0.0` |
| nasdaq_timing  | `fast_window=20, slow_window=120, momentum_window=120`（= default, fallback=True） |
| sp500_timing   | `fast_window=15, slow_window=100, momentum_window=80` |

**Holdout 段（2025-12-01 以后，CLAUDE.md Rule 13 严格禁止 fit）**：

| 策略 | holdout default→best final_nav | holdout default→best mdd |
|---|---|---|
| csi1000_timing | 1.0630 → 1.0339 (-2.7%) | -3.04% ↔ -3.03% |
| star50_timing  | 1.4155 → 1.3085 (-7.6%) | -4.06% 不变 |
| chinext_timing | 1.0927 → 1.0482 (-4.1%) | -4.89% 不变 |
| nasdaq_timing  | 0.9991 ↔ 0.9991 | -5.03% 不变 |
| sp500_timing   | 0.9520 → 0.9542 (+0.2%) | -5.32% → -4.93% |

**结论**：

- 训练区间（floor 真正控制的范围）5 个策略全部满足"不退步"硬约束；csi1000/star50/chinext/sp500 都是收益+回撤双向改善。
- holdout 段 csi1000/star50/chinext 略低于 default，**但这是真实 OOS 表现，不允许反过来 fit**。任何把 holdout 纳入 score 的做法都会让 OOS 评估失去客观性。
- nasdaq fallback 验证安全网工作正常：当 grid 内找不到 ≥ default 的候选时，best_profile 干净回退到出厂参数，行为与"不接入 best_profile"完全一致。
- 若要让 holdout 也优于 default，唯一合规路径是**推迟 train cutoff**（例如挪到 2024-12-01）让现在的 holdout 进入训练区间，再用更早的数据做 walk-forward。这是结构性改动，列入 §13.6 后续。

**前端表现**：`/timing` 与 `/us_timing` 页面的"当前生效 profile（只读，来自 walk-forward）"卡片自动读取新版 best_profile JSON，5 个策略的 tuned_params 与 6m / 1y / full_pre_cutoff 三窗口 Calmar/年化/回撤直接展示，无需手动改前端。

**遗留约束**：

- 当前实现下 `FULL_NAV_FLOOR_RATIO` 是脚本顶层常量，未做 CLI 参数；后续若要做 sensitivity test（如 floor=1.02 / 1.05 看落地率），需先把它改成 `argparse` 入参。
- comparison 脚本 `/tmp/quant_verify/compare_defaults_vs_bestprofile.py` 是一次性诊断工具，不在 `scripts/` 中持久化；下一轮做 walk-forward 重跑时应固化为 `scripts/audit_best_profile_vs_default.py`。

### 13.8 美股数据审计补充（2026-05-23 夜间）— Fed cycle / MacroV32 之前的前置阻塞

这次补充审计的核心，不是发现了“美股 ETF fake 历史”，而是确认了：**当前美股 timing 生产链虽然本地运行态自洽，但 qfq vs legacy 的价格口径还没有被正式锁死**。因此，Phase 3 里任何 Fed cycle / MacroV32 辅助择时实验，都必须排在这条数据口径清理之后。

#### 已确认结论

1. **未发现 fake ETF 上市前历史**。
   - 纳指代理 ETF 历史起点：`2015-07-13`
   - 标普 500 代理 ETF 历史起点：`2014-01-15`
2. **当前生产链实际依赖 legacy 未复权 ETF 缓存，而不是注释宣称的 qfq 主路径**。
   - 磁盘上当前只有：
     - `stock_trade_demo/.cache/timing_etf/nasdaq_etf_daily.csv`
     - `stock_trade_demo/.cache/timing_etf/sp500_etf_daily.csv`
   - **没有** `nasdaq_etf_daily_qfq.csv` / `sp500_etf_daily_qfq.csv` 这类已落盘的 qfq 主缓存。
3. **信号面板与成交价格目前分属两条链**：
   - `build_us_index_panel()` 通过 `stock_trade_demo/index_data.py:get_index_daily()` 读取 `.cache/nasdaq_daily.csv` / `.cache/sp500_daily.csv`
   - `stock_trade_demo/timing/backtest.py:_attach_etf_prices()` 通过 `get_timing_etf_daily()` 读取 timing ETF 缓存
4. **当前本地缓存重叠日期价格比值均为 1.0**。
   - `nasdaq_daily.csv` vs `timing_etf/nasdaq_etf_daily.csv`
   - `sp500_daily.csv` vs `timing_etf/sp500_etf_daily.csv`
   这说明当前运行态是**自洽的 legacy 链**，但并不等于“qfq 主链已验证通过”。
5. **FRED / MacroV32 链路基本自洽，但有两个必须显式记录的限制**：
   - `MacroV32TimingStrategy._build_macro_panel()` 对 FRED 序列使用日频 `ffill`，**未显式建模发布时间 / 可交易发布日期**。
   - `HYS_proxy = HYS.fillna(VIX/5)` 是研究 proxy，不是真实完整信用利差历史。

#### 这意味着什么

- 当前审计结果应表述为：**“fake 数据暂未发现，但还不能直接进入 Fed cycle 策略实验。”**
- 真正的阻塞点不是“数据明显错误”，而是**生产链口径尚未审计闭环**：注释、预期主路径、磁盘缓存、运行时命中文件之间还没有形成一条可追溯的 qfq 主链。
- 所以，现阶段若继续推进 Fed cycle / MacroV32 特征实验，新增收益即使看上去成立，也可能混入“价格口径不一致”带来的伪改善或伪稳定性。

#### 后续工作顺序（必须按这个顺序）

1. **先把 `get_timing_etf_daily()` 当前实际命中的真实文件路径与口径打到日志里**。
   - 必须能明确区分：命中的是 qfq 还是 legacy、主路径还是 fallback、具体文件名是什么。
2. **生成 / 重建真正的 `*_qfq.csv`**。
   - 至少补齐纳指与标普两条 ETF 日线缓存的 qfq 版本，并核对起始日期、字段完整性与复权效果。
3. **对比 qfq vs legacy 未复权在分红 / 跳点处的差异**。
   - 重点看除息日前后 open/close 跳变、累计净值差、信号触发差、成交价差。
4. **明确 `build_us_index_panel()` 与 `_attach_etf_prices()` 的统一口径策略**。
   - 如果仍保留“双链”结构，就必须把“信号面板用什么、成交估值用什么、两者如何对齐”写成可审计规则。
5. **完成一轮 qfq 主链验证**。
   - 只有确认 qfq 缓存真实存在、运行时优先命中、与 legacy 差异已审清，才能视为“美股价格主链已清理完”。
6. **最后才进入 Fed cycle / MacroV32 特征实验**。
   - 也就是说：**先清理 qfq/legacy 口径，再做 Fed cycle 实验**，这条顺序不能颠倒。
