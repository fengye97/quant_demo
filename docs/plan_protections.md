# Plan: 近期 Bug 复盘 → 保护机制清单

**生成日期**: 2026-05-26
**Review 范围**: `eb4d369`（canonical-date dedup）+ `acf2ca2`（holdings/freshness/v3 scoring）+ `6f8827e`（timing P0 realism）+ `0e7508c`（walk-forward）+ `50d0a11`（请求竞态）
**定位**: 此文档列出"已修复 bug 暴露的隐患类别"和"对应的保护机制"，作为 Pillar 1 / Pillar 2 实施时顺手补齐的硬约束。不引入新功能，只防止同类 bug 再次发生。

---

## 一、Bug 复盘：5 类高风险模式

### 模式 A — Silent data integrity（最危险）

| Bug | 表现 | 根因 | 已修方式 |
|---|---|---|---|
| `eb4d369`：同月多个交易日 → 伪换仓期 | 月中 supplement 多次产生 5-11/5-12/5-22/5-25 行；`groupby('交易日期')` 把它们当 4 个独立期回测 | `load_data()` 没有"每月 ≤ 1 个 canonical 日期"不变量 | `load_data()` 内按 YYYY-MM 取 max(date) dedup |
| `eb4d369`：col54 backfill silent skip | 上月「下周期每天涨跌幅」永远填不上 | `stock_data[code][-1]` 在当月有行时指向当月行，`startswith(prev_month)` 永远 False，且**无日志/计数** | 显式 `reversed` 搜索 prev_month_row |
| `acf2ca2`：持仓出现多个 open snapshot | UI 看到重复未平仓行 | 没有 dedup + 没有 "open 只能有 1 笔" assert | dedup + 只保留最后一笔 open snapshot |

**共性**：默认行为符合 happy path，但坏数据进来不会爆炸，只会安静地产生错误结果。**没有任何 assert / count metric / 日志**能让人在结果错之前发现。

---

### 模式 B — 数据源之间的对齐 / 新鲜度

| Bug | 表现 | 根因 | 已修方式 |
|---|---|---|---|
| `acf2ca2`：指数被旧数据覆盖 | `force_refetch=True` 时新抓到的指数 max_date 早于本地缓存仍被写入 | `get_index_daily()` 没有 freshness guard | 抓回数据若 max_date < 本地缓存 max_date，禁止写盘 |
| `acf2ca2`：A 股指数日线 vs ETF 日线日期不一致 | UI 显示"刷新完成"但交易日历漂移 | 没有跨文件一致性检查 | `_check_a_share_index_etf_alignment()` 失败时 raise |

**共性**：每个文件单独看是对的，但**两个文件应该满足的关系**没人在代码里写出来。

---

### 模式 C — 缓存版本号是软规则

`web_app.py` 的 `_CACHE_VERSION` 是手动递增的整数（v12 → v14）。问题：

- `eb4d369` 改了 `load_data()` 的语义（dedup 后行数变少），**必须 bump 版本号**——但完全靠开发者记得手动改。
- 已经有过两次"忘记 bump 导致用旧缓存"的体感，每次解决方案都是再 +1。
- 反向也会出问题：纯加注释也 bump 一次，触发用户不必要地重算 30 分钟。

---

### 模式 D — Timing 引擎 invariant 未写在代码里

`6f8827e` 一口气修了 5 个 P0：
1. Star50 staged `price_series` 取错（用了未复权而非 qfq）
2. `profit_lock_*` API → ctor 之间没穿透
3. `etf_open == 0` 直接除零崩
4. `filter_timing_result` inception 泄漏（slice 跨过了 ETF 第一天）
5. `first_real_etf_date` 泄漏（产生虚构持仓）

外加 staged-mode 设计 bug：`binary_position` 不门控 strength bucket，导致状态机的卖信号在分档暴露下被吞掉。

**共性**：这些 bug 都对应可写成单行 assert 的硬约束：
- "ETF 第一天之前没有任何 trade"
- "任何 fill 的 price > 0"
- "staged exposure 模式下 binary_position 必须 gate 所有 strength bucket"

但代码里没有任何 `assert` 或运行时校验，全靠 6f8827e 那次集中 review 才挖出来。

---

### 模式 E — 请求层参数漂移

`6f8827e` 把 12 个 realism 参数从 query string 经 `setattr` 注入策略实例：

```python
# web_app.py 当前模式（简化）
strat = TimingStrategy(...)
for k in extra_keys:
    setattr(strat, k, request.args.get(k))
```

问题：
- 参数名拼错没人发现（`profit_lock_pct` vs `profit_lock_percent`）
- 类型从 query string 进来全是 `str`，需要每个调用点自己转 float/bool
- 参数有效范围（如 `slippage_bps ∈ [0, 1000]`）没在一处声明

`50d0a11` 修的"stale 响应覆盖新 view"也是请求层问题：旧请求晚到把新视图覆盖。

---

## 二、对应的保护机制（按落地难度排序）

### 保护 1 — `load_data()` 出口处的 schema 不变量（**1 天**）

在 `stock_trade_demo/backtest.py:237` dedup 之后立刻补：

```python
# Invariant 1: 每个 YYYY-MM 只能映射到 1 个 canonical 交易日
ym_to_dates = df.groupby(df['交易日期'].dt.to_period('M'))['交易日期'].nunique()
assert (ym_to_dates == 1).all(), (
    f"load_data() invariant 失败：以下月份有多个 canonical 交易日 "
    f"{ym_to_dates[ym_to_dates > 1].to_dict()}。dedup 逻辑可能被绕过。"
)

# Invariant 2: 月度选股期数 ≈ 月数
n_periods = df['交易日期'].nunique()
ym_span = df['交易日期'].dt.to_period('M').nunique()
assert n_periods == ym_span, f"期数 {n_periods} 与月数 {ym_span} 不一致"
```

收益：模式 A 类 bug 在 load 阶段就爆，不会污染后续回测。

---

### 保护 2 — 跨表回填的 "expected vs actual" 计数（**0.5 天**）

`get_stock_info.py:supplement_csv_incremental()` 收尾打印：

```python
expected_backfills = len([c for c in stock_data
                          if any(r[0].startswith(prev_month_str)
                                 for r in stock_data[c])])
actual_backfills = len(backfill_prev_col54)
ratio = actual_backfills / max(1, expected_backfills)
print(f"col54 backfill: {actual_backfills}/{expected_backfills} "
      f"({ratio:.1%})", file=sys.stderr, flush=True)
if ratio < 0.90:
    raise RuntimeError(
        f"col54 回填覆盖率 {ratio:.1%} < 90%，疑似 backfill 逻辑失效。"
    )
```

收益：模式 A 中"silent skip"类的回归立刻可见，不需要等回测结果异常。

---

### 保护 3 — 缓存版本号自动化（schema fingerprint，**1 天**）

替换 `_CACHE_VERSION = 14` 为：

```python
def _compute_cache_fingerprint() -> str:
    """基于影响缓存内容的代码 + schema 计算 fingerprint。
    任何 load_data / select_and_backtest / result_to_json 改动会自动失效缓存。
    """
    import hashlib
    h = hashlib.sha256()
    for path in [
        'stock_trade_demo/backtest.py',
        'stock_trade_demo/strategies/base.py',
        'stock_trade_demo/strategies/original.py',
        # ... 列出影响缓存输出的关键文件
    ]:
        with open(path, 'rb') as f:
            h.update(f.read())
    # 加入数据 schema 版本
    h.update(b'schema_v=stock_data_csv:gbk:60cols')
    return h.hexdigest()[:12]

_CACHE_FINGERPRINT = _compute_cache_fingerprint()
```

- 缓存文件名改为 `web_cache_{fingerprint}.pkl`
- 启动时只读匹配 fingerprint 的缓存，旧文件保留磁盘但不加载
- 后台脚本定期清理 > 7 天的旧 fingerprint 缓存

收益：模式 C 的"忘记 bump"和"无意义 bump"都消失。注意：必须明确列出参与 hash 的文件白名单，否则一改注释就重算。

---

### 保护 4 — 数据源对齐的统一检查脚本（**1 天**，与 plan_data.md Step 1 协同）

`scripts/check_data_freshness.py`（plan_data.md 已规划）必须包含**关系断言**，不只是单文件检查：

```python
RELATIONS = [
    ('csi1000_daily.csv', 'timing_etf/csi1000_etf_daily.csv', 'max_date_equal'),
    ('chinext_daily.csv', 'timing_etf/chinext_etf_daily.csv', 'max_date_equal'),
    ('star50_daily.csv', 'timing_etf/star50_etf_daily.csv', 'max_date_equal'),
    ('stock_data.csv', 'csi1000_monthly.csv', 'max_month_equal'),
    # A 股交易日历是真值源：任何 ETF 不允许有日期早于 A 股指数 max_date 的"未来"行
    ('a_share_calendar_daily.csv', 'timing_etf/*_etf_daily.csv', 'etf_dates_subset_of_calendar'),
]
```

收益：模式 B 的"两个文件单独看都对、合在一起错"类 bug 在每日刷数据后立刻可见。

---

### 保护 5 — Timing 引擎 invariant 写成 `assert`（**0.5 天**）

`stock_trade_demo/timing/backtest.py:_replay_timing_positions()` 关键位置补：

```python
# 进入 replay 前的数据完整性
assert (etf_df['open'] > 0).all(), "ETF 包含 open=0 的行，无法 fill"
assert (etf_df['close'] > 0).all(), "ETF 包含 close=0 的行"
assert etf_df['date'].is_monotonic_increasing, "ETF 日期未升序"
first_real_date = etf_df['date'].iloc[0]

# 每笔生成的 trade
for trade in trades:
    assert trade['date'] >= first_real_date, (
        f"trade {trade} 早于 ETF 首日 {first_real_date}（inception leak）"
    )
    assert trade['price'] > 0, f"trade {trade} 价格为零或负"

# staged 模式
if mode == 'staged' and binary_position == 0:
    assert all(strength_bucket == 0 for ...), (
        "staged 模式下 binary_position=0 时所有 strength bucket 必须为 0"
    )
```

收益：模式 D 的 5 个 P0 全部在运行时自爆，不需要靠 review。

---

### 保护 6 — 请求层参数 schema 化（**1.5 天**，配合 Pillar 1 Step 4 蓝图层）

引入 `stock_trade_demo/web/params.py`（可用 `dataclasses` 或 `pydantic`）：

```python
from dataclasses import dataclass

@dataclass
class TimingParams:
    profile: str = 'best'
    slippage_bps: float = 5.0
    commission_rate: float = 0.0003
    profit_lock_pct: float | None = None
    profit_lock_drawback: float | None = None
    # ...
    @classmethod
    def from_query(cls, args) -> 'TimingParams':
        # 集中类型转换 + 范围校验
        ...
```

- 所有 `/api/timing/*`、`/api/us_timing/*` 改为 `params = TimingParams.from_query(request.args)`
- 策略 ctor 接收 `**asdict(params)` 而不是 `setattr` 散喂
- 拼错的参数名在 dataclass 构造时立刻失败

收益：模式 E 的"参数名漂移 / 类型漂移"消失。**注意**：此项与 Pillar 1 Step 4（拆 blueprint）一起做，避免修改两次同样的 handler。

---

### 保护 7 — 回归测试套件持续扩张

`6f8827e` 已经新增 `tests/test_timing_realism.py`（16 个 case 覆盖 5 个 P0）。**应作为模板**：

- 规则：**每个修复的 P0 bug 必须补一个会失败的回归测试**，commit 信息引用 test 名。
- 新增：
  - `tests/test_load_data_dedup.py` — 覆盖 eb4d369 同月多行 dedup
  - `tests/test_supplement_backfill.py` — 覆盖 col54 当月有行时回填
  - `tests/test_holdings_open_dedup.py` — 覆盖 acf2ca2 多 open snapshot
  - `tests/test_index_etf_alignment.py` — 覆盖 acf2ca2 freshness guard
- CI 钩子（pre-commit）：变更 `backtest.py` / `timing/backtest.py` / `get_stock_info.py` 触发对应 test 子集。

---

## 三、与 Pillar 1 / Pillar 2 的协同时机

| 保护 | 落地时机 | 与既有计划的关系 |
|---|---|---|
| 1. `load_data()` schema assert | **Pillar 2 Step 2 前**做（独立小 PR） | 给 Step 2（atomic_io）之后的 schema 校验打基础 |
| 2. backfill expected/actual 计数 | **下一次跑 supplement 之前**做 | 独立小 PR，不阻塞两个 pillar |
| 3. cache fingerprint | Pillar 1 Step 6（services/cache_store.py）一起做 | 自然属于缓存层 |
| 4. 数据源对齐脚本 | **Pillar 2 Step 1** 的 `scripts/check_data_freshness.py` 内置 | 同一个脚本 |
| 5. timing assert | Pillar 1 Step 8（engine/execution.py 抽取）一起做 | 抽函数顺手补 |
| 6. 请求参数 schema | Pillar 1 Step 4（blueprint 拆分）一起做 | 同时改 handler |
| 7. 回归测试模板 | **持续做**，不绑定 pillar | 每个 PR 都遵守 |

---

## 四、立刻可做的两个最小动作

1. **下一次 commit 之前**：在 `stock_trade_demo/backtest.py:237` dedup 之后追加 7 行 assert（保护 1）。
2. **下一次跑 `get_stock_info.py --mode supplement` 之前**：补上 backfill expected/actual 计数（保护 2）。

这两个加起来 < 1 小时，但能直接拦住 `eb4d369` 类的两种 bug 复发。

---

## 五、明确不做的事

- **不**给所有函数加 `try/except`——assert 风格更直接、不掩盖 bug。
- **不**追求 100% 覆盖率——只覆盖 P0 bug 类别。
- **不**引入 schema 校验框架（`pydantic`）作为整体依赖——只在 Pillar 1 Step 4 的请求层用，且可降级为 `dataclasses`。
- **不**改变 `data/live_trades.csv` 的写入路径（CLAUDE.md 规则 15），相关 assert 走 plan_data.md 的 `PROTECTED_PATHS` 兜底。
- **不**改 `_CACHE_VERSION` 自动化前的人工流程——fingerprint 上线后再切，且保留人工 override 入口。
