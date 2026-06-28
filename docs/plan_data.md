# quant 数据管理层改造方案 v1

> 调研窗口：2026-05-26
> 调研基线：CLAUDE.md「核心数据路径速查」+ 实盘数据保护条款
> 调研范围：`get_stock_info.py`、`stock_trade_demo/index_data.py`、`stock_trade_demo/web_app.py`、`scripts/*.py`、`stock_trade_demo/.cache/`、`data/`、`strategy/`

## 0. 设计目标

1. **原子写**：任何 `pandas.to_csv` / `pickle.dump` / `os.replace` 全部走统一的 `atomic_write_*` 工具；崩溃后磁盘只可能停在「旧版完整」或「新版完整」两个状态。
2. **格式收敛**：除 `stock_data.csv`（GBK 历史包袱）与 `data/live_trades.csv`（实盘记录、外部要求 CSV）以外，原则上「行情面板」用 parquet、「策略产物 + best_profile」用 JSON/CSV、「Web 启动加载的预热缓存」用 pickle 但加 schema 头。
3. **统一 freshness 探针**：所有「日线/月线/宏观」缓存按同一份 staleness 规则评估。
4. **schema 校验**：关键面板（`stock_data.csv`、各指数日/月线、ETF 日线、FRED 系列）落地前先过一层 `pandera` schema；不通过直接拒写。
5. **数据血缘**：每个生产出来的缓存写一份 `<file>.meta.json`，记录上游脚本 / 数据源 / 行数 / 时间戳 / git sha。

非目标：不重写 web_app 的内存缓存语义；不动 `data/live_trades.csv` 任何字段或编码；不调整 `stock_data.csv` 列序与 GBK 编码。

---

## A. 现状盘点

> 时间戳与大小快照自 `ls -la` + `du -sh`（2026-05-26）。

### A.1 主数据 / 行情面板

| 路径 | 格式 | 大小 | 编码 | 上游脚本 | 下游消费者 | 当前 freshness 校验 |
|---|---|---|---|---|---|---|
| `stock_trade_demo/stock_data.csv` | CSV | 824 MB（703k 行 × 55 列） | **GBK** | `get_stock_info.py --mode supplement` / `POST /api/update_data` | `web_app.load_data()`、`backtest.py`、所有月度策略 | 无；靠人工 + `_CACHE_VERSION` 整库重算 |
| `stock_trade_demo/stock_data.parquet` | Parquet | 395 MB | UTF-8 | `web_app.py` 行 3519 `_df_for_parquet.to_parquet` | `web_app.load_data()` parquet 优先 | 无 |
| `stock_trade_demo/stock_data_weekly_experiment.parquet` | Parquet | — | UTF-8 | `run_weekly_experiment.py` | 同上 | 无 |
| `stock_trade_demo/xingbuxing_stock_data.csv` | CSV | — | GBK | 历史导入 | 仅研究 | 无 |

### A.2 A 股指数 / 交易日历 / ETF

| 路径 | 格式 | 大小 | 编码 | 上游 | 下游 | freshness |
|---|---|---|---|---|---|---|
| `stock_trade_demo/.cache/a_share_calendar_daily.csv` | CSV | 88 K | UTF-8 | `index_data.get_a_share_trading_calendar` | 持仓区间对齐、日线 benchmark | **有 guard**（`fetched_max < cached_max` 时拒写） |
| `stock_trade_demo/.cache/{csi1000,chinext,star50}_daily.csv` | CSV | 88K–224K | UTF-8 | `index_data.get_index_daily` | 同上 | **有 guard** |
| `stock_trade_demo/.cache/{csi1000,chinext,star50}_monthly.csv` | CSV | 4K–8K | UTF-8 | `index_data.get_index_returns` | 月度策略 benchmark | 无 |
| `stock_trade_demo/.cache/csi1000_weekly.csv` | CSV | 20K | UTF-8 | 同上 | 周线实验 | 无 |
| `stock_trade_demo/.cache/timing_etf/{idx}_etf_daily{,_qfq}.csv` | CSV | 9K–157K | UTF-8 | `index_data.get_timing_etf_daily` | `/timing` 页面、择时回测 | 无；仅依赖「不复权 < qfq」字段对齐 |
| `stock_trade_demo/.cache/{nasdaq,sp500}_daily.csv` `{,_monthly}` | CSV | 4K–136K | UTF-8 | `index_data` 美股分支 | macro_v32 / 美股择时 | 无 |

### A.3 宏观 / 估值

| 路径 | 格式 | 大小 | 编码 | 上游 | 下游 | freshness |
|---|---|---|---|---|---|---|
| `data/fred_*.csv`（22 个文件） | CSV | 4K–100K | UTF-8 | `scripts/download_macro_data.py` | macro_v32 / 美股择时 | 无 |
| `data/_etf_summary.csv` / `_fred_summary.csv` / `_idx_summary.csv` | CSV | 1B–1K | UTF-8 | `scripts/download_macro_data.py`、`scripts/download_index_data.py` | 元信息 | 无 |
| `data/a_share_macro/{pe_ttm,cn10y,sse_daily}.csv` | CSV | 4K–124K | UTF-8 | `scripts/fetch_a_share_macro.py`（concat + dedup + to_csv） | `scripts/build_risk_signals.py` → 看多风险面板 | 无 |

### A.4 离线策略产物

| 路径 | 格式 | 大小 | 编码 | 上游 | 下游 |
|---|---|---|---|---|---|
| `strategy/best_profile_*_timing.json` | JSON | ~1–3 K | UTF-8 | `scripts/walk_forward_train.py` | `/api/timing/<id>?profile=best` |
| `strategy/walk_forward_log_*.csv` | CSV | 23K–174K | UTF-8 | 同上 | 调参回溯 |
| `strategy/holdout_report_*.{json,md}` | JSON+MD | 1–2K | UTF-8 | 同上 | 回归报告 |
| `strategy/factor_signals_v{2,31,32}.csv` | CSV | 312K–1.8M | UTF-8 | `scripts/macro_timing_strategy_v*.py` | macro_v32 择时 |
| `strategy/backtest_v{2,31,32}_*.csv` | CSV | 537K–1.2M | UTF-8 | 同上 | 同上 |
| `strategy/sector_weekly_heat.csv` | CSV | 980K | UTF-8 | `scripts/compute_sector_weekly_heat.py` | `/api/sector_heat` |
| `strategy/risk_signals.json` | JSON | 4K | UTF-8 | `scripts/build_risk_signals.py` | 看多风险面板 |

### A.5 启动期 / 实盘缓存（pickle / 特殊）

| 路径 | 格式 | 大小 | 上游 | 下游 | 备注 |
|---|---|---|---|---|---|
| `stock_trade_demo/.cache/web_cache.pkl` | pickle | 3.7 M | `web_app._save_cache` 行 233 | `web_app.load_data()` | 已有 `_CACHE_VERSION=14`；写法 `pickle.dump`（**非原子**） |
| `stock_trade_demo/.cache/single_factor_results.pkl` | pickle | 56 K | `stock_trade_demo/build_single_factor_cache.py` | `/api/single_factor_backtest` | 离线生成，写法 `pickle.dump`（非原子） |
| `stock_trade_demo/.cache/us_timing/{sp500,macro_v32}_timing.pkl` | pickle | 1.5 M、2.3 M | `scripts/build_us_timing_cache.py` 行 125 | `/api/timing/<id>` | 非原子 |
| `stock_trade_demo/.cache/weekly_daily_2021_06_2026_05.pkl` | pickle | **583 MB** | `stock_trade_demo/run_weekly_experiment.py` | 同脚本本身、`/api/weekly_experiment` | **只有 weekly 周线实验需要；按 CLAUDE.md「strategy search guidance #2」weekly 已被证伪，是否还要保留需复核** |
| `stock_trade_demo/.cache/{overheat_ab_test,weekly_experiment}_result.json` | JSON | 2–5 K | 对应 `run_*` 脚本 | 研究面板 | 非原子 |
| `stock_trade_demo/.cache/daily_YYYY-MM.pkl`、`rtquotes_YYYY-MM.pkl` | pickle | 临时 | `get_stock_info.supplement_csv*` | 同脚本自身（重跑加速） | **已经是 pickle.dump，非原子；但属于可重生成 cache，破损只是重抓一次** |
| `stock_trade_demo/.cache/daily_stocks/<code>.csv` | CSV | 每股 1–几十 KB | `get_stock_info._save_daily_cache_one` | `supplement_csv_incremental` | **已经原子写**（行 1297–1309，`tmp + os.replace`） |
| `data/live_trades.csv` | CSV | 98 B | `web_app._write_live_trades` | `web_app` 实盘面板 | **已原子写**（行 2923–2930）；CLAUDE.md 第 15 条受保护 |

### A.6 当前已落地的「原子写」点（避免方案重复造轮子）

| 文件 | 行号 | 写入对象 | 原子性 |
|---|---|---|---|
| `get_stock_info.py` 行 1247 | `supplement_csv`（旧路径） | `stock_data.csv` 全表流式重写 | tmp + `os.replace` |
| `get_stock_info.py` 行 1297-1309 | `_save_daily_cache_one` | 每股 daily CSV | tmp + `os.replace` |
| `get_stock_info.py` 行 1676 | `supplement_csv_incremental`（主路径） | `stock_data.csv` | tmp + `os.replace` |
| `web_app.py` 行 2930 | `_write_live_trades` | `data/live_trades.csv` | tmp + `os.replace` |

### A.7 还**没有**原子写的点（本次主要目标）

| 文件 | 行号 | 写入对象 | 问题 |
|---|---|---|---|
| `get_stock_info.py` 行 1014、1030、1613 | `supplement_csv` / `supplement_csv_incremental` 内部 | `.cache/daily_YYYY-MM.pkl`、`.cache/rtquotes_*.pkl` | `pickle.dump` 直写；可重抓但崩溃后会留半截 pickle，下次 `pickle.load` 抛异常需手删 |
| `stock_trade_demo/index_data.py` 行 261、295、392、408、524 | A 股指数日线 / 月线、A 股交易日历、ETF 日线 | 上述 CSV | `df.to_csv(cache_file)` 直写；已有 freshness guard，但 guard 之后的写仍是非原子 |
| `web_app.py` 行 233、3519 | `web_cache.pkl`、`stock_data.parquet` | 同上 | `pickle.dump` / `to_parquet` 直写 |
| `scripts/build_us_timing_cache.py` 行 125 | `us_timing/*.pkl` | 同上 | `pickle.dump` 直写 |
| `scripts/download_index_data.py`、`download_macro_data.py` | FRED CSV、`_*_summary.csv` | 同上 | `df.to_csv(fp)` 直写 |
| `scripts/fetch_a_share_macro.py` 行 60 | `pe_ttm.csv`、`cn10y.csv`、`sse_daily.csv` | 同上 | `combined.to_csv(path)` 直写 |
| `scripts/macro_timing_strategy_v*.py`、`train_test_validation*.py`、`run_sector_heat_backtest.py`、`walk_forward_train.py`、`audit_*.py`、`compute_sector_weekly_heat.py` | 离线产物 | 同上 | 全部 `df.to_csv(...)` 直写，崩溃后留半截 CSV |

---

## B. 原子写入方案

### B.1 统一工具模块

新建 `stock_trade_demo/utils/atomic_io.py`（**纯新增、无破坏**），暴露：

```python
def atomic_write_text(path: str, data: str, encoding: str = "utf-8") -> None: ...
def atomic_write_bytes(path: str, data: bytes) -> None: ...
def atomic_write_csv(df: "pd.DataFrame", path: str, encoding: str = "utf-8",
                     index: bool = False, **to_csv_kwargs) -> None: ...
def atomic_write_parquet(df: "pd.DataFrame", path: str,
                         engine: str = "pyarrow",
                         compression: str = "snappy", **kwargs) -> None: ...
def atomic_write_pickle(obj, path: str, protocol: int = pickle.HIGHEST_PROTOCOL) -> None: ...
def atomic_write_json(obj, path: str, ensure_ascii: bool = False, indent: int = 2) -> None: ...
```

每个函数统一执行：
1. `tmp = path + f".tmp.{os.getpid()}.{uuid4().hex[:8]}"`
2. `os.makedirs(dirname, exist_ok=True)`
3. 调用对应 writer 到 `tmp`
4. `f.flush(); os.fsync(f.fileno())`（关键，否则掉电仍可能不落盘）
5. （可选）对 `tmp` 做 sha256 → 写到 `tmp + ".sha256"`，**和** 数据文件一起 rename
6. `os.replace(tmp, path)` —— POSIX 原子；同分区跨设备失败要 raise
7. （可选）`os.fsync` 父目录 fd（macOS/Linux 都建议）

**注意**：CSV 不能像 parquet 一样把整个 df 一次性传给 writer 再 fsync，因为 `get_stock_info.supplement_csv_incremental` 是「读旧 CSV 同时流式写新 CSV」的 upsert 模式。所以工具要提供两种 API：

```python
@contextmanager
def atomic_writer(path: str, mode: str = "w", encoding: str = "utf-8"):
    """流式写。with 块内向 yield 的 file handle 写；退出时 fsync + os.replace。"""
```

`supplement_csv_incremental` 的 `tmp_path = csv_path + ".tmp"` + `os.replace` 已经是这套模型，直接换成 `with atomic_writer(csv_path, mode="w", encoding="gbk") as fout` 即可，零行为变化、多一层 fsync。

### B.2 `get_stock_info.py` 改造点

| 行号 | 当前 | 改成 |
|---|---|---|
| 1013-1014 | `with open(daily_cache_file, "wb") as f: _pickle.dump(daily_data, f)` | `atomic_write_pickle(daily_data, daily_cache_file)` |
| 1029-1030 | 同上，`rt_quotes` | `atomic_write_pickle(rt_quotes, rt_cache_file)` |
| 1231-1247 | 已是 `tmp + os.replace`，但**没 fsync** | 切到 `with atomic_writer(csv_path, "w", encoding="gbk", newline="") as fout` |
| 1297-1309 | 同上（已原子，缺 fsync） | 切到 `atomic_writer` |
| 1605-1613 | `rt_quotes` pickle 写 | `atomic_write_pickle` |
| 1655-1676 | `supplement_csv_incremental` 主写 | 同 1231-1247 |

附加 crash-recovery 步骤（写在 `supplement_csv_incremental` 入口）：

```python
# Sweep dangling tmp files from previous crashed runs.
for f in glob(csv_path + ".tmp.*"):
    try: os.remove(f)
    except OSError: pass
```

### B.3 `stock_trade_demo/index_data.py` 改造点

5 处 `df.to_csv(cache_file, index=False)` 全部换成 `atomic_write_csv(df, cache_file, index=False)`。**已有的 freshness guard 不动**，只是 guard 通过之后的写入变成原子。

### B.4 `scripts/*.py` 推广

按风险递减分批：
1. **高频更新脚本**（每天/每月跑）：`scripts/download_index_data.py`、`download_macro_data.py`、`fetch_a_share_macro.py`、`build_us_timing_cache.py`、`walk_forward_train.py`、`compute_sector_weekly_heat.py`、`build_risk_signals.py` —— 全换 atomic。
2. **历史一次性实验**（`macro_timing_strategy_v*`、`train_test_validation*`、`run_sector_heat_backtest`、`audit_*`）：可换可不换；建议换，因为成本仅 1 行 import。
3. `stock_trade_demo/web_app.py` 行 233 `pickle.dump(payload, ..., protocol=HIGHEST)` → `atomic_write_pickle(payload, _CACHE_FILE)`；行 3519 `to_parquet` → `atomic_write_parquet`。

### B.5 实盘文件保护

- `atomic_io.py` 顶部 hardcode 一份 `PROTECTED_PATHS = frozenset({os.path.realpath('data/live_trades.csv')})`。
- `atomic_write_*` 在 `os.replace` 之前 assert `os.path.realpath(target) not in PROTECTED_PATHS`，**除非** 调用方显式传 `_live_trades_override=True`（只有 `web_app._write_live_trades` 允许传）。
- 这样任何 grid search / 数据刷新代码都不可能误覆盖实盘文件。

---

## C. 缓存格式收敛

### C.1 决策表

| 文件 | 当前 | 目标 | 理由 |
|---|---|---|---|
| `stock_data.csv` | CSV GBK | **保持** | 老脚本依赖；体量虽大但工具链稳定；并行 parquet 已经存在 |
| `stock_data.parquet` | parquet | **保持**，且变为 read-path 默认（已经是） | 加载比 CSV 快 10× |
| `.cache/{idx}_daily.csv`、`_monthly.csv`、`_weekly.csv` | CSV | **保持** | 体量小（4K–224K），人工排查友好 |
| `.cache/timing_etf/*.csv` | CSV | **保持** | 同上，最大 157K |
| `data/fred_*.csv` | CSV | **保持** | 同上，FRED 上游就是 CSV |
| `data/a_share_macro/*.csv` | CSV | **保持** | 体量小、需要排查 |
| `.cache/web_cache.pkl` | pickle 3.7M | **保持 pickle**，但加 `meta.json` + atomic | 包含混合 Python 对象（DataFrame + dict + 自定义类型），parquet 化代价大 |
| `.cache/single_factor_results.pkl` | pickle 56K | **保持 pickle** | 同上 |
| `.cache/us_timing/*.pkl` | pickle 1.5M / 2.3M | **保持 pickle** | 同上 |
| `.cache/daily_YYYY-MM.pkl`、`rtquotes_*.pkl` | pickle | **降级为 parquet**（daily）+ **保持 pickle**（rtquotes） | daily 是规整 OHLCV，天然 parquet；rtquotes 是 dict-of-dict，pickle 更直接 |
| **`.cache/weekly_daily_2021_06_2026_05.pkl` (583 MB)** | pickle | **重点：评估是否可以删除** | 见下 |
| `.cache/{overheat_ab_test,weekly_experiment}_result.json` | JSON | **保持 JSON** | 小、可读 |
| `strategy/*.csv` / `*.json` / `*.md` | 混合 | **保持** | 离线产物，跨工具消费 |

### C.2 重点：583 MB 的 `weekly_daily_2021_06_2026_05.pkl`

- 唯一上游：`stock_trade_demo/run_weekly_experiment.py` 行 31 写入。
- 唯一下游：同一个脚本 + `stock_trade_demo/stock_data_weekly_experiment.parquet`（同脚本自产自销）。
- 根据 CLAUDE.md「Strategy search guidance #2」：「weekly / true-weekly 选股换仓」已经被列入「不要优先重试」黑名单。
- 结论：**这份 cache 是一次性实验副产物**，价值低于 583 MB 的占盘成本。
- 建议动作：
  1. 第一阶段：将其移到 `stock_trade_demo/.cache/archive/weekly_daily_2021_06_2026_05.pkl`，与活跃缓存分目录，避免被 `data-refresh-all` skill 误扫描。
  2. 加 `.archive_manifest.json` 记录「来源脚本 / 实验结论 / 是否可删除」。
  3. 第二阶段：90 天内若没有再触发 `run_weekly_experiment.py`，连同 `.parquet` 一并删除；脚本里写明「重跑可重生成」。

> 注意：删除前必须 `git log -- stock_trade_demo/run_weekly_experiment.py` 确认最近 30 天没有依赖该实验的新策略 PR。

### C.3 `daily_YYYY-MM.pkl` → parquet（非阻塞性优化）

`get_stock_info.py` 里的 `daily_data` 结构是 `{code: List[{date, open, high, low, close, volume}]}`，flatten 后是规整长表，可以：

```python
df = pd.DataFrame([{**row, "code": c} for c, rows in daily_data.items() for row in rows])
atomic_write_parquet(df, daily_cache_file.replace(".pkl", ".parquet"))
```

跨版本兼容：保留 `.pkl` 优先 fallback 的读路径 1 个 release 周期。

---

## D. Schema 校验方案

### D.1 选型

| 方案 | 适合度 | 备注 |
|---|---|---|
| `pydantic` | ❌ 不适合 | 面向行对象；70万行 stock_data.csv 验证开销过大 |
| 手写 `dict[col → dtype]` | ⚠️ 可用但易腐 | 表多了维护成本指数上升 |
| **`pandera`** | ✅ **推荐** | DataFrameSchema 原生 pandas、有列级 dtype+range+nullable+unique 约束，可 lazy 校验只跑一次，O(n) 内存友好；社区活跃 |

依赖：`pip install pandera` —— 仅纯 Python，不引入新的 C 扩展。

### D.2 校验落点

不是在「读」时校验（启动会变慢），而是在「写之前」校验：

```python
# stock_trade_demo/schemas/stock_data.py
import pandera as pa
from pandera import Column, DataFrameSchema, Check

STOCK_DATA_SCHEMA = DataFrameSchema({
    "交易日期": Column(str, Check.str_matches(r"^\d{4}-\d{2}-\d{2}$"), nullable=False),
    "股票代码": Column(str, Check.str_matches(r"^[03689]\d{5}$"), nullable=False),
    "股票名称": Column(str, nullable=False),
    "收盘价":   Column(float, Check.gt(0), nullable=True),
    "市盈率倒数": Column(float, Check.in_range(-1.0, 1.0), nullable=True),
    "bias_20":   Column(float, Check.in_range(-1.0, 1.0), nullable=True),
    "MACD":      Column(float, nullable=True),
    "下周期每天涨跌幅": Column(str, nullable=True),  # 序列化为 "[-0.01, 0.02, ...]" 字符串
}, strict=False, coerce=False, ordered=False)
```

`atomic_write_csv(df, path, schema=STOCK_DATA_SCHEMA)` 在 rename 前先 `schema.validate(df, lazy=True)`，校验失败 raise + 不替换原文件。

### D.3 关键面板 schema 一览（先做这几个）

| 文件 | 关键校验项 |
|---|---|
| `stock_data.csv` | 见上 |
| `csi1000_daily.csv` / `chinext_daily.csv` / `star50_daily.csv` | `date` 日期格式 + 单调递增 + 无重复；`open/high/low/close` > 0 且 `low ≤ open,close ≤ high`；`volume ≥ 0` |
| `timing_etf/*_etf_daily{,_qfq}.csv` | 同上；额外要求 `_qfq` 文件 `close` 单调可连续（前复权后不应出现跳变 > 30% 且非除权日） |
| `fred_*.csv` | `DATE` 日期、`VALUE` numeric or NaN |
| `live_trades.csv` | **只在 `web_app._write_live_trades` 里做 schema check**，不在通用 atomic 写里 enforce |

---

## E. 统一 freshness 探针

### E.1 CLI 接口

新建 `scripts/check_data_freshness.py`：

```bash
# 全量
python scripts/check_data_freshness.py

# 指定源
python scripts/check_data_freshness.py --source stock_data,csi1000_daily,fred_VIX

# 只输出 stale 的
python scripts/check_data_freshness.py --only-stale

# JSON 输出（给 CI / 监控）
python scripts/check_data_freshness.py --json
```

### E.2 输出 JSON Schema

```json
{
  "generated_at": "2026-05-26T16:30:00+08:00",
  "calendar_today": "2026-05-26",
  "calendar_last_trading_day": "2026-05-26",
  "sources": [
    {
      "name": "stock_data",
      "path": "/Users/fatcat/Desktop/quant/stock_trade_demo/stock_data.csv",
      "format": "csv-gbk",
      "row_count": 703178,
      "max_date": "2026-04-30",
      "min_date": "2017-01-31",
      "staleness_days": 26,
      "expected_freshness_days": 35,
      "status": "ok",
      "produced_by": "get_stock_info.py --mode supplement",
      "consumed_by": ["web_app.load_data", "backtest.run"],
      "meta_file": "stock_trade_demo/stock_data.csv.meta.json"
    },
    {
      "name": "csi1000_daily",
      "path": ".../csi1000_daily.csv",
      "format": "csv",
      "row_count": 5421,
      "max_date": "2026-05-26",
      "staleness_days": 0,
      "expected_freshness_days": 1,
      "status": "ok",
      "produced_by": "stock_trade_demo/index_data.py::get_index_daily",
      "consumed_by": ["a_share_calendar", "timing"]
    },
    {
      "name": "fred_VIX",
      "status": "warn",
      "max_date": "2026-05-23",
      "staleness_days": 3,
      "expected_freshness_days": 2,
      "produced_by": "scripts/download_macro_data.py"
    }
  ],
  "summary": {
    "ok": 28, "warn": 3, "stale": 1, "missing": 0, "broken": 0
  }
}
```

### E.3 staleness 规则注册表

`scripts/check_data_freshness.py` 顶部内置一份 `SOURCES: list[FreshnessSpec]`，每条：

```python
@dataclass
class FreshnessSpec:
    name: str
    path: str
    fmt: Literal["csv-gbk", "csv", "parquet", "pickle", "json"]
    date_col: str | None              # None 表示 mtime 模式
    encoding: str = "utf-8"
    expected_freshness_days: int = 1  # 默认按 1 个交易日
    weekend_aware: bool = True        # 周末/节假日不算 stale
    produced_by: str = ""
    consumed_by: list[str] = field(default_factory=list)
```

`status` 派生规则：
- `staleness_days <= expected_freshness_days` → `ok`
- `< 2 * expected` → `warn`
- 否则 `stale`
- 文件不存在 → `missing`
- `pd.read_*` 抛异常或行数 0 → `broken`

### E.4 集成

- `POST /api/update_data/status` 返回值里加一段 `freshness_summary`，直接调用本探针，让前端那一行「最后更新于 …」变成可信。
- CI 里可以 `python scripts/check_data_freshness.py --only-stale --json | jq` 做硬门控。

---

## F. 数据血缘元数据（`<cache_file>.meta.json`）

### F.1 通用 schema

```json
{
  "schema_version": 1,
  "file": "csi1000_monthly.csv",
  "absolute_path": "/Users/fatcat/Desktop/quant/stock_trade_demo/.cache/csi1000_monthly.csv",
  "format": "csv",
  "encoding": "utf-8",
  "size_bytes": 4475,
  "sha256": "f2c3...e89",
  "row_count": 110,
  "column_count": 2,
  "columns": ["date", "csi1000_return"],
  "date_range": {"min": "2017-01-31", "max": "2026-04-30", "column": "date"},
  "produced_by": {
    "script": "stock_trade_demo/index_data.py",
    "function": "get_index_returns",
    "args": {"index_id": "csi1000", "frequency": "monthly"},
    "interpreter": "/Users/fatcat/opt/anaconda3/bin/python",
    "python_version": "3.11.5"
  },
  "upstream_sources": [
    {"name": "akshare:index_zh_a_hist", "fetched_at": "2026-05-26T09:51:02+08:00"},
    {"name": ".cache/csi1000_daily.csv", "sha256_at_read": "8a1b...77f"}
  ],
  "git": {
    "sha": "acf2ca2",
    "branch": "main",
    "dirty": true
  },
  "generated_at": "2026-05-26T09:51:08+08:00",
  "atomic_write_tool": "stock_trade_demo/utils/atomic_io.py::atomic_write_csv",
  "schema_validated": true,
  "schema_name": "INDEX_MONTHLY_SCHEMA"
}
```

### F.2 写入约定

`atomic_write_*` 接受可选 `meta_extra: dict` 参数；rename 完成后**同时**原子写一份 `<path>.meta.json`（同样 tmp + replace）。两个 rename 不可能同时原子，因此约定：
1. **先写 data，再写 meta**。
2. 读侧若 meta 缺失 / sha256 不匹配，视为「缺血缘」而非「数据损坏」，仍可加载，但 freshness 探针把它标为 `warn`。

### F.3 `csi1000_monthly.csv` 完整示例

见 F.1（即上方 JSON 已是该文件的示例）。

### F.4 不写 meta 的例外

- `data/live_trades.csv` 不写 meta（受保护、字段稳定、用户手工编辑可能改 mtime）。
- 临时 cache（`daily_YYYY-MM.pkl`、`rtquotes_*.pkl`）不写 meta（可重生成）。

---

## G. 迁移步骤（按风险递增 7 步）

> 每一步独立可回滚；每步完成后跑 `python scripts/check_data_freshness.py` + `lsof -nP -iTCP:8080 -sTCP:LISTEN` 验证 web 仍正常。

### Step 1 — 纯新增工具与探针（风险最低）
1. 新增 `stock_trade_demo/utils/atomic_io.py`（不被任何旧代码 import）。
2. 新增 `scripts/check_data_freshness.py`（只读，不写任何旧文件）。
3. 跑 `python scripts/check_data_freshness.py --json` 把当前基线快照存到 `docs/freshness_baseline_2026-05-26.json`，作为回归对照。

**回滚成本：删两个新文件。**

### Step 2 — `get_stock_info.py` 切原子写
1. `_save_daily_cache_one` 切到 `atomic_writer`（已 tmp+replace，只加 fsync）。
2. `supplement_csv_incremental` 主 upsert 同上。
3. `daily_*.pkl` / `rtquotes_*.pkl` 写入切到 `atomic_write_pickle`。
4. 入口加 dangling tmp sweep。
5. 验证：随机选一个月份跑 `python get_stock_info.py --mode supplement --year 2026 --month 5 --max-stocks 20`，对比 `stock_data.csv` md5 与改造前一致。

**回滚成本：恢复改动的 ~10 行；数据文件本身不变。**

### Step 3 — `index_data.py` + `web_app.py` parquet/pickle 切原子写
1. 替换 5 处 `to_csv` 为 `atomic_write_csv`。
2. `web_app._save_cache` 切 `atomic_write_pickle`。
3. `web_app` 行 3519 `to_parquet` 切 `atomic_write_parquet`。
4. 重启 web，跑 `data-refresh-all` skill 验证。

### Step 4 — `scripts/` 推广 + 实盘保护 assert
1. 高频脚本（B.4 第 1 类）改造。
2. `atomic_io.py` 加 `PROTECTED_PATHS` assert。
3. 在 `data/live_trades.csv` 上跑 1 次手工实验：尝试用 `atomic_write_csv` 写它，应抛 `LiveTradesProtectedError`。

### Step 5 — `pandera` schema enforcement
1. 新增 `stock_trade_demo/schemas/` 目录，先做 `stock_data.py` 与 `index_panel.py`。
2. `atomic_write_csv` 增加 `schema=` 参数，先在 `index_data.py` 写指数日线时开启。
3. 跑一次 `data-refresh-all`，确认没拒写。
4. 再开 `stock_data.csv` 的 schema（大表，要小心；可以加 `validate_sample=10000` 子样校验先观察一周）。

### Step 6 — 数据血缘 meta.json 落地
1. `atomic_write_*` 增加 `produced_by` 必填参数；旧调用点传函数自身的 `__qualname__`。
2. 为每个 `.cache/*.csv`、`.cache/*.pkl` 生成首份 meta（即使是旧文件，也用 `python scripts/seed_meta.py` 一次性补齐 sha256 + date_range + git sha）。
3. 把 `check_data_freshness.py` 升级为读 meta（如果缺则降级为 mtime 模式）。

### Step 7 — 583 MB pkl 归档 + 格式收敛收尾
1. 移动 `weekly_daily_2021_06_2026_05.pkl` 到 `.cache/archive/`。
2. `daily_YYYY-MM.pkl` 切到 parquet（保留 1 个版本的 pickle fallback）。
3. 更新 CLAUDE.md「核心数据路径速查」加一行：「`.cache/archive/` 是已归档的实验缓存，可删除」。

---

## H. 不做什么（红线）

1. **不重排** `stock_data.csv` 列顺序，不改 GBK 编码 —— `get_stock_info.py` 与 `backtest.py` 都按列下标取值（如行 1671 `row[54]`），任何改动都是大爆炸。
2. **不动** `data/live_trades.csv` —— CLAUDE.md 第 15 条；本方案唯一允许的操作是给 `_write_live_trades` 内部加 fsync。
3. **不引入** 新的二进制依赖（pyarrow 已有；pandera 是纯 Python）；不引入 Postgres / DuckDB / SQLite 等真正的数据库 —— 项目是单机脚本生态，过度抽象会让 `data-refresh-all` skill 更难维护。
4. **不修改** `_CACHE_VERSION` 的语义 —— 它已经能驱动整库重算，本方案的 meta.json 是补充信息，不替代它。
5. **不在请求路径上跑 schema 校验** —— 只在 `atomic_write_*` 写入路径上跑（CLAUDE.md 第 12 条：Flask 是只读视图）。
6. **不删除** 历史实验产物（`strategy/backtest_*.csv`、`factor_signals_*.csv`）—— 它们是研究证据链。
7. **不动** `stock_trade_demo/run_weekly_experiment.py` 本身的逻辑 —— 只移动它产出的 pkl 到 `.cache/archive/`，并在脚本顶部加注释「输出已归档」。

---

## I. 关键调研发现摘要

1. **原子写已经部分实现**：`stock_data.csv` 主表 + per-stock daily cache + `live_trades.csv` 都已经 `tmp + os.replace`，但 **没有 fsync**，且所有非主表（pickle、indexCSV、FRED、a_share_macro、parquet）依旧直写。
2. **freshness guard 只在 A 股指数日线上有**（`index_data.py` 行 252-260 + 292-298），其它（FRED、US ETF、a_share_macro、ETF 日线）一律「拉到就覆盖」。
3. **583 MB 的 `weekly_daily_2021_06_2026_05.pkl` 是 weekly 实验副产物**，按 CLAUDE.md 它的策略方向已被劝退，应归档。
4. **schema 版本号 `_CACHE_VERSION=14` 只覆盖 `web_cache.pkl`**，与磁盘上 30+ 份其它缓存解耦；本方案的 `meta.json` 才能填这块。
5. **实盘文件保护**目前完全依赖人不犯错；加 `PROTECTED_PATHS` assert 是廉价但高价值的兜底。
