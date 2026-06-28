---
name: data-refresh-all
description: 把当前所有策略所需要的数据（A 股月度数据 + 指数日线/月度 + 择时 ETF 日线）全部刷新到最新交易日，并验证 Web 端确实读到了新数据。任何"数据落后于最新行情"或"更新数据"类需求都走这个 skill。
---

Use this skill when the user asks to update / refresh the dataset, complains that prices look stale, or before running a fresh round of backtests / walk-forward training that depends on the most recent month.

## Rule

数据刷新必须通过 Web 端两条后台流程完成，**不要**直接绕过去手动跑 `get_stock_info.py` 或 `index_data.py`，否则在内存的回测缓存（`BACKTEST_CACHE` / `TIMING_CACHE` / `web_cache.pkl`）不会失效，前端看到的仍然是旧结果。两条流程都是后台线程，必须轮询 `/status` 直到 `stage == done`，再做前端截图验证。

## Required workflow

1. **确认前端服务在跑**：`lsof -nP -iTCP:8080 -sTCP:LISTEN` 找到 pid，确认是 `/Users/fatcat/opt/anaconda3/bin/python web_app.py`。如果没在跑，按 CLAUDE.md 的方式用 Anaconda Python 重新启动。
2. **记录基线日期**（用于事后比对刷新是否生效）：
   - `stock_data.csv` 最新一行的日期
   - `stock_trade_demo/.cache/timing_etf/csi1000_etf_daily.csv` 末行日期
   - `stock_trade_demo/.cache/csi1000_daily.csv` 末行日期
3. **同时触发两条更新**（彼此独立，可以并行）：
   ```bash
   curl -s -X POST http://localhost:8080/api/update_data         # 股票月度数据
   curl -s -X POST http://localhost:8080/api/update_index_data   # 指数 + ETF + 重建择时缓存
   ```
   两个接口都立刻返回 `{"status":"started"}`，真正的工作在后台线程里跑。
4. **分别轮询状态**直到两条都跑完：
   - `GET /api/update_data/status` → 等 `"stage":"done"`，新版增量阶段顺序：
     `fetching (10%) → rebuilding parquet (72%) → rebuilding_cache (75–95%) → done (100%)`。
     首次冷启 seed 全部 5000+ 股票 ~6 分钟；之后只补差值（cache 命中），整段耗时约 1–2 分钟。
   - `GET /api/update_index_data/status` → 等 `"stage":"done"`，进度从 0 → 100，包含 5 个指数日 K + 月度 + ETF 日线 + 重建择时面板与缓存，整段耗时约 1–3 分钟。
   - 如果 stage 卡在 `error`，立刻把 status JSON 里的 `message` 报给用户，不要继续。
5. **校验产物时间戳确实推进了**：
   - `stock_data.csv` 行数增加，末日期不早于上一个交易日。
   - `stock_trade_demo/.cache/timing_etf/{csi1000,chinext,star50}_etf_daily.csv` 末行日期推进，对应 `_qfq.csv` 也同步推进。
   - `stock_trade_demo/.cache/{csi1000,chinext,star50,nasdaq,sp500}_daily.csv` 末行日期推进，对应 `_monthly.csv` 末月份对应。
   - 如果今天是非交易日（周末/节假日），允许只更新到最近一个交易日，不要为此报错。
6. **前端截图验证**（强制，走 `frontend-screenshot-verify` skill）：
   - `http://localhost:8080/` 顶部 "股票数据: YYYY年M月D日 ~ YYYY年M月D日" badge 已更新到新最大日期。
   - `http://localhost:8080/timing` 三张择时卡片（CSI1000 / 创业板 / 科创50）的"最新数据日期/最新交易日"已经推进，没有"暂无数据/cache miss"。
   - `http://localhost:8080/us-timing`（如有）确认 NASDAQ / S&P500 ETF 卡片日期也是新的。
7. 如果上面任何一个断言失败，**不要**声称更新成功；先回头看 `/api/update_*/status` 的 `error` / 服务端日志，再决定是否重跑。
8. **刷新实时风险因子**（让 live 页的「看多风险 / 看空机会」面板基于最新 VIX / 美债利率 / A 股 ERP / 换手率 / 融资买入占比 等重新触发）：
   ```bash
   # (a) 先拉 A 股估值与情绪因子（PE-TTM 中位数、CN10Y、上交所成交+换手+融资），单跑 ~30s–2min
   /Users/fatcat/opt/anaconda3/bin/python scripts/fetch_a_share_macro.py
   # (b) 再用所有因子产出 risk_signals.json（含美股宏观 + 各指数技术因子 + A 股估值情绪）
   /Users/fatcat/opt/anaconda3/bin/python scripts/build_risk_signals.py
   ```
   预期 stdout：
   - `fetch_a_share_macro`: 三个 csv tail 行显示最新日期。
   - `build_risk_signals`: `[OK] strategy/risk_signals.json` 以及当日 VIX、10Y、各策略触发条数。
   - 失败不影响主流程（web 会自动回退到静态文案），但应汇报给用户。
   - A 股 a_share_macro 的 fetch 是 AkShare 接口，偶尔会单日报 `Length mismatch` —— 这是 AkShare 对非交易日返回畸形 frame 的已知行为，已被静默跳过，不要当成错误。

## 涉及的入口与产物

| 流程 | HTTP 入口 | 内部函数 | 产物 |
| --- | --- | --- | --- |
| 股票月度数据（增量） | `POST /api/update_data` | `web_app._run_data_update` → `get_stock_info.supplement_csv_incremental` → `fetch_daily_batch_incremental` | `.cache/daily_stocks/<code>.csv`（per-stock 日线，原子换名，restart-safe）+ `stock_data.csv` upsert 当月行 + 回填上月 `下周期每天涨跌幅` |
| Parquet 同步 | 同上（fetch 之后自动） | `pd.read_csv(...,encoding='gbk').to_parquet(snappy)` | `stock_data.parquet` 重新生成。**关键**：`backtest.load_data()` 优先读 parquet，不刷新会读到陈旧日期。失败时自动删除旧 parquet 强制 CSV fallback。 |
| 指数 + ETF | `POST /api/update_index_data` | `web_app._run_index_data_update` → `index_data.get_index_daily/get_index_returns/refresh_all_timing_etf_daily` | `.cache/*_daily.csv`、`.cache/*_monthly.csv`、`.cache/timing_etf/*.csv` |
| 内存缓存重建 | 上面两条流程末尾自动做 | `BACKTEST_CACHE / TIMING_CACHE / TIMING_PANEL` 清空 + `init_cache / init_timing_cache` + 写回 `web_cache.pkl` | 前端 `/api/backtest`、`/api/timing/*` 立刻拿到新结果 |

## 注意事项

- **绝不要并发触发同一条更新**：两个 endpoint 都对 `running` 上锁，重复 POST 会得到 HTTP 409。
- **不要用 `/opt/homebrew/bin/python3` 跑 web_app.py**：缺包；并且会启动一个新进程占住 8080，导致原来的更新线程"丢失"。
- **不要在更新过程中重启服务**：后台线程是 daemon thread，进程一退就死，已经拉到一半的数据可能丢；如果必须重启，等当前 `stage` 不是 `fetching` 之后再做。per-stock 缓存是原子写入，**已落盘的部分 restart-safe**，但 stock_data.csv 的 upsert 是流式重写，中途打断会留下残缺 csv，必须重跑。
- **ETF 历史不允许伪造**：若用户报告某 ETF 日期没推进，先确认这只 ETF 的真实交易日，再决定是否是上游数据问题；不要为了让 UI 好看而 forward-fill。
- **轮询节奏**：每 8–15 秒一次足够。`fetching` 阶段进度长时间停留在 10% 是正常的（拉 5000+ 个股票），不是卡死。
- **Sina 限流**：`fetch_daily_batch_incremental` 默认 `max_workers=8`（之前的 20 会在 ~2500 请求后触发 15+ 分钟封禁）。`_fetch_incremental_single` 用 `fresh_gap_days=3` 跳过 3 天内已缓存的股票，进一步减少 API 调用。如果 errors 占比 > 10%，停掉等 10 分钟再重试。
- **Parquet/CSV 双源陷阱**：`backtest.load_data()` 优先读 parquet。任何绕过 `_run_data_update` 直接改 CSV 的脚本，都必须同时刷新 parquet 或删除它，否则 `/api/info` 的 `data_max_date` 会停留在旧日期。
- 整个流程完成后，可以用一句话把"更新了哪些数据 + 新的最大日期"汇报给用户，附上前端截图路径。
