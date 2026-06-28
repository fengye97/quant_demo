# 前端显示界面要求

## 需求检查清单

- [x] 所有需要计算的过程均放在后端计算，前端仅作为展示窗口
- [x] 如果前端需要加载任何东西，请展示进度条，以及对应的加载日志，让使用者知道你在加载什么
- [x] 在前端切换页面时，需要保证充分流畅
- [x] 前端显示的回测收益曲线需要和最新的选股以及择时策略对应

---

## 一、所有计算移至后端

### 已移除的前端计算（index.html）

| 移除的函数 | 原用途 | 替代方案 |
|---|---|---|
| `resampleCurve(curve, resolution)` | 前端按月→季/年重采样，含 `Π(1+r)-1` 复利公式 | 后端 `_resample_curve()` 预计算 |
| `normalizeBenchmarkCurves(curves)` | 前端修复 benchmark 数据结构 | 后端保证 `benchmark_curves` 始终为数组格式 |
| 超额收益计算 `(stratFinal/bmFinal-1)*100` | 前端算 strategy vs benchmark | 后端预计算 `excess_return_pct` |

### 后端预计算字段（web_app.py）

`_resample_curve()` 位于 `web_app.py:472`，在以下位置调用：

| API 响应字段 | 用途 | 数据来源 |
|---|---|---|
| `equity_curve_quarterly` | 策略季线净值曲线 | `_resample_curve(equity_curve, 'quarter')` |
| `equity_curve_yearly` | 策略年线净值曲线 | `_resample_curve(equity_curve, 'year')` |
| `train_equity_curve_quarterly` | 训练集季线 | `_resample_curve(train_curve, 'quarter')` |
| `train_equity_curve_yearly` | 训练集年线 | `_resample_curve(train_curve, 'year')` |
| `test_equity_curve_quarterly` | 测试集季线 | `_resample_curve(test_curve, 'quarter')` |
| `test_equity_curve_yearly` | 测试集年线 | `_resample_curve(test_curve, 'year')` |
| `benchmark_curve_quarterly` | 基准季线 | `_resample_curve(bm_curve, 'quarter')` |
| `benchmark_curve_yearly` | 基准年线 | `_resample_curve(bm_curve, 'year')` |
| `benchmark_curves[].curve_quarterly` | 每基准季线 | 同上 |
| `benchmark_curves[].curve_yearly` | 每基准年线 | 同上 |
| `benchmark_curves[].excess_return_pct` | 策略相对基准超额收益 | 后端计算 |

### 前端现在的角色（纯展示）

- `selectCurveForResolution(monthly, daily, quarterly, yearly, resolution)` — 5 参数纯选择器，无计算
- `selectBenchmarkCurveForResolution(bmItem, resolution)` — 从预计算字段中选择
- `updateOverviewBenchmark(benchmarkCurves)` — 直接读取 `excess_return_pct`
- 周期切换按钮（日/月/季/年）触发 `renderAll()`，**不发起 API 请求**

---

## 二、加载进度条与日志

### 后端 eager loading（web_app.py `__main__`）

服务启动时立即在后台线程预加载，阶段如下：

| 阶段 | `_LOAD_STATUS.stage` | 说明 |
|---|---|---|
| `stock_data` | 加载股票数据 | 读取 823MB CSV (`stock_data.csv`) |
| `index_data` | 加载指数收益 | CSI1000 / 科创50 / 创业板指数月度收益 |
| `strategy_cache` | 预热选股缓存 | 预运行当前专注策略（1 个，非全部 6 个） |
| `timing_cache` | 预热择时缓存 | 预运行全部 3 个择时策略（CSI1000 / 科创50 / 创业板） |
| `ready` | 就绪 | 所有数据加载完成 |

### 状态 API

`GET /api/status` 返回：
```json
{
  "stage": "strategy_cache",
  "message": "正在预热策略回测缓存...",
  "elapsed_sec": 31.0,
  "loading": true,
  "ready": false
}
```

### 前端加载界面（index.html）

- 页面打开即显示 loading overlay
- 每 500ms 轮询 `/api/status`
- 进度条跟随阶段推进：5% → 15% → 40% → 55% → 75-95% → 完成
- 实时显示已用时间（秒 / 分秒格式）
- 显示当前阶段中文描述

### 实测加载时间

| 场景 | 耗时 | 说明 |
|---|---|---|
| 首次启动（无缓存） | ~136s | CSV→Parquet 加载 + 策略计算 + 择时计算 |
| 重启（磁盘缓存命中） | **~14s** | pickle 反序列化 3.8MB，无策略重算 |

各阶段明细（首次）：
- 股票数据 (Parquet 394MB): ~1s（从 CSV 的 14s 优化至 1s，13.7x 加速）
- 指数收益数据: ~3s
- 选股策略缓存 (`original_ensemble`): ~124s（该策略包含 growth timing 双信号逻辑，计算密集）
- 择时策略缓存 (3 个策略): ~5s

### 磁盘缓存机制

缓存文件：`.cache/web_cache.pkl` (~3.8MB pickle)

- 首次启动计算完成后自动保存
- 后续启动优先从磁盘加载，跳过策略重算
- 自动失效条件：
  - 数据文件 (`stock_data.parquet`) 有更新
  - `_CACHE_VERSION` 手动递增（策略代码变更时）
- 每次重启从 136s 降至 **~14s**（约 10x 加速）

> **数据格式**：已从 CSV (823MB) 转换为 Parquet (394MB, Snappy 压缩)。`load_data()` 自动优先读取 Parquet。

---

## 三、页面切换流畅性

### 周期切换（日/月/季/年）

- 所有周期曲线在后端预计算
- 切换时不发起任何网络请求
- 直接调用 `renderAll()` 从已加载数据中选取对应精度曲线
- ECharts 使用 `requestAnimationFrame` + `chart.resize()` 保证渲染平滑

### 页面间导航

- 选股页 ↔ 择时页：通过 `window.location.href` 跳转
- 择时页有独立的 loading 机制（`timing.html` 内建）
- 两个页面共用同一 Flask 进程，数据预热后均受益

### 策略 / 基准切换

- 策略变更 → 加载新因子配置 + 需点击"运行回测"触发
- 基准变更 → 自动触发回测（onchange="runBacktest()"）
- 回测期间显示 loading 遮罩，完成后隐藏

---

## 四、策略对应关系

### 选股策略（STRATEGY_MAP, web_app.py:59）

| ID | 策略名称 | 说明 |
|---|---|---|
| `original` | 原始小盘策略 | 基础 small-cap |
| `original_ensemble` | 择时门控集成策略 | **当前默认策略** (FOCUSED) |
| `chan_enhanced` | Chan 增强策略 | 市值排序 + Chan 倾斜 |
| `chan_only` | Chan 纯策略 | Chan + 规模的加权混合 |
| `method_a` | 方法A策略 | 日级 Chan pipeline 聚合 |
| `quality_value` | 质量价值策略 | Z-score 多因子线性 |

当前专注策略 (`FOCUSED_STRATEGY_ID`): **`original_ensemble`**

### 择时策略（TIMING_STRATEGY_MAP, web_app.py:68）

| ID | 策略名称 | 对应指数 |
|---|---|---|
| `csi1000_timing` | CSI1000 择时 | CSI 1000 |
| `star50_timing` | 科创50 择时 | 科创50 |
| `chinext_timing` | 创业板择时 | 创业板 |

### 回测基准

| 基准 ID | 名称 | 数据来源 |
|---|---|---|
| `csi1000` | CSI 1000 指数 | `.cache/csi1000_monthly.csv` |
| `star50` | 科创50 指数 | `index_data.py` → 指数面板 |
| `chinext` | 创业板指数 | `index_data.py` → 指数面板 |

### 训练 / 测试集拆分

- 拆分日期: **2026-03-31**
- 训练集: ≤ 2026-03-31
- 测试集: > 2026-03-31

---

## 五、关键文件与入口

| 文件 | 用途 |
|---|---|
| `stock_trade_demo/web_app.py` | Flask 后端，所有计算 + API |
| `stock_trade_demo/web/templates/index.html` | 选股前端，纯展示 |
| `stock_trade_demo/web/templates/timing.html` | 择时前端，纯展示 |
| `stock_trade_demo/backtest.py` | 回测引擎 |
| `stock_trade_demo/timing.py` | 择时策略实现 |
| `stock_trade_demo/strategies/` | 6 个选股策略 |
| `stock_trade_demo/stock_data.csv` | 原始数据 (823MB) |

启动命令: `cd stock_trade_demo && python3 web_app.py`
访问地址: `http://localhost:8080`
择时页: `http://localhost:8080/timing`

---

## 六、已知注意事项

1. **Flask `debug=False`**：模板文件修改后必须重启服务器才能生效（模板缓存在内存中）
2. **首次加载**：823MB CSV (~14s) + `original_ensemble` 策略 (~127s) + 择时缓存 (~2s) ≈ 146s。后续请求秒级响应。`init_cache()` 只预热 `FOCUSED_STRATEGY_ID`，不做全量预热
3. **数据截止日期**：由 `stock_data.csv` 最新日期决定，当前为 2026-05-15
4. **自定义日期范围**：选择起止日期后，回测资本重置为 100k，独立计算周期曲线
5. **择时页面参数**：通过 URL query string 传递（`growth_timing_mode`, `weight_3y` 等），后端解析后运行回测
