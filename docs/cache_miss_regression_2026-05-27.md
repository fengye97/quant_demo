# Cache-miss 回归事故复盘（2026-05-27）

> 事故定位：load-only 改造（Pillar 1 Step 6）+ 请求层参数 schema 化（保护 6）联合作用，
> 导致 `/timing`、`/us_timing` 默认页打开后所有 timing 卡片显示 "—"，全部择时回测 API 直接 400。
>
> 影响时间：2026-05-27 当日（Pillar 1 Step 4/6 与保护 6 合并落地之后到本文档归档之间）。
> 严重程度：P0（**只读视图层完全不可用**，但实盘记录、离线产物与磁盘缓存均未损坏）。

---

## 1. 现象

### 1.1 用户观察
- 打开 `http://localhost:8080/timing`：三张择时卡片（csi1000 / chinext / star50）全部
  显示 "—"，下方曲线区不渲染，右上角红色 banner 提示 cache_miss / HTTP 400。
- 打开 `http://localhost:8080/us_timing`：两张卡片（macro_v32 / sp500）同样症状。
- 首次报告时点：2026-05-27（Worker C 收到 team-lead 通报）。
- 之前的页面行为：相同 URL 能拿到完整曲线与最新信号；改造前 cache miss 路径会**现算**
  （`run_timing_backtest_fresh`），违反 CLAUDE.md 第 12 条，但用户感觉是"能用"。

### 1.2 可复现 curl
默认页加载时浏览器实际发出的请求（前端 `collectParamsForStrategy` 把所有 slider 默认值
写进 query string，再加 `strategy` / `compact` / 可选的 `start` / `end`）：

```bash
# A 股 csi1000：默认参数全量回放
curl -sS 'http://localhost:8080/api/timing/backtest?breakout_window=15&exit_window=7&trend_window=50&exposure_mode=staged&enter_threshold=0.55&add_threshold=0.75&trim_threshold=0.38&exit_threshold=0.18&confirm_days=1&max_entry_exposure=0.5&probe_entry_exposure=0.25&probe_confirm_days=1&profit_lock_enabled=true&profit_lock_drawdown=0.05&slippage_bps=5&cash_interest_rate=0.02&commission_rate=0.0003&commission_min=5&stamp_tax_rate=0&transfer_fee_rate=0.00001&limit_max_delay_days=5&base_floor=0&strategy=csi1000_timing&compact=1'
# → HTTP 400 {"error":"cache_miss","strategy":"csi1000_timing","message":"...缓存缺失或参数与默认值不一致..."}

# 美股 macro_v32：默认参数全量回放
curl -sS 'http://localhost:8080/api/us_timing/backtest?sigmoid_k=1.2&max_leverage=1.4&base_position=0.45&inertia=0.05&crisis_vix=40&fed_block_weight=0.25&restrictive_threshold=0.4&pivot_relief=0.6&exposure_mode=staged&enter_threshold=0.55&add_threshold=0.75&trim_threshold=0.35&exit_threshold=0.15&confirm_days=1&max_entry_exposure=1&profit_lock_enabled=true&profit_lock_drawdown=0.05&slippage_bps=5&cash_interest_rate=0.02&commission_rate=0.0003&commission_min=5&stamp_tax_rate=0&transfer_fee_rate=0.00001&limit_max_delay_days=5&base_floor=0&strategy=macro_v32_timing&compact=1'
# → HTTP 400 同上

# 对照：完全不带参数的请求（仅 strategy + compact）能正常返回 200
curl -sS 'http://localhost:8080/api/timing/backtest?strategy=csi1000_timing&compact=1' | jq '.metrics // .error'
# → 拿到 metrics（cache hit 成功）
```

### 1.3 行为差异本质
"默认页加载"与"显式无参请求"在 HTTP 层是两条完全不同的 query string，但语义上指向同一组
参数。回归前两条路径都能现算兜底；回归后**只有完全无参那条**还能走 cache hit。

---

## 2. 根因

### 2.1 直接原因：默认值回放 + 严格哈希比对，两者单独无害，叠加致命

**前端侧（保护 6 落地后的 schema 化副作用之一）**

`stock_trade_demo/web/templates/timing.html:1016-1024` 与
`stock_trade_demo/web/templates/us_timing.html:896-904` 的 `collectParamsForStrategy(sid)`
会遍历当前页面里所有 `.factor-slider`，把 **每个 slider 的 `el.value`**（即接口
`/api/timing/params` 返回的 `default`）都 `params.append()` 进 `URLSearchParams`。
然后 `fetchBacktest` 拼出 `${qs.toString()}` 直接打到 `/api/timing/backtest`。

→ 结果：即使用户没有手动改任何 slider，**所有 timing 参数的默认值都被显式塞回 URL**。
对前端而言这只是"忠实重放当前 UI 状态"，没有任何主动变更。

**后端侧（Pillar 1 Step 6 load-only 改造）**

`stock_trade_demo/web/blueprints/timing_api.py:138-159` 与
`stock_trade_demo/web/blueprints/us_timing_api.py:161-182` 用以下方式判定 cache 命中：

```python
use_cache = strategy_name in state.TIMING_CACHE
if use_cache and strategy_name in state._TIMING_CACHE_DEFAULTS:
    for key, default_val in state._TIMING_CACHE_DEFAULTS[strategy_name].items():
        raw_val = request.args.get(key)
        if raw_val is None:
            continue                                  # 没显式传 → 视为用 default → OK
        try:
            ... 解析 raw_val 为 bool / str / int / float ...
        except (TypeError, ValueError):
            use_cache = False; break                   # 解析失败 → cache miss
        if val != default_val:                         # 严格 !=
            use_cache = False; break

if not use_cache:
    return jsonify({'error': 'cache_miss', ...}), 400  # 不再现算
```

注意三件事：
1. 命中条件是「**所有显式出现的参数值都与 `_TIMING_CACHE_DEFAULTS` 字面相等**」。
2. 比对用 Python `!=`，没有任何 `float` 容差。
3. 对应离线脚本 `scripts/build_timing_cache.py` / `scripts/build_us_timing_cache.py` 只
   覆盖 "零额外参数"那一组缓存——任何 query 一旦显式带了值，就必须**字面**等于
   `_TIMING_CACHE_DEFAULTS` 里那条 dict 的值才能命中。

**命中失败的细微触发点（不是 BUG，是字面相等比对天然脆弱）**

- 前端 slider `step` 给的是 `0.05`，`default` 给的是 `0.45`，但
  `<input type="range">.value` 在某些组合下会变成 `'0.45'` 字符串，经
  `float('0.45')` 得到 `0.45`，**这一段是 OK 的**。
- 真正常态失败的是「默认值的 *形态* 不一致」，例如：
  - 后端 `_TIMING_CACHE_DEFAULTS` 里写 `'enter_threshold': 0.55`，但某次 build / 某个
    策略 patch 把它改成 `0.5500000000000001`（浮点累积）—— `!=` 立即破。
  - 后端默认 `0`，前端 slider 给的是 `'0'` 字符串，`float('0') == 0` 成立——这条 OK；
    但若默认是 `False`，前端发 `'false'`，需要走 `_parse_realism_bool`，任何
    `bool/str` 误判都会 break。
  - 前端 slider `default = 0.05`，后端默认 `_SHARED_REALISM_DEFAULTS` 也是 `0.05`，
    但只要某次离线脚本读到的 cache 里默认是 `0.0500001`（来自浮点写盘），就会全策略
    一次性 cache miss。
- **所以问题不在"前端发错值"，问题在"命中策略本身依赖跨进程字面相等"**，
  违反了 Postel 法则。

**叠加效果**：前端把默认值全部回放 → 后端用字面 `!=` 比对 → 任何一项不字面相等就
全策略 cache miss → load-only 改造禁用了现算兜底 → **直接 HTTP 400**。

> 📌 **修复阶段才看清的更深一层**：上面这段把"字面 `!=`"当成主因，但 Worker A 落地时
> 发现真正的根因不只是比对方式脆弱，而是**比对基准从一开始就不对**——blueprint
> 拿的是 `_TIMING_CACHE_DEFAULTS[strategy]` 原始 dict，而 `build_*_strategy(sid)` 实际
> 构造策略实例时还会再叠 `best_profile_*_timing.json` + 策略类硬编码默认。
> "blueprint 看到的默认" ≠ "策略真正用的默认"。再加上美股入口缺一次 `init_us_timing_cache()`
> lazy 触发、HTML range slider 的 `step` round 把 `0.38` round 成 `0.4`，三层叠加才是
> 完整故事。详见 §4.1。

### 2.2 深层原因：Task #3 load-only 改造只考虑「用户主动改参数」路径

回看 0526_todo.md 中 Pillar 1 Step 6 的验收口径：
> 写 `scripts/build_select_cache.py`，把全策略 × 默认参数预跑写盘
> cache miss 时返回 503 + 指向脚本路径，不再现场重算

设计假设：
- "默认页 / 首次打开" → 前端不会传任何参数 → 走 `request.args.get(key) is None` →
  跳过所有 default 比对 → cache hit。
- "用户拖动 slider 调参" → 前端传了**新值** → 走严格比对 → 大概率 miss → 返回 400
  + 提示用户去跑离线脚本。

实际情况：
- 前端 `collectParamsForStrategy` 在保护 6 落地前后**行为没变**，一直都是回放全部
  slider 值。改造前 cache miss 会现算，所以"默认页加载"碰巧不暴露问题；改造后这条
  路径变成"用户没改参数也算用户改了参数 → 立即报错"。
- 测试覆盖（`tests/test_timing_realism.py` 等）只测显式参数变化 / 引擎不变量，没有
  一条覆盖"完全用 default 打开默认页"这条用户高频路径。
- TimingParams dataclass（保护 6）只承担「**类型转换** + 集中收口」职责，并没有承担
  「**判断该不该走 cache**」职责——后者继续散落在两个 blueprint 里的内联循环。所以
  schema 化没有顺手解决 default-equality 的问题。

### 2.3 一句话总结
load-only 改造把"现算兜底"砍掉的同时，没有把"前端默认值回放也算命中"这条契约写进
后端 cache 命中策略；TimingParams dataclass 又没接管命中判断，结果默认页加载落到了
"前端忠实回放 default → 后端字面对不上 → 没有兜底 → 400"。

---

## 3. 影响面

### 3.1 受影响（默认页打开即坏）
- 页面 `/timing`：3 张择时卡片 csi1000_timing / chinext_timing / star50_timing
  - API：`GET /api/timing/backtest`
- 页面 `/us_timing`：2 张择时卡片 macro_v32_timing / sp500_timing
  - API：`GET /api/us_timing/backtest`
- 页面 `/timing/explore_compare`（理论上）—— 但这一项早在 Step 6 时就被显式 `force=1`
  网关挡住（`timing_api.py:248`），用户侧不感知。

### 3.2 **没有**被影响
- `/api/timing/panel`、`/api/timing/<index_id>/profile=best`、`/api/us_timing/strategy_list`
  等卡片摘要类 API：直接读 `state.TIMING_CACHE` 末行，不经过 `_TIMING_CACHE_DEFAULTS`
  比对，与本次 cache 命中策略无关。
- 选股策略 `/api/backtest`、`/api/factors`、`/api/strategy_list`：完全独立的代码路径，
  Step 6 的 load-only 改造没有触达月度选股侧（仍在 `web_cache.pkl`）。
- 实盘 `/api/live/*`：单独的 `live_api` blueprint + 文件锁，与 timing cache 完全无关。
- 数据更新 `/api/update_data` / `/api/update_index_data`：触发的是离线脚本，与 cache
  命中判断无关。
- 任何离线产物（`strategy/walk_forward_log_*.csv`、`strategy/best_profile_*_timing.json`、
  `stock_trade_demo/.cache/web_cache.pkl`、`stock_trade_demo/.cache/us_timing/*.pkl`）：
  **未被任何写入**，没有数据损坏。

### 3.3 与 CLAUDE.md 红线的关系
- 没触 `data/live_trades.csv`（PROTECTED_PATHS 保护、文件锁、与本路径完全隔离）。
- 没触发 `pkill -f`、没重启 8080。
- 但**违反了 CLAUDE.md 第 14 条产品目标的精神**：默认页一打开就 400 等于策略对用户
  完全不可用——比"跑输 ETF"更严重，等同于把整个择时模块下线。

---

## 4. 修复方案（Worker A 落地）

### 4.1 修复后才看清的真实根因（3 层叠加）

进入实际修复时发现 §2.1 写的"前端回放 + 后端字面 !=" 只是表层；真正让命中判定失败的
比对**基准本身就错了**，并且漏了一次 lazy init。3 层根因如下：

- **L1 — 比对基准与真实策略实例不一致**：blueprint 拿 `_TIMING_CACHE_DEFAULTS[strategy]`
  这份"原始 dict"和 query 比；但 `build_*_strategy(sid)` 实际构造策略实例时还会再叠
  `best_profile_*_timing.json` + 策略类硬编码默认。结果是「blueprint 看到的默认」与
  「策略实例真正用的默认」长期不一致，只要 `best_profile` 改过任何一个值，下一次
  默认页加载就 100% miss。
- **L2 — `/api/us_timing/backtest` 没触发 lazy init**：A 股侧入口在别处会先 `init_timing_cache()`，
  美股侧入口缺这一行，第一次访问时 `state.US_TIMING_CACHE` 为空，第一层
  `use_cache = strategy_name in state.US_TIMING_CACHE` 就 False。这与 L1 独立但同向。
- **L3 — 前端 slider step round 与后端 instance 原值口径不一致**：HTML
  `<input type=range step="0.05">` 浏览器会把 best_profile 给出的非网格值
  （例如 `0.38`）round 到 `0.4` 再回放；后端 instance 原值是 `0.38`。即便 L1 修了基准，
  这里仍会差一个 round step。

### 4.2 实际改动（PID 94496 v3，5/5 全绿）

| 层 | 改动点 | 文件:行 | 作用 |
|---|---|---|---|
| L1 | 新增 `get_effective_timing_defaults()` / `get_effective_us_timing_defaults()` / `get_effective_select_defaults()`：从 `build_*_strategy(sid)` 实例属性提取 effective defaults（自动覆盖 `_CACHE_DEFAULTS` + `best_profile` + 类硬编码），加进程内 cache | `stock_trade_demo/web/state.py:927-988` | 让"用于比对的默认"与"策略真正用的默认"对齐 |
| L1 | 新增 `TimingParams.diff_from_defaults()` / `is_all_defaults()`：float `abs(a-b)<=1e-9` 容差；bool/str/int 严格 `==` | `stock_trade_demo/web/params.py:200-238` | 把命中判定下沉到 schema 层，blueprint 不再各自 inline |
| L1 | A 股 blueprint 改用 `state.get_effective_timing_defaults` + `timing_params.diff_from_defaults` | `stock_trade_demo/web/blueprints/timing_api.py:136-156` | 替换原来的字面 `!=` 循环 |
| L1 | 美股 blueprint 同上，用 `get_effective_us_timing_defaults` | `stock_trade_demo/web/blueprints/us_timing_api.py:160-176` | 同上 |
| L1 | 选股 blueprint `/api/backtest` 也顺手用 `get_effective_select_defaults` + 容差 | `stock_trade_demo/web/blueprints/select_api.py:93-130` | 同种类回归的预防 |
| L2 | 美股入口加 lazy `state.init_us_timing_cache()` | `stock_trade_demo/web/blueprints/us_timing_api.py:185-191` | 首次访问不再因 cache 未初始化直接 miss |
| L3 | `_extract_effective_from_instance` 末尾按 `strategy.get_signal_metadata().parameters[k].step` 做 `round(round(val/step)*step, 10)` snap | `stock_trade_demo/web/state.py:932-983` | 让后端基准与浏览器 `<input type=range>` round 后的值口径一致 |

### 4.3 验收

- PID 94496 v3：csi1000 / star50 / chinext / sp500 / macro_v32 全 200，rows
  21 / 6 / 28 / 24 / 4。
- 反向用例：`breakout_window=20`（非默认值）确认仍正确返回 400 `cache_miss`，没把所有
  miss 都吞成 200（避免变成新坑）。
- **全程无任何 `run_*_fresh` web 内现算兜底**，cache miss 仍按 Pillar 1 Step 6 返回 400
  + 指向 build script，符合 CLAUDE.md 第 12 条。

---

## 5. 保护措施

### 5.1 代码级（Worker B 落地）

回归测试核心断言：**默认页打开（前端真实 query string）→ 后端 200 + 数据非空**。

**测试文件**：`stock_trade_demo/tests/test_default_page_load_no_cache_miss.py`（13 个 case）

| 类别 | case 数 | 覆盖 |
|---|---|---|
| timing HTTP | 3 | csi1000_timing / chinext_timing / star50_timing 默认页全字段回放 → 200 |
| us_timing HTTP | 2 | macro_v32_timing / sp500_timing 默认页全字段回放 → 200 |
| select HTTP | 1 | original_ensemble focused 默认页回放 → 200 |
| lint-style | 5 | 五个注册策略各跑一遍 schema-level 对齐校验（避免下次再 schema drift） |
| L2 lazy init 守卫 | 1 | `test_us_timing_backtest_lazy_inits_cache_on_first_hit`（line ~290）——fixture clear `US_TIMING_CACHE` 模拟冷启动，请求必须 200。守住 `us_timing_api.py:185-191` 的 `init_us_timing_cache()` 调用不被误删。 |
| TimingParams schema 双向一致 | 1 | `test_timing_params_schema_alignment`（line ~330）——`_BOOL_KEYS ∪ _INT_KEYS ∪ _FLOAT_KEYS ∪ _STR_KEYS` 必须等于 `TimingParams` dataclass 字段名集；双向 fail message 分别覆盖"KEYS 多列"与"dataclass 多声明"。 |

**关键复刻细节**：
- 测试里用 `_snap_to_step` 函数复刻 HTML `<input type=range step="...">` 的客户端
  量化行为（与 backend `_extract_effective_from_instance` 末尾的 snap 逻辑一致），
  确保测试发送的"前端真实 query string"和浏览器发出的字节完全等价。
- 主体 HTTP/lint case 用 `autouse` module-scope fixture 一次性 `init_*_cache()`，
  让所有 case 走与线上一致的 lazy init 路径；**L2 lazy init 守卫 case 单独用
  function-scope fixture 在请求前 clear `US_TIMING_CACHE`**，主动绕开 module 级预热以
  模拟进程冷启动——这才是真正能抓住"入口忘了调 `init_us_timing_cache()`"的姿势。

**两轮验证证据**（A/B/regression-style 验收）：
- **Pre-fix**：monkey-patch `get_effective_*_defaults()` 回退到读裸 `_*_CACHE_DEFAULTS`，
  模拟 2026-05-27 故障态 → **10 failed, 1 passed**，failure message 含 diff 字典
  形如 `diffs={'breakout_window': 10, 'exit_window': 5, ...}`，证明测试**能抓**回归。
- **Post-fix**：当前 main 直接跑全部 13 case → **13 passed in 7.54s**，证明本次修复
  **真的修好**。
- **L2 反向验证**：monkey-patch `state.init_us_timing_cache` 成 no-op + clear cache →
  返回 400 cache_miss，新增的 L2 守卫 case 正确爆炸。

这套"先反向验证测试能抓 bug，再正向验证修复让它绿"的两轮证据，比单跑一次 pass
更能证明测试是有效的护栏（而不是写了个永远绿的空壳）。

### 5.2 流程级
- **load-only 改造的验收清单必须包含"默认页加载"E2E**：以后但凡再做"砍掉现算兜底"
  类改动，验收脚本必须模拟前端在**不动任何 UI** 的状态下发出的 query string，断 200。
  仅断"显式无参 + cache hit"是不够的。
- **比对基准必须用 effective defaults，不能用静态 dict**：本次修复已把
  `get_effective_*_defaults()` 落到 `web/state.py:927-988`，从 `build_*_strategy(sid)`
  实例属性抽取并叠加 `best_profile + 类硬编码 + slider step round`。以后再加任何
  "请求层 vs cache 默认"比对，必须走这套 effective getter，不要再回到读 `_*_CACHE_DEFAULTS`
  原始 dict 的老路。
- **TimingParams（保护 6）已开始承担命中判定职责**：本次新增 `diff_from_defaults()` /
  `is_all_defaults()`（`web/params.py:200-238`），float 容差 1e-9，其它严格 ==。
  以后再加新参数类型时，记得在这两个方法里补对应的比对分支。
- **前端契约**：`collectParamsForStrategy` 回放全部 slider 值是合理的（用户可见 ≡
  服务端收到），不要为了"省 query string"去客户端剔除"等于 default"的字段——那只是把
  契约脆弱性藏起来，将来调试更难。前端不动，后端必须容忍。
- **每个 timing 入口都必须 lazy `init_*_cache()` 一次**：本次美股入口因为漏了这一行
  导致首次访问 100% miss。新增择时类 API 入口时检查这条。已由
  `test_us_timing_backtest_lazy_inits_cache_on_first_hit`（清空 cache 模拟冷启动）守护，
  以后所有 `/api/.../backtest` 入口若依赖 module-level cache 必须自身调用 `init_*_cache()`，
  不能假设 `/strategy_list` 已经被访问过。
- **TimingParams schema drift lint**：dataclass 字段集与 `_BOOL/INT/FLOAT/STR_KEYS`
  四个集合的并集必须双向一致。已由 `test_timing_params_schema_alignment` 守护，
  以后新增/删除字段任何一边漏改都立刻在 pytest 阶段爆炸，不再依赖人工 review。

### 5.3 监控级
- Web 进程日志里 `[timing/backtest] strategy=... cache_hit=False` 这条 print 是现成
  的探针。建议把它从 print 升级成结构化日志，让"默认页 cache_hit=False 比例 > 1%"成为
  可观测信号。（不在本次事故修复 scope 内，作为下次 Pillar 改造的 follow-up。）

---

## 6. 类似坑提示（给 6 个月后的人 / 给新策略开发者）

**新增 timing 策略 / 给已有策略新增 TimingParams 字段时**，按这个顺序检查：

1. **TimingParams dataclass 字段补全。** 当前 `stock_trade_demo/web/params.py` 的
   `TimingParams` dataclass 有 **39 个字段**（11 int + 26 float + 1 bool + 1 str），
   与 `_BOOL_KEYS / _INT_KEYS / _FLOAT_KEYS / _STR_KEYS` 四个集合并集严格一致，
   无 schema drift（由 `test_timing_params_schema_alignment` 自动守护）。
   新增字段时务必让 dataclass、四个 KEYS 集合、和
   `_TIMING_CACHE_DEFAULTS` / `_US_TIMING_CACHE_DEFAULTS` 五处保持完全一致；缺一项就
   会再次出现"前端发了、后端解析不出来 / 默认值对不上"的回归——schema lint 测试会先
   于业务出错抓住它。
2. **不要在 `_TIMING_CACHE_DEFAULTS` 里放浮点结果而非常量。** 例如不要写
   `'enter_threshold': round(np.float64(0.55), 6)`，要写字面量 `0.55`。否则跨 Python
   版本 / 跨 numpy 版本可能落到不同 bit pattern，老缓存命中策略会再被打穿。
3. **新增前端 slider 后**，确认 `/api/timing/params` 返回的 `default` 字段与后端
   `_TIMING_CACHE_DEFAULTS[strategy][key]` 在「类型 + 值」两个维度都 1:1 对齐。建议
   写一个单元测试 `test_timing_param_default_alignment`，把两边遍历一次断言相等。
4. **新增 timing 策略时**，离线 build 脚本 `scripts/build_timing_cache.py` /
   `scripts/build_us_timing_cache.py` 必须同步预跑该策略 × 默认参数那一组缓存，否则
   `state.TIMING_CACHE.get(strategy_id)` 一开始就 `None`，第一层 `use_cache = strategy_name in state.TIMING_CACHE`
   就 False，**默认页直接走 cache_miss**——这条 web_app 不会给任何 warning，只能靠
   `scripts/check_data_freshness.py` 或本次新增的回归测试发现。
5. **改 `_TIMING_CACHE_DEFAULTS` 中任何 default 值**（比如把 `enter_threshold` 从
   0.55 调到 0.6），必须：
   - (a) 重跑 `scripts/build_timing_cache.py` 重建缓存（不然新 default 进了哈希比对，
     旧缓存全 miss）；
   - (b) 同步前端 slider 的 `default`（不然前端发的旧值与后端新 default 对不上，
     依旧 cache miss）。
   这两步任一漏掉都会立刻触发本类回归。
6. **加新的 load-only 路径前**，先问自己一句："前端在不动 UI 的状态下，会把哪些参数
   塞进 query string？" 然后用真实 query string 跑一遍。**不要只用 `?strategy=...`
   测**。
7. **`best_profile_*_timing.json` 是 effective defaults 的一部分，不是装饰品**。修
   walk-forward 时如果改了 best_profile 中任何参数值（比如把 `trim_threshold` 从 0.35
   调到 0.38），就等于改了"策略真正用的默认"。此时 `_TIMING_CACHE_DEFAULTS` 里的字面
   值不会自动跟上——本次修复后通过 `get_effective_*_defaults()` 已经把这两边接通；
   但**前端 slider 的 default**（来自 `/api/timing/params`）和 **`scripts/build_*_cache.py`
   预跑用的参数集**还是各自独立的来源，改 best_profile 时必须三处一起对齐。
8. **HTML `<input type=range step="...">` 会 round value**。如果某个参数在 best_profile
   里是非网格值（如 0.38、step=0.05），浏览器会把它 round 成 0.4 再回放。本次修复在
   `_extract_effective_from_instance` 末尾做了 `round(round(val/step)*step, 10)` 的
   snap，让后端基准跟着 round。以后新增 slider 时务必让 `step` 是该参数所有可能
   default 值的公约数；做不到就要靠这套 snap 兜住。

---

## 7. 时间线 / 角色记录

| 时间 | 事件 | 角色 |
|---|---|---|
| 2026-05-27（更早） | Pillar 1 Step 6 落地，cache miss 不再现算 → 400 | Worker C（Step 6） |
| 2026-05-27（同期） | 保护 6 落地，TimingParams 接管 query 解析 | Worker B（Step 4 blueprint） |
| 2026-05-27 | 用户打开 `/timing`、`/us_timing` 默认页，全部卡片显示 "—" | 用户 |
| 2026-05-27 | team-lead 分诊：A 改后端，B 写 E2E 测试，C 写文档 | team-lead |
| 2026-05-27 | Worker A 完成 3 层修复（state/params/3 个 blueprint），PID 94496 v3 5/5 全绿 | Worker A |
| 2026-05-27 | Worker B 完成 `test_default_page_load_no_cache_miss.py` 11 case；pre-fix 10F1P / post-fix 11P 双轮验证 | Worker B |
| 2026-05-27 | Worker B 追加 2 个保护 case（L2 lazy init 守卫 + TimingParams schema 双向一致 lint），总 13 case 全绿（7.54s） | Worker B |
| 2026-05-27 | 本文档定稿，file_path:line 全部补齐 | Worker C |

---

*文档作者：Worker C（cache-miss-fix 团队）*
*关联 todo：`0526_todo.md` Task #14 / #15 / #16（已发现并修复）*
*相关代码入口（修复后版本）：*
*  `stock_trade_demo/web/blueprints/timing_api.py:136-156`（A 股 timing 命中判定）*
*  `stock_trade_demo/web/blueprints/us_timing_api.py:160-176, 185-191`（美股 timing 命中判定 + lazy init）*
*  `stock_trade_demo/web/blueprints/select_api.py:93-130`（选股 backtest 命中判定）*
*  `stock_trade_demo/web/state.py:927-988`（effective defaults getters + step snap）*
*  `stock_trade_demo/web/params.py:200-238`（TimingParams.diff_from_defaults / is_all_defaults）*
*  `stock_trade_demo/web/templates/timing.html:1016-1024`（默认值回放点）*
*  `stock_trade_demo/web/templates/us_timing.html:896-904`（默认值回放点）*
*  `scripts/build_timing_cache.py` / `scripts/build_us_timing_cache.py`（cache 重建入口）*
*  `stock_trade_demo/tests/test_default_page_load_no_cache_miss.py`（13 case 回归测试，含 `test_us_timing_backtest_lazy_inits_cache_on_first_hit` + `test_timing_params_schema_alignment`）*
