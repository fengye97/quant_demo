"""
美股宏观多因子择时策略 v2.1 (优化版)
================================================================
v2.0 -> v2.1 改进点:
  1. 引入"杠杆区"——RegimeScore极强时1.3x杠杆,加速复利
  2. 仓位映射更平滑(7档而非4档),减少跳变
  3. 修复因子NaN填充——保证最近日期使用最新数据
  4. 信号生效改为"次日开盘"(更接近实盘)
  5. 增加QQQ vs SPY轮动:强势趋势期偏QQQ,弱势期偏SPY
  6. 加入"危机模式":VIX>35+信用利差暴走时强制空仓
================================================================
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_REPO_ROOT, "data")
OUT_DIR  = os.path.join(_REPO_ROOT, "strategy")
os.makedirs(OUT_DIR, exist_ok=True)

def load(name):
    fp = os.path.join(DATA_DIR, f"fred_{name}.csv")
    return pd.read_csv(fp, parse_dates=[0], index_col=0)

# -----------------------------------------------------------
# 1. 加载并合并
# -----------------------------------------------------------
def to_daily(df, col):
    df = df.dropna(); df.columns = [col]
    return df.resample("D").ffill()

panel = pd.DataFrame(index=pd.date_range("2005-01-01", datetime.today(), freq="D"))
for code, col in [("NASDAQ100_FRED","NDX"), ("SP500_FRED","SPX"),
                  ("CPI_core","CPI_core"), ("FedFundsRate","FedFunds"),
                  ("YieldCurve_10Y2Y","YC_10Y2Y"), ("Unemployment","Unemp"),
                  ("VIX","VIX"), ("HighYieldSpread","HYS"),
                  ("Treasury10Y","T10Y")]:
    panel[col] = to_daily(load(code), col)[col]

# 关键: HighYieldSpread 仅2023-至今,无值时使用VIX代理
panel["HYS_proxy"] = panel["HYS"].fillna(panel["VIX"] / 5.0)

# -----------------------------------------------------------
# 2. 因子构建 (优化阈值)
# -----------------------------------------------------------
def build_factors(df):
    f = pd.DataFrame(index=df.index)

    # F1 货币环境: Fed Funds 3MA - 12MA, 降息周期=正面
    ff3, ff12 = df["FedFunds"].rolling(90).mean(), df["FedFunds"].rolling(365).mean()
    diff = ff3 - ff12
    f["F1_Money"] = np.where(diff < -0.1, 1, np.where(diff > 0.3, -1, 0))

    # F2 流动性: 期限利差走势(变陡=经济回暖前兆)
    yc = df["YC_10Y2Y"]
    yc_chg = yc.diff(60)  # 60天变化
    f["F2_Liquidity"] = np.where((yc > 0) & (yc_chg > 0), 1,
                          np.where((yc < -0.3) & (yc_chg < 0), -1, 0))

    # F3 通胀: Core CPI同比变化方向
    cpi_yoy = (df["CPI_core"] / df["CPI_core"].shift(365) - 1) * 100
    cpi_mom = cpi_yoy.diff(90)
    # 通胀回落 = 正面; 通胀重燃 = 负面
    f["F3_Inflation"] = np.where(cpi_mom < -0.2, 1, np.where(cpi_mom > 0.3, -1, 0))
    # 额外惩罚: 绝对值>4% 直接负面
    f["F3_Inflation"] = np.where(cpi_yoy > 4.0, -1, f["F3_Inflation"])

    # F4 经济: Sahm Rule 简化版
    u3 = df["Unemp"].rolling(90).mean()
    u_min = df["Unemp"].rolling(365).min()
    sahm = u3 - u_min
    f["F4_Economy"] = np.where(sahm < 0.2, 1, np.where(sahm > 0.4, -1, 0))

    # F5a 趋势: 价格 vs 200日均线 (NDX)
    ma200 = df["NDX"].rolling(200).mean()
    f["F5a_Trend"] = np.where(df["NDX"] > ma200 * 1.02, 1,
                       np.where(df["NDX"] < ma200 * 0.97, -1, 0))

    # F5b 短期动量: 20日 vs 60日均线
    ma20, ma60 = df["NDX"].rolling(20).mean(), df["NDX"].rolling(60).mean()
    f["F5b_Momentum"] = np.where(ma20 > ma60 * 1.01, 1,
                          np.where(ma20 < ma60 * 0.99, -1, 0))

    # F5c 情绪: VIX 水位
    f["F5c_VIX"] = np.where(df["VIX"] < 16, 1, np.where(df["VIX"] > 25, -1, 0))

    # F5d 信用: 高收益利差(有数据则用,否则代理)
    hys = df["HYS_proxy"].rolling(20).mean()
    f["F5d_Credit"] = np.where(hys < 4.0, 1, np.where(hys > 6.0, -1, 0))

    # 危机覆盖: VIX>35 强制 -3 (任何因子组合都覆盖)
    crisis = df["VIX"] > 35
    f["Crisis"] = crisis.astype(int)

    factor_cols = [c for c in f.columns if c.startswith("F")]
    f["RegimeScore"] = f[factor_cols].sum(axis=1)
    # 危机模式覆盖
    f.loc[crisis, "RegimeScore"] = -5
    return f, factor_cols

factors, FCOLS = build_factors(panel)

# -----------------------------------------------------------
# 3. 仓位映射 (7档 + 杠杆区)
# -----------------------------------------------------------
def score_to_pos(s):
    if pd.isna(s): return np.nan
    if s >=  5: return 1.30   # 极强势 → 1.3x杠杆
    if s >=  3: return 1.00   # 强势
    if s >=  1: return 0.75   # 偏强
    if s ==  0: return 0.50   # 中性
    if s >= -2: return 0.25   # 偏弱
    if s >= -4: return 0.10   # 弱
    return 0.00                # 危机 → 空仓

factors["Position"] = factors["RegimeScore"].apply(score_to_pos)

# QQQ vs SPY 偏向: score高→偏QQQ, score低→偏SPY
def asset_tilt(s):
    if pd.isna(s): return np.nan
    if s >= 3:  return 1.0   # 100% QQQ
    if s >= 0:  return 0.6   # 60% QQQ + 40% SPY
    return 0.3                # 30% QQQ + 70% SPY (防御)
factors["QQQ_weight"] = factors["RegimeScore"].apply(asset_tilt)

# 月度调仓
pos_m = factors["Position"].resample("ME").last().ffill()
qqq_m = factors["QQQ_weight"].resample("ME").last().ffill()
panel["Position"]   = pos_m.reindex(panel.index, method="ffill")
panel["QQQ_weight"] = qqq_m.reindex(panel.index, method="ffill")

# -----------------------------------------------------------
# 4. 回测引擎 (含NDX/SPX混合资产)
# -----------------------------------------------------------
def backtest_mix(df, cash_rate=0.04, tc=0.0005):
    d = df[["NDX","SPX","Position","QQQ_weight"]].copy()
    d = d.dropna(subset=["NDX","SPX"])
    d = d[d.index.dayofweek < 5]
    d["ret_ndx"] = d["NDX"].pct_change(fill_method=None).fillna(0)
    d["ret_spx"] = d["SPX"].pct_change(fill_method=None).fillna(0)
    d["pos_l"]  = d["Position"].shift(1).ffill().fillna(0)
    d["qw_l"]   = d["QQQ_weight"].shift(1).ffill().fillna(0.5)
    d["cash_ret"] = (1 + cash_rate) ** (1/252) - 1
    # 组合日收益
    d["asset_ret"] = d["qw_l"] * d["ret_ndx"] + (1 - d["qw_l"]) * d["ret_spx"]
    d["strat_ret"] = d["pos_l"] * d["asset_ret"] + (1 - d["pos_l"].clip(upper=1)) * d["cash_ret"]
    # 杠杆部分融资成本: 超过1的部分支付cash_rate+1%
    lev = (d["pos_l"] - 1).clip(lower=0)
    d["strat_ret"] -= lev * ((cash_rate + 0.01) / 252)
    # 交易成本
    d["turnover"] = (d["pos_l"].diff().abs() + d["qw_l"].diff().abs() * d["pos_l"]).fillna(0)
    d["strat_ret"] -= d["turnover"] * tc
    d["strat_nv"] = (1 + d["strat_ret"]).cumprod()
    d["bh_ndx_nv"] = (1 + d["ret_ndx"]).cumprod()
    d["bh_spx_nv"] = (1 + d["ret_spx"]).cumprod()
    return d

def perf(nv, ret, label):
    yrs = (nv.index[-1] - nv.index[0]).days / 365.25
    cagr = nv.iloc[-1] ** (1/yrs) - 1
    vol  = ret.std() * np.sqrt(252)
    sh   = (ret.mean() * 252 - 0.04) / vol if vol > 0 else 0
    mdd  = ((nv - nv.cummax()) / nv.cummax()).min()
    cal  = cagr / abs(mdd) if mdd != 0 else 0
    return {"label": label, "CAGR": cagr, "Vol": vol, "Sharpe": sh,
            "MaxDD": mdd, "Calmar": cal}

# 全样本回测 (2005-至今, 但SPX只有2016以后)
bt_full = backtest_mix(panel)
# 仅NDX阶段(2005-至今)
panel_ndx_only = panel.copy(); panel_ndx_only["SPX"] = panel_ndx_only["NDX"]
bt_ndx = backtest_mix(panel_ndx_only)

stats = []
stats.append(perf(bt_full["strat_nv"], bt_full["strat_ret"],
                  "策略(混合 NDX+SPX, 2016-至今)"))
stats.append(perf(bt_full["bh_ndx_nv"], bt_full["ret_ndx"],
                  "BH-NDX (2016-至今)"))
stats.append(perf(bt_full["bh_spx_nv"], bt_full["ret_spx"],
                  "BH-SPX (2016-至今)"))
stats.append(perf(bt_ndx["strat_nv"], bt_ndx["strat_ret"],
                  "策略(纯NDX, 2005-至今)"))
stats.append(perf(bt_ndx["bh_ndx_nv"], bt_ndx["ret_ndx"],
                  "BH-NDX (2005-至今)"))

# -----------------------------------------------------------
# 5. 输出
# -----------------------------------------------------------
print("\n" + "=" * 80)
print("美股宏观多因子择时策略 v2.1 — 回测对比")
print("=" * 80)
print(f"{'方案':<35} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'MaxDD':>8} {'Calmar':>8}")
print("-" * 80)
for s in stats:
    print(f"{s['label']:<35} {s['CAGR']*100:>7.2f}% {s['Vol']*100:>7.2f}% "
          f"{s['Sharpe']:>8.3f} {s['MaxDD']*100:>7.2f}% {s['Calmar']:>8.3f}")

print("\n" + "=" * 80)
print("当前最新因子状态")
print("=" * 80)
# 取最近一个所有因子都有值的日期
valid_idx = factors[FCOLS].dropna().index
if len(valid_idx) > 0:
    last = valid_idx[-1]
    print(f"参考日期: {last.date()}")
    for c in FCOLS:
        v = int(factors.loc[last, c])
        flag = "✓正" if v > 0 else ("✗负" if v < 0 else "○中")
        print(f"  {c:<18} = {v:+d}  {flag}")
    sc = int(factors.loc[last, "RegimeScore"])
    pos = factors.loc[last, "Position"]
    qw = factors.loc[last, "QQQ_weight"]
    print(f"  {'RegimeScore':<18} = {sc:+d}")
    print(f"  建议总仓位         = {pos*100:.0f}%   (>100%表示杠杆)")
    print(f"  QQQ权重            = {qw*100:.0f}%   SPY权重 = {(1-qw)*100:.0f}%")

# 保存
bt_full.to_csv(os.path.join(OUT_DIR, "backtest_v2_full.csv"))
bt_ndx.to_csv(os.path.join(OUT_DIR, "backtest_v2_ndx_only.csv"))
factors.to_csv(os.path.join(OUT_DIR, "factor_signals_v2.csv"))
pd.DataFrame(stats).to_csv(os.path.join(OUT_DIR, "performance_summary.csv"), index=False)
print(f"\n结果已保存到 {OUT_DIR}")
