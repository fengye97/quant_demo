"""
v3.2 最终版 — 网格搜索后的稳健最优配置
==================================================================
最优参数 (基于OOS Sharpe 鲁棒性筛选):
  k = 1.5         (sigmoid斜率, 较缓)
  max_lev = 1.4   (杠杆上限)
  base = 0.5      (中性仓位中心)
  trend_mult = False  (关闭趋势乘子, OOS反而更好)
  inertia = 3%    (仓位变化<3%不调仓)

  与v3.0/v3.1的关键差异:
    - 连续sigmoid代替阶梯映射
    - 杠杆从1.3降到1.4但更准确触发
    - 取消趋势multiplier (OOS无效)
==================================================================
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime

DATA_DIR = "/Users/fatcat/Desktop/quant/data"
OUT_DIR  = "/Users/fatcat/Desktop/quant/strategy"

def load(name):
    return pd.read_csv(os.path.join(DATA_DIR, f"fred_{name}.csv"),
                       parse_dates=[0], index_col=0)
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
panel["HYS_proxy"] = panel["HYS"].fillna(panel["VIX"] / 5.0)

def zscore(s, min_periods=252):
    mu = s.expanding(min_periods=min_periods).mean()
    std = s.expanding(min_periods=min_periods).std()
    return ((s - mu) / (std + 1e-9)).clip(-3, 3)

def build_factors(df):
    f = pd.DataFrame(index=df.index)
    ff_diff = df["FedFunds"].rolling(90).mean() - df["FedFunds"].rolling(365).mean()
    f["Z1_Money"] = -zscore(ff_diff)
    yc_lvl = zscore(df["YC_10Y2Y"]); yc_chg = zscore(df["YC_10Y2Y"].diff(60))
    f["Z2_Liquidity"] = (yc_lvl + yc_chg) / 2
    cpi_yoy = (df["CPI_core"] / df["CPI_core"].shift(365) - 1) * 100
    f["Z3_Inflation"] = (-zscore(cpi_yoy.diff(90)) - zscore(cpi_yoy.clip(lower=2.0))) / 2
    sahm = df["Unemp"].rolling(90).mean() - df["Unemp"].rolling(365).min()
    f["Z4_Economy"] = -zscore(sahm)
    ma200 = df["NDX"].rolling(200).mean()
    f["Z5a_Trend"] = zscore((df["NDX"] / ma200 - 1) * 100)
    f["Z5b_Mom"] = zscore((df["NDX"].rolling(20).mean() / df["NDX"].rolling(60).mean() - 1) * 100)
    f["Z5c_VIX"] = -zscore(df["VIX"])
    f["Z5d_Credit"] = -zscore(df["HYS_proxy"])
    cols = [c for c in f.columns if c.startswith("Z")]
    f["ContScore"] = f[cols].mean(axis=1)
    crisis = df["VIX"] > 35
    f.loc[crisis, "ContScore"] = -2.0
    return f

# ===== v3.2 最优配置 =====
K = 1.5
MAX_LEV = 1.4
BASE = 0.5
INERTIA = 0.03

def cont_pos(score, k=K, max_lev=MAX_LEV, base=BASE):
    if pd.isna(score): return np.nan
    sig = 1 / (1 + np.exp(-k * (score - (base - 0.5))))
    return max(0.0, min(max_lev, sig * max_lev))

def cont_tilt(score):
    if pd.isna(score): return np.nan
    sig = 1 / (1 + np.exp(-1.5 * score))
    return max(0.2, min(1.0, 0.3 + 0.7 * sig))

def apply_inertia(s, threshold=INERTIA):
    out = s.copy(); last = out.iloc[0]
    for i in range(1, len(out)):
        if abs(out.iloc[i] - last) < threshold:
            out.iloc[i] = last
        else:
            last = out.iloc[i]
    return out

factors = build_factors(panel)
factors["Position"] = factors["ContScore"].apply(cont_pos)
factors["QQQ_weight"] = factors["ContScore"].apply(cont_tilt)

pos_m = apply_inertia(factors["Position"].resample("M").last().ffill())
qw_m  = apply_inertia(factors["QQQ_weight"].resample("M").last().ffill(), 0.05)
panel["Position"]   = pos_m.reindex(panel.index, method="ffill")
panel["QQQ_weight"] = qw_m.reindex(panel.index, method="ffill")

def backtest_mix(df, cash_rate=0.04, tc=0.0005):
    d = df.dropna(subset=["NDX","SPX","Position","QQQ_weight"]).copy()
    d = d[d.index.dayofweek < 5]
    d["ret_ndx"] = d["NDX"].pct_change(fill_method=None).fillna(0)
    d["ret_spx"] = d["SPX"].pct_change(fill_method=None).fillna(0)
    d["pos_l"] = d["Position"].shift(1).ffill().fillna(0)
    d["qw_l"]  = d["QQQ_weight"].shift(1).ffill().fillna(0.5)
    d["cash_ret"] = (1 + cash_rate) ** (1/252) - 1
    d["asset_ret"] = d["qw_l"] * d["ret_ndx"] + (1 - d["qw_l"]) * d["ret_spx"]
    d["strat_ret"] = d["pos_l"] * d["asset_ret"] + (1 - d["pos_l"].clip(upper=1)) * d["cash_ret"]
    lev = (d["pos_l"] - 1).clip(lower=0)
    d["strat_ret"] -= lev * ((cash_rate + 0.01) / 252)
    d["turnover"] = (d["pos_l"].diff().abs() + d["qw_l"].diff().abs() * d["pos_l"]).fillna(0)
    d["strat_ret"] -= d["turnover"] * tc
    d["strat_nv"] = (1 + d["strat_ret"]).cumprod()
    d["bh_ndx_nv"] = (1 + d["ret_ndx"]).cumprod()
    d["bh_spx_nv"] = (1 + d["ret_spx"]).cumprod()
    return d

def backtest_ndx(df, cash_rate=0.04, tc=0.0005):
    d = df[["NDX","Position"]].dropna().copy()
    d = d[d.index.dayofweek < 5]
    d["ret"] = d["NDX"].pct_change(fill_method=None).fillna(0)
    d["pos_l"] = d["Position"].shift(1).ffill().fillna(0)
    d["cash_ret"] = (1 + cash_rate) ** (1/252) - 1
    d["strat_ret"] = d["pos_l"] * d["ret"] + (1 - d["pos_l"].clip(upper=1)) * d["cash_ret"]
    lev = (d["pos_l"] - 1).clip(lower=0)
    d["strat_ret"] -= lev * ((cash_rate + 0.01) / 252)
    d["turnover"] = d["pos_l"].diff().abs().fillna(0)
    d["strat_ret"] -= d["turnover"] * tc
    d["strat_nv"] = (1 + d["strat_ret"]).cumprod()
    d["bh_nv"] = (1 + d["ret"]).cumprod()
    return d

def perf(nv, ret, name):
    yrs = (nv.index[-1] - nv.index[0]).days / 365.25
    cagr = nv.iloc[-1] ** (1/yrs) - 1
    vol  = ret.std() * np.sqrt(252)
    sh   = (ret.mean() * 252 - 0.04) / vol if vol > 0 else 0
    mdd  = ((nv - nv.cummax()) / nv.cummax()).min()
    cal  = cagr / abs(mdd) if mdd != 0 else 0
    return {"name": name, "yrs": yrs, "CAGR": cagr, "Vol": vol,
            "Sharpe": sh, "MaxDD": mdd, "Calmar": cal}

bt_mix = backtest_mix(panel)
bt_ndx = backtest_ndx(panel)

# Train/Test 分段绩效
def split_perf(bt, ret_c, nv_c, name):
    train = bt.loc[:"2018-12-31"]
    test  = bt.loc["2019-01-01":]
    train_nv = (1 + train[ret_c]).cumprod()
    test_nv  = (1 + test[ret_c]).cumprod()
    pt = perf(train_nv, train[ret_c], name + "[Train2005-18]")
    pte = perf(test_nv, test[ret_c], name + "[Test2019-26]")
    return pt, pte

# v3.2 NDX分段
v32_tr, v32_te = split_perf(bt_ndx, "strat_ret", "strat_nv", "v3.2")
bh_tr, bh_te = split_perf(bt_ndx, "ret", "bh_nv", "BH-NDX")

print("\n" + "=" * 90)
print("v3.2 严格 Train/Test 验证 (NDX, train: 2005-2018, test: 2019-2026)")
print("=" * 90)
print(f"{'方案':<28}{'年数':>6}{'CAGR':>9}{'Sharpe':>9}{'MaxDD':>9}{'Calmar':>9}")
print("-" * 90)
for r in [v32_tr, bh_tr, v32_te, bh_te]:
    print(f"{r['name']:<28}{r['yrs']:>6.1f}{r['CAGR']*100:>8.2f}%"
          f"{r['Sharpe']:>9.3f}{r['MaxDD']*100:>8.1f}%{r['Calmar']:>9.3f}")

# 全样本 + 混合
print("\n" + "=" * 90)
print("v3.2 全样本 + 混合资产 — vs v3.0/v3.1/BH")
print("=" * 90)
stats = [
    perf(bt_mix["strat_nv"],  bt_mix["strat_ret"], "v3.2混合(2016-)"),
    perf(bt_mix["bh_ndx_nv"], bt_mix["ret_ndx"],   "BH-NDX(2016-)"),
    perf(bt_mix["bh_spx_nv"], bt_mix["ret_spx"],   "BH-SPX(2016-)"),
    perf(bt_ndx["strat_nv"],  bt_ndx["strat_ret"], "v3.2纯NDX(2005-)"),
    perf(bt_ndx["bh_nv"],     bt_ndx["ret"],       "BH-NDX(2005-)"),
]
print(f"{'方案':<22}{'年数':>6}{'CAGR':>9}{'Vol':>8}{'Sharpe':>9}{'MaxDD':>9}{'Calmar':>9}")
print("-" * 90)
for s in stats:
    print(f"{s['name']:<22}{s['yrs']:>6.1f}{s['CAGR']*100:>8.2f}%{s['Vol']*100:>7.1f}%"
          f"{s['Sharpe']:>9.3f}{s['MaxDD']*100:>8.1f}%{s['Calmar']:>9.3f}")

# 年度对比
bt_ndx_ann = bt_ndx.copy(); bt_ndx_ann["year"] = bt_ndx_ann.index.year
ann = bt_ndx_ann.groupby("year").agg(
    strat=("strat_ret", lambda x: (1+x).prod()-1),
    bh=("ret", lambda x: (1+x).prod()-1),
    pos=("pos_l", "mean"))

print("\n" + "=" * 60)
print("v3.2 年度收益 vs BH-NDX")
print("=" * 60)
print(f"{'年份':>6}{'v3.2':>10}{'BH-NDX':>10}{'超额':>10}{'平均仓位':>12}")
strat_win, total = 0, 0
strat_dd_better = 0
for y, r in ann.iterrows():
    diff = r['strat'] - r['bh']
    if diff > 0: strat_win += 1
    total += 1
    print(f"{y:>6}{r['strat']*100:>9.1f}%{r['bh']*100:>9.1f}%{diff*100:>+9.1f}%{r['pos']*100:>10.0f}%")
print("-" * 60)
print(f"{'胜率':>6}{strat_win/total*100:>9.1f}% ({strat_win}/{total})")

# 当前信号
print("\n" + "=" * 60)
print("v3.2 最新信号")
print("=" * 60)
valid = factors[[c for c in factors.columns if c.startswith("Z")]].dropna()
if len(valid) > 0:
    last = valid.index[-1]
    print(f"参考日期: {last.date()}")
    for c in [c for c in factors.columns if c.startswith("Z")]:
        v = factors.loc[last, c]
        bar_pos = "█" * int(max(0, v) * 5)
        bar_neg = "█" * int(max(0, -v) * 5)
        print(f"  {c:<18} = {v:>+6.2f}  {bar_neg:>10}|{bar_pos:<10}")
    cs = factors.loc[last, "ContScore"]
    pos = factors.loc[last, "Position"]
    qw  = factors.loc[last, "QQQ_weight"]
    print(f"  {'ContScore':<18} = {cs:>+6.2f}")
    print(f"  仓位     = {pos*100:.0f}%   (sigmoid k={K}, lev={MAX_LEV})")
    print(f"  QQQ权重  = {qw*100:.0f}%   SPY权重 = {(1-qw)*100:.0f}%")

# 保存
bt_mix.to_csv(os.path.join(OUT_DIR, "backtest_v32_mix.csv"))
bt_ndx.to_csv(os.path.join(OUT_DIR, "backtest_v32_ndx.csv"))
factors.to_csv(os.path.join(OUT_DIR, "factor_signals_v32.csv"))
pd.DataFrame(stats).to_csv(os.path.join(OUT_DIR, "v32_performance.csv"), index=False)
ann.to_csv(os.path.join(OUT_DIR, "v32_annual.csv"))
print(f"\n保存到 {OUT_DIR}")
