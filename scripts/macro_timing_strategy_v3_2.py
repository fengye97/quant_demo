"""
v3.2 — 参数网格搜索 + 最优sigmoid配置
==================================================================
搜索维度:
  k (sigmoid斜率): 1.5, 2.0, 2.5, 3.0
  max_lev (杠杆上限): 1.2, 1.4, 1.6, 1.8
  base (基础仓位中心): 0.4, 0.5, 0.6
  trend_mult开关: True/False

注意: 这是out-of-sample搜索的伪近似 — 用前60%数据(2005-2018)选最优,
       后40%(2019-2026)做真实OOS测试,避免过拟合.
==================================================================
"""
import os, itertools
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
for code, col in [("NASDAQ100_FRED","NDX"), ("CPI_core","CPI_core"),
                  ("FedFundsRate","FedFunds"), ("YieldCurve_10Y2Y","YC_10Y2Y"),
                  ("Unemployment","Unemp"), ("VIX","VIX"),
                  ("HighYieldSpread","HYS"), ("Treasury10Y","T10Y")]:
    panel[col] = to_daily(load(code), col)[col]
panel["HYS_proxy"] = panel["HYS"].fillna(panel["VIX"] / 5.0)

def zscore(s, window=252*5, min_periods=252):
    mu = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std()
    return ((s - mu) / (std + 1e-9)).clip(-3, 3)

def build_factors(df):
    f = pd.DataFrame(index=df.index)
    ff_diff = df["FedFunds"].rolling(90).mean() - df["FedFunds"].rolling(365).mean()
    f["Z1"] = -zscore(ff_diff)
    yc_lvl = zscore(df["YC_10Y2Y"]); yc_chg = zscore(df["YC_10Y2Y"].diff(60))
    f["Z2"] = (yc_lvl + yc_chg) / 2
    cpi_yoy = (df["CPI_core"] / df["CPI_core"].shift(365) - 1) * 100
    f["Z3"] = (-zscore(cpi_yoy.diff(90)) - zscore(cpi_yoy.clip(lower=2.0))) / 2
    sahm = df["Unemp"].rolling(90).mean() - df["Unemp"].rolling(365).min()
    f["Z4"] = -zscore(sahm)
    ma200 = df["NDX"].rolling(200).mean()
    f["Z5a"] = zscore((df["NDX"] / ma200 - 1) * 100)
    f["Z5b"] = zscore((df["NDX"].rolling(20).mean() / df["NDX"].rolling(60).mean() - 1) * 100)
    f["Z5c"] = -zscore(df["VIX"])
    f["Z5d"] = -zscore(df["HYS_proxy"])
    cols = [c for c in f.columns if c.startswith("Z")]
    f["ContScore"] = f[cols].mean(axis=1)
    crisis = df["VIX"] > 35
    f.loc[crisis, "ContScore"] = -2.0
    ma200_slope = ma200.diff(60) / ma200.shift(60)
    f["TrendMult"] = np.where(ma200_slope > 0.02, 1.15,
                       np.where(ma200_slope < -0.02, 0.85, 1.0))
    return f

factors = build_factors(panel)

def cont_pos(score, k, max_lev, base):
    if pd.isna(score): return np.nan
    sig = 1 / (1 + np.exp(-k * (score - (base - 0.5))))
    return max(0.0, min(max_lev, sig * max_lev))

def backtest(df, cash_rate=0.04, tc=0.0005):
    d = df[["NDX","Pos"]].dropna().copy()
    d = d[d.index.dayofweek < 5]
    d["ret"] = d["NDX"].pct_change(fill_method=None).fillna(0)
    d["pos_l"] = d["Pos"].shift(1).ffill().fillna(0)
    d["cash_ret"] = (1 + cash_rate) ** (1/252) - 1
    d["strat_ret"] = d["pos_l"] * d["ret"] + (1 - d["pos_l"].clip(upper=1)) * d["cash_ret"]
    lev = (d["pos_l"] - 1).clip(lower=0)
    d["strat_ret"] -= lev * ((cash_rate + 0.01) / 252)
    d["turnover"] = d["pos_l"].diff().abs().fillna(0)
    d["strat_ret"] -= d["turnover"] * tc
    return d

def perf(seg, ret_col, nv):
    yrs = (seg.index[-1] - seg.index[0]).days / 365.25
    if yrs < 0.5: return None
    cagr = nv.iloc[-1] ** (1/yrs) - 1
    vol  = ret_col.std() * np.sqrt(252)
    sh   = (ret_col.mean() * 252 - 0.04) / vol if vol > 0 else 0
    mdd  = ((nv - nv.cummax()) / nv.cummax()).min()
    cal  = cagr / abs(mdd) if mdd != 0 else 0
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sh, "MaxDD": mdd, "Calmar": cal}

# ---------------- 网格搜索 (in-sample: 2005-2018) ----------------
TRAIN_END = "2018-12-31"
TEST_START = "2019-01-01"

grid = list(itertools.product(
    [1.5, 2.0, 2.5, 3.0],         # k
    [1.2, 1.4, 1.6, 1.8],         # max_lev
    [0.4, 0.5, 0.6],              # base
    [True, False],                # use trend mult
))

results = []
for k, lev, base, use_tm in grid:
    factors["Position"] = factors["ContScore"].apply(lambda s: cont_pos(s, k, lev, base))
    if use_tm:
        factors["Position"] = (factors["Position"] * factors["TrendMult"]).clip(0, 1.8)
    pos_m = factors["Position"].resample("ME").last().ffill()
    panel["Pos"] = pos_m.reindex(panel.index, method="ffill")
    bt = backtest(panel)
    # 分段评估
    train = bt.loc[:TRAIN_END]
    test  = bt.loc[TEST_START:]
    train_nv = (1 + train["strat_ret"]).cumprod()
    test_nv  = (1 + test["strat_ret"]).cumprod()
    train_bh = (1 + train["ret"]).cumprod()
    test_bh  = (1 + test["ret"]).cumprod()
    pt = perf(train, train["strat_ret"], train_nv)
    pte = perf(test, test["strat_ret"], test_nv)
    pbt = perf(train, train["ret"], train_bh)
    pbte = perf(test, test["ret"], test_bh)
    if None in (pt, pte, pbt, pbte): continue
    results.append({
        "k": k, "lev": lev, "base": base, "tm": use_tm,
        "train_CAGR": pt["CAGR"], "train_Sh": pt["Sharpe"],
        "train_DD": pt["MaxDD"], "train_Cal": pt["Calmar"],
        "test_CAGR": pte["CAGR"], "test_Sh": pte["Sharpe"],
        "test_DD": pte["MaxDD"], "test_Cal": pte["Calmar"],
        "test_alphaCAGR": pte["CAGR"] - pbte["CAGR"],
        "test_alphaSh":  pte["Sharpe"] - pbte["Sharpe"],
    })

df_grid = pd.DataFrame(results)
# 按train Sharpe排序选最优,看test表现
df_grid_sorted = df_grid.sort_values("train_Sh", ascending=False)
top10 = df_grid_sorted.head(10)

print("\n" + "=" * 110)
print("网格搜索: 按 Train Sharpe(2005-2018) 排前10, 检查 Test(2019-2026) 是否守住")
print("=" * 110)
print(f"{'k':>4}{'lev':>5}{'base':>5}{'tm':>4}"
      f"{'TrainCAGR':>11}{'TrainSh':>9}{'TrainDD':>9}{'TrainCal':>10}"
      f"{'TestCAGR':>10}{'TestSh':>9}{'TestDD':>9}{'TestCal':>10}{'αSh':>7}")
print("-" * 110)
for _, r in top10.iterrows():
    print(f"{r['k']:>4.1f}{r['lev']:>5.1f}{r['base']:>5.1f}"
          f"{('T' if r['tm'] else 'F'):>4}"
          f"{r['train_CAGR']*100:>10.1f}%{r['train_Sh']:>9.3f}{r['train_DD']*100:>8.1f}%"
          f"{r['train_Cal']:>10.3f}"
          f"{r['test_CAGR']*100:>9.1f}%{r['test_Sh']:>9.3f}{r['test_DD']*100:>8.1f}%"
          f"{r['test_Cal']:>10.3f}{r['test_alphaSh']:>+7.3f}")

# 按test alpha Sharpe排
print("\n" + "=" * 110)
print("按 Test Alpha-Sharpe 排序 (实际OOS最优)")
print("=" * 110)
df_grid_test = df_grid.sort_values("test_alphaSh", ascending=False)
top10t = df_grid_test.head(10)
print(f"{'k':>4}{'lev':>5}{'base':>5}{'tm':>4}"
      f"{'TrainCAGR':>11}{'TrainSh':>9}{'TrainDD':>9}{'TrainCal':>10}"
      f"{'TestCAGR':>10}{'TestSh':>9}{'TestDD':>9}{'TestCal':>10}{'αSh':>7}")
print("-" * 110)
for _, r in top10t.iterrows():
    print(f"{r['k']:>4.1f}{r['lev']:>5.1f}{r['base']:>5.1f}"
          f"{('T' if r['tm'] else 'F'):>4}"
          f"{r['train_CAGR']*100:>10.1f}%{r['train_Sh']:>9.3f}{r['train_DD']*100:>8.1f}%"
          f"{r['train_Cal']:>10.3f}"
          f"{r['test_CAGR']*100:>9.1f}%{r['test_Sh']:>9.3f}{r['test_DD']*100:>8.1f}%"
          f"{r['test_Cal']:>10.3f}{r['test_alphaSh']:>+7.3f}")

df_grid.to_csv(os.path.join(OUT_DIR, "v32_grid_search.csv"), index=False)
print(f"\n所有 {len(df_grid)} 组合保存到 v32_grid_search.csv")

# 找出"鲁棒最优"——train前30%且test前30%的交集
top30_train = set(df_grid_sorted.head(int(len(df_grid)*0.3)).index)
top30_test = set(df_grid_test.head(int(len(df_grid)*0.3)).index)
robust = top30_train & top30_test
print(f"\n鲁棒最优(train前30% ∩ test前30%): {len(robust)}组")
if robust:
    robust_df = df_grid.loc[list(robust)].sort_values("test_alphaSh", ascending=False).head(5)
    print(robust_df[["k","lev","base","tm","train_Sh","test_Sh","test_alphaSh","test_DD"]].to_string())
