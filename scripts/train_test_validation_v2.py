"""
Train/Test 验证 v2 — 使用纯NDX(2005+)以获得更长OOS窗口
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_REPO_ROOT, "data")
OUT_DIR  = os.path.join(_REPO_ROOT, "strategy")
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

# v3.0 因子(与生产版一致)
def build_factors(df):
    f = pd.DataFrame(index=df.index)
    ff3, ff12 = df["FedFunds"].rolling(90).mean(), df["FedFunds"].rolling(365).mean()
    diff = ff3 - ff12
    f["F1"] = np.where(diff < -0.1, 1, np.where(diff > 0.3, -1, 0))
    yc, yc_chg = df["YC_10Y2Y"], df["YC_10Y2Y"].diff(60)
    f["F2"] = np.where((yc > 0) & (yc_chg > 0), 1,
                np.where((yc < -0.3) & (yc_chg < 0), -1, 0))
    cpi_yoy = (df["CPI_core"] / df["CPI_core"].shift(365) - 1) * 100
    cpi_mom = cpi_yoy.diff(90)
    f["F3"] = np.where(cpi_mom < -0.2, 1, np.where(cpi_mom > 0.3, -1, 0))
    f["F3"] = np.where(cpi_yoy > 4.0, -1, f["F3"])
    u3, u_min = df["Unemp"].rolling(90).mean(), df["Unemp"].rolling(365).min()
    f["F4"] = np.where(u3 - u_min < 0.2, 1, np.where(u3 - u_min > 0.4, -1, 0))
    ma200 = df["NDX"].rolling(200).mean()
    f["F5a"] = np.where(df["NDX"] > ma200 * 1.02, 1,
                np.where(df["NDX"] < ma200 * 0.97, -1, 0))
    ma20, ma60 = df["NDX"].rolling(20).mean(), df["NDX"].rolling(60).mean()
    f["F5b"] = np.where(ma20 > ma60 * 1.01, 1, np.where(ma20 < ma60 * 0.99, -1, 0))
    f["F5c"] = np.where(df["VIX"] < 16, 1, np.where(df["VIX"] > 25, -1, 0))
    hys = df["HYS_proxy"].rolling(20).mean()
    f["F5d"] = np.where(hys < 4.0, 1, np.where(hys > 6.0, -1, 0))
    crisis = df["VIX"] > 35
    cols = [c for c in f.columns if c.startswith("F")]
    f["Score"] = f[cols].sum(axis=1)
    f.loc[crisis, "Score"] = -5
    return f

def score_to_pos(s):
    if pd.isna(s): return np.nan
    if s >= 5: return 1.30
    if s >= 3: return 1.00
    if s >= 1: return 0.75
    if s == 0: return 0.50
    if s >= -2: return 0.25
    if s >= -4: return 0.10
    return 0.00

factors = build_factors(panel)
factors["Pos"] = factors["Score"].apply(score_to_pos)
panel["Pos"] = factors["Pos"].resample("ME").last().ffill().reindex(panel.index, method="ffill")

# ---------------- NDX-only 回测 ----------------
def backtest_ndx(df, cash_rate=0.04, tc=0.0005):
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

bt = backtest_ndx(panel)

def perf(seg, ret_col, nv_col):
    yrs = (seg.index[-1] - seg.index[0]).days / 365.25
    if yrs < 0.3: return None
    cagr = nv_col.iloc[-1] ** (1/yrs) - 1
    vol  = ret_col.std() * np.sqrt(252)
    sh   = (ret_col.mean() * 252 - 0.04) / vol if vol > 0 else 0
    mdd  = ((nv_col - nv_col.cummax()) / nv_col.cummax()).min()
    cal  = cagr / abs(mdd) if mdd != 0 else 0
    return {"yrs": yrs, "CAGR": cagr, "Vol": vol,
            "Sharpe": sh, "MaxDD": mdd, "Calmar": cal}

def slice_perf(bt, start, end, label):
    seg = bt.loc[start:end].copy()
    if len(seg) < 30: return None
    seg["s_nv"] = (1 + seg["strat_ret"]).cumprod()
    seg["b_nv"] = (1 + seg["ret"]).cumprod()
    s = perf(seg, seg["strat_ret"], seg["s_nv"])
    b = perf(seg, seg["ret"], seg["b_nv"])
    if s is None or b is None: return None
    return {"label": label, "start": str(start)[:10], "end": str(end)[:10],
            "yrs": s["yrs"],
            "s_CAGR": s["CAGR"], "s_Sh": s["Sharpe"], "s_DD": s["MaxDD"], "s_Cal": s["Calmar"],
            "b_CAGR": b["CAGR"], "b_Sh": b["Sharpe"], "b_DD": b["MaxDD"], "b_Cal": b["Calmar"]}

# ---------------- 实验A: 多种train/test切分 ----------------
splits = [
    ("全样本",        "2005-02-01", "2026-05-22"),
    ("Train 05-19",  "2005-02-01", "2019-12-31"),
    ("Test  20-26",  "2020-01-01", "2026-05-22"),
    ("Train 05-17",  "2005-02-01", "2017-12-31"),
    ("Test  18-26",  "2018-01-01", "2026-05-22"),
    ("Train 05-13",  "2005-02-01", "2013-12-31"),
    ("Test  14-26",  "2014-01-01", "2026-05-22"),
    ("Train 05-10",  "2005-02-01", "2010-12-31"),
    ("Test  11-26",  "2011-01-01", "2026-05-22"),
    ("Train 05-08",  "2005-02-01", "2008-12-31"),
    ("Test  09-26",  "2009-01-01", "2026-05-22"),
]
res = [slice_perf(bt, s, e, l) for l, s, e in splits]
res = [r for r in res if r is not None]

print("\n" + "=" * 105)
print("实验A: 多种 Train/Test 切分对比 (纯NDX, 2005-2026)")
print("=" * 105)
print(f"{'切分':<14}{'年数':>5}  {'策略':>22}  {'BuyHold':>22}  {'alpha':>20}")
print(f"{'':14}{'':5}  {'CAGR Sharpe MaxDD':>22}  {'CAGR Sharpe MaxDD':>22}  {'ΔCAGR ΔSh ΔDD':>20}")
print("-" * 105)
for r in res:
    print(f"{r['label']:<14}{r['yrs']:>5.1f}  "
          f"{r['s_CAGR']*100:>6.1f}% {r['s_Sh']:>5.2f} {r['s_DD']*100:>6.1f}%  "
          f"{r['b_CAGR']*100:>6.1f}% {r['b_Sh']:>5.2f} {r['b_DD']*100:>6.1f}%  "
          f"{(r['s_CAGR']-r['b_CAGR'])*100:>+6.1f}% {r['s_Sh']-r['b_Sh']:>+5.2f} "
          f"{(r['s_DD']-r['b_DD'])*100:>+6.1f}%")

# ---------------- 实验B: Walk-Forward 5+1, 3+1, 5+2 ----------------
def walk_forward(bt, train_y, test_y, prefix):
    out = []
    start_year = bt.index.year.min() + train_y
    end_year = bt.index.year.max()
    for ty in range(start_year, end_year + 1):
        ts = pd.Timestamp(f"{ty}-01-01")
        te = pd.Timestamp(f"{min(ty + test_y - 1, end_year)}-12-31")
        if te > bt.index.max(): te = bt.index.max()
        seg = bt.loc[ts:te]
        if len(seg) < 30: continue
        snv = (1 + seg["strat_ret"]).cumprod(); bnv = (1 + seg["ret"]).cumprod()
        sp = perf(seg, seg["strat_ret"], snv); bp = perf(seg, seg["ret"], bnv)
        if sp is None or bp is None: continue
        out.append({"year": ty,
                    "s_ret": float(snv.iloc[-1] - 1),
                    "b_ret": float(bnv.iloc[-1] - 1),
                    "s_Sh": sp["Sharpe"], "b_Sh": bp["Sharpe"],
                    "s_DD": sp["MaxDD"], "b_DD": bp["MaxDD"]})
    return out

wf_5_1 = walk_forward(bt, 5, 1, "WF5+1")

print("\n" + "=" * 90)
print("实验B: Walk-Forward 5年训练+1年OOS (纯NDX 2010-2026)")
print("=" * 90)
print(f"{'年份':>6} {'策略':>9} {'BH-NDX':>9} {'超额':>9} {'策略Sh':>8} {'BH-Sh':>8} {'策略DD':>9} {'BH-DD':>9}")
print("-" * 90)
s_rets, b_rets = [], []
for r in wf_5_1:
    s_rets.append(r["s_ret"]); b_rets.append(r["b_ret"])
    print(f"{r['year']:>6} {r['s_ret']*100:>8.2f}% {r['b_ret']*100:>8.2f}% "
          f"{(r['s_ret']-r['b_ret'])*100:>+8.2f}% "
          f"{r['s_Sh']:>8.3f} {r['b_Sh']:>8.3f} "
          f"{r['s_DD']*100:>8.1f}% {r['b_DD']*100:>8.1f}%")
print("-" * 90)
import statistics as st
print(f"{'均值':>6} {st.mean(s_rets)*100:>8.2f}% {st.mean(b_rets)*100:>8.2f}% "
      f"{(st.mean(s_rets)-st.mean(b_rets))*100:>+8.2f}%")
print(f"{'中位':>6} {st.median(s_rets)*100:>8.2f}% {st.median(b_rets)*100:>8.2f}%")
print(f"{'标差':>6} {st.stdev(s_rets)*100:>8.2f}% {st.stdev(b_rets)*100:>8.2f}%")
win = sum(1 for r in wf_5_1 if r['s_ret'] > r['b_ret']) / len(wf_5_1)
print(f"{'胜率':>6} {win*100:>8.1f}% (策略跑赢BH的年份占比)")
dd_win = sum(1 for r in wf_5_1 if r['s_DD'] > r['b_DD']) / len(wf_5_1)
print(f"{'DD胜率':>6} {dd_win*100:>7.1f}% (策略MaxDD优于BH的年份占比)")

# 保存
pd.DataFrame(res).to_csv(os.path.join(OUT_DIR, "tt_split_v2.csv"), index=False)
pd.DataFrame(wf_5_1).to_csv(os.path.join(OUT_DIR, "walkforward_v2.csv"), index=False)
print(f"\n保存到 {OUT_DIR}")
