"""
美股宏观择时策略 — Train/Test 切分验证与 Walk-Forward 分析
=================================================================
目的: 检验 v3.0 策略的过拟合程度

实验设计:
  实验A: 单次切分 (5种)
    A1: 全样本 (基线)              2005-2026
    A2: 标准切分 70/30              train 2005-2019, test 2020-2026
    A3: 短训练 60/40                train 2005-2017, test 2018-2026
    A4: 极短训练 40/60              train 2005-2013, test 2014-2026
    A5: 最短训练 30/70              train 2005-2010, test 2011-2026

  实验B: Walk-Forward (滚动窗口)
    B1: 5年训练 + 1年测试,每年滚动
    B2: 3年训练 + 1年测试,每年滚动

测试指标: CAGR / Sharpe / MaxDD / Calmar
=================================================================
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

def to_daily(df, col):
    df = df.dropna(); df.columns = [col]
    return df.resample("D").ffill()

# ---------------- 数据加载 ----------------
panel = pd.DataFrame(index=pd.date_range("2005-01-01", datetime.today(), freq="D"))
for code, col in [("NASDAQ100_FRED","NDX"), ("SP500_FRED","SPX"),
                  ("CPI_core","CPI_core"), ("FedFundsRate","FedFunds"),
                  ("YieldCurve_10Y2Y","YC_10Y2Y"), ("Unemployment","Unemp"),
                  ("VIX","VIX"), ("HighYieldSpread","HYS"),
                  ("Treasury10Y","T10Y")]:
    panel[col] = to_daily(load(code), col)[col]
panel["HYS_proxy"] = panel["HYS"].fillna(panel["VIX"] / 5.0)

# ---------------- v3.0 因子构建 (与生产版完全一致) ----------------
def build_factors_v3(df):
    f = pd.DataFrame(index=df.index)
    ff3 = df["FedFunds"].rolling(90).mean()
    ff12 = df["FedFunds"].rolling(365).mean()
    diff = ff3 - ff12
    f["F1_Money"] = np.where(diff < -0.1, 1, np.where(diff > 0.3, -1, 0))

    yc = df["YC_10Y2Y"]; yc_chg = yc.diff(60)
    f["F2_Liquidity"] = np.where((yc > 0) & (yc_chg > 0), 1,
                          np.where((yc < -0.3) & (yc_chg < 0), -1, 0))

    cpi_yoy = (df["CPI_core"] / df["CPI_core"].shift(365) - 1) * 100
    cpi_mom = cpi_yoy.diff(90)
    f["F3_Inflation"] = np.where(cpi_mom < -0.2, 1, np.where(cpi_mom > 0.3, -1, 0))
    f["F3_Inflation"] = np.where(cpi_yoy > 4.0, -1, f["F3_Inflation"])

    u3 = df["Unemp"].rolling(90).mean()
    u_min = df["Unemp"].rolling(365).min()
    sahm = u3 - u_min
    f["F4_Economy"] = np.where(sahm < 0.2, 1, np.where(sahm > 0.4, -1, 0))

    ma200 = df["NDX"].rolling(200).mean()
    f["F5a_Trend"] = np.where(df["NDX"] > ma200 * 1.02, 1,
                       np.where(df["NDX"] < ma200 * 0.97, -1, 0))

    ma20, ma60 = df["NDX"].rolling(20).mean(), df["NDX"].rolling(60).mean()
    f["F5b_Momentum"] = np.where(ma20 > ma60 * 1.01, 1,
                          np.where(ma20 < ma60 * 0.99, -1, 0))

    f["F5c_VIX"] = np.where(df["VIX"] < 16, 1, np.where(df["VIX"] > 25, -1, 0))

    hys = df["HYS_proxy"].rolling(20).mean()
    f["F5d_Credit"] = np.where(hys < 4.0, 1, np.where(hys > 6.0, -1, 0))

    crisis = df["VIX"] > 35
    factor_cols = [c for c in f.columns if c.startswith("F")]
    f["RegimeScore"] = f[factor_cols].sum(axis=1)
    f.loc[crisis, "RegimeScore"] = -5
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

def asset_tilt(s):
    if pd.isna(s): return np.nan
    if s >= 3: return 1.0
    if s >= 0: return 0.6
    return 0.3

# ---------------- 回测引擎 (与v2一致) ----------------
def backtest_mix(df, cash_rate=0.04, tc=0.0005):
    d = df.copy()
    d = d.dropna(subset=["NDX","SPX","Position","QQQ_weight"])
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
    return d

def perf(nv, ret):
    if len(nv) < 30: return None
    yrs = (nv.index[-1] - nv.index[0]).days / 365.25
    if yrs < 0.5: return None
    cagr = nv.iloc[-1] ** (1/yrs) - 1
    vol  = ret.std() * np.sqrt(252)
    sh   = (ret.mean() * 252 - 0.04) / vol if vol > 0 else 0
    mdd  = ((nv - nv.cummax()) / nv.cummax()).min()
    cal  = cagr / abs(mdd) if mdd != 0 else 0
    return {"yrs": yrs, "CAGR": cagr, "Vol": vol,
            "Sharpe": sh, "MaxDD": mdd, "Calmar": cal}

# ---------------- 准备完整因子+仓位 ----------------
factors = build_factors_v3(panel)
factors["Position"]   = factors["RegimeScore"].apply(score_to_pos)
factors["QQQ_weight"] = factors["RegimeScore"].apply(asset_tilt)
panel["Position"]     = factors["Position"].resample("ME").last().ffill().reindex(panel.index, method="ffill")
panel["QQQ_weight"]   = factors["QQQ_weight"].resample("ME").last().ffill().reindex(panel.index, method="ffill")

# 全样本回测一次 → 取子区间分析
bt_full = backtest_mix(panel)

# ---------------- 实验A: 单次切分 ----------------
def slice_perf(bt, start, end, label):
    seg = bt.loc[start:end].copy()
    if len(seg) < 30: return None
    # 重新规范化净值从1开始
    seg["strat_nv"] = (1 + seg["strat_ret"]).cumprod()
    seg["bh_ndx_nv"] = (1 + seg["ret_ndx"]).cumprod()
    s = perf(seg["strat_nv"], seg["strat_ret"])
    b = perf(seg["bh_ndx_nv"], seg["ret_ndx"])
    if s is None or b is None: return None
    return {
        "label": label, "start": str(start), "end": str(end),
        "n_days": len(seg),
        "strat_CAGR": s["CAGR"], "strat_Sharpe": s["Sharpe"],
        "strat_MaxDD": s["MaxDD"], "strat_Calmar": s["Calmar"],
        "bh_CAGR": b["CAGR"], "bh_Sharpe": b["Sharpe"],
        "bh_MaxDD": b["MaxDD"], "bh_Calmar": b["Calmar"],
        "alpha_CAGR": s["CAGR"] - b["CAGR"],
        "alpha_Sharpe": s["Sharpe"] - b["Sharpe"],
        "dd_improve": s["MaxDD"] - b["MaxDD"],  # 正数好
    }

splits = [
    ("A1_FullSample",     "2005-02-01", "2026-05-22", "全样本(基线)"),
    ("A2_Train_05-19",    "2005-02-01", "2019-12-31", "Train 2005-2019 (70%)"),
    ("A2_Test_20-26",     "2020-01-01", "2026-05-22", "Test 2020-2026 (30%)"),
    ("A3_Train_05-17",    "2005-02-01", "2017-12-31", "Train 2005-2017 (60%)"),
    ("A3_Test_18-26",     "2018-01-01", "2026-05-22", "Test 2018-2026 (40%)"),
    ("A4_Train_05-13",    "2005-02-01", "2013-12-31", "Train 2005-2013 (40%)"),
    ("A4_Test_14-26",     "2014-01-01", "2026-05-22", "Test 2014-2026 (60%)"),
    ("A5_Train_05-10",    "2005-02-01", "2010-12-31", "Train 2005-2010 (30%)"),
    ("A5_Test_11-26",     "2011-01-01", "2026-05-22", "Test 2011-2026 (70%)"),
]

results_A = []
for tag, start, end, label in splits:
    r = slice_perf(bt_full, start, end, label)
    if r is not None:
        r["tag"] = tag
        results_A.append(r)

print("\n" + "=" * 110)
print("实验A: 单次切分对比 (策略 vs Buy & Hold NDX)")
print("=" * 110)
print(f"{'区间':<28} {'年数':>5} {'策略CAGR':>9} {'BH-CAGR':>9} {'αCAGR':>8} "
      f"{'策略Sh':>7} {'BH-Sh':>7} {'αSh':>7} {'策略DD':>8} {'BH-DD':>8} {'DD改善':>8}")
print("-" * 110)
for r in results_A:
    print(f"{r['label']:<28} {r['n_days']/252:>5.1f} "
          f"{r['strat_CAGR']*100:>8.2f}% {r['bh_CAGR']*100:>8.2f}% "
          f"{r['alpha_CAGR']*100:>+7.2f}% "
          f"{r['strat_Sharpe']:>7.3f} {r['bh_Sharpe']:>7.3f} {r['alpha_Sharpe']:>+7.3f} "
          f"{r['strat_MaxDD']*100:>7.1f}% {r['bh_MaxDD']*100:>7.1f}% "
          f"{r['dd_improve']*100:>+7.1f}%")

# ---------------- 实验B: Walk-Forward ----------------
def walk_forward(bt, train_years, test_years, label_prefix):
    results = []
    start_year = bt.index.year.min() + train_years
    end_year = bt.index.year.max()
    for ty in range(start_year, end_year + 1):
        test_start = pd.Timestamp(f"{ty}-01-01")
        test_end   = pd.Timestamp(f"{ty + test_years - 1}-12-31")
        if test_end > bt.index.max(): test_end = bt.index.max()
        seg = bt.loc[test_start:test_end]
        if len(seg) < 30: continue
        sret = seg["strat_ret"]; bret = seg["ret_ndx"]
        snv = (1 + sret).cumprod(); bnv = (1 + bret).cumprod()
        sp = perf(snv, sret); bp = perf(bnv, bret)
        if sp is None or bp is None: continue
        results.append({
            "year": ty, "label": f"{label_prefix}-{ty}",
            "strat_ret": float(snv.iloc[-1] - 1),
            "bh_ret":   float(bnv.iloc[-1] - 1),
            "strat_Sharpe": sp["Sharpe"], "bh_Sharpe": bp["Sharpe"],
            "strat_MaxDD": sp["MaxDD"],   "bh_MaxDD": bp["MaxDD"],
        })
    return results

wf_5_1 = walk_forward(bt_full, train_years=5, test_years=1, label_prefix="WF5+1")
wf_3_1 = walk_forward(bt_full, train_years=3, test_years=1, label_prefix="WF3+1")

print("\n" + "=" * 95)
print("实验B1: Walk-Forward 5年训练+1年测试 (年度OOS表现)")
print("=" * 95)
print(f"{'测试年':>6} {'策略年收益':>11} {'BH年收益':>11} {'超额':>10} "
      f"{'策略Sh':>8} {'BH-Sh':>8} {'策略DD':>9} {'BH-DD':>9}")
print("-" * 95)
strat_returns_5 = []; bh_returns_5 = []
for r in wf_5_1:
    strat_returns_5.append(r["strat_ret"]); bh_returns_5.append(r["bh_ret"])
    print(f"{r['year']:>6} {r['strat_ret']*100:>10.2f}% {r['bh_ret']*100:>10.2f}% "
          f"{(r['strat_ret']-r['bh_ret'])*100:>+9.2f}% "
          f"{r['strat_Sharpe']:>8.3f} {r['bh_Sharpe']:>8.3f} "
          f"{r['strat_MaxDD']*100:>8.1f}% {r['bh_MaxDD']*100:>8.1f}%")

print("-" * 95)
import statistics as st
print(f"{'均值':>6} {st.mean(strat_returns_5)*100:>10.2f}% {st.mean(bh_returns_5)*100:>10.2f}% "
      f"{(st.mean(strat_returns_5)-st.mean(bh_returns_5))*100:>+9.2f}%")
print(f"{'中位':>6} {st.median(strat_returns_5)*100:>10.2f}% {st.median(bh_returns_5)*100:>10.2f}%")
win_rate = sum(1 for r in wf_5_1 if r['strat_ret'] > r['bh_ret']) / len(wf_5_1)
print(f"{'胜率':>6} {win_rate*100:>10.1f}% (策略跑赢BH的年份占比)")

# 保存所有结果
pd.DataFrame(results_A).to_csv(os.path.join(OUT_DIR, "tt_split_results.csv"), index=False)
pd.DataFrame(wf_5_1).to_csv(os.path.join(OUT_DIR, "walkforward_5_1.csv"), index=False)
pd.DataFrame(wf_3_1).to_csv(os.path.join(OUT_DIR, "walkforward_3_1.csv"), index=False)
print(f"\n结果已保存到 {OUT_DIR}")
