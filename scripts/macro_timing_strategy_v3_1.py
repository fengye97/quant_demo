"""
美股宏观多因子择时策略 v3.1 — Continuous Staged Position
==================================================================
v3.0 → v3.1 核心改进:

  1. **因子从 ternary {-1,0,+1} 升级为 continuous z-score**
     - 用滚动 z-score 量化每个因子的偏离程度
     - 保留方向信号但增加幅度信息

  2. **RegimeScore → 连续平滑仓位映射**
     - 用 logistic sigmoid 函数: pos = sig(score * k) * max_lev
     - 取消硬阈值跳变 (3档/0.5/0.75/1.0/1.3 → 连续区间)
     - 在score边界附近避免一次性大幅调仓

  3. **更激进的仓位上限** (响应v3.0过保守问题)
     - max_lev 提升到 1.5x (vs v3.0 1.3x)
     - 牛市 score 高时容易触及上限

  4. **加入趋势确认 multiplier**
     - 当NDX处于明确上升趋势(MA200趋势向上)时,position * 1.15
     - 减少在2023这类强趋势年份的踏空

  5. **降低交易摩擦**
     - 仓位变化 <3% 不调仓 (避免噪音换手)
     - 调仓阈值从月度收紧到双月

执行结果应该:
  ✓ 牛市跟随性更好 (2019/2020/2023)
  ✓ 仍保持回撤优势 (2008/2022)
  ✓ Sharpe / Calmar 进一步提升
==================================================================
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime

DATA_DIR = "/Users/fatcat/Desktop/quant/data"
OUT_DIR  = "/Users/fatcat/Desktop/quant/strategy"
os.makedirs(OUT_DIR, exist_ok=True)

def load(name):
    return pd.read_csv(os.path.join(DATA_DIR, f"fred_{name}.csv"),
                       parse_dates=[0], index_col=0)

def to_daily(df, col):
    df = df.dropna(); df.columns = [col]
    return df.resample("D").ffill()

# ---------------- 数据 ----------------
panel = pd.DataFrame(index=pd.date_range("2005-01-01", datetime.today(), freq="D"))
for code, col in [("NASDAQ100_FRED","NDX"), ("SP500_FRED","SPX"),
                  ("CPI_core","CPI_core"), ("FedFundsRate","FedFunds"),
                  ("YieldCurve_10Y2Y","YC_10Y2Y"), ("Unemployment","Unemp"),
                  ("VIX","VIX"), ("HighYieldSpread","HYS"),
                  ("Treasury10Y","T10Y")]:
    panel[col] = to_daily(load(code), col)[col]
panel["HYS_proxy"] = panel["HYS"].fillna(panel["VIX"] / 5.0)

# ---------------- 连续因子 (z-score化) ----------------
def zscore(s, window=252*5, min_periods=252):
    """5年滚动z-score, 避免lookahead"""
    mu  = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std()
    z = (s - mu) / (std + 1e-9)
    return z.clip(-3, 3)  # 截断极端值

def build_continuous_factors(df):
    """连续因子 — 每个返回[-3,+3]范围的z-score(已含方向)"""
    f = pd.DataFrame(index=df.index)

    # F1 货币: 降息周期=正面 (符号已反转: ff下降 → z为正)
    ff_diff = df["FedFunds"].rolling(90).mean() - df["FedFunds"].rolling(365).mean()
    f["Z1_Money"] = -zscore(ff_diff)  # ff_diff 越负 → 因子越正

    # F2 流动性: 期限利差水平(>0=健康) + 变化方向
    yc_lvl  = zscore(df["YC_10Y2Y"])
    yc_chg  = zscore(df["YC_10Y2Y"].diff(60))
    f["Z2_Liquidity"] = (yc_lvl + yc_chg) / 2

    # F3 通胀: CPI同比变化方向 (回落=正面)
    cpi_yoy = (df["CPI_core"] / df["CPI_core"].shift(365) - 1) * 100
    cpi_mom = cpi_yoy.diff(90)
    f["Z3_Inflation"] = -zscore(cpi_mom)  # mom下降=正面
    # 高通胀绝对水平惩罚
    cpi_lvl_penalty = -zscore(cpi_yoy.clip(lower=2.0))
    f["Z3_Inflation"] = (f["Z3_Inflation"] + cpi_lvl_penalty) / 2

    # F4 经济: Sahm变量 (失业上升=负面)
    sahm = df["Unemp"].rolling(90).mean() - df["Unemp"].rolling(365).min()
    f["Z4_Economy"] = -zscore(sahm)

    # F5a 趋势: 价格相对MA200的偏离
    ma200 = df["NDX"].rolling(200).mean()
    trend_bias = (df["NDX"] / ma200 - 1) * 100
    f["Z5a_Trend"] = zscore(trend_bias)

    # F5b 短期动量: MA20/MA60
    mom_bias = (df["NDX"].rolling(20).mean() / df["NDX"].rolling(60).mean() - 1) * 100
    f["Z5b_Mom"] = zscore(mom_bias)

    # F5c 情绪: VIX (低VIX=正面)
    f["Z5c_VIX"] = -zscore(df["VIX"])

    # F5d 信用: HighYieldSpread
    f["Z5d_Credit"] = -zscore(df["HYS_proxy"])

    # 汇总: 简单平均(等权)所有8个因子的z-score
    zcols = [c for c in f.columns if c.startswith("Z")]
    f["ContScore"] = f[zcols].mean(axis=1)  # 均值范围约[-1.5, +1.5]

    # 危机覆盖
    crisis = df["VIX"] > 35
    f["Crisis"] = crisis.astype(int)
    f.loc[crisis, "ContScore"] = -2.0  # 危机模式

    # 趋势确认multiplier
    ma200_slope = ma200.diff(60) / ma200.shift(60)  # 60日斜率
    f["TrendMult"] = np.where(ma200_slope > 0.02, 1.15,
                       np.where(ma200_slope < -0.02, 0.85, 1.0))
    return f

# ---------------- 连续仓位映射 (sigmoid) ----------------
def cont_score_to_pos(score, k=2.0, max_lev=1.5, base=0.5):
    """
    sigmoid 映射: score 越正,仓位越大,平滑过渡
    score = 0 → pos = base * max_lev = 0.75
    score = +1.5 → pos ≈ max_lev*0.95 ≈ 1.43
    score = -1.5 → pos ≈ max_lev*0.05 ≈ 0.075
    """
    if pd.isna(score): return np.nan
    sig = 1 / (1 + np.exp(-k * score))  # logistic
    return max(0.0, min(max_lev, sig * max_lev))

def cont_score_to_tilt(score):
    """QQQ vs SPY 倾斜 — 连续版"""
    if pd.isna(score): return np.nan
    # score>0 倾向QQQ, score<0 倾向SPY
    sig = 1 / (1 + np.exp(-1.5 * score))
    return max(0.2, min(1.0, 0.3 + 0.7 * sig))

factors = build_continuous_factors(panel)
factors["Position_raw"] = factors["ContScore"].apply(cont_score_to_pos)
factors["Position"] = factors["Position_raw"] * factors["TrendMult"]
factors["Position"] = factors["Position"].clip(0.0, 1.5)
factors["QQQ_weight"] = factors["ContScore"].apply(cont_score_to_tilt)

# 调仓: 月度,但变化<3%不动 (减少摩擦)
pos_m = factors["Position"].resample("ME").last().ffill()
qw_m  = factors["QQQ_weight"].resample("ME").last().ffill()

# 加入"小变动免调仓"
def apply_inertia(series, threshold=0.03):
    out = series.copy()
    last = out.iloc[0]
    for i in range(1, len(out)):
        if abs(out.iloc[i] - last) < threshold:
            out.iloc[i] = last
        else:
            last = out.iloc[i]
    return out

pos_m = apply_inertia(pos_m, 0.03)
qw_m  = apply_inertia(qw_m,  0.05)

panel["Position"]   = pos_m.reindex(panel.index, method="ffill")
panel["QQQ_weight"] = qw_m.reindex(panel.index, method="ffill")

# ---------------- 回测 ----------------
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

def backtest_ndx_only(df, cash_rate=0.04, tc=0.0005):
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
bt_ndx = backtest_ndx_only(panel)

stats = [
    perf(bt_mix["strat_nv"],   bt_mix["strat_ret"],  "v3.1混合(2016-)"),
    perf(bt_mix["bh_ndx_nv"],  bt_mix["ret_ndx"],    "BH-NDX(2016-)"),
    perf(bt_mix["bh_spx_nv"],  bt_mix["ret_spx"],    "BH-SPX(2016-)"),
    perf(bt_ndx["strat_nv"],   bt_ndx["strat_ret"],  "v3.1纯NDX(2005-)"),
    perf(bt_ndx["bh_nv"],      bt_ndx["ret"],        "BH-NDX(2005-)"),
]

print("\n" + "=" * 90)
print("v3.1 Continuous Staged Position — 回测")
print("=" * 90)
print(f"{'方案':<25}{'年数':>6}{'CAGR':>9}{'Vol':>8}{'Sharpe':>9}{'MaxDD':>9}{'Calmar':>9}")
print("-" * 90)
for s in stats:
    print(f"{s['name']:<25}{s['yrs']:>6.1f}{s['CAGR']*100:>8.2f}%{s['Vol']*100:>7.1f}%"
          f"{s['Sharpe']:>9.3f}{s['MaxDD']*100:>8.1f}%{s['Calmar']:>9.3f}")

# 年度对比
bt_ndx_ann = bt_ndx.copy()
bt_ndx_ann["year"] = bt_ndx_ann.index.year
ann = bt_ndx_ann.groupby("year").agg(
    strat=("strat_ret", lambda x: (1+x).prod()-1),
    bh=("ret", lambda x: (1+x).prod()-1),
    pos=("pos_l", "mean"))

print("\n" + "=" * 60)
print("年度收益 v3.1 vs BH-NDX")
print("=" * 60)
print(f"{'年份':>6}{'v3.1':>10}{'BH-NDX':>10}{'超额':>10}{'平均仓位':>12}")
print("-" * 60)
strat_win = 0; total = 0
for y, r in ann.iterrows():
    diff = r['strat'] - r['bh']
    if diff > 0: strat_win += 1
    total += 1
    print(f"{y:>6}{r['strat']*100:>9.1f}%{r['bh']*100:>9.1f}%{diff*100:>+9.1f}%{r['pos']*100:>10.0f}%")
print("-" * 60)
print(f"{'胜率':>6}{strat_win/total*100:>9.1f}% ({strat_win}/{total})")

# 当前信号
print("\n" + "=" * 60)
print("最新信号 (v3.1)")
print("=" * 60)
valid = factors[[c for c in factors.columns if c.startswith("Z")]].dropna()
if len(valid) > 0:
    last = valid.index[-1]
    print(f"参考日期: {last.date()}")
    for c in [c for c in factors.columns if c.startswith("Z")]:
        v = factors.loc[last, c]
        bar = "█" * int(abs(v) * 5)
        sign = "+" if v >= 0 else "-"
        print(f"  {c:<18} = {v:>+6.2f}  {sign}{bar}")
    cs = factors.loc[last, "ContScore"]
    tm = factors.loc[last, "TrendMult"]
    raw = factors.loc[last, "Position_raw"]
    pos = factors.loc[last, "Position"]
    qw  = factors.loc[last, "QQQ_weight"]
    print(f"  {'ContScore':<18} = {cs:>+6.2f}")
    print(f"  {'TrendMult':<18} = {tm:>+6.2f}")
    print(f"  Position(原始) = {raw*100:.0f}%  → ×趋势 = {pos*100:.0f}%")
    print(f"  QQQ权重         = {qw*100:.0f}%  SPY权重 = {(1-qw)*100:.0f}%")

# 保存
bt_mix.to_csv(os.path.join(OUT_DIR, "backtest_v31_mix.csv"))
bt_ndx.to_csv(os.path.join(OUT_DIR, "backtest_v31_ndx.csv"))
factors.to_csv(os.path.join(OUT_DIR, "factor_signals_v31.csv"))
pd.DataFrame(stats).to_csv(os.path.join(OUT_DIR, "v31_performance.csv"), index=False)
ann.to_csv(os.path.join(OUT_DIR, "v31_annual.csv"))
print(f"\n保存到 {OUT_DIR}")
