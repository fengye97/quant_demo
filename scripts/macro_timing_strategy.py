"""
美股宏观多因子择时策略 v2.0
================================================================
核心思想:
  使用宏观因子构建市场"健康度"评分,在不利环境降低仓位/空仓,
  在有利环境满仓持有,以追求超越买入持有(BH)的风险调整收益.

因子体系 (5大类, 8因子):
  1) 货币环境:    Fed Funds Rate 趋势 + 利率方向
  2) 流动性:      10Y-2Y期限利差 (倒挂=衰退前兆)
  3) 通胀压力:    Core CPI同比变化方向
  4) 经济动能:    失业率3MA vs 12MA (Sahm Rule 简化版)
  5) 市场情绪:    VIX水位 + 200日均线 + 信用利差

每个因子独立打分(-1/0/+1),加权汇总得到 RegimeScore.
仓位映射:
   score >=  3 : 100% 持仓 (QQQ优先)
   score in [1,2]:  60% 持仓
   score in [-1,0]: 30% 持仓
   score <= -2 :   0% 持仓 (持现金/短债)

回测对象: 纳指100 (NASDAQ100) + 标普500 (SP500)
回测窗口: 2005-至今 (纳指), 2016-至今 (标普500)
基准: 买入持有 (Buy & Hold)
================================================================
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime

DATA_DIR = "/Users/fatcat/Desktop/quant/data"
OUT_DIR  = "/Users/fatcat/Desktop/quant/strategy"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------------------------------------
# 1. 读取数据
# -----------------------------------------------------------
def load(name):
    fp = os.path.join(DATA_DIR, f"fred_{name}.csv")
    df = pd.read_csv(fp, parse_dates=[0], index_col=0)
    return df

ndx  = load("NASDAQ100_FRED")
spx  = load("SP500_FRED")
cpi  = load("CPI_core")
ff   = load("FedFundsRate")
yc   = load("YieldCurve_10Y2Y")
ur   = load("Unemployment")
vix  = load("VIX")
hys  = load("HighYieldSpread")
t10  = load("Treasury10Y")

# 合并到日频
def to_daily(df, name):
    df = df.dropna()
    df.columns = [name]
    return df.resample("D").ffill()

px_ndx = to_daily(ndx, "NDX")
px_spx = to_daily(spx, "SPX")

# 用最长可用区间(由NDX决定)作为主面板
panel = pd.DataFrame(index=pd.date_range("2005-01-01", datetime.today(), freq="D"))
panel["NDX"] = to_daily(ndx, "NDX")["NDX"]
panel["SPX"] = to_daily(spx, "SPX")["SPX"]
panel["CPI_core"] = to_daily(cpi, "CPI_core")["CPI_core"]
panel["FedFunds"] = to_daily(ff, "FedFunds")["FedFunds"]
panel["YC_10Y2Y"] = to_daily(yc, "YC_10Y2Y")["YC_10Y2Y"]
panel["Unemp"]    = to_daily(ur, "Unemp")["Unemp"]
panel["VIX"]      = to_daily(vix, "VIX")["VIX"]
panel["HYS"]      = to_daily(hys, "HYS")["HYS"]

# -----------------------------------------------------------
# 2. 构建因子信号 (5大类)
# -----------------------------------------------------------
def build_factors(df):
    f = pd.DataFrame(index=df.index)

    # 因子1: 货币环境 — Fed Funds 3MA 同比变化
    ff_3m  = df["FedFunds"].rolling(90).mean()
    ff_12m = df["FedFunds"].rolling(365).mean()
    f["F1_Money"] = np.where(ff_3m < ff_12m, 1,
                     np.where(ff_3m > ff_12m + 0.5, -1, 0))

    # 因子2: 流动性 — 期限利差
    # YC > 0 健康;  -0.5 < YC < 0 警戒; YC < -0.5 严重倒挂
    f["F2_Liquidity"] = np.where(df["YC_10Y2Y"] > 0.5, 1,
                          np.where(df["YC_10Y2Y"] < -0.3, -1, 0))

    # 因子3: 通胀方向 — Core CPI 同比变化 (3M momentum)
    cpi_yoy = df["CPI_core"].pct_change(365) * 100
    cpi_chg = cpi_yoy.diff(90)  # 3个月前同比 vs 当前同比
    f["F3_Inflation"] = np.where(cpi_chg < -0.3, 1,           # 通胀回落
                          np.where(cpi_chg > 0.5, -1, 0))     # 通胀重燃

    # 因子4: 经济动能 — Sahm Rule 简化版 (失业率3MA vs 12个月最低)
    u3m  = df["Unemp"].rolling(90).mean()
    u_min12m = df["Unemp"].rolling(365).min()
    sahm = u3m - u_min12m
    f["F4_Economy"] = np.where(sahm < 0.2, 1,
                        np.where(sahm > 0.5, -1, 0))

    # 因子5a: 趋势 — 价格 vs 200日均线
    ma200 = df["NDX"].rolling(200).mean()
    f["F5a_Trend"] = np.where(df["NDX"] > ma200 * 1.02, 1,
                       np.where(df["NDX"] < ma200 * 0.98, -1, 0))

    # 因子5b: 情绪 — VIX 水位
    f["F5b_VIX"] = np.where(df["VIX"] < 15, 1,
                    np.where(df["VIX"] > 25, -1, 0))

    # 因子5c: 信用利差 — 高收益债利差
    hys_ma20 = df["HYS"].rolling(20).mean()
    f["F5c_Credit"] = np.where(hys_ma20 < 4.0, 1,
                        np.where(hys_ma20 > 6.0, -1, 0))

    # 汇总分数
    factor_cols = ["F1_Money", "F2_Liquidity", "F3_Inflation",
                   "F4_Economy", "F5a_Trend", "F5b_VIX", "F5c_Credit"]
    f["RegimeScore"] = f[factor_cols].sum(axis=1)
    return f

factors = build_factors(panel)

# -----------------------------------------------------------
# 3. 仓位映射
# -----------------------------------------------------------
def score_to_position(s):
    if pd.isna(s): return np.nan
    if s >= 3:  return 1.00
    if s >= 1:  return 0.60
    if s >= -1: return 0.30
    return 0.00

factors["Position"] = factors["RegimeScore"].apply(score_to_position)

# 月度调仓 (避免频繁交易,每月第1个交易日按上月末信号)
positions_monthly = factors["Position"].resample("M").last().ffill()
panel["Position"] = positions_monthly.reindex(panel.index, method="ffill")

# -----------------------------------------------------------
# 4. 回测引擎
# -----------------------------------------------------------
def backtest(price, position, name, cash_rate=0.02, tc=0.0005):
    df = pd.DataFrame({"price": price, "pos": position}).dropna()
    df = df[df.index.dayofweek < 5]  # 仅交易日
    df["ret"]   = df["price"].pct_change().fillna(0)
    df["pos_l"] = df["pos"].shift(1).fillna(0)  # T-1信号
    # 现金部分按年化cash_rate
    df["cash_ret"] = (1 + cash_rate)**(1/252) - 1
    # 策略收益
    df["strat_ret"] = df["pos_l"] * df["ret"] + (1 - df["pos_l"]) * df["cash_ret"]
    # 交易成本
    df["turnover"] = df["pos_l"].diff().abs().fillna(0)
    df["strat_ret"] -= df["turnover"] * tc
    df["strat_nv"] = (1 + df["strat_ret"]).cumprod()
    df["bh_nv"]    = (1 + df["ret"]).cumprod()

    # 绩效统计
    yrs = (df.index[-1] - df.index[0]).days / 365.25
    def perf(nv_col, ret_col):
        cagr = nv_col.iloc[-1] ** (1/yrs) - 1
        vol  = ret_col.std() * np.sqrt(252)
        sharpe = (ret_col.mean() * 252 - 0.02) / vol if vol > 0 else 0
        roll_max = nv_col.cummax()
        mdd = ((nv_col - roll_max) / roll_max).min()
        calmar = cagr / abs(mdd) if mdd != 0 else 0
        return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe,
                "MaxDD": mdd, "Calmar": calmar}
    strat_stats = perf(df["strat_nv"], df["strat_ret"])
    bh_stats    = perf(df["bh_nv"], df["ret"])

    # 持仓占比
    avg_pos = df["pos_l"].mean()
    turnover = df["turnover"].sum() / yrs

    return df, strat_stats, bh_stats, avg_pos, turnover

# -----------------------------------------------------------
# 5. 对NDX和SPX分别回测
# -----------------------------------------------------------
results = {}
for name, col in [("NDX", "NDX"), ("SPX", "SPX")]:
    df_bt, strat, bh, avg_pos, turn = backtest(panel[col], panel["Position"], name)
    df_bt.to_csv(os.path.join(OUT_DIR, f"backtest_{name}.csv"))
    results[name] = {"strat": strat, "bh": bh, "avg_pos": avg_pos,
                     "turnover": turn, "df": df_bt}

# -----------------------------------------------------------
# 6. 输出结果
# -----------------------------------------------------------
print("\n" + "=" * 70)
print("美股宏观多因子择时策略 v2.0 — 回测结果")
print("=" * 70)
for name, r in results.items():
    df = r["df"]
    print(f"\n【{name}】  样本: {df.index[0].date()} → {df.index[-1].date()}  "
          f"共 {len(df)} 个交易日")
    print(f"  平均仓位 = {r['avg_pos']*100:.1f}%   年换手 = {r['turnover']:.2f}")
    print(f"  {'指标':<10}{'策略':>15}{'买入持有(BH)':>20}{'超额':>15}")
    print(f"  {'-'*60}")
    for key in ["CAGR", "Vol", "Sharpe", "MaxDD", "Calmar"]:
        s = r["strat"][key]; b = r["bh"][key]
        if key in ("CAGR", "Vol", "MaxDD"):
            print(f"  {key:<10}{s*100:>14.2f}%{b*100:>19.2f}%{(s-b)*100:>14.2f}%")
        else:
            print(f"  {key:<10}{s:>15.3f}{b:>20.3f}{s-b:>15.3f}")

# 当前最新信号
print("\n" + "=" * 70)
print(f"当前因子状态 (截至 {factors.dropna().index[-1].date()})")
print("=" * 70)
latest = factors.dropna().iloc[-1]
factor_cols = ["F1_Money", "F2_Liquidity", "F3_Inflation",
               "F4_Economy", "F5a_Trend", "F5b_VIX", "F5c_Credit"]
for c in factor_cols:
    v = int(latest[c])
    flag = "✓ 正面" if v > 0 else ("✗ 负面" if v < 0 else "○ 中性")
    print(f"  {c:<15} = {v:+d}   {flag}")
print(f"  {'RegimeScore':<15} = {int(latest['RegimeScore']):+d}")
print(f"  建议仓位     = {latest['Position']*100:.0f}%")

# 保存信号文件
factors.to_csv(os.path.join(OUT_DIR, "factor_signals_history.csv"))
print(f"\n输出文件 → {OUT_DIR}")
