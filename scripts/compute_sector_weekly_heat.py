"""
按申万一级行业计算周级别涨跌幅热度。

原理：
  - stock_data 是月度频率，每行包含 `下周期每天涨跌幅`（下一月内每日收益列表）
  - 把每月的日线按 5 个交易日一周分组，计算该周的累计收益
  - 按流通市值加权聚合到申万一级行业
  - 输出 strategy/sector_weekly_heat.csv

用法：
  python scripts/compute_sector_weekly_heat.py
"""

import ast, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "stock_trade_demo" / "stock_data.parquet"
OUT_PATH = ROOT / "strategy" / "sector_weekly_heat.csv"

INDUSTRY_COL = "新版申万一级行业名称"
DAYS_PER_WEEK = 5
MIN_STOCKS = 5          # 一个 (行业, 周) 最少需要几只股票才纳入

def parse_daily(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    return []

def week_cumret(daily_returns, week_idx: int) -> float:
    """取第 week_idx 周（0-based）5 个交易日的累计收益"""
    start = week_idx * DAYS_PER_WEEK
    chunk = daily_returns[start : start + DAYS_PER_WEEK]
    if not chunk:
        return np.nan
    return np.prod([1 + r for r in chunk]) - 1

def month_offset(date: pd.Timestamp, months: int) -> pd.Timestamp:
    """向后推 months 个月（返回该月最后一天，用 period 做对齐）"""
    p = date.to_period("M") + months
    return p.to_timestamp("M")  # 月末

def main():
    print("读取数据...")
    df = pd.read_parquet(DATA_PATH)

    # 只保留有日线数据且行业标签有效的行
    df = df[df["是否交易"] == 1].copy()
    df["daily_rets"] = df["下周期每天涨跌幅"].apply(parse_daily)
    df = df[df["daily_rets"].apply(len) >= DAYS_PER_WEEK].copy()
    df = df[df[INDUSTRY_COL].notna() & (df[INDUSTRY_COL] != "")].copy()

    # 下周期 = 交易日期的下一个月
    df["next_month"] = df["交易日期"].apply(lambda d: month_offset(d, 1))
    # 最大有几周
    max_weeks = df["daily_rets"].apply(len).max() // DAYS_PER_WEEK
    print(f"月度条数: {len(df):,}，每月最多 {max_weeks} 周")

    # 展开：每行 → 最多 max_weeks 行（对应该下周期的各周）
    records = []
    for row in df.itertuples():
        daily = row.daily_rets
        mktcap = row.流通市值 if not np.isnan(row.流通市值) else 0.0
        n_weeks = len(daily) // DAYS_PER_WEEK
        for w in range(n_weeks):
            cr = week_cumret(daily, w)
            if not np.isnan(cr):
                records.append({
                    "year_month": row.next_month.to_period("M"),
                    "week_in_month": w + 1,           # 1-based
                    "stock_code": row.股票代码,
                    "industry": getattr(row, INDUSTRY_COL),
                    "mktcap": mktcap,
                    "weekly_ret": cr,
                })

    long_df = pd.DataFrame(records)
    long_df["year_month_str"] = long_df["year_month"].astype(str)

    # 构建全局唯一的周标签，如 "2025-04 W2"
    long_df["week_label"] = long_df["year_month_str"] + " W" + long_df["week_in_month"].astype(str)

    print(f"展开后条数: {len(long_df):,}")

    # 按 (周标签, 行业) 聚合 → 流通市值加权平均收益
    def wavg(g):
        w = g["mktcap"].clip(lower=0)
        if w.sum() == 0:
            w = pd.Series(np.ones(len(g)), index=g.index)
        return np.average(g["weekly_ret"], weights=w)

    grp = long_df.groupby(["year_month", "week_in_month", "week_label", "industry"])
    agg = grp.apply(
        lambda g: pd.Series({
            "weekly_ret": wavg(g),
            "n_stocks": len(g),
        })
    ).reset_index()

    agg = agg[agg["n_stocks"] >= MIN_STOCKS].copy()
    agg["weekly_ret_pct"] = (agg["weekly_ret"] * 100).round(2)
    agg.sort_values(["year_month", "week_in_month", "weekly_ret"], ascending=[True, True, False], inplace=True)

    # 保存
    out_cols = ["year_month", "week_in_month", "week_label", "industry", "weekly_ret_pct", "n_stocks"]
    agg[out_cols].to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"已保存 → {OUT_PATH}  (行数: {len(agg):,})")

    # ── 最近 8 周行业热度预览 ──
    latest_weeks = sorted(agg["week_label"].unique())[-8:]
    pivot = (
        agg[agg["week_label"].isin(latest_weeks)]
        .pivot_table(index="industry", columns="week_label", values="weekly_ret_pct")
    )
    # 排列列顺序
    pivot = pivot[sorted(pivot.columns)]
    # 按最新一周排序
    pivot = pivot.sort_values(pivot.columns[-1], ascending=False)

    print("\n===== 最近 8 周行业涨跌幅（%，市值加权）=====")
    print(pivot.to_string())

    # 热度汇总：最近 4 周平均收益 × 行业排名
    recent_4w = sorted(agg["week_label"].unique())[-4:]
    heat = (
        agg[agg["week_label"].isin(recent_4w)]
        .groupby("industry")["weekly_ret_pct"]
        .mean()
        .sort_values(ascending=False)
        .rename("近4周平均周收益(%)")
    )
    print("\n===== 近 4 周行业热度排名 =====")
    print(heat.to_string())

if __name__ == "__main__":
    main()
