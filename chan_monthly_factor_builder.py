#!/usr/bin/env python3
"""
Method A: Run the full daily Chan Theory pipeline per stock via ChanTheoryAnalyzer,
then aggregate into monthly factors. Replaces proxy-based approximations.

Output: chan_monthly_factors.csv
"""

import os, sys, time, hashlib, json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from get_stock_info import fetch_a_share_daily_from_sina
from chan_theory_factors import ChanTheoryAnalyzer, FractalType, Direction

MAX_WORKERS = 8
DATA_LEN = 500  # daily bars
CACHE_DIR = ".cache/chan_factors_v2"
NUM_TEST_STOCKS = 100  # for full run, cover the CSV universe


def safe_value(val, default=0.0):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return val


def fetch_and_aggregate(
    code: str, verbose: bool = False
) -> Tuple[str, Optional[List[dict]], Optional[str]]:
    """Fetch daily → ChanTheoryAnalyzer → monthly factors."""
    try:
        daily = fetch_a_share_daily_from_sina(code, datalen=DATA_LEN)
        if not daily or len(daily) < 60:
            return (code, None, f"Only {len(daily) if daily else 0} bars")

        df = pd.DataFrame(daily)
        df = df.rename(columns={
            "date": "date", "open": "open", "high": "high",
            "low": "low", "close": "close", "volume": "volume",
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        analyzer = ChanTheoryAnalyzer()
        results = analyzer.analyze(df)

        # ── Build monthly rows ──
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        month_groups = df.groupby(["year", "month"])

        monthly_rows = []
        for (y, m), md in month_groups:
            m_start = md.index[0]
            m_end = md.index[-1]

            # Fractals in this month
            fractals_in_month = [
                f for f in analyzer.fractals
                if m_start <= f.index <= m_end and f.is_confirmed
            ]
            top_cnt = sum(1 for f in fractals_in_month if f.fractal_type == FractalType.TOP)
            bot_cnt = sum(1 for f in fractals_in_month if f.fractal_type == FractalType.BOTTOM)

            # Strokes ending in this month
            strokes_in_month = [
                s for s in analyzer.strokes
                if m_start <= s.end_idx <= m_end
            ]
            last_stroke_dir = 0
            stroke_strengths = []
            for s in strokes_in_month:
                if s.start_price and s.start_price != 0:
                    stroke_strengths.append(abs(s.end_price - s.start_price) / abs(s.start_price))
            if strokes_in_month:
                last_stroke_dir = 1 if strokes_in_month[-1].direction == Direction.UP else -1

            # Zhongshu at month end
            zs_list = analyzer.zhongshu_list if hasattr(analyzer, 'zhongshu_list') else []
            active_zs = [z for z in zs_list if z.start_idx <= m_end <= z.end_idx]
            if not active_zs:
                active_zs = [z for z in zs_list if z.end_idx <= m_end]
            zs_pos = 0.0
            zs_width = 0.0
            if active_zs:
                zs = active_zs[-1]
                close_now = df["close"].iloc[m_end]
                if close_now > zs.ZG:
                    zs_pos = 1.0
                elif close_now < zs.ZD:
                    zs_pos = -1.0
                mid = (zs.ZG + zs.ZD) / 2
                if mid > 0:
                    zs_width = (zs.ZG - zs.ZD) / mid

            # Divergence signals from divergence_df (index-aligned with df)
            div_top = 0
            div_bot = 0
            div_df = results.get("divergence_df", pd.DataFrame())
            if not div_df.empty:
                md_div = div_df.iloc[m_start:m_end + 1]
                div_signals = md_div["divergence_signal"].values
                div_bot = int((div_signals == -1).sum())
                div_top = int((div_signals == 1).sum())

            # Trade signals from trade_signals_df (index-aligned with df)
            buy_sig = 0
            sell_sig = 0
            sig_df = results.get("trade_signals_df", pd.DataFrame())
            if not sig_df.empty:
                md_sig = sig_df.iloc[m_start:m_end + 1]
                buy_sig = int((md_sig["buy_signal"] > 0).sum())
                sell_sig = int((md_sig["sell_signal"] > 0).sum())

            div_sig = 1 if div_bot > 0 else (-1 if div_top > 0 else 0)

            # Use the actual last trading date of this month
            last_date = md["date"].iloc[-1].strftime("%Y-%m-%d")

            row = {
                "交易日期": last_date,
                "股票代码": code,
                "chan_top_fractal": top_cnt,
                "chan_bottom_fractal": bot_cnt,
                "chan_fractal_ratio": safe_value(bot_cnt / max(top_cnt + bot_cnt, 1), 0.5),
                "chan_stroke_dir": last_stroke_dir,
                "chan_stroke_count": len(strokes_in_month),
                "chan_stroke_strength": safe_value(np.mean(stroke_strengths) if stroke_strengths else 0.0),
                "chan_zhongshu_count": len(active_zs),
                "chan_zhongshu_position": zs_pos,
                "chan_zhongshu_width": zs_width,
                "chan_top_div": div_top,
                "chan_bottom_div": div_bot,
                "chan_div_signal": div_sig,
                "chan_buy_signals": buy_sig,
                "chan_sell_signals": sell_sig,
                "chan_segment_count": len(analyzer.segments),
            }
            monthly_rows.append(row)

        return (code, monthly_rows, None)

    except Exception as e:
        return (code, None, str(e)[:120])


def build_factor_csv(
    stock_codes: List[str],
    max_stocks: int = NUM_TEST_STOCKS,
    cache_path: str = "",
) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)

    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached: {cache_path}")
        return pd.read_csv(cache_path)

    codes = stock_codes[:max_stocks]
    all_rows = []
    errors = []
    done = 0
    t0 = time.time()

    print(f"Building Chan monthly factors for {len(codes)} stocks ({MAX_WORKERS} workers)...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futmap = {ex.submit(fetch_and_aggregate, c): c for c in codes}
        for fu in as_completed(futmap):
            code, rows, err = fu.result()
            done += 1
            if rows:
                all_rows.extend(rows)
            else:
                errors.append(f"{code}: {err}")
            if done % 25 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed else 0
                eta = (len(codes) - done) / rate if rate else 0
                print(f"  {done}/{len(codes)}  rate={rate:.1f}/s  errors={len(errors)}  ETA={eta/60:.1f}min")

    if not all_rows:
        print("No rows generated — all stocks failed.")
        if errors:
            print(f"Errors ({len(errors)} / {len(codes)}):")
            for e in errors[:10]:
                print(f"  {e}")
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    if cache_path:
        df.to_csv(cache_path, index=False, encoding="gbk")
        print(f"Saved {len(df)} rows → {cache_path}")
    if errors:
        print(f"Errors: {len(errors)} / {len(codes)}")
        for e in errors[:5]:
            print(f"  {e}")
    return df


if __name__ == "__main__":
    # Sample run
    csv_path = "stock_trade_demo/stock_data.csv"
    df_all = pd.read_csv(csv_path, encoding="gbk", low_memory=False)
    all_codes = sorted(df_all["股票代码"].unique())
    all_codes = [c for c in all_codes if c.startswith(("sh", "sz"))]
    print(f"Universe: {len(all_codes)} stocks")

    test_n = min(30, len(all_codes))
    cache_file = os.path.join(CACHE_DIR, f"chan_factors_{test_n}.csv")
    df = build_factor_csv(all_codes, max_stocks=test_n, cache_path=cache_file)
    print(f"\nResult: {len(df)} rows, {df['股票代码'].nunique()} stocks")
    print(df.describe())
