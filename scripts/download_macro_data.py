"""
美股宏观因子 + ETF价格历史数据下载脚本
数据源: FRED (St. Louis Fed) + Yahoo Finance
保存路径: /Users/fatcat/Desktop/quant/data/
"""
import os
import pandas as pd
from datetime import datetime
from pandas_datareader import data as pdr
import yfinance as yf

DATA_DIR = "/Users/fatcat/Desktop/quant/data"
os.makedirs(DATA_DIR, exist_ok=True)

START = "2005-01-01"
END = datetime.today().strftime("%Y-%m-%d")

# FRED 宏观指标代码映射
FRED_SERIES = {
    "CPIAUCSL":   "CPI_headline",         # 整体CPI
    "CPILFESL":   "CPI_core",             # 核心CPI
    "PCEPI":      "PCE_headline",         # PCE
    "PCEPILFE":   "PCE_core",             # 核心PCE
    "UNRATE":     "Unemployment",         # 失业率
    "PAYEMS":     "NonFarmPayrolls",      # 非农就业人数
    "FEDFUNDS":   "FedFundsRate",         # 联邦基金利率
    "DGS10":      "Treasury10Y",          # 10年期美债收益率
    "DGS2":       "Treasury2Y",           # 2年期美债收益率
    "T10Y2Y":     "YieldCurve_10Y2Y",     # 期限利差
    "DTWEXBGS":   "DollarIndex",          # 美元指数(广义)
    "VIXCLS":     "VIX",                  # 恐慌指数
    "DCOILWTICO": "WTI_Oil",              # WTI原油
    "GDPC1":      "GDP_Real",             # 实际GDP
    "INDPRO":     "IndustrialProduction", # 工业生产指数
    "UMCSENT":    "ConsumerSentiment",    # 密歇根消费者信心
    "M2SL":       "M2_MoneySupply",       # M2货币供应
    "BAMLH0A0HYM2": "HighYieldSpread",    # 高收益债利差
}

# ETF / 指数代码
TICKERS = {
    "QQQ":  "Nasdaq100_ETF",
    "SPY":  "SP500_ETF",
    "TQQQ": "Nasdaq100_3xETF",
    "DIA":  "DowJones_ETF",
    "IWM":  "Russell2000_ETF",
    "TLT":  "Treasury20Y_ETF",
    "GLD":  "Gold_ETF",
    "^VIX": "VIX_Index",
}

def download_fred():
    print("=" * 60)
    print("下载 FRED 宏观数据")
    print("=" * 60)
    summary = []
    for code, name in FRED_SERIES.items():
        try:
            df = pdr.DataReader(code, "fred", START, END)
            df.columns = [name]
            fp = os.path.join(DATA_DIR, f"fred_{name}.csv")
            df.to_csv(fp)
            n = len(df)
            first = df.index.min().strftime("%Y-%m-%d")
            last  = df.index.max().strftime("%Y-%m-%d")
            print(f"  ✓ {name:25s} {n:5d} rows  {first} → {last}")
            summary.append({"name": name, "code": code, "rows": n,
                            "start": first, "end": last,
                            "latest_value": float(df.iloc[-1, 0])})
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    pd.DataFrame(summary).to_csv(os.path.join(DATA_DIR, "_fred_summary.csv"), index=False)

def download_yf():
    print("=" * 60)
    print("下载 Yahoo Finance ETF/指数数据")
    print("=" * 60)
    summary = []
    for ticker, name in TICKERS.items():
        try:
            df = yf.download(ticker, start=START, end=END,
                             progress=False, auto_adjust=True)
            if df.empty:
                print(f"  ✗ {name}: empty")
                continue
            # 处理多层列
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            fp = os.path.join(DATA_DIR, f"yf_{name}.csv")
            df.to_csv(fp)
            n = len(df)
            first = df.index.min().strftime("%Y-%m-%d")
            last  = df.index.max().strftime("%Y-%m-%d")
            close_col = "Close" if "Close" in df.columns else df.columns[0]
            latest = float(df[close_col].iloc[-1])
            print(f"  ✓ {name:25s} {n:5d} rows  {first} → {last}  last={latest:.2f}")
            summary.append({"name": name, "ticker": ticker, "rows": n,
                            "start": first, "end": last, "latest_close": latest})
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    pd.DataFrame(summary).to_csv(os.path.join(DATA_DIR, "_etf_summary.csv"), index=False)

if __name__ == "__main__":
    download_fred()
    download_yf()
    print("\n全部完成。数据保存于:", DATA_DIR)
